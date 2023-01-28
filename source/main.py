import os
import argparse
from datetime import datetime
from loguru import logger

import numpy as np
import wandb
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from ivadomed.losses import DiceLoss as ivadoDiceLoss
from ivadomed.metrics import precision_score, recall_score

from monai.utils import set_determinism
from monai.losses import DiceCELoss, DiceLoss, FocalLoss
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, DynUNet, BasicUNet, SegResNet, UNETR
from monai.data import (DataLoader, Dataset, CacheDataset, load_decathlon_datalist, decollate_batch, 
                        pad_list_data_collate, PatchIterd, GridPatchDataset)
from monai.transforms import (Compose, EnsureType, EnsureTyped, Invertd, SaveImaged)

import utils.transforms as utils_transforms

# create a "model"-agnostic class with PL to use different models on both datasets
class Model(pl.LightningModule):
    def __init__(self, args, data_root, fold_num, net, loss_function, optimizer_class, 
                    load_pretrained=True, load_path=None, exp_id=None):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=['net'])

        # if self.args.unet_depth == 3:
        #     from models import ModifiedUNet3DEncoder, ModifiedUNet3DDecoder   # this is 3-level UNet
        #     logger.info("Using UNet with Depth = 3! ")
        # else:
        #     from models_original import ModifiedUNet3DEncoder, ModifiedUNet3DDecoder
        #     logger.info("Using UNet with Depth = 4! ")

        self.root = data_root
        self.fold_num = fold_num
        self.net = net
        # self.load_pretrained = load_pretrained
        self.lr = args.learning_rate
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class
        self.save_exp_id = exp_id

        # TODO: figure out how to load pretrained models
        # # instantiate the model
        # if not self.load_pretrained:
        #     logger.info("INITIALIZING ENCODER WEIGHTS FROM SCRATCH!")
        #     self.encoder = ModifiedUNet3DEncoder(in_channels=1, base_n_filter=args.init_filters, attention=False)
        # else:
        #     logger.info(f"LOADING PRETRAINED WEIGHTS FOR THE ENCODER TRAINED ON { centers_list[self.center_idx - 1] }!")
        #     self.encoder = ModifiedUNet3DEncoder(in_channels=1, base_n_filter=args.init_filters, attention=False)
        #     self.encoder.load_state_dict(torch.load(load_path))
        #     self.encoder.eval()   
        # self.decoder = ModifiedUNet3DDecoder(n_classes=1, base_n_filter=args.init_filters)

        self.best_val_dice, self.best_val_epoch = 0, 0
        self.metric_values = []
        self.epoch_losses, self.epoch_soft_dice_train, self.epoch_hard_dice_train = [], [], []

        # define cropping and padding dimensions
        # TODO: hard-coded for now, but should be changed to args.patch_size when optimal crop_size is determined
        self.voxel_cropping_size = (96, 96, 96) # (args.patch_size,) * 3 
        self.inference_roi_size = (96, 96, 96) # (args.patch_size,) * 3 

        # define post-processing transforms for validation, nothing fancy just making sure that it's a tensor (default)
        self.val_post_pred = Compose([EnsureType()]) 
        self.val_post_label = Compose([EnsureType()])

        # define evaluation metric
        self.ivado_dice_metric = ivadoDiceLoss(smooth=1.0)

    def forward(self, x):
        # x, context_features = self.encoder(x)
        # preds = self.decoder(x, context_features)
        
        out = self.net(x)
        # NOTE: MONAI's models only output the logits, not the output after the final activation function
        # https://docs.monai.io/en/0.9.0/_modules/monai/networks/nets/unetr.html#UNETR.forward refers to the 
        # UnetOutBlock (https://docs.monai.io/en/0.9.0/_modules/monai/networks/blocks/dynunet_block.html#UnetOutBlock) 
        # as the final block applied to the input, which is just a convolutional layer with no activation function
        # Hence, we are used Normalized ReLU to normalize the logits to the final output
        normalized_out = F.relu(out) / F.relu(out).max() if bool(F.relu(out).max()) else F.relu(out)

        return normalized_out

    def prepare_data(self):
        # set deterministic training for reproducibility
        set_determinism(seed=self.args.seed)

        # NOTE: Volume-level preprocessing is done
        
        # define training and validation transforms
        train_transforms = utils_transforms.train_transforms(
            crop_size=self.voxel_cropping_size, 
            num_samples_pv=self.args.num_samples_per_volume,
            lbl_key='label'
        )
        val_transforms = utils_transforms.val_transforms(lbl_key='label')
        
        # load the dataset
        dataset = self.root + f"bavaria-qc_fold-{self.fold_num}.json"
        train_files = load_decathlon_datalist(dataset, True, "training")
        val_files = load_decathlon_datalist(dataset, True, "validation")
        test_files = load_decathlon_datalist(dataset, True, "test")

        self.train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.25, num_workers=4)
        self.val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)

        # define test transforms
        test_transforms = utils_transforms.test_transforms(lbl_key='label')
        
        # define post-processing transforms for testing; taken (with explanations) from 
        # https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py#L66
        self.test_post_pred = Compose([
            EnsureTyped(keys=["pred", "label"]),
            Invertd(keys="pred", transform=test_transforms, orig_keys="image",  meta_keys="pred_meta_dict",  
                    orig_meta_keys=["image_meta_dict"],  meta_key_postfix="meta_dict", nearest_interp=False, to_tensor=True),
            ])

        self.test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0, num_workers=4)


    def train_dataloader(self):
        # NOTE: if num_samples=4 in RandCropByPosNegLabeld and batch_size=2, then 2 x 4 images are generated for network training
        return DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=4, 
                            pin_memory=True, collate_fn=pad_list_data_collate)
        # list_data_collate is only useful when each input in the batch has different shape

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    def test_dataloader(self):
        test_dataloaders_list = []
        
        for i in range(len(self.test_datasets_list)):
            test_dataloaders_list.append(
                DataLoader(self.test_datasets_list[i], batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
            )
        
        return test_dataloaders_list
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        inputs, labels = batch["image"], batch["label"]
        output = self.forward(inputs)

        # calculate training loss
        # ivadomed dice loss returns - 2.0 x ...., so we first make it positive and subtract from 1.0
        loss = 1.0 - (self.loss_function(output, labels) * -1.0)

        # calculate train dice
        # NOTE: this is done on patches (and not entire 3D volume) because SlidingWindowInference is not used here
        train_soft_dice = self.ivado_dice_metric(output, labels) 
        train_hard_dice = self.ivado_dice_metric((output.detach() > 0.5).float(), (labels.detach() > 0.5).float())

        return {
            "loss": loss,
            "train_soft_dice": train_soft_dice,
            "train_hard_dice": train_hard_dice,
            "train_number": len(inputs)
        }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_soft_dice_train = torch.stack([x["train_soft_dice"] for x in outputs]).mean()
        avg_hard_dice_train = torch.stack([x["train_hard_dice"] for x in outputs]).mean()
        
        self.log('train_soft_dice', avg_soft_dice_train, on_step=False, on_epoch=True)

        self.epoch_losses.append(avg_loss.detach().cpu().numpy())
        self.epoch_soft_dice_train.append(avg_soft_dice_train.detach().cpu().numpy())
        self.epoch_hard_dice_train.append(avg_hard_dice_train.detach().cpu().numpy())

    
    def validation_step(self, batch, batch_idx):
        
        inputs, labels = batch["image"], batch["label"]

        inference_roi_size = self.inference_roi_size
        sw_batch_size = 4
        outputs = sliding_window_inference(inputs, inference_roi_size, sw_batch_size, self.forward, overlap=0.5,) 
        # outputs shape: (B, C, <original H x W x D>)
        
        # calculate validation loss
        # ivadomed dice loss returns - 2.0 x ...., so we first make it positive and subtract from 1.0
        loss = 1.0 - (self.loss_function(outputs, labels) * -1.0)
        
        # post-process for calculating the evaluation metric
        post_outputs = [self.val_post_pred(i) for i in decollate_batch(outputs)]
        post_labels = [self.val_post_label(i) for i in decollate_batch(labels)]
        val_soft_dice = -1.0 * self.ivado_dice_metric(post_outputs[0], post_labels[0])
        val_hard_dice = -1.0 * self.ivado_dice_metric((post_outputs[0].detach() > 0.5).float(), (post_labels[0].detach() > 0.5).float())
        
        return {
            "val_loss": loss, 
            "val_soft_dice": val_soft_dice,
            "val_hard_dice": val_hard_dice,
            "val_number": len(post_outputs),
            }

    def validation_epoch_end(self, outputs):
        val_loss, num_val_items, val_soft_dice, val_hard_dice = 0, 0, 0.0, 0.0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            val_soft_dice += output["val_soft_dice"].sum().item()
            val_hard_dice += output["val_hard_dice"].sum().item()
            num_val_items += output["val_number"]
        
        mean_val_loss = torch.tensor(val_loss / num_val_items)
        mean_val_soft_dice = torch.tensor(val_soft_dice / num_val_items)
        mean_val_hard_dice = torch.tensor(val_hard_dice / num_val_items)
        
        wandb_logs = {
            "val_soft_dice": mean_val_soft_dice,
            "val_hard_dice": mean_val_hard_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_soft_dice > self.best_val_dice:
            self.best_val_dice = mean_val_soft_dice
            self.best_val_epoch = self.current_epoch

        print(
            f"Current epoch: {self.current_epoch}"
            f"\nCurrent Mean Soft Dice: {mean_val_soft_dice:.4f}"
            f"\nCurrent Mean Hard Dice: {mean_val_hard_dice:.4f}"
            f"\nBest Mean Dice: {self.best_val_dice:.4f} at Epoch: {self.best_val_epoch}"
            f"\n----------------------------------------------------")
        
        self.metric_values.append(mean_val_soft_dice)

        # log on to wandb
        self.log_dict(wandb_logs)
        
        return {"log": wandb_logs}


    def test_step(self, batch, batch_idx, dataloader_idx):
        # Sequentially computes the things below for each dataloader
        
        test_input, test_label = batch["image"], batch["label"]
        roi_size = self.inference_roi_size
        sw_batch_size = 4
        batch["pred"] = sliding_window_inference(test_input, roi_size, sw_batch_size, self.forward, overlap=0.5)

        # upon fsleyes visualization, observed that very small values need to be set to zero, but NOT fully binarizing the pred
        batch["pred"][batch["pred"] < 0.099] = 0.0

        post_test_out = [self.test_post_pred(i) for i in decollate_batch(batch)]

        # NOTE: exceptionally for this ms_brain_spine dataset, we're using this method to save the images. This is because
        # the dataset is not bidsified, i.e. the subject names do not appear in the file names due to which, they are 
        # overwritten when included in test_post_pred. 
        # Using nib.save() as done in original ms-challenge seems to mess up outputs for some reason
        # subject_name = (batch["label_meta_dict"]["filename_or_obj"][0]).split(os.sep)[9]
        # self.predictions_save_path = os.path.join(
        #                 self.args.results_dir, centers_order, f"ER_seed={self.args.seed}", self.save_exp_id
        #                 )
        # save_transform = Compose([
        #     # AsDiscreted(keys="pred", argmax=False, threshold=0.1), #, to_onehot=2), ANIMA only needs binary predictions
        #     # NOTE: despite the small threshold, it just binarizes everything, hence not using
        #     SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir=os.path.join(self.predictions_save_path, subject_name), 
        #                 output_postfix="pred", resample=False),
        #     SaveImaged(keys="label", meta_keys="image_meta_dict", output_dir=os.path.join(self.predictions_save_path, subject_name), 
        #                 output_postfix="gt", resample=False),
        #     ])
        # post_test_out = [save_transform(i) for i in decollate_batch(batch)]

        # make sure that the shapes of prediction and GT label are the same
        assert post_test_out[0]['pred'].shape == post_test_out[0]['label'].shape
        
        pred, label = post_test_out[0]['pred'].cpu(), post_test_out[0]['label'].cpu()

        # NOTE: Important point from the SoftSeg paper - binarize predictions before computing metrics

        # calculate all metrics here
        # 1. Dice Score
        test_soft_dice = -1.0 * self.ivado_dice_metric(pred, label)

        # binarizing the predictions 
        pred = (post_test_out[0]['pred'].detach().cpu() > 0.5).float()
        label = (post_test_out[0]['label'].detach().cpu() > 0.5).float()

        # 1.1 Hard Dice Score
        test_hard_dice = -1.0 * self.ivado_dice_metric(pred, label)
        # 2. Precision Score
        test_precision = precision_score(pred.numpy(), label.numpy())
        # 3. Recall Score
        test_recall = recall_score(pred.numpy(), label.numpy())

        return {
            "test_hard_dice": test_hard_dice,
            "test_soft_dice": test_soft_dice,
            "test_precision": test_precision,
            "test_recall": test_recall,
        }

    def test_epoch_end(self, outputs):

        avg_soft_dice_test, avg_hard_dice_test = {}, {}
        avg_precision_test, avg_recall_test = {}, {}
        
        avg_hard_dice_test[self.center_name] = torch.stack([x["test_hard_dice"] for x in outputs]).mean().cpu().numpy()        
        avg_soft_dice_test[self.center_name] = torch.stack([x["test_soft_dice"] for x in outputs]).mean().cpu().numpy()
        avg_precision_test[self.center_name] = (np.stack([x["test_precision"] for x in outputs]).mean())
        avg_recall_test[self.center_name] = (np.stack([x["test_recall"] for x in outputs]).mean())

        logger.info(f"Test (Soft) Dice for Center {self.center_name}: {avg_soft_dice_test}")
        logger.info(f"Test (Hard) Dice for Center {self.center_name}: {avg_hard_dice_test}")
        logger.info(f"Test Precision Score for Center {self.center_name}: {avg_precision_test}")
        logger.info(f"Test Recall Score for Center {self.center_name}: {avg_recall_test}")
        
        self.avg_test_dice = avg_soft_dice_test
        self.avg_test_dice_hard = avg_hard_dice_test
        self.avg_test_precision = avg_precision_test
        self.avg_test_recall = avg_recall_test



def main(args):
    # Setting the seed
    pl.seed_everything(args.seed, workers=True)

    # final matrix of test metrics. Note: there is only 1 dataset, so 1x1 matrix
    final_dice_scores = np.zeros((1, 1))
    final_hard_dice_scores = np.zeros((1, 1))
    final_precision_scores = np.zeros((1, 1))
    final_recall_scores = np.zeros((1, 1))

    # define root path for finding datalists
    dataset_root = "/home/GRAMES.POLYMTL.CA/u114716/tum-poly/custom_model_3D/datalists/"

    # define optimizer
    if args.optimizer in ["adamw", "AdamW", "Adamw"]:
        optimizer_class = torch.optim.AdamW
    elif args.optimizer in ["SGD", "sgd"]:
        optimizer_class = torch.optim.SGD

    # define models
    if args.model in ["unet", "UNet"]:            
        net = UNet(spatial_dims=2 if args.model_type == "2d" else 3, 
                    in_channels=1, out_channels=1,
                    channels=(
                        args.init_filters, 
                        args.init_filters * 2, 
                        args.init_filters * 4, 
                        args.init_filters * 8, 
                        args.init_filters * 16
                    ),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                )
        save_exp_id =f"{args.model}_lr={args.learning_rate}"
    elif args.model in ["unetr", "UNETR"]:
        # define image size to be fed to the model
        img_size = (96, 96, 96) if args.model_type == "3d" else (96, 96)
        
        # define model
        net = UNETR(spatial_dims=2 if args.model_type == "2d" else 3,
                    in_channels=1, out_channels=1, 
                    img_size=img_size,
                    feature_size=args.feature_size, 
                    hidden_size=args.hidden_size, 
                    mlp_dim=args.mlp_dim, 
                    num_heads=args.num_heads,
                    pos_embed="perceptron", 
                    norm_name="instance", 
                    res_block=True, 
                    dropout_rate=0.2,
                )
        save_exp_id = f"{args.model}_lr={args.learning_rate}" \
                        f"_fs={args.feature_size}_hs={args.hidden_size}_mlpd={args.mlp_dim}_nh={args.num_heads}"
    elif args.model in ["segresnet", "SegResNet"]:
        net = SegResNet(
            in_channels=1, out_channels=2,
            init_filters=16,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            dropout_prob=0.2,)
        save_exp_id =f"{args.model}_lr={args.learning_rate}"

    # define loss function
    loss_func = ivadoDiceLoss(smooth=1.0)

    # to save the best model on validation
    save_path = os.path.join(args.save_path, f"{args.model}_{args.model_type}")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    pretrained_load_path = None

    # train across all folds of the dataset
    for fold in range(args.num_cv_folds):
        logger.info(f" Training on fold {fold} out of {args.num_cv_folds} folds! ")

        timestamp = datetime.now().strftime(f"%Y%m%d-%H%M%S")   # prints in YYYYMMDD-HHMMSS format
        save_exp_id = f"{save_exp_id}_fold={fold}_{timestamp}"

        # TODO: figure out pretraining stuff
        if pretrained_load_path is not None:
            pl_model = Model(args, data_root=dataset_root, fold_num=fold, optimizer_class=optimizer_class,
                                loss_function=loss_func, load_pretrained=True, load_path=pretrained_load_path, 
                                exp_id=save_exp_id, net=None)
            
        else:
            # i.e. train by loading weights from scratch
            pl_model = Model(args, data_root=dataset_root, fold_num=fold, optimizer_class=optimizer_class,
                                loss_function=loss_func, net=net, load_pretrained=False, load_path=None, 
                                exp_id=save_exp_id)

        wandb_logger = pl.loggers.WandbLogger(
                            name=save_exp_id,
                            group=f"{args.model}", 
                            log_model=True, # save best model using checkpoint callback
                            project='bavaria-quebec',
                            entity='naga-karthik',
                            config=args)
        
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=save_path, filename=save_exp_id, monitor='val_loss', 
            save_top_k=1, mode="min", save_last=False, save_weights_only=True)
        
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        
        early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, 
                            verbose=False, mode="min")

        # initialise Lightning's trainer.
        trainer = pl.Trainer(
            devices=1, accelerator="gpu", # strategy="ddp",
            logger=wandb_logger, 
            callbacks=[checkpoint_callback, lr_monitor, early_stopping],
            check_val_every_n_epoch=args.check_val_every_n_epochs,
            max_epochs=args.max_epochs, 
            precision=32,
            deterministic=True,
            enable_progress_bar=args.enable_progress_bar)

        # Train!
        trainer.fit(pl_model)        
        logger.info(f" Training Done!") # --> TRAINED ON CENTER: {center}; TESTING ON ALL CENTERS !! ")

        # TODO: Come back to testing when hyperparamters have been fixed after cross-validation
        # # Test!
        # trainer.test(pl_model)

        # final_dice_scores[0, 0] = np.fromiter(pl_model.avg_test_dice.values(), dtype=float)
        # final_hard_dice_scores[0, 0] = np.fromiter(pl_model.avg_test_dice_hard.values(), dtype=float)
        # final_precision_scores[0, 0] = np.fromiter(pl_model.avg_test_precision.values(), dtype=float)
        # final_recall_scores[0, 0] = np.fromiter(pl_model.avg_test_recall.values(), dtype=float)
        # print(final_hard_dice_scores)

        # logger.info(f"TESTING DONE!")

        # closing the current wandb instance so that a new one is created for the next fold
        wandb.finish()

        # TODO: Figure out saving test metrics to a file
        # with open(os.path.join(save_path, 'test_metrics.txt'), 'a') as f:
        #     print('\n-------------- Test Metrics from training on Individual Centers ----------------', file=f)
        #     print(f"\nSeed Used: {args.seed}", file=f)
        #     print(f"\ninitf={args.init_filters}_patch={args.patch_size}_lr={args.learning_rate}_bs={args.batch_size}_{timestamp[4:]}", file=f)
        #     print(f"\n{np.array(centers_list)[None, :]}", file=f)
        #     print(f"\n{np.array(centers_list)[:, None]}", file=f)
            
        #     print('\n-------------- Test Hard Dice Scores ----------------', file=f)
        #     print(f" { repr(final_hard_dice_scores)}", file=f)

        #     print('\n-------------- Test Precision Scores ----------------', file=f)
        #     print(f" { repr(final_precision_scores)}", file=f)

        #     print('\n-------------- Test Recall Scores ----------------', file=f)
        #     print(f" { repr(final_recall_scores)}", file=f)

        #     print('\n-------------- Test Soft Dice Scores ----------------', file=f)
        #     print(f" { repr(final_dice_scores)}", file=f)

        #     print('-----------------------------------------------------------------', file=f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script for training custom models for SCI Lesion Segmentation.')
    # Arguments for model, data, and training and saving
    parser.add_argument('-e', '--only_eval', default=False, action='store_true', help='Only do evaluation, i.e. skip training!')
    parser.add_argument('-m', '--model', 
                        choices=['unet', 'UNet', 'unetr', 'UNETR', 'segresnet', 'SegResNet'], 
                        default='unet', type=str, help='Model type to be used')
    parser.add_argument('-mt', '--model_type', choices=['2d', '3d'],
                        default='3d', type=str, help='Model type to be used')
    # dataset
    parser.add_argument('-nspv', '--num_samples_per_volume', default=4, type=int, help="Number of samples to crop per volume")    
    parser.add_argument('-ncv', '--num_cv_folds', default=5, type=int, help="Number of cross validation folds")
    
    # unet model 
    parser.add_argument('-initf', '--init_filters', default=16, type=int, help="Number of Filters in Init Layer")
    parser.add_argument('-ps', '--patch_size', type=int, default=128, help='List containing subvolume size')
    parser.add_argument('-dep', '--unet_depth', default=3, type=int, help="Depth of UNet model")

    # unetr model 
    parser.add_argument('-fs', '--feature_size', default=16, type=int, help="Feature Size")
    parser.add_argument('-hs', '--hidden_size', default=768, type=int, help='Dimensionality of hidden embeddings')
    parser.add_argument('-mlpd', '--mlp_dim', default=2048, type=int, help='Dimensionality of MLP layer')
    parser.add_argument('-nh', '--num_heads', default=12, type=int, help='Number of heads in Multi-head Attention')

    # optimizations
    parser.add_argument('-lf', '--loss_func', choices=['ivado_dice', 'dice', 'dice_ce', 'dice_f'],
                         default='dice', type=str, help="Loss function to use")
    parser.add_argument('-me', '--max_epochs', default=1000, type=int, help='Number of epochs for the training process')
    parser.add_argument('-bs', '--batch_size', default=2, type=int, help='Batch size of the training and validation processes')
    parser.add_argument('-opt', '--optimizer', 
                        choices=['adamw', 'AdamW', 'SGD', 'sgd'], 
                        default='adamw', type=str, help='Optimizer to use')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='Learning rate for training the model')
    parser.add_argument('-pat', '--patience', default=200, type=int, 
                            help='number of validation steps (val_every_n_iters) to wait before early stopping')
    parser.add_argument('-epb', '--enable_progress_bar', default=False, action='store_true', 
                            help='by default is disabled since it doesnt work in colab')
    parser.add_argument('-cve', '--check_val_every_n_epochs', default=1, type=int, help='num of epochs to wait before validation')
    # saving
    parser.add_argument('-sp', '--save_path', 
                        default=f"/home/GRAMES.POLYMTL.CA/u114716/tum-poly/custom_model_3D/saved_models", 
                        type=str, help='Path to the saved models directory')
    parser.add_argument('-c', '--continue_from_checkpoint', default=False, action='store_true', 
                            help='Load model from checkpoint and continue training')
    parser.add_argument('-se', '--seed', default=42, type=int, help='Set seeds for reproducibility')
    # testing
    parser.add_argument('-rd', '--results_dir', 
                    default=f"/home/GRAMES.POLYMTL.CA/u114716/tum-poly/custom_model_3D/model_predictions", 
                    type=str, help='Path to the model prediction results directory')


    args = parser.parse_args()

    main(args)
