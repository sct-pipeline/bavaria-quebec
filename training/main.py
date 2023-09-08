import os
import argparse
from datetime import datetime
from loguru import logger
import yaml
import json

import wandb
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import PolyLRScheduler, dict2obj, multi_class_dice_score
from transforms import train_transforms, val_transforms
from models import create_nnunet_from_plans

from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR
from monai.data import (DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch)
from monai.transforms import (Compose, EnsureType, EnsureTyped, Invertd, SaveImaged, SaveImage, AsDiscrete, AsDiscreted,)


# create a "model"-agnostic class with PL to use different models
class Model(pl.LightningModule):
    def __init__(self, config, data_root, fold_num, net, loss_function, optimizer_class, 
                 exp_id=None, results_path=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['net', 'loss_function'])

        self.root = data_root
        self.fold_num = fold_num
        self.net = net
        self.num_classes = config.MODEL.NUM_CLASSES
        self.lr = config.OPTIM.LR
        self.optimizer_class = optimizer_class
        self.loss_function = loss_function
        self.save_exp_id = exp_id
        self.results_path = results_path

        # define cropping and padding dimensions
        self.inference_roi_size = self.voxel_cropping_size = config.MODEL.PATCH_SIZE
        self.batch_size = config.OPTIM.BATCH_SIZE

        # define post-processing transforms for validation, nothing fancy just making sure that it's a tensor (default)
        self.val_post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=self.num_classes)]) 
        self.val_post_label = Compose([EnsureType(), AsDiscrete(to_onehot=self.num_classes)])

        # define evaluation metric
        self.train_dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.val_dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        # self.test_dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

        # temp lists for storing outputs from training, validation, and testing
        self.best_val_dice, self.best_val_epoch = 0, 0
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.test_centers_list = ["bavaria", "deepseg"]
        # # specify example_input_array for model summary
        # self.example_input_array = torch.rand(1, 1, 160, 224, 96)


    # --------------------------------
    # FORWARD PASS
    # --------------------------------
    def forward(self, x):
        # x, context_features = self.encoder(x)
        # preds = self.decoder(x, context_features)
        
        out = self.net(x)  
        # # NOTE: MONAI's models only output the logits, not the output after the final activation function
        # # https://docs.monai.io/en/0.9.0/_modules/monai/networks/nets/unetr.html#UNETR.forward refers to the 
        # # UnetOutBlock (https://docs.monai.io/en/0.9.0/_modules/monai/networks/blocks/dynunet_block.html#UnetOutBlock) 
        # # as the final block applied to the input, which is just a convolutional layer with no activation function
        # # Hence, we are used Normalized ReLU to normalize the logits to the final output
        # normalized_out = F.relu(out) / F.relu(out).max() if bool(F.relu(out).max()) else F.relu(out)

        return out  # returns logits


    # --------------------------------
    # DATA PREPARATION
    # --------------------------------   
    def prepare_data(self):
        # set deterministic training for reproducibility
        set_determinism(seed=self.config.SEED)
        
        # define training and validation transforms
        transforms_train = train_transforms(
            crop_size=self.voxel_cropping_size, 
            num_samples_pv=self.config.DATA.NUM_SAMPLES_PER_IMAGE,
            lbl_key='label'
        )
        transforms_val = val_transforms(crop_size=self.voxel_cropping_size, lbl_key='label')
        
        # load the dataset
        dataset = os.path.join(self.root, f"dataset_bavaria-deepseg_seed{self.config.SEED}.json")
        train_files = load_decathlon_datalist(dataset, True, "train")
        val_files = load_decathlon_datalist(dataset, True, "validation")
        test_files_bavaria = load_decathlon_datalist(dataset, True, "test_bavaria")
        test_files_deepseg = load_decathlon_datalist(dataset, True, "test_deepseg")

        if self.config.DEBUG:
            train_files = train_files[:10]
            val_files = val_files[:10]
            test_files_bavaria = test_files_bavaria[:6]
            test_files_deepseg = test_files_deepseg[:6]
        
        train_cache_rate = 0.25 if self.config.DEBUG else 0.5
        self.train_ds = CacheDataset(data=train_files, transform=transforms_train, cache_rate=train_cache_rate, num_workers=4)
        self.val_ds = CacheDataset(data=val_files, transform=transforms_val, cache_rate=0.25, num_workers=4)

        # define test transforms
        transforms_test = val_transforms(crop_size=self.voxel_cropping_size, lbl_key='label')
        
        # define post-processing transforms for testing; taken (with explanations) from 
        # https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py#L66
        self.test_post_pred = Compose([
            EnsureTyped(keys=["pred", "label"]),
            AsDiscreted(keys=["pred"], argmax=True, to_onehot=self.num_classes),
            AsDiscreted(keys=["label"], to_onehot=self.num_classes),
            Invertd(keys=["pred", "label"], transform=transforms_test, 
                    orig_keys=["image", "label"], 
                    meta_keys=["pred_meta_dict", "label_meta_dict"],
                    nearest_interp=False, to_tensor=True),
            ])

        self.test_ds_list = [
            CacheDataset(data=test_files_bavaria, transform=transforms_test, cache_rate=0.1, num_workers=4),
            CacheDataset(data=test_files_deepseg, transform=transforms_test, cache_rate=0.1, num_workers=4)
        ]

        # self.test_ds_bavaria = CacheDataset(data=test_files_bavaria, transform=transforms_test, cache_rate=0.1, num_workers=4)
        # self.test_ds_deepseg = CacheDataset(data=test_files_deepseg, transform=transforms_test, cache_rate=0.1, num_workers=4)


    # --------------------------------
    # DATA LOADERS
    # --------------------------------
    def train_dataloader(self):
        # NOTE: if num_samples=4 in RandCropByPosNegLabeld and batch_size=2, then 2 x 4 images are generated for network training
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=16, 
                            pin_memory=True, persistent_workers=True) # collate_fn=pad_list_data_collate)
        # list_data_collate is only useful when each input in the batch has different shape

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=16, pin_memory=True, 
                          persistent_workers=True)
    
    def test_dataloader(self):
        test_dataloaders_list = []
        
        for i in range(len(self.test_ds_list)):
            test_dataloaders_list.append(
                DataLoader(self.test_ds_list[i], batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
            )
        
        return test_dataloaders_list

        # return DataLoader(self.test_ds_bavaria, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        # TODO: add the deepseg test dataset as well

    
    # --------------------------------
    # OPTIMIZATION
    # --------------------------------
    def configure_optimizers(self):
        if self.config.OPTIM.OPTIMIZER == "sgd":
            optimizer = self.optimizer_class(self.parameters(), lr=self.lr, momentum=0.99, weight_decay=3e-5, nesterov=True)
        else:
            optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        scheduler = PolyLRScheduler(optimizer, self.lr, max_steps=self.config.OPTIM.MAX_EPOCHS)
        return [optimizer], [scheduler]


    # --------------------------------
    # TRAINING
    # --------------------------------
    def training_step(self, batch, batch_idx):

        inputs, labels = batch["image"], batch["label"]

        # NOTE: surprisingly, filtering out empty patches is adding more CSA bias
        # # check if any label image patch is empty in the batch
        # if check_empty_patch(labels) is None:
        #     # print(f"Empty label patch found. Skipping training step ...")
        #     return None

        output = self.forward(inputs)   # logits
        # print(f"labels.shape: {labels.shape} \t output.shape: {output.shape}")
        
        if self.config.MODEL.TYPE == "nnunet" and self.config.MODEL.ENABLE_DEEP_SUPERVISION:

            # calculate dice loss for each output
            loss, train_dice = 0.0, 0.0
            for i in range(len(output)):
                # give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                # NOTE: outputs[0] is the final pred, outputs[-1] is the lowest resolution pred (at the bottleneck)
                # we're downsampling the GT to the resolution of each deepsupervision feature map output 
                # (instead of upsampling each deepsupervision feature map output to the final resolution)
                downsampled_gt = F.interpolate(labels, size=output[i].shape[-3:], mode='trilinear', align_corners=False)
                # print(f"downsampled_gt.shape: {downsampled_gt.shape} \t output[i].shape: {output[i].shape}")
                
                # apply softmax to the logits
                out = F.softmax(output[i], dim=1)

                # calculate training loss
                loss += (0.5 ** i) * self.loss_function(out, downsampled_gt)

                # calculate train dice
                # NOTE: this is done on patches (and not entire 3D volume) because SlidingWindowInference is not used here
                # So, take this dice score with a lot of salt
                self.train_dice_metric(out, downsampled_gt) 
            
            # average dice loss across the outputs
            loss = loss / len(output)
            # train_dice = train_dice / len(output)

        else:
            # apply softmax to the logits
            output = F.softmax(output, dim=1)

            # calculate training loss   
            # NOTE: the diceLoss expects the input to be logits (which it then normalizes inside)
            loss = self.loss_function(output, labels)

            # calculate train dice
            # NOTE: this is done on patches (and not entire 3D volume) because SlidingWindowInference is not used here
            # So, take this dice score with a lot of salt
            self.train_dice_metric(y_pred=output, y=labels)

        metrics_dict = {
            "loss": loss.cpu(),
            "train_number": len(inputs),
            # "train_dice": self.train_dice_metric.aggregate().item().cpu(),
            # "train_image": inputs[0].detach().cpu().squeeze(),
            # "train_gt": labels[0].detach().cpu().squeeze(),
            # "train_pred": output[0].detach().cpu().squeeze()
        }
        self.train_step_outputs.append(metrics_dict)

        return metrics_dict

    def on_train_epoch_end(self):

        train_loss, num_items = 0, 0
        for output in self.train_step_outputs:
            train_loss += output["loss"].sum().item()
            num_items += output["train_number"]
        
        mean_train_loss = (train_loss / num_items)
        mean_train_dice = self.train_dice_metric.aggregate().item()
        self.train_dice_metric.reset()

        wandb_logs = {
            "train_loss": mean_train_loss,
            "train_dice": mean_train_dice,
        }
        self.log_dict(wandb_logs)

        # # plot the training images
        # fig = plot_slices(image=self.train_step_outputs[0]["train_image"],
        #                   gt=self.train_step_outputs[0]["train_gt"],
        #                   pred=self.train_step_outputs[0]["train_pred"],
        #                   debug=args.debug)
        # wandb.log({"training images": wandb.Image(fig)})

        # free up memory
        self.train_step_outputs.clear()
        wandb_logs.clear()
        # plt.close(fig)


    # --------------------------------
    # VALIDATION
    # --------------------------------    
    def validation_step(self, batch, batch_idx):
        
        inputs, labels = batch["image"], batch["label"]
        outputs = sliding_window_inference(inputs, self.inference_roi_size, mode="gaussian",
                                           sw_batch_size=4, predictor=self.forward, overlap=0.5,) 
        # outputs shape: (B, C, <original H x W x D>)
        
        if self.config.MODEL.TYPE == "nnunet" and self.config.MODEL.ENABLE_DEEP_SUPERVISION:
            # we only need the output with the highest resolution
            outputs = outputs[0]

        # apply softmax to the logits
        outputs = F.softmax(outputs, dim=1)        
        
        # calculate validation loss
        loss = self.loss_function(outputs, labels)

        # post-process for calculating the evaluation metric
        post_outputs = [self.val_post_pred(i) for i in decollate_batch(outputs)]
        post_labels = [self.val_post_label(i) for i in decollate_batch(labels)]

        # NOTE: preds and labels need to be 1-hot encoded for the dice metric
        self.val_dice_metric(post_outputs[0], post_labels[0])

        # NOTE: there was a massive memory leak when storing cuda tensors in this dict. Hence,
        # using .detach() to avoid storing the whole computation graph
        # Ref: https://discuss.pytorch.org/t/cuda-memory-leak-while-training/82855/2
        metrics_dict = {
            "val_loss": loss.detach().cpu(),
            "val_number": len(post_outputs),
            # "val_image": inputs[0].detach().cpu().squeeze(),
            # "val_gt": labels[0].detach().cpu().squeeze(),
            # "val_pred": post_outputs[0].detach().cpu().squeeze(),
        }
        self.val_step_outputs.append(metrics_dict)
        
        return metrics_dict

    def on_validation_epoch_end(self):

        val_loss, num_items = 0, 0
        for output in self.val_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        
        mean_val_loss = (val_loss / num_items)
        mean_val_dice = self.val_dice_metric.aggregate().item()
        self.val_dice_metric.reset()
                
        wandb_logs = {
            "val_loss": mean_val_loss,
            "val_dice": mean_val_dice,
        }
        # save the best model based on validation dice score
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        
        print(
            f"Current epoch: {self.current_epoch}"
            f"\nAverage Dice (VAL): {mean_val_dice:.4f}"
            f"\nBest Average Dice: {self.best_val_dice:.4f} at Epoch: {self.best_val_epoch}"
            f"\n----------------------------------------------------")

        # log on to wandb
        self.log_dict(wandb_logs)

        # # plot the validation images
        # fig = plot_slices(image=self.val_step_outputs[0]["val_image"],
        #                   gt=self.val_step_outputs[0]["val_gt"],
        #                   pred=self.val_step_outputs[0]["val_pred"],)
        # wandb.log({"validation images": wandb.Image(fig)})

        # free up memory
        self.val_step_outputs.clear()
        wandb_logs.clear()
        # plt.close(fig)
        
        # return {"log": wandb_logs}

    # --------------------------------
    # TESTING
    # --------------------------------
    def test_step(self, batch, batch_idx, dataloader_idx):
        
        test_input, test_label = batch["image"], batch["label"]
        # print(batch["label_meta_dict"]["filename_or_obj"][0])
        # print(f"test_input.shape: {test_input.shape} \t test_label.shape: {test_label.shape}")
        batch["pred"] = sliding_window_inference(test_input, self.inference_roi_size, 
                                                 sw_batch_size=4, predictor=self.forward, overlap=0.5)
        # print(f"batch['pred'].shape: {batch['pred'].shape}")

        if self.config.MODEL.TYPE == "nnunet" and self.config.MODEL.ENABLE_DEEP_SUPERVISION:
            # we only need the output with the highest resolution
            batch["pred"] = batch["pred"][0]

        # normalize the logits
        batch["pred"] = F.softmax(batch["pred"], dim=1)

        # # upon fsleyes visualization, observed that very small values need to be set to zero, but NOT fully binarizing the pred
        # batch["pred"][batch["pred"] < 0.099] = 0.0

        post_test_out = [self.test_post_pred(i) for i in decollate_batch(batch)]

        # make sure that the shapes of prediction and GT label are the same
        # print(f"pred shape: {post_test_out[0]['pred'].shape}, label shape: {post_test_out[0]['label'].shape}")
        assert post_test_out[0]['pred'].shape == post_test_out[0]['label'].shape
        
        pred, label = post_test_out[0]['pred'].cpu(), post_test_out[0]['label'].cpu()

        # save the prediction and label
        if self.config.SAVE_TEST_PREDS:

            subject_name = (batch["image_meta_dict"]["filename_or_obj"][0]).split("/")[-1].replace(".nii.gz", "")
            print(f"Saving subject: {subject_name}")

            # image saver class
            save_folder = os.path.join(self.results_path, subject_name.split("_")[0])
            pred_saver = SaveImage(
                output_dir=save_folder, output_postfix="pred", output_ext=".nii.gz", 
                separate_folder=False, print_log=False)
            # save the prediction
            pred_saver(pred)

            label_saver = SaveImage(
                output_dir=save_folder, output_postfix="gt", output_ext=".nii.gz", 
                separate_folder=False, print_log=False)
            # save the label
            label_saver(label)
            
        # calculate only dice score, rest of the metrics will be computed by ANIMA on the saved predictions
        test_dice = multi_class_dice_score(pred, label)
        metrics_dict = {
            self.test_centers_list[dataloader_idx]: [test_dice],
        }
        self.test_step_outputs.append(metrics_dict)

        return metrics_dict

    def on_test_epoch_end(self):
        
        # outputs = self.test_step_outputs
        avg_test_dice_bavaria = []
        avg_test_dice_deepseg = []
        avg_test_dice = {}

        # split the outputs into two lists corresponding to each cente
        for output in self.test_step_outputs:
            for k, v in output.items():
                if k == "bavaria":
                    avg_test_dice_bavaria.append(v[0])
                else:
                    avg_test_dice_deepseg.append(v[0])
        
        # stack the dice scores for each center
        avg_test_dice_bavaria = (torch.stack(avg_test_dice_bavaria).mean()).cpu().numpy()
        avg_test_dice_deepseg = (torch.stack(avg_test_dice_deepseg).mean()).cpu().numpy()
        avg_test_dice["bavaria"] = avg_test_dice_bavaria
        avg_test_dice["deepseg"] = avg_test_dice_deepseg

        # test_mean_dice = self.test_dice_metric.aggregate().item()
        logger.info(f"Test Dice: {[(k, v.item()) for k, v in avg_test_dice.items()]}")
        
        self.avg_test_dice = avg_test_dice

        # # free up memory
        # self.test_step_outputs.clear()


# --------------------------------
# MAIN
# --------------------------------
def main(args):

    # get the config file and unyaml it
    with open(args.config, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config_dict)

    # Setting the seed
    pl.seed_everything(config.SEED, workers=True)
    
    # check if nnunet plans file is provided
    if config.MODEL.TYPE == "nnunet":
        if args.nnunet_plans is None:
            raise ValueError(f"Please provide the nnUNet plans file in the config file!")
        else:
            # load the json file
            with open(args.nnunet_plans, "r") as f:
                nnunet_plans_json = json.load(f)

    # define root path for finding datalists
    dataset_root = config.DIRS.DATASET

    # define optimizer
    if config.OPTIM.OPTIMIZER in ["adam", "Adam"]:
        optimizer_class = torch.optim.Adam
    elif config.OPTIM.OPTIMIZER in ["SGD", "sgd"]:
        optimizer_class = torch.optim.SGD

    # define models
    if config.MODEL.TYPE in ["unet"]:
        logger.info(f" Using ivadomed's UNet model! ")
        # # this is the ivadomed unet model
        # net = ModifiedUNet3D(
        #     in_channels=config.MODEL.IN_CHANNELS, 
        #     out_channels=config.MODEL.NUM_CLASSES, 
        #     init_filters=config.MODEL.INIT_FILTERS,
        # )
        # patch_size = config.MODEL.PATCH_SIZE
        # patch_size = f"{patch_size[0]}x{patch_size[1]}x{patch_size[2]}"
        # # save_exp_id =f"ivado_{args.model}_nf={args.init_filters}_opt={args.optimizer}_lr={args.learning_rate}" \
        # #                 f"_CSAdiceL_nspv={args.num_samples_per_volume}" \
        # #                 f"_bs={args.batch_size}_{patch_size}"
        # save_exp_id =f"ivado_{args.model}_nf={args.init_filters}_opt={args.optimizer}_lr={args.learning_rate}" \
        #                 f"_CSAdiceL_CCrop_bs={args.batch_size}_{patch_size}"


    elif config.MODEL.TYPE in ["unetr"]:
        # define image size to be fed to the model
        img_size = config.MODEL.PATCH_SIZE
        
        # TODO: fix the arguments when using unetr
        # define model
        net = UNETR(spatial_dims=3,
                    in_channels=1, out_channels=1, 
                    img_size=img_size,
                    feature_size=args.feature_size, 
                    hidden_size=args.hidden_size, 
                    mlp_dim=args.mlp_dim, 
                    num_heads=args.num_heads,
                    pos_embed="conv", 
                    norm_name="instance", 
                    res_block=True, 
                    dropout_rate=0.2,
                )
        img_size = f"{img_size[0]}x{img_size[1]}x{img_size[2]}"
        save_exp_id = f"{args.model}_opt={args.optimizer}_lr={args.learning_rate}" \
                        f"_fs={args.feature_size}_hs={args.hidden_size}_mlpd={args.mlp_dim}_nh={args.num_heads}" \
                        f"_CSAdiceL_nspv={args.num_samples_per_volume}_bs={args.batch_size}_{img_size}" \

    elif config.MODEL.TYPE in ["nnunet"]:
        if config.MODEL.ENABLE_DEEP_SUPERVISION:
            logger.info(f" Using nnUNet model WITH deep supervision! ")
        else:
            logger.info(f" Using nnUNet model WITHOUT deep supervision! ")

        # define model
        net = create_nnunet_from_plans(
            plans=nnunet_plans_json, 
            num_input_channels=config.MODEL.IN_CHANNELS, 
            num_classes=config.MODEL.NUM_CLASSES,
            deep_supervision=config.MODEL.ENABLE_DEEP_SUPERVISION
        )
        patch_size = config.MODEL.PATCH_SIZE
        patch_size = f"{patch_size[0]}x{patch_size[1]}x{patch_size[2]}"
        save_exp_id =f"{config.MODEL.TYPE}_nf={config.MODEL.INIT_FILTERS}_DS={int(config.MODEL.ENABLE_DEEP_SUPERVISION)}" \
                        f"_opt={config.OPTIM.OPTIMIZER}_lr={config.OPTIM.LR}" \
                        f"_DiceCE_nspv={config.DATA.NUM_SAMPLES_PER_IMAGE}" \
                        f"_bs={config.OPTIM.BATCH_SIZE}_{patch_size}"

    # define loss function
    # reduction "mean" averages the losses for both classes and returns a single number
    loss_func = DiceCELoss(to_onehot_y=True, reduction="mean")

    # TODO: move this inside the for loop when using more folds
    timestamp = datetime.now().strftime(f"%Y%m%d-%H%M")   # prints in YYYYMMDD-HHMMSS format
    save_exp_id = f"{save_exp_id}_{timestamp}"

    # to save the best model on validation
    save_path = os.path.join(config.DIRS.SAVE_PATH, f"{save_exp_id}")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # to save the results/model predictions 
    results_path = os.path.join(config.DIRS.RESULTS_DIR, f"{save_exp_id}")
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    # train across all folds of the dataset
    num_cv_folds = 1
    for fold in range(num_cv_folds):
        logger.info(f" Training on fold {fold+1} out of {num_cv_folds} folds! ")

        # i.e. train by loading weights from scratch
        pl_model = Model(config, data_root=dataset_root, fold_num=fold, 
                         optimizer_class=optimizer_class, loss_function=loss_func, net=net, 
                         exp_id=save_exp_id, results_path=results_path)

        # don't use wandb logger if in debug mode
        # if not args.debug:
        exp_logger = pl.loggers.WandbLogger(
                            name=save_exp_id,
                            save_dir=config.DIRS.SAVE_PATH,
                            group=config.WANDB.GRP_NAME,
                            log_model=True, # save best model using checkpoint callback
                            project='bavaria-quebec',
                            entity='naga-karthik',
                            config=config_dict)
        
        # saving the best model based on soft validation dice score
        checkpoint_callback_dice = pl.callbacks.ModelCheckpoint(
            dirpath=save_path, filename='best_model', monitor='val_dice', 
            save_top_k=1, mode="max", save_last=False, save_weights_only=True)
        
        early_stopping = pl.callbacks.EarlyStopping(monitor="val_dice", min_delta=0.00, 
                            patience=config.OPTIM.PATIENCE, verbose=False, mode="max")

        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

        # Saving training script to wandb
        wandb.save("main.py")
        wandb.save("transforms.py")
        wandb.save(args.config)

        # initialise Lightning's trainer.
        trainer = pl.Trainer(
            devices=1, accelerator="gpu", 
            logger=exp_logger,
            callbacks=[checkpoint_callback_dice, lr_monitor, early_stopping],
            check_val_every_n_epoch=config.OPTIM.VAL_EVERY_N_EPOCHS,
            max_epochs=config.OPTIM.MAX_EPOCHS, 
            precision=32,   # TODO: see if 16-bit precision is stable
            # deterministic=True,
            enable_progress_bar=False,) 
            # profiler="simple",)     # to profile the training time taken for each step

        # Train!
        trainer.fit(pl_model)        
        logger.info(f" Training Done!")

        # Test!
        trainer.test(pl_model)
        logger.info(f"TESTING DONE!")

        # closing the current wandb instance so that a new one is created for the next fold
        wandb.finish()
        
        # TODO: Figure out saving test metrics to a file
        with open(os.path.join(results_path, 'test_metrics.txt'), 'a') as f:
            print('\n-------------- Test Metrics ----------------', file=f)
            print(f"\nSeed Used: {config.SEED}", file=f)
            print(f"\ninitf={config.MODEL.INIT_FILTERS}_lr={config.OPTIM.LR}_bs={config.OPTIM.BATCH_SIZE}_{timestamp}", file=f)
            print(f"\npatch_size={pl_model.voxel_cropping_size}", file=f)
            
            print('\n-------------- Test Dice Scores ----------------', file=f)
            print(f"Average Multi-class Dice: {[(k, round(v.item(), 3)) for k, v in pl_model.avg_test_dice.items()]}", file=f)

            print('-------------------------------------------------------', file=f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script for training custom models for Spinal Cord MS Lesion Segmentation.')
    # get the config file
    parser.add_argument('-c', '--config', required=True, type=str, help='Path to the config file')
    # if training an nnunet model, get the nnunet_plans file
    parser.add_argument('--nnunet_plans', default=None, type=str, 
                        help='Path to the nnunet plans file for initializing the nnunet model')

    args = parser.parse_args()

    main(args)