from hmac import new
import os
import shutil
import argparse
import glob

def copy_and_rename_lesion_files(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Glob all lesion files and copy them including their full directory paths
    lesion_files = sorted(glob.glob(os.path.join(input_dir, '**', '*lesions-manual_T2w.nii.gz'), recursive=True))
    seg_files = [str(x).replace("lesions","seg") for x in lesion_files]
    ax_files = [str(x).replace("lesions-manual_","") for x in lesion_files]

    for file in lesion_files:
        relative_path = os.path.relpath(file, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(file, output_path)


    for file in seg_files:
        relative_path = os.path.relpath(file, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(file, output_path)

    for file in ax_files:
        relative_path = os.path.relpath(file, input_dir)
        output_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(file, output_path)

    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy and rename lesion files")
    parser.add_argument("--input_dir", help="Input directory containing lesion files")
    parser.add_argument("--output_dir", help="Output directory for copied and renamed files")
    args = parser.parse_args()

    copy_and_rename_lesion_files(args.input_dir, args.output_dir)
