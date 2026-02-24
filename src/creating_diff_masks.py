# Generate diff images between ground truth and predictions

from pathlib import Path
import numpy as np
from imageio.v3 import imread, imwrite
from tqdm import tqdm
from skimage.color import rgb2lab, gray2rgb
from PIL import Image


def my_rgb2lab(x):
    if len(x.shape) == 2:
        x = gray2rgb(x)
    x = rgb2lab(x)
    return x / np.array([[[100, 127 * 2, 127 * 2]]]) + np.array([[[0, 0.5, 0.5]]])


root_path = Path(__file__).parent

dataset_basename = "LIU4K_v2_train"
subdatasets = ["Animal", "Building", "Mountain", "Street" ]

dataset_path = root_path / "datasets" / dataset_basename
output_path = root_path / "output_sinsr_adaptive" / dataset_basename
diffs_path = root_path / "masks" / dataset_basename

# Create output directory
diffs_path.mkdir(parents=True, exist_ok=True)

# Find prediction files only in the specified subfolders
pred_files = []
for subfolder in subdatasets:
    subfolder_path = output_path / subfolder
    if subfolder_path.exists():
        # Using glob on the specific subfolder path
        pred_files.extend(list(subfolder_path.glob("**/*_pred.npy")))
    else:
        print(f"Warning: Subfolder not found - {subfolder_path}")

for pred_file in tqdm(pred_files, desc="Generating diffs"):
    # Extract relative path (e.g., "Animal/image_name")
    relative_path = pred_file.relative_to(output_path)
    fname = str(relative_path).replace("_pred.npy", "")
    gt_image = imread(dataset_path / f"{fname}.png")
    # Load ground truth
    gt_image = my_rgb2lab(gt_image)
    
    # Load prediction
    pred = np.load(pred_file)
    pred = my_rgb2lab(pred)

    min_h = min(gt_image.shape[0], pred.shape[0])
    min_w = min(gt_image.shape[1], pred.shape[1])

    gt_image = gt_image[:min_h, :min_w, :]
    pred = pred[:min_h, :min_w, :]
    # Calculate absolute difference (L1 norm across channels)
    diff = np.linalg.norm(gt_image - pred, ord=1, axis=-1)
    
    # Define output path
    diff = diff.astype(np.float16)
    save_file = diffs_path / f"{fname}.npy"

    # Create the subdirectory (e.g., "Animal") if it doesn't exist
    save_file.parent.mkdir(parents=True, exist_ok=True) 

    np.save(save_file, diff)

print(f"Diffs saved to {diffs_path}")