# Authors: Eduardo Adame, Daniel Csillag, Guilherme Tegoni Goedert
# Main calibration script for LIU4K dataset using conformal prediction

import os
import json
from pathlib import Path

import numpy as np
from torchvision.transforms import CenterCrop
from skimage.color import rgb2lab, gray2rgb
from skimage.transform import rescale
from imageio.v3 import imread
from PIL import Image
from tqdm import tqdm

from src.conformal import ConformalCalibratedModel
from skimage.filters import gaussian

root_path = Path(__file__).parent 


dataset_basename = "LIU4K_v2_train"
subdatasets = ["Animal", "Building", "Mountain", "Street"]

downscaled_path = root_path / "datasets" / "LR" / dataset_basename 
output_path = root_path / "output_sinsr_adaptive" / dataset_basename
dataset_path = root_path / "datasets" / dataset_basename
diffs_path = root_path / "masks" / dataset_basename


def my_rgb2lab(x):
    if len(x.shape) == 2:
        x = gray2rgb(x)
    x = rgb2lab(x)
    return x / np.array([[[100, 127 * 2, 127 * 2]]]) + np.array([[[0, 0.5, 0.5]]])


def my_center_crop(x):
    return np.asarray(CenterCrop(1024)(Image.fromarray(x.astype(np.uint8))))


def my_left_corner_crop(x):
    return x[:2048, :2048]


valid_fnames = [
    fname.parent.name + "/" + fname.stem
    for fname in diffs_path.resolve().glob("**/*.npy")
    if fname.is_file()
]


diffs = [
    np.load(diffs_path / f"{i}.npy")
    for i in tqdm(valid_fnames, desc="load diffs")
]

mask_sigmas =  [64,] 
diff_sigmas =  [64,]
kernel_sizes = [
    31,
]

for diff_sigma in diff_sigmas:
    gdiffs = [
        gaussian(diff, sigma=diff_sigma) for diff in tqdm(diffs, desc="gaussian diffs")
    ]

    if len(diff_sigmas) == 1:
        del diffs 
        print("Deleted diffs to save memory")

    for kernel_size in kernel_sizes:

        prob_masks = [
            np.load(output_path / f"{i}_ker{kernel_size}_runs10_varmask.npy") 
            for i in tqdm(valid_fnames, desc=f"load masks")
        ]

        for mask_sigma in mask_sigmas:
            print(
                f"kernel_size: {kernel_size}, mask_sigma: {mask_sigma}, diff_sigma: {diff_sigma}"
            )

            # clip masks into .95 quantile
            gprob_masks = [
                gaussian(np.clip(mask, 0, np.quantile(mask, 0.95)), sigma=mask_sigma)
                for mask in tqdm(prob_masks, desc="clip masks")
            ]

            if mask_sigma == mask_sigmas[-1]:
                del prob_masks
                print("Deleted prob_masks to save memory")

            dummy_preds = [None] * len(gprob_masks)

            conformal = ConformalCalibratedModel.calibrate(
                None,
                None,
                zip(dummy_preds, gprob_masks),
                alphas=list(np.linspace(0.025, 0.50, 39))
                + [
                    2.5,
                ],
                diffs=gdiffs,
                method="bisection",
            )

            save_path = root_path / "conformal_adaptive" 
            save_path /= f"sinsr-thresholds-k{kernel_size}-d{diff_sigma}-m{mask_sigma}-bs-fix.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "w") as file:
                json.dump(conformal.thresholds, file)
