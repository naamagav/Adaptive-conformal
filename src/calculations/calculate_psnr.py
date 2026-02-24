# Authors: Eduardo Adame, Daniel Csillag, Guilherme Tegoni Goedert
# PSNR metric computation for super-resolution evaluation

import os
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision.transforms import CenterCrop
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import rescale
from imageio.v3 import imread
from PIL import Image
from tqdm import tqdm
import pandas as pd
from icecream import ic
from scipy.signal import convolve2d  # type: ignore
from typing import Callable, List, Tuple
from scipy.stats import bootstrap  # type: ignore
from skimage.filters import gaussian

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true")
args = parser.parse_args()
force = args.force
Image.MAX_IMAGE_PIXELS = None

root_path = Path(__file__).parent.parent


dataset_basename = "LIU4K_v2_valid"
subdatasets = ["Animal", "Building", "Mountain", "Street"]

downscaled_path = root_path / "datasets" / "LR" / dataset_basename 
sr_path = root_path / "output_sinsr_adaptive" / dataset_basename
hr_path = root_path / "datasets" / dataset_basename
diffs_path = root_path / "masks" / dataset_basename
psnr_path = root_path / "psnr"
psnr_path.mkdir(exist_ok=True)

matplotlib.rc("text.latex", preamble=r"\usepackage{amsmath}")

COLORS = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]


valid_fnames = [
    fname.parent.name + "/" + fname.stem
    for fname in hr_path.resolve().glob("**/*.png")
    if fname.is_file() and fname.parent.name != "Capture"
    and fname.parent.name in subdatasets
]
valid_fnames.sort()

dfs = []
kernel_size = 31
mask_filename_suffix = "ker31_runs10_varmask"


def my_left_corner_crop(x):
    return x[:2048, :2048]


def my_rgb2lab(x):
    x = rgb2lab(x)
    return x / np.array([[[100, 127 * 2, 127 * 2]]]) + np.array([[[0, 0.5, 0.5]]])


def safemax(x):
    if len(x) == 0:
        return 0
    else:
        return np.max(x)


def safemean(x):
    if len(x) == 0:
        return 0
    else:
        return np.mean(x)


def my_psnr(orig, gen, mask=None):
    if mask is None:
        mask = np.ones(orig.shape[:2], dtype=bool)
    if orig.ndim == 3 and mask.ndim == 2:
        mask = np.repeat(mask[:, :, np.newaxis], orig.shape[2], axis=2)
    diff_sq = (orig - gen) ** 2
    mse = safemean(diff_sq[mask])
    if mse == 0:
        return float("inf")
    # max_val = safemax(orig[mask])
    max_val = 1.0 # since the images are normalized to [0, 1], the max pixel value is 1.0
    if max_val == 0:
        return 0
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def infer_mean(x: list[float]) -> str:
    result = bootstrap(
        (x,),
        np.mean,
        confidence_level=0.95,
        n_resamples=500,
        method="basic",
        random_state=np.random.default_rng(),
    )
    return (
        result.confidence_interval.low + result.confidence_interval.high
    ) * 0.5, result.confidence_interval.high - result.confidence_interval.low


alphas = []
psnr_semantic = {}
psnr_non_semantic = {}

with open(
    root_path / f"conformal_adaptive/sinsr-thresholds-k{kernel_size}-d64-m64-bs-fix.json"
) as file:
    thresholds_non_semantic = json.load(file)


configs = [
    {
        "name": "non-semantic",
        "samples": psnr_non_semantic,
        "thresholds": thresholds_non_semantic,
    },

]
mask_sigma = 64
for base in tqdm(valid_fnames, desc="Calibration images"):
    hr_img = Image.open(hr_path / f"{base}.png").convert("RGB")
    hr_array = np.array(hr_img)
    hr_array = my_left_corner_crop(hr_array) / 255

    sr_array = np.load(sr_path / f"{base}_pred.npy")
    mask = np.load(sr_path / f"{base}_{mask_filename_suffix}.npy")
    sr_array = my_left_corner_crop(sr_array) / 255

    mask = my_left_corner_crop(mask)

    mask = gaussian(np.clip(mask, 0, np.quantile(mask, 0.95)), sigma=mask_sigma)

    for config in configs:
        thresholds = config["thresholds"]
        if config["samples"].get("baseline") is None:
            config["samples"]["baseline"] = []
        psnr = my_psnr(hr_array, sr_array, None)
        if not (psnr == float("inf")):
            config["samples"]["baseline"].append(psnr)

        for alpha, threshold in thresholds.items():
            alpha = float(alpha)
            if alpha > 0.50 or alpha < 0.05:
                continue
            if config["samples"].get(alpha) is None:
                config["samples"][alpha] = []
            psnr = my_psnr(hr_array, sr_array, mask < threshold)
            if psnr == float("inf"):
                continue
            config["samples"][alpha].append(psnr)

    del hr_img, hr_array, sr_array, mask

# save the samples
for config in configs:
    psnr_means = [infer_mean(v) for v in config["samples"].values()]
    baseline_psnr = psnr_means[0]
    psnr_means = psnr_means[1:]

    lower_bound = -20 * np.log10(np.array(list(config["samples"].keys())[1:]))
    df = pd.DataFrame(
        {
            "FidelityLevel": list(config["samples"].keys())[1:],
            "LowerBound": lower_bound,
            "PSNR": [psnr_mean[0] for psnr_mean in psnr_means],
            "PSNRInterval": [psnr_mean[1] for psnr_mean in psnr_means],
            "BaselinePSNR": [baseline_psnr[0]] * len(psnr_means),
            "BaselinePSNRInterval": [baseline_psnr[1]] * len(psnr_means),
        }
    )
    df.to_csv(psnr_path / f"psnr_{config['name']}_dim{kernel_size}_adaptive.csv", index=False)
