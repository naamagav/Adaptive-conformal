# Adapted from experiments-conformal-superres by Eduardo Adame, Daniel Csillag, Guilherme Tegoni Goedert
# Compute average mask coverage statistics

import os
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision.transforms import CenterCrop
from skimage.color import rgb2lab
from skimage.transform import rescale
from imageio.v3 import imread
from PIL import Image
from tqdm import tqdm
import pandas as pd
from icecream import ic
from scipy.signal import convolve2d  # type: ignore
from scipy.stats import bootstrap  # type: ignore
from tueplots import figsizes, fonts
from skimage.filters import gaussian


root_path = Path(__file__).parent.parent


dataset_basename = "LIU4K_v2_valid"
subdatasets = ["Animal", "Building", "Mountain", "Street"]

downscaled_path = root_path / "datasets" / "LR" / dataset_basename 
output_path = root_path / "output_sinsr_adaptive" / dataset_basename
dataset_path = root_path / "datasets" / dataset_basename
diffs_path = root_path / "masks" / dataset_basename
table_path = root_path / "table_data"
table_path.mkdir(exist_ok=True)


valid_fnames = [
    fname.parent.name + "/" + fname.stem
    for fname in dataset_path.resolve().glob("**/*.png")
    if fname.is_file() and fname.parent.name != "Capture"
    and fname.parent.name in subdatasets
]

non_semantic_kernel_sizes = [64]

mask = "ker31_runs10_varmask"
configs = [
    {
        "name": "Non-semantic $D_p$",
        "save_path": table_path
        / f"avg_mask_fidelity_non-semantic_dim31_adaptive.csv",
        "thresholds_path": root_path
        / f"conformal_adaptive/sinsr-thresholds-k31-d64-m64-bs-fix.json",
        "kernel_size": kernel_size,
    }
    for i, kernel_size in enumerate(non_semantic_kernel_sizes)
]


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

prob_masks = [
    my_left_corner_crop(np.load(output_path / f"{i}_{mask}.npy"))
    for i in tqdm(valid_fnames, desc="load masks")
]

diffs = [
        my_left_corner_crop(np.load(diffs_path / f"{i}.npy"))
        for i in tqdm(valid_fnames, desc="load diffs")
    ]

for config in tqdm(configs, desc="configurations"):
    sigma = config["kernel_size"]


    gdiffs = [
        gaussian(diff, sigma=sigma) for diff in tqdm(diffs, desc="gaussian diffs")
    ]

    if len(configs) == 1:
        del diffs
        print("Deleted diffs to save memory")


    gprob_masks = [
    gaussian(np.clip(mask, 0, np.quantile(mask, 0.95)), sigma=sigma)
        for mask in tqdm(prob_masks, desc="clip masks")
    ]

    if len(configs) == 1:
        del prob_masks
        print("Deleted prob_masks to save memory")
    

    alphas = []
    method_fidelity_error = []
    method_fidelity_error_interval = []

    avg_mask_size_error = []
    avg_mask_size_error_interval = []
    with open(config["thresholds_path"], "r", encoding="utf-8") as file:
        thresholds = json.load(file)

    for alpha, threshold in tqdm(thresholds.items()):
        alpha = float(alpha)
        if alpha > 0.50:            
            continue

        samples = [
            safemax(gdiff[gprob_mask < threshold])
            for gdiff, gprob_mask in tqdm(zip(gdiffs, gprob_masks), desc="load samples")
        ]

        eff_samples = [np.mean(gprob_mask > threshold) for gprob_mask in gprob_masks]

        coverage_err = infer_mean(samples)
        mask_size_err = infer_mean(eff_samples)

        alphas.append(alpha)
        method_fidelity_error.append(coverage_err[0])
        method_fidelity_error_interval.append(coverage_err[1])
        avg_mask_size_error.append(mask_size_err[0])
        avg_mask_size_error_interval.append(mask_size_err[1])

    samples = [safemax(gdiff) for gdiff in tqdm(gdiffs, desc="load samples")]

    base_infer = infer_mean(samples)

    baseline_fidelity_error = [base_infer[0]] * len(alphas)
    baseline_fidelity_error_interval = [base_infer[1]] * len(alphas)

    df = pd.DataFrame(
        {
            "FidelityLevel": alphas,
            "BaselineFidelityError": baseline_fidelity_error,
            "BaselineFidelityErrorInterval": baseline_fidelity_error_interval,
            "MethodFidelityError": method_fidelity_error,
            "MethodFidelityErrorInterval": method_fidelity_error_interval,
            "AverageConformalMaskSize": avg_mask_size_error,
            "AverageConformalMaskSizeInterval": avg_mask_size_error_interval,
        }
    )

    # Save DF
    config["df"] = df
    df.to_csv(config["save_path"], index=False)

