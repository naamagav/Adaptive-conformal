# Authors: Eduardo Adame, Daniel Csillag, Guilherme Tegoni Goedert
# Generate results tables for paper

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
table_path = root_path / "table_data"
psnr_path = root_path / "psnr"
psnr_path.mkdir(exist_ok=True)
table_path.mkdir(exist_ok=True)

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

dfs = []
kernel_size = 31
mask_name = "ker31_runs10_varmask"


alpha = 0.1
name = "non-semantic"
configs = [
    {
        "name": name,
        "psnr": pd.read_csv(psnr_path / f"psnr_{name}_dim{kernel_size}_adaptive.csv")[0:],
        "fidelity": pd.read_csv(
            table_path / f"avg_mask_fidelity_{name}_dim{kernel_size}_adaptive.csv"
        )[0:],
    }
]


for config in configs:
    config["psnr"]["FidelityLevel"] = config["psnr"]["FidelityLevel"].round(3)
    config["fidelity"]["FidelityLevel"] = config["fidelity"]["FidelityLevel"].round(3)
    baseline_psnr = config["psnr"]["BaselinePSNR"].iloc[0]
    
    baseline_psnr_interval = config["psnr"]["BaselinePSNRInterval"].iloc[0]
    avg_fidelity_error = config["fidelity"]["BaselineFidelityError"].iloc[0]
    avg_fidelity_error_interval = config["fidelity"][
        "BaselineFidelityErrorInterval"
    ].iloc[0]
    baseline = {
        "FidelityLevel": "W/o our method",
        "Avg. PSNR": rf"{baseline_psnr:.2f} $\pm$ {baseline_psnr_interval:.2f}",
        "Avg. Fidelity Error": rf"{avg_fidelity_error:.2f} $\pm$ {avg_fidelity_error_interval:.2f}",
        "Avg. Conformal Mask Size": np.nan,
    }

    # format avg as PSNR +- interval
    config["psnr"]["Avg. PSNR"] = config["psnr"][
        ["FidelityLevel", "PSNR", "PSNRInterval"]
    ].apply(lambda x: rf"{x['PSNR']:.2f} $\pm$ {x['PSNRInterval']:.2f}", axis=1)

    config["fidelity"]["Avg. Fidelity Error"] = config["fidelity"][
        ["FidelityLevel", "MethodFidelityError", "MethodFidelityErrorInterval"]
    ].apply(
        lambda x: rf"{x['MethodFidelityError']:.2f} $\pm$ {x['MethodFidelityErrorInterval']:.2f}",
        axis=1,
    )
    config["fidelity"]["Avg. Conformal Mask Size"] = config["fidelity"][
        [
            "FidelityLevel",
            "AverageConformalMaskSize",
            "AverageConformalMaskSizeInterval",
        ]
    ].apply(
        lambda x: rf"{x['AverageConformalMaskSize']:.2f} $\pm$ {x['AverageConformalMaskSizeInterval']:.2f}",
        axis=1,
    )

    config["psnr"] = config["psnr"][["FidelityLevel", "Avg. PSNR"]]
    config["fidelity"] = config["fidelity"][
        ["FidelityLevel", "Avg. Fidelity Error", "Avg. Conformal Mask Size"]
    ]

    joined = pd.merge(
        config["psnr"],
        config["fidelity"],
        on="FidelityLevel",
        suffixes=("_psnr", "_fidelity"),
    )

    # add baseline
    joined = pd.concat([pd.DataFrame([baseline]), joined], ignore_index=True)

    config["joined"] = joined

    print(joined.columns)


table = joined

selected_alphas = ["W/o our method", 0.075, 0.1, 0.2, 0.3]
filtered_table = table.loc[table["FidelityLevel"].isin(selected_alphas)].copy()

filtered_table["FidelityLevel"] = filtered_table["FidelityLevel"].apply(
    lambda x: rf"$\alpha = {x:.3f}$" if isinstance(x, float) else x
)

print(filtered_table)
# save csv
filtered_table.to_csv(table_path / f"table_{kernel_size}_adaptive.csv", index=False)
