# Adapted from experiments-conformal-superres by Eduardo Adame, Daniel Csillag, Guilherme Tegoni Goedert
# Generate Figure 2 

from pathlib import Path
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale
from imageio.v3 import imread
import json, os
from skimage.color import lab2rgb
from tueplots import figsizes, fonts
from skimage.filters import gaussian

matplotlib.rc("text", usetex=True)
matplotlib.rc("text.latex", preamble=r"\usepackage{amsmath}")
plt.rcParams.update({"figure.dpi": 300})
plt.rcParams.update(fonts.neurips2023())
plt.rcParams.update(figsizes.neurips2023())
plt.rcParams.update({"font.size": 24})


def my_left_corner_crop(x):
    return x[:2048, :2048]


root_path = Path(__file__).parent.parent


dataset_basename = "LIU4K_v2_valid"
subdatasets = ["Animal", "Building", "Mountain", "Street"]

downscaled_path = root_path / "datasets" / "LR" / dataset_basename #(dataset_basename + "_downscaled")
output_path_adaptive = root_path / "output_sinsr_adaptive" / dataset_basename
output_path = root_path / "output_sinsr" / dataset_basename
dataset_path = root_path / "datasets" / dataset_basename
diffs_path = root_path / "masks" / dataset_basename
fig_output_path = root_path / "Results" / "figures"
fig_output_path.mkdir(exist_ok=True)

valid_fnames = [
    fname.parent.name + "/" + fname.stem
    for fname in dataset_path.resolve().glob("**/*.png")
    if fname.is_file() and fname.parent.name != "Capture"
    and fname.parent.name in subdatasets]

valid_fnames.sort()

chosen_ids = [185, 83, 132]

IS = [valid_fnames[i] for i in chosen_ids]
LEVEL_1 = "0.1" 
with open(root_path / "conformal/sinsr-thresholds-k31-d64-m64-bs-fix.json") as file:
    THRESHOLD_non_adapt = json.load(file)[LEVEL_1]
with open(root_path / "conformal_adaptive/sinsr-thresholds-k31-d64-m64-bs-fix.json") as file:
    THRESHOLD_adapt = json.load(file)[LEVEL_1]


n_rows = len(IS)
n_cols = 7

fig, axs = plt.subplots(
    n_rows,
    n_cols,
    figsize=(24 , 4.25 * n_rows),
    squeeze=False,
    gridspec_kw={
        "wspace": 0.125,  # horizontal spacing between plots
        "hspace": 0.125,  # vertical spacing between plots
    },
    constrained_layout=True,
)
mask_sigma = 64
for row, I in enumerate(IS):

    real_image = (imread(dataset_path / f"{I}.png")) / 255


    pred = np.load(output_path_adaptive / f"{I}_pred.npy")
    pred = pred / 255.0

    real_image = my_left_corner_crop(real_image)
    pred = my_left_corner_crop(pred)
    low_res = rescale(real_image, 0.25, channel_axis=2)

    mask = np.load(output_path / f"{I}_ker31_runs10_varmask.npy") 
    mask_adaptive = np.load(output_path_adaptive / f"{I}_ker31_runs10_varmask.npy")
    mask_adaptive = gaussian(np.clip(mask_adaptive, 0, np.quantile(mask_adaptive, 0.95)), sigma=mask_sigma)
    mask = gaussian(np.clip(mask, 0, np.quantile(mask, 0.95)), sigma=mask_sigma)
    ALPHA = 0.3

    mask_adaptive = my_left_corner_crop(mask_adaptive)
    mask = my_left_corner_crop(mask)

    # Calculate prediction with conformal mask and pixel difference
    red_mask = np.stack(
        (np.ones((2048, 2048)), np.zeros((2048, 2048)), np.zeros((2048, 2048))),
        axis=2,
    )

    selection_nsem_non_adapt = np.stack((mask, mask, mask), axis=2) >= THRESHOLD_non_adapt
    pred_with_mask_nsem_non_adapt = np.where(
        selection_nsem_non_adapt, (1 - ALPHA) * red_mask + ALPHA * pred, pred
    )

    selection_nsem_adapt = np.stack((mask_adaptive, mask_adaptive, mask_adaptive), axis=2) >= THRESHOLD_adapt
    pred_with_mask_nsem_adapt = np.where(
        selection_nsem_adapt, (1 - ALPHA) * red_mask + ALPHA * pred, pred
    )

    pixel_diff = np.abs(pred - real_image)

    axs[row, 0].imshow(mask)
    axs[row, 1].imshow(mask_adaptive)
    axs[row, 2].imshow(low_res)
    axs[row, 3].imshow(real_image)
    axs[row, 4].imshow(pred_with_mask_nsem_non_adapt)
    axs[row, 5].imshow(pred_with_mask_nsem_adapt)
    # Enhance pixel difference visualization
    pixel_diff_gray = (
        np.mean(pixel_diff, axis=2) if len(pixel_diff.shape) == 3 else pixel_diff
    )
    p95_diff = np.percentile(pixel_diff_gray, 95)
    axs[row, 6].imshow(pixel_diff_gray, cmap="afmhot", vmin=0, vmax=p95_diff)


for ax in np.ravel(axs):
    ax.set_xticks([])
    ax.set_yticks([])

axs[0, 0].set_title(r"$\sigma$ non-adaptive" + "\n64-pixel-wide" + "\nGaussian blur", y=1.05)
axs[0, 1].set_title(r"$\sigma$ adaptive" + "\n64-pixel-wide\nGaussian blur", y=1.05)
axs[0, 2].set_title("Low resolution", y=1.05)
axs[0, 3].set_title("Ground truth", y=1.05)
axs[0, 4].set_title("Prediction with\nconformal mask\nnon-adaptive", y=1.05)
axs[0, 5].set_title("Prediction with\nconformal mask\nadaptive", y=1.05)
axs[0, 6].set_title(r"Pixel difference" + "\n" + r"($|$pred $-$ gt$|$)", y=1.05)

fig.savefig(fig_output_path / "fig2.png", bbox_inches="tight", dpi=150)
