# Authors: Eduardo Adame, Daniel Csillag, Guilherme Tegoni Goedert
# Variance-based uncertainty mask generation

import torch
import torch.nn.functional as nnf
import numpy as np
from pathlib import Path
from imageio.v3 import imread
from skimage.transform import rescale
from skimage.color import rgb2lab
from tqdm import tqdm
from icecream import ic

root_path = Path(__file__).parent.parent

dataset_base = "LIU4K_v2_train"
subdatasets = ["Animal", "Building", "Mountain", "Street"] 

runs = 10


def save_images(images, save_path: Path, filename: str = ""):
    directory = save_path / f"{filename}"
    directory.mkdir(exist_ok=True)

    for i, img in enumerate(images):
        np.save(directory / f"upscaled_{i}.npy", img)


@torch.compile
def calculate_mask(img, kernel_size, iters):
    var_mask = img
    for _ in range(iters):
        var_mask = (
            nnf.avg_pool2d(
                var_mask**2, kernel_size=kernel_size, padding=0, stride=1
            )
            - nnf.avg_pool2d(
                var_mask, kernel_size=kernel_size, padding=0, stride=1
            )
            ** 2
        )
    return var_mask.mean(axis=(0, 1)).cpu().numpy()


if __name__ == "__main__":
    # Load generated images

    for subdataset in subdatasets:
        dataset_fullname = dataset_base + "/" + subdataset

        result_folder_in = root_path / "output_adaptive" / dataset_fullname
        first_result_folder_in = result_folder_in / "output_0"

        mask_folder_out = root_path / "output_sinsr_adaptive" / dataset_fullname
        mask_folder_out.mkdir(parents=True, exist_ok=True)


        # result_folder_in / run / id.png
        for image in tqdm(first_result_folder_in.glob("*.png")):
            image_id = image.stem
            pred_file_path = mask_folder_out / f"{image_id}_pred.npy"

            generated_images = np.stack(
                [
                    imread(str(result_folder_in / f"output_{run}" / f"{image_id}.png"))
                    for run in range(runs)
                ]
            )
            np.save(pred_file_path, generated_images[0])
            generated_images = rgb2lab(generated_images)
            upscaled_tensor = (
                torch.tensor(generated_images, dtype=torch.float16)
                .permute(0, 3, 1, 2)
                .cpu()
            )  # [N, C, H, W]

            iters = 1
            for kernel_size in [31]: 
                mask_file_path = mask_folder_out / f"{image_id}_ker{kernel_size}_runs{runs}_varmask.npy"

                var_mask = upscaled_tensor
                var_mask_array = calculate_mask(var_mask, kernel_size, iters)
                np.save(mask_file_path, var_mask_array)
