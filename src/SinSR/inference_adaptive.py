# SinSR model adaptive inference wrapper
# Modified from the original SinSR codebase: https://github.com/wyf0912/SinSR

import os, sys
import argparse
from pathlib import Path
from tqdm import trange
import torch
import torch.nn.functional as nnf
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import shutil
from skimage.filters import threshold_otsu
import time

from omegaconf import OmegaConf
from sampler import Sampler

from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url



def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument(
        "-o", "--out_path", type=str, default="./results", help="Output path."
    )
    parser.add_argument(
        "-r", "--ref_path", type=str, default=None, help="reference image"
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=15,
        help="Diffusion length. (The number of steps that the model trained on.)",
    )
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument(
        "-is",
        "--infer_steps",
        type=int,
        default=None,
        help="Diffusion length for inference",
    )
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--one_step", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument(
        "--chop_size",
        type=int,
        default=512,
        choices=[512, 256],
        help="Chopping forward.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="SinSR",
        choices=["SinSR", "realsrx4", "bicsrx4_opencv", "bicsrx4_matlab"],
        help="Chopping forward.",
    )
    parser.add_argument("--ddim", action="store_true")
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)


    args = parser.parse_args()
    if args.infer_steps is None:
        args.infer_steps = args.steps
    print(f"[INFO] Using the inference step: {args.steps}")
    return args


def get_configs(args):
    if args.config is None:
        if args.task == "SinSR":
            configs = OmegaConf.load("./configs/SinSR.yaml")
        elif args.task == "realsrx4":
            configs = OmegaConf.load("./configs/realsr_swinunet_realesrgan256.yaml")
    else:
        configs = OmegaConf.load(args.config)
    # prepare the checkpoint
    ckpt_dir = Path("./weights")
    if args.ckpt is None:
        if not ckpt_dir.exists():
            ckpt_dir.mkdir()
        if args.task == "SinSR":
            ckpt_path = ckpt_dir / f"SinSR_v1.pth"
        elif args.task == "realsrx4":
            ckpt_path = ckpt_dir / f"resshift_{args.task}_s{args.steps}_v1.pth"
    else:
        ckpt_path = Path(args.ckpt)
    print(f"[INFO] Using the checkpoint {ckpt_path}")

    if not ckpt_path.exists():
        if args.task == "SinSR":
            load_file_from_url(
                url=f"https://github.com/wyf0912/SinSR/releases/download/v1.0/{ckpt_path.name}",
                model_dir=ckpt_dir,
                progress=True,
                file_name=ckpt_path.name,
            )
        else:
            load_file_from_url(
                url=f"https://github.com/zsyOAOA/ResShift/releases/download/v2.0/{ckpt_path.name}",
                model_dir=ckpt_dir,
                progress=True,
                file_name=ckpt_path.name,
            )
    vqgan_path = ckpt_dir / f"autoencoder_vq_f4.pth"
    if not vqgan_path.exists():
        load_file_from_url(
            url="https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth",
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
        )

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.timestep_respacing = args.infer_steps
    configs.diffusion.params.sf = args.scale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    if args.chop_size == 512:
        chop_stride = 448
    elif args.chop_size == 256:
        chop_stride = 224
    else:
        raise ValueError("Chop size must be in [512, 384, 256]")

    return configs, chop_stride

def calculate_variance_map_lab(img1_lab, img2_lab, img3_lab, kernel_size=31):
    """
    Calculates variance between three Lab images using the moment formula:
    Var(X) = E[X^2] - (E[X])^2.
    
    Args:
        img1_lab, img2_lab, img3_lab: Tensors of shape (C, H, W) in Lab space.
    """
    # Stack images: (3, C, H, W)
    imgs = torch.stack([img1_lab, img2_lab, img3_lab], dim=0)
    
    var_mask = imgs # (3, C, H, W)
    
    # E[Smooth(X^2)]
    term1 = nnf.avg_pool2d(
        var_mask**2, kernel_size=kernel_size, padding=kernel_size // 2, stride=1
    )
    
    # Smooth(E[X]))^2  
    term2 = nnf.avg_pool2d(
        var_mask, kernel_size=kernel_size, padding=kernel_size // 2, stride=1
    ) ** 2
    
    # logic: Smooth(X^2) - (Smooth(X))^2
    # This approximates local variance.
    local_var = term1 - term2
    
    # Collapse (3, C, H, W) -> Mean over N(3) and C(3) -> (H, W)
    # We want a single 2D map for decision making.
    return local_var.mean(dim=(0, 1)).cpu().numpy()

def generate_multiple_outputs():
    # Load the model configuration
    args = get_parser()
    configs, chop_stride = get_configs(args)
    sampler = Sampler(
        configs,
        chop_size=args.chop_size,
        chop_stride=chop_stride,
        chop_bs=1,
        use_fp16=True,
        seed=args.seed,
        ddim=args.ddim,
    )

    # Ensure output directory exists

    # Loop through input images
    # input_images = list(Path(input_folder).glob("*.png"))  
    input_master = Path(args.in_path)
    output_master = Path(args.out_path)
    # paths = [p for p in input_master.glob('*') if p.is_dir()]
    # for path in paths:

    # --- Parameters ---
    SCOUT_RUNS = 3            
    PATCH_SIZE = args.chop_size
    SCALE = args.scale
    KERNEL_SIZE = 31  
    t_start_total = time.time()      

    # --- Phase 1: Scout Runs (Full Image) ---
    print(f"[Info] Running Scout Phase ({SCOUT_RUNS} runs)...")
    for m in range(SCOUT_RUNS):
        output_folder = output_master / f"output_{m}"
        output_folder.mkdir(parents=True, exist_ok=True)
        
        if not (output_folder / "completed.marker").exists():
            with torch.no_grad():
                sampler.inference(
                    input_master, output_folder, bs=args.batch_size, 
                    noise_repeat=False, one_step=args.one_step
                )

    # --- Phase 2: Analyze Variance & Select Patches ---
    print("[Info] Analyzing variance (Lab Space)...")
    img_files = sorted(list((output_master / "output_0").glob("*.png")))
    
    for img_file in img_files:
        fname = img_file.name
        stem = img_file.stem
        
        # Load images
        path0 = output_master / "output_0" / fname
        path1 = output_master / "output_1" / fname
        path2 = output_master / "output_2" / fname
        
        # Read as RGB (0-255)
        img0_rgb = np.array(Image.open(path0).convert("RGB"))
        img1_rgb = np.array(Image.open(path1).convert("RGB"))
        img2_rgb = np.array(Image.open(path2).convert("RGB"))
        
        # Convert to LAB
        img0_lab = rgb2lab(img0_rgb).astype(np.float32)
        img1_lab = rgb2lab(img1_rgb).astype(np.float32)
        img2_lab = rgb2lab(img2_rgb).astype(np.float32)

        
        # Convert to Tensor [C, H, W]
        t0 = torch.from_numpy(img0_lab).permute(2, 0, 1).cuda()
        t1 = torch.from_numpy(img1_lab).permute(2, 0, 1).cuda()
        t2 = torch.from_numpy(img2_lab).permute(2, 0, 1).cuda()
        
        # Calculate Variance Map 
        var_map = calculate_variance_map_lab(t0, t1, t2, kernel_size=KERNEL_SIZE)

        
        # Downsample to Patch Grid to find "Bad Patches"
        # We use simple average pooling on the calculated variance map
        var_map_tensor = torch.from_numpy(var_map).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        
        patch_grid = nnf.avg_pool2d(
            var_map_tensor, 
            kernel_size=PATCH_SIZE, 
            stride=PATCH_SIZE
        ).squeeze() # (Grid_H, Grid_W)

        # --- SMART THRESHOLD CALCULATION ---
        scores = patch_grid.cpu().numpy().flatten()
        
        # Safety check: if the image is perfectly stable (variance is 0), skip everything
        if scores.max() <= 1e-6:
             VARIANCE_THRESH = 9999.0 # Impossible threshold
             print(f"Image {fname}: Completely stable (Max variance ~0). Skipping all.")
        else:
             # Otsu finds the natural separation point
             try:
                 VARIANCE_THRESH = threshold_otsu(scores)
                 # Otsu can sometimes be too aggressive. 
                 # we multiply it by a safety factor (e.g. 0.8) to include a bit more context.
                 VARIANCE_THRESH *= 0.8 
             except Exception:
                 # Fallback if statistical separation fails
                 VARIANCE_THRESH = 5.0 
                 
        print(f"Image {fname}: Auto-Threshold = {VARIANCE_THRESH:.5f}")

        # Create a binary mask (1 = Active/Re-run, 0 = Stable/Skipped)
        # We perform the comparison on the GPU tensor before moving to cpu

        decision_mask = (patch_grid > VARIANCE_THRESH).cpu().numpy().astype(np.uint8)
        
        # Save to the output directory
        mask_save_path = output_master / f"{stem}_decision_mask.npy"
        np.save(mask_save_path, decision_mask)
        print(f"Saved decision mask to: {mask_save_path}")

        # Identify Active Patches
        h_grid, w_grid = patch_grid.shape
        total_patches = h_grid * w_grid
        print(f"Total patches available: {total_patches}")
        active_patches = []
        
        for gy in range(h_grid):
            for gx in range(w_grid):
                score = patch_grid[gy, gx].item()
                
                # Check Threshold
                if score > VARIANCE_THRESH:
                    # Coordinates
                    hr_y, hr_x = gy * PATCH_SIZE, gx * PATCH_SIZE
                    hr_y_end, hr_x_end = hr_y + PATCH_SIZE, hr_x + PATCH_SIZE
                    
                    lr_y, lr_x = hr_y // SCALE, hr_x // SCALE
                    lr_y_end, lr_x_end = hr_y_end // SCALE, hr_x_end // SCALE
                    
                    active_patches.append((lr_x, lr_y, lr_x_end, lr_y_end))

        print(f"Image {fname}: Found {len(active_patches)} active patches (Thresh: {VARIANCE_THRESH}).")

        # --- Phase 3: Adaptive Inference ---
        if len(active_patches) > 0:
            temp_in_dir = output_master / "temp_crops_in"
            temp_out_dir = output_master / "temp_crops_out"
            
            if temp_in_dir.exists(): shutil.rmtree(temp_in_dir)
            if temp_out_dir.exists(): shutil.rmtree(temp_out_dir)
            temp_in_dir.mkdir(parents=True, exist_ok=True)
            temp_out_dir.mkdir(parents=True, exist_ok=True)
            
            original_lr = Image.open(input_master / fname).convert("RGB")
            patch_map = {} 

            # Save Crops
            for idx, (x, y, xe, ye) in enumerate(active_patches):
                crop = original_lr.crop((x, y, xe, ye))
                crop_name = f"{stem}_p{idx:05d}.png"
                crop.save(temp_in_dir / crop_name)
                patch_map[crop_name] = (x, y, xe, ye)

            # Run for remaining M
            for m in range(SCOUT_RUNS, args.M):
                print(f"Generating output {m} (Adaptive)...")
                current_crop_out = temp_out_dir / f"run_{m}"
                current_crop_out.mkdir(parents=True, exist_ok=True)
                
                # Inference on crops
                with torch.no_grad():
                    sampler.inference(
                        temp_in_dir, current_crop_out, bs=args.batch_size, 
                        noise_repeat=False, one_step=args.one_step
                    )
                
                # Reconstruct
                base_img = Image.open(path0).copy() # Start with stable Run 0
                
                for crop_name, (lx, ly, rx, ry) in patch_map.items():
                    pred_path = current_crop_out / crop_name
                    
                    if pred_path.exists():
                        patch_up = Image.open(pred_path)
                        base_img.paste(patch_up, (lx * SCALE, ly * SCALE))
                    else:
                        print(f"Error: Expected output file missing: {pred_path}")

                
                final_folder = output_master / f"output_{m}"
                final_folder.mkdir(exist_ok=True)
                base_img.save(final_folder / fname)
        else:
            print(f"Image {fname}: Stable. Copying Run 0.")
            for m in range(SCOUT_RUNS, args.M):
                final_folder = output_master / f"output_{m}"
                final_folder.mkdir(exist_ok=True)
                shutil.copy(path0, final_folder / fname)

    total_elapsed = time.time() - t_start_total
    print(f"TOTAL ADAPTIVE TIME: {total_elapsed:.2f} s")
    print(f"TOTAL ADAPTIVE TIME: {total_elapsed/3600:.2f} h")

def main():
    args = get_parser()

    configs, chop_stride = get_configs(args)

    resshift_sampler = Sampler(
        configs,
        chop_size=args.chop_size,
        chop_stride=chop_stride,
        chop_bs=1,
        use_fp16=True,
        seed=args.seed,
        ddim=args.ddim,
    )

    resshift_sampler.inference(
        args.in_path, args.out_path, bs=1, noise_repeat=False, one_step=args.one_step
    )


if __name__ == "__main__":
    # main()
    generate_multiple_outputs()
