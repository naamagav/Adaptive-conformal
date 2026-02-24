# SinSR model inference wrapper
# Modified from the original SinSR codebase: https://github.com/wyf0912/SinSR

import os, sys
import argparse
from pathlib import Path
from tqdm import trange
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
    # input_images = list(Path(input_folder).glob("*.png"))  # Update for other formats as needed
    input_master = Path(args.in_path)
    output_master = Path(args.out_path)
    # paths = [p for p in input_master.glob('*') if p.is_dir()]
    # for path in paths:
    t_start_total = time.time()
    for m in trange(args.M):
        print(f"Generating output {m}")
        output_folder = output_master / f"output_{m}"
        output_folder.mkdir(parents=True, exist_ok=True)

        # Run inference
        sampler.inference(
            input_master,
            output_folder,
            bs=args.batch_size,
            noise_repeat=False,
            one_step=args.one_step,
        )

    total_elapsed = time.time() - t_start_total
    print(f"TOTAL TIME: {total_elapsed:.2f} s")
    print(f"TOTAL TIME: {total_elapsed/3600:.2f} h")

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