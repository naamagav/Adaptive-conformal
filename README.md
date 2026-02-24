# Image Super-Resolution with Guarantees via Conformalized Generative Models
This repository contains the code for our adaptation of the Image Super-Resolution with Guarantees via Conformalized Generative Models project. We used an adaptive variance approach and compared our results to the standard variance calculation from the paper.

## Attribution

This code is heavily based on:
- [Image Super-Resolution with Guarantees via Conformalized Generative Models](https://github.com/adamesalles/experiments-conformal-superres) 
- [SinSR](https://github.com/wyf0912/SinSR)

## Installation

### Requirements
- Python 3.11 (required, <3.12)

### Setup
```bash
conda create -n super_res "python=3.11"
conda activate super_res
pip install -r requirements.txt
```

## Project Structure

### Core Modules

- **`src/base_model.py`**: Protocol definition for super-resolution model interface
- **`src/conformal.py`**: Core conformal calibration implementation with dynamic programming and bisection search methods for computing risk-controlling thresholds
- **`src/SinSR/inference_adaptive.py`**: Our adaptation for a faster and more efficient inference process.

### Calibration Scripts

- **`src/calibrate.py`**: Main calibration script for LIU4K dataset using conformal prediction

### Mask Generation

Uncertainty masks are generated based on variance across multiple super-resolution predictions:

- **`src/gen_masks/generate_masks.py`**: Variance-based mask generation

### Calculations and Metrics

- **`src/calculations/`**: Scripts for computing metrics
  - **`src/calculations/calculate_psnr.py`**: PSNR metric computation
  - **`src/calculations/calculate_avg_mask.py`**: Compute average mask coverage statistics and fidelity errors
  - **`src/calculations/calculate_table.py`**: Generate results tables

### Utilities and Setup

- **`src/create_LR.py`**: Utility to generate low-resolution images from datasets
- **`src/creating_diff_masks.py`**: Utility to generate difference masks between the full resolution images and the SinSR predictions
- **`src/figures/`**: Figure generation scripts (e.g., `fig1.py`, `fig2.py`)

## Usage

### Running SinSR Inference
The `SinSR/` directory contains the scripts for running the inference process. You can use the `run_folders.sh` file, or run the Python scripts directly. 

For the standard inference (M full runs), use:
```bash
cd src/SinSR

python inference.py \
    --in_path "$CURRENT_IN_PATH" \
    --out_path "$CURRENT_OUT_PATH" \
    --chop_size 256 \
    --one_step \
    --M 10
```

For the adaptive inference (3 full runs and M-3 adaptive runs), use:
```bash
cd src/SinSR

python inference_adaptive.py \
    --in_path "$CURRENT_IN_PATH" \
    --out_path "$CURRENT_OUT_PATH" \
    --chop_size 256 \
    --one_step \
    --M 10
```

### Generating Masks

```bash
# Generate uncertainty masks
python -m src.gen_masks.generate_masks
```

### Running Calibration

```bash
# Calibrate on LIU4K dataset
python -m src.calibrate
```

### Computing Metrics

```bash
# Calculate PSNR
python -m src.calculations.calculate_psnr

# Calculate average mask coverage and fidelity error
python -m src.calculations.calculate_avg_mask

# Calculate the full tables (Table 1 or Table 2)
python -m src.calculations.calculate_table
```

## Key Implementation Details

- **Color Space**: All metrics and conformal risk computations are performed in LAB color space (not RGB).
- **Risk Function**: Supremum of L1 error over confident regions in LAB space.
- **Data**: Calibration is performed on the train dataset. Evaluation (metrics calculation and conformal masks visualizations) is performed on the valid dataset.

## Data

This code expects datasets to be organized in the following structure:

```text
src/datasets/
  LIU4K_v2_train/
      Animal/
      Building/
      Mountain/
      Street/
  LIU4K_v2_valid/
      Animal/
      Building/
      Mountain/
      Street/
```

You can change the location of the datasets folder and change the `root_path` variable in all scripts accordingly.

## Citation

```bibtex
@inproceedings{adame2025conformal,
  title={Image Super-Resolution with Guarantees via Conformalized Generative Models},
  author={Adame, Eduardo and Csillag, Daniel and Goedert, Guilherme Tegoni},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}

@inproceedings{wang2024sinsr,
  title={Sinsr: diffusion-based image super-resolution in a single step},
  author={Wang, Yufei and Yang, Wenhan and Chen, Xinyuan and Wang, Yaohui and Guo, Lanqing and Chau, Lap-Pui and Liu, Ziwei and Qiao, Yu and Kot, Alex C and Wen, Bihan},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={25796--25805},
  year={2024}
}
```
