#!/bin/bash
ROOT_PATH=$(realpath "$(dirname "$0")/..")
# --- Configuration ---
BASE_IN_PATH="$ROOT_PATH/datasets/LR/LIU4K_v2_train"
BASE_OUT_PATH="$ROOT_PATH/output_adaptive/LIU4K_v2_train"

# List of subfolders to process
SUBFOLDERS=("Animal" "Building" "Mountain" "Street")

export CUDA_VISIBLE_DEVICES=1

# Check CUDA environment once
echo "--- Checking CUDA Environment ---"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Torch Version: {torch.__version__}')"
echo "---------------------------------"

# --- Loop Through Subfolders ---
for subfolder in "${SUBFOLDERS[@]}"; do
    echo "Processing subfolder: $subfolder"
    
    # Define specific input and output paths
    CURRENT_IN_PATH="${BASE_IN_PATH}/${subfolder}"
    CURRENT_OUT_PATH="${BASE_OUT_PATH}/${subfolder}"
    
    # Create the output directory if it doesn't exist
    mkdir -p "$CURRENT_OUT_PATH"
    
    # Run the Command
    python inference_adaptive.py \
        --in_path "$CURRENT_IN_PATH" \
        --out_path "$CURRENT_OUT_PATH" \
        --chop_size 256 \
        --one_step \
        --M 10 

    echo "Finished processing: $subfolder"
    echo "---------------------------------"
done

echo "--- All Subfolders Finished ---"