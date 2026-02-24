from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

root_path = Path(__file__).parent.parent

def main():
    dataset_path = root_path / "datasets" / "LIU4K_v2_train" / "Animal"
    valid_fnames = sorted(list(dataset_path.glob("*.png")))
    
    if len(valid_fnames) == 0:
        print("Error: No images found!")
        return

    chosen_ids = [0]
    IS = [valid_fnames[i] for i in chosen_ids]

    # Create figure with 5 columns (Original, Runs10, Adaptive, Difference, Decision Mask)
    fig, axes = plt.subplots(len(IS), 5, figsize=(25, 5 * len(IS)), squeeze=False)


    for i, img_path in enumerate(IS):
        print(f"Processing: {img_path.name}...")

        img = np.array(Image.open(img_path))
        
        output_path = root_path / "output_sinsr" / "LIU4K_v2_train" / "Animal"
        output_path_adaptive = root_path / "output_sinsr_adaptive" / "LIU4K_v2_train" / "Animal"

        # Load the uncertainty maps
        map_runs10 = np.load(output_path / (img_path.stem + "_ker31_runs10_varmask.npy"))
        map_adaptive = np.load(output_path_adaptive / (img_path.stem + "_ker31_runs10_varmask.npy"))

        decision_mask_path = root_path / "output_adaptive/LIU4K_v2_train/Animal" / f"{img_path.stem}_decision_mask.npy" 
        
        if decision_mask_path.exists():
            decision_mask = np.load(decision_mask_path)
        else:
            print(f"Warning: Decision mask not found for {img_path.stem}")

        # Calculate Difference (Absolute difference to show magnitude of change) 
        diff_map = np.abs(map_runs10 - map_adaptive)  

        ax_row = axes[i] 

    
        # We determine max value across both maps to share the same color scale
        global_max = max(np.max(map_adaptive), np.max(map_runs10))

        def add_colorbar(im, ax, visible=True, position="right"):
            divider = make_axes_locatable(ax)
            # We append the axes to the requested side (left or right)
            cax = divider.append_axes(position, size="5%", pad=0.05)
            if visible:
                plt.colorbar(im, cax=cax)
            else:
                cax.axis("off") 
                # If it's on the left, we also turn off the frame so it's truly invisible
                cax.set_frame_on(False)

        # --- Column 1: Original Image ---
        im0 = ax_row[0].imshow(img)
        ax_row[0].set_title("Original Image", fontsize=24, y=1.05)
        ax_row[0].axis("off")
        add_colorbar(im0, ax_row[0], visible=False, position="left")
        
        # --- Column 2: Var Mask (10 Runs) ---
        im1 = ax_row[1].imshow(map_runs10, cmap="hot", vmin=0, vmax=global_max)
        ax_row[1].set_title("Var Mask (10 Runs)", fontsize=24, y=1.05)
        ax_row[1].axis("off")
        add_colorbar(im1, ax_row[1], visible=True, position="right")

        # --- Column 3: Adaptive ---
        im2 = ax_row[2].imshow(map_adaptive, cmap="hot", vmin=0, vmax=global_max)
        ax_row[2].set_title("Var Mask (Adaptive)", fontsize=24, y=1.05)
        ax_row[2].axis("off")
        add_colorbar(im2, ax_row[2], visible=True, position="right")

        # --- Column 4: Difference ---
        im3 = ax_row[3].imshow(diff_map, cmap="inferno") 
        ax_row[3].set_title("Difference\n|10 Runs - Adaptive|", fontsize=24, y=1.05)
        ax_row[3].axis("off")
        add_colorbar(im3, ax_row[3], visible=True, position="right")

        # --- Column 5: Decision Mask ---
        im4 = ax_row[4].imshow(diff_map, cmap="inferno") 
        ax_row[4].set_title("Difference\n|10 Runs - Adaptive|\nRed=Active\nBlue=Stable", fontsize=24, y=1.05)
        ax_row[4].axis("off")
        add_colorbar(im4, ax_row[4], visible=True, position="right")

        # ===  Draw Grid Overlay ===
        if decision_mask is not None:
            h_img, w_img = map_adaptive.shape[:2]
            h_grid, w_grid = decision_mask.shape
            
            # Calculate patch size relative to the variance map
            patch_h = h_img / h_grid
            patch_w = w_img / w_grid

            for gy in range(h_grid):
                for gx in range(w_grid):
                    is_active = decision_mask[gy, gx]
                    
                    # Style settings
                    if is_active:
                        edge_color = 'red'
                        line_style = '-'
                        line_width = 1
                        alpha = 1
                    else:
                        edge_color = 'deepskyblue' 
                        line_style = '-'
                        line_width = 1
                        alpha = 0.3

                    # Create Rectangle
                    # (x, y), width, height
                    rect = patches.Rectangle(
                        (gx * patch_w, gy * patch_h), 
                        patch_w, patch_h, 
                        linewidth=line_width, 
                        edgecolor=edge_color, 
                        linestyle=line_style,
                        facecolor='none', 
                        alpha=alpha
                    )
                    ax_row[4].add_patch(rect)

    # Save the new comparison figure
    save_path = root_path / "Results" / "figures" / "fig1.png"
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    
    print(f"\nCreated Uncertainty Comparison figure at:\n{save_path}")

if __name__ == "__main__":
    main()