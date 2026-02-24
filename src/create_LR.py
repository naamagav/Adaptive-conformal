
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
root_path = Path(__file__).parent
# Base path where the original 4K dataset is located
base_source_path = root_path / "datasets/LIU4K_v2_train"

# Base path where the Low Resolution versions will go
base_dest_path = root_path / "datasets/LR/LIU4K_v2_train"

# List of subfolders to process
subdatasets = ["Animal", "Building", "Mountain", "Street"]

for category in subdatasets:
    # Define source and destination for this specific category
    source_folder = base_source_path / category
    dest_folder = base_dest_path / category
    
    # Create the destination subfolder if it doesn't exist
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all PNG images 
    files = list(source_folder.glob("*.png"))
    
    print(f"Processing '{category}': Found {len(files)} images.")

    for file_path in tqdm(files, desc=f"Downsampling {category}"):
        try:
            with Image.open(file_path) as img:
                # Calculate new size (1/4 scale)
                new_w = img.width // 4
                new_h = img.height // 4
                
                # Resize using High Quality resampling (LANCZOS)
                img_lr = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # Save to the specific category folder in destination
                save_path = dest_folder / file_path.name
                img_lr.save(save_path)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

print("\nDone! All subdatasets processed.")
print("New LR root path:", base_dest_path)