import os
from collections import Counter
import os
root_dir = "data/npy" 

def get_folder_size(folder_path):
    """Return total size of all files in a folder (in bytes)."""
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                if os.path.isfile(fp):  # Skip broken symlinks
                    total_size += os.path.getsize(fp)
            except Exception as e:
                print(f"Could not access {fp}: {e}")
    return total_size

def list_folder_sizes(parent_dir):
    """Iterate through subfolders and print their sizes."""
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path):
            size_bytes = get_folder_size(item_path)
            size_mb = size_bytes / (1024 * 1024)
            print(f"{item}: {size_mb:.2f} MB")

if __name__ == "__main__":
    # target_directory = "C:\\Program Files" 
    # list_folder_sizes(target_directory)
    
    for cls in os.listdir(root_dir):
        cpath = os.path.join(root_dir, cls)
        n = sum(1 for r,_,f in os.walk(cpath) for x in f if x.endswith(".npy"))
        print(f"{cls:12s}: {n} samples")

