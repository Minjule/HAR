# original dataset ee uurt heregteigeer uurchluh 
"""
knife -> 0
scissors -> 1
axe -> 2
socket -> 3
windows -> 4
sckrewdriver -> 5
"""
import os
from collections import Counter
import os

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

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

def remove_label_from_dataset(root_dir, class_idxs_to_remove=None):
    """
    Remove all samples whose label file's first non-whitespace character is in class_idxs_to_remove.

    Args:
        root_dir: dataset root containing class folders
        class_idxs_to_remove: iterable of characters (e.g. ["0","3"]) or a comma-separated string "0,3"
    Returns:
        dict with counts: {'removed_txt': int, 'removed_imgs': int, 'skipped': int}
    """
    if class_idxs_to_remove is None:
        print("[WARN] no class_idxs_to_remove provided, nothing to do.")
        return {"removed_txt": 0, "removed_imgs": 0, "skipped": 0}

    # normalize to set of single-char strings
    if isinstance(class_idxs_to_remove, str):
        class_idxs = {c.strip() for c in class_idxs_to_remove.split(",") if c.strip() != ""}
    else:
        class_idxs = {str(c) for c in class_idxs_to_remove}

    removed_txt = 0
    removed_imgs = 0
    skipped = 0

    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for folder_name in sorted(os.listdir(cls_path)):
            label_path = os.path.join(cls_path, folder_name)
            if not os.path.isdir(label_path):
                continue
            for name in sorted(os.listdir(label_path)):
                if not name.lower().endswith(".txt"):
                    continue
                txt_path = os.path.join(label_path, name)
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    # find first non-whitespace character
                    stripped = content.lstrip()
                    first_ch = stripped[0]
                    if first_ch in class_idxs:
                        os.remove(txt_path)
                        removed_txt += 1
                        # remove matching image(s) with same base name in several candidate locations
                        base = os.path.splitext(name)[0]
                        img_path = os.path.join(cls_path, "images", base + ".jpg")
                        os.remove(img_path)
                        removed_imgs += 1
                except Exception as e:
                    print(f"[ERR] Processing {txt_path}: {e}")

    summary = {"removed_txt": removed_txt, "removed_imgs": removed_imgs, "skipped": skipped}
    print(f"[INFO] done. removed {removed_txt} .txt files, removed {removed_imgs} image files, skipped {skipped} files.")
    return summary

def update_mapping(root_dir, mapping=None):
    updated_txt = 0
    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for folder_name in sorted(os.listdir(cls_path)):
            label_path = os.path.join(cls_path, folder_name)
            if not os.path.isdir(label_path):
                continue
            for name in sorted(os.listdir(label_path)):
                if not name.lower().endswith(".txt"):
                    continue
                txt_path = os.path.join(label_path, name)
                try:
                    with open(txt_path, "r+", encoding="utf-8") as f:
                        content = f.read()
                        first_ch = content[0]
                        if first_ch in mapping:
                            new_ch = mapping[first_ch]
                            new_content = new_ch + content[1:]
                            f.seek(0)
                            f.truncate()
                            f.write(new_content)
                            updated_txt += 1
                except Exception as e:
                    print(f"[ERR] updating the label {txt_path}: {e}")
    print(f"[INFO] done. Updated {updated_txt} .txt files.")

def number_of_instances_train_test_val(root_dir):
    """Count and print number of label files and image files per class and totals."""
    label_counts = Counter()
    img_counts = Counter()
    total_labels = 0
    total_images = 0

    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        lbl_cnt = 0
        img_cnt = 0
        for folder_name in sorted(os.listdir(cls_path)):
            label_path = os.path.join(cls_path, folder_name)
            if not os.path.isdir(label_path):
                continue
            for name in sorted(os.listdir(label_path)):
                lname = name.lower()
                if lname.endswith(".txt"):
                    lbl_cnt += 1
                elif any(lname.endswith(ext) for ext in IMAGE_EXTS):
                    img_cnt += 1
        label_counts[cls] = lbl_cnt
        img_counts[cls] = img_cnt
        total_labels += lbl_cnt
        total_images += img_cnt

    print("Number of instances per class:")
    for cls in sorted(label_counts.keys()):
        print(f" Class '{cls}': {label_counts[cls]} label files, {img_counts[cls]} images")
    print(f"Totals: {total_labels} label files, {total_images} images")

def count_first_chars_per_class(root_dir):
    """
    For each class folder under root_dir:
      - walks into each subfolder and reads .txt files
      - extracts the first non-whitespace character from each txt
      - counts occurrences and distinct characters per class
    Prints per-class counters and global totals.
    """
    global_counter = Counter()
    classes_stats = {}

    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        cls_counter = Counter()
        # iterate subfolders (existing dataset layout)
        for folder_name in sorted(os.listdir(cls_path)):
            label_path = os.path.join(cls_path, folder_name)
            if not os.path.isdir(label_path):
                continue
            for fname in sorted(os.listdir(label_path)):
                if not fname.lower().endswith(".txt"):
                    continue
                txt_path = os.path.join(label_path, fname)
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    # find first non-whitespace char
                    stripped = content.lstrip()
                    if not stripped:
                        continue
                    first_ch = stripped[0]
                    cls_counter[first_ch] += 1
                    global_counter[first_ch] += 1
                except Exception as e:
                    print(f"[ERR] reading {txt_path}: {e}")

        classes_stats[cls] = cls_counter

    # print detailed per-class stats
    print("First-char label counts per class:")
    for cls, counter in classes_stats.items():
        distinct = len(counter)
        total = sum(counter.values())
        print(f" Class '{cls}': {total} labels, {distinct} distinct first-chars -> {dict(counter)}")
    print(f"Global totals: {sum(global_counter.values())} labels, {len(global_counter)} distinct first-chars -> {dict(global_counter)}")
    return classes_stats, global_counter

if __name__ == "__main__":
    root_dir = "D:\\HAR\\harmful_12"
    mapping = {"0": "2", "8": "0", "9": "1", "10": "5"}
    #remove_label_from_dataset(root_dir, class_idxs_to_remove=["1", "2", "3", "4", "5", "6", "7", "11"])
    #update_mapping(root_dir, mapping)
    #number_of_instances_train_test_val(root_dir)
    count_first_chars_per_class(root_dir)

