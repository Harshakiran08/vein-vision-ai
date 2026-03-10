# ==========================================================
#
#  Prepare the HDF5 datasets of the STARE database
#
# ==========================================================

import os
import h5py
import numpy as np
from PIL import Image

def write_hdf5(arr, outfile):
    """Write a numpy array to an HDF5 file"""
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)

# ---------------- Paths of the images ---------------------
STARE_DIR = './STARE/'

# Training paths
original_imgs_train = os.path.join(STARE_DIR, "training/Images/")
groundTruth_imgs_train = os.path.join(STARE_DIR, "training/Labels/")
borderMasks_imgs_train = os.path.join(STARE_DIR, "training/Masks/")

# Testing paths
IMG_TEST_DIR = os.path.join(STARE_DIR, "testing/Images/")
groundTruth_imgs_test = os.path.join(STARE_DIR, "testing/Labels/")
MASK_TEST_DIR = os.path.join(STARE_DIR, "testing/Masks/")

# Output dataset path
dataset_path = "./stare_datasets/"
os.makedirs(dataset_path, exist_ok=True)

# ---------------- Image parameters -----------------------
channels = 3
height = 605
width = 700

# ---------------- Dataset loader -------------------------
def get_datasets(imgs_dir, groundTruth_dir, borderMasks_dir):
    """Load images, ground truth, and border masks into numpy arrays"""
    files = sorted([f for f in os.listdir(imgs_dir) if f.endswith('.ppm')])
    Nimgs = len(files)

    imgs = np.empty((Nimgs, height, width, channels), dtype=np.uint8)
    groundTruth = np.empty((Nimgs, height, width), dtype=np.uint8)
    border_masks = np.empty((Nimgs, height, width), dtype=np.uint8)

    for i, file in enumerate(files):
        base_name = os.path.splitext(file)[0]

        # Original image
        img_path = os.path.join(imgs_dir, file)
        img = Image.open(img_path).convert("RGB").resize((width, height))
        imgs[i] = np.asarray(img, dtype=np.uint8)

        # Ground truth
        g_path = os.path.join(groundTruth_dir, base_name + ".ah.ppm")
        if not os.path.exists(g_path):
            raise FileNotFoundError(f"Ground truth file not found: {g_path}")
        g_img = Image.open(g_path).convert("L").resize((width, height))
        g_arr = np.asarray(g_img, dtype=np.uint8)
        if g_arr.max() <= 1:
            g_arr = (g_arr * 255).astype(np.uint8)
        groundTruth[i] = g_arr

        # Border mask
        b_path = os.path.join(borderMasks_dir, base_name + "_mask.png")
        if not os.path.exists(b_path):
            raise FileNotFoundError(f"Border mask file not found: {b_path}")
        b_img = Image.open(b_path).convert("L").resize((width, height))
        b_arr = np.asarray(b_img, dtype=np.uint8)
        if b_arr.max() <= 1:
            b_arr = (b_arr * 255).astype(np.uint8)
        border_masks[i] = b_arr

    # Transpose and reshape
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    groundTruth = groundTruth.reshape(Nimgs, 1, height, width)
    border_masks = border_masks.reshape(Nimgs, 1, height, width)

    return imgs, groundTruth, border_masks

# ---------------- Prepare datasets -----------------------
# Training
imgs_train, groundTruth_train, border_masks_train = get_datasets(
    original_imgs_train, groundTruth_imgs_train, borderMasks_imgs_train
)
write_hdf5(imgs_train, os.path.join(dataset_path, "STARE_dataset_img_train.hdf5"))
write_hdf5(groundTruth_train, os.path.join(dataset_path, "STARE_dataset_groundTruth_train.hdf5"))
write_hdf5(border_masks_train, os.path.join(dataset_path, "STARE_dataset_mask_train.hdf5"))

# Testing
imgs_test, groundTruth_test, mask_test = get_datasets(
    IMG_TEST_DIR, groundTruth_imgs_test, MASK_TEST_DIR
)
write_hdf5(imgs_test, os.path.join(dataset_path, "STARE_dataset_img_test.hdf5"))
write_hdf5(groundTruth_test, os.path.join(dataset_path, "STARE_dataset_groundTruth_test.hdf5"))
write_hdf5(mask_test, os.path.join(dataset_path, "STARE_dataset_mask_test.hdf5"))

print("All datasets created successfully!")
