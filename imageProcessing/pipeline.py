import os, re
import tqdm
import rasterio
import numpy as np
import pandas as pd

from pathlib import Path
from rasterio.warp import *
from .utilities import *

#####################################################################################

def create_msk_lbl_img(img_paths, labels, bbox, proj_dir, msk_out_dir, lbl_out_dir, img_out_dir, res=None, nrow=None, ncol=None, binary_mask=True, prim_crop=['maize','rice']):
    r""" Create rasterized mask, labels, and images for chipping uses

    Arguments:
    ----------
    img_paths (list): list of paths to orthophotos
    labels (geopandas Dataframe): dataframe of labels of each orthophoto
    bbox (geopandas Dataframe): dataframe of bounding boxes of each orthophoto
    proj_dir (str): path to project directory
    msk_out_dir (str): path to output directory for masks
    lbl_out_dir (str): path to output directory for labels
    img_out_dir (str): path to output directory for images
    res (float): resolution of output images. Warning: Using the res option might result in tiles with different sizes(nrows, ncols)
    nrow (int): number of rows for output images. Must be specified if res is None
    ncol (int): number of columns for output images. Must be specified if res is None
    binary_mask (bool): whether to use binary mask or not. Selecting one of the prmary crop as the positive class is recommended
    prim_crop (list): list of primary crops to be used for binary mask. Default is ['maize','rice'].

    Returns:
    --------
    img_list (list): list of paths to output images
    lbl_list (list): list of paths to output labels
    msk_list (list): list of paths to output masks
    """
    
    
    
    # output resolution for images--you might want to change this
    # original was 2.5e-07
    # res = 10e-07 is used in this example.
    # Warning: Using the res option might result in tiles with different sizes(nrows, ncols)


    img_list = [] # list to catch output image paths
    lbl_list = [] # list to catch output label paths
    msk_list = [] # list to catch output mask paths

    for img in tqdm.tqdm(img_paths):
        ortho_idx = os.path.basename(img)
        ortho_labels = labels[labels.file_name == ortho_idx].reset_index()
        label_mask = bbox[bbox.file_name == ortho_idx].reset_index()

        #remove all 0 items
        ortho_labels = ortho_labels[~ortho_labels.is_empty]

        if len(label_mask) == 0:
            # print(f"Skipping {ortho_idx} because not labelled")
            continue

        # Make raster template, defining extent, resolution, and mask
        msk_name = f"mask_{ortho_idx}"
        msk_path = Path(msk_out_dir) / msk_name
        msk_list.append(re.sub(f"{str(proj_dir)}/", "", str(msk_path)))

        if not os.path.isfile(msk_path):
            # print(f"..Writing mask for {msk_name}")
            if res:
                dst_rst, dst_meta = make_rst_template(
                    label_mask, nrow=None, ncol=None, res=res, na=0, meta_only=False
                )
            elif res is None:
                assert nrow is not None and ncol is not None, "Must specify nrow and ncol if res is None"
                dst_rst, dst_meta = make_rst_template(
                    label_mask, nrow=nrow, ncol=ncol, res=None, na=0, meta_only=False
                )
            # write out mask file
            with rasterio.open(msk_path, "w", **dst_meta) as dst:
                dst.write(dst_rst, indexes=1)
        # else:
        #     print(f"..{msk_name} exists, skipping")

        # Make image labels
        lbl_name = f"label_{ortho_idx}"
        lbl_path = Path(lbl_out_dir) / lbl_name
        if len(ortho_labels) > 0:
            if not os.path.isfile(lbl_path):
                # print(f"..Making labels for {lbl_name}")
                dst_meta['nodata'] = None
                msk_lbls = rasterize_mask_labels(
                    ortho_labels, label_mask, dst_meta, None, lbl_out_dir, lbl_name,
                    na=255, binary_mask=binary_mask, prim_crop=prim_crop
                )
            # else:
            #     print(f"..{lbl_name} exists, skipping")

            lbl_list.append(
                f'{re.sub(str(proj_dir) + "/", "", str(lbl_out_dir))}/{lbl_name}'
            )
        else:
            # print(f"..No labels for {ortho_idx}, you can use mask for labels")
            lbl_list.append("None")

        # Process orthophoto (crop, resample, mask)
        # orthophoto
        img_name = f"img_{ortho_idx}"
        img_path = Path(img_out_dir) / img_name
        img_list.append(
            f'{re.sub(str(proj_dir) + "/", "", str(img_path))}'
        )
        if not os.path.isfile(img_path):
            # print(f"..Cropping, resampling {ortho_idx}")
            crop_resample_image(img, dst_rst, dst_meta, label_mask, 0, np.uint16,
                                img_out_dir, img_name)
        # else:
        #     print(f"..{img_name} exists, skipping")

    return img_list, lbl_list, msk_list

#####################################################################################

def train_test_split_cat(img_list, lbl_list, msk_list, proj_dir, train_prop=0.8):
    r""" Create training and validation catalogs
    
    Arguments:
    ----------
    img_list (list): list of paths to output images
    lbl_list (list): list of paths to output labels
    msk_list (list): list of paths to output masks
    proj_dir (str): path to project directory
    train_prop (float): proportion of orthophotos to use for training

    Returns:
    --------
    chipping_cat (pandas dataframe): dataframe of training and validation catalogs

    """
    p = train_prop  # proportion of orthophotos to use for training
    chipping_cat = pd.DataFrame(
        {"labels": lbl_list, "masks": msk_list, "images": img_list}
    )
    train = chipping_cat[chipping_cat.labels != "None"].sample(
        frac=p, random_state=1).copy()
    train['usage'] = "train"
    validate = chipping_cat[~chipping_cat.images.isin(train.images)].copy()
    validate['usage'] = "validate"

    chipping_cat = pd.concat([train, validate], axis=0, ignore_index=True)
    chipping_cat.to_csv(
        Path(proj_dir) / "working" / "chipping_catalog.csv"
    )


    # chipping_cat = chipping_cat.drop(index = 50)
    return chipping_cat

#####################################################################################

def chipping(proj_dir, lbl_chip_dir, img_chip_dir, out_format='tif', chipping_cat=None, patch_size=256, overlap=32, positive_class_threshold=0.1):


    r""" Chip images and labels into smaller images for training and validation
    
    Arguments:
    ----------
    proj_dir (str): path to project directory
    lbl_chip_dir (str): path to output directory for labels
    img_chip_dir (str): path to output directory for images
    out_format (str): format of output images. Options are 'tif', 'npz', and 'pkl'. Default is 'tif'
    chipping_cat (pandas dataframe): dataframe of training and validation catalogs. If None, it will be read from proj_dir/working/chipping_catalog.csv
    patch_size (int): size of output images. Default is 256
    overlap (int): overlap between output images. Default is 32
    positive_class_threshold (float): threshold for positive class. Default is 0.1

    Returns:
    --------
    chipping_cat (pandas dataframe): dataframe of training and validation catalogs
    Chipped images and labels will be written to disk as geotiffs if out_format is 'tif' and as npz or pkl files if out_format is 'npz' or 'pkl', respectively.

    """


    assert out_format in ["tif", "npz", "pkl"], "out_format must be geotiff, npz, or pkl"

    patch_size = patch_size
    overlap = overlap
    positive_class_threshold = positive_class_threshold

    if chipping_cat is None:
        # print(f"Reading in chipping catalog")
        chipping_cat = pd.read_csv(
            Path(proj_dir) / "working" / "chipping_catalog.csv"
        )

    chips_cat = []
    usage_list = []
    img_chips_list = []
    lbl_chips_list = []
    for i, lrow in chipping_cat.iterrows():
    # Need to remove .loc (only runs through 5 lines)
        # print(f"Processing {lrow['labels']}")

        if lrow['labels'] == "None":
            print(f"..no labels in {lrow['images']}, continuing")
            continue

        else:
            # print(f"..chipping {lrow['labels']}")
            # Read in mask, labels, image, normalize image, and get indexes for
            # centers of pixels meeting criteria for minimum coverage by positive
            # class
            idx, meta, msk, lbl, img = label_mask_image_loader(
                lrow, proj_dir, patch_size, overlap, positive_class_threshold,
                verbose=False
            )

            # Process labels and images into chips, written to disk as geotiffs
            cat, lbl_chips, img_chips = chipper(
                lrow, idx, meta, patch_size, lbl, img, proj_dir, lbl_chip_dir,
                img_chip_dir, out_format
            )
            usage_list.append(lrow['usage'])
            img_chips_list.append(img_chips)
            lbl_chips_list.append(lbl_chips)
            chips_cat.append(cat)

    if out_format in ["npz", "pkl"]:
        packing_data(usage_list, img_chips_list, lbl_chips_list, out_format, proj_dir)

    chips_catdf = pd.concat(chips_cat, ignore_index=True)
    chips_catdf.to_csv(Path(proj_dir) / "working" / "chips_catalog.csv")



