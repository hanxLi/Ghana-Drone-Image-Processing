import os, re
from pathlib import Path
import math, random
import pickle
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt

from skimage import exposure
from rasterio.warp import *
from rasterio import features
from rasterio.mask import mask
from rasterio.crs import CRS

#####################################################################################

def list_full_path(folder, pattern=None):
    r"""List full paths of files in a folder.

    Arguments:
    ----------
    folder (str): the folder to list files from.
    pattern (str): a regular expression pattern to match files with.

    Returns:
    --------
    list: a list of full paths of files in the folder.
    """
    files = [f for f in os.listdir(folder) if not f.startswith('.')]
    
    if pattern:
        file_paths = [f'{folder}/{file}' for file in files
                      if re.search(pattern, file)]
    else:
        file_paths = [f'{folder}/{file}' for file in files]
    return file_paths

#####################################################################################

def make_rst_template(ply, res=None, nrow=None, ncol=None, na=0,
                      meta_only=True):
    r"""Make a raster template based on a Shapely Polygon.

    Arguments:
    ----------
    ply (Shapely.Polygon): the polygon.
    nrow (int): the height of the image.
    ncol (int): the width of the image.
    res (numeric): Desired resolution of output image.
        Used in place of nrow, ncol
    meta_only (bool) : Whether to produce meta data only or raster
        template also

    Returns:
    --------
    tuple: a raster numpy.ndarray and a meta dictionary.
    """
    #commented out old debugging code, kept for future debugging.
    # # convert ply to numpy array
    # ply_arr = np.array(ply.geometry.apply(lambda geom: np.array(geom.exterior.coords)).values.tolist())

    # # check for NaN values in ply_arr
    # nan_count = np.count_nonzero(np.isnan(ply_arr))
    # print("the nan_count is", nan_count)
    # if nan_count > 0:
    #     print(f"WARNING: {nan_count} NaN values detected in the input polygon.")

    minx, miny, maxx, maxy = ply['geometry'].total_bounds
    # Add print statements to check the values of minx, maxx, miny, and maxy
    # print(f"minx: {minx}, maxx: {maxx}, miny: {miny}, maxy: {maxy}")
    # print("ply is a: ", ply)


    if res:
        # print("Using resolution")
        dst_transform = (res, 0.0, minx, 0.0, -res, maxy)
        ncol = int((maxx - minx) / res)
        nrow = int((maxy - miny) / res)

    else:
        dst_transform = ((maxx - minx) / ncol, 0.0, minx, 0.0,
                         -(maxx - minx) / ncol, maxy)
    meta = ({
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': na,
        'width': ncol,
        'height': nrow,
        'count': 1,
        'crs': CRS.from_epsg(ply.crs.to_epsg()),
        'transform': dst_transform})

    if meta_only:
        return meta
    else:
        rst = features.rasterize(ply['geometry'], out_shape=[nrow, ncol],
                                fill=na, dtype=meta['dtype'],
                                transform=meta['transform'])
        return rst, meta

#####################################################################################

def rasterize_mask_labels(labels, mask, dst_meta=None, res=None,
                          out_folder=None, outname=None, na=None, nrow=None,
                          ncol=None, binary_mask=True, prim_crop=['maize','rice']):

    r"""Rasterize polygon labels to extent of mask and mask out NA areas

    Arguments:
    ----------
    labels (GeoDataFrame) : Labels to be rasterized
    mask (GeoDataFrame) : Mask to be used to clip labels
    dst_meta (dict) : Rasterio metadata dictionary providing extent and
        resolution of output
    res (float) : Resolution of output raster
    out_folder (str) : Where we are writing this
    outname (str) : Optional name to write. Otherwise it is composed of fields
        from the tile
    na (int) : Value to set NA areas to
    nrow (int) : Number of rows in output raster
    ncol (int) : Number of columns in output raster
    binary_mask (binary) : If set to True, the label mask will be binary
    prim_crop (list) : List of primary crops to be used in rasterizing labels

    Returns:
    -------
    Labels on disk and output numpy array and rasterio meta

    """

    if not dst_meta and res:
        dst_meta = make_rst_template(mask, res=res, na=None, nrow=None,
                                     ncol=None)
    elif not dst_meta and not res:
        print("If dst_meta is not supplied you must provide res")
        return
    # print(dst_meta)

    class_list = np.unique(labels.prim_crop)
    other_crops = [x for x in class_list if x not in prim_crop]
    
    
########################    
# All Binary Mask Method
    if binary_mask:    
        clipped_labels = gpd.clip(labels.buffer(0), mask)
        geom = [shapes for shapes in clipped_labels.geometry]
    
########################
# Crop Specific Mask Method
    elif not binary_mask:
        geom = []
        for i in range(len(labels)):
            
            temp_geom = gpd.clip(labels.loc[[i]].buffer(0), mask).reset_index()

            # print(temp_geom.geometry.empty)
            if not temp_geom.geometry.empty:
                if labels.loc[i].prim_crop in other_crops:
                    geom.append([temp_geom.geometry[0], 0])
                else:
                    for j in range(len(prim_crop)):
                        if labels.loc[i].prim_crop == prim_crop[j]:
                            geom.append([temp_geom.geometry[0], 3])
    else:
        assert ValueError, "binary_mask must be a boolean"
            
    
########################
    if not outname:
        grid_name = list(mask.grid_name)
        fname = f"label_{grid_name}.tif"
        outpath = os.path.join(out_folder, fname)

    else:
        outpath = os.path.join(out_folder, f"{outname}")

    with rasterio.open(outpath, 'w+', **dst_meta) as dst:
        out_arr = dst.read(1)
        burned = features.rasterize(
            geom, out=out_arr, fill=0, transform=dst.transform,
            all_touched = True, default_value=1
        )
        dst.write_band(1, burned)

    with rasterio.open(outpath) as src:
        out_image, out_trans = rasterio.mask.mask(
            src, mask['geometry'], nodata=na, crop=True
        )
        out_meta = src.meta

    out_meta['nodata'] = na
    out_meta['dtype'] = 'uint8'
    with rasterio.open(outpath, "w", **out_meta) as dst2:
        dst2.write(out_image)

    return {"labels": out_image, "meta": out_meta}

#####################################################################################

def crop_resample_image(img_path, dst_rst, dst_meta, tile, nodata,
                        dst_dtype, out_folder, outname=None):
    r"""Crop and resample image to match mask

    Arguments:
    ----------
    img_path (str) : Path to image to be cropped and resampled
    dst_rst (numpy.ndarray) : Rasterized mask to crop and resample image to
    dst_meta (dict) : Rasterio metadata dictionary providing extent and
        resolution of output
    tile (GeoDataFrame) : Tile to crop and resample image to
    nodata (int) : Value to set NA areas to
    dst_dtype (str) : Data type of the output image chips
    out_folder (str) : Where we are writing this
    outname (str) : Optional name to write. Otherwise it is composed of fields
        from the tile
    
    Returns:
    --------
    Image on disk and output numpy array and rasterio meta
    """

    if not outname:
        fname = f"{tile.grid_name}_rs.tif"
        outpath = os.path.join(out_folder, fname)
    else:
        outpath = os.path.join(out_folder, outname)

    with rasterio.open(img_path) as src:
        imgdat = src.read()
        img_meta = src.meta

    img_dst_rst = np.stack((dst_rst,) * imgdat.shape[0], axis=0)

        # with rasterio.open(outpath, 'w', **dst_meta) as dst:
    reproject(imgdat, img_dst_rst, src_transform=img_meta['transform'],
              src_crs=img_meta['crs'],
              dst_transform=dst_meta['transform'],
              dst_crs=dst_meta['crs'], resampling=Resampling.bilinear,
              num_threads=os.cpu_count() - 2)
    #print(dst_meta)
    #return img_dst_rst

    with rasterio.open(outpath,'w', driver='GTiff', width=dst_meta['width'],
                       height=dst_meta['height'], count=imgdat.shape[0],
                       dtype=dst_dtype, nodata=nodata,
                       transform=dst_meta['transform'],
                       crs=dst_meta['crs']) as dst:
        dst.write(img_dst_rst)

    with rasterio.open(outpath) as src2:
        out_image, out_trans = rasterio.mask.mask(
            src2, tile['geometry'], nodata=65535, crop=True
        )
    out_meta = src2.meta
    out_meta['nodata'] = 65535
    # out_meta['dtype'] = dst_dtype
    with rasterio.open(outpath, "w", **out_meta) as dst2:
        dst2.write(out_image)

    return {"labels": out_image, "meta": out_meta}

#####################################################################################

def patch_center_index(cropping_ref, patch_size, overlap, usage, binary_mask=True,
                       positive_class_threshold=None, verbose=False):
    r"""
    Generate index to divide the scene into small chips.
    Each index marks the location of corresponding chip center.

    Arguments:
    ----------
    cropping_ref (list) : Reference raster layers, to be used to generate the
        index. In our case, it is study area binary mask and label mask.
    patch_size (int) : Size of each clipped patches.
    overlap (int) : amount of overlap between the extracted chips.
    usage (str) : Either 'train', 'val'. Chipping strategy is different for
        different usage.
    binary_mask (binary) : If set to True, the mask is assumed to be binary
    positive_class_threshold (float) : a real value as a threshold for the
        proportion of positive class to the total areal of the chip. Used to
        decide if the chip should be considered as a positive chip in the
        sampling process.
    verbose (binary) : If set to True prints on screen the detailed list of
        center coordinates of the sampled chips.

    Returns:
    --------
    proportional_patch_index : A list of index recording the center of
        patches to extract from the input
    """

    assert usage in ["train", "validate"]

    mask, label = cropping_ref

    max_lbl = label.max()

    half_size = patch_size // 2
    step_size = patch_size - 2 * overlap

    proportional_patch_index = []
    non_proportional_patch_index = []
    neg_patch_index = []

    # Get the index of all the non-zero elements in the mask.
    x = np.argwhere(mask)

    # First col of x shows the row indices (height) of the mask layer (iterate over
    # the y axis or latitude).
    y_min = min(x[:, 0]) + half_size
    y_max = max(x[:, 0]) - half_size
    # Second col of x shows the column indices (width) of the mask layer
    # (iterate over the x axis or longitude).
    x_min = min(x[:, 1]) + half_size
    x_max = max(x[:, 1]) - half_size

    # Generate index for the center of each patch considering the proportion of
    # each category falling into each patch.
    for row in range(y_min, y_max + 1, step_size):

        for col in range(x_min, x_max + 1, step_size):

            # Split the mask and label layers into patches based on the index
            # of the center of the patch
            mask_ref = mask[row - half_size: row + half_size,
                            col - half_size: col + half_size]
            label_ref = label[row - half_size: row + half_size,
                              col - half_size: col + half_size]
            # plt.imshow(mask_ref)
            if (usage == "train") and mask_ref.all():
                print("Passed mask_ref check...")
                if label_ref.any() != 0:
                    print("Passed label_ref check...")
                    if binary_mask:
                        pond_ratio = np.sum(label_ref == 1) / label_ref.size
                    else:
                        sum_val = 0
                        for i in range(1, max_lbl):
                            if i in label_ref:
                                sum_val += np.sum(label_ref == i)
                        
                        pond_ratio = sum_val / label_ref.size

                    if pond_ratio >= positive_class_threshold:

                        proportional_patch_index.append([row, col])
                else:
                    print("failed label_ref check...")
                    neg_patch_index.append([row, col])
                
            if (usage == "validate") and (label_ref.any() != 0) and mask_ref.all():
                non_proportional_patch_index.append([row, col])

    if usage == "train":

        num_negative_samples = min(
            math.ceil(0.2 * len(proportional_patch_index)), 5
        )
        # print(num_negative_samples, len(proportional_patch_index))

        # if num_negative_samples <= len(neg_patch_index):
        #     num_negative_samples = len(neg_patch_index)

        if num_negative_samples > 0 and len(neg_patch_index) > 0:
            if num_negative_samples <= len(neg_patch_index):
                neg_samples = random.sample(neg_patch_index, num_negative_samples)
                # print(neg_samples)
                proportional_patch_index.extend(neg_samples)

    # For test set use the indices generated from mask without considering the
    # class proportions.
    if usage == "validate":
        proportional_patch_index = non_proportional_patch_index

    if verbose:
        print("Number of negative patches:", len(neg_patch_index))
        print("Number of negative samples:", num_negative_samples)
        print("Number of patches:", len(proportional_patch_index))
        print("Patched from:\n{}".format(proportional_patch_index))

    return proportional_patch_index

#####################################################################################

def min_max_normalize_image(image):
    r""" Normalizes the input image to the range [0, 1] using min-max
    Arguments:
    ------
    image(str) : Input numpy array--should already be a float with NA set to
      np.nan

    Returns:
    -------
    Normalized image as np.float32
    """

    # Calculate the minimum and maximum values for each band
    min_values = np.nanmin(image, axis=(1, 2))[:, np.newaxis, np.newaxis]
    max_values = np.nanmax(image, axis=(1, 2))[:, np.newaxis, np.newaxis]

    # Normalize the image data to the range [0, 1]
    normalized_img = (image - min_values) / (max_values - min_values)

    # Return the normalized image data
    return normalized_img

#####################################################################################

def load_data(data_path, is_label=False, apply_normalization=False,
              na=65535, dtype=np.float32, verbose=False):
    r"""
    Open data using rasterio, read it as an array and normalize it.

    Arguments:
    ----------
    data_path (str) : Full path including filename of the data source we
        wish to load.
    is_label (bool) : If True then the layer is a ground truth (category
          index) and if set to False the layer is a reflectance band.
    na (int or np.nan) : Integer value of NA in imagery. np.nan if float.
    apply_normalization (bool) : If true min/max normalization will be
          applied on each band.
    dtype (np.dtype) : Data type of the output image chips.
    verbose (binary) : if set to true, print a screen statement on the
          loaded band.

    Returns:
    --------
    image : Returns the loaded image as a 32-bit float numpy ndarray.
    """

    # Inform user of the file names being loaded from the Dataset.
    if verbose:
        print('loading file:{}'.format(data_path))

    # open dataset using rasterio library.
    with rasterio.open(data_path, "r") as src:

        if is_label:
            if src.count != 1:
                raise ValueError("Expected Label to have exactly one channel.")
            img = src.read(1)

        else:
            img_data = src.read()
            if src.count != 3:
                print("Warning: Expected Image to have exactly three channels.")
                img_data = img_data[:3, :, :]
            if apply_normalization:
                img = np.where(img_data == na, np.nan, img_data)
                img = min_max_normalize_image(img)
                img = img.astype(dtype)
            else:
                img = img_data.astype(dtype)

    return img

#####################################################################################

def update_meta_transform_xy(meta, left, top, nrow, ncol):
    r""" Updates the transform and size of a rasterio meta dictionary

    Arguments:
    ----------
    meta (dict) : Rasterio metadata dictionary
    left (float) : Left coordinate of the chip
    top (float) : Top coordinate of the chip
    nrow (int) : Number of rows in the chip
    ncol (int) : Number of columns in the chip

    Returns:
    --------
    meta (dict) : Updated rasterio metadata dictionary
    """
    new_meta = meta
    tr = list(new_meta['transform'])
    tr[2] = left  # replace transform lon with left
    tr[5] = top  # replace transform lat with top
    newtrans = rasterio.Affine(tr[0], tr[1], tr[2], tr[3], tr[4], tr[5])
    new_meta['transform'] = newtrans
    new_meta['height'] = nrow
    new_meta['width'] = ncol
    return new_meta

#####################################################################################

def plot_image_and_label(image_array, label_array, band_composite=None, stretch=None):
    r"""
    Plots a loaded image and its corresponding label using matplotlib and rasterio.plot.

    Arguments:
    ----------
    image_array (numpy.ndarray) -- A numpy array containing image data.
    label_array (numpy.ndarray) -- A numpy array containing label data.
    stretch (str) -- A string specifying the contrast stretch to apply to the
      image (e.g. 'linear', 'sqrt', or 'log').
    band_composite (tuple of int) -- A tuple specifying the band indices to use
      for an RGB image (e.g. (3, 2, 1)).

    Returns:
    --------
    None, plots image and label side by side using matplotlib and rasterio.plot
    """
    # Set up the figure and axes
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

    image_size=[image_array.shape[0], image_array.shape[1]]
    label_size=[label_array.shape[0], label_array.shape[1]]

    # If a band composite is specified, create an RGB image
    red_band = image_array[:, :, band_composite[0] - 1]
    green_band = image_array[:, :, band_composite[1] - 1]
    blue_band = image_array[:, :, band_composite[2] - 1]

    # Stack the bands to create an RGB image
    rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)

    # Apply the contrast stretch if specified
    if stretch:
        p_min, p_max = np.percentile(rgb_image, (2, 98))
        stretched = exposure.rescale_intensity(rgb_image,
                                               in_range=(p_min, p_max),
                                               out_range=(0, 1))
    else:
        stretched = rgb_image

    # Display the RGB image using matplotlib
    axs[0].imshow(stretched)

    # Add a title and axis labels
    axs[0].set_title('Image')
    axs[0].set_xlabel(f'# Column: {image_size[0]}')
    axs[0].set_ylabel(f'# Row: {image_size[1]}')

    # Show the plot
    axs[0].axis('off')

    # Plot label chip
    #axs[1].imshow(np.repeat(label_array, 3, axis=0).transpose(1, 2, 0), cmap='gray')
    axs[1].imshow(label_array, cmap='viridis')
    #rp.show(label_array, ax=axs[1], cmap='viridis')
    axs[1].set_title('Label')
    axs[1].set_xlabel(f'# Column: {label_size[0]}')
    axs[1].set_ylabel(f'# Row: {label_size[1]}')

    # Show the plot
    axs[1].axis('off')
    plt.show()

#####################################################################################
    
def label_mask_image_loader(cat_row, proj_dir, patch_size, overlap,
                            positive_class_threshold, verbose=True, binary_mask=True):
    r"""Loads up mask, labels, and image for a given row to be processed in a
    catalog of input images

    Arguments
    ---------
    cat_row (pd_series) : One row of a pandas dataframe that is the chipping
      catalog. It should have columns containing the paths to masks, labels, and
      images, and one for usage.
    proj_dir (str) : The front part of the project path that will allow a full
      path to be made to each label/mask/image input
    patch_size (int) : Size of each clipped patches.
    overlap (int) : amount of overlap between the extracted chips.
    positive_class_threshold (float) : a real value as a threshold for the
        proportion of positive class to the total areal of the chip. Used to
        decide if the chip should be considered as a positive chip in the
        sampling process.
    verbose (binary) : If set to True prints on screen the detailed list of
        center coordinates of the sampled chips.
    binary_mask (binary) : If set to True, the mask is assumed to be binary

    Returns
    -------
    Proportional patch index, metadata for labels (for further modification),
    loaded mask, labels, and normalized image

    """
    # Load mask, labels, images for the ortho to be chipped
    msk = load_data(Path(proj_dir) / cat_row['masks'], is_label=True)
    lbl = load_data(Path(proj_dir) / cat_row['labels'], is_label=True)
    img = load_data(Path(proj_dir) / cat_row['images'],
                    apply_normalization=True)
    # if img.shape[0] == np.min(img.shape):
    #     img = img.transpose((1, 2, 0))

    # find chip indices
    index = patch_center_index([msk, lbl], patch_size, overlap,
                               cat_row['usage'], binary_mask=binary_mask, positive_class_threshold = positive_class_threshold,)
    lbl_meta = rasterio.open(Path(proj_dir) / cat_row['labels']).meta
    # msk_trans = rasterio.open(Path(proj_dir) / cat_row['mask']).transform

    return index, lbl_meta, msk, lbl, img

#####################################################################################

def chipper(cat_row, index, meta, patch_size, labels, image, proj_dir,
            label_dir, image_dir, out_format="tif"):
    r"""Loops through indexes and writes out label and image chips to geotiffs

    Arguments:
    ----------
    cat_row (pd_series) : One row of a pandas dataframe that is the chipping
      catalog. It should have columns containing the paths to masks, labels, and
      images, and one for usage.
    index (list) : List of patch center indices
    meta (dict) : Metadata of label image
    patch_size (int) : Size of each clipped patches
    labels (np.ndarray) : Label array
    image (np.ndarry) : Normalized image array in (band, row, col) order
    proj_dir (str) : Directory for main project
    label_dir (str) : Directory to write label chips into
    image_dir (str) : Directory to write image chips into
    out_format (str) : Format to write chips into, defaults to geotiff

    Returns:
    --------
    Chips written to disk when geotiff is selected as the output format, lists of chips and images as nd.arrays, pd.DataFrame
    with paths to chips and their usage

    """

    coor = []
    xys = []
    lbl_chips = []
    lbl_paths = []
    img_chips = []
    img_paths = []
    half_size = patch_size // 2
    transform = meta['transform']
    for i in range(len(index)):
        # dimensions
        col = index[i][1]  # col index (x)
        row = index[i][0]  # row index (y)
        coor.append([row, col])

        xs = [col - half_size, col + half_size]  # left and right column index
        ys = [row - half_size, row + half_size]  # top and bottom column index
        left, top = rasterio.transform.xy(transform, ys[0], xs[0], offset='ul') # xy long lat
        xys.append([left, top])

        # Use the x, y coordinates of the center indices to chip the image
        # and label and add each chip to its corresponding list.
        # Labels
        # Paths
        lbl_chip_name = re.sub(".tif", f"_{i}.tif",
                               os.path.basename(cat_row['labels']))
        lbl_chip_path = Path(label_dir) / lbl_chip_name
        lbl_paths.append(
            f'{re.sub(str(proj_dir) + "/", "", str(lbl_chip_path))}'
        )

        # create label chip and write out
        lbl_chip = labels[ys[0]:ys[1], xs[0]:xs[1]]
        lbl_chips.append(lbl_chip)
        lchp_meta = update_meta_transform_xy(
            meta.copy(), left, top, patch_size, patch_size
        )
        if out_format == "tif":
            with rasterio.open(lbl_chip_path, "w", **lchp_meta) as cdst:
                cdst.write(lbl_chip, indexes=1)

        # create image chip and write out
        img_chip_name = re.sub(".tif", f"_{i}.tif",
                               os.path.basename(cat_row['images']))
        img_chip_path = Path(image_dir) / img_chip_name
        img_paths.append(
            f'{re.sub(str(proj_dir) + "/", "", str(img_chip_path))}'
        )
        # note we keep rasterio order for creating the chip, to write it out.
        # (band, row, col). That should be transposed before going into loader
        img_chip = image[:, ys[0]:ys[1], xs[0]:xs[1]]
        # print(img_chip.shape)
        # assert img_chip.shape[0] != 3, f"There is a mismatch in image shape with {img_chip_name}, {img_chip.shape}"
            
        img_chips.append(img_chip)

        # write out image chip, after first updating metadata
        ichp_meta = lchp_meta.copy()
        ichp_meta['dtype'] = image.dtype
        ichp_meta['nodata'] = np.nan
        ichp_meta['count'] = 3
        if out_format == "tif":
            with rasterio.open(img_chip_path, "w", **ichp_meta) as idst:
                idst.write(img_chip)

    catalog = pd.DataFrame({"name": os.path.basename(cat_row["images"]),
                            "labels": lbl_paths, "images": img_paths,
                            "usage": cat_row['usage']})

    return catalog, lbl_chips, img_chips

#####################################################################################

def packing_data(usage_list, image_chips, lbl_chips, out_format, out_dir):
    r"""Packs up image and label chips into npz or pkl files for training
    
    Arguments:
    ----------
    usage_list (list) : List of strings indicating usage of each tile
    image_chips (list) : List of image chips as nd.arrays in (band, row, col) order
    lbl_chips (list) : List of label chips as nd.arrays in (row, col) order
    out_format (str) : Format to write chips into
    out_dir (str) : Directory to write chips into

    Returns:
    --------
    None, writes out npz or pkl files of image and label chips

    """


    training_data_img = []
    training_data_lbl = []
    validation_data_img = []
    validation_data_lbl = []

    for i in range(len(usage_list)):
        for j in range(len(image_chips[i])):
            if usage_list[i] == "train":
                training_data_img.append(image_chips[i][j])
                training_data_lbl.append(lbl_chips[i][j])
            if usage_list[i] == "validate":
                validation_data_img.append(image_chips[i][j])
                validation_data_lbl.append(lbl_chips[i][j])
    
    assert out_format in ["npz", "pkl"]
    
    if out_format == "npz":
        np.savez_compressed(Path(out_dir) / "image_chips/training_data_img.npz", training_data_img)
        np.savez_compressed(Path(out_dir) / "label_chips/training_data_lbl.npz", training_data_lbl)
        np.savez_compressed(Path(out_dir) / "image_chips/validation_data_img.npz", validation_data_img)
        np.savez_compressed(Path(out_dir) / "label_chips/validation_data_lbl.npz", validation_data_lbl)
    elif out_format == "pkl":
        with open(Path(out_dir) / "image_chips/training_data_img.pkl", "wb") as f:
            pickle.dump(training_data_img, f)
        with open(Path(out_dir) / "image_chips/validation_data_img.pkl", "wb") as f:
            pickle.dump(validation_data_img, f)
        with open(Path(out_dir) / "label_chips/training_data_lbl.pkl", "wb") as f:
            pickle.dump(training_data_lbl, f)
        with open(Path(out_dir) / "label_chips/validation_data_lbl.pkl", "wb") as f:
            pickle.dump(validation_data_lbl, f)

#####################################################################################

def plot_img_lbl_pair(img_path, lbl_path, band_composite=None, stretch=None):
    r""" Plots a loaded image and its corresponding label using matplotlib and rasterio.plot.

    Arguments:
    ----------
    img_path (str) : Full path including filename of the image source we wish to load.
    lbl_path (str) : Full path including filename of the label source we wish to load.
    stretch (str) : A string specifying the contrast stretch to apply to the image (e.g. 'linear', 'sqrt', or 'log').
    band_composite (tuple of int) : A tuple specifying the band indices to use for an RGB image (e.g. (3, 2, 1)).

    Returns:
    --------
    None, plots image and label side by side using matplotlib and rasterio.plot
    """
    
    # Using the plot_image_and_label function to plot the image and label

    with rasterio.open(img_path) as src:
        img = src.read()
        img = np.moveaxis(img, 0, -1)
    
    with rasterio.open(lbl_path) as src:
        lbl = src.read(1)

    plot_image_and_label(img, lbl, band_composite, stretch)
