import os
import gdal 
import numpy as np
import geopandas as gpd
from SplittedImage.SplittedImage import SplittedImage
np.random.seed(19950304)

def normalize(x):
    x_min = x.min(axis=(1, 2), keepdims=True)
    x_max = x.max(axis=(1, 2), keepdims=True)
    return (x - x_min)/(x_max-x_min)

def load_tif_andSplit(tif_path, box_size=128):
    ds = gdal.Open(tif_path)
    if len(np.shape(ds.ReadAsArray())) == 2:
        img_src_arr = np.expand_dims(ds.ReadAsArray(), 0)
    else:
        img_src_arr = ds.ReadAsArray()
    X = np.transpose(img_src_arr, axes=[1,2,0])
    return SplittedImage(X, box_size, ds.GetGeoTransform(), ds.GetProjection())


def load_data(or_tif, gt_tif, box_size, val_ratio=0.2, normalizztion=False):
    splitted_gt = load_tif_andSplit(gt_tif)
    splitted_im = load_tif_andSplit(or_tif)
    tmp_X = splitted_im.get_splitted_images()
    tmp_y = splitted_gt.get_splitted_images()
    label_data_idx = np.nonzero(np.sum(tmp_y, axis=(1,2,3)))[0].tolist() # only using have label data
    val_list = np.random.choice(label_data_idx, int(len(label_data_idx)*val_ratio))
    train_list = np.array([item for item in label_data_idx if item not in val_list])

    X, X_val = tmp_X[train_list], tmp_X[val_list]
    y, y_val = tmp_y[train_list], tmp_y[val_list]
    if normalizztion:
        X = normalize(X); X_val = normalize(X_val)
        print(X.shape, X_val.shape)
    return X, y, X_val, y_val 

if __name__ == '__main__':
    target_tif_path = "./test.tif"
    box_size = 128
    ref_path = "../data/Sentinel-2/projected_tif/T50QRM_20180322T022651_B2348.tif"
    X, y, X_val, y_val = load_data(ref_path, target_tif_path, box_size)
    print("All numbers of data:",X.shape)


