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

def load_data_from_folder(or_tif_folder, gt_tif, box_size, val_ratio=0.2, normalize=False):
    tif_list = os.listdir(or_tif_folder)#[:3]
    y = load_tif_andSplit(gt_tif).get_splitted_images()
    label_data_idx = np.nonzero(np.sum(y, axis=(1,2,3)))[0].tolist() # only using have label data
    val_list = np.random.choice(label_data_idx, int(len(label_data_idx)*val_ratio))
    train_list = np.array([item for item in label_data_idx if item not in val_list])
    y, y_val = y[train_list], y[val_list]
    
    X, X_val = [[] for i in range(len(tif_list))], [[] for i in range(len(tif_list))]

    for tif_path, x_buffer, x_val_buffer in zip(tif_list, X, X_val):
        print(tif_path)
        ds = gdal.Open(os.path.join(or_tif_folder, tif_path))
        img_src_arr = np.expand_dims(ds.ReadAsArray(), 0)
        x = np.transpose(img_src_arr, axes=[1,2,0])
        splitted_im = SplittedImage(x, box_size, ds.GetGeoTransform(), ds.GetProjection())
        x = splitted_im.get_splitted_images()
        x_buffer.append(x[train_list])
        x_val_buffer.append(x[val_list])
    # sys.exit()
    X, y = np.stack(np.squeeze(X), axis=-1), np.array(y)
    X_val, y_val = np.stack(np.squeeze(X_val), axis=-1), np.array(y_val)
    if normalize:
        X = normalize(X); X_val = normalize(X_val)
    return X, y, X_val, y_val 

def load_data_from_folder_list(or_tif_folder_list, gt_tif, box_size, val_ratio=0.2, normalize=False):
    for or_tif_folder in or_tif_folder_list:
        tif_list = os.listdir(or_tif_folder)#[:3]
        y = load_tif_andSplit(gt_tif).get_splitted_images()
        label_data_idx = np.nonzero(np.sum(y, axis=(1,2,3)))[0].tolist() # only using have label data
        val_list = np.random.choice(label_data_idx, int(len(label_data_idx)*val_ratio))
        train_list = np.array([item for item in label_data_idx if item not in val_list])
        y, y_val = y[train_list], y[val_list]
        
        X, X_val = [[] for i in range(len(tif_list))], [[] for i in range(len(tif_list))]

        for tif_path, x_buffer, x_val_buffer in zip(tif_list, X, X_val):
            print(tif_path)
            ds = gdal.Open(os.path.join(or_tif_folder, tif_path))
            img_src_arr = np.expand_dims(ds.ReadAsArray(), 0)
            x = np.transpose(img_src_arr, axes=[1,2,0])
            splitted_im = SplittedImage(x, box_size, ds.GetGeoTransform(), ds.GetProjection())
            x = splitted_im.get_splitted_images()
            x_buffer.append(x[train_list])
            x_val_buffer.append(x[val_list])
        
        X, y = np.stack(np.squeeze(X), axis=-1), np.array(y)
        X_val, y_val = np.stack(np.squeeze(X_val), axis=-1), np.array(y_val)
    if normalize:
        X = normalize(X); X_val = normalize(X_val)
    return X, y, X_val, y_val 

if __name__ == '__main__':
    target_tif_path = "./test.tif"
    box_size = 128

    ## case 1: if all band in one tif file
    ref_path = "../data/Sentinel-2/projected_tif/T50QRM_20180322T022651_B2348.tif"
    # X, y = load_data(ref_path, target_tif_path, box_size)
    ## case 2: if band in multiple tif file but in same folder
    ref_path = "../data/Sentinel-2/T50QRM_20180322_FullBand"
    X, y, X_val, y_val = load_data_from_folder(ref_path, target_tif_path, box_size)
    
    print("All numbers of data:",X.shape)
