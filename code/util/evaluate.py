import argparse, sys, os, time, ast, distutils.util, time, subprocess, osr
from util.ziyu_model import *
from util.train import *
from util.data_parser import *
import tensorflow as tf
from keras import backend as K
from keras.models import load_model

def write_predict_tif(self, X, dst_tif_path):
    rows = self.source_image.shape[0]
    cols = self.source_image.shape[1]
    bands = X.shape[3]
    dst_ds = gdal.GetDriverByName('GTiff').Create(dst_tif_path, cols, rows, 1, gdal.GDT_Float64 )
    dst_ds.SetGeoTransform(self.geo_transform)
    dst_ds.SetProjection(self.projection)
    
    
    X_combined = np.empty((rows, cols))
    X_combined[:] = np.nan
    for i in range(len(X)):
        idx_h , idx_w = self.convert_order_to_location_index(i)
        h_start_inner, h_stop_inner = self.convert_to_inner_index_h(idx_h, idx_h)
        w_start_inner, w_stop_inner = self.convert_to_inner_index_w(idx_w, idx_w)
        h_length, w_length = X_combined[h_start_inner:h_stop_inner, w_start_inner:w_stop_inner].shape
        pred_patch = X[i, :h_length, :w_length, 0]
        pred_patch = pred_patch - 1
        pred_patch[np.nan_to_num(pred_patch) < 0] = np.nan
        pred_patch[pred_patch == 0] = 1
        # pred_patch[pred_patch == 1] = 0
        X_combined[h_start_inner:h_stop_inner, w_start_inner:w_stop_inner] = pred_patch
    band = dst_ds.GetRasterBand(1)
    band.WriteArray(X_combined)
    band.FlushCache()
    
    dst_ds = None

def read_tif_file(filename):
    filehandle = gdal.Open(filename)
    band1 = filehandle.GetRasterBand(1)
    geotransform = filehandle.GetGeoTransform()
    geoproj = filehandle.GetProjection()
    Z = band1.ReadAsArray()
    xsize = filehandle.RasterXSize
    ysize = filehandle.RasterYSize
    return xsize,ysize,geotransform,geoproj,Z

def write_tif_file(filename, geotransform, geoprojection, data):
    (x,y) = data.shape
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(filename, y, x, 1, gdal.GDT_Float32)
    dst_ds.GetRasterBand(1).WriteArray(data)
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(geoprojection)
    dst_ds.GetRasterBand(1).SetNoDataValue(-9999)
    return 1

def clip_by_shp_file(shp_path, tif_path, args):
    os.makedirs(os.path.join(args.model_dest,"clip_result"), exist_ok=True)
    tif_path = os.path.abspath(tif_path)
    clip_result_path = tif_path.replace("original", "clip_result").replace("pred_", "clip_")
    os.environ["GDAL_DATA"] = "C:\Program Files\QGIS 3.8\share\gdal"
    input_raster = gdal.Open(tif_path)
    proj = osr.SpatialReference(wkt=input_raster.GetProjection())
    src_csr = proj.GetAttrValue('AUTHORITY',1)
    cmd = 'gdalwarp -s_srs EPSG:{} -t_srs EPSG:3826 -of GTiff -cutline {} -crop_to_cutline {} {}'.format(src_csr, shp_path, tif_path, clip_result_path)
    retcode = subprocess.call(cmd, shell=True)
    [xsize,ysize,geotransform,geoproj,Z] = read_tif_file(clip_result_path)
    Z[Z==0]= np.nan
    write_tif_file(clip_result_path, geotransform, geoproj, Z)
    return 

def evaluate(model_path, tif_list, args, normalizztion=False, clip_shp_path=None):
    model_path = os.path.join(model_path, "model_loss.h5")
    model = load_model(model_path, { loss_arg(args.loss)[0]:loss_arg(args.loss)[1], 'om':om, 'tf':tf})
    os.makedirs(os.path.join(args.model_dest,"original"), exist_ok=True)
    for tif_path in tif_list:
        tif_folder, tif_name = os.path.split(tif_path)
        print("\033[96mLoad data: {}\033[0m".format(tif_name))
        splitted_im = load_tif_andSplit(tif_path, args.box_size)
        X = splitted_im.get_splitted_images()
        if normalizztion:
            X = normalize(X)
        pred = model.predict(X, batch_size=args.batch_size, verbose=1)
        SplittedImage.write_predict_tif = write_predict_tif
        pred = np.where(pred > 0.5, 1, np.nan) # binary
        pred_tif_path = os.path.join(args.model_dest,"original/pred_" + tif_name)
        splitted_im.write_predict_tif(pred, pred_tif_path)
        print("Save prediction to", pred_tif_path)
        if clip_shp_path != None:
            clip_by_shp_file(os.path.abspath(clip_shp_path), pred_tif_path, args)


    return 


if __name__ == '__main__':
    file_name = [
        "T50QRM_20190327T022551_B2348.tif",
        "T50QRM_20190327T022551_B2348.tif"
        ]
    model_path = "./experiments/台灣區決賽所用模型及預測結果/model_loss.h5"
    box_size = 128
    batch_size = 50
    model = load_model(model_path, { loss_arg("DL")[0]:loss_arg("DL")[1], 'om':om, 'tf':tf})
    for i in file_name:
        ref_path = "../data/Sentinel-2/projected_tif/"+i     
        print(ref_path)
        splitted_im = load_tif_andSplit(ref_path, box_size)
        X = splitted_im.get_splitted_images()     
        pred = model.predict(X, batch_size=batch_size, verbose=0)
        print(pred.shape)
        SplittedImage.write_predict_tif = write_predict_tif
        splitted_im.write_predict_tif(pred,"./pred_"+i)
    



