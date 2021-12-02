import cv2
import gdal 
import ogr, osr
import numpy as np
import geopandas as gpd

import time, os, sys, json
from util.ziyu_model import *
from util.train import *
from util.data_parser import *
from util.evaluate import *
from osgeo import gdal



def convert_geotiff_to_shp(src_tif_path, dst_shp_path):
    # this allows GDAL to throw Python Exceptions
    gdal.UseExceptions()

    #  get raster datasource
    src_ds = gdal.Open(src_tif_path)
    if src_ds is None:
        print('Unable to open %s' % src_tif_path)
        sys.exit(1)

    band_num = 1
    try:
        srcband = src_ds.GetRasterBand(band_num)
    except RuntimeError:
        # for example, try GetRasterBand(10)
        print ('Band ( %i ) not found' % band_num)
        sys.exit(1)


    #  create output datasource
    dst_layername = "output"
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(dst_shp_path)
    
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromWkt(src_ds.GetProjection())
    dst_layer = dst_ds.CreateLayer(dst_layername, srs = dst_srs )

    gdal.Polygonize(srcband, None, dst_layer, -1, [], callback=None )

    dst_ds = None
    src_ds = None

def yield_per_hec_val(filename):
    if "2017" in filename:
        return 5.852675
    elif "2018" in filename:
        return 6.867416
    elif "2019" in filename:
        return 5.76519876
    else:
        pass

class ArgumentForLoss():
    def __init__(self, model_dest, loss, batch_size):
        self.model_dest = model_dest
        self.loss = loss
        self.box_size = 128
        self.batch_size = batch_size #30

def create_clip_result(model_path, src_path):
    os.makedirs(model_path+"shp_result/", exist_ok=True) 
    os.makedirs(model_path+"statistics/", exist_ok=True) 
    os.makedirs(model_path+"add_column_shp/", exist_ok=True)
    os.makedirs(model_path+"projected_tif/", exist_ok=True)
    

    loss = "DL" # using which loss function e.g. "DL" is Dice loss
    batch_size = 30
    args = ArgumentForLoss(model_path, loss, batch_size)
    
    ################################################
    # evaluate each tif data by the training model:
    print("\033[96mStart to predict data\033[0m")
    clip_shp_path = "../data/Sentinel-2/雲林縣/yunlin.shp" # clip the prediction result by the shp file 
    tif_list = [f for f in os.listdir(src_path) if f.endswith('.tif')]
    for filename in tif_list:
        input_raster = gdal.Open(os.path.join(src_path, filename))
        output_raster = model_path+"projected_tif/"+filename
        gdal.Warp(output_raster,input_raster,dstSRS='EPSG:3826')
    tif_list = [os.path.join(src_path, f) for f in os.listdir(model_path+"projected_tif/") if f.endswith('.tif')]
    evaluate(model_path, tif_list, args, clip_shp_path=clip_shp_path)
    dst_path = model_path+"clip_result/"
    file_name = os.listdir(dst_path) 
    predict_dict = {}
    for i in file_name:
        print("Start to convert {} to shp file.".format(i))
        convert_geotiff_to_shp(dst_path+i, model_path+"shp_result/"+i[:-3]+"shp")
        shp_pred = gpd.read_file(model_path+"shp_result/"+i[:-3]+"shp")
        shp_pred["area(hec)"] = shp_pred.geometry.area / 1e4 # 原本是 m2
        shp_pred["yield_per_hec"] = shp_pred.geometry.area / 1e4 * (yield_per_hec_val(i))
        shp_pred.to_file(model_path+"add_column_shp/"+i[:-3]+"shp", driver='ESRI Shapefile')
        shp_pred = gpd.read_file(model_path+"add_column_shp/"+i[:-3]+"shp")
        total_area = np.sum(shp_pred.geometry.area)/1e4 # 公尺平方 -> 公頃
        predict_dict[i] = {
            'farms': len(shp_pred),
            'avg_area': total_area / len(shp_pred),
            'total_area': total_area,
            'yield_per_hec' : yield_per_hec_val(i),
            'yield': total_area * yield_per_hec_val(i)
        }

        with open(model_path+'/statistics/衛星圖統計{}.txt'.format(i.replace(".tif","")), 'w') as file:
            file.write(json.dumps(predict_dict[i])) 
        print("Done! Save file to {}.".format(model_path+"shp_result/"+i[:-3]+"shp"))
if __name__ == '__main__':
    model_path = "./experiments/test/"
    using_data = "../data/Sentinel-2/using_data/"
    create_clip_result(model_path, using_data)
    