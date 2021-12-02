import gdal, ogr
import numpy as np
import geopandas as gpd
import os, time
from SplittedImage import SplittedImage


def convert_shp_to_geotiff(src_shp_path, dst_tif_path, ref_tif_path):
    df_shp = gpd.read_file(src_shp_path) # Open your shapefile
    df_shp['positive'] = 1
    src_shp_ds = ogr.Open(df_shp.to_json())
    src_shp_layer = src_shp_ds.GetLayer()

    ref_tif_ds = gdal.Open(ref_tif_path)
    ref_tif_cols, ref_tif_rows = ref_tif_ds.RasterXSize, ref_tif_ds.RasterYSize
    
    dst_tif_ds = gdal.GetDriverByName('GTiff').Create(dst_tif_path, ref_tif_cols, ref_tif_rows, 1, gdal.GDT_Byte)
    dst_tif_ds.SetGeoTransform(ref_tif_ds.GetGeoTransform())

    band = dst_tif_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.FlushCache()

    # set it to the attribute that contains the relevant unique
    gdal.RasterizeLayer(dst_tif_ds, [1], src_shp_layer, options = ["ATTRIBUTE=positive"])

    # Add a spatial reference
    dst_tif_ds.SetProjection(ref_tif_ds.GetProjection())
    ref_tif_ds = None
    dst_ds = None

src_path = "../data/Sentinel-2/farm/YL20181Rice.shp"
ref_path = "../data/Sentinel-2/projected_tif/T50QRM_20180322T022651_B2348.tif"
target_tif_path = "./test.tif"
target_tif_save_dir = "./util/split_gt_tif/"
os.makedirs(target_tif_save_dir, exist_ok=True)
box_size = 128
    

start_time = time.time()
convert_shp_to_geotiff(src_path, target_tif_path, ref_path)
print("Using %s seconds for convert shp file to target tif" % (time.time() - start_time))

# create splitted target tif
ds = gdal.Open(target_tif_path)
img_src_arr = np.expand_dims(ds.ReadAsArray(), 0)
X = np.transpose(img_src_arr, axes=[1,2,0])
splitted_gt = SplittedImage(X, box_size, ds.GetGeoTransform(), ds.GetProjection())
splitted_gt.write_splitted_images(target_tif_save_dir, 'P0015913_SP5_006_001_002_021_002_005')
