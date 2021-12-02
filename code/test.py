import os, subprocess
import numpy as np
import gdal 
import numpy as np
import geopandas as gpd

clip_shp_path = "../data/Sentinel-2/雲林縣/yunlin.shp"
pred_tif_path = "./experiments/test/original/pred_T50QRM_20180322T022651_B2348.tif"
shp_path, tif_path = os.path.abspath(clip_shp_path), pred_tif_path

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

os.makedirs(os.path.join("../experiments/test/clip_result"), exist_ok=True)
tif_path = os.path.abspath(tif_path)
print(tif_path)
print()
print("HEHE", os.path.isfile(tif_path))
clip_result_path = tif_path.replace("original", "clip_result").replace("pred_", "clip_")
os.environ["GDAL_DATA"] = "C:\Program Files\QGIS 3.8\share\gdal"
cmd = 'gdalwarp -s_srs EPSG:32650 -t_srs EPSG:3826 -of GTiff -cutline {} -crop_to_cutline {} {}'.format(shp_path, tif_path, clip_result_path)
retcode = subprocess.call(cmd, shell=True)
print(clip_result_path)
[xsize,ysize,geotransform,geoproj,Z] = read_tif_file(clip_result_path)
Z[Z==0]= np.nan
write_tif_file(clip_result_path, geotransform, geoproj, Z)