import os
import re
from osgeo import gdal
import numpy as np
import pdb

def read_tif_files( folder_path, variable_name):
    tif_files = []
    tif_names = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if re.match(r'^{}.*\.tif$'.format(variable_name), file): 
                file_path = os.path.join(root, file)        
                img = gdal.Open(file_path, gdal.GA_ReadOnly)
                tif_array = img.ReadAsArray()
                tif_files.append(tif_array)
                tif_names.append(file)
                if len(tif_files) >= 9:
                    return tif_files
    print(tif_names)
    return tif_files

folder_path = '/nfs-data2/ys/ChangeDetection/code/senseearth/data/samples/train/Times/'
variable_name = '6_38560'

read_tif_files(folder_path, variable_name)
