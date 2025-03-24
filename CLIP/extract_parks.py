import cv2
import rasterio
import rasterio.mask
import geopandas as gpd
import numpy as np
import os
import pprint as p
from shapely.geometry import box
from extract_farrms import clip_satellite_image 
import cv2
import rasterio
import numpy as np

def create_color_image(red_band_path, green_band_path, blue_band_path, output_path):
    red_band = cv2.imread(red_band_path, cv2.IMREAD_GRAYSCALE)
    green_band = cv2.imread(green_band_path, cv2.IMREAD_GRAYSCALE)
    blue_band = cv2.imread(blue_band_path, cv2.IMREAD_GRAYSCALE)

    if red_band is None or green_band is None or blue_band is None:
        raise ValueError("One or more input images could not be read. Check file paths.")

    if not (red_band.shape == green_band.shape == blue_band.shape):
        raise ValueError("Input images must have the same dimensions.")

    color_image = cv2.merge([red_band, green_band, blue_band]) 

    alpha_channel = np.where((red_band == 0) & (green_band == 0) & (blue_band == 0), 0, 255).astype(np.uint8)

    rgba_image = cv2.merge([red_band, green_band, blue_band, alpha_channel])

    with rasterio.open(red_band_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs

    meta.update({
        "count": 4, 
        "dtype": "uint8", 
        "driver": "GTiff"
    })

if __name__ == '__main__':
    red_band_path = 'selected_data/park_data/LC09_L2SP_040036_20250224_20250225_02_T1_SR_B4.TIF'
    green_band_path = 'selected_data/park_data/LC09_L2SP_040036_20250224_20250225_02_T1_SR_B3.TIF'
    blue_band_path = 'selected_data/park_data/LC09_L2SP_040036_20250224_20250225_02_T1_SR_B2.TIF'

    vector_data = 'selected_data/parks.shp'

    output_dir = 'seg_images_parks'

    color_image_path = 'color_image.tif'
    try:
        create_color_image(red_band_path, green_band_path, blue_band_path, color_image_path)
    except ValueError as e:
        print(f"Error creating color image: {e}")

    try:
        clip_satellite_image(color_image_path, vector_data, output_dir)
    except ValueError as e:
        print(f"Error Converitng sattlie imagery: {e}")