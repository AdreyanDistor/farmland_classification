import cv2
import rasterio
import numpy as np

def create_color_image(red_band_path, green_band_path, blue_band_path, output_path):
    """
    Combines grayscale bands into a color image using OpenCV and saves it with CRS metadata.

    Args:
        red_band_path: Path to the grayscale image file for the red band.
        green_band_path: Path to the grayscale image file for the green band.
        blue_band_path: Path to the grayscale image file for the blue band.
        output_path: Path to save the resulting color image.
    """
    # Read the bands using OpenCV
    red_band = cv2.imread(red_band_path, cv2.IMREAD_GRAYSCALE)
    green_band = cv2.imread(green_band_path, cv2.IMREAD_GRAYSCALE)
    blue_band = cv2.imread(blue_band_path, cv2.IMREAD_GRAYSCALE)

    if red_band is None or green_band is None or blue_band is None:
        raise ValueError("One or more input images could not be read. Check file paths.")

    # Ensure bands have the same dimensions
    if not (red_band.shape == green_band.shape == blue_band.shape):
        raise ValueError("Input images must have the same dimensions.")

    # Merge the bands into a single color image (RGB format)
    color_image = cv2.merge([red_band, green_band, blue_band])  # RGB order

    # Read metadata (CRS, transform, etc.) from one of the input bands using rasterio
    with rasterio.open(red_band_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs

    # Update metadata for the output color image
    meta.update({
        "count": 3,  # Number of bands (RGB)
        "dtype": "uint8",  # Save as 8-bit
        "driver": "GTiff"  # Save as GeoTIFF
    })

    # Save the color image with the same CRS and metadata using rasterio
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(color_image.transpose(2, 0, 1))  # Write RGB bands

    print(f"Color image saved to {output_path} with CRS: {crs}")

if __name__ == '__main__':
    red_band_path = 'selected_data/LC09_L2SP_038037_20250226_20250227_02_T1_SR_B4.TIF'
    green_band_path = 'selected_data/LC09_L2SP_038037_20250226_20250227_02_T1_SR_B3.TIF'
    blue_band_path = 'selected_data/LC09_L2SP_038037_20250226_20250227_02_T1_SR_B2.TIF'

    output_path = 'color_image.tif'

    try:
        create_color_image(red_band_path, green_band_path, blue_band_path, output_path)
    except ValueError as e:
        print(f"Error: {e}")