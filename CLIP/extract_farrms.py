import rasterio
import rasterio.mask
import geopandas as gpd
import pprint as p
import os

def clip_satellite_image(satellite_image, vector_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with rasterio.open(satellite_image) as src:
        gdf = gpd.read_file(vector_data)
        
        if src.crs != gdf.crs:
            gdf = gdf.to_crs(src.crs)
            print("Reprojected Vector CRS to match Raster CRS.")

        i = 0
        for index, row in gdf.iterrows():
            g = row['geometry']
            p.pprint(g)

            try:
                clipped_image, transform = rasterio.mask.mask(src, [g], crop=True)

                clipped_image[clipped_image == src.nodata] = 0 

                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": clipped_image.shape[1],
                    "width": clipped_image.shape[2],
                    "transform": transform,
                })

                output_path = os.path.join(output_dir, f"data_{i}.tiff")
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(clipped_image)

                print(f"Clipped image saved to {output_path}")
                i += 1
            except ValueError as e:
                print(f"Skipping geometry {index} due to error: {e}")

satellite_image = 'test_data/338-1252_quad.tif'
vector_data = 'test_data/selected_test_data.shp'
output_dir = 'test_data/test_images'

clip_satellite_image(satellite_image, vector_data, output_dir)
