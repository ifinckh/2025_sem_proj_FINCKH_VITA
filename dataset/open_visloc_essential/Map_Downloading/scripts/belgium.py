#https://wms.ngi.be/inspire/ortho/service?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&FORMAT=image%2Fpng&TRANSPARENT=true&LAYERS=orthoimage_coverage_2022%2Corthoimage_coverage_2021%2Corthoimage_coverage_2020%2Corthoimage_coverage_2019%2Corthoimage_coverage_2018%2Corthoimage_coverage_2017%2Corthoimage_coverage_2016%2Corthoimage_coverage&WIDTH=512&HEIGHT=512&CRS=EPSG%3A3857&STYLES=&BBOX=483693.51498859376%2C6596209.792897571%2C483999.26310173445%2C6596515.541010711
import requests
import os
from pyproj import Proj, transform
from PIL import Image
import io
RESOLUTION = 0.2
RATIO = 1.58
proj_mercator = Proj(init='epsg:3857')
proj_wgs84 = Proj(init='epsg:4326')

def lonlat_to_mercator(lon, lat):
    x, y = transform(proj_wgs84, proj_mercator, lon, lat)
    return x, y

def latlon_to_mercator(lat, lon):
    return lonlat_to_mercator(lon, lat)

def mercator_to_lonlat(x, y):
    lon, lat = transform(proj_mercator, proj_wgs84, x, y)
    return lon, lat

def mercator_to_latlon(x, y):
    lon, lat = mercator_to_lonlat(x, y)
    return lat, lon

def calculate_bbox(x, y, side_length_m):
    """Calculate bounding box for given center point and side length"""
    half_side = side_length_m / 2
    min_x = x - half_side
    min_y = y - half_side
    max_x = x + half_side
    max_y = y + half_side
    return min_x, min_y, max_x, max_y

def belgium_exact_position_image(lat, lon, side_length_m):
    """
    Get image for specified location and side length
    
    :param lat: latitude
    :param lon: longitude
    :param side_length_m: required side length (meters)
    :return: PIL.Image object
    """
    x, y = latlon_to_mercator(lat, lon)
    min_x, min_y, max_x, max_y = calculate_bbox(x, y, side_length_m * RATIO)
    size = int(side_length_m / RESOLUTION)
    layers = 'orthoimage_coverage_2022,orthoimage_coverage_2021,orthoimage_coverage_2020,orthoimage_coverage_2019,orthoimage_coverage_2018,orthoimage_coverage_2017,orthoimage_coverage_2016,orthoimage_coverage'
    format = 'image/png'
    crs = 'EPSG:3857'
    
    wms_url = (
        f"https://wms.ngi.be/inspire/ortho/service?"
        f"SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&FORMAT={format}&"
        f"TRANSPARENT=true&LAYERS={layers}&WIDTH={size}&HEIGHT={size}&"
        f"CRS={crs}&STYLES=&BBOX={min_x},{min_y},{max_x},{max_y}"
    )
    
    response = requests.get(wms_url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        print("failed", response.status_code)
        return None

if __name__ == "__main__":
    lat = 50.844582225194024
    lon = 4.346398819971392
    side_length_m = 200  # specify side length as 200 meters
    
    image = belgium_exact_position_image(lat, lon, side_length_m)
    
    if image:
        # save the image
        output_dir = "outputs/belgium"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"belgium_lat{lat}_lon{lon}_side{side_length_m}m.png"
        )
        image.save(output_path)
        print(f"saved to {output_path}")

# The image obtained using bbox request should be unbiased