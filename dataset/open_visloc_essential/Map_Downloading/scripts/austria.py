url = "https://maps.bev.gv.at"
url_example = 'https://maps.bev.gv.at/#/center/14.00932,48.074/zoom/17.3/basis/ortho/compare/epo_1'

import requests
from pyproj import Proj, transform
import os
from PIL import Image
import io
RESOLUTION = 0.2
RATIO = 1.496
def lonlat_to_mercator(lon, lat):
    proj_wgs84 = Proj(init='epsg:4326')  # WGS84
    proj_mercator = Proj(init='epsg:3857')  # Web Mercator
    x, y = transform(proj_wgs84, proj_mercator, lon, lat)
    return x, y

def latlon_to_mercator(lat, lon):
    return lonlat_to_mercator(lon, lat)

def mercator_to_lonlat(x, y):
    proj_wgs84 = Proj(init='epsg:4326')  # WGS84
    proj_mercator = Proj(init='epsg:3857')  # Web Mercator
    lon, lat = transform(proj_mercator, proj_wgs84, x, y)
    return lon, lat

def mercator_to_latlon(x, y):
    lon, lat = mercator_to_lonlat(x, y)
    return lat, lon

def austria_exact_position_image(lat, lon, side_length_m, format='image/jpeg', crs='EPSG:3857'):
    """
    Download Austria map image for specified location and side length
    
    :param lat: latitude
    :param lon: longitude
    :param side_length_m: required side length (meters)
    :param format: image format
    :param crs: coordinate system
    :return: PIL.Image object
    """
    x, y = latlon_to_mercator(lat, lon)
    
    # 计算边界框
    half_size = side_length_m / 2 * RATIO
    min_x = x - half_size
    min_y = y - half_size
    max_x = x + half_size
    max_y = y + half_size

    bbox = f"{min_x},{min_y},{max_x},{max_y}"

    pixels = int(side_length_m / RESOLUTION)

    wms_url = (
        f"https://kataster.bev.gv.at/ortho/ows?"
        f"REQUEST=GetMap&SERVICE=WMS&VERSION=1.3.0"
        f"&FORMAT={format}"
        f"&STYLES=&TRANSPARENT=true"
        f"&LAYERS=inspire:AT_BEV_OI"
        f"&WIDTH={pixels}&HEIGHT={pixels}"
        f"&CRS={crs}"
        f"&BBOX={bbox}"
    )

    response = requests.get(wms_url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        print("failed", response.status_code)
        return None

if __name__ == "__main__":
    lat = 48.074
    lon = 14.00932
    side_length_m = 200  # specify side length as 200 meters
    
    image = austria_exact_position_image(lat, lon, side_length_m)
    
    if image:
        # save the image
        output_dir = "outputs/austria"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"austria_lat{lat}_lon{lon}_side{side_length_m}m.jpg"
        )
        image.save(output_path)
        print(f"saved to {output_path}")