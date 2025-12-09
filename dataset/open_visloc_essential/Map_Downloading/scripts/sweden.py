import math
import requests
from pyproj import Transformer
import os
from PIL import Image
import io
import time
RESOLUTION = 0.067
def latlon_to_bbox(lat, lon, side_length_m):
    """
    Convert given WGS84 (lat, lon) to EPSG:3006 projection coordinates, and return
    a box centered at (x, y) with side length of side_length_m.
    Returns in order (minx, miny, maxx, maxy),
    which can be directly used for the WMS BBOX parameter.
    """
    wgs84_to_3006 = Transformer.from_crs("EPSG:4326", "EPSG:3006", always_xy=True)
    x, y = wgs84_to_3006.transform(lon, lat)

    half_side = side_length_m / 2
    minx = x - half_side
    miny = y - half_side
    maxx = x + half_side
    maxy = y + half_side

    return (minx, miny, maxx, maxy)


def sweden_exact_position_image(lat, lon, side_length_m=50, retries=5, delay=2):
    """
    Download and return an image for the specified area based on latitude, longitude and side length.
    Parameters:
        lat (float): Latitude of the center point
        lon (float): Longitude of the center point
        side_length_m (float): Side length of the square area (m)
    Returns:
        bytes: Image data
    """
    size = side_length_m / RESOLUTION   
    minx, miny, maxx, maxy = latlon_to_bbox(lat, lon, side_length_m)

    url = (
        "https://minkarta.lantmateriet.se/map/ortofoto?"
        "REQUEST=GetMap&SERVICE=WMS&VERSION=1.1.1&FORMAT=image/png&STYLES=&TRANSPARENT=false&"
        "LAYERS=Ortofoto_0.5,Ortofoto_0.4,Ortofoto_0.25,Ortofoto_0.16&TILED=true&MAP_RESOLUTION=180&"
        f"WIDTH={size}&HEIGHT={size}&SRS=EPSG:3006&"
        f"BBOX={minx},{miny},{maxx},{maxy}"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ),
        "Accept": "image/png,*/*;q=0.9",
    }

    # # print(f"Generated URL: {url}")
    # response = requests.get(url, headers=headers, stream=True)

    # if response.status_code == 200:
    #     content_type = response.headers.get('Content-Type')
    #     if content_type == 'image/png':
    #         return Image.open(io.BytesIO(response.content))
    #     else:
    #         print("The server did not return a PNG image. Content-Type:", content_type)
    #         return None
    # else:
    #     print(f"Failed to download image, status code: {response.status_code}")
    #     return None

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=10)
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type')
                if content_type == 'image/png':
                    return Image.open(io.BytesIO(response.content))
                else:
                    print("The server did not return a PNG image. Content-Type:", content_type)
                    return None
            else:
                print(f"Attempt {attempt+1}: Failed with status code {response.status_code}")
        except requests.RequestException as e:
            print(f"Attempt {attempt+1}: Request failed with error: {e}")
            time.sleep(delay)
            
    print("All attempts to download the image failed.")
    return None

# if __name__ == "__main__":
#     lat = 59.3450206741838
#     lon = 18.039668509182004
#     side_length_m = 50
#     output_dir = "outputs/sweden"
#     output_filename = f"{output_dir}/lat_{lat}_lon_{lon}_side_{side_length_m}.png"
    
#     if os.path.dirname(output_filename):
#         os.makedirs(os.path.dirname(output_filename), exist_ok=True)

#     image_data = sweden_exact_position_image(lat, lon, side_length_m)
    
#     if image_data:
#         with open(output_filename, "wb") as f:
#             f.write(image_data)
#         print(f"Image successfully saved to {output_filename}")
