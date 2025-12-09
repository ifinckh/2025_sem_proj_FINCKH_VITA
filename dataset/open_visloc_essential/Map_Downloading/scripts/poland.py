import requests
from pyproj import Transformer
import os
RESOLUTION = 0.1
from PIL import Image
import io

def poland_exact_position_image(
    lat, lon,
    side_length_m=500,
    format="image/png",
    retries=3
):
    """
    Download and return an image for the specified area based on latitude and longitude.
    Parameters:
        lat (float): Latitude of the center point
        lon (float): Longitude of the center point
        side_length_m (float): Side length of the square area (m), each meter corresponds to 10 pixels
        format (str): Image format, default is PNG, can be changed to JPEG
    Returns:
        bytes: Image data
    """
    # EPSG:2180 projection coordinates
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True)
    x_center, y_center = transformer.transform(lon, lat)

    # Create square area based on center point
    half_side = side_length_m / 2
    x_min = x_center - half_side
    x_max = x_center + half_side
    y_min = y_center - half_side
    y_max = y_center + half_side

    bbox_str = f"{x_min}%2C{y_min}%2C{x_max}%2C{y_max}"

    base_url = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WCS/HighResolution"
    full_url = (
        f"{base_url}?service=wcs"
        f"&request=GetCoverage"
        f"&version=1.0.0"
        f"&coverage=Orthoimagery_High_Resolution"
        f"&format={format}"
        f"&bbox={bbox_str}"
        f"&resx=0.1"
        f"&resy=0.1"
        f"&crs=EPSG%3A2180"
    )

    # Mimic browser
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:58.0) "
            "Gecko/20100101 Firefox/58.0"
        )
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(full_url, headers=headers, timeout=60)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            print(f"failed, {e}")
            return None

if __name__ == "__main__":
    lat = 52.231480139793376
    lon = 21.0049362013031
    side_length_m = 100
    #resolution = 0.1
    output_path = f"outputs/poland/poland_tile_lon{lon}_lat{lat}_side{side_length_m}.png"
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    image_data = poland_exact_position_image(
        lat=lat,
        lon=lon,
        side_length_m=side_length_m
    )
    
    if image_data:
        image_data.save(output_path)
        print(f"saved to {output_path}")