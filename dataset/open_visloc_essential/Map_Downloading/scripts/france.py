import math
import requests
from pyproj import Transformer
import os
from PIL import Image, ImageDraw
import io
import tqdm
# Constants definition
ZOOM_LEVEL = 19
SCALE_DENOMINATOR = 1066.3647919248930199
RESOLUTION = SCALE_DENOMINATOR * 0.28e-3
TRUE_RESOLUTION = 0.2032520325203252
# The French map website has some issues with resolution settings that require manual adjustment.
# We need to use the given resolution for all tile fetching operations,
# but use the actual resolution when calculating the number of tiles to stitch and for final cropping.
TILE_WIDTH = 256
TILE_HEIGHT = 256
TOP_LEFT_X = -20037508.342789199
TOP_LEFT_Y = 20037508.342789199

# Initialize coordinate transformer: WGS84 (EPSG:4326) to Web Mercator (EPSG:3857)
transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)

def lonlat_to_tile(lon, lat, zoom):
    """
    Convert longitude/latitude to tile row and column numbers
    """
    x, y = transformer.transform(lon, lat)
    
    tile_size_meters = TILE_WIDTH * RESOLUTION
    tile_col = math.floor((x - TOP_LEFT_X) / tile_size_meters)
    tile_row = math.floor((TOP_LEFT_Y - y) / tile_size_meters)
    
    return tile_row, tile_col

def save_images(stitched_image, cropped_image, lon, lat, zoom, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    stitched_filename = os.path.join(
        save_folder,
        f"stitched_zoom{zoom}_lon{lon}_lat{lat}.jpg"
    )
    cropped_filename = os.path.join(
        save_folder,
        f"cropped_zoom{zoom}_lon{lon}_lat{lat}.jpg"
    )
    marked_filename = os.path.join(
        save_folder,
        f"marked_zoom{zoom}_lon{lon}_lat{lat}.jpg"
    )

    stitched_image.save(stitched_filename)

    cropped_image.save(cropped_filename)
    
    marked_image = cropped_image.copy()
    draw = ImageDraw.Draw(marked_image)
    center_x = marked_image.width // 2
    center_y = marked_image.height // 2
    point_size = 5
    draw.ellipse((center_x - point_size, center_y - point_size, 
                  center_x + point_size, center_y + point_size), fill=(255, 0, 0))
    marked_image.save(marked_filename)
    
def fetch_tile_image(lat, lon, zoom, tile_row=None, tile_col=None, retries=3):
    """
    Download image for specified tile
    """
    if tile_row is None or tile_col is None:
        tile_row, tile_col = lonlat_to_tile(lon, lat, zoom)

    url = (
        f"https://data.geopf.fr/wmts?"
        f"layer=ORTHOIMAGERY.ORTHOPHOTOS&style=normal&tilematrixset=PM&Service=WMTS&"
        f"Request=GetTile&Version=1.0.0&Format=image/jpeg&TileMatrix={zoom}&"
        f"TileCol={tile_col}&TileRow={tile_row}"
    )
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            return img
        except requests.RequestException as e:
            print(f"Request failed (Row={tile_row}, Col={tile_col}), attempt {attempt}/{retries}: {e}")
            if attempt == retries:
                print(f"All retry attempts failed. Using black image instead (Row={tile_row}, Col={tile_col})")
        except Exception as e:
            print(f"Image processing failed (Row={tile_row}, Col={tile_col}): {e}")
            break

    return Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), (0, 0, 0))

def calculate_grid_size(lat, lon, side_length_m, zoom):
    """
    Calculate required tile grid size for given side length
    """
    tile_meter = TILE_WIDTH * TRUE_RESOLUTION  # actual meters per tile
    
    tiles_needed = math.ceil(side_length_m / 2 / tile_meter)
    
    return tiles_needed

def fetch_and_stitch_tiles(lat, lon, zoom, side_length_m):
    """
    Download sufficient tile grid and stitch them together
    """
    center_row, center_col = lonlat_to_tile(lon, lat, zoom)
    tiles_needed = calculate_grid_size(lat, lon, side_length_m, zoom)
    
    grid_size = 2 * tiles_needed + 1

    start_row = center_row - (grid_size // 2)
    start_col = center_col - (grid_size // 2)
    
    # Download all tiles
    tiles = []
    for row in range(start_row, start_row + grid_size):
        row_tiles = []
        for col in range(start_col, start_col + grid_size):
            img = fetch_tile_image(lat, lon, zoom, tile_row=row, tile_col=col)
            row_tiles.append(img)
        tiles.append(row_tiles)

    # Stitch images together
    stitched_image = Image.new('RGB', (TILE_WIDTH * grid_size, TILE_HEIGHT * grid_size))
    for i, row_tiles in enumerate(tiles):
        for j, img in enumerate(row_tiles):
            stitched_image.paste(img, (j * TILE_WIDTH, i * TILE_HEIGHT))

    # Calculate center point position in stitched image
    x, y = transformer.transform(lon, lat)
    
    stitched_origin_x = TOP_LEFT_X + start_col * TILE_WIDTH * RESOLUTION
    stitched_origin_y = TOP_LEFT_Y - start_row * TILE_HEIGHT * RESOLUTION
    
    delta_x = x - stitched_origin_x
    delta_y = stitched_origin_y - y
    pixel_x = delta_x / RESOLUTION
    pixel_y = delta_y / RESOLUTION
    
    return stitched_image, (pixel_x, pixel_y)

def crop_image_center(stitched_image, center_pixel, crop_size):
    """
    Crop image of specified size from stitched image, ensuring correct center point position
    """
    pixel_x, pixel_y = center_pixel
    half_crop = crop_size // 2
    
    left = int(pixel_x - half_crop)
    upper = int(pixel_y - half_crop)
    right = left + crop_size
    lower = upper + crop_size
    
    # Ensure cropping area is within image range
    left = max(left, 0)
    upper = max(upper, 0)
    right = min(right, stitched_image.width)
    lower = min(lower, stitched_image.height)
    
    cropped_image = stitched_image.crop((left, upper, right, lower))
    return cropped_image

def france_exact_position_image(lat, lon, side_length_m, zoom=ZOOM_LEVEL):
    """
    Get image for specified location and side length
    
    :param lat: latitude
    :param lon: longitude
    :param side_length_m: required side length (meters)
    :param zoom: zoom level
    :return: PIL.Image object
    """
    # Calculate required pixel size
    pixels_needed = int(side_length_m / TRUE_RESOLUTION)
    
    # Get stitched image and center point position
    stitched_image, center_pixel = fetch_and_stitch_tiles(lat, lon, zoom, side_length_m)
    
    # Crop image of specified size
    cropped_image = crop_image_center(stitched_image, center_pixel, pixels_needed)
    # save_images(stitched_image, cropped_image, lon, lat, zoom, "outputs/france")
    return cropped_image

if __name__ == "__main__":
    lat = 47.204877443106774
    lon = -1.5657045783975607
    side_length_m = 150
    
    image = france_exact_position_image(lat, lon, side_length_m)
    
    # Save image
    # output_dir = "outputs/france"
    # os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(
    #     output_dir,
    #     f"france_lon{lon}_lat{lat}_side{side_length_m}m.jpg"
    # )
    # image.save(output_path)
    # print(f"saved to {output_path}")


