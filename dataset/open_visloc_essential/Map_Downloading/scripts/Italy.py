import os
import math
import requests
from pyproj import Transformer
from PIL import Image
import io
import tqdm
resolutions = {
        0: 1587.5031750063501,
        1: 793.7515875031751,
        2: 264.5838625010584,
        3: 132.2919312505292,
        4: 66.1459656252646,
        5: 26.458386250105836,
        6: 19.843789687579378,
        7: 13.229193125052918,
        8: 10.583354500042335,
        9: 6.614596562526459,
        10: 5.291677250021167,
        11: 3.9687579375158752,
        12: 2.6458386250105836,
        13: 1.9843789687579376,
        14: 1.3229193125052918,
        15: 1.0583354500042335,
        16: 0.7937515875031751,
        17: 0.5291677250021167,
        18: 0.26458386250105836
    }
def latlon_to_tile(lat, lon, zoom):
    """
    Convert latitude and longitude to tile row and column numbers.

    :param lat: latitude
    :param lon: longitude
    :param zoom: zoom level
    :return: (row, col)
    """
    # Create coordinate transformer: WGS84 (EPSG:4326) -> UTM zone 33N (EPSG:32633)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)

    # Transform (lon, lat) to (x, y) in EPSG:32633
    x, y = transformer.transform(lon, lat)

    # Resolution lookup table

    resolution = resolutions.get(zoom)
    if resolution is None:
        raise ValueError(f"Zoom level {zoom} is not supported.")

    origin_x = -5120900.0
    origin_y = 9998100.0

    tile_size = 512  # Pixel size of each tile

    col = int((x - origin_x) / (tile_size * resolution))
    row = int((origin_y - y) / (tile_size * resolution))

    return row, col

def fetch_tile_image(lat, lon, zoom, tile_row=None, tile_col=None, retries=3):
    """
    Download image for specified tile, return black image if download fails.

    :param lat: latitude
    :param lon: longitude
    :param zoom: zoom level
    :param tile_row: tile row number
    :param tile_col: tile column number
    :param retries: number of retry attempts
    :return: PIL.Image object
    """
    if tile_row is None or tile_col is None:
        tile_row, tile_col = latlon_to_tile(lat, lon, zoom)

    # print(f"Zoom: {zoom}, Row: {tile_row}, Col: {tile_col}")

    base_url = "http://www.pcn.minambiente.it/arcgis/rest/services/immagini/ortofoto_colore_12/MapServer/tile"
    tile_url = f"{base_url}/{zoom}/{tile_row}/{tile_col}"

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(tile_url, timeout=10)
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

    # If download fails, return a black image
    black_img = Image.new("RGB", (512, 512), (0, 0, 0))
    return black_img

def calculate_grid_size(lat, lon, side_length_m, zoom):
    """
    Calculate required tile grid size for given side length
    Returns: required number of tiles (width, height) and additional padding tiles needed
    """
    resolution = resolutions.get(zoom)
    if resolution is None:
        raise ValueError(f"Zoom level {zoom} is not supported.")
    
    tile_size = 512  # pixels
    tile_meter = tile_size * resolution  # actual meters represented by one tile
    # print(f"tile_meter: {tile_meter}")
    # Calculate required number of tiles (rounded up)
    tiles_needed = math.ceil(side_length_m / tile_meter)
    # print(f"tiles_needed: {tiles_needed}")
    # Add one extra tile in each direction to ensure sufficient padding
    padding_tiles = 1
    
    return tiles_needed, padding_tiles

def fetch_and_stitch_tiles(lat, lon, zoom, side_length_m):
    """
    Download sufficient tile grid and stitch them together
    """
    center_row, center_col = latlon_to_tile(lat, lon, zoom)
    tiles_needed, padding_tiles = calculate_grid_size(lat, lon, side_length_m, zoom)
    
    # Calculate total grid size (considering padding)
    grid_size = 2 * tiles_needed + 1
    # print(f"grid_size: {grid_size}")
    # Calculate starting row and column
    start_row = center_row - (grid_size // 2)
    start_col = center_col - (grid_size // 2)
    end_row = center_row + (grid_size // 2)
    end_col = center_col + (grid_size // 2)
    
    # Download all tiles
    tiles = []
    for row in tqdm.tqdm(range(start_row, start_row + grid_size), desc="Downloading tiles"):
        row_tiles = []
        for col in range(start_col, start_col + grid_size):
            # print(f"Fetching tile: Row={row}, Col={col}")
            img = fetch_tile_image(lat, lon, zoom, tile_row=row, tile_col=col)
            row_tiles.append(img)
        tiles.append(row_tiles)

    # Stitch images together
    tile_size = 512
    stitched_image = Image.new('RGB', (tile_size * grid_size, tile_size * grid_size))
    for i, row_tiles in enumerate(tiles):
        for j, img in enumerate(row_tiles):
            stitched_image.paste(img, (j * tile_size, i * tile_size))

    # Calculate center point position in stitched image
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    x, y = transformer.transform(lon, lat)
    
    resolution = resolutions.get(zoom)
    origin_x = -5120900.0
    origin_y = 9998100.0
    
    stitched_origin_x = origin_x + start_col * tile_size * resolution
    stitched_origin_y = origin_y - start_row * tile_size * resolution
    
    delta_x = x - stitched_origin_x
    delta_y = stitched_origin_y - y
    pixel_x = delta_x / resolution
    pixel_y = delta_y / resolution
    
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
    
    # Ensure crop area is within image bounds
    left = max(left, 0)
    upper = max(upper, 0)
    right = min(right, stitched_image.width)
    lower = min(lower, stitched_image.height)
    
    cropped_image = stitched_image.crop((left, upper, right, lower))
    return cropped_image

def save_images(stitched_image, cropped_image, lon, lat, zoom, save_folder="outputs/italia"):
    os.makedirs(save_folder, exist_ok=True)

    stitched_filename = os.path.join(
        save_folder,
        f"stitched_zoom{zoom}_lon{lon}_lat{lat}.jpg"
    )
    cropped_filename = os.path.join(
        save_folder,
        f"cropped_zoom{zoom}_lon{lon}_lat{lat}.jpg"
    )

    stitched_image.save(stitched_filename)
    print(f"saved to {stitched_filename}")

    cropped_image.save(cropped_filename)
    print(f"saved to {cropped_filename}")

def Italy_exact_position_image(lat, lon, side_length_m, zoom=18):
    """
    Get image for specified location and side length
    
    :param lat: latitude
    :param lon: longitude
    :param side_length_m: required side length (meters)
    :param zoom: zoom level
    :return: PIL.Image object
    """
    resolution = resolutions.get(zoom)
    if resolution is None:
        raise ValueError(f"Zoom level {zoom} is not supported.")
    
    # Calculate required pixel size
    pixels_needed = int(side_length_m / resolution)
    
    # Get stitched image and center point position
    stitched_image, center_pixel = fetch_and_stitch_tiles(lat, lon, zoom, side_length_m)
    
    # Crop image to required size
    cropped_image = crop_image_center(stitched_image, center_pixel, pixels_needed)
    
    return cropped_image

if __name__ == "__main__":
    lat = 41.89332322921207
    lon = 12.482934948791554
    side_length_m = 200
    
    image = Italy_exact_position_image(lat, lon, side_length_m)
    
    # Save image
    output_dir = "outputs/italia"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"italy_lon{lon}_lat{lat}_side{side_length_m}m.jpg"
    )
    image.save(output_path)
    print(f"saved to {output_path}")
