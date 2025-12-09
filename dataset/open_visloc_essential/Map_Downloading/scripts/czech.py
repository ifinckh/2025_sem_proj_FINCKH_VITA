import math
import requests
from pyproj import Proj, Transformer
import os
from PIL import Image
import io
import tqdm
# Projection: WGS84 (EPSG:4326) -> S-JTSK / Krovak East North (EPSG:5514)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=True)

# WMTS TileMatrix parameters
tile_matrices = {
    0:  {"TopLeftCorner": (-925000.0, -920000.0), "TileWidth": 256, "TileHeight": 256, "ScaleDenominator": 7315200.0},
    1:  {"TopLeftCorner": (-925000.0, -920000.0), "TileWidth": 256, "TileHeight": 256, "ScaleDenominator": 3657600.0},
    2:  {"TopLeftCorner": (-925000.0, -920000.0), "TileWidth": 256, "TileHeight": 256, "ScaleDenominator": 1828800.0},
    3:  {"TopLeftCorner": (-925000.0, -920000.0), "TileWidth": 256, "TileHeight": 256, "ScaleDenominator": 914400.0},
    4:  {"TopLeftCorner": (-925000.0, -920000.0), "TileWidth": 256, "TileHeight": 256, "ScaleDenominator": 457200.0},
    5:  {"TopLeftCorner": (-925000.0, -920000.0), "TileWidth": 256, "TileHeight": 256, "ScaleDenominator": 228600.0},
    6:  {"TopLeftCorner": (-925000.0, -920000.0), "TileWidth": 256, "TileHeight": 256, "ScaleDenominator": 114300.0},
    7:  {"TopLeftCorner": (-925000.0, -920000.0), "TileWidth": 256, "TileHeight": 256, "ScaleDenominator": 57150.0},
    8:  {"TopLeftCorner": (-925000.0, -920000.0), "TileWidth": 256, "TileHeight": 256, "ScaleDenominator": 28575.0},
    9:  {"TopLeftCorner": (-925000.0, -920000.0), "TileWidth": 256, "TileHeight": 256, "ScaleDenominator": 14287.5},
    10: {"TopLeftCorner": (-925000.0, -920000.0), "TileWidth": 256, "TileHeight": 256, "ScaleDenominator": 7143.75},
    11: {"TopLeftCorner": (-925000.0, -920000.0), "TileWidth": 256, "TileHeight": 256, "ScaleDenominator": 3571.875},
    12: {"TopLeftCorner": (-925000.0, -920000.0), "TileWidth": 256, "TileHeight": 256, "ScaleDenominator": 1785.9375},
    13: {"TopLeftCorner": (-925000.0, -920000.0), "TileWidth": 256, "TileHeight": 256, "ScaleDenominator": 892.96875},
    14: {"TopLeftCorner": (-925000.0, -920000.0), "TileWidth": 256, "TileHeight": 256, "ScaleDenominator": 446.484375},
    15: {"TopLeftCorner": (-925000.0, -920000.0), "TileWidth": 256, "TileHeight": 256, "ScaleDenominator": 223.2421875},
}

pixel_size = 0.00028

def latlon_to_tile(lat, lon, zoom):
    """
    Convert latitude/longitude to tile row/column numbers
    """
    x, y = transformer.transform(lon, lat)
    params = tile_matrices[zoom]
    X0, Y0 = params["TopLeftCorner"]
    tile_width = params["TileWidth"]
    tile_height = params["TileHeight"]
    scale_denominator = params["ScaleDenominator"]
    resolution = scale_denominator * pixel_size
    
    tile_col = math.floor((x - X0) / (tile_width * resolution))
    tile_row = math.floor((Y0 - y) / (tile_height * resolution))
    
    return tile_row, tile_col

def fetch_tile_image(lat, lon, zoom, tile_row=None, tile_col=None, retries=3):
    """
    Download image for specified tile
    """
    if tile_row is None or tile_col is None:
        tile_row, tile_col = latlon_to_tile(lat, lon, zoom)

    layer = "orto"
    style = "default"
    tile_matrix_set = "jtsk:epsg:5514"
    fmt = "image/jpeg"
    
    wmts_url = (
        f"https://geoportal.cuzk.cz/WMTS_ORTOFOTO/WMTService.aspx?"
        f"SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER={layer}&STYLE={style}&"
        f"TILEMATRIXSET={tile_matrix_set}&TILEMATRIX={zoom}&"
        f"TILEROW={tile_row}&TILECOL={tile_col}&FORMAT={fmt}"
    )

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(wmts_url, timeout=10)
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

    return Image.new("RGB", (256, 256), (0, 0, 0))

def calculate_grid_size(lat, lon, side_length_m, zoom):
    """
    Calculate required tile grid size for given side length
    """
    params = tile_matrices[zoom]
    resolution = params["ScaleDenominator"] * pixel_size
    tile_size = params["TileWidth"]
    
    tile_meter = tile_size * resolution  # Actual meters represented by one tile
    print(f"tile_meter: {tile_meter}")
    
    # Calculate required number of tiles (rounded up)
    tiles_needed = 2 * math.ceil(side_length_m / tile_meter) + 1

    return tiles_needed

def fetch_and_stitch_tiles(lat, lon, zoom, side_length_m):
    """
    Download sufficient tile grid and stitch them together
    """
    center_row, center_col = latlon_to_tile(lat, lon, zoom)
    tiles_needed = calculate_grid_size(lat, lon, side_length_m, zoom)
    
    # Calculate total grid size (considering padding)
    grid_size = tiles_needed
    print(f"grid_size: {grid_size}")
    
    # Calculate starting row and column
    start_row = center_row - (grid_size // 2)
    start_col = center_col - (grid_size // 2)
    
    # Download all tiles
    tiles = []
    for row in tqdm.tqdm(range(start_row, start_row + grid_size), desc="Downloading tiles"):
        row_tiles = []
        for col in range(start_col, start_col + grid_size):
            print(f"Fetching tile: Row={row}, Col={col}")
            img = fetch_tile_image(lat, lon, zoom, tile_row=row, tile_col=col)
            row_tiles.append(img)
        tiles.append(row_tiles)

    params = tile_matrices[zoom]
    tile_width = params["TileWidth"]
    tile_height = params["TileHeight"]

    # Stitch images together
    stitched_image = Image.new('RGB', (tile_width * grid_size, tile_height * grid_size))
    for i, row_tiles in enumerate(tiles):
        for j, img in enumerate(row_tiles):
            stitched_image.paste(img, (j * tile_width, i * tile_height))

    # Calculate center point position in stitched image
    x, y = transformer.transform(lon, lat)
    params = tile_matrices[zoom]
    X0, Y0 = params["TopLeftCorner"]
    resolution = params["ScaleDenominator"] * pixel_size
    
    stitched_origin_x = X0 + start_col * tile_width * resolution
    stitched_origin_y = Y0 - start_row * tile_height * resolution
    
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

def czech_exact_position_image(lat, lon, side_length_m, zoom=13):
    """
    Get image for specified location and side length
    
    :param lat: latitude
    :param lon: longitude
    :param side_length_m: required side length (meters)
    :param zoom: zoom level
    :return: PIL.Image object
    """
    params = tile_matrices[zoom]
    resolution = params["ScaleDenominator"] * pixel_size
    
    # Calculate required pixel size
    pixels_needed = int(side_length_m / resolution)
    
    # Get stitched image and center point position
    stitched_image, center_pixel = fetch_and_stitch_tiles(lat, lon, zoom, side_length_m)
    
    # Crop image to required size
    cropped_image = crop_image_center(stitched_image, center_pixel, pixels_needed)
    
    return cropped_image

if __name__ == "__main__":
    lat = 50.081676
    lon = 14.420829
    side_length_m = 60
    
    image = czech_exact_position_image(lat, lon, side_length_m)
    
    output_dir = "outputs/czech"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"czech_lon{lon}_lat{lat}_side{side_length_m}m.jpg"
    )
    image.save(output_path)
    print(f"saved to {output_path}")