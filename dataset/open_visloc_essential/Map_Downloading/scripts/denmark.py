import math
import requests
from pyproj import Transformer
import os
from PIL import Image
import io
import tqdm

# Username and password, obtained from url, might need to be updated regularly
USERNAME = "STGGAEECCJ"
PASSWORD = "een!oM8HJ7_!aCw6"

# WMTS TileMatrix information
TILE_MATRIX_INFO = {
    str(i): {
        "ScaleDenominator": scale,
        "TopLeftCorner": (120000.0, 6500000.0),
        "TileWidth": 256,
        "TileHeight": 256,
        "MatrixWidth": width,
        "MatrixHeight": height,
    }
    for i, (scale, width, height) in enumerate([
        (5851428.571428571, 3, 2),
        (2925714.285714286, 5, 3),
        (1462857.142857143, 9, 6),
        (731428.571428571, 17, 12),
        (365714.285714286, 34, 23),
        (182857.142857143, 68, 46),
        (91428.571428571, 135, 92),
        (45714.285714286, 269, 184),
        (22857.142857143, 538, 367),
        (11428.571428571, 1075, 733),
        (5714.285714286, 2149, 1465),
        (2857.142857143, 4297, 2930),
        (1428.571428571, 8594, 5860),
        (714.285714286, 17188, 11719),
        (357.142857143, 34375, 23438),
        (178.571428571, 68750, 46875),
    ])
}

# Convert latitude/longitude to EPSG:25832
transformer = Transformer.from_crs("epsg:4326", "epsg:25832", always_xy=True)

def latlon_to_tile(lat, lon, level):
    """
    Convert latitude/longitude to tile column and row numbers.
    
    :param lat: latitude
    :param lon: longitude
    :param level: zoom level (string type)
    :return: (tile_col, tile_row)
    """
    tile_matrix_info = TILE_MATRIX_INFO[level]
    scale = tile_matrix_info["ScaleDenominator"]
    top_left_x, top_left_y = tile_matrix_info["TopLeftCorner"]
    tile_width = tile_matrix_info["TileWidth"]
    tile_height = tile_matrix_info["TileHeight"]
    resolution = scale * 0.28e-3  # meters per pixel
    
    # EPSG:25832 coordinates
    x, y = transformer.transform(lon, lat)

    dx = x - top_left_x
    dy = top_left_y - y
    tile_col = int(math.floor(dx / (tile_width * resolution)))
    tile_row = int(math.floor(dy / (tile_height * resolution)))

    return tile_col, tile_row

def fetch_tile_image(level, tile_row=None, tile_col=None, retries=3):
    """
    Download image for specified tile
    """
    url_template = (
        "https://services.datafordeler.dk/GeoDanmarkOrto/orto_foraar_wmts/1.0.0/wmts?"
        "username={username}&password={password}&layer=orto_foraar_wmts&style=default&"
        "tilematrixset=KortforsyningTilingDK&Service=WMTS&Request=GetTile&Version=1.0.0&"
        "Format=image/jpeg&TileMatrix={tilematrix}&TileCol={tilecol}&TileRow={tilerow}"
    )
    url = url_template.format(
        username=USERNAME,
        password=PASSWORD,
        tilematrix=level,
        tilecol=tile_col,
        tilerow=tile_row
    )


    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
            return img
        except requests.RequestException as e:
            print(f"请求失败 (Row={tile_row}, Col={tile_col}), 尝试 {attempt}/{retries}: {e}")
            if attempt == retries:
                print(f"所有重试次数均失败。使用黑色图像替代 (Row={tile_row}, Col={tile_col})")
        except Exception as e:
            print(f"图像处理失败 (Row={tile_row}, Col={tile_col}): {e}")
            break

    tile_matrix_info = TILE_MATRIX_INFO[level]
    return Image.new("RGB", (tile_matrix_info["TileWidth"], tile_matrix_info["TileHeight"]), (0, 0, 0))

def calculate_grid_size(lat, lon, side_length_m, level):
    """
    Calculate required tile grid size for given side length
    """
    tile_matrix_info = TILE_MATRIX_INFO[level]
    resolution = tile_matrix_info["ScaleDenominator"] * 0.28e-3
    tile_size = tile_matrix_info["TileWidth"]
    tile_meter = tile_size * resolution  # 一个瓦片对应的实际米数
    print(f"tile_meter: {tile_meter}")
    
    # Calculate required number of tiles (rounded up)
    tiles_needed = math.ceil(side_length_m / tile_meter)
    print(f"tiles_needed: {tiles_needed}")
    
    return tiles_needed, 1  # 1 为padding_tiles

def fetch_and_stitch_tiles(lat, lon, level, side_length_m):
    """
    Download sufficient tile grid and stitch them together
    """
    center_col, center_row = latlon_to_tile(lat, lon, level)

    tiles_needed, padding_tiles = calculate_grid_size(lat, lon, side_length_m, level)
    

    grid_size = 2 * tiles_needed + 1
    print(f"grid_size: {grid_size}")
    
    # Calculate starting row and column
    start_row = center_row - (grid_size // 2)
    start_col = center_col - (grid_size // 2)
    
    # Download all tiles
    tiles = []
    for row in range(start_row, start_row + grid_size):
        row_tiles = []
        for col in range(start_col, start_col + grid_size):
            # print(f"Fetching tile: Row={row}, Col={col}")
            img = fetch_tile_image(level, tile_row=row, tile_col=col)
            row_tiles.append(img)
        tiles.append(row_tiles)

    tile_matrix_info = TILE_MATRIX_INFO[level]
    tile_width = tile_matrix_info["TileWidth"]
    tile_height = tile_matrix_info["TileHeight"]
    
    # Stitch images together
    stitched_image = Image.new('RGB', (tile_width * grid_size, tile_height * grid_size))
    for i, row_tiles in enumerate(tiles):
        for j, img in enumerate(row_tiles):
            stitched_image.paste(img, (j * tile_width, i * tile_height))

    # Calculate center point position in stitched image
    x, y = transformer.transform(lon, lat)
    resolution = tile_matrix_info["ScaleDenominator"] * 0.28e-3
    top_left_x, top_left_y = tile_matrix_info["TopLeftCorner"]
    
    stitched_origin_x = top_left_x + start_col * tile_width * resolution
    stitched_origin_y = top_left_y - start_row * tile_height * resolution
    
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
    
    # Ensure cropping area is within image range
    left = max(left, 0)
    upper = max(upper, 0)
    right = min(right, stitched_image.width)
    lower = min(lower, stitched_image.height)
    
    cropped_image = stitched_image.crop((left, upper, right, lower))
    return cropped_image

def denmark_exact_position_image(lat, lon, side_length_m, level="15"):
    """
    Get image for specified location and side length
    
    :param lat: latitude
    :param lon: longitude
    :param side_length_m: required side length (meters)
    :param level: zoom level
    :return: PIL.Image object
    """
    tile_matrix_info = TILE_MATRIX_INFO[level]
    resolution = tile_matrix_info["ScaleDenominator"] * 0.28e-3
    
    # Calculate required pixel size
    pixels_needed = int(side_length_m / resolution)
    
    # Get stitched image and center point position
    stitched_image, center_pixel = fetch_and_stitch_tiles(lat, lon, level, side_length_m)
    
    # Crop image of specified size
    cropped_image = crop_image_center(stitched_image, center_pixel, pixels_needed)
    
    return cropped_image

if __name__ == "__main__":
    lat = 55.71659837626672
    lon = 12.533825234946141

    side_length_m = 100
    
    image = denmark_exact_position_image(lat, lon, side_length_m)
    
    # Save image
    output_dir = "outputs/denmark"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"denmark_lon{lon}_lat{lat}_side{side_length_m}m.jpg"
    )
    image.save(output_path)
    print(f"saved to {output_path}")
