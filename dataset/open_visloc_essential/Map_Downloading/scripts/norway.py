#https://gatekeeper1.geonorge.no/BaatGatekeeper/gk/gk.nib_utm33_wmts_v2?&gkt=BEC147D107F0FC8D49140BE086DD2F1CA054BD31C709747A75C8369E135220FA8A644E52F69F1772459B5CCE850E31C95687B02F87FB7BE38D018F0853CD0DA1&layer=Nibcache_UTM33_EUREF89&style=default&tilematrixset=default028mm&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image%2Fpng&TileMatrix=17&TileCol=65097&TileRow=57304

# if token expires: go to https://www.norgeskart.no/, switch to satellite view. zoom in to max zoom level. Press Ctrl+Shift+I, go to develop mode. Go to "Network" and select an image patch. Find the url of a displayed image patch. Replace the WMTS_TOKEN below



import math
import requests
import os
from PIL import Image
import io
from pyproj import CRS, Transformer
import tqdm
# Constants definition
ZOOM_LEVEL = 17
WMTS_TOKEN = "4FA69CF965830FC136A3980D41F654D3ADD2F04F00E07A0F27056F8C1C888B834F125DE714C7A9B1C0EF229332EBA5BAA597CD703B546D648D018F0853CD0DA1"
# the token has to be manually updated regularly, from the url of map website
WMTS_BASE_URL = (
    "https://gatekeeper1.geonorge.no/BaatGatekeeper/gk/gk.nib_utm33_wmts_v2"
    f"?&gkt={WMTS_TOKEN}"
)
LAYER_ID = "Nibcache_UTM33_EUREF89"
TILEMATRIXSET_ID = "default028mm"
STYLE_ID = "default"
FORMAT_ID = "image/png"
WMTS_VERSION = "1.0.0"
# wmts service: https://gatekeeper1.geonorge.no/BaatGatekeeper/gk/gk.nib_utm33_wmts_v2?gkt=BEC147D107F0FC8D49140BE086DD2F1CA054BD31C709747A75C8369E135220FA8A644E52F69F1772459B5CCE850E31C95687B02F87FB7BE38D018F0853CD0DA1&Service=WMTS&Request=GetCapabilities&Version=1.0.0
# <ows:Identifier>17</ows:Identifier>
# <ScaleDenominator>590.2971540177829</ScaleDenominator>
# <TopLeftCorner>-2500000.0 9045984.0</TopLeftCorner>
# <TileWidth>256</TileWidth>
# <TileHeight>256</TileHeight>
# <MatrixWidth>131073</MatrixWidth>
# <MatrixHeight>131073</MatrixHeight>
SCALE_DENOMINATOR_16 = 590.2971540177829
TOPLEFT_X            = -2500000.0
TOPLEFT_Y            = 9045984.0
TILE_SIZE            = 256
MATRIX_WIDTH         = 131073
MATRIX_HEIGHT        = 131073

RESOLUTION = SCALE_DENOMINATOR_16 * 0.28e-3
TILE_SIZE_METERS = TILE_SIZE * RESOLUTION

crs_wgs84 = CRS.from_epsg(4326)
crs_utm33 = CRS.from_epsg(25833)
transformer = Transformer.from_crs(crs_wgs84, crs_utm33, always_xy=True)

def latlon_to_tile(lon, lat, zoom=ZOOM_LEVEL):
    """
    Convert latitude and longitude to tile row and column numbers
    """
    x, y = transformer.transform(lon, lat)
    
    dx = x - TOPLEFT_X
    dy = TOPLEFT_Y - y
    
    tile_col = math.floor(dx / TILE_SIZE_METERS)
    tile_row = math.floor(dy / TILE_SIZE_METERS)
    
    tile_col = max(0, min(tile_col, MATRIX_WIDTH - 1))
    tile_row = max(0, min(tile_row, MATRIX_HEIGHT - 1))
    
    return tile_row, tile_col

def fetch_tile_image(lat, lon, zoom, tile_row=None, tile_col=None, retries=5):
    """
    Download image for specified tile
    """
    if tile_row is None or tile_col is None:
        tile_row, tile_col = latlon_to_tile(lon, lat, zoom)

    url = construct_wmts_url(tile_col, tile_row, zoom)
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

    return Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))

def calculate_grid_size(lat, lon, side_length_m, zoom):
    """
    Calculate required tile grid size for given side length
    """
    tile_meter = TILE_SIZE * RESOLUTION
    
    tiles_needed = math.ceil(side_length_m / tile_meter)
    
    return tiles_needed

def fetch_and_stitch_tiles(lat, lon, zoom, side_length_m):
    """
    Download sufficient tile grid and stitch them together
    """
    center_row, center_col = latlon_to_tile(lon, lat, zoom)
    tiles_needed= calculate_grid_size(lat, lon, side_length_m, zoom)
    
    # Calculate total grid size (considering padding)
    grid_size = 2 * tiles_needed + 1
    print(f"grid_size: {grid_size}")
    
    start_row = center_row - (grid_size // 2)
    start_col = center_col - (grid_size // 2)
    
    tiles = []
    for row in tqdm.tqdm(range(start_row, start_row + grid_size), desc="Downloading tiles"):
        row_tiles = []
        for col in range(start_col, start_col + grid_size):
            img = fetch_tile_image(lat, lon, zoom, tile_row=row, tile_col=col)
            row_tiles.append(img)
        tiles.append(row_tiles)

    # Stitch images together
    stitched_image = Image.new('RGB', (TILE_SIZE * grid_size, TILE_SIZE * grid_size))
    for i, row_tiles in enumerate(tiles):
        for j, img in enumerate(row_tiles):
            stitched_image.paste(img, (j * TILE_SIZE, i * TILE_SIZE))

    # Calculate center point position in stitched image
    x, y = transformer.transform(lon, lat)
    
    stitched_origin_x = TOPLEFT_X + start_col * TILE_SIZE * RESOLUTION
    stitched_origin_y = TOPLEFT_Y - start_row * TILE_SIZE * RESOLUTION
    
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

def construct_wmts_url(tile_col, tile_row, zoom=ZOOM_LEVEL):
    """
    Construct WMTS KVP request URL
    """
    # Format exactly matching the provided URL structure
    url = (
        f"{WMTS_BASE_URL}"
        f"&layer={LAYER_ID}"
        f"&style={STYLE_ID}"
        f"&tilematrixset={TILEMATRIXSET_ID}"
        f"&Service=WMTS"
        f"&Request=GetTile"
        f"&Version={WMTS_VERSION}"
        f"&Format={FORMAT_ID}"
        f"&TileMatrix={zoom}"
        f"&TileCol={tile_col}"
        f"&TileRow={tile_row}"
    )
    return url

def norway_exact_position_image(lat, lon, side_length_m, zoom=ZOOM_LEVEL):
    """
    Get image for specified location and side length
    
    :param lat: latitude
    :param lon: longitude
    :param side_length_m: required side length (meters)
    :param zoom: zoom level
    :return: PIL.Image object
    """
    # Calculate required pixel size
    pixels_needed = int(side_length_m / RESOLUTION)
    
    # Get stitched image and center point position
    stitched_image, center_pixel = fetch_and_stitch_tiles(lat, lon, zoom, side_length_m)
    
    # Crop image of specified size
    cropped_image = crop_image_center(stitched_image, center_pixel, pixels_needed)
    
    return cropped_image

if __name__ == "__main__":
    # Website key needs to be manually updated periodically in WMTS_TOKEN
    lat = 59.933216567569026
    lon = 10.77852647122692
    side_length_m = 30  # specify side length as 50 meters
    
    image = norway_exact_position_image(lat, lon, side_length_m)
    
    # Save image
    output_dir = "outputs/norway"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"norway_lon{lon}_lat{lat}_side{side_length_m}m.jpg"
    )
    image.save(output_path)
    print(f"saved to {output_path}")

