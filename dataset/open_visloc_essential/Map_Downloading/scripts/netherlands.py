# import streamlit as st

# import glob
# import json
# import datetime

# from PIL import Image
# import numpy as np

# import matplotlib.pyplot as plt
# import numpy as np

import sys
sys.path.append('/work/vita/zimin/open_visloc_benchmark/rosbag-dataset-devkit')
from dsutils import worldproj
from dsutils import tileset

RESOLUTION = 0.04545455
# RESOLUTION = 0.07465472191116088 # 0.07464553543474242 
zoom = 21 

# ----
TILE_SERVERS = [
    dict(
        name = 'PDOK Luchtfoto RGB',
        url = 'https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0?request=GetCapabilities&service=wmts',
        tileset_name = 'EPSG:3857', # Pseudo-Mercator
        layer_name = 'Actueel_ortho25',
        zoom = 21, # can go up to 19
        crs = None,
    ),
    dict(
        name = 'PDOK Luchtfoto IR',
        url = 'https://service.pdok.nl/hwh/luchtfotocir/wmts/v1_0?request=GetCapabilities&service=wmts',
        tileset_name = 'EPSG:3857', # Pseudo-Mercator
        layer_name = 'Actueel_ortho25IR',
        zoom = 19,
        crs = None,
    ),
    dict(
        name = 'PDOK AHN3 (RGB)',
        url = 'https://service.pdok.nl/rws/ahn3/wmts/v1_0?request=getcapabilities&service=wmts',
        tileset_name = 'EPSG:3857', # Pseudo-Mercator
        layer_name = 'ahn3_05m_dsm',
        zoom = 19,
        crs = None,
    ),
    dict( # NOTE: this is a WCS server
        name = 'PDOK AHN3 (Float)',
        url = 'https://service.pdok.nl/rws/ahn3/wcs/v1_0?request=getcapabilities&service=wcs',
        tileset_name = 'EPSG:3857', # Pseudo-Mercator
        layer_name = 'ahn3_05m_dsm',
        zoom = 19,
        crs = None,
    ),
    dict(
        name = 'PDOK OpenTopo',
        url = 'https://geodata.nationaalgeoregister.nl/tiles/service/wmts?request=GetCapabilities&service=WMTS',
        tileset_name = 'EPSG:28992',
        layer_name = 'Actueel_ortho25IR',
        zoom = 14,
        crs = None,
    ),
    dict(
        name = 'Sinica TW',
        url = 'http://gis.sinica.edu.tw/worldmap/wmts',
        tileset_name = 'GoogleMapsCompatible', 
        layer_name = 'OSM',
        zoom = 13, # Pseudo-Mercator
        crs = 'EPSG:3857', 
    ),
]

# ----

TILE_SERVER = TILE_SERVERS[0]
url = TILE_SERVER['url']
tileset_name = TILE_SERVER['tileset_name']
layer_name = TILE_SERVER['layer_name']
# zoom = TILE_SERVER['zoom']
crs = TILE_SERVER['crs']

tile_cache_dir = './tilecache'

service = tileset.TileService(url, cache_dir=tile_cache_dir)
layer_names = service.layer_names
tileset_names = service.tileset_names


tile_source = service.get_tilesource(zoom, tileset_name, layer_name)



def nl_exact_position_image(lat, lon, side_length_m=50, retries=3, delay=2):
    lonlat = [lon, lat]
    imsize_meters = [side_length_m, side_length_m]
    imsize_pixels = [side_length_m / RESOLUTION, side_length_m / RESOLUTION]

    for attempt in range(retries):
        try:
            # img, proj = tile_source.get_PIL_centered_lonlat_meters(lonlat, return_proj=True, imsize_meters=imsize_meters)
            img = tile_source.get_PIL_centered_lonlat(lonlat, imsize_px=imsize_pixels)
            return img
        except Exception as e:
            print(f"Tile download failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError("Failed to download tile after multiple attempts.")
    