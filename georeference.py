#!/usr/bin/env python3

import os
import argparse
import numpy as np
import rasterio
from rasterio.transform import Affine
import glob
import json
from PIL import Image
import re

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stitch processed tiles back into a single image')
    
    # Input directories
    parser.add_argument('--tiles_dir', type=str, required=True, 
                        help='Directory containing processed tiles (output from super-resolution)')
    parser.add_argument('--input_projection', type=str, required=True,
                        help='Directory containing reference tiles with projection and georeference files')
    
    # Output file
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output GeoTIFF file path')
    
    # Optional parameters
    parser.add_argument('--tile_pattern', type=str, default='tile_*_*.png',
                        help='Pattern to match tile files (default: tile_*_*.png)')
    parser.add_argument('--flip_horizontal', action='store_true',
                        help='Flip the output image horizontally to correct mirroring issues')
    parser.add_argument('--flip_vertical', default=True, action='store_true',
                        help='Flip the output image vertically to correct mirroring issues')
    parser.add_argument('--reverse_i', action='store_true',
                        help='Reverse the i indices (columns) when placing tiles')
    parser.add_argument('--reverse_j', default=True,  action='store_true',
                        help='Reverse the j indices (rows) when placing tiles')
    
    return parser.parse_args()

def extract_tile_indices(filename):
    """Extract tile indices (i, j) from filename"""
    match = re.search(r'tile_(\d+)_(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def get_tile_metadata(tile_path, ref_dir):
    """Get metadata for a tile from its corresponding reference JSON file"""
    # Extract base filename (e.g., tile_0_0)
    base_name = os.path.splitext(os.path.basename(tile_path))[0]
    
    # Look for corresponding JSON file in reference directory
    json_path = os.path.join(ref_dir, f"{base_name}.json")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: No metadata found for {base_name} in {ref_dir}")
        return None

def get_projection_from_prj(tile_path, ref_dir):
    """Get projection from corresponding reference PRJ file"""
    # Extract base filename (e.g., tile_0_0)
    base_name = os.path.splitext(os.path.basename(tile_path))[0]
    
    # Look for corresponding PRJ file in reference directory
    prj_path = os.path.join(ref_dir, f"{base_name}.prj")
    
    if os.path.exists(prj_path):
        with open(prj_path, 'r') as f:
            return f.read()
    else:
        print(f"Warning: No PRJ file found for {base_name} in {ref_dir}")
        return None

def get_geotransform_from_wld(tile_path, ref_dir):
    """Get geotransform from corresponding reference WLD file"""
    # Extract base filename (e.g., tile_0_0)
    base_name = os.path.splitext(os.path.basename(tile_path))[0]
    
    # Look for corresponding WLD file in reference directory
    wld_path = os.path.join(ref_dir, f"{base_name}.wld")
    
    if os.path.exists(wld_path):
        with open(wld_path, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 6:
                # Parse world file parameters
                # Format: pixel width, rotation, rotation, pixel height, top-left x, top-left y
                return {
                    "pixel_width": float(lines[0].strip()),
                    "rotation_1": float(lines[1].strip()),
                    "rotation_2": float(lines[2].strip()),
                    "pixel_height": float(lines[3].strip()),
                    "top_left_x": float(lines[4].strip()),
                    "top_left_y": float(lines[5].strip())
                }
    
    print(f"Warning: No valid WLD file found for {base_name} in {ref_dir}")
    return None

def stitch_tiles(tiles_dir, ref_dir, output_file, tile_pattern='tile_*_*.png', 
                flip_horizontal=False, flip_vertical=False, reverse_i=False, reverse_j=False):
    """Stitch processed tiles into a single GeoTIFF using reference projection files"""
    # Find all tile files
    tile_paths = glob.glob(os.path.join(tiles_dir, tile_pattern))
    
    if not tile_paths:
        print(f"Error: No tiles found in {tiles_dir} matching pattern {tile_pattern}")
        return False
    
    print(f"Found {len(tile_paths)} tiles to stitch")
    
    # Extract tile indices and organize tiles
    tiles_info = []
    for tile_path in tile_paths:
        indices = extract_tile_indices(tile_path)
        if indices:
            i, j = indices
            
            # Get projection and geotransform from reference files
            metadata = get_tile_metadata(tile_path, ref_dir)
            projection = get_projection_from_prj(tile_path, ref_dir)
            geotransform = get_geotransform_from_wld(tile_path, ref_dir)
            
            if not projection or not geotransform:
                print(f"Warning: Missing projection or geotransform for {os.path.basename(tile_path)}")
                continue
                
            tiles_info.append((i, j, tile_path, metadata, projection, geotransform))
    
    if not tiles_info:
        print("Error: Could not extract indices or georeference information from tiles")
        return False
    
    # Determine grid dimensions
    max_i = max(info[0] for info in tiles_info) + 1
    max_j = max(info[1] for info in tiles_info) + 1
    
    print(f"Tile grid dimensions: {max_i}x{max_j}")
    
    # Load a sample tile to get dimensions and bands
    sample_img = Image.open(tiles_info[0][2])
    tile_width, tile_height = sample_img.size
    bands = len(sample_img.getbands())
    
    print(f"Tile dimensions: {tile_width}x{tile_height}, Bands: {bands}")
    
    # Get CRS from the first tile's projection
    from rasterio.crs import CRS
    crs = CRS.from_wkt(tiles_info[0][4])
    
    # Calculate bounds of the full image using geotransform information
    # We need to find the top-left corner of the entire image and the bottom-right corner
    
    # Get the top-left corner from the (0,0) tile if available
    top_left_tile = next((t for t in tiles_info if t[0] == 0 and t[1] == 0), None)
    
    if top_left_tile:
        # Use geotransform from the top-left tile
        geotransform = top_left_tile[5]
        left = geotransform["top_left_x"]
        top = geotransform["top_left_y"]
    else:
        # If no (0,0) tile, use the first tile and calculate back to where (0,0) would be
        first_tile = tiles_info[0]
        i, j = first_tile[0], first_tile[1]
        geotransform = first_tile[5]
        
        # Calculate where (0,0) would be
        pixel_width = geotransform["pixel_width"]
        pixel_height = geotransform["pixel_height"]
        left = geotransform["top_left_x"] - (i * tile_width * pixel_width)
        top = geotransform["top_left_y"] - (j * tile_height * pixel_height)
    
    # Calculate the bottom-right corner
    pixel_width = geotransform["pixel_width"]
    
    # PERUBAHAN: Selalu menegatifkan nilai pixel_height untuk membalik gambar secara vertikal
    pixel_height = abs(geotransform["pixel_height"])
    
    # Correct the pixel_width sign if we're flipping horizontally
    if flip_horizontal:
        pixel_width = -abs(pixel_width)
        # Adjust left coordinate to be at the right edge when flipped
        left = left + (max_i * tile_width * abs(pixel_width))
    
    # Bagian flip_vertical tidak perlu diubah karena kita sudah menegatifkan pixel_height
    # Namun kita perlu menyesuaikan koordinat top karena pixel_height sekarang negatif
    # Adjust top coordinate for the negative pixel_height
    top = top - (max_j * tile_height * abs(pixel_height))
    
    right = left + (max_i * tile_width * pixel_width)
    bottom = top + (max_j * tile_height * pixel_height)
    
    # Calculate output dimensions
    output_width = max_i * tile_width
    output_height = max_j * tile_height
    
    print(f"Output dimensions: {output_width}x{output_height}")
    print(f"Bounds: Left={left}, Top={top}, Right={right}, Bottom={bottom}")
    print(f"Pixel size: Width={pixel_width}, Height={pixel_height}")
    
    # Create output transform
    output_transform = Affine(
        pixel_width, 0.0, left,
        0.0, pixel_height, top
    )
    
    print(f"Output transform: {output_transform}")
    
    # Create output raster
    output_profile = {
        'driver': 'GTiff',
        'height': output_height,
        'width': output_width,
        'count': bands,
        'dtype': np.uint8,
        'crs': crs,
        'transform': output_transform,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
    }
    
    with rasterio.open(output_file, 'w', **output_profile) as dst:
        # Create an empty array to hold the full image
        full_img = np.zeros((bands, output_height, output_width), dtype=np.uint8)
        
        # Place each tile in the correct position
        for i, j, tile_path, _, _, _ in tiles_info:
            img = Image.open(tile_path)
            
            # Apply transformations to the tile if needed
            if flip_horizontal:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_vertical:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
            # Adjust indices based on reverse options
            i_adjusted = max_i - i - 1 if reverse_i else i
            j_adjusted = max_j - j - 1 if reverse_j else j
                
            img_data = np.array(img)
            
            # Handle different band arrangements
            if bands == 1:
                if len(img_data.shape) == 3:  # If image has extra dimensions
                    img_data = img_data[:,:,0]
                img_data = img_data[np.newaxis, :, :]  # Add band dimension
            else:
                # Ensure band dimension is first (rasterio convention)
                if len(img_data.shape) == 3 and img_data.shape[2] != bands:
                    print(f"Warning: Band count mismatch for {tile_path}. Expected {bands}, got {img_data.shape[2]}")
                    continue
                img_data = np.moveaxis(img_data, 2, 0)
            
            # Calculate position in the full image
            y_start = j_adjusted * tile_height
            y_end = y_start + tile_height
            x_start = i_adjusted * tile_width
            x_end = x_start + tile_width
            
            # Check bounds
            if y_end > output_height or x_end > output_width:
                print(f"Warning: Tile {i}_{j} exceeds output dimensions, will be cropped")
                # Adjust end coordinates to fit within output dimensions
                y_end = min(y_end, output_height)
                x_end = min(x_end, output_width)
                # Crop tile data accordingly
                img_data = img_data[:, :(y_end-y_start), :(x_end-x_start)]
            
            # Place tile in the full image
            full_img[:, y_start:y_end, x_start:x_end] = img_data
            
            print(f"Placed tile {i}_{j} at position ({x_start},{y_start})")
        
        # Write the full image to the output file
        dst.write(full_img)
    
    print(f"Successfully stitched tiles to {output_file}")
    return True

def main():
    """Main function"""
    args = parse_arguments()
    
    # Check if input directory exists
    if not os.path.isdir(args.tiles_dir):
        print(f"Error: Tiles directory {args.tiles_dir} does not exist")
        return
    
    # Check if reference directory exists
    if not os.path.isdir(args.input_projection):
        print(f"Error: Reference directory {args.input_projection} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Stitch tiles
    success = stitch_tiles(
        args.tiles_dir,
        args.input_projection,
        args.output_file,
        args.tile_pattern,
        args.flip_horizontal,
        args.flip_vertical,
        args.reverse_i,
        args.reverse_j
    )
    
    if success:
        print("Stitching completed successfully")
    else:
        print("Stitching failed")

if __name__ == "__main__":
    main()
