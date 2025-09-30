#!/usr/bin/env python3

import os
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
import sys
from enum import Enum
from PIL import Image
import json

class ImageType(Enum):
    HR = "hr"
    REF = "ref"
    SENTINEL = "sentinel"

class TileProcessor:
    def __init__(self, image_type, input_path, output_folder, tile_size):
        self.image_type = image_type
        self.input_path = input_path
        self.output_folder = output_folder
        self.tile_size = tile_size
        
        # Set default tile sizes based on image type
        self.default_sizes = {
            ImageType.HR: 256,
            ImageType.REF: 256,
            ImageType.SENTINEL: 26
        }
        
        if tile_size is None:
            self.tile_size = self.default_sizes[image_type]

    def create_tiles(self):
        """Create tiles from input TIF file"""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        try:
            with rasterio.open(self.input_path) as src:
                # Get image dimensions
                width = src.width
                height = src.height
                bands = src.count
                
                # Extract projection information
                projection = src.crs.wkt if src.crs else None
                transform = src.transform
                
                # Calculate the number of tiles
                num_tiles_x = int(np.ceil(width / self.tile_size))
                num_tiles_y = int(np.ceil(height / self.tile_size))
                
                print(f"\nProcessing {self.image_type.value.upper()} image:")
                print(f"Image size: {width}x{height} pixels, {bands} bands")
                print(f"Creating {num_tiles_x}x{num_tiles_y} tiles of size {self.tile_size}x{self.tile_size}")
                
                # Get input filename without extension
                base_filename = os.path.splitext(os.path.basename(self.input_path))[0]
                
                # Create tiles
                for i in range(num_tiles_x):
                    for j in range(num_tiles_y):
                        self._process_single_tile(src, base_filename, i, j, width, height, bands, projection, transform)
                
                return True
                
        except rasterio.errors.RasterioIOError:
            print(f"Error: Could not open {self.input_path}")
            return False

    def _process_single_tile(self, src, base_filename, i, j, width, height, bands, projection, transform):
        """Process a single tile"""
        # Calculate pixel coordinates
        x_offset = i * self.tile_size
        y_offset = j * self.tile_size
        
        # Handle edge cases where tile would go beyond image boundaries
        x_size = min(self.tile_size, width - x_offset)
        y_size = min(self.tile_size, height - y_offset)
        
        # Skip if tile size is 0 in any dimension
        if x_size <= 0 or y_size <= 0:
            return
            
        # Skip if the tile size doesn't match the requested tile size
        # This will skip residual tiles at the edges that aren't the full size
        if x_size != self.tile_size or y_size != self.tile_size:
            print(f"Skipping residual tile at ({i},{j}) with size {x_size}x{y_size} (expected {self.tile_size}x{self.tile_size})")
            return
        
        # Create output filename base (without extension)
        output_filename_base = f"tile_{i}_{j}"
        output_path_base = os.path.join(self.output_folder, output_filename_base)
        
        # Create a window for reading the data
        window = Window(x_offset, y_offset, x_size, y_size)
        
        # Read the data for all bands
        data = src.read(window=window)
        
        # Convert to 8-bit if necessary and handle different band counts
        if data.dtype != np.uint8:
            # Normalize and convert to 8-bit
            data = ((data - data.min()) * (255.0 / (data.max() - data.min()))).astype(np.uint8)
        
        # Transpose data from (bands, height, width) to (height, width, bands)
        data = np.transpose(data, (1, 2, 0))
        
        # Handle different numbers of bands
        if bands == 1:
            # Single band - create grayscale image
            img = Image.fromarray(data.squeeze(), 'L')
        elif bands == 3:
            # Three bands - create RGB image
            img = Image.fromarray(data, 'RGB')
        elif bands == 4:
            # Four bands - create RGBA image
            img = Image.fromarray(data, 'RGBA')
        else:
            # For other band counts, take first three bands as RGB
            img = Image.fromarray(data[:, :, :3], 'RGB')
        
        # Save as PNG
        img.save(f"{output_path_base}.png", 'PNG')
        
        # Calculate and save the georeference information for this tile
        if projection:
            # Calculate the new geotransform for this tile
            # Get the coordinates of the top-left corner of the tile
            tile_transform = rasterio.transform.from_origin(
                transform.c + x_offset * transform.a,  # x_min (left)
                transform.f + y_offset * transform.e,  # y_max (top)
                transform.a,  # pixel width
                transform.e   # pixel height (negative)
            )
            
            # Save projection to .prj file
            with open(f"{output_path_base}.prj", 'w') as prj_file:
                prj_file.write(projection)
            
            # Save georeference information to .wld file (world file)
            with open(f"{output_path_base}.wld", 'w') as wld_file:
                wld_file.write(f"{tile_transform.a}\n")  # pixel width
                wld_file.write(f"{tile_transform.b}\n")  # rotation (usually 0)
                wld_file.write(f"{tile_transform.d}\n")  # rotation (usually 0)
                wld_file.write(f"{tile_transform.e}\n")  # pixel height (negative)
                wld_file.write(f"{tile_transform.c}\n")  # x-coordinate of upper-left pixel
                wld_file.write(f"{tile_transform.f}\n")  # y-coordinate of upper-left pixel
            
            # Also save metadata in a more readable JSON format
            metadata = {
                "projection": projection,
                "geotransform": {
                    "pixel_width": tile_transform.a,
                    "rotation_1": tile_transform.b,
                    "rotation_2": tile_transform.d,
                    "pixel_height": tile_transform.e,
                    "top_left_x": tile_transform.c,
                    "top_left_y": tile_transform.f
                },
                "tile_info": {
                    "tile_index_x": i,
                    "tile_index_y": j,
                    "width": x_size,
                    "height": y_size,
                    "bands": bands
                },
                "source_file": os.path.basename(self.input_path)
            }
            
            with open(f"{output_path_base}.json", 'w') as json_file:
                json.dump(metadata, json_file, indent=4)
        
        print(f"Created tile: {output_filename_base}.png with projection files")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process and tile multiple types of satellite imagery')
    
    # Input files
    parser.add_argument('--hr_input', type=str, help='Path to input HR TIF file')
    parser.add_argument('--ref_input', type=str, help='Path to input Reference TIF file')
    parser.add_argument('--sentinel_input', type=str, help='Path to input Sentinel-2 TIF file')
    
    # Output folders
    parser.add_argument('--output_base', type=str, default='./data_img', help='Base output directory for all tiles')
    
    # Optional tile sizes
    parser.add_argument('--hr_tile_size', type=int, help='Tile size for HR image (default: 800)')
    parser.add_argument('--ref_tile_size', type=int, help='Tile size for Reference image (default: 800)')
    parser.add_argument('--sentinel_tile_size', type=int, help='Tile size for Sentinel image (default: 48)')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create base output directory
    os.makedirs(args.output_base, exist_ok=True)
    
    # Define processing configurations
    configs = [
        (ImageType.HR, args.hr_input, "hr_tiles", args.hr_tile_size),
        (ImageType.REF, args.ref_input, "ref_tiles", args.ref_tile_size),
        (ImageType.SENTINEL, args.sentinel_input, "sentinel_tiles", args.sentinel_tile_size)
    ]
    
    # Process each image type
    for img_type, input_path, subfolder, tile_size in configs:
        if input_path:
            if not os.path.isfile(input_path):
                print(f"Error: Input file {input_path} does not exist")
                continue
                
            output_folder = os.path.join(args.output_base, subfolder)
            processor = TileProcessor(img_type, input_path, output_folder, tile_size)
            
            print(f"\nProcessing {img_type.value.upper()} image:")
            print(f"Input: {input_path}")
            print(f"Output: {output_folder}")
            print(f"Tile size: {processor.tile_size}x{processor.tile_size}")
            
            success = processor.create_tiles()
            
            if success:
                print(f"{img_type.value.upper()} tiling completed successfully")
            else:
                print(f"{img_type.value.upper()} tiling failed")

if __name__ == "__main__":
    main()