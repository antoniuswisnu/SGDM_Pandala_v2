import rasterio
from rasterio.enums import Resampling
import numpy as np
import os
from pathlib import Path

def resample_to_1m_resolution(input_path, output_path=None, delete_input=True):
    """
    Resample a GeoTIFF file to 1 meter per pixel resolution while preserving georeference data.
    Optionally delete the input file after successful processing.
    
    Args:
        input_path (str): Path to the input GeoTIFF file
        output_path (str, optional): Path for the output file. If None, will use input filename with '_1m' suffix.
        delete_input (bool): Whether to delete the input file after successful processing
    
    Returns:
        str: Path to the resampled output file
    """
    # Create output path if not provided
    if output_path is None:
        input_file = Path(input_path)
        output_path = str(input_file.with_stem(f"{input_file.stem}_1m"))
    
    try:
        # Open the dataset
        with rasterio.open(input_path) as src:
            # Get the current transform and CRS
            transform = src.transform
            crs = src.crs
            
            # Calculate scaling factors to achieve 1 meter/pixel
            # Get current resolution
            current_res_x = abs(transform.a)  # Width of a pixel in map units
            current_res_y = abs(transform.e)  # Height of a pixel in map units
            
            print(f"Current resolution: {current_res_x:.2f} x {current_res_y:.2f} meters/pixel")
            
            # Calculate scaling factors
            scale_factor_x = current_res_x / 1.0
            scale_factor_y = current_res_y / 1.0
            
            # Calculate new dimensions
            new_width = int(src.width * scale_factor_x)
            new_height = int(src.height * scale_factor_y)
            
            print(f"Scaling by factors: {scale_factor_x:.2f} x {scale_factor_y:.2f}")
            print(f"New dimensions: {new_width} x {new_height} pixels")
            
            # Calculate the new transform
            new_transform = rasterio.transform.from_bounds(
                src.bounds.left, src.bounds.bottom, 
                src.bounds.right, src.bounds.top, 
                new_width, new_height
            )
            
            # Create the resampled dataset
            profile = src.profile.copy()
            profile.update({
                'width': new_width,
                'height': new_height,
                'transform': new_transform,
            })
            
            # Read and resample data
            with rasterio.open(output_path, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    resampled = src.read(
                        i,
                        out_shape=(new_height, new_width),
                        resampling=Resampling.bilinear
                    )
                    dst.write(resampled, i)
        
        # Verify the output file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Resampled image saved to: {output_path}")
            print(f"New resolution: 1.00 x 1.00 meters/pixel")
            
            # Delete input file if requested
            if delete_input:
                os.remove(input_path)
                print(f"Input file deleted: {input_path}")
        else:
            print(f"Error: Output file {output_path} was not created properly.")
            return None
            
        return output_path
    
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return None

def verify_resolution(tif_path):
    """Verify the resolution of a GeoTIFF file"""
    with rasterio.open(tif_path) as src:
        transform = src.transform
        res_x = abs(transform.a)
        res_y = abs(transform.e)
        print(f"File: {tif_path}")
        print(f"Resolution: {res_x:.2f} x {res_y:.2f} meters/pixel")
        print(f"CRS: {src.crs}")
        print(f"Dimensions: {src.width} x {src.height} pixels")
        print(f"Bounds: {src.bounds}")

def batch_process_directory(directory, pattern="*.tif", delete_input=False):
    """
    Process all matching TIF files in a directory
    
    Args:
        directory (str): Directory containing TIF files
        pattern (str): Glob pattern to match files
        delete_input (bool): Whether to delete input files after processing
    """
    directory_path = Path(directory)
    files = list(directory_path.glob(pattern))
    
    if not files:
        print(f"No files matching '{pattern}' found in {directory}")
        return
    
    print(f"Found {len(files)} files to process")
    
    success_count = 0
    for file_path in files:
        print(f"\nProcessing: {file_path}")
        output_path = resample_to_1m_resolution(str(file_path), delete_input=delete_input)
        if output_path:
            success_count += 1
    
    print(f"\nProcessing complete: {success_count}/{len(files)} files successfully resampled")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Resample a GeoTIFF to 1 meter/pixel resolution')
    parser.add_argument('--input', help='Path to the input GeoTIFF file or directory')
    parser.add_argument('--output', '-o', help='Path for the output file (only for single file processing)')
    parser.add_argument('--verify', '-v', action='store_true', help='Verify the resolution after processing')
    parser.add_argument('--delete', '-d', action='store_true', help='Delete input files after successful processing')
    parser.add_argument('--batch', '-b', action='store_true', help='Process all TIF files in the input directory')
    parser.add_argument('--pattern', '-p', default="*.tif", help='File pattern for batch processing (default: *.tif)')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch process a directory
        batch_process_directory(args.input, args.pattern, args.delete)
    else:
        # Process a single file
        output_path = resample_to_1m_resolution(args.input, args.output, args.delete)
        
        # Verify if requested and processing was successful
        if args.verify and output_path:
            if not args.delete:  # Only verify input if it wasn't deleted
                print("\nVerifying input file:")
                verify_resolution(args.input)
            print("\nVerifying output file:")
            verify_resolution(output_path)