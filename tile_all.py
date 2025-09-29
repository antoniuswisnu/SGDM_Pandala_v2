#!/usr/bin/env python3

"""Tile raster datasets into 256x256 PNGs and export georeferencing metadata."""

import argparse
import json
import os
from enum import Enum
from typing import Any, Dict, List

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window


class ImageType(Enum):
    HR = "hr"
    REF = "ref"
    SENTINEL = "sentinel"


class TileProcessor:
    """Handle tiling of a single raster dataset."""

    def __init__(self, image_type: ImageType, input_path: str, output_folder: str, tile_size: int | None) -> None:
        self.image_type = image_type
        self.input_path = input_path
        self.output_folder = output_folder
        self.tile_size = tile_size

        self.default_sizes = {
            ImageType.HR: 256,
            ImageType.REF: 256,
            ImageType.SENTINEL: 16,
        }

        if self.tile_size is None:
            self.tile_size = self.default_sizes[image_type]

        self.tile_records: List[Dict[str, Any]] = []
        self.source_transform_gdal: List[float] | None = None
        self.source_crs_wkt: str | None = None
        self.output_bands: int | None = None

    def create_tiles(self) -> bool:
        """Create tiles from the configured raster input."""
        os.makedirs(self.output_folder, exist_ok=True)

        try:
            with rasterio.open(self.input_path) as src:
                width = src.width
                height = src.height
                source_bands = src.count

                num_tiles_x = width // self.tile_size
                num_tiles_y = height // self.tile_size
                remainder_x = width % self.tile_size
                remainder_y = height % self.tile_size

                print(f"\nProcessing {self.image_type.value.upper()} image:")
                print(f"Input: {self.input_path}")
                print(f"Output: {self.output_folder}")
                print(f"Image size: {width}x{height} pixels, {source_bands} bands")
                print(f"Creating {num_tiles_x}x{num_tiles_y} tiles of size {self.tile_size}x{self.tile_size}")

                if remainder_x or remainder_y:
                    print(
                        f"Skipping tiles smaller than {self.tile_size}px "
                        f"(remainder_x={remainder_x}, remainder_y={remainder_y})"
                    )

                if num_tiles_x == 0 or num_tiles_y == 0:
                    print("Image is smaller than requested tile size; nothing to do.")
                    return False

                self.tile_records = []
                self.source_transform_gdal = list(src.transform.to_gdal())
                self.source_crs_wkt = src.crs.to_wkt() if src.crs else None
                self.output_bands = None

                for i in range(num_tiles_x):
                    for j in range(num_tiles_y):
                        self._process_single_tile(src, i, j, width, height)

                if not self.tile_records:
                    print("No tiles were generated.")
                    return False

                self._write_index_file(width, height, num_tiles_x, num_tiles_y, remainder_x, remainder_y)
                print(f"Saved {len(self.tile_records)} tiles to {self.output_folder}")
                return True

        except rasterio.errors.RasterioIOError:
            print(f"Error: Could not open {self.input_path}")
            return False

    def _process_single_tile(self, src: rasterio.io.DatasetReader, i: int, j: int, width: int, height: int) -> None:
        x_offset = i * self.tile_size
        y_offset = j * self.tile_size

        x_size = min(self.tile_size, width - x_offset)
        y_size = min(self.tile_size, height - y_offset)

        if x_size <= 0 or y_size <= 0:
            return

        if x_size != self.tile_size or y_size != self.tile_size:
            return

        window = Window(x_offset, y_offset, x_size, y_size)
        data = src.read(window=window)
        data_uint8 = self._ensure_uint8(data)
        array = np.transpose(data_uint8, (1, 2, 0))

        if array.shape[2] == 1:
            img = Image.fromarray(array[:, :, 0], mode="L")
        elif array.shape[2] == 3:
            img = Image.fromarray(array, mode="RGB")
        elif array.shape[2] >= 4:
            img = Image.fromarray(array[:, :, :4], mode="RGBA")
        else:
            img = Image.fromarray(array[:, :, :3], mode="RGB")

        bands_out = len(img.getbands())
        if self.output_bands is None:
            self.output_bands = bands_out
        elif bands_out != self.output_bands:
            raise ValueError(
                f"Tile band count mismatch: expected {self.output_bands}, got {bands_out}"
            )

        output_filename = f"tile_{i}_{j}.png"
        output_path = os.path.join(self.output_folder, output_filename)
        img.save(output_path, "PNG")

        tile_transform = src.window_transform(window)
        bounds = src.window_bounds(window)

        metadata_filename = f"tile_{i}_{j}.json"
        metadata_path = os.path.join(self.output_folder, metadata_filename)
        tile_metadata = {
            "tile_index": [i, j],
            "x_offset": int(x_offset),
            "y_offset": int(y_offset),
            "width": int(x_size),
            "height": int(y_size),
            "bounds": [float(b) for b in bounds],
            "transform_gdal": [float(v) for v in tile_transform.to_gdal()],
            "crs_wkt": self.source_crs_wkt,
            "source_transform_gdal": self.source_transform_gdal,
            "source_path": self.input_path,
            "image_type": self.image_type.value,
        }

        with open(metadata_path, "w", encoding="utf-8") as meta_file:
            json.dump(tile_metadata, meta_file, indent=2)

        tile_record = {
            "tile_index": [i, j],
            "png": output_filename,
            "metadata": metadata_filename,
            "x_offset": int(x_offset),
            "y_offset": int(y_offset),
            "width": int(x_size),
            "height": int(y_size),
            "bounds": [float(b) for b in bounds],
            "transform_gdal": [float(v) for v in tile_transform.to_gdal()],
        }
        self.tile_records.append(tile_record)

    def _ensure_uint8(self, data: np.ndarray) -> np.ndarray:
        if data.dtype == np.uint8:
            return data

        data_float = data.astype(np.float32, copy=False)
        data_min = float(data_float.min())
        data_max = float(data_float.max())

        if data_max == data_min:
            return np.zeros_like(data_float, dtype=np.uint8)

        scale = 255.0 / (data_max - data_min)
        scaled = (data_float - data_min) * scale
        clipped = np.clip(scaled, 0, 255)
        return clipped.astype(np.uint8)

    def _write_index_file(
        self,
        width: int,
        height: int,
        num_tiles_x: int,
        num_tiles_y: int,
        remainder_x: int,
        remainder_y: int,
    ) -> None:
        index_data = {
            "image_type": self.image_type.value,
            "source_path": self.input_path,
            "output_folder": self.output_folder,
            "tile_size": self.tile_size,
            "source_width": int(width),
            "source_height": int(height),
            "num_tiles_x": int(num_tiles_x),
            "num_tiles_y": int(num_tiles_y),
            "remainder_x": int(remainder_x),
            "remainder_y": int(remainder_y),
            "transform_gdal": self.source_transform_gdal,
            "crs_wkt": self.source_crs_wkt,
            "output_bands": self.output_bands,
            "output_dtype": "uint8",
            "tile_count": len(self.tile_records),
            "tiles": self.tile_records,
        }

        index_path = os.path.join(self.output_folder, "tiles_index.json")
        with open(index_path, "w", encoding="utf-8") as index_file:
            json.dump(index_data, index_file, indent=2)
        print(f"Saved tile metadata index: {index_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process and tile multiple types of satellite imagery",
    )

    parser.add_argument("--hr_input", type=str, help="Path to input HR TIF file")
    parser.add_argument("--ref_input", type=str, help="Path to input Reference TIF file")
    parser.add_argument("--sentinel_input", type=str, help="Path to input Sentinel-2 TIF file")
    parser.add_argument(
        "--output_base",
        type=str,
        default="./data_img",
        help="Base output directory for all tiles",
    )
    parser.add_argument(
        "--hr_tile_size",
        type=int,
        help="Tile size for HR image (default: 256)",
    )
    parser.add_argument(
        "--ref_tile_size",
        type=int,
        help="Tile size for Reference image (default: 256)",
    )
    parser.add_argument(
        "--sentinel_tile_size",
        type=int,
        help="Tile size for Sentinel image (default: 16)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    os.makedirs(args.output_base, exist_ok=True)

    configs = [
        (ImageType.HR, args.hr_input, "hr_tiles", args.hr_tile_size),
        (ImageType.REF, args.ref_input, "ref_tiles", args.ref_tile_size),
        (ImageType.SENTINEL, args.sentinel_input, "sentinel_tiles", args.sentinel_tile_size),
    ]

    for img_type, input_path, subfolder, tile_size in configs:
        if not input_path:
            continue

        if not os.path.isfile(input_path):
            print(f"Error: Input file {input_path} does not exist")
            continue

        output_folder = os.path.join(args.output_base, subfolder)
        processor = TileProcessor(img_type, input_path, output_folder, tile_size)

        success = processor.create_tiles()
        if success:
            print(f"{img_type.value.upper()} tiling completed successfully")
        else:
            print(f"{img_type.value.upper()} tiling failed")


if __name__ == "__main__":
    main()
