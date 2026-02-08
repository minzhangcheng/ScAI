"""
Image processing utilities for pathology images
"""
import base64
import io
import math
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np


# Set PIL decompression bomb limit at module level to protect all Image.open() calls
Image.MAX_IMAGE_PIXELS = 200_000_000  # ~200 megapixels


def load_image(image_path: str) -> Image.Image:
    """Load an image from file path with size safety check"""
    return Image.open(image_path)


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_image(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data))


def is_blank_tile(image: Image.Image, threshold: float = 0.95) -> bool:
    """
    Check if a tile is mostly blank/white
    
    Args:
        image: PIL Image to check
        threshold: Ratio of white pixels to consider blank
        
    Returns:
        True if the tile is blank
    """
    # Convert to grayscale
    gray = image.convert('L')
    arr = np.array(gray)
    
    # Consider pixels > 240 as white
    white_pixels = np.sum(arr > 240)
    total_pixels = arr.size
    
    return (white_pixels / total_pixels) > threshold


def split_image_to_tiles(
    image: Image.Image,
    tile_size: int = 1024,
    overlap: int = 64,
    blank_threshold: float = 0.95
) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
    """
    Split a large image into tiles, removing blank tiles
    
    Args:
        image: Large PIL Image to split
        tile_size: Size of each tile (square)
        overlap: Overlap between adjacent tiles
        blank_threshold: Threshold for blank detection
        
    Returns:
        List of (tile_image, (x1, y1, x2, y2)) tuples
    """
    width, height = image.size
    tiles = []
    
    step = tile_size - overlap
    
    for y in range(0, height, step):
        for x in range(0, width, step):
            # Calculate tile boundaries
            x1 = x
            y1 = y
            x2 = min(x + tile_size, width)
            y2 = min(y + tile_size, height)
            
            # Extract tile
            tile = image.crop((x1, y1, x2, y2))
            
            # Skip blank tiles
            if not is_blank_tile(tile, blank_threshold):
                tiles.append((tile, (x1, y1, x2, y2)))
    
    return tiles


def process_pathology_image(
    image_path: str,
    tile_size: int = 1024,
    overlap: int = 64,
    blank_threshold: float = 0.95,
    max_tiles: int = 50
) -> List[str]:
    """
    Process a pathology image for vision model analysis
    
    Args:
        image_path: Path to the pathology image
        tile_size: Size of each tile
        overlap: Overlap between tiles
        blank_threshold: Threshold for blank detection
        max_tiles: Maximum number of tiles to return
        
    Returns:
        List of base64 encoded tile images
    """
    # Load image
    image = load_image(image_path)
    
    # Get image size
    width, height = image.size
    
    # If image is small enough, return as-is
    if width <= tile_size and height <= tile_size:
        return [image_to_base64(image)]
    
    # Split into tiles
    tiles = split_image_to_tiles(
        image,
        tile_size=tile_size,
        overlap=overlap,
        blank_threshold=blank_threshold
    )
    
    # If too many tiles, sample evenly
    if len(tiles) > max_tiles:
        indices = np.linspace(0, len(tiles) - 1, max_tiles, dtype=int)
        tiles = [tiles[i] for i in indices]
    
    # Convert to base64
    return [image_to_base64(tile) for tile, _ in tiles]


def create_overview_image(
    image: Image.Image,
    max_size: int = 1024
) -> str:
    """
    Create a downscaled overview image for context
    
    Args:
        image: Original image
        max_size: Maximum dimension
        
    Returns:
        Base64 encoded overview image
    """
    width, height = image.size
    
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image_to_base64(resized)


def annotate_tile_positions(
    original_size: Tuple[int, int],
    tiles: List[Tuple[Image.Image, Tuple[int, int, int, int]]]
) -> str:
    """
    Create a description of tile positions for context
    
    Args:
        original_size: (width, height) of original image
        tiles: List of tiles with their positions
        
    Returns:
        Description string
    """
    width, height = original_size
    
    descriptions = [f"Original image size: {width}x{height} pixels"]
    descriptions.append(f"Total tiles: {len(tiles)}")
    
    for i, (_, (x1, y1, x2, y2)) in enumerate(tiles):
        # Calculate relative position
        rel_x = (x1 + x2) / 2 / width
        rel_y = (y1 + y2) / 2 / height
        
        position = ""
        if rel_y < 0.33:
            position = "top"
        elif rel_y < 0.67:
            position = "middle"
        else:
            position = "bottom"

        if rel_x < 0.33:
            position = "left-" + position
        elif rel_x < 0.67:
            position = "center-" + position
        else:
            position = "right-" + position

        descriptions.append(f"Tile {i+1}: {position} (position: {x1},{y1} to {x2},{y2})")
    
    return "\n".join(descriptions)
