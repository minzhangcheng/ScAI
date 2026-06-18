"""
Image processing utilities for pathology images
"""
import base64
import io
import logging
import math
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from PIL import Image
import numpy as np

_logger = logging.getLogger(__name__)


# Set PIL decompression bomb limit at module level to protect all Image.open() calls
Image.MAX_IMAGE_PIXELS = 200_000_000  # ~200 megapixels


def compress_image_to_base64(image_path: str, max_width: int = 800, quality: int = 85) -> str:
    """
    Compress an image and convert to base64 for HTML embedding.

    Args:
        image_path: Path to the image file
        max_width: Maximum width for resizing (preserves aspect ratio)
        quality: JPEG quality (1-100)

    Returns:
        Base64 encoded string with data URI prefix, or empty string on failure
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (for PNG with transparency)
            if img.mode in ('RGBA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize if too large
            width, height = img.size
            if width > max_width:
                ratio = max_width / width
                new_size = (max_width, int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Save to buffer as JPEG
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)

            b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_data}"
    except Exception as e:
        _logger.exception(f"Error compressing image {image_path}")
        return ""


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
    max_tiles: int = 50,
    preloaded_image: "Image.Image | None" = None
) -> List[str]:
    """
    Process a pathology image for vision model analysis

    Args:
        image_path: Path to the pathology image
        tile_size: Size of each tile
        overlap: Overlap between tiles
        blank_threshold: Threshold for blank detection
        max_tiles: Maximum number of tiles to return
        preloaded_image: Optional pre-loaded PIL Image to avoid redundant disk reads

    Returns:
        List of base64 encoded tile images
    """
    # Load image (reuse preloaded if provided)
    image = preloaded_image if preloaded_image is not None else load_image(image_path)
    
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


def categorize_figures(figures_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Scan figures directory and categorize images by their content type.

    Returns a dictionary mapping category names to lists of figure info.
    """
    categories = {
        "qc": [],           # Quality control figures
        "umap": [],         # UMAP visualizations
        "clustering": [],   # Clustering related
        "markers": [],      # Marker gene analysis
        "de": [],           # Differential expression
        "heatmap": [],      # Heatmaps
        "violin": [],       # Violin plots
        "other": []         # Other figures
    }

    if not figures_dir.exists():
        return categories

    # Scan for both JPG and PNG (JPG preferred for new sessions, PNG for backward compatibility)
    seen_stems = set()
    image_files = sorted(figures_dir.glob("*.jpg")) + sorted(figures_dir.glob("*.png"))
    for img_file in image_files:
        if img_file.stem in seen_stems:
            continue  # Skip PNG if JPG already found for same stem
        seen_stems.add(img_file.stem)
        filename = img_file.stem.lower()
        pdf_file = img_file.with_suffix('.pdf')

        fig_info = {
            "name": img_file.stem,
            "png_path": str(img_file),  # Field name kept for compatibility; may be JPG
            "pdf_path": str(pdf_file) if pdf_file.exists() else None,
            "filename": img_file.name
        }

        # Categorize based on filename
        if "qc" in filename or "quality" in filename:
            categories["qc"].append(fig_info)
        elif "celltype" in filename or "cell_type" in filename or "annotation" in filename:
            categories["umap"].append(fig_info)  # Cell type annotation UMAPs grouped with UMAP
        elif "umap" in filename:
            categories["umap"].append(fig_info)
        elif "cluster" in filename or "leiden" in filename:
            categories["clustering"].append(fig_info)
        elif "marker" in filename:
            categories["markers"].append(fig_info)
        elif "volcano" in filename or "_de_" in filename or "differential" in filename:
            categories["de"].append(fig_info)
        elif "heatmap" in filename:
            categories["heatmap"].append(fig_info)
        elif "violin" in filename:
            categories["violin"].append(fig_info)
        elif "dotplot" in filename:
            categories["markers"].append(fig_info)
        elif "stacked_bar" in filename or "composition" in filename or "fold_change" in filename:
            categories["clustering"].append(fig_info)
        elif "feature" in filename or "expression" in filename:
            categories["de"].append(fig_info)
        else:
            categories["other"].append(fig_info)

    return categories


def save_graphic_abstract_image(image_base64: str, figures_dir) -> Optional[str]:
    """Decode a base64 graphic-abstract image, flatten RGBA/palette modes onto a
    white background, and save it as ``figures_dir/graphic_abstract.jpg`` (JPEG,
    quality 90). Returns the output path, or None when no image data is given.

    Shared by the live supervisor report path and services/report/generator so
    both produce identical, valid JPEGs (see memory: two-report-generators)."""
    if not image_base64:
        return None
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "graphic_abstract.jpg"
    img = Image.open(io.BytesIO(base64.b64decode(image_base64)))
    if img.mode in ("RGBA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = bg
    img.save(out_path, "JPEG", quality=90, optimize=True)
    return str(out_path)
