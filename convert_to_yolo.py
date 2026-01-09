"""
Convert Pascal VOC XML annotations to YOLO format for Face Mask Detection.
Splits data into train/val/test sets (80/15/5).
"""

import os
import xml.etree.ElementTree as ET
import shutil
import random
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "archive" / "images"
ANNOTATIONS_DIR = BASE_DIR / "archive" / "annotations"
OUTPUT_DIR = BASE_DIR / "dataset"

# Class mapping
CLASS_MAPPING = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.15
TEST_RATIO = 0.05


def parse_voc_annotation(xml_path):
    """Parse Pascal VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    
    # Get filename
    filename = root.find("filename").text
    
    # Parse objects
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_MAPPING:
            continue
            
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        
        objects.append({
            "class": name,
            "class_id": CLASS_MAPPING[name],
            "bbox": (xmin, ymin, xmax, ymax)
        })
    
    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects
    }


def convert_to_yolo(annotation, width, height):
    """Convert bounding box to YOLO format (normalized center x, y, w, h)."""
    xmin, ymin, xmax, ymax = annotation["bbox"]
    
    # Calculate center and dimensions
    x_center = (xmin + xmax) / 2.0 / width
    y_center = (ymin + ymax) / 2.0 / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    
    return annotation["class_id"], x_center, y_center, box_width, box_height


def create_yolo_label(parsed_annotation):
    """Create YOLO format label string."""
    width = parsed_annotation["width"]
    height = parsed_annotation["height"]
    
    lines = []
    for obj in parsed_annotation["objects"]:
        class_id, x, y, w, h = convert_to_yolo(obj, width, height)
        lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    
    return "\n".join(lines)


def setup_directories():
    """Create YOLO dataset directory structure."""
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    print(f"Created directory structure at {OUTPUT_DIR}")


def main():
    print("=" * 60)
    print("Face Mask Detection - VOC to YOLO Converter")
    print("=" * 60)
    
    # Setup directories
    setup_directories()
    
    # Get all annotation files
    xml_files = list(ANNOTATIONS_DIR.glob("*.xml"))
    print(f"\nFound {len(xml_files)} annotation files")
    
    # Shuffle and split
    random.seed(42)  # For reproducibility
    random.shuffle(xml_files)
    
    n_total = len(xml_files)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    splits = {
        "train": xml_files[:n_train],
        "val": xml_files[n_train:n_train + n_val],
        "test": xml_files[n_train + n_val:]
    }
    
    print(f"\nSplit distribution:")
    for split, files in splits.items():
        print(f"  {split}: {len(files)} images")
    
    # Process each split
    stats = {"total_objects": 0, "classes": {name: 0 for name in CLASS_MAPPING}}
    
    for split_name, files in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        for xml_path in files:
            # Parse annotation
            annotation = parse_voc_annotation(xml_path)
            
            # Get corresponding image
            image_name = annotation["filename"]
            image_path = IMAGES_DIR / image_name
            
            if not image_path.exists():
                print(f"  Warning: Image not found: {image_path}")
                continue
            
            # Create YOLO label
            yolo_label = create_yolo_label(annotation)
            
            # Copy image
            dest_image = OUTPUT_DIR / split_name / "images" / image_name
            shutil.copy2(image_path, dest_image)
            
            # Write label
            label_name = image_name.rsplit(".", 1)[0] + ".txt"
            dest_label = OUTPUT_DIR / split_name / "labels" / label_name
            with open(dest_label, "w") as f:
                f.write(yolo_label)
            
            # Update stats
            for obj in annotation["objects"]:
                stats["total_objects"] += 1
                stats["classes"][obj["class"]] += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"\nTotal objects: {stats['total_objects']}")
    print("\nClass distribution:")
    for class_name, count in stats["classes"].items():
        print(f"  {class_name}: {count}")
    
    print(f"\nDataset saved to: {OUTPUT_DIR}")
    print("\nNext step: Run 'python train.py' to train the model")


if __name__ == "__main__":
    main()
