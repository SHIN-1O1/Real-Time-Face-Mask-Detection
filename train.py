"""
Train YOLOv8 model for Face Mask Detection.
Uses transfer learning from pretrained weights.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(epochs=50, batch_size=4, img_size=320, device="0", run_name="face_mask"):
    """Train YOLOv8 model on face mask dataset."""
    
    print("=" * 60)
    print("Face Mask Detection - YOLOv8 Training")
    print("=" * 60)
    
    # Load pretrained model
    print("\nLoading pretrained YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    # Training configuration
    data_yaml = Path(__file__).parent / "data.yaml"
    
    print(f"\nTraining Configuration:")
    print(f"  Dataset: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Device: {device}")
    
    # Train the model
    print("\nStarting training...\n")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project="runs/detect",
        name=run_name,
        exist_ok=True,
        pretrained=True,
        optimizer="Adam",
        lr0=0.001,
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        workers=2,  # Reduced workers for stability
        cache=False  # Disable caching to save memory
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest model saved at: runs/detect/face_mask/weights/best.pt")
    print(f"Training metrics saved at: runs/detect/face_mask/")
    
    # Validate the model
    print("\nRunning validation...")
    metrics = model.val()
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Face Mask Detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=2, help="Batch size (use 2 for low VRAM GPUs)")
    parser.add_argument("--img-size", type=int, default=320, help="Image size (use 320 for low VRAM)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device (0, 1, etc.) or 'cpu'")
    parser.add_argument("--name", type=str, default="face_mask", help="Run name for saving results")
    parser.add_argument("--validate", action="store_true", help="Run quick validation only")
    
    args = parser.parse_args()
    
    if args.validate:
        # Quick validation run
        train(epochs=5, batch_size=8, img_size=640, device=args.device, run_name="validate")
    else:
        train(
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            device=args.device,
            run_name=args.name
        )


if __name__ == "__main__":
    main()
