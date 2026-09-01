from ultralytics import YOLO
import os

def run_training():
    """
    Fine-tunes YOLOv8n on the processed BCCD dataset using custom parameters.
    """
    # 1. Load the pre-trained base model
    model = YOLO('yolov8n.pt')

    # 2. Define data path and hyperparameters
    data_manifest = "data.yaml"
    
    if not os.path.exists(data_manifest):
        raise FileNotFoundError(f"Manifest file '{data_manifest}' not found.")

    print("Starting YOLO fine-tuning pipeline...")

    # 3. Execute training loop
    results = model.train(
        data=data_manifest,
        epochs=3,             # Number of training passes over full dataset
        imgsz=640,             # Input image resolution expected by YOLO
        batch=16,              # Samples per batch (adjust lower if out of memory)
        workers=2,             # Data loader threads
        name='bccd_v1',         # Experiment name
        project='./assets/models', # Save destination for trained weights
        exist_ok=True          # Overwrite existing experiment folder with same name
    )
    
    print(f"Training completed. Weights saved at: assets/models/bccd_v1/weights/best.pt")

if __name__ == "__main__":
    run_training()