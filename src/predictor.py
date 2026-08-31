import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Any
import logging

# 1. Setup module-level logger (Crucial for production monitoring)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BloodCellDetector:
    """
    Core inference engine for object detection.
    Kept strictly independent from any UI framework (Streamlit) to ensure scalability.
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """
        Initializes the YOLO model. 
        Loads weights into memory (RAM/VRAM) upon instantiation.
        """
        logger.info(f"Loading YOLO model from {model_path}...")
        try:
            # For the MVP, if the file doesn't exist locally, Ultralytics downloads the base model.
            self.model = YOLO(model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load the inference model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def analyze_sample(self, image: np.ndarray, conf_threshold: float = 0.25) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Performs inference on a preprocessed image array.
        
        Args:
            image: RGB numpy array from the preprocessing layer.
            conf_threshold: Minimum confidence score to consider a valid detection.
            
        Returns:
            Tuple containing:
            - Annotated image (numpy array with drawn bounding boxes)
            - List of dictionaries containing detection metadata for the Analytics layer.
        """
        logger.info("Running inference on the clinical sample...")
        
        # Run YOLO prediction. verbose=False keeps our terminal clean from YOLO spam.
        results = self.model.predict(source=image, conf=conf_threshold, verbose=False)
        
        # We only pass one image at a time, so we take the first result object
        result = results[0]
        
        # 1. VISUAL OUTPUT: Get the annotated image (YOLO draws boxes automatically)
        # YOLO's .plot() returns BGR. We convert to RGB for standard UI rendering.
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[..., ::-1] # Numpy slicing trick to reverse BGR to RGB
        
        # 2. DATA OUTPUT: Extract metadata for our Analytics Layer (Pandas)
        detections = []
        for box in result.boxes:
            detection = {
                "class_id": int(box.cls[0]),
                "class_name": self.model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max] format
            }
            detections.append(detection)
            
        logger.info(f"Detection complete. Found {len(detections)} objects.")
        
        return annotated_rgb, detections