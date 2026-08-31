import streamlit as st
import cv2
import os

# Import our custom modules (Clean Architecture approach)
from src.preprocessing import load_and_preprocess
from src.predictor import BloodCellDetector

# 1. THE CACHE DECORATOR (Crucial for performance)
# @st.cache_resource is used for global resources like ML models or Database connections
@st.cache_resource
def get_inference_engine() -> BloodCellDetector:
    """
    Initializes the YOLO model only once per session.
    Prevents memory leaks and slow re-renders.
    """
    # For MVP, using the base YOLOv8 nano model
    return BloodCellDetector(model_path='yolov8n.pt')

def main():
    st.set_page_config(
        page_title="Rangelyze | Clinical WBC Counter",
        page_icon="🔬",
        layout="wide"
    )

    st.title("🔬 Rangelyze: White Blood Cell Counter")
    st.markdown("### MVP - Inference Layer Validation")
    st.markdown("---")

    # 2. INSTANTIATE THE ENGINE
    # The first time this runs, it takes a few seconds. 
    # Subsequent interactions will be instantaneous.
    detector = get_inference_engine()

    sample_path = "datasets/bccd_sample/BloodImage_00000.jpg"

    if not os.path.exists(sample_path):
        st.error(f"Sample not found at path: {sample_path}.")
        return

    try:
        # Load and preprocess (Layer 1)
        original_bgr = cv2.imread(sample_path)
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        enhanced_image = load_and_preprocess(sample_path)

        # 3. RUN INFERENCE (Layer 2)
        # Using a slider to let the user adjust the confidence threshold dynamically
        st.sidebar.header("Microscope Settings")
        conf_threshold = st.sidebar.slider(
            "Confidence Threshold", 
            min_value=0.1, max_value=1.0, value=0.25, step=0.05,
            help="Filters out weak detections. Higher = Stricter."
        )
        
        # Execute the prediction on the preprocessed image
        annotated_image, metadata = detector.analyze_sample(
            image=enhanced_image, 
            conf_threshold=conf_threshold
        )

        # 4. UI Layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. Original Smear")
            st.image(original_rgb, use_container_width=True)

        with col2:
            st.subheader(f"2. YOLO Detections (Found: {len(metadata)})")
            st.image(annotated_image, use_container_width=True)

        # 5. DISPLAY RAW METADATA (For the upcoming Layer 4)
        st.markdown("### Detection Metadata (Raw JSON)")
        st.write(metadata)

    except Exception as e:
        st.error(f"System Error: {str(e)}")

if __name__ == "__main__":
    main()