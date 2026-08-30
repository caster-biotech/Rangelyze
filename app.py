import streamlit as st
import cv2
import os

# Import our custom preprocessing module
# Note: Streamlit runs from the root directory, so 'src' is accessible
from src.preprocessing import load_and_preprocess

def main():
    # 1. Page Configuration (Must be the first Streamlit command)
    st.set_page_config(
        page_title="Rangelyze | Clinical WBC Counter",
        page_icon="🔬",
        layout="wide"
    )

    # 2. Dashboard Header
    st.title("🔬 Rangelyze: White Blood Cell Counter")
    st.markdown("### MVP - Blood Smear Preprocessing Validation")
    st.markdown("---")

    # 3. Define the path to our local sample
    sample_path = "datasets/bccd_sample/BloodImage_00000.jpg"

    # 4. Error Handling: Verify file existence before processing
    if not os.path.exists(sample_path):
        st.error(f"Sample not found at path: {sample_path}. Please check your datasets folder.")
        return  # Stop execution if there is no data

    try:
        # Load the original image (OpenCV loads in BGR, we need RGB for the UI)
        original_bgr = cv2.imread(sample_path)
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        # Load the preprocessed image using our Capa 1 pipeline
        enhanced_image = load_and_preprocess(sample_path)

        # 5. UI Layout: Side-by-side comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Smear")
            # use_container_width dynamically adjusts the image to the column size
            st.image(original_rgb, caption="Raw Image (RGB)", use_container_width=True)

        with col2:
            st.subheader("Enhanced Smear (CLAHE)")
            st.image(enhanced_image, caption="Preprocessed Image for YOLO Inference", use_container_width=True)

    except Exception as e:
        # Catch and display any clinical software error gracefully
        st.error(f"An error occurred during sample processing: {str(e)}")

if __name__ == "__main__":
    main()