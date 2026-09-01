import os
import xml.etree.ElementTree as ET
from typing import Tuple
import shutil
import random
from pathlib import Path

# 1.- Sample estandarization 

def convert_voc_to_yolo_math(
    size: Tuple[int, int], 
    box: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    """
    Converts PASCAL VOC absolute coordinates to YOLO normalized coordinates.
    
    Args:
        size: A tuple containing (Image_Width, Image_Height).
        box: A tuple containing absolute coordinates (x_min, x_max, y_min, y_max).
        
    Returns:
        A tuple of normalized YOLO coordinates (x_center, y_center, width, height) 
        scaled between 0.0 and 1.0.
    """
    image_w, image_h = size
    x_min, x_max, y_min, y_max = box

    # Calculate absolute center coordinates
    abs_x_center = (x_min + x_max) / 2.0
    abs_y_center = (y_min + y_max) / 2.0
    
    # Calculate absolute width and height
    abs_width = x_max - x_min
    abs_height = y_max - y_min

    # Normalize by dividing by the total image dimensions
    x_center_norm = abs_x_center / image_w
    y_center_norm = abs_y_center / image_h
    width_norm = abs_width / image_w
    height_norm = abs_height / image_h

    # Return rounded to 6 decimal places to keep the txt files clean and small
    return (
        round(x_center_norm, 6),
        round(y_center_norm, 6),
        round(width_norm, 6),
        round(height_norm, 6)
    )


# 2.-  BCCD class mapping to numerical YOLO IDs


#MVP Scope: Limited to 3 cell types. Fine-grained classification planned for future releases.
# Class ranked from highest to lowest frequency 

CLASSES = {"RBC": 0, "WBC": 1, "Platelets": 2}

def parse_voc_xml_to_yolo(xml_file_path: str, output_txt_path: str) -> bool:
    """
    Reads a single PASCAL VOC XML file, extracts bounding boxes, 
    converts them to YOLO format, and writes them to a TXT file.
    
    Args:
        xml_file_path: Path to the input .xml annotation file.
        output_txt_path: Path to save the resulting .txt YOLO file.
        
    Returns:
        True if the file was processed successfully, False otherwise.
    """
    try:
        # 1. Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # 2. Extract total image dimensions
        size_node = root.find('size')
        if size_node is None:
            return False
            
        img_width = int(size_node.find('width').text)
        img_height = int(size_node.find('height').text)
        
        # 3. Open the output TXT file to write the YOLO annotations
        with open(output_txt_path, 'w') as out_file:
            # 4. Iterate over each detected object (cell) in the XML
            for obj in root.iter('object'):
                difficult = obj.find('difficult')
                if difficult is not None and int(difficult.text) == 1:
                    continue # Skip difficult/blurry objects to avoid confusing the model
                    
                class_name = obj.find('name').text
                # Only process classes we defined in our dictionary
                if class_name not in CLASSES:
                    continue
                    
                class_id = CLASSES[class_name]
                
                # 5. Extract VOC coordinates
                xml_box = obj.find('bndbox')
                x_min = float(xml_box.find('xmin').text)
                x_max = float(xml_box.find('xmax').text)
                y_min = float(xml_box.find('ymin').text)
                y_max = float(xml_box.find('ymax').text)
                
                # Setup tuple for the math function
                box_voc = (x_min, x_max, y_min, y_max)
                size_img = (img_width, img_height)
                
                # 6. Call our math function to get normalized YOLO coordinates
                yolo_coords = convert_voc_to_yolo_math(size_img, box_voc)
                
                # 7. Write to file in YOLO format: <class_id> <x_center> <y_center> <width> <height>
                # Example: 0 0.523 0.441 0.12 0.09
                out_file.write(f"{class_id} {yolo_coords[0]} {yolo_coords[1]} {yolo_coords[2]} {yolo_coords[3]}\n")
                
        return True
    
    except Exception as e:
        print(f"Error processing {xml_file_path}: {e}")
        return False

# 3.- Principal Loop

def split_and_prepare_dataset(
    source_images_dir: str, 
    source_annotations_dir: str, 
    output_base_dir: str, 
    val_split: float = 0.2
):
    """
    Orchestrates the conversion process. Reads raw BCCD files, converts labels to YOLO format,
    and splits the data into train and validation sets.
    """
    # 1. Define the exact folder structure YOLO requires
    paths = {
        "images_train": Path(output_base_dir) / "images" / "train",
        "images_val": Path(output_base_dir) / "images" / "val",
        "labels_train": Path(output_base_dir) / "labels" / "train",
        "labels_val": Path(output_base_dir) / "labels" / "val"
    }

    # Create the directories if they don't exist
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # 2. Get all available XML annotations
    xml_files = [f for f in os.listdir(source_annotations_dir) if f.endswith('.xml')]
    
    # 3. Randomly shuffle to ensure unbiased splitting
    random.seed(42) # Fixed seed for reproducible results
    random.shuffle(xml_files)
    
    # Calculate how many files go to validation
    num_val = int(len(xml_files) * val_split)
    
    # Slice the list into train and val sets
    val_files = xml_files[:num_val]
    train_files = xml_files[num_val:]
    
    print(f"Total samples: {len(xml_files)}")
    print(f"Allocating {len(train_files)} to Training and {len(val_files)} to Validation.")

    # 4. The main processing loop
    def process_set(file_list: list, img_dest_dir: Path, lbl_dest_dir: Path):
        successful_conversions = 0
        for xml_file in file_list:
            # Construct full paths
            base_name = xml_file.replace('.xml', '')
            source_xml_path = os.path.join(source_annotations_dir, xml_file)
            source_img_path = os.path.join(source_images_dir, f"{base_name}.jpg")
            
            dest_txt_path = lbl_dest_dir / f"{base_name}.txt"
            dest_img_path = img_dest_dir / f"{base_name}.jpg"
            
            # Check if the corresponding image exists before processing
            if not os.path.exists(source_img_path):
                print(f"Warning: Image missing for {xml_file}. Skipping.")
                continue
                
            # Process the XML. If successful, copy the image to the new YOLO structure
            if parse_voc_xml_to_yolo(source_xml_path, str(dest_txt_path)):
                shutil.copy2(source_img_path, dest_img_path)
                successful_conversions += 1
                
        return successful_conversions

    print("\nProcessing Training Set...")
    train_count = process_set(train_files, paths["images_train"], paths["labels_train"])
    
    print("Processing Validation Set...")
    val_count = process_set(val_files, paths["images_val"], paths["labels_val"])
    
    print(f"\nPreparation Complete! Successfully prepared {train_count} train samples and {val_count} val samples.")

#  ==== EXECUTION BLOCK =====   

# Execute the script ONLY if run directly from the terminal
if __name__ == "__main__":
    # Relative paths pointing to the folder extracted by setup_offline.py
    RAW_IMAGES = "datasets/bccd_sample/BCCD_Dataset-master/BCCD/JPEGImages"
    RAW_ANNOTATIONS = "datasets/bccd_sample/BCCD_Dataset-master/BCCD/Annotations"
    OUTPUT_YOLO = "datasets/bccd_yolo"

    print("Starting BCCD to YOLO dataset conversion pipeline...")
    split_and_prepare_dataset(
        source_images_dir=RAW_IMAGES,
        source_annotations_dir=RAW_ANNOTATIONS,
        output_base_dir=OUTPUT_YOLO,
        val_split=0.2
    )