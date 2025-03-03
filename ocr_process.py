import os
from paddlex import create_pipeline

# Initialize OCR pipeline
try:
    ocr_pipeline = create_pipeline(
        pipeline="OCR",
        device="gpu:0",  # Ensure that you have GPU support or change it to "cpu"
        use_hpip=False
    )
    print("OCR pipeline initialized successfully.")
except Exception as e:
    print(f"Failed to initialize OCR pipeline: {e}")
    exit()

def process_images(image_folder, output_folder):
    """Process all images in the folder and save OCR results to output folder"""
    
    # Get image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Image folder contains {len(image_files)} images.")
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing image: {image_path}")
        
        # Perform OCR prediction
        output = ocr_pipeline.predict(
            input=image_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        
        # Process and save the OCR results
        for res in output:
            res.print()  # Print OCR result to console
            res.save_to_img(os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_result.png"))  # Save result as image
            res.save_to_json(os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_result.json"))  # Save result as JSON
    
    print(f"OCR results saved to {output_folder}")

# Run OCR prediction on all images in the folder
image_folder = '/root/workspace/paddle/pdx_train/train_data/rec/train'  # Update this path
output_folder = './output'  # Folder to save the OCR results (images and JSON files)
process_images(image_folder, output_folder)
