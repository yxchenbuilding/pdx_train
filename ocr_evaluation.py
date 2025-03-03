import os
import json
import numpy as np
from paddlex import create_pipeline
from difflib import SequenceMatcher

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

def calculate_accuracy(pred_text, gt_text):
    """Calculate the character matching accuracy based on Levenshtein distance"""
    matcher = SequenceMatcher(None, pred_text, gt_text)
    return matcher.ratio()

def evaluate_ocr(image_folder, gt_txt, output_json="ocr_results.json"):
    """Evaluate OCR accuracy for images in the given folder and save results to JSON"""
    
    # Read the ground truth file
    gt_dict = {}
    with open(gt_txt, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')  # Assuming format: filename text (tab-separated)
            if len(parts) >= 2:
                # Strip the 'train_data/rec/train/' prefix
                filename = parts[0].strip().replace('train_data/rec/train/', '')
                gt_text = ' '.join(parts[1:]).strip()
                gt_dict[filename] = gt_text

    print(f"Ground truth file contains {len(gt_dict)} entries.")
    
    # Get image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Image folder contains {len(image_files)} images.")
    
    # Variables to track results
    results_list = []
    accuracies = []
    
    # Check if image filenames match ground truth file
    missing_gt_files = []
    
    # Process each image
    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        
        if image_name in gt_dict:
            print(f"Processing image: {image_name}")
            results = ocr_pipeline.predict(input=image_path, use_text_detection=True)
            
            if results:
                # Extract OCR text results
                pred_text = ' '.join(res.rec_texts for res in results)
                gt_text = gt_dict[image_name]
                
                # Calculate accuracy
                acc = calculate_accuracy(pred_text, gt_text)
                accuracies.append(acc)
                
                # Store results in JSON
                results_list.append({
                    "image": image_name,
                    "predicted_text": pred_text,
                    "ground_truth": gt_text,
                    "accuracy": round(acc, 4)
                })
                
                print(f"{image_name}: Accuracy = {acc:.4f}")
            else:
                print(f"No OCR results for {image_name}")
        else:
            missing_gt_files.append(image_name)  # Track missing ground truth

    if missing_gt_files:
        print("The following images have no ground truth:")
        for file in missing_gt_files:
            print(file)
    
    # Calculate average accuracy and standard deviation
    if accuracies:
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"\nAverage Accuracy: {avg_acc:.4f}")
        print(f"Standard Deviation: {std_acc:.4f}")
    else:
        avg_acc = std_acc = 0.0

    # Save results to JSON file
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump({
            "average_accuracy": round(avg_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "results": results_list
        }, json_file, ensure_ascii=False, indent=4)

    print(f"OCR results saved to {output_json}")

# Run OCR evaluation
image_folder = '/root/workspace/paddle/pdx_train/train_data/rec/train'
gt_txt = '/root/workspace/paddle/pdx_train/train_data/rec/rec_gt_train.txt'
evaluate_ocr(image_folder, gt_txt)

