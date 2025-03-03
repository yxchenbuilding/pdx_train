import os
import json
import numpy as np
from difflib import SequenceMatcher

def calculate_accuracy(pred_text, gt_text):
    """Calculate the character matching accuracy based on Levenshtein distance"""
    matcher = SequenceMatcher(None, pred_text, gt_text)
    return matcher.ratio()

def evaluate_ocr_results(output_folder, gt_txt):
    """Evaluate OCR accuracy based on the JSON files in output_folder and the GT txt file"""
    
    # Read the ground truth file
    gt_dict = {}
    with open(gt_txt, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')  # Assuming format: filename text (tab-separated)
            if len(parts) >= 2:
                image_path = parts[0].strip()  # Relative path
                gt_text = ' '.join(parts[1:]).strip()
                gt_dict[image_path] = gt_text

    print(f"Ground truth file contains {len(gt_dict)} entries.")
    
    # Get all JSON files from the output folder
    json_files = [f for f in os.listdir(output_folder) if f.lower().endswith('.json')]
    print(f"Output folder contains {len(json_files)} JSON files.")
    
    # Variables to track results
    accuracies = []
    results_list = []
    
    # Process each JSON file
    for json_file in json_files:
        json_path = os.path.join(output_folder, json_file)
        
        # Extract image name without '_result' suffix and file extension
        image_name = os.path.splitext(json_file)[0].replace('_result', '')
        
        # Check if corresponding GT exists
        if image_name in gt_dict:
            gt_text = gt_dict[image_name]
            
            # Load the OCR result from JSON file
            with open(json_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            
            # Extract predicted text from JSON (rec_texts)
            pred_text = ' '.join([text for text in ocr_data.get('rec_texts', [])]).strip()
            
            # Calculate accuracy
            acc = calculate_accuracy(pred_text, gt_text)
            accuracies.append(acc)
            
            # Store the evaluation results
            results_list.append({
                "image": image_name,
                "predicted_text": pred_text,
                "ground_truth": gt_text,
                "accuracy": round(acc, 4)
            })
            
            print(f"{image_name}: Accuracy = {acc:.4f}")
        else:
            print(f"No ground truth found for {image_name}")

    # Calculate average accuracy and standard deviation
    if accuracies:
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"\nAverage Accuracy: {avg_acc:.4f}")
        print(f"Standard Deviation: {std_acc:.4f}")
    else:
        avg_acc = std_acc = 0.0

    # Optionally, save the results in a summary JSON file
    summary_file = os.path.join(output_folder, "ocr_evaluation_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "average_accuracy": round(avg_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "results": results_list
        }, f, ensure_ascii=False, indent=4)

    print(f"OCR evaluation results saved to {summary_file}")

# Run OCR evaluation
output_folder = './output'  # Path to the output folder containing OCR JSON results
gt_txt = '/root/workspace/paddle/pdx_train/train_data/rec/rec_gt_train.txt'  # Path to the ground truth text file
evaluate_ocr_results(output_folder, gt_txt)
