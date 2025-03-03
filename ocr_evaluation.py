import os
import re
from paddlex import create_pipeline

def load_ground_truth(filepath):
    """Load ground truth text from annotation file"""
    ground_truth = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                image_path, text = line.strip().split('\t')
                ground_truth[image_path] = text
        return ground_truth
    except Exception as e:
        print(f"Error loading ground truth: {str(e)}")
        return {}

def normalize_text(text):
    """Normalize text for better comparison"""
    return re.sub(r'\s+', '', text).lower()

def evaluate_ocr(folder_path, ground_truth):
    """Evaluate OCR performance against ground truth"""
    pipeline = create_pipeline(pipeline="OCR")
    total = 0
    correct = 0
    
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        
        if not os.path.isfile(image_path):
            continue
            
        try:
            output = pipeline.predict(
                input=image_path,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            
            for res in output:
                total += 1
                ocr_text = res.transcription  # Updated attribute
                confidence = res.score
                gt_text = ground_truth.get(image_path, "")
                
                # Normalize both texts
                ocr_norm = normalize_text(ocr_text)
                gt_norm = normalize_text(gt_text)
                match = ocr_norm == gt_norm
                
                if match:
                    correct += 1
                
                # Detailed output
                print(f"\nImage: {image_name}")
                print(f"OCR Raw: {ocr_text}")
                print(f"OCR Normalized: {ocr_norm}")
                print(f"Ground Truth Raw: {gt_text}")
                print(f"Ground Truth Normalized: {gt_norm}")
                print(f"Confidence: {confidence:.2f}")
                print(f"Match: {match}")
                print("-" * 60)
                
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
    
    # Calculate accuracy
    if total > 0:
        accuracy = correct / total
        print(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{total})")
    else:
        print("No images processed")

if __name__ == "__main__":
    ground_truth_file = "./train_data/rec/rec_gt_train.txt"
    images_folder = "./train_data/rec/train"
    
    ground_truth = load_ground_truth(ground_truth_file)
    if ground_truth:
        evaluate_ocr(images_folder, ground_truth)
    else:
        print("Failed to load ground truth data")
