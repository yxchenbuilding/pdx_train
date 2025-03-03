import os
from paddlex import create_pipeline

def load_ground_truth(filepath):
    ground_truth = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            image_path, text = line.strip().split('\t')
            ground_truth[image_path] = text
    return ground_truth

def evaluate_ocr(folder_path, ground_truth):
    pipeline = create_pipeline(pipeline="OCR")
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if os.path.isfile(image_path):
            output = pipeline.predict(
                input=image_path,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            for res in output:
                ocr_text = res.text
                gt_text = ground_truth.get(image_path, "")
                print(f"Image: {image_name}")
                print(f"OCR Text: {ocr_text}")
                print(f"Ground Truth: {gt_text}")
                print(f"Match: {ocr_text == gt_text}")
                print("-" * 50)

if __name__ == "__main__":
    ground_truth_file = "./train_data/rec/rec_gt_train.txt"
    images_folder = "./train_data/rec/train"
    ground_truth = load_ground_truth(ground_truth_file)
    evaluate_ocr(images_folder, ground_truth)