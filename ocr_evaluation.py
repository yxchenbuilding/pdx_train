import os
import json
import numpy as np
from paddlex import create_pipeline
from difflib import SequenceMatcher

# 初始化OCR产线
ocr_pipeline = create_pipeline(
    pipeline="OCR",
    device="gpu:0", 
    use_hpip=False
)

def calculate_accuracy(pred_text, gt_text):
    """计算Levenshtein距离的字符匹配精度"""
    matcher = SequenceMatcher(None, pred_text, gt_text)
    return matcher.ratio()

def evaluate_ocr(image_folder, gt_txt, output_json="ocr_results.json"):
    """评估OCR模型在指定文件夹下的识别精度，并保存结果为JSON"""
    # 读取ground truth
    gt_dict = {}
    with open(gt_txt, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('	')  # 假设格式：filename text（空格分隔）
            if len(parts) >= 2:
                filename = parts[0].strip()
                gt_text = ' '.join(parts[1:]).strip()
                gt_dict[filename] = gt_text

    # 打印读取的GT文件中包含多少条ground truth信息
    print(f"Ground truth file contains {len(gt_dict)} entries.")
    
    # 统计文件夹中的图片数量
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    print(f"Image folder contains {len(image_files)} images.")

    # 记录找到的图像和GT配对的数量
    found_images = 0
    found_gt = 0
    results_list = []
    accuracies = []

    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        if image_name in gt_dict:
            found_images += 1
            found_gt += 1
            results = ocr_pipeline.predict(input=image_path, use_doc_orientation_classify=False)
            
            # 解析 Result 对象获取预测文本
            pred_text = ' '.join(res.rec_texts for res in results)
            acc = calculate_accuracy(pred_text, gt_dict[image_name])
            accuracies.append(acc)

            # 存入JSON的结果
            results_list.append({
                "image": image_name,
                "predicted_text": pred_text,
                "ground_truth": gt_dict[image_name],
                "accuracy": round(acc, 4)
            })

            print(f"{image_name}: Accuracy = {acc:.4f}")

    # 计算平均值和标准差
    avg_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"\nAverage Accuracy: {avg_acc:.4f}")
    print(f"Standard Deviation: {std_acc:.4f}")

    # 将所有结果保存到 JSON 文件
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump({
            "average_accuracy": round(avg_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "results": results_list
        }, json_file, ensure_ascii=False, indent=4)

    print(f"\nOCR results saved to {output_json}")
    print(f"\nFound {found_images} images and {found_gt} ground truth entries.")

# 设置路径
evaluate_ocr('/root/workspace/paddle/data/train_data/rec/train', '/root/workspace/paddle/data/train_data/rec/rec_gt_train.txt')
