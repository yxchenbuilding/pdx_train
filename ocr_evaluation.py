import os
import json
import numpy as np
import cv2
from paddleocr import PaddleOCR
from difflib import SequenceMatcher

# 初始化OCR模型
ocr = PaddleOCR(rec_model_dir='/root/workspace/paddle/inference/P-OCRv4_server_rec', use_angle_cls=True, use_gpu=True)

def calculate_accuracy(pred_text, gt_text):
    """计算Levenshtein距离的字符匹配精度"""
    matcher = SequenceMatcher(None, pred_text, gt_text)
    return matcher.ratio()

def evaluate_ocr(image_folder, gt_txt, output_json="ocr_results.json"):
    """评估OCR模型在指定文件夹下的识别精度，并保存结果为JSON"""
    
    print("\n========== [1] 读取 Ground Truth ========== ")
    gt_dict = {}
    try:
        with open(gt_txt, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('	')  # 假设格式：filename text（空格分隔）
                if len(parts) >= 2:
                    filename = parts[0].strip()
                    gt_text = ' '.join(parts[1:]).strip()
                    gt_dict[filename] = gt_text
        print(f"成功读取 Ground Truth，共 {len(gt_dict)} 条记录")
    except Exception as e:
        print(f"错误：无法读取 ground truth 文件 - {str(e)}")
        return

    print("\n========== [2] 检查图像文件夹 ========== ")
    if not os.path.exists(image_folder):
        print(f"错误：文件夹 {image_folder} 不存在！")
        return
    
    image_files = os.listdir(image_folder)
    if not image_files:
        print("错误：图片文件夹为空！")
        return
    print(f"发现 {len(image_files)} 张图片")

    results_list = []
    accuracies = []

    print("\n========== [3] 开始OCR识别 ========== ")
    images_found = False  # 标志是否有有效图片

    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        
        print(f"\n>>> 处理图片: {image_name}")
        
        # 检查图片是否可以加载
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告：无法载入 {image_name}，可能是无效图像或路径错误")
            continue
        print("✅ 图片载入成功")

        # 检查 ground truth 是否存在
        if image_name not in gt_dict:
            print(f"⚠️ 警告：{image_name} 没有对应的ground truth，跳过！")
            continue
        
        images_found = True
        gt_text = gt_dict[image_name]
        print(f"Ground Truth: {gt_text}")

        # 运行 OCR 识别
        result = ocr.ocr(image_path, cls=True)
        
        if not result or not result[0]:
            print(f"⚠️ 警告：{image_name} OCR 结果为空")
            pred_text = ""
        else:
            pred_text = ''.join([res[1][0] for res in result[0]])
        
        print(f"OCR 识别结果: {pred_text}")

        acc = calculate_accuracy(pred_text, gt_text)
        accuracies.append(acc)

        # 存入 JSON 结果
        results_list.append({
            "image": image_name,
            "predicted_text": pred_text,
            "ground_truth": gt_text,
            "accuracy": round(acc, 4)
        })

        print(f"✅ 计算精度: {acc:.4f}")

    print("\n========== [4] 计算最终准确率 ========== ")
    if accuracies:  # 确保非空
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
    else:
        avg_acc = 0
        std_acc = 0
        print("⚠️ 警告：没有可用的 OCR 结果，平均准确率和标准差设为 0")

    print(f"\n📊 平均准确率: {avg_acc:.4f}")
    print(f"📊 标准差: {std_acc:.4f}")

    # 保存 JSON 结果
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump({
            "average_accuracy": round(avg_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "results": results_list
        }, json_file, ensure_ascii=False, indent=4)

    print(f"\n✅ OCR 结果已保存到 {output_json}")

# 设置路径
evaluate_ocr('/root/workspace/paddle/data/train_data/rec/train', '/root/workspace/paddle/data/train_data/rec/rec_gt_train.txt')
