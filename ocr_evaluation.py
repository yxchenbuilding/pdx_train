import os
import cv2
import pandas as pd
from paddleocr import PaddleOCR
from sklearn.metrics import precision_score, recall_score
from leven import levenshtein

class OCRValidator:
    def __init__(self, image_dir, gt_path, output_file):
        self.image_dir = image_dir
        self.gt_path = gt_path
        self.output_file = output_file
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', 
                           det_model_dir='en_PP-OCRv3_det_infer',
                           rec_model_dir='en_PP-OCRv3_rec_infer')
        
        self.results = []

    def load_ground_truth(self):
        """Load ground truth from text file"""
        self.gt_mapping = {}
        with open(self.gt_path, 'r') as f:
            for line in f:
                img_name, gt_text = line.strip().split('\t')
                self.gt_mapping[img_name] = gt_text

    def process_images(self):
        """Process all images in directory"""
        for img_file in os.listdir(self.image_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.process_single_image(img_file)

    def process_single_image(self, img_file):
        """Process single image and compare with GT"""
        img_path = os.path.join(self.image_dir, img_file)
        result = self.ocr.ocr(img_path, cls=True)
        
        # Extract recognized text
        ocr_text = ' '.join([line[1][0] for line in result[0]] if result else [])
        
        # Get ground truth
        gt_text = self.gt_mapping.get(img_file, '')
        
        # Calculate metrics
        cer = levenshtein(ocr_text, gt_text) / max(len(gt_text), 1)
        precision = precision_score(list(gt_text), list(ocr_text[:len(gt_text)]), average='micro')
        recall = recall_score(list(gt_text), list(ocr_text[:len(gt_text)]), average='micro')
        
        # Visualize results
        vis_path = os.path.join('outputs', 'visualized', img_file)
        self.visualize_results(img_path, result, vis_path)
        
        self.results.append({
            'Image': img_file,
            'OCR_Text': ocr_text,
            'GT_Text': gt_text,
            'CER': cer,
            'Precision': precision,
            'Recall': recall,
            'Visualization': vis_path
        })

    def visualize_results(self, img_path, result, output_path):
        """Draw OCR results on image"""
        image = cv2.imread(img_path)
        for line in result[0]:
            box = line[0]
            text = line[1][0]
            confidence = line[1][1]
            
            # Draw bounding box
            pts = np.array(box, np.int32).reshape((-1,1,2))
            cv2.polylines(image, [pts], True, (0,255,0), 2)
            
            # Put text
            cv2.putText(image, f"{text} ({confidence:.2f})", 
                       tuple(map(int, box[0])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.imwrite(output_path, image)

    def generate_report(self):
        """Generate Excel report with metrics"""
        df = pd.DataFrame(self.results)
        with pd.ExcelWriter(self.output_file) as writer:
            df.to_excel(writer, sheet_name='OCR Results', index=False)
            
            # Add summary statistics
            summary = pd.DataFrame({
                'Metric': ['Average CER', 'Average Precision', 'Average Recall'],
                'Value': [df['CER'].mean(), df['Precision'].mean(), df['Recall'].mean()]
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)

if __name__ == "__main__":
    validator = OCRValidator(
        image_dir='images',
        gt_path='ground_truth/labels.txt',
        output_file='outputs/results.xlsx'
    )
    validator.load_ground_truth()
    validator.process_images()
    validator.generate_report()
