import torch
import cv2
import numpy as np
import os
import random
from model import CRNN

def decode_predictions(preds, alphabet):
    """CTC 貪婪解碼 (Greedy Decoder)"""
    preds = preds.permute(1, 0, 2) # [batch, seq_len, num_classes]
    _, max_indices = torch.max(preds, 2)
    
    decoded_results = []
    for i in range(max_indices.size(0)):
        res = ""
        prev = 0
        for idx in max_indices[i]:
            if idx != 0 and idx != prev: # 0 是 blank 標籤
                res += alphabet[idx - 1]
            prev = idx
        decoded_results.append(res)
    return decoded_results

def test_random_samples(data_dir, model_path, num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    
    # 1. 載入模型
    model = CRNN(num_classes=len(alphabet) + 1).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ 成功載入權重: {model_path}")
    else:
        print("❌ 找不到權重檔案！")
        return

    # 2. 隨機抓取檔案
    all_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(all_files) < num_samples:
        num_samples = len(all_files)
    
    samples = random.sample(all_files, num_samples)
    
    print("-" * 50)
    print(f"{'真實標籤 (Label)':<20} | {'模型預測 (Predict)':<20} | {'結果'}")
    print("-" * 50)

    for file_name in samples:
        # 解析真實標籤 (假設格式為 label_id.png)
        ground_truth = file_name.split('_')[0].lower()
        
        # 影像預處理
        img_path = os.path.join(data_dir, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 32))
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
        
        # 推論
        with torch.no_grad():
            preds = model(img_tensor)
            predict_text = decode_predictions(preds, alphabet)[0]
        
        status = "✅" if ground_truth == predict_text else "❌"
        print(f"{ground_truth:<20} | {predict_text:<20} | {status}")

if __name__ == "__main__":
    # 設定你的路徑
    REAL_DATA_DIR = "./raw_captcha"
    WEIGHTS = "./checkpoints/crnn_last.pth"
    
    test_random_samples(REAL_DATA_DIR, WEIGHTS, num_samples=5)