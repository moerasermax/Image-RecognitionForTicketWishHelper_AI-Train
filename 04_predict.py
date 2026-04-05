import torch
import cv2
import numpy as np
from model import CRNN # 確保 model.py 在同目錄

def decode_predictions(preds, alphabet):
    """CTC 貪婪解碼邏輯"""
    # preds shape: [seq_len, batch, num_classes]
    preds = preds.permute(1, 0, 2) # [batch, seq_len, num_classes]
    _, max_indices = torch.max(preds, 2) # 取最大機率索引
    
    decoded_results = []
    for i in range(max_indices.size(0)):
        res = ""
        prev = 0
        for idx in max_indices[i]:
            if idx != 0 and idx != prev: # 非 blank 且不與前一個字元重複
                res += alphabet[idx - 1]
            prev = idx
        decoded_results.append(res)
    return decoded_results

def predict_single_image(img_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    
    # 1. 載入模型
    model = CRNN(num_classes=len(alphabet) + 1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. 影像預處理 (必須與訓練時完全一致)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 32))
    img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. 推論
    with torch.no_grad():
        preds = model(img_tensor)
        result = decode_predictions(preds, alphabet)
    
    return result[0]

if __name__ == "__main__":
    # 測試一張你 raw_captcha 資料夾裡的圖
    test_img = "./raw_captcha/vpvo_312482997.png" # 改成你實際存在的檔名
    weight_file = "./checkpoints/crnn_last.pth"
    pred_text = predict_single_image(test_img, weight_file)
    print(f"📷 影像: {test_img}")
    print(f"🔮 模型預測結果: {pred_text}")
    
    
    test_img = "./raw_captcha/rwto_537958183.png" # 改成你實際存在的檔名
    weight_file = "./checkpoints/crnn_last.pth"
    pred_text = predict_single_image(test_img, weight_file)
    print(f"📷 影像: {test_img}")
    print(f"🔮 模型預測結果: {pred_text}")
    
    
    test_img = "./raw_captcha/iwi__102283177.png" # 改成你實際存在的檔名
    weight_file = "./checkpoints/crnn_last.pth"
    pred_text = predict_single_image(test_img, weight_file)
    print(f"📷 影像: {test_img}")
    print(f"🔮 模型預測結果: {pred_text}")
    
    
    test_img = "./raw_captcha/juje_215719840.png" # 改成你實際存在的檔名
    weight_file = "./checkpoints/crnn_last.pth"
    pred_text = predict_single_image(test_img, weight_file)
    print(f"📷 影像: {test_img}")
    print(f"🔮 模型預測結果: {pred_text}")
    
    
    test_img = "./raw_captcha/keua_800421967.png" # 改成你實際存在的檔名
    weight_file = "./checkpoints/crnn_last.pth"
    pred_text = predict_single_image(test_img, weight_file)
    print(f"📷 影像: {test_img}")
    print(f"🔮 模型預測結果: {pred_text}")