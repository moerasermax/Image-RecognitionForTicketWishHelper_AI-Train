import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# 確保你的檔案名稱正確
from dataset import CaptchaDataset, ocr_collate_fn
from model import CRNN

def train():
    # --- 1. 核心配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHARS = "abcdefghijklmnopqrstuvwxyz"
    BATCH_SIZE = 64
    # 微調階段建議降低學習率，防止破壞已學到的特徵
    LEARNING_RATE = 0.000005
    EPOCHS = 12
    SAVE_PATH = "./checkpoints"
    MODEL_FILE = os.path.join(SAVE_PATH, "crnn_last.pth")
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # --- 2. 數據載入 ---
    # 注意：這裡的合成數據應包含你最新生成的 Varela Round 字體
    train_ds = CaptchaDataset("./synthetic_data", CHARS)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ocr_collate_fn)

    val_ds = CaptchaDataset("./raw_captcha", CHARS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ocr_collate_fn)

    # --- 3. 初始化模型 ---
    model = CRNN(num_classes=len(CHARS) + 1).to(device)
    
    # 【關鍵：自動載入權重邏輯】
    if os.path.exists(MODEL_FILE):
        print(f"♻️ 偵測到舊權重：{MODEL_FILE}，正在載入並進行續訓...")
        # map_location 確保在不同設備間切換不會報錯
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    else:
        print("🆕 未發現舊權重，將從頭開始訓練（使用 Kaiming 初始化）...")

    # --- 4. 優化器與損失函數 ---
    # 加入 weight_decay (權重衰減) 幫助進一步防止過擬合
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # 使用 CosineAnnealing 讓學習率更平滑地下降
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"🚀 戰鬥開始！設備: {device} | 訓練集規模: {len(train_ds)}")

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        
        for i, (imgs, tars, tar_lens) in enumerate(train_loader):
            imgs, tars = imgs.to(device), tars.to(device)
            
            preds = model(imgs)
            batch_size = imgs.size(0)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size).to(device)
            
            loss = criterion(preds.log_softmax(2), tars, preds_size, tar_lens)
            
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪：防止 LSTM 在處理複雜黏連時梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
            epoch_train_loss += loss.item()

        # 驗證環節
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for imgs, tars, tar_lens in val_loader:
                imgs, tars = imgs.to(device), tars.to(device)
                preds = model(imgs)
                preds_size = torch.IntTensor([preds.size(0)] * imgs.size(0)).to(device)
                v_loss = criterion(preds.log_softmax(2), tars, preds_size, tar_lens)
                epoch_val_loss += v_loss.item()
        
        scheduler.step()
        
        avg_train = epoch_train_loss / len(train_loader)
        avg_val = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 每 5 輪存檔一次，確保進度不遺失
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), MODEL_FILE)
            print(f"💾 進度已保存至 {MODEL_FILE}")

    print("✨ 訓練流程結束！")

if __name__ == "__main__":
    train()