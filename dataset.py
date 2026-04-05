import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A

class CaptchaDataset(Dataset):
    def __init__(self, root_dir, characters, img_width=128, img_height=32, transform=None):
        """
        OCR 數據集類別
        :param root_dir: 圖片資料夾路徑
        :param characters: 字元集 (例如 "abcdefg...")
        :param transform: Albumentations 轉換流
        """
        self.root_dir = root_dir
        # 僅讀取影像檔案，排除隱藏檔
        self.img_names = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.img_width = img_width
        self.img_height = img_height
        self.transform = transform
        
        # --- CTC 字典建置 ---
        # 重要：0 必須保留給 CTC 的 'blank'，所以索引從 1 開始
        self.characters = characters
        self.char_to_idx = {char: i + 1 for i, char in enumerate(characters)}
        self.idx_to_char = {i + 1: char for i, char in enumerate(characters)}
        self.num_classes = len(characters) + 1

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # 1. 讀取影像 (BGR 轉 RGB)
        image = cv2.imread(img_path)
        if image is None:
            # 容錯處理：如果讀取失敗，隨機抓下一張
            return self.__getitem__(np.random.randint(0, len(self.img_names)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 解析標籤 (假設檔名格式: label_id.png)
        # 例如 nipa_123.png -> label_str 為 nipa
        label_str = img_name.split('_')[0].lower()
        
        # 3. 數據增強 (Albumentations)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # 4. 影像預處理
        # 轉灰階 -> 縮放至指定尺寸 -> 歸一化至 [-1, 1]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (self.img_width, self.img_height))
        image = (image.astype(np.float32) / 255.0 - 0.5) / 0.5
        
        # 轉為 Tensor [Channel, Height, Width]
        image = torch.from_numpy(image).unsqueeze(0) 
        
        # 5. 標籤轉索引序列
        target = [self.char_to_idx[c] for c in label_str if c in self.char_to_idx]
        target = torch.LongTensor(target)
        target_length = torch.IntTensor([len(target)])
        
        return image, target, target_length

def ocr_collate_fn(batch):
    """
    自定義 Batch 收集函式
    解決標籤長度不一 (Variable length labels) 無法使用預設 stack 的問題
    """
    images, targets, target_lengths = zip(*batch)
    
    # 影像維度固定，直接疊加: [Batch, 1, 32, 128]
    images = torch.stack(images, 0)
    
    # 標籤拼接成一個一維向量: [Total_Chars_in_Batch]
    targets = torch.cat(targets, 0)
    
    # 記錄每個樣本標籤的原始長度: [Batch]
    target_lengths = torch.cat(target_lengths, 0)
    
    return images, targets, target_lengths

# ==========================================
# 🔧 專業配置範例
# ==========================================

# 1. 字元集 (請確保包含你所有會出現的字母)
CHAR_SET = "abcdefghijklmnopqrstuvwxyz" 

# 2. 訓練增強策略 (對抗黏連、背景干擾)
train_transform = A.Compose([
    # 彈性形變：模擬字母被擠壓、拉伸的狀態
    A.ElasticTransform(alpha=1, sigma=50, p=0.5),
    # 高斯雜訊與模糊
    A.GaussNoise(p=0.2),
    A.Blur(blur_limit=3, p=0.2),
    # 亮度與對比度
    A.RandomBrightnessContrast(p=0.3),
])

# 3. 實例化測試區塊
if __name__ == "__main__":
    # 這裡可以根據你的資料夾名稱自行修改路徑測試
    SYN_DIR = "./synthetic_data"
    REAL_DIR = "./raw_captcha"
    
    if os.path.exists(SYN_DIR):
        ds = CaptchaDataset(SYN_DIR, CHAR_SET, transform=train_transform)
        print(f"✅ 成功載入合成樣本: {len(ds)} 張")
        
        # 測試 DataLoader
        loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=ocr_collate_fn)
        imgs, tars, lens = next(iter(loader))
        print(f"📊 Batch 影像維度: {imgs.shape}") # 預期 [4, 1, 32, 128]
        print(f"📝 標籤長度序列: {lens}")
    else:
        print(f"⚠️ 找不到路徑: {SYN_DIR}")