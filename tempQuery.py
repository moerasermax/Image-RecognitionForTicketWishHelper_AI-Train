import os
data_dir = "./raw_captcha"
files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg'))]
for f in files:
    label = f.split('_')[0]
    if len(label) != 4:
        print(f"⚠️ 標籤長度異常: {f} (長度: {len(label)})")