import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def generate_synthetic_data(output_dir, count=5000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    width, height = 128, 32
    bg_color = (0, 112, 221)  # 你的樣本經典藍
    font_color = (255, 255, 255) # 白色
    
    # 【關鍵】請確保路徑指向你下載的 Varela Round 檔案
    font_path = "VarelaRound-Regular.ttf" 
    
    try:
        # 增加字體大小到 28~30，讓它更「胖」一點，增加黏連感
        font = ImageFont.truetype(font_path, 30)
    except:
        print("字體載入失敗，請檢查路徑！")
        font = ImageFont.load_default()

    print(f"正在生成 {count} 張更真實的合成影像...")

    for i in range(count):
        chars = ''.join(random.choices(string.ascii_lowercase, k=4))
        img = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # 【關鍵】縮小間距：原本 22，現在改為 18~20，強制重疊
        start_x = random.randint(5, 15)
        for idx, char in enumerate(chars):
            char_x = start_x + (idx * 19) + random.randint(-1, 1)
            char_y = random.randint(-2, 2) # 稍微上下抖動
            draw.text((char_x, char_y), char, font=font, fill=font_color)

        # 模擬真圖的微弱模糊感
        if random.random() > 0.5:
            img = img.filter(ImageFilter.SMOOTH)

        file_name = f"{chars}_{i}.png"
        img.save(os.path.join(output_dir, file_name))

if __name__ == "__main__":
    # 先把舊的 synthetic_data 刪除或清空，再重新生成
    generate_synthetic_data("./synthetic_data", count=5000)