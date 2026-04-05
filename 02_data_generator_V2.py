import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def generate_perfect_fit_data(output_dir, count=5000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    width, height = 128, 32
    # 擴展顏色範圍，模擬不同光學環境下的藍色與白色
    # 真圖藍色約 (0, 112, 221)，我們給它上下 10 點的抖動
    
    font_path = "VarelaRound-Regular.ttf"
    
    print(f"正在生成 {count} 張高擬合度影像...")

    for i in range(count):
        chars = ''.join(random.choices(string.ascii_lowercase, k=4))
        
        # 1. 顏色微抖動
        bg_r = random.randint(0, 10); bg_g = random.randint(100, 120); bg_b = random.randint(210, 230)
        img = Image.new('RGB', (width, height), color=(bg_r, bg_g, bg_b))
        draw = ImageDraw.Draw(img)
        
        # 2. 隨機字體大小 (27~31)，模擬縮放感
        f_size = random.randint(27, 31)
        font = ImageFont.truetype(font_path, f_size)
        
        # 3. 極度黏連邏輯：動態計算每個字的 x 位置
        # 讓字與字之間的步進 (Step) 甚至小於字體寬度
        current_x = random.randint(5, 12)
        for char in chars:
            # 設置白色文字，並帶有微小的透明度或灰度變化
            txt_color = (random.randint(245, 255), random.randint(245, 255), random.randint(245, 255))
            draw.text((current_x, random.randint(-2, 2)), char, font=font, fill=txt_color)
            # 關鍵：步進僅 17~19 像素，強制重疊
            current_x += random.randint(17, 20) 

        # 4. 邊緣平滑處理 (這是關鍵！)
        # 真圖看起來不是銳利的像素，而是有平滑過渡
        img = img.filter(ImageFilter.SMOOTH_MORE)
        
        # 5. 模擬輕微的形狀扭曲 (使用 numpy 進行仿射變換或彈性變形)
        # 這裡簡單使用內建的濾波，若需要更強可加入彈性形變
        if random.random() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        file_name = f"{chars}_{i}.png"
        img.save(os.path.join(output_dir, file_name))

if __name__ == "__main__":
    generate_perfect_fit_data("./synthetic_data", count=8000) # B 組實驗建議量大一點