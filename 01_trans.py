import os
import random
import string
from pathlib import Path

def batch_rename_images_only(target_dir: str):
    """
    僅針對圖片檔案進行重新命名：[前四碼]_[9位隨機數字].[原副檔名]
    """
    # 定義圖片副檔名清單（轉換為小寫以利比對）
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    folder = Path(target_dir)
    
    # 檢查路徑是否存在
    if not folder.is_dir():
        print(f"錯誤：找不到資料夾 {target_dir}")
        return

    # 1. 過濾邏輯：僅選取在 IMAGE_EXTENSIONS 清單內的檔案
    # 這裡展現了 Python 的可讀性與 functional programming 特色
    image_files = [
        f for f in folder.iterdir() 
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"預計處理 {len(image_files)} 個圖片檔案...")

    for file_path in image_files:
        # 取得不含副檔名的主檔名 (stem) 與 副檔名 (suffix)
        old_stem = file_path.stem
        ext = file_path.suffix
        
        # 2. 擷取前四碼：Python slice 極具彈性，若長度不足則取全部
        prefix = old_stem[:4]
        
        # 3. 生成 9 位純數字：確保包含前導零且不因數值運算丟失位數
        # 使用 random.choices 確保每一位都是獨立隨機選取
        random_digits = ''.join(random.choices(string.digits, k=9))
        
        new_name = f"{prefix}_{random_digits}{ext}"
        new_file_path = folder / new_name
        
        # 4. 碰撞檢查：防止極端機率下的檔名重複導致錯誤
        # 在 C# 中通常會用 File.Exists() 判斷
        while new_file_path.exists():
            random_digits = ''.join(random.choices(string.digits, k=9))
            new_name = f"{prefix}_{random_digits}{ext}"
            new_file_path = folder / new_name

        try:
            # 執行重命名動作
            file_path.rename(new_file_path)
            print(f"[成功] {file_path.name} -> {new_name}")
        except Exception as e:
            # 捕捉 I/O 異常（如檔案被鎖定、權限不足）
            print(f"[失敗] 無法重新命名 {file_path.name}，原因：{e}")

if __name__ == "__main__":
    # 預設執行目錄為程式所在位置
    batch_rename_images_only(".")