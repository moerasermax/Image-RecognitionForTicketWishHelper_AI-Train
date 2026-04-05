# 🚀 CRNN Captcha Recognition System (Expert Implementation)

這是一個基於 **CRNN (CNN + Bi-LSTM + CTC Loss)** 架構的端到端驗證碼識別系統。本專案紀錄了從數據生成、圖像預處理、模型架構設計到最終部署的完整流程，並成功解決了字元黏連與標籤污染等訓練痛點。

---

## 👤 專家角色與開發理念
本專案由精通電腦視覺 (CV) 與深度學習的資深工程師指導開發，核心理念如下：
*   **架構設計**：優先採用 **CRNN (CNN+Bi-LSTM+CTC)** 結構，利用其處理不定長度序列、無須預先分割字元的原生優勢。
*   **數據策略**：利用 Pillow 生成高擬合合成數據，並模擬噪點、扭曲、陰影等對抗干擾，縮小 **Domain Gap**。
*   **問題診斷**：系統化解決過擬合、無法對齊及 CTC 塌陷等常見 OCR 訓練問題。
*   **代碼實作**：提供模組化、乾淨的 PyTorch 實作。

---

## 🛠️ 專業工具箱 (Tech Stack)
*   **影像處理**：OpenCV, Albumentations (數據增強), Scikit-image, Pillow。
*   **核心模型**：ResNet-based Backbone, Bi-LSTM (Neck), **CTC Loss (Head)**。
*   **優化技巧**：CosineAnnealingLR, 混合精度訓練 (AMP), 預測置信度分析。

---

## 📂 專案目錄結構

```text
captured_data/
├── checkpoints/          # 存放訓練好的權重 (.pth)
├── raw_captcha/          # 修正標籤後的真實樣本 (Label Cleaning)
├── synthetic_data/       # 高擬合度萬級合成數據集
├── 01_trans.py           # 真題預處理與重新命名腳本
├── 02_data_generator_V2.py # V2 高擬合影像生成器 (核心)
├── 03_train.py           # 支援斷點續訓與 LR 自動衰減的訓練腳本
├── 04_predict.py         # 單張影像推論測試工具
├── 05_RandomTest.py      # 隨機抽樣比對與結果診斷工具
├── dataset.py            # PyTorch Dataset 封裝 (處理影像張量化)
├── model.py              # CRNN 模型架構定義 (CNN + Bi-LSTM + CTC)
└── VarelaRound-Regular.ttf # 目標驗證碼使用的核心字型檔案
```

## 🛠️ 全流程操作 SOP
1. 環境配置 (Environment Setup)
* Bash
* pip install torch torchvision opencv-python pillow albumentations
* 2. 數據流水線 (Data Pipeline)
* 清洗真題：將真實驗證碼放入 raw_captcha，確保檔名為 label_uuid.png 且標籤長度嚴格等於 4（避坑指南：標籤長度偏差會直接破壞 CNN 特徵響應）。

* 大規模合成 (02_data_generator_V2.py)：執行生成器產生約 10,000 張合成圖。

* 技術亮點：引入 Smooth Filter 與字元重疊邏輯，模擬真實物理質感。

* 數據混合：將真題重複複製數次併入 synthetic_data，增加模型對真實分佈的敏感度。

3. 模型訓練 (Training Strategy)
* 本專案採用 B 組實驗模式 (From Scratch)，避開早期標籤污染導致的權重偏移。

* 參數設置：學習率 0.0001 起步，配合 CosineAnnealingLR 自動衰減至 0.000003。

* 架構優勢：無須分割字元，由 LSTM 處理水平序列特徵，CTC 負責自動對齊。

* Bash
* python 03_train.py
4. 驗證與診斷 (Validation & Diagnosis)
* Bash
# 隨機選取真題進行對比測試，診斷相似字元 (v/w, a/o) 的辨識邊界
* python 05_RandomTest.py
# 📈 訓練進度報告 (B 組實驗心得)
* 在優化過程中，我們驗證了 「數據質量」遠比「模型複雜度」更重要 的 AI 核心原則：

* 階段,現象診斷,解決對策,最終成果
* 初期 (Fail),標籤長度 3/4 碼混雜導致漏字,啟動 B 組實驗，清空權重並全面修正標籤,排除特徵抑制，建立正確對齊邏輯
* 中期 (V2),合成圖太乾淨導致真題辨識率低,開發 V2 生成器，加入模糊與重疊處理,Val Loss 穩定降至 0.08 以下
* 後期 (Final),"相似字元視覺混淆 (v/w, a/o)",強化發射端針對性數據增強,真題準確率達到 95% - 100%

# 💡 核心技術心得 (Best Practices)
* 標籤即真理：OCR 對標籤長度極其敏感。一個 4 字標成 3 字的錯誤標籤會導致 CNN 濾波器學會忽略該區域的特徵響應。

* 縮小 Domain Gap：合成數據不能太「乾淨」。模擬真圖的「抗鋸齒模糊」與「色彩抖動」是模型實戰不失效的關鍵。

* 學習率排程：前期大步幅快速跳出隨機狀態，後期使用極小學習率 (0.000003) 進行微觀權重修正。

* 層次化解答：優先確認驗證碼物理特性（如字體、背景），再給出針對性方案，而非盲目堆疊卷積層。

# 🧑‍💻 作者與貢獻
# 🤖 CAPTCHA 辨識模型訓練專家 & 專案合作夥伴

* 本專案旨在提供一套從數據生成到模型部署的工業級 OCR SOP，實現具備「直覺辨識力」的 AI 引擎。歡迎討論與交流技術細節！


---

### 💡 上傳 GitHub 的小提醒：
1. **建立檔案**：在 GitHub 網頁版點擊 `Add file` -> `Create new file`。
2. **命名**：輸入 `README.md`。
3. **貼上**：把上面區塊的原始碼全部貼進去。
4. **預覽**：點擊 `Preview` 標籤頁。你會看到所有的 **Emoji、表格、加粗、程式碼區塊** 都會完美整齊地呈現。

這樣一來，你的專案門面就非常專業了！祝你的專案在 GitHub 上大獲成功！
