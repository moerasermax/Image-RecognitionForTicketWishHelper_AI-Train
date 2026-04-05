import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut, dropout=0.3):
        super(BidirectionalLSTM, self).__init__()
        # bidirectional=True 讓模型同時看左邊與右邊的上下文，對處理黏連字元至關重要
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        # 加入 Dropout 防止模型過度擬合合成數據
        self.dropout = nn.Dropout(dropout)
        # 雙向輸出維度是 nHidden * 2
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        t, b, h = recurrent.size()
        # 序列展平後進行隨機失活與線性映射
        t_rec = recurrent.view(t * b, h)
        output = self.dropout(t_rec) 
        output = self.embedding(output)
        # 重新回到 [序列長度, Batch, 類別數]
        output = output.view(t, b, -1)
        return output

class CRNN(nn.Module):
    def __init__(self, num_classes, img_channel=1, nh=256):
        super(CRNN, self).__init__()
        
        # 1. CNN Backbone: 提取視覺特徵
        # 輸入: [Batch, 1, 32, 128] (灰階圖)
        self.cnn = nn.Sequential(
            # 第一層：快速捕捉筆畫邊緣
            nn.Conv2d(img_channel, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),      # [64, 16, 64]
            # 第二層：深層特徵
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),             # [128, 8, 32]
            # 第三層：加入 BatchNorm 穩定藍底白字的高對比度梯度
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), 
            # 寬度方向不大幅縮減，保留給序列長度
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),                                       # [256, 4, 33]
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), 
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),                                       # [512, 2, 34]
            # 最後一層將高度壓縮到 1
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True)                                 # [512, 1, 33]
        )
        
        # 2. RNN Neck: 序列建模 (將圖像切片視為時間序列)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, num_classes)
        )
        
        # 3. 權重初始化 (Kaiming Normal) - 解決 Loss 停滯的藥方
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # CNN 提取特徵 -> [b, 512, 1, 33]
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "CNN 輸出高度必須壓縮至 1"
        
        # 調整維度以符合 RNN 輸入格式: [序列長度(w), Batch(b), 特徵通道(c)]
        conv = conv.squeeze(2)          # [b, 512, 33]
        conv = conv.permute(2, 0, 1)    # [33, b, 512]
        
        # RNN 識別序列
        output = self.rnn(conv)         # [33, b, num_classes]
        return output