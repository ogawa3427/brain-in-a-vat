import torch
import torch.nn as nn

class ImageCaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        # --- CNN部分 ---
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 64x64 → 32ch
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             # 32x32
            nn.Conv2d(32, 64, 3, padding=1),# 32x32 → 64ch
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             # 16x16
        )
        self.fc_image = nn.Linear(64*16*16, hidden_dim)
        
        # --- LSTM部分 ---
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, text_ids):
        """
        images: (B, 3, 64, 64)
        text_ids: (B, max_len)
        """
        B = images.size(0)
        
        # 1) 画像をエンコード
        x = self.conv(images)           # (B, 64, 16, 16)
        x = x.view(B, -1)              # (B, 64*16*16)
        img_features = self.fc_image(x) # (B, hidden_dim)
        
        # LSTMの初期状態 (h0, c0) を画像特徴から作る
        h0 = img_features.unsqueeze(0)  # (1, B, hidden_dim)
        c0 = torch.zeros_like(h0)

        # 2) テキストをEmbedding
        emb = self.embedding(text_ids)  # (B, max_len, embed_dim)

        # 3) LSTMに通す
        outputs, (hn, cn) = self.lstm(emb, (h0, c0))
        # outputs: (B, max_len, hidden_dim)

        # 4) vocabサイズに変換
        logits = self.fc_out(outputs)   # (B, max_len, vocab_size)

        return logits 