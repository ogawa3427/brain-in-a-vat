import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import csv
import os
from model import ImageCaptionModel  # モデルをインポート

# データセットクラスの実装
class TwitterImageCaptionDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, max_len=30, pad_id=0):
        self.transform = transform
        self.max_len = max_len
        self.pad_id = pad_id
        self.img_dir = img_dir
        
        self.data = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row["image_path"]
                # text_ids をスペース区切り → 整数化
                ids = list(map(int, row["text_ids"].split()))
                self.data.append((img_path, ids))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, ids = self.data[idx]
        
        # 画像を読み込み
        img = Image.open(os.path.join(self.img_dir, img_path)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        # パディング / トリミング
        if len(ids) < self.max_len:
            ids = ids + [self.pad_id] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        
        text_tensor = torch.LongTensor(ids)  # (max_len,)
        
        return img, text_tensor

# モデルの定義
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

def main():
    # ハイパーパラメータ
    VOCAB_SIZE = 8000  # SPMのvocabサイズ
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    MAX_LEN = 30
    
    # データ変換
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    # データセットの準備
    dataset = TwitterImageCaptionDataset(
        csv_path="data/train_data.csv",
        img_dir="resized_images",
        transform=transform,
        max_len=MAX_LEN,
        pad_id=0
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # モデルの準備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ImageCaptionModel(vocab_size=VOCAB_SIZE).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD_ID=0を無視
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 学習ループ
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for images, text_ids in dataloader:
            images = images.to(device)
            text_ids = text_ids.to(device)
            
            optimizer.zero_grad()
            
            # (B, max_len, vocab_size)
            logits = model(images, text_ids[:, :-1])
            
            # logits: (B, max_len-1, vocab_size)
            #  正解: (B, max_len-1)
            target = text_ids[:, 1:]
            
            loss = criterion(logits.transpose(1,2), target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{num_batches}]  Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Average Loss: {avg_loss:.4f}")
        
        # モデルの保存
        torch.save(model.state_dict(), f"model_epoch{epoch+1}.pth")

if __name__ == "__main__":
    main() 