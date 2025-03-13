import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import sentencepiece as spm
import os
from typing import List
import torch.nn.functional as F

# 04train.pyからモデルクラスをインポート
from model import ImageCaptionModel

class ImageCaptionInference:
    def __init__(self, model_path: str, spm_model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # SPMトークナイザーの読み込み
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)
        
        # モデルの準備
        self.model = ImageCaptionModel(vocab_size=8000).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 画像の前処理
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
    
    def generate_caption(self, image_path: str, max_len: int = 30) -> str:
        """画像からキャプションを生成"""
        # 画像の読み込みと前処理
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)  # (1, 3, 64, 64)
        
        # 生成開始
        with torch.no_grad():
            # 画像特徴量の抽出
            B = image.size(0)
            x = self.model.conv(image)
            x = x.view(B, -1)
            img_features = self.model.fc_image(x)
            
            # LSTMの初期状態を設定
            h = img_features.unsqueeze(0)  # (1, 1, hidden_dim)
            c = torch.zeros_like(h)
            
            # 最初のトークン (BOS=1) から生成開始
            curr_ids = torch.LongTensor([[1]]).to(self.device)  # (1, 1)
            
            generated_ids = [1]  # BOSから開始
            
            for _ in range(max_len):
                emb = self.model.embedding(curr_ids)
                output, (h, c) = self.model.lstm(emb, (h, c))
                
                # 次のトークンを予測（確率的生成を導入）
                logits = self.model.fc_out(output[:, -1])  # (1, vocab_size)
                probs = F.softmax(logits / 0.7, dim=-1)  # temperature=0.7で確率を調整
                next_id = torch.multinomial(probs, 1).item()
                
                # デバッグ情報
                token = self.sp.decode_ids([next_id])
                print(f"Probabilities for next token: max={probs.max().item():.4f}")
                print(f"Selected token: {token} (ID={next_id})")
                
                generated_ids.append(next_id)
                
                # EOSが出たら終了（ただし最小長は設定）
                if next_id == 2 and len(generated_ids) > 5:  # 最小5トークン
                    break
                
                curr_ids = torch.LongTensor([[next_id]]).to(self.device)
            
            # 画像特徴量の確認
            print(f"Image features shape: {img_features.shape}")
            print(f"Image features mean: {img_features.mean().item()}")
            
            # 生成過程の確認
            for i, id in enumerate(generated_ids):
                token = self.sp.decode([id])
                print(f"Step {i}: ID={id}, Token='{token}'")
        
        # BOS/EOSトークンを除外してデコード
        caption = self.sp.decode_ids(generated_ids[1:-1])  # BOSとEOSを除外
        return caption

def main():
    # 設定
    MODEL_PATH = "model_epoch10.pth"  # 学習済みモデル
    SPM_MODEL_PATH = "tweets_spm.model"  # SPMモデル
    
    # 推論クラスの初期化
    inferencer = ImageCaptionInference(
        model_path=MODEL_PATH,
        spm_model_path=SPM_MODEL_PATH
    )
    
    # テスト用の画像を何枚か試す
    test_images = [
        "resized_images/F-O1azAaMAAtX7a.jpg",
        "resized_images/F-MO9m7bIAAa6bX.jpg",
        "resized_images/F53wMkxaEAAuMMR.png",
        "resized_images/F53sEiAbIAAgq31.png",
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n画像: {img_path}")
            caption = inferencer.generate_caption(img_path)
            print(f"生成されたキャプション: {caption}")
        else:
            print(f"画像が見つかりません: {img_path}")

if __name__ == "__main__":
    main() 