import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import sentencepiece as spm
import os
from typing import List
import torch.nn.functional as F
from PIL import ImageFont, ImageDraw

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
                if next_id == 2 and len(generated_ids) > 7:  # 最小7トークン
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

    def generate_captioned_image(self, image_path: str, output_path: str = None) -> None:
        """画像にキャプションを追加して保存する関数"""
        # 元画像の読み込み
        original_image = Image.open(image_path).convert('RGB')
        
        # キャプションの生成
        caption = self.generate_caption(image_path)
        
        # 新しい画像サイズの計算（元画像 + キャプション用の余白）
        margin = 50  # キャプション用の余白
        new_width = original_image.width
        new_height = original_image.height + margin
        
        # 新しい画像の作成（白背景）
        new_image = Image.new('RGB', (new_width, new_height), 'white')
        
        # 元画像を配置
        new_image.paste(original_image, (0, 0))
        
        # フォントの設定
        try:
            font = ImageFont.truetype('NotoSansJP-Regular.ttf', 30)
        except:
            # フォントが見つからない場合はデフォルトフォント
            font = ImageFont.load_default()
        
        # テキスト描画
        draw = ImageDraw.Draw(new_image)
        # テキストのサイズを取得して中央揃えに
        text_bbox = draw.textbbox((0, 0), caption, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (new_width - text_width) // 2
        text_y = original_image.height + 10
        
        draw.text((text_x, text_y), caption, font=font, fill='black')
        
        # 保存
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_captioned{ext}"
        
        new_image.save(output_path)
        print(f"キャプション付き画像を保存しました: {output_path}")

def main():
    # 設定
    MODEL_PATH = "model_epoch10.pth"
    SPM_MODEL_PATH = "tweets_spm.model"
    
    # 推論クラスの初期化
    inferencer = ImageCaptionInference(
        model_path=MODEL_PATH,
        spm_model_path=SPM_MODEL_PATH
    )
    
    # 単一画像の処理
    input_image = "test.jpg"  # 処理したい画像のパス
    inferencer.generate_captioned_image(input_image)

if __name__ == "__main__":
    main() 