import json
import os
import requests
from PIL import Image
from io import BytesIO

# 出力ディレクトリの作成
output_dir = "resized_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 元のJSONデータを読み込み
with open("media_tweets.json", "r", encoding="utf-8") as f:
    tweets = json.load(f)

# 新しい辞書を作成
image_text_map = {}

# 合計ツイート数とカウンターの初期化
total_media_tweets = sum(1 for tweet in tweets if "media" in tweet and tweet["media"])
processed_count = 0

for tweet in tweets:
    # メディアが存在し、少なくとも1つの画像/動画がある場合
    if "media" in tweet and tweet["media"]:
        media = tweet["media"][0]  # 最初のメディアのみを使用
        processed_count += 1
        
        # media_urlを取得
        media_url = media.get("media_url_https") or media.get("media_url")
        if not media_url:
            continue

        try:
            print(f"画像を処理中... {processed_count}/{total_media_tweets}")
            # 画像をダウンロード
            response = requests.get(media_url)
            img = Image.open(BytesIO(response.content))

            # 64x64にリサイズ
            img = img.resize((64, 64), Image.Resampling.LANCZOS)

            # ファイル名を生成（media_url の最後の部分を使用）
            filename = os.path.basename(media_url)
            output_path = os.path.join(output_dir, filename)

            # 保存
            img.save(output_path)

            # URLパターンを除去してから辞書に追加
            text = tweet["text"]
            # https:// で始まるURLを除去
            text = ' '.join(word for word in text.split() if not word.startswith('https://'))
            # 空白の除去
            text = text.strip()
            
            # テキストが空でない場合のみ辞書に追加
            if text:
                image_text_map[filename] = text

        except Exception as e:
            print(f"Error processing {media_url}: {e}")

# 新しい辞書をJSONとして保存
with open("image_text_map.json", "w", encoding="utf-8") as f:
    json.dump(image_text_map, f, ensure_ascii=False, indent=2)