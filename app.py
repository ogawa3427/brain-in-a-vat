from flask import Flask, request, send_from_directory, jsonify
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename

# ImageCaptionInferenceをインポート
from _06single import ImageCaptionInference

print(os.getcwd())

app = Flask(__name__, static_folder='frontend', static_url_path='')

# Inferenceクラスのインスタンスを作成
inferencer = ImageCaptionInference(
    model_path="model_epoch10.pth",
    spm_model_path="tweets_spm.model"
)

@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'ファイルがありません'}), 400
        
        file = request.files['file']
        # 画像を読み込む
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # RGBAの場合はRGBに変換
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # 一時的に画像を保存
        temp_path = "temp_image.jpg"
        image.save(temp_path)
    
        # キャプションを生成
        caption = inferencer.generate_caption(temp_path)
        return jsonify({'caption': caption})
    finally:
        # 一時ファイルを削除
        if os.path.exists(temp_path):
            os.remove(temp_path)

# @app.route('/generate_')
# def generate_():
#     return jsonify({'message': 'Hello, World!'})

@app.route('/')
def read_root():
    return send_from_directory('frontend', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)