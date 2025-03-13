from huggingface_hub import HfApi
import os

import env

def upload_model_to_hub(
    model_path: str,
    tokenizer_path: str,
    repo_id: str,
    token: str
):
    """
    モデルとトークナイザーをHugging Face Hubにアップロードする
    
    Args:
        model_path: モデルファイルのパス（.pth）
        tokenizer_path: トークナイザーファイルのパス（.model）
        repo_id: 'ユーザー名/リポジトリ名'の形式
        token: Hugging Faceのアクセストークン
    """
    api = HfApi()
    
    # リポジトリが存在しない場合は作成
    try:
        api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
        print(f"リポジトリの準備完了: {repo_id}")
    except Exception as e:
        print(f"リポジトリ作成エラー: {e}")
        return

    # モデルファイルのアップロード
    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=os.path.basename(model_path),
            repo_id=repo_id,
            token=token
        )
        print(f"モデルファイルのアップロード完了: {model_path}")
    except Exception as e:
        print(f"モデルアップロードエラー: {e}")

    # トークナイザーファイルのアップロード
    try:
        api.upload_file(
            path_or_fileobj=tokenizer_path,
            path_in_repo=os.path.basename(tokenizer_path),
            repo_id=repo_id,
            token=token
        )
        print(f"トークナイザーファイルのアップロード完了: {tokenizer_path}")
    except Exception as e:
        print(f"トークナイザーアップロードエラー: {e}")

def main():
    # 設定
    MODEL_PATH = env.MODEL_PATH_
    TOKENIZER_PATH = env.TOKENIZER_PATH_
    REPO_ID = env.REPO_ID_
    TOKEN = env.TOKEN_
    
    # アップロード実行
    upload_model_to_hub(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        repo_id=REPO_ID,
        token=TOKEN
    )

if __name__ == "__main__":
    main()