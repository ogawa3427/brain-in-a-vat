import sentencepiece as spm
import csv
import json
from pathlib import Path

def train_tokenizer(input_file: str, model_prefix: str, vocab_size: int = 8000):
    """SentencePieceモデルを学習する"""
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        user_defined_symbols=["<BOS>", "<EOS>"]
    )

def create_token_dataset(
    data_pairs: list,
    output_file: str,
    model_path: str,
    add_special_tokens: bool = True
):
    """トークナイズしたデータセットを作成する"""
    # トークナイザーの準備
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    # 出力ディレクトリの作成
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # データセットの作成
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "text_ids"])

        for img_path, text in data_pairs:
            # テキストをトークナイズ
            ids = sp.encode_as_ids(text)
            
            # 必要に応じてBOS/EOSトークンを追加
            if add_special_tokens:
                ids = [sp.bos_id()] + ids + [sp.eos_id()]
            
            # ID列を文字列に変換
            ids_str = " ".join(map(str, ids))
            writer.writerow([img_path, ids_str])

def load_image_text_pairs(json_file: str) -> list:
    """JSONファイルから画像とテキストのペアを読み込む"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # (image_path, text) のリストを作成
    pairs = [(k, v) for k, v in data.items()]
    return pairs

def create_corpus_file(pairs: list, output_file: str):
    """トークナイザー学習用のコーパスファイルを作成"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, text in pairs:
            f.write(text + '\n')

def main():
    # JSONファイルから画像-テキストペアを読み込む
    data_pairs = load_image_text_pairs('image_text_map.json')
    
    # コーパスファイルの作成
    create_corpus_file(data_pairs, 'tweets_corpus.txt')
    
    # トークナイザーの学習
    train_tokenizer(
        input_file="tweets_corpus.txt",
        model_prefix="tweets_spm",
        vocab_size=8000
    )

    # データセットの作成
    create_token_dataset(
        data_pairs=data_pairs,
        output_file="data/train_data.csv",
        model_path="tweets_spm.model",
        add_special_tokens=True
    )

if __name__ == "__main__":
    main()
