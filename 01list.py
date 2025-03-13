import json

def extract_media_tweets(json_file_path):
    # JSONファイルを読み込む
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 画像付きで非RTツイートを抽出
    media_tweets = []
    for item in data:
        tweet = item['tweet']  # ツイートデータは'tweet'キーの中にある
        
        # RTでないことを確認（retweetedフラグを使用）
        if not tweet['full_text'].startswith('RT @'):
            # メディアが含まれているか確認
            if 'extended_entities' in tweet and 'media' in tweet['extended_entities']:
                media_tweets.append({
                    'text': tweet['full_text'],
                    'created_at': tweet['created_at'],
                    'media': tweet['extended_entities']['media']
                })

    return media_tweets

# 使用例
json_file_path = 'twitter-2023-11-09-5729099829981f915f2ca25761d24dc6be3f3f613d340c650d8da49e41b92774/data/tweets.json'
result = extract_media_tweets(json_file_path)

# 結果を確認
print(f"抽出されたツイート数: {len(result)}")

# 結果を新しいJSONファイルに保存
with open('media_tweets.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
