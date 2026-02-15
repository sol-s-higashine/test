# Qwen2.5-VL 画像生成システム

`Qwen2.5-VL` を使ってユーザーの指示文を高品質な画像生成プロンプトへ変換し、`Stable Diffusion XL` で画像を生成する CLI です。

> 画像生成機能は `src/qwen_image_generator.py` に新規追加しています（`src/main.py` は既存の電卓のまま）。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 実行

```bash
python src/qwen_image_generator.py \
  --prompt "雨上がりの東京の夜景、ネオンが反射する路地、映画のワンシーン風" \
  --output outputs/tokyo.png \
  --width 1024 \
  --height 1024 \
  --steps 30 \
  --guidance-scale 7.5 \
  --seed 42
```

## 構成

1. `Qwen2.5-VL` が入力文を解釈し、英語の詳細プロンプトを生成
2. 生成プロンプト + ネガティブプロンプトを `SDXL` に入力
3. 指定パスへ画像を保存

## 注意

- 初回実行時にモデルダウンロードが発生するため時間とディスク容量を要します。
- GPU 環境を推奨します（CPU 実行も可能ですが非常に遅いです）。
