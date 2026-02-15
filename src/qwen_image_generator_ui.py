from __future__ import annotations

from pathlib import Path

import streamlit as st

from qwen_image_generator import Qwen25VLImageGenerationSystem


@st.cache_resource
def load_system() -> Qwen25VLImageGenerationSystem:
    return Qwen25VLImageGenerationSystem()


def main() -> None:
    st.set_page_config(page_title="Qwen2.5-VL 画像生成", page_icon="🎨", layout="centered")
    st.title("🎨 Qwen2.5-VL 画像生成システム")
    st.write("日本語の説明文から画像を生成します。")

    prompt = st.text_area(
        "生成したい画像の説明",
        value="雨上がりの東京の夜景、ネオンが反射する路地、映画のワンシーン風",
        height=120,
    )

    col1, col2 = st.columns(2)
    with col1:
        width = st.selectbox("幅", options=[512, 768, 1024], index=2)
        steps = st.slider("推論ステップ", min_value=10, max_value=50, value=30)
    with col2:
        height = st.selectbox("高さ", options=[512, 768, 1024], index=2)
        guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=12.0, value=7.5, step=0.5)

    seed_input = st.text_input("Seed（空欄でランダム）", value="42")
    output_name = st.text_input("出力ファイル名", value="generated_ui.png")

    if st.button("画像を生成", type="primary"):
        if not prompt.strip():
            st.error("プロンプトを入力してください。")
            return

        output_path = Path("outputs") / output_name

        seed = None
        if seed_input.strip():
            try:
                seed = int(seed_input)
            except ValueError:
                st.error("Seed は整数で入力してください。")
                return

        with st.spinner("モデルを読み込み、画像を生成中です...（初回は時間がかかります）"):
            try:
                system = load_system()
                saved = system.generate_image(
                    user_request=prompt,
                    output_path=str(output_path),
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )
            except Exception as exc:
                st.exception(exc)
                return

        st.success(f"画像を保存しました: {saved}")
        st.image(str(saved), caption="生成結果", use_container_width=True)


if __name__ == "__main__":
    main()
