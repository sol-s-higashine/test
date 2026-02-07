import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_model(model_id: str) -> Qwen2_5_VLForConditionalGeneration:
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )


def extract_dialogue(image_path: Path, model_id: str, max_new_tokens: int) -> str:
    image = Image.open(image_path).convert("RGB")
    processor = AutoProcessor.from_pretrained(model_id)
    model = load_model(model_id)

    prompt = (
        "You are a manga OCR assistant. Extract only the speech balloon dialogue "
        "from the image. Return the dialogue as plain text, line by line, without "
        "extra commentary."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract speech balloon dialogue from a manga image using Qwen2.5-VL",
    )
    parser.add_argument("image", type=Path, help="Path to a manga image file")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Hugging Face model id for Qwen2.5-VL",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate for dialogue",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    dialogue = extract_dialogue(args.image, args.model, args.max_new_tokens)
    print(dialogue)


if __name__ == "__main__":
    main()
