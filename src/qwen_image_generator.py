from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class Qwen25VLImageGenerationSystem:
    """Qwen2.5-VL でプロンプトを生成し、SDXL で画像を作るシステム。"""

    def __init__(
        self,
        qwen_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        diffusion_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(qwen_model)
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_model,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )

        pipeline_kwargs: dict[str, object] = {
            "torch_dtype": self.torch_dtype,
            "use_safetensors": True,
        }
        if self.device == "cuda":
            pipeline_kwargs["variant"] = "fp16"

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            diffusion_model,
            **pipeline_kwargs,
        )
        self.pipeline = self.pipeline.to(self.device)

    def build_prompt(
        self,
        user_request: str,
        negative_prompt: str = "low quality, blurry, distorted",
    ) -> tuple[str, str]:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a visual prompt engineer. Convert user requests into rich English "
                            "prompts for text-to-image generation."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Generate a concise but vivid prompt for a diffusion model. "
                            f"User request: {user_request}"
                        ),
                    }
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], images=None, videos=None, return_tensors="pt")
        inputs = {name: tensor.to(self.qwen.device) for name, tensor in inputs.items()}

        with torch.inference_mode():
            generated = self.qwen.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
            )

        generated_ids = generated[:, inputs["input_ids"].shape[1] :]
        prompt = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        return prompt, negative_prompt

    def generate_image(
        self,
        user_request: str,
        output_path: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> Path:
        prompt, negative_prompt = self.build_prompt(user_request)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(save_path)
        return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL ベースの画像生成システム")
    parser.add_argument("--prompt", required=True, help="生成したい画像の説明（日本語可）")
    parser.add_argument("--output", default="outputs/generated.png", help="出力画像パス")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    system = Qwen25VLImageGenerationSystem()
    output = system.generate_image(
        user_request=args.prompt,
        output_path=args.output,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    print(f"画像を保存しました: {output}")


if __name__ == "__main__":
    main()
