# generate_cluster1_image_captions.py

import os
import base64
import json
import time
from pathlib import Path

from openai import OpenAI, APIError, RateLimitError
from tqdm.auto import tqdm


# Config
max_captions = 20
retry_limit = 5
sleep_base = 3
skip_existing = True

image_dir = Path("data/clustered_images/cluster_1_no_artifacts")
caption_dir = Path("data/gpt_captions_cluster1")
caption_dir.mkdir(parents=True, exist_ok=True)

system_prompt = "Describe the content of each image concisely."
user_prompt = "Describe very concisely content of this image."


def get_client() -> OpenAI:
    """Return an OpenAI client using OPENAI_API_KEY from environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it before running this script."
        )
    return OpenAI(api_key=api_key)


def get_caption_for_image(client: OpenAI, b64_image: str, filename: str) -> str | None:
    """Call OpenAI vision model to generate a caption for a single image."""
    for attempt in range(retry_limit):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}"
                                },
                            },
                        ],
                    },
                ],
            )

            content = response.choices[0].message.content
            return content.strip()

        except (RateLimitError, APIError) as exc:
            print(f"[{filename}] API error on attempt {attempt + 1}: {exc}")
            if attempt < retry_limit - 1:
                time.sleep(sleep_base * (attempt + 1))
            else:
                return None
        except Exception as exc:
            print(f"[{filename}] Unexpected error: {exc}")
            return None

    return None


def main() -> None:
    """Generate captions for cluster 1 images and save txt + json outputs."""
    client = get_client()
    captions_json: dict[str, str] = {}
    processed = 0

    image_list = sorted(
        [f for f in image_dir.iterdir() if f.suffix.lower() == ".png"],
        key=lambda p: p.name,
    )
    image_list = image_list[:max_captions]

    for img_path in tqdm(image_list, desc="Captioning images", unit="image"):
        if processed >= max_captions:
            break

        caption_txt_path = caption_dir / img_path.with_suffix(".txt").name
        if skip_existing and caption_txt_path.exists():
            # Optionally you could load it into captions_json here if needed
            continue

        with img_path.open("rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")

        caption = get_caption_for_image(client, b64_image, img_path.name)
        if caption is None:
            continue

        with caption_txt_path.open("w", encoding="utf-8") as f:
            f.write(caption)

        captions_json[img_path.name] = caption
        processed += 1

    with Path("gpt_captions_cluster1.json").open("w", encoding="utf-8") as jf:
        json.dump(captions_json, jf, indent=2, ensure_ascii=False)

    print("Completed:", processed, "captions.")


if __name__ == "__main__":
    main()


# # Export OpenAI API key example:
# export OPENAI_API_KEY="your_key_here

# # Run the script:
# python generate_cluster1_image_captions.py