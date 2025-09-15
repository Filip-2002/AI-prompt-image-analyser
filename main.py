import os
import torch
import cv2
from groundingdino.util.inference import load_model, load_image, predict, annotate

CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"


IMAGE_DIR = "images"
OUTPUT_DIR = os.path.join(IMAGE_DIR, "outputs")

def main():
    device = "cpu"

    print("Loading model...")
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)

    image_name = input("Enter image file name (e.g. room.jpg): ").strip()
    image_path = os.path.join(IMAGE_DIR, image_name)

    if not os.path.exists(image_path):
        print(f"Could not find image at {image_path}")
        return

    image_source, image_tensor = load_image(image_path)

    text_prompt = input("Enter prompt (e.g. 'dog, person'): ").strip()

    print(f"Running detection for prompt: {text_prompt}")
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=0.17,  #change depending on accuracy wanted
        text_threshold=0.17, #change depending on accuracy wanted
        device=device
    )

    annotated_frame = annotate(image_source, boxes, logits, phrases)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base, ext = os.path.splitext(image_name)
    out_path = os.path.join(OUTPUT_DIR, f"{base}_output{ext}")
    cv2.imwrite(out_path, annotated_frame)

    print(f"Done! Results saved to {out_path}")

if __name__ == "__main__":
    main()
