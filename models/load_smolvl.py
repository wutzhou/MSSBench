# --------------------------------------------------------------
#  SmolVLM-500M-Instruct  (≈500 M params, ~2 GB VRAM in FP16)
# --------------------------------------------------------------
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import json
import tqdm
import random

# ------------------------------------------------------------------
# 1. Load processor + model (trust_remote_code is required for SmolVLM)
# ------------------------------------------------------------------
model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"

# Load model directly
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct", 
    device_map="auto"
)

# ------------------------------------------------------------------
# 2. Helper that mimics the original `call_model(image_path, query)`
# ------------------------------------------------------------------
def call_model(image_path: str, query: str) -> str:
    """
    Args:
        image_path: path to a local image file
        query:      text prompt (e.g. "What do you see in the image?")

    Returns:
        The model's textual answer.
    """
    # Load the image with PIL (SmolVLM expects a PIL.Image)
    ## pil_image = Image.open(image_path).convert("RGB")

    # ------------------------------------------------------------------
    # Build the chat message exactly as SmolVLM expects:
    #   - a list of dicts with role "user"/"assistant"
    #   - the image is passed separately to `processor`
    # ------------------------------------------------------------------
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": query}
            ]
        },
    ]

    inputs = processor.apply_chat_template(
	    messages,
	    add_generation_prompt=True,
	    tokenize=True,
	    return_dict=True,
	    return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=40)

    # Remove the prompt tokens – keep only the newly generated part
    answer_ids = generated_ids[0, inputs["input_ids"].shape[1] :]
    answer = processor.tokenizer.decode(answer_ids, skip_special_tokens=True)

    return answer


# --------------------------------------------------------------
# 3. Example usage
# --------------------------------------------------------------
if __name__ == "__main__":
    img_path = "candy.JPG"          # <-- replace with your image
    question = "What animal is on the candy?"

    print("Question:", question)
    print("Answer  :", call_model(img_path, question))
