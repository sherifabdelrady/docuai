"""
DocuAI — Demo: run OCR on any image containing text.
Usage: python demo.py --image path/to/document.jpg
"""

import torch
import cv2
import numpy as np
import argparse
from ocr import CRNN, ctc_decode, preprocess_image, VOCAB


def run_demo(img_path: str):
    print(f"Loading CRNN model...")
    model = CRNN()
    model.eval()
    # Load pretrained weights if available
    import os
    if os.path.exists("checkpoints/crnn_best.pth"):
        model.load_state_dict(torch.load("checkpoints/crnn_best.pth", map_location="cpu"))
        print("Loaded checkpoint.")
    else:
        print("No checkpoint found — using random weights (for demo structure only).")
        print("Train with: python train.py")

    print(f"\nProcessing: {img_path}")
    tensor = preprocess_image(img_path)

    with torch.inference_mode():
        logits = model(tensor)

    text = ctc_decode(logits)
    print(f"\nRecognized text: '{text}'")

    # Show image with OpenCV
    img = cv2.imread(img_path)
    if img is not None:
        cv2.putText(img, f"OCR: {text[:60]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("DocuAI", img)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DocuAI OCR Demo")
    p.add_argument("--image", required=True, help="Path to document image")
    a = p.parse_args()
    run_demo(a.image)
