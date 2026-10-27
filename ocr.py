"""
DocuAI - OCR & Document Intelligence (CRNN + CRAFT)

Audit fixes (audit/production-remediation):
- P1 FIX: Added Arabic Unicode characters to VOCAB.
  Original VOCAB was ASCII-only (96 chars) — zero Arabic support
  despite the repository claiming "96.4% word accuracy on Arabic + Latin".
- P1 FIX: CTC decoder no longer strips spaces. Original .replace(" ","")
  destroyed word boundaries, returning one merged string for multi-word text.
- P1 FIX: preprocess_image() now raises FileNotFoundError rather than
  crashing with AttributeError on None.shape when the file is missing.
- P1 FIX: ctc_decode handles batch dimension correctly (squeeze is unsafe
  when batch_size > 1 — replaced with explicit indexing).
- P2 FIX: CRNN input channel is 1 (grayscale) — documented explicitly.
- P2 FIX: Added model checkpoint loading path with clear error message
  when weights are missing (was silently using random weights).
- P2 FIX: Added basic Arabic text reshaping note — proper RTL rendering
  requires python-arabic-reshaper + bidi algorithm in downstream display code.

NOTE: Arabic-language WEIGHTS are not included in this repository.
The architecture supports Arabic characters but model accuracy on Arabic
text is UNVERIFIED without trained weights. See README for training instructions.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
# ASCII printable (excluding control chars): space through tilde
_ASCII = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"

# Arabic Unicode block: U+0600–U+06FF (core Arabic letters and diacritics)
# Common Arabic letters and punctuation used in printed text
_ARABIC = (
    "\u0621\u0622\u0623\u0624\u0625\u0626\u0627\u0628\u0629\u062A"  # ء آ أ ؤ إ ئ ا ب ة ت
    "\u062B\u062C\u062D\u062E\u062F\u0630\u0631\u0632\u0633\u0634"  # ث ج ح خ د ذ ر ز س ش
    "\u0635\u0636\u0637\u0638\u0639\u063A\u0641\u0642\u0643\u0644"  # ص ض ط ظ ع غ ف ق ك ل
    "\u0645\u0646\u0647\u0648\u0649\u064A"                          # م ن ه و ى ي
    "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"  # Arabic-Indic digits ٠١٢٣٤٥٦٧٨٩
    "\u060C\u061B\u061F\u0640"                                       # ، ؛ ؟ ـ (kashida)
)

VOCAB: str = _ASCII + _ARABIC
# CTC blank token index = len(VOCAB)  (convention: blank = last class)
BLANK_IDX: int = len(VOCAB)
NUM_CLASSES: int = len(VOCAB) + 1   # +1 for CTC blank


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BidirectionalLSTM(nn.Module):
    def __init__(self, in_size: int, hidden: int, out_size: int):
        super().__init__()
        self.rnn  = nn.LSTM(in_size, hidden, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden * 2, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.proj(out)


class CRNN(nn.Module):
    """
    Convolutional-Recurrent Neural Network for sequence recognition.

    Input:  (B, 1, H, W) — grayscale, H=32 px
    Output: (B, T, num_classes) — log-softmax over vocabulary + CTC blank
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),   nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2), nn.ReLU(),
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)               # (B, C, 1, W')
        feat = feat.squeeze(2)           # (B, C, W')
        feat = feat.permute(0, 2, 1)     # (B, W', C) = (B, T, C)
        return self.rnn(feat)            # (B, T, num_classes)


def load_model(weights_path: str | None = None, device: str = "cpu") -> CRNN:
    """
    Create CRNN and optionally load pretrained weights.
    Raises FileNotFoundError clearly if weights_path is given but missing.
    """
    model = CRNN(NUM_CLASSES).to(device)
    if weights_path is not None:
        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Weights file not found: {p}\n"
                "Train a model first: see README.md → Training section."
            )
        state = torch.load(p, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded weights: {p}")
    else:
        print("WARNING: No weights loaded — model uses random initialisation.")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# CTC Greedy Decoder
# ---------------------------------------------------------------------------

def ctc_decode(logits: torch.Tensor, vocab: str = VOCAB) -> list[str]:
    """
    Greedy CTC decoder — handles a batch of sequences.

    Parameters
    ----------
    logits : (B, T, num_classes) or (T, num_classes) for single sample
    vocab  : character vocabulary (len == num_classes - 1; blank = last index)

    Returns
    -------
    List of decoded strings, one per batch element.
    """
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)   # (1, T, C)

    blank_idx = len(vocab)
    results: list[str] = []
    for b in range(logits.size(0)):
        indices = logits[b].argmax(-1).tolist()   # (T,)
        chars: list[str] = []
        prev: int | None = None
        for idx in indices:
            if idx != prev and idx != blank_idx:
                if idx < len(vocab):
                    chars.append(vocab[idx])
            prev = idx
        # NOTE: do NOT strip spaces — they separate words in multi-word text.
        results.append("".join(chars))
    return results


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(img_path: str, height: int = 32) -> torch.Tensor:
    """
    Load a grayscale image and resize to fixed height, preserving aspect ratio.

    Returns (1, 1, height, width) tensor, float32, normalised to [0, 1].
    Raises FileNotFoundError if the file does not exist.
    """
    path = Path(img_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")

    h, w = img.shape
    new_w = max(1, int(w * height / h))
    img = cv2.resize(img, (new_w, height)).astype(np.float32) / 255.0
    return torch.tensor(img).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocuAI OCR inference")
    parser.add_argument("--image",   default=None,  help="Path to input image")
    parser.add_argument("--weights", default=None,  help="Path to trained .pt weights")
    parser.add_argument("--device",  default="cpu", help="cpu | cuda")
    args = parser.parse_args()

    model = load_model(args.weights, args.device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CRNN ready. Params: {total_params:.1f}M | Vocab size: {len(VOCAB)} chars")
    print(f"  ASCII: {len(_ASCII)} chars | Arabic: {len(_ARABIC)} chars | Blank idx: {BLANK_IDX}")

    if args.image:
        tensor = preprocess_image(args.image).to(args.device)
        with torch.no_grad():
            logits = model(tensor)
        texts = ctc_decode(logits)
        print(f"Prediction: {texts[0]!r}")
    else:
        # Smoke test with random input
        dummy = torch.zeros(1, 1, 32, 128).to(args.device)
        with torch.no_grad():
            out = model(dummy)
        print(f"Output shape: {out.shape}  (batch=1, seq_len={out.shape[1]}, classes={out.shape[2]})")
