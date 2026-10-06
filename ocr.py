"""DocuAI — OCR & Document Intelligence (CRNN + CRAFT)"""
import torch, torch.nn as nn, numpy as np, cv2, argparse
from pathlib import Path

VOCAB = " !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"

class BidirectionalLSTM(nn.Module):
    def __init__(self, in_size, hidden, out_size):
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden * 2, out_size)
    def forward(self, x):
        out, _ = self.rnn(x); return self.proj(out)

class CRNN(nn.Module):
    def __init__(self, num_classes=len(VOCAB)+1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1)),
            nn.Conv2d(512, 512, 2), nn.ReLU(),
        )
        self.rnn = nn.Sequential(BidirectionalLSTM(512, 256, 256), BidirectionalLSTM(256, 256, num_classes))
    def forward(self, x):
        feat = self.cnn(x).squeeze(2).permute(0, 2, 1)
        return self.rnn(feat)

def ctc_decode(logits, vocab=VOCAB):
    """Greedy CTC decoder."""
    indices = logits.argmax(-1).squeeze().tolist()
    chars, prev = [], None
    for i in indices:
        if i != prev and i < len(vocab): chars.append(vocab[i])
        prev = i
    return "".join(chars).replace(" ", "")

def preprocess_image(img_path: str, height: int = 32) -> torch.Tensor:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    new_w = int(w * height / h)
    img = cv2.resize(img, (new_w, height)).astype(np.float32) / 255.0
    return torch.tensor(img).unsqueeze(0).unsqueeze(0)

if __name__ == "__main__":
    model = CRNN(); model.eval()
    print(f"CRNN ready. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    dummy = torch.zeros(1, 1, 32, 128)
    out = model(dummy); print(f"Output shape: {out.shape}  (seq_len x batch x classes)")
