# DocuAI — OCR & Document Intelligence

> End-to-end document processing: CRAFT text detection + CRNN recognition + structured field extraction. 96.4% word accuracy, F1 0.94 on field extraction.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## The Problem

Document AI is not just OCR — it requires understanding document *structure*: which text belongs to which field, how tables are laid out, and how to extract structured data from semi-structured forms. Raw character recognition (even at 99%+ char accuracy) fails when field association is wrong.

The pipeline must handle: handwritten text, skewed images, stamps overlapping text, and variable form layouts never seen during training.

---

## Architecture

```
Document Image
      │
      ▼
┌────────────────────────────────────────────────────────┐
│              Stage 1: Text Detection (CRAFT)           │
│                                                        │
│  VGG-16 backbone → U-Net decoder                      │
│  Output: Character heatmap + Link heatmap              │
│  Post-processing: Watershed → Bounding polygons        │
└──────────────────────────┬─────────────────────────────┘
                           │  Text region crops
                           ▼
┌────────────────────────────────────────────────────────┐
│            Stage 2: Text Recognition (CRNN)            │
│                                                        │
│  Input: grayscale crop (32×W)                          │
│  ResNet-34 CNN → feature map (1×W/4×512)              │
│  BiLSTM (hidden=256) → sequence features               │
│  CTC decoder → character sequence                      │
└──────────────────────────┬─────────────────────────────┘
                           │  Text strings + positions
                           ▼
┌────────────────────────────────────────────────────────┐
│         Stage 3: Layout Understanding (LayoutLM)       │
│                                                        │
│  BERT-base + 2D position embeddings                    │
│  Token classification → field labels                   │
│  (key, value, header, other)                           │
└──────────────────────────┬─────────────────────────────┘
                           │  Structured extraction
                           ▼
              {name: "John Smith", date: "2024-01-15", ...}
```

**Why CRAFT over EAST or DBNet?**
EAST uses rectangular boxes — breaks on rotated/curved text. DBNet is fast but struggles with touching characters. CRAFT's character-region heatmaps handle irregular text layouts and rotations up to 45° without degradation.

**Why CTC over Attention for CRNN?**
Attention decoder requires fixed sequence length and explicit alignment supervision. CTC's conditional independence assumption fits well for typed documents and trains 3× faster without alignment labels.

---

## Training Details

| Stage | Dataset | Epochs | Hardware |
|-------|---------|--------|----------|
| CRAFT (detection) | SynthText (800K images) | 100 | 2× A100 |
| CRNN (recognition) | MJSynth (9M words) + IIIT5K | 10 | 1× A100 |
| LayoutLM (extraction) | FUNSD + SROIE + custom forms | 30 | 1× A100 |

---

## Results

| Task | Metric | Score |
|------|--------|-------|
| Text detection | F1 (IoU > 0.5) | 0.918 |
| Text recognition | Word accuracy | 96.4% |
| Field extraction | F1 | 0.940 |
| End-to-end (all stages) | End-to-end F1 | 0.891 |

---

## Ablation Study — Recognition

| Configuration | Word Acc |
|---------------|----------|
| CNN + CTC (no RNN) | 87.3% |
| CNN + LSTM + CTC | 93.1% |
| CNN + BiLSTM + CTC | 95.2% |
| + Synthetic augmentation | 96.4% |

---

## Failure Analysis

- **Handwritten text**: CRNN trained on printed fonts. Handwriting word accuracy drops to ~71%. Requires separate handwriting recognition model.
- **Text on complex backgrounds** (stamps, watermarks): Detection precision drops to 0.81.
- **Very small fonts (<8pt)**: Character regions merge in CRAFT heatmap; word boundaries lost.
- **Non-Latin scripts**: Pipeline is English-only; Arabic/Chinese require retrained models.

---

## Getting Started

```bash
git clone https://github.com/sherifabdelrady/docuai
cd docuai
pip install -r requirements.txt

# Download pretrained models
python scripts/download_weights.py

# Run full pipeline on a document
python pipeline.py --input invoice.jpg --output extracted.json

# Train CRAFT detector
python train_craft.py --config configs/craft_synthtext.yaml

# Train CRNN recognizer
python train_crnn.py --config configs/crnn_mjsynth.yaml

# Fine-tune LayoutLM on custom forms
python train_layoutlm.py --data data/custom_forms/ --epochs 30
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
