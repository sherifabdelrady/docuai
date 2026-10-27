"""
tests/test_ocr.py — DocuAI OCR unit tests
Runs without any trained model weights.
"""
import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from ocr import (
    VOCAB, BLANK_IDX, NUM_CLASSES, _ASCII, _ARABIC,
    CRNN, ctc_decode, preprocess_image,
)


class TestVocab:
    def test_arabic_present(self):
        """Arabic characters must be in VOCAB — was missing before fix."""
        assert "\u0627" in VOCAB, "Arabic letter alef (ا) not in VOCAB"
        assert "\u064A" in VOCAB, "Arabic letter ya (ي) not in VOCAB"

    def test_ascii_present(self):
        assert "A" in VOCAB
        assert "0" in VOCAB
        assert " " in VOCAB

    def test_no_duplicates(self):
        assert len(VOCAB) == len(set(VOCAB)), "VOCAB contains duplicate characters"

    def test_blank_idx_correct(self):
        assert BLANK_IDX == len(VOCAB)
        assert NUM_CLASSES == len(VOCAB) + 1

    def test_arabic_indic_digits(self):
        """Arabic-Indic digits ٠١٢٣٤٥٦٧٨٩ must be in VOCAB."""
        assert "\u0660" in VOCAB, "Arabic-Indic zero not in VOCAB"
        assert "\u0669" in VOCAB, "Arabic-Indic nine not in VOCAB"


class TestCTCDecode:
    def test_single_char(self):
        """Greedy decode of a single time-step with one dominant class."""
        idx = VOCAB.index("A")
        logits = torch.zeros(1, 1, NUM_CLASSES)
        logits[0, 0, idx] = 10.0
        texts = ctc_decode(logits)
        assert texts[0] == "A"

    def test_repeated_collapse(self):
        """CTC collapses repeated indices to a single char."""
        idx = VOCAB.index("B")
        logits = torch.zeros(1, 5, NUM_CLASSES)
        logits[0, :, idx] = 10.0   # 5 repeated B's → should decode to "B"
        texts = ctc_decode(logits)
        assert texts[0] == "B"

    def test_blank_removed(self):
        """Blank token must not appear in decoded output."""
        logits = torch.zeros(1, 3, NUM_CLASSES)
        logits[0, :, BLANK_IDX] = 10.0   # all blanks → empty string
        texts = ctc_decode(logits)
        assert texts[0] == ""

    def test_spaces_preserved(self):
        """
        REGRESSION: original code did .replace(' ', '') — spaces were lost.
        Decoded spaces must now be preserved.
        """
        space_idx = VOCAB.index(" ")
        a_idx = VOCAB.index("A")
        b_idx = VOCAB.index("B")
        # Sequence: A, blank, space, blank, B
        logits = torch.zeros(1, 5, NUM_CLASSES)
        logits[0, 0, a_idx]     = 10.0
        logits[0, 1, BLANK_IDX] = 10.0
        logits[0, 2, space_idx] = 10.0
        logits[0, 3, BLANK_IDX] = 10.0
        logits[0, 4, b_idx]     = 10.0
        texts = ctc_decode(logits)
        assert " " in texts[0], f"Space was stripped — got: {texts[0]!r}"
        assert texts[0] == "A B"

    def test_batch_decode(self):
        """Decode works correctly for a batch > 1."""
        logits = torch.zeros(3, 4, NUM_CLASSES)
        for b, char in enumerate(["X", "Y", "Z"]):
            logits[b, :, VOCAB.index(char)] = 10.0
        texts = ctc_decode(logits)
        assert texts == ["X", "Y", "Z"]

    def test_2d_input_promoted(self):
        """A (T, C) tensor (no batch dim) should be handled without error."""
        logits = torch.zeros(4, NUM_CLASSES)
        logits[:, VOCAB.index("C")] = 10.0
        texts = ctc_decode(logits)
        assert texts[0] == "C"

    def test_arabic_decode(self):
        """Arabic characters in VOCAB must be decodable."""
        alef = "\u0627"
        idx  = VOCAB.index(alef)
        logits = torch.zeros(1, 1, NUM_CLASSES)
        logits[0, 0, idx] = 10.0
        texts = ctc_decode(logits)
        assert texts[0] == alef


class TestPreprocessImage:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            preprocess_image(str(tmp_path / "nonexistent.png"))

    def test_valid_image(self, tmp_path):
        import cv2, numpy as np
        p = tmp_path / "test.png"
        cv2.imwrite(str(p), np.ones((64, 128), dtype=np.uint8) * 128)
        tensor = preprocess_image(str(p), height=32)
        assert tensor.shape[0] == 1   # batch
        assert tensor.shape[1] == 1   # channel (grayscale)
        assert tensor.shape[2] == 32  # height


class TestCRNN:
    def test_output_shape(self):
        model = CRNN(NUM_CLASSES).eval()
        dummy = torch.zeros(1, 1, 32, 128)
        with torch.no_grad():
            out = model(dummy)
        assert out.dim() == 3
        assert out.shape[0] == 1          # batch
        assert out.shape[2] == NUM_CLASSES

    def test_num_params_reasonable(self):
        model = CRNN(NUM_CLASSES)
        n = sum(p.numel() for p in model.parameters()) / 1e6
        assert 5.0 <= n <= 20.0, f"Unexpected param count: {n:.1f}M"
