#!/usr/bin/env python3
"""
Generated PyTorch Lightning training script — Natural Language Processing.
"""

import os
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# __LIGHTNING_IMPORT_BLOCK__

# ──────────────────────────────────────────────────────────────────────────────
# 1. Hyperparameters & Configurations
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = "__DATA_DIR__"
LOG_DIR = "__LOG_DIR__"
EXPERIMENT_NAME = "__EXPERIMENT_NAME__"
LR = __LR__
BATCH_SIZE = __BATCH_SIZE__
NUM_WORKERS = __NUM_WORKERS__
MAX_EPOCHS = __MAX_EPOCHS__
ACCELERATOR = "__ACCELERATOR__"
DEVICES = __DEVICES__
PRECISION = __PRECISION__
LOG_EVERY_N_STEPS = __LOG_EVERY_N_STEPS__
SEED = __SEED__

# NLP Specific Settings
MAX_SEQ_LEN = __MAX_SEQ_LEN__
MODEL_TYPE = "__MODEL_TYPE__"  # "attention" or "seq2seq"

# Simple Tokenizer & Vocab Builder
class SimpleTokenizer:
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<sos>", 3: "<eos>"}
        
    def build_vocab(self, texts):
        words = []
        for text in texts:
            words.extend(self._tokenize(text))
        counter = Counter(words)
        most_common = counter.most_common(self.max_vocab_size - 4)
        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                
    def _tokenize(self, text):
        return re.findall(r"\w+", str(text).lower())
        
    def encode(self, text, max_len):
        tokens = self._tokenize(text)
        ids = [self.word2idx.get(t, 1) for t in tokens[:max_len]]
        padding = [0] * (max_len - len(ids))
        return ids + padding

    def encode_seq2seq(self, text, max_len):
        tokens = self._tokenize(text)
        ids = [2] + [self.word2idx.get(t, 1) for t in tokens[:max_len - 2]] + [3]
        padding = [0] * (max_len - len(ids))
        return ids + padding


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, is_seq2seq=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_seq2seq = is_seq2seq

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.is_seq2seq:
            # For seq2seq, target is also text (shifted version or output text sequence)
            src_ids = torch.tensor(self.tokenizer.encode_seq2seq(text, self.max_len), dtype=torch.long)
            tgt_text = self.labels[idx]
            tgt_ids = torch.tensor(self.tokenizer.encode_seq2seq(tgt_text, self.max_len), dtype=torch.long)
            return src_ids, tgt_ids
        else:
            ids = torch.tensor(self.tokenizer.encode(text, self.max_len), dtype=torch.long)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return ids, label


# ── DataModule ──
class LitDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = SimpleTokenizer()
        
    def setup(self, stage=None):
        # __DATASET_SETUP__
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


# ── Model Architectures ──
# __MODEL_BLOCK__


# ── Logger Setup ─────────────────────────────────────────────────────────────
def build_loggers():
    loggers = [
        # __LOGGER_LINES__
    ]
    return [lg for lg in loggers if lg is not False]


# ── Main Entrypoint & Training Loop ────────────────────────────────────────────
def main():
    L.seed_everything(SEED, workers=True)
    datamodule = LitDataModule()
    datamodule.setup()
    
    vocab_size = len(datamodule.tokenizer.word2idx)
    num_classes = getattr(datamodule, "num_classes", 2)
    
    model = LitModel(vocab_size=vocab_size, num_classes=num_classes)
    loggers = build_loggers()
    
    # __CALLBACK_BLOCK__
    
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        logger=loggers if loggers else None,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
