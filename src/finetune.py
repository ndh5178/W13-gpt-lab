# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 유틸리티."""

import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def _read_nsmc_tsv(path: str | Path) -> list[dict]:
    """NSMC TSV 파일을 {text, label} 딕셔너리 리스트로 읽습니다."""
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = (row.get("document") or "").strip()
            label = row.get("label")
            if not text or label is None:
                continue
            try:
                label_id = int(label)
            except ValueError:
                continue
            if label_id not in (0, 1):
                continue
            rows.append({"text": text, "label": label_id})
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """딕셔너리 리스트를 JSONL 파일로 저장합니다."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    train_rows = _read_nsmc_tsv(train_tsv_path)
    test_rows = _read_nsmc_tsv(test_tsv_path) if test_tsv_path is not None else []

    rng = random.Random(seed)
    rng.shuffle(train_rows)

    if val_ratio <= 0 or len(train_rows) <= 1:
        val_size = 0
    else:
        val_size = int(len(train_rows) * val_ratio)
        val_size = max(1, min(val_size, len(train_rows) - 1))

    val_rows = train_rows[:val_size]
    train_rows = train_rows[val_size:]

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_path / "nsmc_sentiment_train.jsonl", train_rows)
        _write_jsonl(output_path / "nsmc_sentiment_val.jsonl", val_rows)
        _write_jsonl(output_path / "nsmc_sentiment_test.jsonl", test_rows)

    return train_rows, val_rows, test_rows


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

        input_rows = []
        labels = []

        for item in data:
            token_ids = tokenizer.encode(item["text"], add_bos_eos=True)
            token_ids = token_ids[: self.max_length]

            if len(token_ids) < self.max_length:
                token_ids = token_ids + [self.pad_id] * (self.max_length - len(token_ids))

            input_rows.append(token_ids)
            labels.append(int(item["label"]))

        self.input_ids = torch.tensor(input_rows, dtype=torch.long)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """미리 encode해 둔 input_ids와 label을 반환합니다."""
        return self.input_ids[idx], self.labels[idx]

class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        self.pad_id = 0
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        hidden = self.gpt.embedding(input_ids)
        for block in self.gpt.blocks:
            hidden = block(hidden, causal_mask=True)
        hidden = self.gpt.final_norm(hidden)

        non_pad_counts = (input_ids != self.pad_id).sum(dim=1)
        last_token_idx = torch.clamp(non_pad_counts - 1, min=0)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        pooled = hidden[batch_idx, last_token_idx]

        logits = self.classifier(self.dropout(pooled))

        if labels is None:
            return logits

        labels = labels.long()
        loss = F.cross_entropy(logits, labels)
        return loss, logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    model.to(device)
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        return float("nan"), 0.0

    return total_loss / total_examples, total_correct / total_examples


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    model.to(device)
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device).long()

            loss, logits = model(input_ids, labels=labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_examples += batch_size

    if was_training:
        model.train()

    if total_examples == 0:
        return float("nan"), 0.0

    return total_loss / total_examples, total_correct / total_examples