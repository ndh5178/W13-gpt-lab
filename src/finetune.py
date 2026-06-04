# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    TODO: NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    def read_tsv(path):
        rows = []
        with open(path, encoding="utf-8") as f:
            next(f)  # 헤더 스킵
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                _, document, label = parts
                if not document.strip():  # 빈 리뷰 제거
                    continue
                rows.append({"text": document.strip(), "label": int(label)})
        return rows

    all_train = read_tsv(train_tsv_path)

    rng = random.Random(seed)
    rng.shuffle(all_train)

    n_val = max(1, int(len(all_train) * val_ratio))
    val_data = all_train[:n_val]
    train_data = all_train[n_val:]

    test_data = read_tsv(test_tsv_path) if test_tsv_path is not None else []

    return train_data, val_data, test_data


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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """TODO: text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        item = self.data[idx]
        ids = self.tokenizer.encode(item["text"], add_bos_eos=False)

        # max_length보다 길면 자르기
        ids = ids[: self.max_length]

        # max_length보다 짧으면 pad_id로 패딩
        pad_len = self.max_length - len(ids)
        ids = ids + [self.pad_id] * pad_len

        input_ids = torch.tensor(ids, dtype=torch.long)
        return input_ids, item["label"]


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
        pad_id: int = 0,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        self.pad_id = pad_id
        # TODO: dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        emb_dim = gpt_model.config["emb_dim"]
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(emb_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        # embedding → blocks → norm 까지만 실행 (lm_head 제외)
        x = self.gpt.embedding(input_ids)
        for block in self.gpt.blocks:
            x = block(x)
        x = self.gpt.norm(x)

        # 마지막 non-pad 토큰 위치의 벡터를 문장 대표 벡터로 사용
        non_pad = input_ids.ne(self.pad_id)
        last_token_idx = non_pad.long().sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        x = x[batch_idx, last_token_idx]           # (B, emb_dim)
        x = self.dropout(x)
        logits = self.classifier(x)    # (B, num_labels)

        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return loss, logits

        return logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    try:
        with torch.no_grad():
            for input_ids, labels in data_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                loss, logits = model(input_ids, labels=labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
    finally:
        if was_training:
            model.train()

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy
