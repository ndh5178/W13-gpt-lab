from pathlib import Path
import hashlib
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.bpe import BPETokenizer
from src.finetune import (
    GPTForSequenceClassification,
    evaluate_sentiment,
    make_sentiment_dataset,
    train_epoch_sentiment,
)
from src.model import GPTModel
from src.train import load_checkpoint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

vocab_size = 5000
context_length = 128
num_epochs = 8
batch_size = 32
learning_rate = 1e-4
classifier_drop_rate = 0.2
vocab_path = Path("data/nsmc_bpe_vocab_5000_500k.json")
checkpoint_path = Path("checkpoints/pretrain_report_v5000_tok500k_corpus1m_ctx128_emb192_l4_drop02_bias.pt")
train_tsv_path = Path("data/ratings_train.txt")
test_tsv_path = Path("data/ratings_test.txt")
val_ratio = 0.1
split_seed = 42
sentiment_cache_path = Path(f"data/nsmc_sentiment_cache_v{vocab_size}_len{context_length}_seed{split_seed}.pt")


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_sentiment_split(
    name: str,
    data: list[dict],
    tokenizer: BPETokenizer,
    max_length: int,
    pad_id: int,
    log_every: int = 5000,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.full((len(data), max_length), pad_id, dtype=torch.long)
    labels = torch.empty(len(data), dtype=torch.long)
    start = time.time()

    for row_idx, item in enumerate(data):
        token_ids = tokenizer.encode(item["text"], add_bos_eos=True)
        token_ids = token_ids[:max_length]
        input_ids[row_idx, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
        labels[row_idx] = int(item["label"])

        done = row_idx + 1
        if done % log_every == 0 or done == len(data):
            elapsed = time.time() - start
            print(f"{name} cache encode: {done}/{len(data)} ({elapsed:.1f}s)")

    return input_ids, labels

if not vocab_path.exists():
    raise FileNotFoundError(f"먼저 run_pretrain_report.py를 실행해서 vocab을 만드세요: {vocab_path}")

if not checkpoint_path.exists():
    raise FileNotFoundError(f"먼저 run_pretrain_report.py를 실행해서 checkpoint를 만드세요: {checkpoint_path}")

tokenizer = BPETokenizer(vocab_size=vocab_size)
tokenizer.load(vocab_path)

config = {
    "vocab_size": vocab_size,
    "context_length": context_length,
    "emb_dim": 192,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.2,
    "qkv_bias": True,
}

model = GPTModel(config).to(device)
load_checkpoint(model, None, str(checkpoint_path), device)

expected_cache_meta = {
    "vocab_size": vocab_size,
    "context_length": context_length,
    "vocab_path": str(vocab_path),
    "vocab_sha1": file_sha1(vocab_path),
    "pad_id": tokenizer.get_pad_id(),
    "train_tsv_sha1": file_sha1(train_tsv_path),
    "test_tsv_sha1": file_sha1(test_tsv_path),
    "val_ratio": val_ratio,
    "seed": split_seed,
}

cache = None
if sentiment_cache_path.exists():
    loaded_cache = torch.load(sentiment_cache_path, map_location="cpu")
    if loaded_cache.get("meta") == expected_cache_meta:
        cache = loaded_cache
        print("loaded sentiment cache:", sentiment_cache_path)
    else:
        print("ignored stale sentiment cache:", sentiment_cache_path)

if cache is None:
    train_data, val_data, test_data = make_sentiment_dataset(
        train_tsv_path=train_tsv_path,
        test_tsv_path=test_tsv_path,
        val_ratio=val_ratio,
        seed=split_seed,
        output_dir="data",
    )

    pad_id = tokenizer.get_pad_id()
    train_input_ids, train_labels = encode_sentiment_split(
        "train",
        train_data,
        tokenizer,
        context_length,
        pad_id,
    )
    val_input_ids, val_labels = encode_sentiment_split(
        "val",
        val_data,
        tokenizer,
        context_length,
        pad_id,
    )
    test_input_ids, test_labels = encode_sentiment_split(
        "test",
        test_data,
        tokenizer,
        context_length,
        pad_id,
    )

    cache = {
        "meta": expected_cache_meta,
        "train_input_ids": train_input_ids,
        "train_labels": train_labels,
        "val_input_ids": val_input_ids,
        "val_labels": val_labels,
        "test_input_ids": test_input_ids,
        "test_labels": test_labels,
    }
    torch.save(cache, sentiment_cache_path)
    print("saved sentiment cache:", sentiment_cache_path)

train_ds = TensorDataset(cache["train_input_ids"], cache["train_labels"])
val_ds = TensorDataset(cache["val_input_ids"], cache["val_labels"])
test_ds = TensorDataset(cache["test_input_ids"], cache["test_labels"])

pin_memory = device.type == "cuda"
train_loader_cls = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory)
val_loader_cls = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
test_loader_cls = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

clf_model = GPTForSequenceClassification(
    gpt_model=model,
    num_labels=2,
    drop_rate=classifier_drop_rate,
).to(device)

optimizer_cls = torch.optim.AdamW(clf_model.parameters(), lr=learning_rate)

start = time.time()

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch_sentiment(
        clf_model,
        train_loader_cls,
        optimizer_cls,
        device,
    )

    val_loss, val_acc = evaluate_sentiment(
        clf_model,
        val_loader_cls,
        device,
    )

    print(
        f"epoch {epoch + 1} | "
        f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
        f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
    )

test_loss, test_acc = evaluate_sentiment(
    clf_model,
    test_loader_cls,
    device,
)

finetune_time = time.time() - start

print("test_loss:", test_loss)
print("test_acc:", test_acc)
print("finetune_time:", finetune_time)
