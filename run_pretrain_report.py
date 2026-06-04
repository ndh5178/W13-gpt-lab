from pathlib import Path
import time

import torch

from src.bpe import BPETokenizer
from src.dataset import create_dataloader
from src.model import GPTModel
from src.train import calc_loss_loader, generate, train_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

train_text = Path("data/nsmc_lm_train.txt").read_text(encoding="utf-8")
val_text = Path("data/nsmc_lm_val.txt").read_text(encoding="utf-8")

vocab_size = 5000
tokenizer_train_chars = 500_000
corpus_size = 1_000_000
val_size = 150_000
context_length = 128
batch_size = 8
num_epochs = 10
learning_rate = 3e-4

tokenizer = BPETokenizer(vocab_size=vocab_size)
vocab_path = Path("data/nsmc_bpe_vocab_5000_500k.json")
ids_cache_path = Path("data/v5000_chars500000_m4740_traine2b0df65adaf_val0bba341ced33_ids.pt")

start = time.time()
if vocab_path.exists():
    tokenizer.load(vocab_path)
    bpe_time = 0.0
    print("loaded vocab:", vocab_path)
else:
    tokenizer.train(train_text[:tokenizer_train_chars])
    bpe_time = time.time() - start
    tokenizer.save(vocab_path)
    print("saved vocab:", vocab_path)

print("BPE time:", bpe_time)

if ids_cache_path.exists():
    ids_cache = torch.load(ids_cache_path, map_location="cpu")
    train_ids = ids_cache["train_ids"]
    val_ids = ids_cache["val_ids"]
    print("loaded ids cache:", ids_cache_path)
    print("ids cache meta:", ids_cache.get("meta", {}))
else:
    train_ids = tokenizer.encode(train_text[:corpus_size])
    val_ids = tokenizer.encode(val_text[:val_size])
    torch.save(
        {
            "meta": {
                "vocab_size": vocab_size,
                "tokenizer_train_chars": tokenizer_train_chars,
                "corpus_size": corpus_size,
                "val_size": val_size,
                "vocab_path": str(vocab_path),
            },
            "train_ids": train_ids,
            "val_ids": val_ids,
        },
        ids_cache_path,
    )
    print("saved ids cache:", ids_cache_path)

train_loader = create_dataloader(
    train_ids,
    context_length=context_length,
    batch_size=batch_size,
    stride=context_length,
    shuffle=True,
    drop_last=True,
    pin_memory=device.type == "cuda",
)

val_loader = create_dataloader(
    val_ids,
    context_length=context_length,
    batch_size=batch_size,
    stride=context_length,
    shuffle=False,
    drop_last=False,
    pin_memory=device.type == "cuda",
)

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
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

num_params = sum(p.numel() for p in model.parameters())
print(f"num_params: {num_params:,}")

start = time.time()
train_losses = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs,
    eval_freq=200,
    eval_iter=20,
    start_context="\uc601\ud654",
    tokenizer=tokenizer,
    ckpt_freq=None,
)
pretrain_time = time.time() - start

checkpoint_path = checkpoint_dir / "pretrain_report_v5000_tok500k_corpus1m_ctx128_emb192_l4_drop02_bias.pt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": num_epochs,
        "global_step": len(train_loader) * num_epochs,
        "config": config,
        "tokenizer_train_chars": tokenizer_train_chars,
        "corpus_size": corpus_size,
        "learning_rate": learning_rate,
        "ids_cache_path": str(ids_cache_path),
        "vocab_path": str(vocab_path),
    },
    checkpoint_path,
)
print("saved checkpoint:", checkpoint_path)

final_train_loss = calc_loss_loader(train_loader, model, device, num_batches=50)
final_val_loss = calc_loss_loader(val_loader, model, device, num_batches=50)

print("train_losses:", train_losses)
print("final_train_loss:", final_train_loss)
print("final_val_loss:", final_val_loss)
print("pretrain_time:", pretrain_time)

start_ids = tokenizer.encode("\uc601\ud654", add_bos_eos=False)
idx = torch.tensor(start_ids, dtype=torch.long).unsqueeze(0).to(device)

out = generate(
    model=model,
    idx=idx,
    max_new_tokens=60,
    context_size=context_length,
    temperature=0.8,
    top_k=40,
    eos_id=tokenizer.get_eos_id(),
)

print("sample:")
print(tokenizer.decode(out[0].tolist(), skip_special=True))
