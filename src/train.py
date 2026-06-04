# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    """TODO: 한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    loss, _ = model(input_batch, targets=target_batch)
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    total_loss = 0.0
    n = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if num_batches is not None and i >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            n += 1
    
    if was_training:
        model.train()
    return total_loss / n if n > 0 else 0.0


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("global_step", 0)


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """TODO: temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)          # (B, T, vocab_size)
            if isinstance(logits, tuple):
                logits = logits[1]
            logits = logits[:, -1, :]         # 마지막 위치만 (B, vocab_size)

            # top-k: 상위 k개 외 나머지를 -inf로 마스킹
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                top_values, _ = torch.topk(logits, top_k, dim=-1)
                threshold = top_values[:, -1, None]  # k번째 값
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            if temperature == 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # temperature 적용 후 확률 분포로 변환
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)

                # 확률 분포에서 샘플링
                next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)

            # eos 토큰이 나오면 조기 종료
            if eos_id is not None and (next_id == eos_id).all():
                break

    if was_training:
        model.train()

    return idx


def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> None:
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    was_training = model.training
    model.eval()
    encoded = tokenizer.encode(start_context, add_bos_eos=False)
    idx = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    eos_id = tokenizer.get_eos_id() if hasattr(tokenizer, "get_eos_id") else None
    out = generate(model, idx, max_new_tokens, context_size, temperature, top_k, eos_id=eos_id)
    decoded = tokenizer.decode(out[0].tolist(), skip_special=True)
    print(decoded)
    if was_training:
        model.train()


def train_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
) -> list[float]:
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    train_losses = []
    model.to(device)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            # eval_freq 스텝마다 검증
            if eval_freq > 0 and global_step % eval_freq == 0:
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                print(f"Epoch {epoch+1} | Step {global_step} | val_loss: {val_loss:.4f}")

        avg_train_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        train_losses.append(avg_train_loss)

        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Epoch {epoch+1} done | train_loss: {avg_train_loss:.4f} | val_loss: {val_loss:.4f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

        if ckpt_freq is not None and (epoch + 1) % ckpt_freq == 0:
            save_checkpoint(model, optimizer, epoch + 1, global_step, f"ckpt_epoch{epoch+1}.pt")

    return train_losses


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
