import os

import torch


def load_model_weights(model, checkpoint_path, device):
    """
    兼容两种权重格式:
    1. 纯模型权重: model.state_dict()
    2. 完整训练检查点: 包含 model_state_dict 的字典
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint, "full_checkpoint"

    model.load_state_dict(checkpoint)
    return checkpoint, "model_state_dict"


def resolve_resume_checkpoint_path(cfg, latest_checkpoint_path):
    resume_path = str(getattr(cfg, "resume_checkpoint_path", "") or "").strip()
    if resume_path:
        return resume_path

    if getattr(cfg, "resume_training", False):
        return latest_checkpoint_path

    return ""


def load_training_checkpoint(model, optimizer, scaler, checkpoint_path, device, logger):
    """
    加载完整训练状态。

    支持两种情况:
    1. 完整 checkpoint:
       恢复模型、优化器、AMP、epoch、best_val_loss、loss 历史
    2. 纯模型权重:
       只恢复模型参数，从 epoch 0 开始重新训练
    """

    start_epoch = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    if not checkpoint_path:
        return start_epoch, best_val_loss, train_losses, val_losses, False

    if not os.path.exists(checkpoint_path):
        logger.warning(f"Resume checkpoint not found: {checkpoint_path}")
        return start_epoch, best_val_loss, train_losses, val_losses, False

    checkpoint, checkpoint_type = load_model_weights(model, checkpoint_path, device)
    if checkpoint_type == "full_checkpoint":
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state:
            scaler.load_state_dict(scaler_state)

        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        train_losses = list(checkpoint.get("train_losses", []))
        val_losses = list(checkpoint.get("val_losses", []))

        logger.info(f"Resumed full checkpoint from: {checkpoint_path}")
        logger.info(
            f"Resume state: next_epoch={start_epoch + 1}, "
            f"best_val_loss={best_val_loss:.4f}, "
            f"history_len={len(train_losses)}"
        )
        return start_epoch, best_val_loss, train_losses, val_losses, True

    logger.warning(
        "Loaded a model-only checkpoint for resume. "
        "Optimizer, scaler, epoch, and best_val_loss were not restored."
    )
    logger.warning(f"Model-only checkpoint path: {checkpoint_path}")
    return start_epoch, best_val_loss, train_losses, val_losses, True


def save_training_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scaler,
    epoch,
    best_val_loss,
    train_losses,
    val_losses,
    cfg,
):
    scaler_state = scaler.state_dict() if scaler.is_enabled() else None

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler_state,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "model_name": cfg.model_name,
        "experiment_name": getattr(cfg, "experiment_name", ""),
    }
    torch.save(checkpoint, checkpoint_path)
