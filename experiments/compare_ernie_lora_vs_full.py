"""Compare LoRA fine-tuning vs full fine-tuning on ERNIE 3.0 Medium."""
import argparse
import dataclasses
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import aistudio_sdk.hub as aistudio_hub
except Exception:  # pragma: no cover - optional dependency
    aistudio_hub = None

if aistudio_hub is not None and not hasattr(aistudio_hub, "download"):
    def _unsupported_download(*args, **kwargs):  # pragma: no cover - environment guard
        raise RuntimeError(
            "aistudio_sdk.hub.download is unavailable in this environment. "
            "Set an AI Studio token or install a compatible SDK to enable it."
        )

    aistudio_hub.download = _unsupported_download
    sys.modules["aistudio_sdk.hub"] = aistudio_hub

try:
    import paddle
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "This script requires PaddlePaddle. Install it with `pip install paddlepaddle paddlenlp`."
    ) from exc

from paddlenlp.data import Pad, Stack, Tuple as TupleBatch
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.peft import LoRAConfig, LoRAModel
from datasets import load_dataset as load_hf_dataset


@dataclass
class ExperimentConfig:
    """Configuration controlling the fine-tuning comparison."""

    model_name: str = "ernie-3.0-medium-zh"
    max_seq_length: int = 128
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 2
    max_train_samples: int = 1024
    max_eval_samples: int = 512
    seed: int = 42
    output_dir: str = "reports"
    report_file: str = "ernie_lora_vs_full_report.md"
    save_artifacts: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1


@dataclass
class RunResult:
    """Summary of a single fine-tuning strategy."""

    strategy: str
    train_loss: float
    eval_loss: float
    test_loss: float
    eval_accuracy: float
    test_accuracy: float
    training_time: float
    total_params: int
    trainable_params: int
    config: Dict[str, float]
    artifact_dir: str

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


class ListDataset(paddle.io.Dataset):
    """Simple paddle dataset backed by an in-memory list."""

    def __init__(self, data: List[Dict[str, np.ndarray]]):
        super().__init__()
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self._data[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def select_subset(dataset, max_samples: int, seed: int):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def convert_example(example: Dict[str, str], tokenizer, max_seq_length: int) -> Dict[str, np.ndarray]:
    encoded = tokenizer(
        example["sentence"],
        max_length=max_seq_length,
        truncation=True,
        return_attention_mask=True,
    )
    encoded["labels"] = np.array([int(example["label"])], dtype="int64")
    return encoded


def create_dataloader(dataset, batch_size: int, tokenizer, shuffle: bool = False):
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
    pad_token_type_id = getattr(tokenizer, "pad_token_type_id", 0) or 0
    batchify_fn = TupleBatch(
        Pad(axis=0, pad_val=pad_token_id, dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=pad_token_type_id, dtype="int64"),  # token_type_ids
        Pad(axis=0, pad_val=0, dtype="int64"),  # attention_mask
        Stack(dtype="int64"),  # labels
    )

    def collate_fn(samples: List[Dict[str, np.ndarray]]):
        input_ids = [sample["input_ids"] for sample in samples]
        token_type_ids = [sample["token_type_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        labels = [sample["labels"] for sample in samples]
        input_ids, token_type_ids, attention_mask, labels = batchify_fn(
            list(zip(input_ids, token_type_ids, attention_mask, labels))
        )
        return input_ids, token_type_ids, attention_mask, labels

    sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    return paddle.io.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, return_list=True)


def count_parameters(model: paddle.nn.Layer) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        numel = int(np.prod(param.shape))
        total += numel
        if not getattr(param, "stop_gradient", False):
            trainable += numel
    return total, trainable


def evaluate(model: paddle.nn.Layer, dataloader, loss_fn) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with paddle.no_grad():
        for batch in dataloader:
            input_ids, token_type_ids, attention_mask, labels = batch
            labels = labels.squeeze(-1)
            logits = model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            loss = loss_fn(logits, labels)
            total_loss += float(loss.numpy())
            preds = paddle.argmax(logits, axis=-1)
            total_correct += int((preds == labels).astype("int64").sum().numpy())
            total_samples += labels.shape[0]
    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def train_epoch(
    model: paddle.nn.Layer,
    dataloader,
    optimizer: paddle.optimizer.Optimizer,
    loss_fn: paddle.nn.Layer,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        input_ids, token_type_ids, attention_mask, labels = batch
        labels = labels.squeeze(-1)
        logits = model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        total_loss += float(loss.numpy()) * labels.shape[0]
        total_samples += labels.shape[0]
    return total_loss / max(total_samples, 1)


def instantiate_model(config: ExperimentConfig, strategy: str, num_classes: int) -> paddle.nn.Layer:
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_classes=num_classes,
    )
    if strategy == "lora":
        lora_config = LoRAConfig(
            target_modules=[r".*q_proj.*", r".*k_proj.*", r".*v_proj.*", r".*out_proj.*"],
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
        model = LoRAModel(base_model, lora_config)
        model.mark_only_lora_as_trainable()
        return model
    if strategy == "full_finetune":
        return base_model
    raise ValueError(f"Unknown strategy: {strategy}")


def run_strategy(
    strategy: str,
    config: ExperimentConfig,
    datasets,
    tokenizer,
    device: str,
) -> RunResult:
    paddle.set_device(device)
    train_dataset, eval_dataset, test_dataset = datasets
    model = instantiate_model(config, strategy, num_classes=15)
    total_params, trainable_params = count_parameters(model)

    train_loader = create_dataloader(train_dataset, config.batch_size, tokenizer, shuffle=True)
    eval_loader = create_dataloader(eval_dataset, config.batch_size, tokenizer, shuffle=False)
    test_loader = create_dataloader(test_dataset, config.batch_size, tokenizer, shuffle=False)

    loss_fn = paddle.nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.AdamW(
        learning_rate=config.learning_rate,
        parameters=[p for p in model.parameters() if not getattr(p, "stop_gradient", False)],
        weight_decay=config.weight_decay,
    )

    start_time = time.time()
    for _ in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
    training_time = time.time() - start_time

    eval_loss, eval_accuracy = evaluate(model, eval_loader, loss_fn)
    test_loss, test_accuracy = evaluate(model, test_loader, loss_fn)

    artifact_dir = ""
    if config.save_artifacts:
        artifact_dir = os.path.join(config.output_dir, f"{strategy}_artifacts")
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(os.path.join(artifact_dir, "model"))
        tokenizer.save_pretrained(os.path.join(artifact_dir, "tokenizer"))

    return RunResult(
        strategy=strategy,
        train_loss=float(train_loss),
        eval_loss=float(eval_loss),
        test_loss=float(test_loss),
        eval_accuracy=float(eval_accuracy),
        test_accuracy=float(test_accuracy),
        training_time=float(training_time),
        total_params=total_params,
        trainable_params=trainable_params,
        config={
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
        },
        artifact_dir=artifact_dir,
    )


def prepare_datasets(tokenizer, config: ExperimentConfig):
    train_ds = load_hf_dataset("clue", "tnews", split="train")
    val_ds = load_hf_dataset("clue", "tnews", split="validation")

    train_ds = select_subset(train_ds, config.max_train_samples, config.seed)
    val_shuffled = val_ds.shuffle(seed=config.seed)
    dev_count = min(config.max_eval_samples, len(val_shuffled))
    test_count = min(config.max_eval_samples, max(len(val_shuffled) - dev_count, 0))
    if test_count == 0:
        test_count = dev_count

    dev_ds = val_shuffled.select(range(dev_count))
    test_start = dev_count
    test_end = min(len(val_shuffled), dev_count + test_count)
    test_ds = val_shuffled.select(range(test_start, test_end))

    def preprocess(example):
        return convert_example(example, tokenizer, config.max_seq_length)

    train_processed = [preprocess(example) for example in train_ds]
    dev_processed = [preprocess(example) for example in dev_ds]
    test_processed = [preprocess(example) for example in test_ds]
    return ListDataset(train_processed), ListDataset(dev_processed), ListDataset(test_processed)


def generate_report(results: List[RunResult], config: ExperimentConfig) -> str:
    report_dir = Path(config.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / config.report_file

    lines = [
        "# ERNIE 3.0 Medium: LoRA vs Full Fine-Tuning",
        "",
        "This report summarizes a small-scale comparison between low-rank adaptation",
        "(LoRA) and full-parameter fine-tuning on the Chinese TNEWS text classification",
        "dataset using the ERNIE 3.0 Medium model.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(dataclasses.asdict(config), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Results",
        "",
        "| Strategy | Trainable Params | Total Params | Training Time (s) | Dev Accuracy | Test Accuracy | Dev Loss | Test Loss |",
        "|----------|-----------------:|-------------:|------------------:|-------------:|--------------:|---------:|----------:|",
    ]

    for result in results:
        lines.append(
            f"| {result.strategy} | {result.trainable_params:,} | {result.total_params:,} | "
            f"{result.training_time:.2f} | {result.eval_accuracy:.4f} | {result.test_accuracy:.4f} | "
            f"{result.eval_loss:.4f} | {result.test_loss:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Observations",
            "",
            "- LoRA confines training to a small subset of parameters, reducing the number of",
            "  trainable weights compared with full fine-tuning while aiming to maintain",
            "  competitive accuracy.",
            "- The training times in this lightweight run primarily reflect CPU-bound",
            "  execution; expect larger speedups on GPU hardware when leveraging LoRA's",
            "  memory savings to increase batch size.",
            "",
            "## Artifacts",
            "",
        ]
    )

    for result in results:
        if result.artifact_dir:
            lines.append(f"- **{result.strategy}** artifacts saved under `{result.artifact_dir}`")

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return str(report_path)


def run(config: ExperimentConfig) -> None:
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    datasets = prepare_datasets(tokenizer, config)
    device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
    results: List[RunResult] = []
    for strategy in ("lora", "full_finetune"):
        print(f"Running strategy: {strategy}")
        result = run_strategy(strategy, config, datasets, tokenizer, device)
        results.append(result)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    report_path = generate_report(results, config)
    print(f"Report written to {report_path}")


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Compare LoRA with full fine-tuning on ERNIE 3.0 Medium")
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-eval-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout probability")
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--report-file", type=str, default="ernie_lora_vs_full_report.md")
    args = parser.parse_args()
    return ExperimentConfig(
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        save_artifacts=not args.no_artifacts,
        output_dir=args.output_dir,
        report_file=args.report_file,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )


if __name__ == "__main__":
    config = parse_args()
    run(config)
