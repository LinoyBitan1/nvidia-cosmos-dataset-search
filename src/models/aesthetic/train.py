# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Training script for aesthetic predictor."""

import random
from pathlib import Path

import braceexpand
import click
import pytorch_lightning as pl
import torch

from src.models.aesthetic import AestheticPredictor, make_loader


@click.group
def cli() -> None: ...


@cli.command
@click.option("--url", type=str, required=True)
@click.option("--output_path", type=click.Path(), required=True)
@click.option("--val_fraction", type=float, show_default=True, default=0.1)
@click.option("--max_epochs", type=int, show_default=True, default=100)
@click.option("--batches_per_epoch", type=int, show_default=True, default=10_000)
@click.option("--batch_size", type=int, show_default=True, default=256)
@click.option("--num_workers", type=int, show_default=True, default=8)
@click.option("--shuffle_buffer", type=int, show_default=True, default=5000)
@click.option("--embedding_size", type=int, show_default=True, default=768)
@click.option(
    "--probabilistic/--no-probabilistic",
    type=bool,
    is_flag=True,
    show_default=True,
    default=False,
)
@click.option("--lr", type=float, show_default=True, default=1e-3)
@click.option("--name", type=str, show_default=True, default="")
@click.option("--cache_dir", type=str, show_default=True, default="")
def train(
    url: str,
    output_path: Path,
    val_fraction: float,
    max_epochs: int,
    batches_per_epoch: int,
    batch_size: int,
    num_workers: int,
    shuffle_buffer: int,
    embedding_size: int,
    probabilistic: bool,
    lr: float,
    name: str,
    cache_dir: str,
) -> None:
    """Training entrypoint for aesthetic predictor model."""

    urls = list(braceexpand.braceexpand(url))
    random.seed(42)
    random.shuffle(urls)

    split_id = int(len(urls) * val_fraction)
    train_urls, val_urls = urls[split_id:], urls[:split_id]
    print(f"Training on {len(train_urls)} URLs")
    print(f"Validating on {len(val_urls)} URLs")

    cache_path = None if not cache_dir else Path(cache_dir)
    train_cache_dir, val_cache_dir = None, None
    if cache_path:
        train_cache_path = cache_path / "train"
        train_cache_path.mkdir(exist_ok=True)
        train_cache_dir = train_cache_path.as_posix()
        val_cache_path = cache_path / "val"
        val_cache_path.mkdir(exist_ok=True)
        val_cache_dir = val_cache_path.as_posix()

    train_loader = make_loader(
        train_urls,
        mode="train",
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_buffer=shuffle_buffer,
        cache_dir=train_cache_dir,
    )

    val_loader = make_loader(
        val_urls,
        mode="val",
        batch_size=batch_size,
        num_workers=num_workers,
        cache_dir=val_cache_dir,
    )

    model = AestheticPredictor(embedding_size, probabilistic=probabilistic)

    def configure_optimizer(cls: AestheticPredictor):
        """Over-ride optimizer"""
        optimizer = torch.optim.Adam(cls.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=5, min_lr=1e-5
        )
        return [optimizer], [scheduler]

    model.configure_optimizer = configure_optimizer

    save_root = Path(output_path) / name
    save_root.mkdir(exist_ok=True, parents=True)

    trainer = pl.Trainer(
        devices=torch.cuda.device_count(),
        accelerator="gpu",
        strategy="ddp",
        max_epochs=max_epochs,
        limit_train_batches=batches_per_epoch,
        logger=pl.loggers.TensorBoardLogger(save_root.as_posix()),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=save_root.as_posix(), monitor="val_loss"
            ),
            pl.callbacks.LearningRateMonitor("epoch"),
        ],
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    train()
