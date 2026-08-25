#!/usr/bin/env python3
"""train_autoencoder.py - Autoencoder training for unsupervised representation learning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from config import ExperimentConfig, get_paths, get_data_path
from data import load_region_timeseries, ClimateNormalizer, ClimateDataset
from models import ConvLSTMAutoencoder
from training import AutoencoderTrainer
from utils import set_reproducible_seeds
from utils.logger import Logger


def main():
    """
    Main entry point for autoencoder training.

    The autoencoder learns a compressed representation of climate data
    that can be used for transfer learning with the predictor model.
    """
    # =====================================================================
    # INITIALIZATION
    # =====================================================================

    config = ExperimentConfig()
    logger = Logger()
    set_reproducible_seeds(config.random_seed)

    paths = get_paths(config.region)

    autoencoder_path = paths["autoencoder_dir"] / "model.pth"
    normalizer_path = paths["autoencoder_dir"] / "normalizer.json"

    # Check if autoencoder already exists
    if autoencoder_path.exists() and not config.autoencoder.train:
        logger.info(f"✅ Autoencoder already exists: {autoencoder_path}")
        logger.info("   Use --force or set autoencoder.train=True to retrain")
        return

    # =====================================================================
    # HEADER
    # =====================================================================

    logger.header(f"🧠 AUTOENCODER TRAINING - Region: {config.region}")

    # =====================================================================
    # LOAD DATA
    # =====================================================================

    base_data_path = get_data_path()
    out = load_region_timeseries(base_data_path, config)

    # ✅ CORREÇÃO: Usar apenas dados de treino (até train_end)
    # Evita vazamento de dados da validação para o autoencoder
    split_info_gs = config.split.get_end_indices_gs()
    time_idx = np.array([y * 12 + (m - 1) for y, m in zip(out["years"], out["months"])])

    # ✅ Mudança: usar train_end em vez de val_end
    train_mask = time_idx <= split_info_gs["train_end"]
    data_trainval = out["data"][train_mask]
    months_trainval = out["months"][train_mask]

    p = config.autoencoder.p

    # Display data information
    logger.info(f"📊 Available data: {len(data_trainval)} months")
    logger.info(
        f"   Period: {months_trainval[0]}/{out['years'][train_mask][0]} "
        f"to {months_trainval[-1]}/{out['years'][train_mask][-1]}"
    )
    logger.info(f"   Sequence length (p): {p}")

    # =====================================================================
    # VALIDATE MASK
    # =====================================================================

    if out["valid_mask"] is not None:
        total_pixels = out["valid_mask"].size
        valid_pixels = out["valid_mask"].sum()
        invalid_pixels = total_pixels - valid_pixels

        logger.info(f"\n📊 Validity Mask:")
        logger.info(f"   Total pixels: {total_pixels:,}")
        logger.info(f"   Valid: {valid_pixels:,} ({valid_pixels/total_pixels:.1%})")
        logger.info(f"   Invalid: {invalid_pixels:,} ({invalid_pixels/total_pixels:.1%})")

        # Check for non-drought pixels (value 0) in the mask
        zero_pixels = (data_trainval[0] == 0).all(axis=-1)
        zero_valid = zero_pixels & out["valid_mask"]

        if zero_valid.sum() == 0 and zero_pixels.sum() > 0:
            logger.warning(
                "\n⚠️  WARNING: No non-drought pixels are considered valid!"
            )
            logger.warning(
                "   The autoencoder will NEVER see non-drought examples."
            )
            logger.warning(
                "   This may affect the quality of learned representations."
            )

    # =====================================================================
    # TRAIN/VALIDATION SPLIT
    # =====================================================================

    train_size = int(len(data_trainval) * 0.8)
    val_size = len(data_trainval) - train_size

    logger.info(f"\n📊 Data Split:")
    logger.info(f"   Total: {len(data_trainval)} months")
    logger.info(f"   Training: {train_size} months (80%)")
    logger.info(f"   Validation: {val_size} months (20%)")

    if val_size <= p:
        logger.warning(f"   ⚠️  Validation too small ({val_size} months <= p={p})")
        logger.warning("   Using training data only for validation (fallback)")

        data_train = data_trainval
        months_train = months_trainval
        data_val = data_train[-min(12, len(data_train) // 4):]
        months_val = months_train[-min(12, len(months_train) // 4):]
    else:
        data_train = data_trainval[:train_size]
        data_val = data_trainval[train_size:]
        months_train = months_trainval[:train_size]
        months_val = months_trainval[train_size:]

    logger.info(f"\n📊 After Split:")
    logger.info(f"   Training: {len(data_train)} months → {max(0, len(data_train) - p)} samples")
    logger.info(f"   Validation: {len(data_val)} months → {max(0, len(data_val) - p)} samples")

    # =====================================================================
    # NORMALIZATION
    # =====================================================================

    logger.info(f"\n📊 Normalizing data...")

    normalizer = ClimateNormalizer(config.data.bands)
    data_train_norm = normalizer.fit_transform(data_train, months_train, out["valid_mask"])
    data_val_norm = normalizer.transform(data_val, months_val, out["valid_mask"])

    normalizer.save(normalizer_path)
    logger.success(f"✅ Normalizer saved to: {normalizer_path}")

    # =====================================================================
    # CREATE DATASETS
    # =====================================================================

    logger.info(f"\n📊 Creating datasets...")
    logger.info(f"\n📊 Using anomalies: {config.autoencoder.use_anomalies}")

    train_ds = ClimateDataset(
        data=np.nan_to_num(data_train_norm, nan=0.0),
        spi=None,
        months=months_train,
        p=p,
        q=0,
        spi_threshold=None,
        valid_mask=out["valid_mask"],
        mode="autoencoder",
        temporal_decay=False,
        seed=config.random_seed,
        verbose=True,
    )

    val_ds = None
    if len(data_val) > p:
        val_ds = ClimateDataset(
            data=np.nan_to_num(data_val_norm, nan=0.0),
            spi=None,
            months=months_val,
            p=p,
            q=0,
            spi_threshold=None,
            valid_mask=out["valid_mask"],
            mode="autoencoder",
            temporal_decay=False,
            seed=config.random_seed,
            verbose=False,
        )
        logger.success(f"✅ Validation dataset: {len(val_ds)} samples")
    else:
        logger.warning("⚠️  No validation dataset (insufficient data)")

    # =====================================================================
    # CREATE MODEL
    # =====================================================================

    logger.info(f"\n📊 Creating autoencoder model...")

    model_config = {
        "input_dim": config.num_bands,
        "hidden_dims": config.model.hidden_dims,
        "kernel_size": config.model.kernel_size,
        "use_attention": config.model.use_attention,
        "output_dim": config.num_bands,
        "attention_dropout": getattr(config.model, 'attention_dropout', 0.3),
        "reconstruct_full_sequence": getattr(
            config.autoencoder, 'reconstruct_full_sequence', True
        ),
        "sequence_length": getattr(config.autoencoder, 'sequence_length', 12),
        "decay_rate": getattr(config.autoencoder, 'decay_rate', 0.6),
    }

    model = ConvLSTMAutoencoder(model_config)

    # Print model architecture summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"   Model architecture:")
    logger.info(f"   - Hidden dimensions: {config.model.hidden_dims}")
    logger.info(f"   - Kernel size: {config.model.kernel_size}")
    logger.info(f"   - Use attention: {config.model.use_attention}")
    logger.info(f"   - Total parameters: {total_params:,}")
    logger.info(f"   - Trainable parameters: {trainable_params:,}")

    # Variable weights for weighted loss
    variable_weights = torch.tensor(config.autoencoder.variable_weights)
    logger.info(f"   - Variable weights: {variable_weights.tolist()}")

    # =====================================================================
    # TRAINING
    # =====================================================================

    logger.info(f"\n🚀 Starting training...")

    trainer = AutoencoderTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        save_path=autoencoder_path,
        variable_weights=variable_weights,
    )

    best_loss = trainer.train()

    # =====================================================================
    # COMPLETION
    # =====================================================================

    logger.header("✅ TRAINING COMPLETED")

    logger.info(f"📊 Results:")
    logger.info(f"   Best validation loss: {best_loss:.4f}")
    logger.info(f"   Model saved to: {autoencoder_path}")
    logger.info(f"   Normalizer saved to: {normalizer_path}")

    # Check if model was successfully saved
    if autoencoder_path.exists():
        file_size = autoencoder_path.stat().st_size / (1024 * 1024)
        logger.success(f"✅ Model saved successfully ({file_size:.2f} MB)")
    else:
        logger.error("❌ Model not saved! Check permissions and disk space.")

    # =====================================================================
    # ENCODER READY FOR TRANSFER LEARNING
    # =====================================================================

    logger.info(f"\n💡 Next steps:")
    logger.info(f"   1. Run grid search with transfer learning:")
    logger.info(f"      python main.py grid-search --use-transfer-learning")
    logger.info(f"   2. Or run inference with pretrained model:")
    logger.info(f"      python main.py inference --model-type pretrained")


if __name__ == "__main__":
    main()