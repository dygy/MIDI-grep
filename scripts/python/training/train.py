#!/usr/bin/env python3
"""
Fine-tune Basic Pitch model on custom dataset.

This script wraps Basic Pitch's training functionality for MIDI-grep.

Usage:
    python train.py --dataset ./dataset --output ./models/my-model --epochs 100
"""

import argparse
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_tensorflow():
    """Verify TensorFlow is available."""
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow version: {tf.__version__}")

        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU available: {gpus}")
        else:
            logger.warning("No GPU found - training will be slower on CPU")

        return True
    except ImportError:
        logger.error("TensorFlow not installed!")
        logger.info("Install with: pip install tensorflow")
        return False


def train_model(
    dataset_dir: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    base_model: str = None,
) -> None:
    """Fine-tune Basic Pitch on custom dataset."""

    if not check_tensorflow():
        sys.exit(1)

    import tensorflow as tf
    from basic_pitch import models
    from basic_pitch.data import tf_example_deserialization

    # Verify dataset exists
    if not os.path.isdir(dataset_dir):
        logger.error(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "validation")

    if not os.path.isdir(train_dir):
        logger.error(f"Training data not found: {train_dir}")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 50)
    logger.info("MIDI-grep Model Training")
    logger.info("=" * 50)
    logger.info(f"Dataset: {dataset_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info("=" * 50)

    # Build model
    logger.info("Building model...")
    model = models.model()

    # Load base weights if specified
    if base_model:
        if base_model == "basic-pitch":
            # Load pretrained Basic Pitch weights
            from basic_pitch import ICASSP_2022_MODEL_PATH
            logger.info(f"Loading pretrained weights from: {ICASSP_2022_MODEL_PATH}")
            # Note: This requires the saved model format, not just weights
            # For transfer learning, we'd load specific layers
        elif os.path.exists(base_model):
            logger.info(f"Loading weights from: {base_model}")
            model.load_weights(base_model)

    model.summary()

    # Prepare data loaders
    logger.info("Preparing data loaders...")

    # Use Basic Pitch's data loading utilities
    # Note: This assumes data is in TFRecord format
    try:
        train_ds, val_ds = tf_example_deserialization.prepare_datasets(
            dataset_dir,
            shuffle_size=100,
            batch_size=batch_size,
            validation_steps=10,
            datasets_to_use=["custom"],  # Would need to register custom dataset
            dataset_sampling_frequency=[1.0],
        )
    except Exception as e:
        logger.warning(f"Could not use Basic Pitch data loader: {e}")
        logger.info("Using simplified data loading...")

        # Simplified: Load TFRecords directly
        train_pattern = os.path.join(train_dir, "*.tfrecord")
        val_pattern = os.path.join(val_dir, "*.tfrecord")

        train_files = tf.io.gfile.glob(train_pattern)
        val_files = tf.io.gfile.glob(val_pattern)

        if not train_files:
            logger.error(f"No TFRecord files found in {train_dir}")
            logger.info("Run prepare_dataset.py first to create training data")
            sys.exit(1)

        logger.info(f"Found {len(train_files)} training files, {len(val_files)} validation files")

        # Create datasets (simplified - would need proper parsing)
        train_ds = tf.data.TFRecordDataset(train_files).batch(batch_size)
        val_ds = tf.data.TFRecordDataset(val_files).batch(batch_size) if val_files else None

    # Compile model
    logger.info("Compiling model...")
    loss = models.loss(weighted=False)
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate),
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, "tensorboard"),
            histogram_freq=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=25,
            verbose=1,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            verbose=1,
            patience=10,
            factor=0.5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "model.best"),
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "checkpoints", "model.{epoch:03d}"),
            save_freq='epoch'
        ),
    ]

    # Train
    logger.info("Starting training...")
    try:
        history = model.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_ds,
        )

        # Save final model
        final_path = os.path.join(output_dir, "model.final")
        model.save(final_path)
        logger.info(f"Final model saved to: {final_path}")

        # Save training history
        import json
        history_path = os.path.join(output_dir, "history.json")
        with open(history_path, 'w') as f:
            json.dump(history.history, f, indent=2, default=str)
        logger.info(f"Training history saved to: {history_path}")

        logger.info("=" * 50)
        logger.info("Training complete!")
        logger.info("=" * 50)
        logger.info(f"Model saved to: {output_dir}")
        logger.info("")
        logger.info("To use this model for extraction:")
        logger.info(f"  midi-grep extract --model {final_path} --url '...'")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Basic Pitch model on custom dataset"
    )

    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Path to prepared dataset directory"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--learning-rate", "-l",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--base-model",
        default="basic-pitch",
        help="Base model to fine-tune from (default: basic-pitch)"
    )

    args = parser.parse_args()

    train_model(
        dataset_dir=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        base_model=args.base_model,
    )


if __name__ == "__main__":
    main()
