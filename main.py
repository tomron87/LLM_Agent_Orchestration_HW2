"""
Main Execution Script
=====================

Complete pipeline for LSTM frequency extraction system.

Usage:
    # Train L=1 stateful model
    python main.py --model stateful --epochs 50

    # Train L>1 sequence model
    python main.py --model sequence --sequence-length 10 --epochs 50

    # Evaluate existing model
    python main.py --mode evaluate --checkpoint outputs/models/best_model.pth

    # Quick test run
    python main.py --quick-test
"""

import argparse
import torch
import numpy as np
import os
from pathlib import Path

# Import project modules
from src.data import generate_dataset, create_train_val_loaders, create_test_loader
from src.models import StatefulLSTM, SequenceLSTM
from src.training import TrainingConfig, train_model
from src.evaluation import Evaluator, Visualizer, plot_all_results, print_evaluation_summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='LSTM Frequency Extraction System'
    )

    # Mode
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'visualize'],
                        help='Execution mode')

    # Model settings
    parser.add_argument('--model', type=str, default='stateful',
                        choices=['stateful', 'sequence'],
                        help='Model type: stateful (L=1) or sequence (L>1)')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Number of LSTM layers')
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='Sequence length for L>1 model')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')

    # Training settings
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')

    # Data settings
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split fraction')

    # Paths
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test run with small model and few epochs')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu/mps)')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_pipeline(args):
    """Complete training pipeline."""

    print("\n" + "="*70)
    print(" "*20 + "LSTM FREQUENCY EXTRACTION SYSTEM")
    print("="*70)

    # Set random seed
    set_seed(args.seed)

    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_dir}/models").mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_dir}/figures").mkdir(parents=True, exist_ok=True)

    # Step 1: Generate or load datasets
    print("\n" + "-"*70)
    print("STEP 1: DATA GENERATION")
    print("-"*70)

    train_data_path = f"{args.output_dir}/train_data.pkl"
    test_data_path = f"{args.output_dir}/test_data.pkl"

    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        print("Loading existing datasets...")
        from src.data.signal_generator import load_dataset
        train_data = load_dataset(train_data_path)
        test_data = load_dataset(test_data_path)
    else:
        print("Generating training dataset (seed=1)...")
        train_data = generate_dataset(seed=1, save_path=train_data_path)

        print("Generating test dataset (seed=2)...")
        test_data = generate_dataset(seed=2, save_path=test_data_path)

    print(f"✓ Training samples: {len(train_data['S'])} time points")
    print(f"✓ Test samples: {len(test_data['S'])} time points")

    # Step 2: Create data loaders
    print("\n" + "-"*70)
    print("STEP 2: DATA LOADERS")
    print("-"*70)

    train_loader, val_loader = create_train_val_loaders(
        train_data,
        val_split=args.val_split,
        batch_size=args.batch_size,
        model_type=args.model,
        sequence_length=args.sequence_length if args.model == 'sequence' else 1
    )

    test_loader = create_test_loader(
        test_data,
        batch_size=args.batch_size,
        model_type=args.model,
        sequence_length=args.sequence_length if args.model == 'sequence' else 1
    )

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")

    # Step 3: Create model
    print("\n" + "-"*70)
    print("STEP 3: MODEL CREATION")
    print("-"*70)

    if args.model == 'stateful':
        model = StatefulLSTM(
            input_size=5,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        model = SequenceLSTM(
            input_size=5,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            sequence_length=args.sequence_length,
            dropout=args.dropout
        )

    print(f"✓ Model type: {args.model}")
    print(f"✓ Parameters: {model.num_parameters():,}")
    print(model)

    # Step 4: Training configuration
    print("\n" + "-"*70)
    print("STEP 4: TRAINING CONFIGURATION")
    print("-"*70)

    config = TrainingConfig(
        model_type=args.model,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        sequence_length=args.sequence_length if args.model == 'sequence' else 1,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        val_split=args.val_split,
        patience=args.patience,
        checkpoint_dir=f"{args.output_dir}/models",
        device=args.device if args.device else ('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'),
        seed=args.seed
    )

    print(config)

    # Step 5: Train model
    print("\n" + "-"*70)
    print("STEP 5: TRAINING")
    print("-"*70)

    trained_model, history = train_model(model, train_loader, val_loader, config)

    print("\n✓ Training complete!")

    # Step 6: Evaluation
    print("\n" + "-"*70)
    print("STEP 6: EVALUATION")
    print("-"*70)

    evaluator = Evaluator(trained_model, device=config.device, model_type=args.model)

    # Evaluate on training set
    print("Evaluating on training set...")
    train_mse, train_pred, train_targets = evaluator.evaluate(train_loader)
    freq_metrics_train = evaluator.evaluate_per_frequency(train_loader)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_mse, test_pred, test_targets = evaluator.evaluate(test_loader)
    freq_metrics_test = evaluator.evaluate_per_frequency(test_loader)

    # Print summary
    print_evaluation_summary(
        train_mse, test_mse,
        freq_metrics_train, freq_metrics_test
    )

    # Step 7: Visualization
    print("\n" + "-"*70)
    print("STEP 7: VISUALIZATION")
    print("-"*70)

    visualizer = Visualizer(save_dir=f"{args.output_dir}/figures")
    plot_all_results(visualizer, test_data, test_pred, history)

    print("\n" + "="*70)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutputs saved to: {args.output_dir}/")
    print(f"  - Models: {args.output_dir}/models/")
    print(f"  - Figures: {args.output_dir}/figures/")
    print("\n")


def evaluate_pipeline(args):
    """Evaluation pipeline for existing model."""

    print("\n" + "="*70)
    print(" "*25 + "MODEL EVALUATION")
    print("="*70)

    if args.checkpoint is None:
        raise ValueError("Must provide --checkpoint path for evaluation mode")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    config_dict = checkpoint['config']
    model_type = config_dict['model_type']

    # Create model
    if model_type == 'stateful':
        model = StatefulLSTM(
            input_size=5,
            hidden_size=config_dict['hidden_size'],
            num_layers=config_dict['num_layers']
        )
    else:
        model = SequenceLSTM(
            input_size=5,
            hidden_size=config_dict['hidden_size'],
            num_layers=config_dict['num_layers'],
            sequence_length=config_dict['sequence_length']
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded: {model_type}")

    # Load test data
    print("\nLoading test data...")
    test_data_path = f"{args.output_dir}/test_data.pkl"
    if not os.path.exists(test_data_path):
        print("Generating test dataset...")
        test_data = generate_dataset(seed=2, save_path=test_data_path)
    else:
        from src.data.signal_generator import load_dataset
        test_data = load_dataset(test_data_path)

    # Create test loader
    test_loader = create_test_loader(
        test_data,
        batch_size=config_dict['batch_size'],
        model_type=model_type,
        sequence_length=config_dict.get('sequence_length', 1)
    )

    # Evaluate
    evaluator = Evaluator(model, device=args.device or 'cpu', model_type=model_type)

    print("\nEvaluating...")
    test_mse, predictions, targets = evaluator.evaluate(test_loader)
    freq_metrics = evaluator.evaluate_per_frequency(test_loader)

    # Print results
    print(f"\nTest MSE: {test_mse:.6f}")
    print("\nPer-frequency metrics:")
    for freq_idx, metrics in freq_metrics.items():
        print(f"  Frequency {freq_idx+1}: MSE={metrics['mse']:.6f}, "
              f"Correlation={metrics['correlation']:.4f}")

    print("\n" + "="*70)


def main():
    """Main entry point."""
    args = parse_args()

    # Quick test mode
    if args.quick_test:
        print("\n*** QUICK TEST MODE ***")
        args.epochs = 5
        args.hidden_size = 32
        args.batch_size = 64
        args.patience = 3
        args.model = 'stateful'

    # Execute appropriate pipeline
    if args.mode == 'train':
        train_pipeline(args)
    elif args.mode == 'evaluate':
        evaluate_pipeline(args)
    else:
        print(f"Mode '{args.mode}' not implemented yet")


if __name__ == '__main__':
    main()
