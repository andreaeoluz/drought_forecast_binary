#!/usr/bin/env python3
"""main.py - Unified entry point for the framework."""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import ExperimentConfig, get_data_path
from utils import set_reproducible_seeds, Logger


def print_config_summary(config: ExperimentConfig, optimization_metric: str = None):
    """Print configuration summary."""
    # ✅ CORREÇÃO: usar config.get_downsample() em vez de config.data.get_downsample()
    ds_h, ds_w = config.get_downsample(config.region)
    ds_info = config.get_downsample_info(config.region)
    
    print(f"\n{'='*60}")
    print(f"📍 Region: {config.region}")
    print(f"📊 SPI-{config.spi.scale} | Threshold: {config.spi.threshold} ({config.spi.threshold_name})")
    print(f"📈 Expected prevalence: {config.spi.expected_prevalence:.2%}")
    print(f"📉 Downsampling: {ds_h}x{ds_w} ({ds_info['preservation_estimate']})")
    print(f"🔄 Transfer Learning: {config.use_transfer_learning}")
    
    if optimization_metric:
        print(f"🎯 Optimization: {optimization_metric.upper()}")
    
    print(f"{'='*60}\n")


def setup_parser() -> argparse.ArgumentParser:
    """Configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Drought Forecast Binary Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Precompute SPI
  python main.py precompute-spi

  # Train autoencoder
  python main.py train-ae

  # Run grid search
  python main.py grid-search --optimization-metric mcc

  # Run inference with calibration (RECOMMENDED)
  python main.py inference --recalibrate --use-calibration --optimization-metric csi

  # Inference without calibration
  python main.py inference --p 9 --q 1 --model-type pretrained --no-calibration
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Command to execute')
    
    # Precompute SPI
    subparsers.add_parser('precompute-spi', help='Precompute SPI for the region')
    
    # Train Autoencoder
    ae_parser = subparsers.add_parser('train-ae', help='Train autoencoder')
    ae_parser.add_argument('--force', action='store_true', help='Force retraining')
    
    # Grid Search
    gs_parser = subparsers.add_parser('grid-search', help='Run hyperparameter grid search')
    gs_parser.add_argument('--threshold-spi', type=float, default=None, help='SPI threshold')
    gs_parser.add_argument('--region', type=str, default=None, help='Region')
    gs_parser.add_argument('--use-transfer-learning', action='store_true', default=None, help='Use transfer learning')
    gs_parser.add_argument('--optimization-metric', choices=['mcc', 'csi'], default='mcc',
                      help='Metric for model selection [default: mcc]')  
    
    # Inference
    inf_parser = subparsers.add_parser('inference', help='Run inference on test period')
    inf_parser.add_argument('--p', type=int, help='History length (default: best model)')
    inf_parser.add_argument('--q', type=int, help='Forecast horizon (default: best model)')
    inf_parser.add_argument('--model-type', choices=['pretrained', 'scratch'], 
                           help='Model type (default: auto-detected)')
    
    # Calibration and threshold options
    inf_parser.add_argument('--use-calibration', action='store_true', default=True,
                           help='Enable probability calibration [default: True]')
    inf_parser.add_argument('--no-calibration', action='store_true',
                           help='Disable probability calibration')
    inf_parser.add_argument('--recalibrate', action='store_true',
                           help='Recalibrate threshold with validation data')
    inf_parser.add_argument('--use-validation-fallback', action='store_true', default=True,
                           help='Use validation data for threshold if test samples insufficient [default: True]')
    inf_parser.add_argument('--threshold', type=float, help='Custom threshold (overrides optimization)')
    
    # Model discovery
    inf_parser.add_argument('--list-models', action='store_true', help='List available models')
    
    # Region and SPI
    inf_parser.add_argument('--region', type=str, default=None, help='Region')
    inf_parser.add_argument('--threshold-spi', type=float, default=None, help='SPI threshold')
    inf_parser.add_argument('--optimization-metric', choices=['mcc', 'csi'], default='mcc',
                           help='Metric for threshold optimization [default: mcc]')  
    
    return parser


def apply_args_to_config(config: ExperimentConfig, args) -> None:
    """Apply command-line arguments to configuration."""
    if hasattr(args, 'region') and args.region is not None:
        config.region = args.region
    
    if hasattr(args, 'threshold_spi') and args.threshold_spi is not None:
        config.spi.threshold = args.threshold_spi
    
    if hasattr(args, 'use_transfer_learning') and args.use_transfer_learning is not None:
        config.use_transfer_learning = args.use_transfer_learning


def execute_command(args, config: ExperimentConfig, base_data_path: Path):
    """Execute the requested command."""
    logger = Logger()
    
    if args.command == 'precompute-spi':
        from experiments import precompute_spi
        precompute_spi()
    
    elif args.command == 'train-ae':
        from experiments import train_autoencoder
        train_autoencoder()
    
    elif args.command == 'grid-search':
        from experiments import GridSearch
        optimization_metric = getattr(args, 'optimization_metric', 'mcc')
        
        grid_search = GridSearch(
            config,
            base_data_path,
            optimization_metric=optimization_metric
        )
        grid_search.run()
    
    elif args.command == 'inference':
        from inference.run import main as inference_main
        
        argv = [sys.argv[0]]
        
        # Model parameters
        if args.p is not None:
            argv.extend(['--p', str(args.p)])
        if args.q is not None:
            argv.extend(['--q', str(args.q)])
        if args.model_type:
            argv.extend(['--model-type', args.model_type])
        
        # Calibration flags
        if args.no_calibration:
            argv.append('--no-calibration')
        elif args.use_calibration:
            argv.append('--use-calibration')
        
        if args.recalibrate:
            argv.append('--recalibrate')
        if args.use_validation_fallback:
            argv.append('--use-validation')
        
        # Threshold
        if args.threshold is not None:
            argv.extend(['--threshold', str(args.threshold)])
        
        # Utilities
        if args.list_models:
            argv.append('--list-models')
        
        # Region and SPI
        if args.region:
            argv.extend(['--region', args.region])
        if args.threshold_spi is not None:
            argv.extend(['--threshold-spi', str(args.threshold_spi)])
        if hasattr(args, 'optimization_metric') and args.optimization_metric:
            argv.extend(['--optimization-metric', args.optimization_metric])
        
        sys.argv = argv
        inference_main()
    
    else:
        logger.error(f"Unknown command: {args.command}")
        logger.info("Use --help to see available commands")


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    config = ExperimentConfig()
    apply_args_to_config(config, args)
    
    optimization_metric = getattr(args, 'optimization_metric', None)
    print_config_summary(config, optimization_metric)
    
    set_reproducible_seeds(config.random_seed)
    base_data_path = get_data_path()
    
    execute_command(args, config, base_data_path)


if __name__ == "__main__":
    main()