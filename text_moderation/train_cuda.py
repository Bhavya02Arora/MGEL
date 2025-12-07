"""
train_cuda.py - Memory-Optimized Training Pipeline for MGEL
Trains on all datasets with proper train/validation/test splits
Includes early stopping and memory management for low-memory GPUs

OPTIMIZED VERSION - Memory-efficient BERT training with early stopping
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
import json
from datetime import datetime
import argparse
import torch
import gc

# Import MGEL components
import sys
sys.path.append('.')

from mgel_system_cuda import (
    NBLRClassifier,
    EnhancedBERT,
    MGELEnsemble,
    evaluate_model,
    TRANSFORMERS_AVAILABLE
)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

class MGELTrainingPipeline:
    """
    Memory-Optimized MGEL Training Pipeline

    Features:
    - Automatic memory cleanup between training runs
    - Early stopping to prevent overfitting
    - Configurable batch sizes and gradient accumulation
    - Support for low-memory GPUs
    """

    def __init__(self, data_dir='./datasets', results_dir='./results'):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.datasets = {}
        self.models = {}
        self.results = {}

    # ----------------------------------------------------------
    # LOAD DATASETS
    # ----------------------------------------------------------
    def load_all_datasets(self):
        """Load all available datasets"""
        print("="*70)
        print("📂 LOADING DATASETS")
        print("="*70)

        dataset_files = {
            'twitter': 'twitter_hate_speech.csv',
            'finance': 'yahoo_finance_sample.csv',
            'news': 'yahoo_news_sample.csv',
            'toxic_obfuscated': 'toxic_obfuscated_dataset.csv',
            'openai': 'openai_moderation_1680.csv'
        }

        for name, filename in dataset_files.items():
            filepath = self.data_dir / filename

            if filepath.exists():
                df = pd.read_csv(filepath)

                # Standardize columns
                if 'comment' in df.columns:
                    df = df.rename(columns={'comment': 'text'})
                if 'tweet' in df.columns:
                    df = df.rename(columns={'tweet': 'text'})

                df = df.dropna(subset=['text', 'label'])
                df['text'] = df['text'].astype(str)
                df['label'] = df['label'].astype(int)

                self.datasets[name] = df

                toxic_pct = (df['label'] == 1).mean() * 100
                print(f"\n✅ {name.upper()}")
                print(f"   Samples: {len(df):,}")
                print(f"   Toxic: {(df['label']==1).sum():,} ({toxic_pct:.1f}%)")
                print(f"   Clean: {(df['label']==0).sum():,} ({100-toxic_pct:.1f}%)")

            else:
                print(f"⚠️  Missing dataset: {filepath}")

        if not self.datasets:
            raise RuntimeError("No datasets loaded!")

        return self.datasets

    # ----------------------------------------------------------
    # DATA SPLIT: 80% train, 10% val, 10% test
    # ----------------------------------------------------------
    def split_dataset(self, df, random_state=42):
        """
        Split following the paper:
        - 80% training
        - 10% validation (for hyperparameter tuning & early stopping)
        - 10% test (for final evaluation)
        """
        # Step 1: 10% test
        train_val, test = train_test_split(
            df, test_size=0.10, random_state=random_state,
            stratify=df['label']
        )

        # Step 2: 10% val from remaining 90%
        val_frac = 0.10 / 0.90  # ~0.111
        train, val = train_test_split(
            train_val, test_size=val_frac, random_state=random_state,
            stratify=train_val['label']
        )

        return {'train': train, 'val': val, 'test': test}

    # ----------------------------------------------------------
    # MEMORY CLEANUP
    # ----------------------------------------------------------
    def cleanup_memory(self):
        """Aggressive memory cleanup between training runs"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ----------------------------------------------------------
    # TRAIN SINGLE DATASET
    # ----------------------------------------------------------
    def train_single_dataset(self, dataset_name, use_ensemble=False,
                            bert_epochs=3, bert_batch_size=2,
                            gradient_accumulation_steps=4,
                            early_stopping_patience=3,
                            max_length=128):
        """
        Train on a single dataset with memory optimizations

        Args:
            dataset_name: Name of the dataset to train on
            use_ensemble: Whether to use full MGEL ensemble with BERT
            bert_epochs: Maximum epochs for BERT training
            bert_batch_size: Batch size for BERT (use 2 for low memory)
            gradient_accumulation_steps: Simulate larger batches (4-8 recommended)
            early_stopping_patience: Stop if no improvement for N epochs
            max_length: Maximum sequence length (reduce for memory savings)
        """
        print("\n" + "="*70)
        print(f"🎯 TRAINING ON {dataset_name.upper()}")
        print("="*70)

        if not TRANSFORMERS_AVAILABLE and use_ensemble:
            print("\n⚠️  WARNING: transformers not installed!")
            print("   Install with: pip install transformers torch")
            print("   Falling back to NBLR only...\n")
            use_ensemble = False

        # Memory cleanup before starting
        self.cleanup_memory()

        df = self.datasets[dataset_name]
        splits = self.split_dataset(df)

        print("\n📊 DATA SPLIT:")
        for k, v in splits.items():
            pct = len(v) / len(df) * 100
            print(f"   {k.capitalize()}: {len(v):,} samples ({pct:.1f}%)")

        X_train = splits['train']['text'].values
        y_train = splits['train']['label'].values
        X_val = splits['val']['text'].values
        y_val = splits['val']['label'].values
        X_test = splits['test']['text'].values
        y_test = splits['test']['label'].values

        results = {}

        # TRAIN NBLR (Fast baseline)
        print("\n🔧 Training NBLR...")
        nblr = NBLRClassifier()
        nblr.fit(X_train, y_train)

        val_pred = nblr.predict_proba(X_val)
        val_metrics = evaluate_model(y_val, val_pred)

        print(f"   Validation AUC@ROC: {val_metrics['AUC@ROC']:.4f}")

        results['nblr'] = {
            'model': nblr,
            'val_metrics': val_metrics
        }

        # TRAIN ENSEMBLE with MEMORY-OPTIMIZED BERT
        if use_ensemble:
            print("\n🔧 Training MGEL Ensemble with Memory-Optimized BERT...")
            print(f"   Settings: batch_size={bert_batch_size}, "
                  f"grad_accum={gradient_accumulation_steps}, "
                  f"early_stopping={early_stopping_patience}")

            # Byte-level BERT
            print("\n   [1/3] Training Enhanced BERT (byte level)...")
            self.cleanup_memory()

            bert_byte = EnhancedBERT(
                granularity='byte',
                max_length=max_length
            )
            bert_byte.fit(
                X_train, y_train,
                val_texts=X_val, val_labels=y_val,
                epochs=bert_epochs,
                batch_size=bert_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                early_stopping_patience=early_stopping_patience
            )

            # Character-level BERT
            print("\n   [2/3] Training Enhanced BERT (character level)...")
            self.cleanup_memory()

            bert_char = EnhancedBERT(
                granularity='character',
                max_length=max_length
            )
            bert_char.fit(
                X_train, y_train,
                val_texts=X_val, val_labels=y_val,
                epochs=bert_epochs,
                batch_size=bert_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                early_stopping_patience=early_stopping_patience
            )

            # Subword-level BERT
            print("\n   [3/3] Training Enhanced BERT (subword level)...")
            self.cleanup_memory()

            bert_sub = EnhancedBERT(
                granularity='subword',
                max_length=max_length
            )
            bert_sub.fit(
                X_train, y_train,
                val_texts=X_val, val_labels=y_val,
                epochs=bert_epochs,
                batch_size=bert_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                early_stopping_patience=early_stopping_patience
            )

            # Create ensemble
            print("\n   Creating MGEL ensemble...")
            mgel = MGELEnsemble()
            mgel.add_model('nblr', nblr)
            mgel.add_model('bert_byte', bert_byte)
            mgel.add_model('bert_char', bert_char)
            mgel.add_model('bert_subword', bert_sub)

            print("   Optimizing ensemble weights...")
            mgel.optimize_weights_grid_search(X_val, y_val, step=0.1)

            val_pred = mgel.predict_proba(X_val)
            val_metrics = evaluate_model(y_val, val_pred)

            print(f"   Ensemble Validation AUC@ROC: {val_metrics['AUC@ROC']:.4f}")

            results['mgel'] = {
                'model': mgel,
                'val_metrics': val_metrics
            }

            self.cleanup_memory()

        # EVALUATE ON TEST SET
        print("\n📊 FINAL TEST RESULTS:")
        print("-"*70)

        for mname, mdata in results.items():
            pred = mdata['model'].predict_proba(X_test)
            mdata['test_metrics'] = evaluate_model(y_test, pred)

            print(f"\n{mname.upper()}:")
            for k, v in mdata['test_metrics'].items():
                print(f"   {k}: {v:.4f}")

        # SAVE RESULTS
        self.results[dataset_name] = {
            'splits': {k: len(v) for k, v in splits.items()},
            'training_config': {
                'bert_epochs': bert_epochs,
                'bert_batch_size': bert_batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'early_stopping_patience': early_stopping_patience,
                'max_length': max_length
            },
            'models': {
                name: {
                    'val_metrics': data['val_metrics'],
                    'test_metrics': data['test_metrics']
                }
                for name, data in results.items()
            }
        }

        # Save models
        self.models[dataset_name] = results
        self.save_model(dataset_name, results)

        return results

    # ----------------------------------------------------------
    # TRAIN COMBINED DATASETS
    # ----------------------------------------------------------
    def train_combined_datasets(self, dataset_names=None, use_ensemble=False,
                                bert_epochs=3, bert_batch_size=2,
                                gradient_accumulation_steps=4,
                                early_stopping_patience=3,
                                max_length=128):
        """
        Train on combined datasets with memory optimizations
        """
        print("\n" + "="*70)
        print("🎯 TRAINING ON COMBINED DATASETS")
        print("="*70)

        if not TRANSFORMERS_AVAILABLE and use_ensemble:
            print("\n⚠️  transformers not installed - using NBLR only")
            use_ensemble = False

        if dataset_names is None:
            dataset_names = list(self.datasets.keys())

        print(f"\nCombining datasets: {', '.join(dataset_names)}")

        self.cleanup_memory()

        df = pd.concat(
            [self.datasets[n].assign(source=n) for n in dataset_names],
            ignore_index=True
        )

        print(f"📊 Combined dataset: {len(df):,} samples")
        print(f"   Toxic: {(df['label']==1).sum():,} ({(df['label']==1).mean()*100:.1f}%)")

        splits = self.split_dataset(df)

        X_train = splits['train']['text'].values
        y_train = splits['train']['label'].values
        X_val = splits['val']['text'].values
        y_val = splits['val']['label'].values
        X_test = splits['test']['text'].values
        y_test = splits['test']['label'].values

        results = {}

        # Train NBLR
        print("\n🔧 Training NBLR on combined data...")
        nblr = NBLRClassifier()
        nblr.fit(X_train, y_train)

        val_pred = nblr.predict_proba(X_val)
        val_metrics = evaluate_model(y_val, val_pred)

        results['nblr'] = {
            'model': nblr,
            'val_metrics': val_metrics
        }

        # Train ensemble with MEMORY-OPTIMIZED BERT
        if use_ensemble:
            print("\n🔧 Training MGEL Ensemble on combined data...")
            print(f"   Settings: batch_size={bert_batch_size}, "
                  f"grad_accum={gradient_accumulation_steps}, "
                  f"early_stopping={early_stopping_patience}")

            # Byte-level
            print("\n   [1/3] Training byte-level BERT...")
            self.cleanup_memory()
            bert_byte = EnhancedBERT(granularity='byte', max_length=max_length)
            bert_byte.fit(
                X_train, y_train, X_val, y_val,
                epochs=bert_epochs,
                batch_size=bert_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                early_stopping_patience=early_stopping_patience
            )

            # Character-level
            print("\n   [2/3] Training character-level BERT...")
            self.cleanup_memory()
            bert_char = EnhancedBERT(granularity='character', max_length=max_length)
            bert_char.fit(
                X_train, y_train, X_val, y_val,
                epochs=bert_epochs,
                batch_size=bert_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                early_stopping_patience=early_stopping_patience
            )

            # Subword-level
            print("\n   [3/3] Training subword-level BERT...")
            self.cleanup_memory()
            bert_sub = EnhancedBERT(granularity='subword', max_length=max_length)
            bert_sub.fit(
                X_train, y_train, X_val, y_val,
                epochs=bert_epochs,
                batch_size=bert_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                early_stopping_patience=early_stopping_patience
            )

            # Ensemble
            ensemble = MGELEnsemble()
            ensemble.add_model('nblr', nblr)
            ensemble.add_model('bert_byte', bert_byte)
            ensemble.add_model('bert_char', bert_char)
            ensemble.add_model('bert_subword', bert_sub)

            ensemble.optimize_weights_grid_search(X_val, y_val)
            val_pred = ensemble.predict_proba(X_val)

            results['mgel'] = {
                'model': ensemble,
                'val_metrics': evaluate_model(y_val, val_pred)
            }

            self.cleanup_memory()

        # Test evaluation
        print("\n📊 FINAL TEST PERFORMANCE")
        print("-"*70)

        for name, data in results.items():
            pred = data['model'].predict_proba(X_test)
            data['test_metrics'] = evaluate_model(y_test, pred)

            print(f"\n{name.upper()}:")
            for k, v in data['test_metrics'].items():
                print(f"   {k}: {v:.4f}")

        # Save results
        self.results['combined'] = {
            'splits': {k: len(v) for k, v in splits.items()},
            'training_config': {
                'bert_epochs': bert_epochs,
                'bert_batch_size': bert_batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'early_stopping_patience': early_stopping_patience,
                'max_length': max_length
            },
            'models': {
                name: {
                    'val_metrics': data['val_metrics'],
                    'test_metrics': data['test_metrics']
                }
                for name, data in results.items()
            }
        }

        self.models['combined'] = results
        self.save_model('combined', results)

        return results

    # ----------------------------------------------------------
    def save_model(self, name, results):
        """Save trained models"""
        model_dir = self.results_dir / 'models'
        model_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_name, model_data in results.items():
            # Special handling for BERT models
            if 'bert' in model_name and hasattr(model_data['model'], 'save_model'):
                bert_path = model_dir / f"{name}_{model_name}_{timestamp}_bert"
                model_data['model'].save_model(str(bert_path))
            else:
                # Regular pickle save
                filepath = model_dir / f"{name}_{model_name}_{timestamp}.pkl"
                with open(filepath, "wb") as f:
                    pickle.dump(model_data["model"], f)
                print(f"💾 Saved {name}_{model_name} to {filepath}")

    # ----------------------------------------------------------
    def save_results(self):
        """Save all results to JSON"""
        results_file = self.results_dir / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")

    # ----------------------------------------------------------
    def compare_models(self):
        """Print comparison table"""
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON TABLE (TEST PERFORMANCE)")
        print("="*70)

        print(f"\n{'Dataset':<15} {'Model':<10} {'AUC@ROC':<10} {'AUC@PR':<10} {'F1':<8} {'MCC':<8}")
        print("-"*70)

        for dataset, data in self.results.items():
            for model_name, metrics in data['models'].items():
                t = metrics['test_metrics']
                print(f"{dataset:<15} {model_name:<10} "
                      f"{t['AUC@ROC']:<10.4f} "
                      f"{t['AUC@PR']:<10.4f} "
                      f"{t['F1']:<8.4f} "
                      f"{t['MCC']:<8.4f}")

        print("="*70)


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Memory-Optimized MGEL Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast training (NBLR only)
  python train_cuda.py --mode quick --dataset twitter

  # Low-memory GPU training with early stopping
  python train_cuda.py --mode quick --dataset twitter --ensemble \\
      --epochs 10 --batch-size 2 --grad-accum 4 --patience 3

  # Combined datasets with aggressive memory settings
  python train_cuda.py --mode combined --ensemble \\
      --batch-size 1 --grad-accum 8 --max-length 64

  # Full training with optimal settings
  python train_cuda.py --mode full --ensemble \\
      --epochs 5 --batch-size 2 --grad-accum 4 --patience 3
        """
    )

    parser.add_argument('--mode',
                       choices=['quick', 'full', 'combined'],
                       default='quick',
                       help='Training mode')

    parser.add_argument('--dataset',
                       type=str,
                       default='twitter',
                       help='Dataset name for quick mode')

    parser.add_argument('--ensemble',
                       action='store_true',
                       help='Use full MGEL ensemble with BERT')

    # BERT training parameters
    parser.add_argument('--epochs',
                       type=int,
                       default=3,
                       help='Maximum epochs for BERT (default: 3)')

    parser.add_argument('--batch-size',
                       type=int,
                       default=2,
                       help='Batch size for BERT (use 1-2 for low memory, default: 2)')

    parser.add_argument('--grad-accum',
                       type=int,
                       default=4,
                       help='Gradient accumulation steps (default: 4, effective_batch=batch_size*grad_accum)')

    parser.add_argument('--patience',
                       type=int,
                       default=3,
                       help='Early stopping patience in epochs (default: 3)')

    parser.add_argument('--max-length',
                       type=int,
                       default=128,
                       help='Maximum sequence length (reduce to 64/32 for memory savings, default: 128)')

    args = parser.parse_args()

    # Print configuration
    print("="*70)
    print("🚀 MEMORY-OPTIMIZED MGEL TRAINING PIPELINE")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Ensemble: {'Yes (BERT)' if args.ensemble else 'No (NBLR only)'}")

    if args.ensemble:
        print(f"\n💡 BERT Configuration:")
        print(f"   Max Epochs: {args.epochs}")
        print(f"   Batch Size: {args.batch_size}")
        print(f"   Gradient Accumulation: {args.grad_accum}")
        print(f"   Effective Batch Size: {args.batch_size * args.grad_accum}")
        print(f"   Early Stopping Patience: {args.patience} epochs")
        print(f"   Max Sequence Length: {args.max_length}")
        print(f"   Transformers: {'✅ Available' if TRANSFORMERS_AVAILABLE else '❌ Not installed'}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        else:
            print(f"   Device: CPU (training will be slow)")

    print("="*70)

    # Memory tips
    if args.ensemble and torch.cuda.is_available():
        print("\n💡 MEMORY TIPS:")
        print("   • If OOM error occurs, try:")
        print("     - Reduce --batch-size to 1")
        print("     - Increase --grad-accum to 8 or 16")
        print("     - Reduce --max-length to 64 or 32")
        print("     - Use DistilBERT in mgel_system.py (40% smaller)")
        print("="*70)

    # Run selected mode
    pipeline = MGELTrainingPipeline()
    pipeline.load_all_datasets()

    if args.mode == 'quick':
        pipeline.train_single_dataset(
            args.dataset,
            use_ensemble=args.ensemble,
            bert_epochs=args.epochs,
            bert_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            early_stopping_patience=args.patience,
            max_length=args.max_length
        )
        pipeline.save_results()

    elif args.mode == 'combined':
        pipeline.train_combined_datasets(
            use_ensemble=args.ensemble,
            bert_epochs=args.epochs,
            bert_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            early_stopping_patience=args.patience,
            max_length=args.max_length
        )
        pipeline.save_results()

    elif args.mode == 'full':
        # Train all datasets individually
        for dataset_name in pipeline.datasets.keys():
            pipeline.train_single_dataset(
                dataset_name,
                use_ensemble=args.ensemble,
                bert_epochs=args.epochs,
                bert_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                early_stopping_patience=args.patience,
                max_length=args.max_length
            )

        # Train combined
        pipeline.train_combined_datasets(
            use_ensemble=args.ensemble,
            bert_epochs=args.epochs,
            bert_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            early_stopping_patience=args.patience,
            max_length=args.max_length
        )

        pipeline.compare_models()
        pipeline.save_results()

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\n📖 Next steps:")
    print("   1. Check results/ directory for saved models and metrics")
    print("   2. Load model: pickle.load(open('results/models/...pkl', 'rb'))")
    print("   3. Use for predictions on new data")
    print("="*70)

    #python train_cuda.py --mode quick --dataset twitter --ensemble --batch-size 1 --grad-accum 8 --max-length 64 --patience 3
    # python train_cuda.py --mode quick --dataset twitter
    # python train_cuda.py --mode quick --dataset twitter --ensemble --epochs 10 --batch-size 2 --grad-accum 4 --patience 3