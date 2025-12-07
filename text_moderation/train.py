"""
Complete Training Pipeline for MGEL
Trains on all datasets with proper train/validation/test splits
Reproduces the paper's experimental setup

CORRECTED VERSION - Uses Real BERT instead of simulators
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
import json
from datetime import datetime

# Import MGEL components
import sys
sys.path.append('.')

# CHANGE 1: Import from the corrected module with Real BERT
from mgel_system import (
    NBLRClassifier,
    EnhancedBERT,  # Real BERT implementation
    MGELEnsemble,
    evaluate_model,
    TRANSFORMERS_AVAILABLE  # Check if transformers is installed
)


class MGELTrainingPipeline:

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
                print(f"⚠️ Missing dataset: {filepath}")

        if not self.datasets:
            raise RuntimeError("No datasets loaded!")

        return self.datasets

    # ----------------------------------------------------------
    # CORRECT SPLIT: 80% train, 10% val, 10% test
    # ----------------------------------------------------------
    def split_dataset(self, df, random_state=42):
        """
        Split following the paper:
        - 80% training
        - 10% validation (for hyperparameter tuning)
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
    # TRAIN SINGLE DATASET
    # ----------------------------------------------------------
    def train_single_dataset(self, dataset_name, use_ensemble=False, 
                            bert_epochs=3, bert_batch_size=8):
        """
        Train on a single dataset
        
        CHANGE 2: Added bert_epochs and bert_batch_size parameters
        """
        print("\n" + "="*70)
        print(f"🎯 TRAINING ON {dataset_name.upper()}")
        print("="*70)

        if not TRANSFORMERS_AVAILABLE and use_ensemble:
            print("\n⚠️  WARNING: transformers not installed!")
            print("   Install with: pip install transformers torch")
            print("   Falling back to NBLR only...\n")
            use_ensemble = False

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

        # TRAIN ENSEMBLE with REAL BERT
        if use_ensemble:
            print("\n🔧 Training MGEL Ensemble with REAL BERT...")
            
            # CHANGE 3: Use Real BERT instead of simulators
            print("   Training Enhanced BERT (byte level)...")
            bert_byte = EnhancedBERT(
                granularity='byte',
                max_length=128
            )
            bert_byte.fit(
                X_train, y_train,
                val_texts=X_val, val_labels=y_val,
                epochs=bert_epochs,
                batch_size=bert_batch_size
            )

            print("   Training Enhanced BERT (character level)...")
            bert_char = EnhancedBERT(
                granularity='character',
                max_length=128
            )
            bert_char.fit(
                X_train, y_train,
                val_texts=X_val, val_labels=y_val,
                epochs=bert_epochs,
                batch_size=bert_batch_size
            )

            print("   Training Enhanced BERT (subword level)...")
            bert_sub = EnhancedBERT(
                granularity='subword',
                max_length=128
            )
            bert_sub.fit(
                X_train, y_train,
                val_texts=X_val, val_labels=y_val,
                epochs=bert_epochs,
                batch_size=bert_batch_size
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
            
            print(f"   Validation AUC@ROC: {val_metrics['AUC@ROC']:.4f}")
            
            results['mgel'] = {
                'model': mgel,
                'val_metrics': val_metrics
            }

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
                                bert_epochs=3, bert_batch_size=8):
        """
        Train on combined datasets
        
        CHANGE 4: Added bert parameters
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

        # Train ensemble with REAL BERT
        if use_ensemble:
            print("\n🔧 Training MGEL Ensemble on combined data...")
            
            # CHANGE 5: Real BERT models
            bert_byte = EnhancedBERT(granularity='byte', max_length=128)
            bert_byte.fit(X_train, y_train, X_val, y_val, 
                         epochs=bert_epochs, batch_size=bert_batch_size)
            
            bert_char = EnhancedBERT(granularity='character', max_length=128)
            bert_char.fit(X_train, y_train, X_val, y_val,
                         epochs=bert_epochs, batch_size=bert_batch_size)
            
            bert_sub = EnhancedBERT(granularity='subword', max_length=128)
            bert_sub.fit(X_train, y_train, X_val, y_val,
                        epochs=bert_epochs, batch_size=bert_batch_size)

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
            filepath = model_dir / f"{name}_{model_name}_{timestamp}.pkl"
            
            # CHANGE 6: Special handling for BERT models
            if 'bert' in model_name and hasattr(model_data['model'], 'save_model'):
                # Save BERT separately
                bert_path = model_dir / f"{name}_{model_name}_{timestamp}_bert"
                model_data['model'].save_model(str(bert_path))
            else:
                # Regular pickle save
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
# EXAMPLE FUNCTIONS
# ----------------------------------------------------------

def quick_train_example():
    """Quick example - NBLR only on single dataset"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║            QUICK EXAMPLE: Train NBLR on Single Dataset          ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    pipeline = MGELTrainingPipeline()
    pipeline.load_all_datasets()

    # Fast training with NBLR only
    pipeline.train_single_dataset('twitter', use_ensemble=False)

    pipeline.save_results()
    print("\n✅ Quick training complete!")


def full_training_example():
    """Full training with Real BERT ensemble"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║       FULL TRAINING: All Datasets + Real BERT Ensemble          ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    if not TRANSFORMERS_AVAILABLE:
        print("\n⚠️  ERROR: transformers not installed!")
        print("   Install with: pip install transformers torch")
        return

    pipeline = MGELTrainingPipeline()
    pipeline.load_all_datasets()

    # Train each dataset individually with ensemble
    for dataset_name in pipeline.datasets.keys():
        pipeline.train_single_dataset(
            dataset_name, 
            use_ensemble=True,
            bert_epochs=3,  # Adjust for your compute
            bert_batch_size=8
        )

    # Train on all combined
    pipeline.train_combined_datasets(
        use_ensemble=True,
        bert_epochs=3,
        bert_batch_size=8
    )

    # Compare all models
    pipeline.compare_models()

    # Save all results
    pipeline.save_results()

    print("\n✅ Full training complete!")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Train MGEL models with Real BERT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast training (NBLR only, single dataset)
  python train.py --mode quick --dataset twitter

  # Best model (Combined datasets with Real BERT)
  python train.py --mode combined --ensemble --epochs 3

  # Full paper reproduction (all datasets + Real BERT)
  python train.py --mode full --epochs 5
        """
    )

    parser.add_argument('--mode', 
                       choices=['quick', 'full', 'combined'],
                       default='quick',
                       help='Training mode to run')

    parser.add_argument('--dataset', 
                       type=str, 
                       default='twitter',
                       help='Dataset name for quick mode')

    parser.add_argument('--ensemble', 
                       action='store_true',
                       help='Use full MGEL ensemble with Real BERT')

    # CHANGE 7: Added BERT-specific arguments
    parser.add_argument('--epochs',
                       type=int,
                       default=3,
                       help='Number of epochs for BERT training (default: 3)')
    
    parser.add_argument('--batch-size',
                       type=int,
                       default=8,
                       help='Batch size for BERT training (default: 8)')

    args = parser.parse_args()

    # Print configuration
    print("="*70)
    print("🚀 MGEL TRAINING PIPELINE")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Ensemble: {'Yes (Real BERT)' if args.ensemble else 'No (NBLR only)'}")
    if args.ensemble:
        print(f"BERT Epochs: {args.epochs}")
        print(f"BERT Batch Size: {args.batch_size}")
        print(f"Transformers Available: {'✅ Yes' if TRANSFORMERS_AVAILABLE else '❌ No'}")
    print("="*70)

    # Run selected mode
    if args.mode == 'quick':
        pipeline = MGELTrainingPipeline()
        pipeline.load_all_datasets()
        pipeline.train_single_dataset(
            args.dataset, 
            use_ensemble=args.ensemble,
            bert_epochs=args.epochs,
            bert_batch_size=args.batch_size
        )
        pipeline.save_results()

    elif args.mode == 'combined':
        pipeline = MGELTrainingPipeline()
        pipeline.load_all_datasets()
        pipeline.train_combined_datasets(
            use_ensemble=args.ensemble,
            bert_epochs=args.epochs,
            bert_batch_size=args.batch_size
        )
        pipeline.save_results()

    elif args.mode == 'full':
        full_training_example()

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\n📖 Next steps:")
    print("   1. Check results/ directory for saved models and metrics")
    print("   2. Load model: pickle.load(open('results/models/...pkl', 'rb'))")
    print("   3. Use for predictions on new data")
    print("="*70)