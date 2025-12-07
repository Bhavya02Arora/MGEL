"""
MGEL: Memory-Optimized Version
Fixes CUDA OOM errors with gradient accumulation, smaller batches, and memory management
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, matthews_corrcoef
from typing import List, Dict
import warnings
import gc
warnings.filterwarnings('ignore')

# Check if transformers is available
try:
    import torch
    from transformers import (
        BertTokenizer,
        BertForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
        EarlyStoppingCallback
    )
    from torch.utils.data import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  WARNING: transformers not installed. Using simulator instead.")


class MultihotByteEncoder:
    """Multihot byte-level encoding"""

    def __init__(self, max_length: int = 300):
        self.max_length = max_length
        self.byte_vocab = {}
        self.vocab_size = 0

    def build_vocab(self, texts: List[str]):
        byte_set = set()
        for text in texts:
            encoded = text.encode('utf-8')
            for byte in encoded:
                byte_set.add(byte)

        self.byte_vocab = {byte: idx for idx, byte in enumerate(sorted(byte_set))}
        self.vocab_size = len(self.byte_vocab)
        print(f"Byte vocabulary size: {self.vocab_size}")

    def encode_text(self, text: str) -> np.ndarray:
        encoded_bytes = text.encode('utf-8')
        matrix = np.zeros((self.vocab_size, self.max_length), dtype=np.int8)

        char_idx = 0
        byte_idx = 0

        while byte_idx < len(encoded_bytes) and char_idx < self.max_length:
            byte_val = encoded_bytes[byte_idx]

            if byte_val < 0x80:
                char_bytes = [byte_val]
                byte_idx += 1
            elif byte_val < 0xE0:
                char_bytes = list(encoded_bytes[byte_idx:byte_idx+2])
                byte_idx += 2
            elif byte_val < 0xF0:
                char_bytes = list(encoded_bytes[byte_idx:byte_idx+3])
                byte_idx += 3
            else:
                char_bytes = list(encoded_bytes[byte_idx:byte_idx+4])
                byte_idx += 4

            for b in char_bytes:
                if b in self.byte_vocab:
                    matrix[self.byte_vocab[b], char_idx] += 1

            char_idx += 1

        return matrix

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.encode_text(text).flatten() for text in texts])


class NBLRClassifier:
    """Naive Bayes Logistic Regression"""

    def __init__(self, word_ngram_range=(1, 2), char_ngram_range=(1, 4),
                 max_features=None, C=1.0):
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.max_features = max_features
        self.C = C

        self.word_vectorizer = TfidfVectorizer(
            ngram_range=word_ngram_range,
            max_features=max_features,
            analyzer='word',
            lowercase=True
        )

        self.char_vectorizer = TfidfVectorizer(
            ngram_range=char_ngram_range,
            max_features=max_features,
            analyzer='char',
            lowercase=True
        )

        self.lr_model = None
        self.nb_ratios = None

    def compute_nb_ratios(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        pos_mask = y == 1
        neg_mask = y == 0

        pos_avg = np.array(X[pos_mask].mean(axis=0)).flatten()
        neg_avg = np.array(X[neg_mask].mean(axis=0)).flatten()

        smoothing = 1e-10
        pos_avg += smoothing
        neg_avg += smoothing

        nb_ratios = np.log(pos_avg / neg_avg)
        return nb_ratios

    def fit(self, texts: List[str], y: np.ndarray):
        print("Extracting word n-grams...")
        X_word = self.word_vectorizer.fit_transform(texts)

        print("Extracting character n-grams...")
        X_char = self.char_vectorizer.fit_transform(texts)

        from scipy.sparse import hstack
        X = hstack([X_word, X_char])

        print(f"Total features: {X.shape[1]}")

        print("Computing Naive Bayes ratios...")
        self.nb_ratios = self.compute_nb_ratios(X, y)

        X_scaled = X.multiply(self.nb_ratios)

        print("Training logistic regression...")
        self.lr_model = LogisticRegression(C=self.C, max_iter=1000, random_state=42)
        self.lr_model.fit(X_scaled, y)

        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X_word = self.word_vectorizer.transform(texts)
        X_char = self.char_vectorizer.transform(texts)

        from scipy.sparse import hstack
        X = hstack([X_word, X_char])
        X_scaled = X.multiply(self.nb_ratios)

        return self.lr_model.predict_proba(X_scaled)[:, 1]

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(texts)
        return (proba >= threshold).astype(int)


# ==================== MEMORY-OPTIMIZED BERT ====================

if TRANSFORMERS_AVAILABLE:

    class ToxicCommentsDataset(Dataset):
        """PyTorch Dataset for toxic comments"""

        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = int(self.labels[idx])

            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }


    class EnhancedBERT:
        """
        🚀 MEMORY-OPTIMIZED Enhanced BERT

        Key optimizations:
        1. Smaller batch sizes (2-4)
        2. Gradient accumulation (simulates larger batches)
        3. Aggressive memory cleanup
        4. FP16 mixed precision training
        5. Gradient checkpointing for large models
        """

        def __init__(self, granularity='subword', model_name='bert-base-uncased',
                     max_length=128, device=None):
            self.granularity = granularity
            self.max_length = max_length
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

            print(f"🔧 Initializing Memory-Optimized BERT ({granularity})")
            print(f"   Device: {self.device}")

            # Clear cache before starting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            self.tokenizer = BertTokenizer.from_pretrained(
                model_name,
                do_lower_case=(granularity not in ['byte', 'character'])
            )

            self.model = BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                problem_type="single_label_classification"
            )

            # Enable gradient checkpointing to save memory
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("   ✅ Gradient checkpointing enabled")

            self.model.to(self.device)
            self.trainer = None

        def preprocess_text(self, text):
            """Preprocess text based on granularity"""
            if self.granularity == 'byte':
                return ' '.join([f'{b:02x}' for b in text.encode('utf-8')])
            elif self.granularity == 'character':
                return ' '.join(list(text))
            else:
                return text

        def fit(self, texts, labels, val_texts=None, val_labels=None,
                epochs=3, batch_size=2, learning_rate=2e-5,
                gradient_accumulation_steps=4, early_stopping_patience=3):
            """
            Fine-tune BERT with memory optimizations and early stopping

            Args:
                gradient_accumulation_steps: Accumulate gradients over N steps
                    Effective batch size = batch_size * gradient_accumulation_steps
                    Use 4-8 for low memory GPUs
                early_stopping_patience: Stop if no improvement for N epochs (default: 3)
            """
            print(f"\n🎓 Training Memory-Optimized BERT...")
            print(f"   Training samples: {len(texts)}")
            print(f"   Epochs: {epochs}")
            print(f"   Batch size: {batch_size}")
            print(f"   Gradient accumulation: {gradient_accumulation_steps}")
            print(f"   Effective batch size: {batch_size * gradient_accumulation_steps}")

            # Clear memory before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

                # Print memory info
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                print(f"   GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

            # Preprocess
            processed_texts = [self.preprocess_text(t) for t in texts]

            # Create dataset
            train_dataset = ToxicCommentsDataset(
                processed_texts,
                labels,
                self.tokenizer,
                self.max_length
            )

            eval_dataset = None
            if val_texts is not None and val_labels is not None:
                processed_val = [self.preprocess_text(t) for t in val_texts]
                eval_dataset = ToxicCommentsDataset(
                    processed_val,
                    val_labels,
                    self.tokenizer,
                    self.max_length
                )
                print(f"   Validation samples: {len(val_texts)}")

            # MEMORY-OPTIMIZED Training arguments with EARLY STOPPING
            training_args_dict = {
                'output_dir': './results',
                'num_train_epochs': epochs,
                'per_device_train_batch_size': batch_size,
                'per_device_eval_batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'learning_rate': learning_rate,
                'weight_decay': 0.01,
                'logging_dir': './logs',
                'logging_steps': 50,
                'save_strategy': 'epoch',
                'warmup_steps': 100,
                'disable_tqdm': False,
                'max_grad_norm': 1.0,  # Gradient clipping
                'dataloader_num_workers': 0,  # Disable multiprocessing to save memory
            }

            # Add evaluation strategy and EARLY STOPPING
            if eval_dataset:
                training_args_dict['eval_strategy'] = 'epoch'
                training_args_dict['load_best_model_at_end'] = True
                training_args_dict['metric_for_best_model'] = 'loss'  # Monitor validation loss
                training_args_dict['greater_is_better'] = False  # Lower loss is better
                print("   ✅ Early stopping enabled (patience=3 epochs)")
            else:
                training_args_dict['eval_strategy'] = 'no'
                print("   ⚠️  No validation set - early stopping disabled")

            # FP16 for GPU (saves ~50% memory)
            if torch.cuda.is_available():
                training_args_dict['fp16'] = True
                print("   ✅ FP16 mixed precision enabled")

            training_args = TrainingArguments(**training_args_dict)

            # Setup callbacks
            callbacks = []
            if eval_dataset:
                # Add early stopping callback
                early_stopping = EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_threshold=0.0  # Any improvement counts
                )
                callbacks.append(early_stopping)

            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=DataCollatorWithPadding(self.tokenizer),
                callbacks=callbacks,  # Add callbacks here
            )

            # Train with error handling
            print("\n🚀 Starting BERT training...")
            try:
                self.trainer.train()
                print("✅ BERT training complete!")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("\n❌ Still out of memory!")
                    print("Try these solutions:")
                    print("  1. Reduce batch_size to 1")
                    print("  2. Increase gradient_accumulation_steps to 8-16")
                    print("  3. Reduce max_length to 64 or 32")
                    print("  4. Use a smaller model: 'distilbert-base-uncased'")
                    raise
                else:
                    raise
            finally:
                # Clean up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

            return self

        def predict_proba(self, texts, batch_size=8):
            """Predict with memory-efficient batching"""
            self.model.eval()
            all_probs = []

            # Process in batches to avoid OOM
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                processed_texts = [self.preprocess_text(t) for t in batch_texts]

                encodings = self.tokenizer(
                    processed_texts,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )

                encodings = {k: v.to(self.device) for k, v in encodings.items()}

                with torch.no_grad():
                    outputs = self.model(**encodings)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                    all_probs.append(probs[:, 1].cpu().numpy())

                # Clean up batch memory
                del encodings, outputs, logits, probs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return np.concatenate(all_probs)

        def predict(self, texts, threshold=0.5, batch_size=8):
            """Predict class labels"""
            proba = self.predict_proba(texts, batch_size=batch_size)
            return (proba >= threshold).astype(int)

        def save_model(self, path):
            """Save model"""
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"💾 Model saved to: {path}")

        def load_model(self, path):
            """Load model"""
            self.model = BertForSequenceClassification.from_pretrained(path)
            self.tokenizer = BertTokenizer.from_pretrained(path)
            self.model.to(self.device)
            print(f"📂 Model loaded from: {path}")

else:
    # Fallback simulator
    class EnhancedBERT:
        """Fallback simulator"""

        def __init__(self, granularity='subword', **kwargs):
            self.granularity = granularity
            self.model = None
            print(f"⚠️  Using simulator for {granularity}")

        def fit(self, texts, labels, **kwargs):
            print(f"Training simulator ({self.granularity})...")
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
            X = vectorizer.fit_transform(texts)

            self.model = LogisticRegression(C=1.0, max_iter=1000)
            self.model.fit(X, labels)
            self.vectorizer = vectorizer
            return self

        def predict_proba(self, texts):
            X = self.vectorizer.transform(texts)
            return self.model.predict_proba(X)[:, 1]


# ==================== ENSEMBLE ====================

class MGELEnsemble:
    """Multigrained Ensemble Learning"""

    def __init__(self):
        self.models = {}
        self.weights = None

    def add_model(self, name: str, model):
        self.models[name] = model

    def optimize_weights_grid_search(self, texts: List[str], y: np.ndarray,
                                     step: float = 0.1):
        print("Optimizing ensemble weights...")

        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(texts)

        best_score = -np.inf
        best_weights = None

        for w_bert in np.arange(0, 1.1, step):
            w_nblr = 1 - w_bert

            n_bert_models = len([k for k in self.models.keys() if 'bert' in k.lower()])
            w_per_bert = w_bert / n_bert_models if n_bert_models > 0 else 0

            ensemble_pred = np.zeros(len(texts))

            for name, pred in predictions.items():
                if 'nblr' in name.lower():
                    ensemble_pred += w_nblr * pred
                else:
                    ensemble_pred += w_per_bert * pred

            score = roc_auc_score(y, ensemble_pred)

            if score > best_score:
                best_score = score
                best_weights = {'bert': w_bert, 'nblr': w_nblr}

        self.weights = best_weights
        print(f"Best weights: {best_weights}, Score: {best_score:.4f}")

        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Weights not optimized. Call optimize_weights first.")

        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(texts)

        n_bert_models = len([k for k in self.models.keys() if 'bert' in k.lower()])
        w_per_bert = self.weights['bert'] / n_bert_models if n_bert_models > 0 else 0

        ensemble_pred = np.zeros(len(texts))
        for name, pred in predictions.items():
            if 'nblr' in name.lower():
                ensemble_pred += self.weights['nblr'] * pred
            else:
                ensemble_pred += w_per_bert * pred

        return ensemble_pred

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(texts)
        return (proba >= threshold).astype(int)


def evaluate_model(y_true: np.ndarray, y_pred_proba: np.ndarray,
                   threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate model"""
    auc_roc = roc_auc_score(y_true, y_pred_proba)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = auc(recall, precision)

    y_pred = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        'AUC@ROC': auc_roc,
        'AUC@PR': auc_pr,
        'F1': f1,
        'MCC': mcc
    }


# ==================== DEMO ====================

def demo_memory_optimized():
    """Demo with memory optimizations"""
    print("=" * 70)
    print("🚀 MEMORY-OPTIMIZED MGEL DEMO")
    print("=" * 70)

    if not TRANSFORMERS_AVAILABLE:
        print("\n⚠️  transformers not installed!")
        return

    # Smaller training set for demo
    clean_texts = [
        "I love this product, amazing quality!",
        "Great article, very informative",
        "Excellent work on this project",
        "Thank you so much for your help",
    ] * 3

    toxic_texts = [
        "You're an idiot, go away",
        "F*ck you and your stupid ideas",
        "I h@te you, you're trash",
        "You piece of sh1t",
    ] * 3

    texts_train = clean_texts + toxic_texts
    y_train = np.array([0]*len(clean_texts) + [1]*len(toxic_texts))

    texts_val = [
        "Nice work", "You're an @ss",
        "Good job", "Die in a fire"
    ]
    y_val = np.array([0, 1, 0, 1])

    print(f"\n📊 Data: {len(texts_train)} train, {len(texts_val)} val")

    # Train NBLR
    print("\n1. Training NBLR...")
    nblr = NBLRClassifier()
    nblr.fit(texts_train, y_train)

    # Train memory-optimized BERT
    print("\n2. Training Memory-Optimized BERT...")
    bert = EnhancedBERT(
        granularity='subword',
        max_length=64  # Shorter sequences = less memory
    )

    try:
        bert.fit(
            texts_train, y_train,
            val_texts=texts_val, val_labels=y_val,
            epochs=2,
            batch_size=2,  # Small batch
            gradient_accumulation_steps=4,  # Simulates batch_size=8
            early_stopping_patience=3  # Stop if no improvement for 3 epochs
        )
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n💡 Still OOM? Try these settings in your code:")
            print("   batch_size=1")
            print("   gradient_accumulation_steps=8")
            print("   max_length=32")
            print("   Or use 'distilbert-base-uncased' (smaller model)")
            return
        raise

    # Ensemble
    print("\n3. Creating ensemble...")
    mgel = MGELEnsemble()
    mgel.add_model('nblr', nblr)
    mgel.add_model('bert', bert)
    mgel.optimize_weights_grid_search(texts_val, y_val)

    # Test
    test_texts = [
        "You are wonderful!",
        "I h8te you so much",
        "Great content!"
    ]

    predictions = mgel.predict_proba(test_texts)
    labels = mgel.predict(test_texts)

    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)

    for text, prob, label in zip(test_texts, predictions, labels):
        emoji = "🔴" if label == 1 else "✅"
        label_text = "TOXIC" if label == 1 else "CLEAN"
        print(f"{emoji} {label_text} ({prob:.1%}): {text}")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         MGEL - Memory-Optimized for Low-Memory GPUs             ║
╚══════════════════════════════════════════════════════════════════╝

💡 Key Optimizations:
   • Batch size: 2 (instead of 8)
   • Gradient accumulation: 4 steps (effective batch = 8)
   • FP16 mixed precision (saves 50% memory)
   • Gradient checkpointing
   • Shorter sequences (max_length=64)
   • Aggressive memory cleanup
   • ✨ Early stopping (patience=3 epochs) to prevent overfitting
    """)

    if TRANSFORMERS_AVAILABLE:
        demo_memory_optimized()
    else:
        print("⚠️  Install transformers: pip install transformers torch")