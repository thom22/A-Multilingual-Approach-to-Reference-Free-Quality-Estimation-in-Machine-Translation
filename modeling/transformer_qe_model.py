"""
Transformer-Based Quality Estimation Model
Fine-tunes XLM-RoBERTa for MT quality prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("ERROR: transformers library not available")


class QEDataset(Dataset):
    """Dataset for Quality Estimation with real data"""
    
    def __init__(self, df, tokenizer, max_length=128, use_cleaned=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_cleaned = use_cleaned
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Use cleaned or original text
        src_col = 'src_clean' if self.use_cleaned and 'src_clean' in self.df.columns else 'src'
        mt_col = 'mt_clean' if self.use_cleaned and 'mt_clean' in self.df.columns else 'mt'
        
        # Combine source and translation
        text = f"{row[src_col]} [SEP] {row[mt_col]}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(row['score'], dtype=torch.float)
        }


class XLMRQEModel(nn.Module):
    """XLM-RoBERTa based Quality Estimation model"""
    
    def __init__(self, model_name='xlm-roberta-base', dropout=0.1):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Regression head with better architecture
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.Tanh(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Predict quality score
        score = self.regressor(cls_output)
        
        return score.squeeze(-1)


class TransformerQETrainer:
    """Trainer for transformer QE model"""
    
    def __init__(self, model_name='xlm-roberta-base', device=None):
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")
        
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = XLMRQEModel(model_name).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def load_data(self, data_dir='data'):
        """Load the real dataset"""
        
        data_dir = Path(data_dir)
        
        print("\nLoading datasets...")
        train_df = pd.read_csv(data_dir / 'train_combined.csv')
        val_df = pd.read_csv(data_dir / 'validation_combined.csv')
        test_df = pd.read_csv(data_dir / 'test_combined.csv')
        
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def prepare_dataloaders(self, train_df, val_df, test_df, 
                           batch_size=16, max_length=128, use_cleaned=True):
        """Prepare data loaders"""
        
        print("\nPreparing data loaders...")
        print(f"  Batch size: {batch_size}")
        print(f"  Max length: {max_length}")
        print(f"  Use cleaned text: {use_cleaned}")
        
        train_dataset = QEDataset(train_df, self.tokenizer, max_length, use_cleaned)
        val_dataset = QEDataset(val_df, self.tokenizer, max_length, use_cleaned)
        test_dataset = QEDataset(test_df, self.tokenizer, max_length, use_cleaned)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                       shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                      shuffle=False, num_workers=0)
        
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")
    
    def train_epoch(self, optimizer, scheduler, criterion):
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            predictions = self.model(input_ids, attention_mask)
            loss = criterion(predictions, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        
        # Compute metrics
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        pearson = pearsonr(labels, predictions)[0]
        
        return avg_loss, pearson
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                predictions = self.model(input_ids, attention_mask)
                loss = criterion(predictions, labels)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'pearson': pearsonr(labels, predictions)[0],
            'spearman': spearmanr(labels, predictions)[0],
            'rmse': np.sqrt(mean_squared_error(labels, predictions)),
            'mae': mean_absolute_error(labels, predictions)
        }
        
        return metrics, predictions, labels
    
    def train(self, epochs=3, learning_rate=2e-5, warmup_steps=100):
        """Train the model"""
        
        print("\n" + "="*80)
        print("TRAINING TRANSFORMER MODEL")
        print("="*80)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(self.train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_pearson = -1
        history = {
            'train_loss': [], 'train_pearson': [],
            'val_loss': [], 'val_pearson': []
        }
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_pearson = self.train_epoch(optimizer, scheduler, criterion)
            history['train_loss'].append(train_loss)
            history['train_pearson'].append(train_pearson)
            
            # Validate
            val_metrics, _, _ = self.evaluate(self.val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_pearson'].append(val_metrics['pearson'])
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Pearson: {train_pearson:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Pearson: {val_metrics['pearson']:.4f}")
            print(f"  Val Spearman: {val_metrics['spearman']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
            
            # Save best model
            if val_metrics['pearson'] > best_val_pearson:
                best_val_pearson = val_metrics['pearson']
                torch.save(self.model.state_dict(), 'results/best_transformer_model.pt')
                print(f"  ✓ New best model (Val Pearson: {best_val_pearson:.4f})")
        
        return history
    
    def final_evaluation(self):
        """Final evaluation on all splits"""
        
        print("\n" + "="*80)
        print("FINAL EVALUATION")
        print("="*80)
        
        # Load best model
        self.model.load_state_dict(torch.load('results/best_transformer_model.pt'))
        
        results = {}
        
        for split_name, loader in [('Val', self.val_loader), ('Test', self.test_loader)]:
            metrics, predictions, labels = self.evaluate(loader)
            results[split_name] = {
                'metrics': metrics,
                'predictions': predictions,
                'labels': labels
            }
            
            print(f"\n{split_name} Set:")
            print(f"  Pearson:  {metrics['pearson']:.4f}")
            print(f"  Spearman: {metrics['spearman']:.4f}")
            print(f"  RMSE:     {metrics['rmse']:.4f}")
            print(f"  MAE:      {metrics['mae']:.4f}")
        
        return results
    
    def plot_results(self, history, results):
        """Create result visualizations"""
        
        figures_dir = Path('figures')
        
        # Training history
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch', fontweight='bold')
        axes[0].set_ylabel('Loss', fontweight='bold')
        axes[0].set_title('Training and Validation Loss', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Pearson
        axes[1].plot(epochs, history['train_pearson'], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, history['val_pearson'], 'r-', label='Val', linewidth=2)
        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('Pearson Correlation', fontweight='bold')
        axes[1].set_title('Training and Validation Pearson', fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(figures_dir / '08_transformer_training_history.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: 08_transformer_training_history.png")
        plt.close()
        
        # Predictions
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Transformer Model Predictions', fontsize=16, fontweight='bold')
        
        for i, (split_name, data) in enumerate(results.items()):
            # Scatter plot
            ax = axes[i, 0]
            ax.scatter(data['labels'], data['predictions'], alpha=0.3, s=10)
            lims = [min(data['labels'].min(), data['predictions'].min()),
                   max(data['labels'].max(), data['predictions'].max())]
            ax.plot(lims, lims, 'r--', lw=2)
            ax.set_xlabel('Actual Score', fontweight='bold')
            ax.set_ylabel('Predicted Score', fontweight='bold')
            pearson = data['metrics']['pearson']
            ax.set_title(f'{split_name} Set (Pearson: {pearson:.4f})', fontweight='bold')
            ax.grid(alpha=0.3)
            
            # Residual plot
            ax = axes[i, 1]
            residuals = data['labels'] - data['predictions']
            ax.scatter(data['predictions'], residuals, alpha=0.3, s=10)
            ax.axhline(0, color='r', linestyle='--', lw=2)
            ax.set_xlabel('Predicted Score', fontweight='bold')
            ax.set_ylabel('Residuals', fontweight='bold')
            ax.set_title(f'{split_name} Residuals', fontweight='bold')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(figures_dir / '09_transformer_predictions.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: 09_transformer_predictions.png")
        plt.close()
    
    def save_results(self, results):
        """Save final results"""
        
        results_df = pd.DataFrame([
            {
                'Model': 'XLM-RoBERTa',
                'Val_Pearson': results['Val']['metrics']['pearson'],
                'Val_Spearman': results['Val']['metrics']['spearman'],
                'Val_RMSE': results['Val']['metrics']['rmse'],
                'Val_MAE': results['Val']['metrics']['mae'],
                'Test_Pearson': results['Test']['metrics']['pearson'],
                'Test_Spearman': results['Test']['metrics']['spearman'],
                'Test_RMSE': results['Test']['metrics']['rmse'],
                'Test_MAE': results['Test']['metrics']['mae']
            }
        ])
        
        results_df.to_csv('results/transformer_results.csv', index=False)
        print(f"\n✓ Saved results to results/transformer_results.csv")


def main():
    """Main execution"""
    
    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: transformers library not available")
        print("Install with: pip install transformers torch")
        return
    
    print("="*80)
    print("TRANSFORMER-BASED QUALITY ESTIMATION")
    print("="*80)
    
    # Initialize trainer
    trainer = TransformerQETrainer(model_name='xlm-roberta-base')
    
    # Load data
    train_df, val_df, test_df = trainer.load_data()
    
    # Prepare dataloaders
    trainer.prepare_dataloaders(train_df, val_df, test_df, 
                               batch_size=16, max_length=128, use_cleaned=True)
    
    # Train
    history = trainer.train(epochs=3, learning_rate=2e-5)
    
    # Final evaluation
    results = trainer.final_evaluation()
    
    # Plot and save
    trainer.plot_results(history, results)
    trainer.save_results(results)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    main()
