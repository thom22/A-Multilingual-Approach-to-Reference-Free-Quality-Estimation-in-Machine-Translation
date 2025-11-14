"""
Advanced Preprocessing with Multilingual Embeddings
Extracts features using LaBSE and XLM-RoBERTa embeddings
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import embedding models
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available")


class EmbeddingExtractor:
    """Extract multilingual embeddings for MT Quality Estimation"""
    
    def __init__(self, use_cleaned_text=True):
        """
        Args:
            use_cleaned_text: If True, use src_clean/mt_clean; otherwise use src/mt
        """
        self.use_cleaned_text = use_cleaned_text
        self.labse_model = None
        self.xlmr_model = None
        self.xlmr_tokenizer = None
        self.device = 'cuda' if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {self.device}")
    
    def load_labse(self):
        """Load LaBSE model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("⚠ LaBSE not available (install sentence-transformers)")
            return False
        
        try:
            print("Loading LaBSE model...")
            self.labse_model = SentenceTransformer('sentence-transformers/LaBSE')
            print("✓ LaBSE loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Error loading LaBSE: {e}")
            return False
    
    def load_xlmr(self):
        """Load XLM-RoBERTa model"""
        if not TRANSFORMERS_AVAILABLE:
            print("⚠ XLM-R not available (install transformers and torch)")
            return False
        
        try:
            print("Loading XLM-RoBERTa model...")
            self.xlmr_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
            self.xlmr_model = AutoModel.from_pretrained('xlm-roberta-base').to(self.device)
            self.xlmr_model.eval()
            print("✓ XLM-RoBERTa loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Error loading XLM-R: {e}")
            return False
    
    def extract_labse_embeddings(self, texts, batch_size=32):
        """Extract LaBSE embeddings"""
        if self.labse_model is None:
            return None
        
        print(f"  Extracting LaBSE embeddings for {len(texts)} texts...")
        embeddings = self.labse_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def extract_xlmr_embeddings(self, texts, batch_size=16, max_length=128):
        """Extract XLM-RoBERTa embeddings"""
        if self.xlmr_model is None or self.xlmr_tokenizer is None:
            return None
        
        print(f"  Extracting XLM-R embeddings for {len(texts)} texts...")
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="XLM-R batches"):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = self.xlmr_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings
                outputs = self.xlmr_model(**inputs)
                # Use [CLS] token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        return embeddings
    
    def compute_embedding_features(self, src_emb, mt_emb):
        """
        Compute features from source and MT embeddings
        
        Returns:
            numpy array of features
        """
        features = []
        
        for src, mt in zip(src_emb, mt_emb):
            # Cosine similarity
            cos_sim = np.dot(src, mt) / (np.linalg.norm(src) * np.linalg.norm(mt) + 1e-8)
            
            # Euclidean distance
            euclidean_dist = np.linalg.norm(src - mt)
            
            # Manhattan distance
            manhattan_dist = np.sum(np.abs(src - mt))
            
            # Element-wise statistics
            diff = src - mt
            abs_diff = np.abs(diff)
            
            feat_vector = [
                cos_sim,
                euclidean_dist,
                manhattan_dist,
                np.mean(abs_diff),
                np.std(abs_diff),
                np.min(abs_diff),
                np.max(abs_diff),
                np.median(abs_diff)
            ]
            
            features.append(feat_vector)
        
        return np.array(features)
    
    def compute_simple_features(self, df):
        """Compute simple linguistic features"""
        
        print("  Computing simple linguistic features...")
        
        # Determine which columns to use
        src_col = 'src_clean' if self.use_cleaned_text else 'src'
        mt_col = 'mt_clean' if self.use_cleaned_text else 'mt'
        
        features = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Simple features"):
            src = str(row[src_col])
            mt = str(row[mt_col])
            
            # Length features
            src_len = len(src)
            mt_len = len(mt)
            len_ratio = mt_len / (src_len + 1e-8)
            len_diff = abs(src_len - mt_len)
            
            # Token features
            src_tokens = len(src.split())
            mt_tokens = len(mt.split())
            token_ratio = mt_tokens / (src_tokens + 1e-8)
            token_diff = abs(src_tokens - mt_tokens)
            
            # Character overlap
            src_chars = set(src.lower())
            mt_chars = set(mt.lower())
            char_overlap = len(src_chars & mt_chars) / (len(src_chars | mt_chars) + 1e-8)
            
            # Word overlap (simple)
            src_words = set(src.lower().split())
            mt_words = set(mt.lower().split())
            word_overlap = len(src_words & mt_words) / (len(src_words | mt_words) + 1e-8)
            
            # Average word length
            avg_src_word_len = src_len / (src_tokens + 1e-8)
            avg_mt_word_len = mt_len / (mt_tokens + 1e-8)
            
            feat_vector = [
                src_len,
                mt_len,
                len_ratio,
                len_diff,
                src_tokens,
                mt_tokens,
                token_ratio,
                token_diff,
                char_overlap,
                word_overlap,
                avg_src_word_len,
                avg_mt_word_len,
                abs(len_ratio - 1.0),  # deviation from perfect ratio
                abs(token_ratio - 1.0)
            ]
            
            features.append(feat_vector)
        
        return np.array(features)
    
    def process_dataset(self, df, extract_labse=True, extract_xlmr=True):
        """
        Process dataset and extract all features
        
        Returns:
            Dictionary with feature arrays and names
        """
        print(f"\nProcessing {len(df)} samples...")
        
        # Determine which text columns to use
        src_col = 'src_clean' if self.use_cleaned_text else 'src'
        mt_col = 'mt_clean' if self.use_cleaned_text else 'mt'
        
        print(f"Using text from: {src_col}, {mt_col}")
        
        src_texts = df[src_col].fillna('').astype(str).tolist()
        mt_texts = df[mt_col].fillna('').astype(str).tolist()
        
        all_features = []
        feature_names = []
        
        # 1. Simple features
        print("\n1. Simple Features:")
        simple_feats = self.compute_simple_features(df)
        all_features.append(simple_feats)
        simple_feat_names = [
            'src_len', 'mt_len', 'len_ratio', 'len_diff',
            'src_tokens', 'mt_tokens', 'token_ratio', 'token_diff',
            'char_overlap', 'word_overlap',
            'avg_src_word_len', 'avg_mt_word_len',
            'len_ratio_dev', 'token_ratio_dev'
        ]
        feature_names.extend(simple_feat_names)
        print(f"  ✓ Extracted {simple_feats.shape[1]} simple features")
        
        # 2. LaBSE embeddings
        if extract_labse and self.labse_model is not None:
            print("\n2. LaBSE Embeddings:")
            src_labse = self.extract_labse_embeddings(src_texts)
            mt_labse = self.extract_labse_embeddings(mt_texts)
            
            if src_labse is not None and mt_labse is not None:
                labse_feats = self.compute_embedding_features(src_labse, mt_labse)
                all_features.append(labse_feats)
                labse_feat_names = [
                    'labse_cos_sim', 'labse_euclidean', 'labse_manhattan',
                    'labse_diff_mean', 'labse_diff_std', 'labse_diff_min',
                    'labse_diff_max', 'labse_diff_median'
                ]
                feature_names.extend(labse_feat_names)
                print(f"  ✓ Extracted {labse_feats.shape[1]} LaBSE features")
        
        # 3. XLM-R embeddings
        if extract_xlmr and self.xlmr_model is not None:
            print("\n3. XLM-RoBERTa Embeddings:")
            src_xlmr = self.extract_xlmr_embeddings(src_texts)
            mt_xlmr = self.extract_xlmr_embeddings(mt_texts)
            
            if src_xlmr is not None and mt_xlmr is not None:
                xlmr_feats = self.compute_embedding_features(src_xlmr, mt_xlmr)
                all_features.append(xlmr_feats)
                xlmr_feat_names = [
                    'xlmr_cos_sim', 'xlmr_euclidean', 'xlmr_manhattan',
                    'xlmr_diff_mean', 'xlmr_diff_std', 'xlmr_diff_min',
                    'xlmr_diff_max', 'xlmr_diff_median'
                ]
                feature_names.extend(xlmr_feat_names)
                print(f"  ✓ Extracted {xlmr_feats.shape[1]} XLM-R features")
        
        # Combine all features
        combined_features = np.hstack(all_features)
        
        print(f"\n✓ Total features: {combined_features.shape[1]}")
        
        return {
            'features': combined_features,
            'feature_names': feature_names
        }


def main():
    """Main preprocessing pipeline"""
    
    print("="*80)
    print("MT QUALITY ESTIMATION - ADVANCED PREPROCESSING")
    print("="*80)
    
    # Load data
    data_dir = Path('data')
    
    print("\nLoading datasets...")
    train_df = pd.read_csv(data_dir / 'train_combined.csv')
    val_df = pd.read_csv(data_dir / 'validation_combined.csv')
    test_df = pd.read_csv(data_dir / 'test_combined.csv')
    
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Initialize extractor
    print("\n" + "="*80)
    print("Initializing embedding models...")
    print("="*80)
    
    extractor = EmbeddingExtractor(use_cleaned_text=True)
    
    # Load models
    has_labse = extractor.load_labse()
    has_xlmr = extractor.load_xlmr()
    
    if not has_labse and not has_xlmr:
        print("\n⚠ No embedding models available!")
        print("Will use simple features only.")
        print("To use embeddings, install: pip install sentence-transformers transformers torch")
    
    # Process each split
    for split_name, df in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        
        print("\n" + "="*80)
        print(f"Processing {split_name.upper()} split")
        print("="*80)
        
        result = extractor.process_dataset(
            df,
            extract_labse=has_labse,
            extract_xlmr=has_xlmr
        )
        
        # Create feature DataFrame
        feature_df = pd.DataFrame(
            result['features'],
            columns=result['feature_names']
        )
        
        # Add metadata
        feature_df['score'] = df['score'].values
        feature_df['lang_pair'] = df['lang_pair'].values
        feature_df['resource_level'] = df['resource_level'].values
        
        # Save
        output_file = data_dir / f'{split_name}_features.csv'
        feature_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved {split_name} features to {output_file}")
        print(f"  Shape: {feature_df.shape}")
    
    # Save feature names
    feature_info = pd.DataFrame({
        'feature_name': result['feature_names'],
        'feature_index': range(len(result['feature_names']))
    })
    feature_info.to_csv(data_dir / 'feature_names.csv', index=False)
    print(f"\n✓ Saved feature names to {data_dir / 'feature_names.csv'}")
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
