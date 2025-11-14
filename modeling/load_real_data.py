"""
Data Loader for Real WMT20 MLQE Dataset
Loads the downloaded dataset from final_dataset folder
"""

import pandas as pd
from pathlib import Path
import numpy as np

class RealDataLoader:
    """Loads the actual WMT20 MLQE dataset from final_dataset folder"""
    
    def __init__(self, data_root='datasets/final_dataset'):
        self.data_root = Path(data_root)
        
        # Language pairs mapping
        self.language_pairs = {
            'en-de': 'high',     # English-German
            'ro-en': 'medium',   # Romanian-English
            'si-en': 'low'       # Sinhala-English
        }
        
    def load_split(self, lang_pair, split='train'):
        """
        Load a specific split for a language pair
        
        Args:
            lang_pair: 'en-de', 'ro-en', or 'si-en'
            split: 'train', 'test', or 'validation'
            
        Returns:
            DataFrame with loaded data
        """
        file_path = self.data_root / lang_pair / f"{lang_pair}_{split}.csv"
        
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            return None
        
        df = pd.read_csv(file_path)
        
        # Add metadata
        df['lang_pair'] = lang_pair
        df['split'] = split
        df['resource_level'] = self.language_pairs[lang_pair]
        
        # Use cleaned versions if available, otherwise use original
        if 'src_clean' not in df.columns:
            df['src_clean'] = df['src']
        if 'mt_clean' not in df.columns:
            df['mt_clean'] = df['mt']
            
        # Rename score column for consistency
        if 'z_mean' in df.columns and 'score' not in df.columns:
            df['score'] = df['z_mean']
        elif 'score' not in df.columns and 'mean' in df.columns:
            df['score'] = df['mean']
        
        return df
    
    def load_all_data(self):
        """
        Load all language pairs and splits
        
        Returns:
            Dictionary with train/validation/test DataFrames
        """
        all_data = {
            'train': [],
            'validation': [],
            'test': []
        }
        
        for lang_pair in self.language_pairs.keys():
            print(f"\nLoading {lang_pair}...")
            
            for split in ['train', 'validation', 'test']:
                df = self.load_split(lang_pair, split)
                
                if df is not None:
                    all_data[split].append(df)
                    print(f"  {split}: {len(df)} samples")
        
        # Combine all language pairs
        combined_data = {}
        for split in ['train', 'validation', 'test']:
            if all_data[split]:
                combined_data[split] = pd.concat(all_data[split], ignore_index=True)
                print(f"\nTotal {split}: {len(combined_data[split])} samples")
            else:
                combined_data[split] = None
        
        return combined_data
    
    def get_data_info(self):
        """Print information about the dataset"""
        print("\n" + "="*60)
        print("DATASET INFORMATION")
        print("="*60)
        
        for lang_pair in self.language_pairs.keys():
            print(f"\n{lang_pair} ({self.language_pairs[lang_pair]}-resource):")
            
            for split in ['train', 'validation', 'test']:
                df = self.load_split(lang_pair, split)
                if df is not None:
                    print(f"  {split:12s}: {len(df):5d} samples")
                    
                    # Show columns
                    if split == 'train':
                        print(f"  Columns: {', '.join(df.columns[:10])}...")
                        print(f"  Score range: [{df['score'].min():.3f}, {df['score'].max():.3f}]")


def main():
    """Test the data loader"""
    
    print("="*60)
    print("Testing Real Data Loader")
    print("="*60)
    
    # Initialize loader
    loader = RealDataLoader(data_root='datasets/final_dataset')
    
    # Get dataset info
    loader.get_data_info()
    
    # Load all data
    print("\n" + "="*60)
    print("Loading All Data")
    print("="*60)
    
    data = loader.load_all_data()
    
    # Save combined data for easy access
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    
    for split, df in data.items():
        if df is not None:
            output_file = output_dir / f'{split}_combined.csv'
            
            # Select key columns for modeling
            key_columns = ['src', 'mt', 'src_clean', 'mt_clean', 'score', 
                          'lang_pair', 'resource_level', 'split']
            
            # Keep only columns that exist
            cols_to_save = [col for col in key_columns if col in df.columns]
            df_save = df[cols_to_save]
            
            df_save.to_csv(output_file, index=False)
            print(f"Saved {split} data to {output_file}")
    
    print("\n" + "="*60)
    print("Data loading complete!")
    print("="*60)


if __name__ == "__main__":
    main()
