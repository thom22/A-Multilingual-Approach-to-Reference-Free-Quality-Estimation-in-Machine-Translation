# A-Multilingual-Approach-to-Reference-Free-Quality-Estimation-in-Machine-Translation


# Machine Translation Quality Estimation - Updated Implementation

**A comprehensive MT Quality Estimation system using real WMT20 MLQE data with advanced embeddings and multiple modeling approaches.**

## Overview

This implementation works with your **downloaded WMT20 MLQE dataset** and provides:
- Comprehensive EDA with publication-quality visualizations
-  Advanced preprocessing with **LaBSE** and **XLM-RoBERTa** embeddings
-  Comparison of **8+ regression models**
-  Transformer-based model (XLM-RoBERTa fine-tuning)
-  Support for 3 language pairs: **en-de** (high), **ro-en** (medium), **si-en** (low)

## Key Question: Should I Use Cleaned or Original Text?

**Recommendation: Use cleaned text (`src_clean`, `mt_clean`)**

**Why?**
1. **Better embeddings**: Cleaned text produces more consistent embeddings
2. **Faster processing**: Shorter text = faster tokenization
3. **Less noise**: Removes URLs, normalizes Unicode
4. **Your data prep already did this**: The notebook you shared already created these columns

**When to use original (`src`, `mt`)?**
- If you want to preserve exact formatting
- If URLs or special characters are meaningful
- For final production system comparison

**Current implementation**: All scripts default to using **cleaned text** but can be easily switched by changing `use_cleaned_text=False` in the code.

## Your Data Structure

```
datasets/final_dataset/
├── en-de/
│   ├── en-de_train.csv
│   ├── en-de_validation.csv
│   └── en-de_test.csv
├── ro-en/
│   ├── ro-en_train.csv
│   ├── ro-en_validation.csv
│   └── ro-en_test.csv
└── si-en/
    ├── si-en_train.csv
    ├── si-en_validation.csv
    └── si-en_test.csv
```

**CSV Columns** (from your data):
- `src` - Source text
- `mt` - Machine translation
- `src_clean` - Cleaned source (lowercased, normalized)
- `mt_clean` - Cleaned translation
- `score` or `z_mean` - Quality score (target variable)
- `lp` - Language pair
- Other metadata columns

## Quick Start

### 1. Install Dependencies

```bash
# Core packages (required)
pip install pandas numpy matplotlib seaborn scikit-learn scipy tqdm

# For embeddings (highly recommended)
pip install sentence-transformers

# For transformer model (recommended)
pip install torch transformers

# Optional: for XGBoost
pip install xgboost
```

### 2. Run the Complete Pipeline

```bash
python run_complete_pipeline.py
```

This will:
1. Load your data from `datasets/final_dataset/`
2. Perform comprehensive EDA
3. Extract features (simple + LaBSE + XLM-R)
4. Train 8+ regression models
5. Fine-tune XLM-RoBERTa transformer
6. Generate all visualizations

### 3. Or Run Step-by-Step

```bash
# Step 1: Load data
python load_real_data.py

# Step 2: EDA with visualizations
python enhanced_eda.py

# Step 3: Extract features with embeddings
python advanced_preprocessing.py

# Step 4: Train regression models
python regression_models.py

# Step 5: Train transformer
python transformer_qe_model.py
```

## What You'll Get

### Data Files (`data/`)
- `*_combined.csv` - Combined datasets (7K train + 1K val + 1K test per language)
- `*_features.csv` - Extracted features (14 simple + 8 LaBSE + 8 XLM-R = 30 features)
- `feature_names.csv` - Feature descriptions

### Visualizations (`figures/`)

**Figure 1: Dataset Overview** (`01_dataset_overview.png`)
- Sample distribution by language pair
- Resource level breakdown
- Score distributions
- Statistical summary table

**Figure 2: Length Analysis** (`02_length_analysis.png`)
- Source/MT length distributions
- Length ratio analysis
- Token count comparisons
- Source vs MT scatter plots

**Figure 3: Score Analysis** (`03_score_analysis.png`)
- Score vs length relationships
- Score vs length ratio
- Distribution by language pair
- Violin plots

**Figure 4: Correlation Analysis** (`04_correlation_analysis.png`)
- Feature correlation matrix
- Correlation with quality score

**Figure 5: Model Comparison** (`05_model_comparison.png`)
- Pearson/Spearman correlations
- RMSE and MAE comparison
- Ranked model performance

**Figure 6: Train/Val/Test** (`06_train_val_test_comparison.png`)
- Performance across splits
- Overfitting detection

**Figure 7: Best Model Predictions** (`07_best_model_predictions.png`)
- Scatter plots
- Residual plots
- All three splits

**Figure 8-9: Transformer Results**
- Training history curves
- Prediction scatter plots

### Results (`results/`)
- `model_comparison.csv` - All regression model metrics
- `transformer_results.csv` - Transformer model metrics
- `best_transformer_model.pt` - Trained model checkpoint

### Transformer Model:
- **XLM-RoBERTa**: Pearson ~0.55-0.75

*Note: Actual performance depends on data quality and hyperparameters*

##  Scripts Description

### 1. `load_real_data.py`
**What it does:**
- Loads your downloaded WMT20 data
- Combines all language pairs
- Creates train/val/test CSV files
- Validates data structure

**Output:** `data/*_combined.csv`

### 2. `enhanced_eda.py`
**What it does:**
- Comprehensive exploratory data analysis
- Creates 4 publication-quality figures
- Generates summary statistics
- Computes linguistic features

**Output:** 4 PNG figures, summary statistics

**Key insights:**
- Score distributions differ by language pair
- Length ratio correlates with quality
- Low-resource (si-en) shows more variance

### 3. `advanced_preprocessing.py`
**What it does:**
- Extracts 14 simple linguistic features
- Computes LaBSE embeddings (768-dim → 8 features)
- Computes XLM-R embeddings (768-dim → 8 features)
- Creates feature similarity metrics

**Output:** `data/*_features.csv` (30 total features)

**Features extracted:**
- Length-based: character/token counts, ratios
- Overlap-based: character/word overlap
- Embedding-based: cosine similarity, distances

### 4. `regression_models.py`
**What it does:**
- Trains 6+ regression models
- Compares performance metrics
- Creates visualization comparing all models
- Identifies best model

**Models compared:**
1. Linear Regression
2. Ridge (α=1.0)
3. ExtraTrees
4. Lasso
5. Random Forest
6. Gradient Boosting
7. XGBoost

