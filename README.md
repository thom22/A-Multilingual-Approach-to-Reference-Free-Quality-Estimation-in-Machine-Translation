# A-Multilingual-Approach-to-Reference-Free-Quality-Estimation-in-Machine-Translation


Quick Start
1. Install Dependencies
bash# Core packages (required)
pip install pandas numpy matplotlib seaborn scikit-learn scipy tqdm

# For embeddings (highly recommended)
pip install sentence-transformers

# For transformer model (recommended)
pip install torch transformers

# Optional: for XGBoost
pip install xgboost
2. Run the Complete Pipeline
bashpython run_complete_pipeline.py
This will:

Load your data from datasets/final_dataset/
Perform comprehensive EDA
Extract features (simple + LaBSE + XLM-R)
Train 8+ regression models
Fine-tune XLM-RoBERTa transformer
Generate all visualizations

Total time: ~40-60 minutes
3. Or Run Step-by-Step
bash# Step 1: Load data
python load_real_data.py

# Step 2: EDA with visualizations
python enhanced_eda.py

# Step 3: Extract features with embeddings
python advanced_preprocessing.py

# Step 4: Train regression models
python regression_models.py

# Step 5: Train transformer
python transformer_qe_model.py