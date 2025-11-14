"""
Comprehensive Regression Modeling for MT Quality Estimation
Compares multiple regression algorithms
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Try xgboost
try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not available. Install with: pip install xgboost")


class RegressionModelComparison:
    """Compare multiple regression models for QE"""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.feature_names = None

    def initialize_models(self):
        """Initialize regression models with default parameters"""

        print("Initializing models...")

        self.models = {
            "Ridge": Ridge(),
            # "Lasso": Lasso(),
            "Lasso": Lasso(alpha=0.01, max_iter=5000),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Extra Trees": ExtraTreesRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models["XGBoost"] = XGBRegressor(random_state=42)

        print(f"✓ Initialized {len(self.models)} models")

    def load_data(self, data_dir="data"):
        """Load preprocessed features"""

        data_dir = Path(data_dir)

        print("\nLoading feature data...")
        train_df = pd.read_csv(data_dir / "train_features.csv")
        val_df = pd.read_csv(data_dir / "validation_features.csv")
        test_df = pd.read_csv(data_dir / "test_features.csv")

        print(f"  Train: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")

        # Separate features and target
        meta_cols = ["score", "lang_pair", "resource_level"]
        feature_cols = [col for col in train_df.columns if col not in meta_cols]

        self.feature_names = feature_cols
        print(f"  Features: {len(feature_cols)}")

        X_train = train_df[feature_cols].values
        y_train = train_df["score"].values
        meta_train = train_df[meta_cols]

        X_val = val_df[feature_cols].values
        y_val = val_df["score"].values
        meta_val = val_df[meta_cols]

        X_test = test_df[feature_cols].values
        y_test = test_df["score"].values
        meta_test = test_df[meta_cols]

        # Check for NaN or inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "meta_train": meta_train,
            "X_val": X_val,
            "y_val": y_val,
            "meta_val": meta_val,
            "X_test": X_test,
            "y_test": y_test,
            "meta_test": meta_test,
        }

    def compute_metrics(self, y_true, y_pred):
        """Compute evaluation metrics"""

        # Clip predictions to valid range
        y_pred = np.clip(y_pred, -5, 5)  # Reasonable range for z-scores

        metrics = {
            "pearson": pearsonr(y_true, y_pred)[0],
            "spearman": spearmanr(y_true, y_pred)[0],
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

        return metrics

    def train_and_evaluate(self, data):
        """Train and evaluate all models"""

        print("\n" + "=" * 80)
        print("TRAINING AND EVALUATION")
        print("=" * 80)

        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(data["X_train"])
        X_val_scaled = self.scaler.transform(data["X_val"])
        X_test_scaled = self.scaler.transform(data["X_test"])

        results_list = []

        for model_name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training: {model_name}")
            print(f"{'='*60}")

            try:
                # Train
                print("  Training...")
                model.fit(X_train_scaled, data["y_train"])

                # Predict
                print("  Predicting...")
                train_pred = model.predict(X_train_scaled)
                val_pred = model.predict(X_val_scaled)
                test_pred = model.predict(X_test_scaled)

                # Evaluate
                print("  Evaluating...")
                train_metrics = self.compute_metrics(data["y_train"], train_pred)
                val_metrics = self.compute_metrics(data["y_val"], val_pred)
                test_metrics = self.compute_metrics(data["y_test"], test_pred)

                # Store results
                self.results[model_name] = {
                    "model": model,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "train_pred": train_pred,
                    "val_pred": val_pred,
                    "test_pred": test_pred,
                }

                # Print results
                print(f"\n  Results:")
                print(
                    f"    Train - Pearson: {train_metrics['pearson']:.4f}, RMSE: {train_metrics['rmse']:.4f}"
                )
                print(
                    f"    Val   - Pearson: {val_metrics['pearson']:.4f}, RMSE: {val_metrics['rmse']:.4f}"
                )
                print(
                    f"    Test  - Pearson: {test_metrics['pearson']:.4f}, RMSE: {test_metrics['rmse']:.4f}"
                )

                # Store for comparison
                results_list.append(
                    {
                        "Model": model_name,
                        "Train_Pearson": train_metrics["pearson"],
                        "Train_Spearman": train_metrics["spearman"],
                        "Train_RMSE": train_metrics["rmse"],
                        "Train_MAE": train_metrics["mae"],
                        "Val_Pearson": val_metrics["pearson"],
                        "Val_Spearman": val_metrics["spearman"],
                        "Val_RMSE": val_metrics["rmse"],
                        "Val_MAE": val_metrics["mae"],
                        "Test_Pearson": test_metrics["pearson"],
                        "Test_Spearman": test_metrics["spearman"],
                        "Test_RMSE": test_metrics["rmse"],
                        "Test_MAE": test_metrics["mae"],
                        "Test_R2": test_metrics["r2"],
                    }
                )

            except Exception as e:
                print(f"  ✗ Error training {model_name}: {str(e)}")
                continue

        # Create comparison DataFrame
        self.comparison_df = pd.DataFrame(results_list)

        return data

    def plot_comparison(self):
        """Create comparison visualizations"""

        print("\n" + "=" * 80)
        print("CREATING COMPARISON VISUALIZATIONS")
        print("=" * 80)

        figures_dir = Path("figures")
        figures_dir.mkdir(exist_ok=True)

        # Figure 1: Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

        # Pearson correlation
        ax = axes[0, 0]
        df_plot = self.comparison_df.sort_values("Test_Pearson", ascending=False)
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_plot)))
        bars = ax.barh(
            range(len(df_plot)),
            df_plot["Test_Pearson"],
            color=colors,
            edgecolor="black",
        )
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot["Model"])
        ax.set_xlabel("Pearson Correlation", fontweight="bold")
        ax.set_title("(A) Test Set Pearson Correlation", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        for i, val in enumerate(df_plot["Test_Pearson"]):
            ax.text(val + 0.01, i, f"{val:.4f}", va="center", fontweight="bold")

        # Spearman correlation
        ax = axes[0, 1]
        df_plot = self.comparison_df.sort_values("Test_Spearman", ascending=False)
        bars = ax.barh(
            range(len(df_plot)),
            df_plot["Test_Spearman"],
            color=colors,
            edgecolor="black",
        )
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot["Model"])
        ax.set_xlabel("Spearman Correlation", fontweight="bold")
        ax.set_title("(B) Test Set Spearman Correlation", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        for i, val in enumerate(df_plot["Test_Spearman"]):
            ax.text(val + 0.01, i, f"{val:.4f}", va="center", fontweight="bold")

        # RMSE
        ax = axes[1, 0]
        df_plot = self.comparison_df.sort_values("Test_RMSE", ascending=True)
        bars = ax.barh(
            range(len(df_plot)), df_plot["Test_RMSE"], color=colors, edgecolor="black"
        )
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot["Model"])
        ax.set_xlabel("RMSE (lower is better)", fontweight="bold")
        ax.set_title("(C) Test Set RMSE", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        for i, val in enumerate(df_plot["Test_RMSE"]):
            ax.text(val + 0.01, i, f"{val:.4f}", va="center", fontweight="bold")

        # MAE
        ax = axes[1, 1]
        df_plot = self.comparison_df.sort_values("Test_MAE", ascending=True)
        bars = ax.barh(
            range(len(df_plot)), df_plot["Test_MAE"], color=colors, edgecolor="black"
        )
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot["Model"])
        ax.set_xlabel("MAE (lower is better)", fontweight="bold")
        ax.set_title("(D) Test Set MAE", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        for i, val in enumerate(df_plot["Test_MAE"]):
            ax.text(val + 0.01, i, f"{val:.4f}", va="center", fontweight="bold")

        plt.tight_layout()
        plt.savefig(
            figures_dir / "05_model_comparison.png", dpi=300, bbox_inches="tight"
        )
        print(f"✓ Saved: 05_model_comparison.png")
        plt.close()

        # Figure 2: Train vs Val vs Test performance
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(self.comparison_df))
        width = 0.25

        ax.bar(
            x - width,
            self.comparison_df["Train_Pearson"],
            width,
            label="Train",
            alpha=0.8,
            color="#2E86AB",
        )
        ax.bar(
            x,
            self.comparison_df["Val_Pearson"],
            width,
            label="Validation",
            alpha=0.8,
            color="#A23B72",
        )
        ax.bar(
            x + width,
            self.comparison_df["Test_Pearson"],
            width,
            label="Test",
            alpha=0.8,
            color="#F18F01",
        )

        ax.set_xlabel("Model", fontweight="bold")
        ax.set_ylabel("Pearson Correlation", fontweight="bold")
        ax.set_title(
            "Train/Val/Test Performance Comparison", fontweight="bold", fontsize=14
        )
        ax.set_xticks(x)
        ax.set_xticklabels(self.comparison_df["Model"], rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            figures_dir / "06_train_val_test_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"✓ Saved: 06_train_val_test_comparison.png")
        plt.close()

    def plot_best_model_predictions(self, data):
        """Plot predictions for the best model"""

        # Find best model by test Pearson
        best_model_name = self.comparison_df.sort_values(
            "Test_Pearson", ascending=False
        ).iloc[0]["Model"]
        print(f"\nBest model: {best_model_name}")

        best_result = self.results[best_model_name]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"Best Model Predictions: {best_model_name}", fontsize=16, fontweight="bold"
        )

        # Train predictions
        ax = axes[0, 0]
        ax.scatter(data["y_train"], best_result["train_pred"], alpha=0.3, s=10)
        ax.plot(
            [data["y_train"].min(), data["y_train"].max()],
            [data["y_train"].min(), data["y_train"].max()],
            "r--",
            lw=2,
        )
        ax.set_xlabel("Actual Score", fontweight="bold")
        ax.set_ylabel("Predicted Score", fontweight="bold")
        ax.set_title(
            f"Train (Pearson: {best_result['train_metrics']['pearson']:.4f})",
            fontweight="bold",
        )
        ax.grid(alpha=0.3)

        # Val predictions
        ax = axes[0, 1]
        ax.scatter(data["y_val"], best_result["val_pred"], alpha=0.3, s=10)
        ax.plot(
            [data["y_val"].min(), data["y_val"].max()],
            [data["y_val"].min(), data["y_val"].max()],
            "r--",
            lw=2,
        )
        ax.set_xlabel("Actual Score", fontweight="bold")
        ax.set_ylabel("Predicted Score", fontweight="bold")
        ax.set_title(
            f"Validation (Pearson: {best_result['val_metrics']['pearson']:.4f})",
            fontweight="bold",
        )
        ax.grid(alpha=0.3)

        # Test predictions
        ax = axes[0, 2]
        ax.scatter(data["y_test"], best_result["test_pred"], alpha=0.3, s=10)
        ax.plot(
            [data["y_test"].min(), data["y_test"].max()],
            [data["y_test"].min(), data["y_test"].max()],
            "r--",
            lw=2,
        )
        ax.set_xlabel("Actual Score", fontweight="bold")
        ax.set_ylabel("Predicted Score", fontweight="bold")
        ax.set_title(
            f"Test (Pearson: {best_result['test_metrics']['pearson']:.4f})",
            fontweight="bold",
        )
        ax.grid(alpha=0.3)

        # Residual plots
        for i, (split_name, y_true, y_pred) in enumerate(
            [
                ("Train", data["y_train"], best_result["train_pred"]),
                ("Val", data["y_val"], best_result["val_pred"]),
                ("Test", data["y_test"], best_result["test_pred"]),
            ]
        ):
            ax = axes[1, i]
            residuals = y_true - y_pred
            ax.scatter(y_pred, residuals, alpha=0.3, s=10)
            ax.axhline(0, color="r", linestyle="--", lw=2)
            ax.set_xlabel("Predicted Score", fontweight="bold")
            ax.set_ylabel("Residuals", fontweight="bold")
            ax.set_title(f"{split_name} Residuals", fontweight="bold")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            Path("figures") / "07_best_model_predictions.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"✓ Saved: 07_best_model_predictions.png")
        plt.close()

    def save_results(self):
        """Save all results"""

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Save comparison table
        self.comparison_df.to_csv(results_dir / "model_comparison.csv", index=False)
        print(f"\n✓ Saved results to results/model_comparison.csv")

        # Print summary table
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        print(self.comparison_df.to_string(index=False))


def main():
    """Main execution"""

    print("=" * 80)
    print("REGRESSION MODEL COMPARISON")
    print("=" * 80)

    # Initialize
    comparison = RegressionModelComparison()
    comparison.initialize_models()

    # Load data
    data = comparison.load_data()

    # Train and evaluate
    data = comparison.train_and_evaluate(data)

    # Create visualizations
    comparison.plot_comparison()
    comparison.plot_best_model_predictions(data)

    # Save results
    comparison.save_results()

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
