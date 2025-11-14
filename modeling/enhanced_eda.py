"""
Comprehensive Exploratory Data Analysis for MT Quality Estimation
Creates publication-quality visualizations and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set style for attractive plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.3)


class MTQualityEDA:
    """Comprehensive EDA for MT Quality Estimation"""

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.figures_dir = Path("figures")
        self.figures_dir.mkdir(exist_ok=True)

        # Color palette for language pairs
        self.colors = {
            "en-de": "#2E86AB",  # Blue
            "ro-en": "#A23B72",  # Purple
            "si-en": "#F18F01",  # Orange
        }

    def load_data(self):
        """Load the combined dataset"""
        train_file = self.data_dir / "train_combined.csv"

        if not train_file.exists():
            print(f"Error: {train_file} not found.")
            print("Please run load_real_data.py first!")
            return None

        df = pd.read_csv(train_file)
        print(f"Loaded {len(df)} training samples")
        return df

    def compute_features(self, df):
        """Compute linguistic features for analysis"""

        # Use cleaned text for analysis
        df["src_len"] = df["src_clean"].str.len()
        df["mt_len"] = df["mt_clean"].str.len()
        df["len_ratio"] = df["mt_len"] / (df["src_len"] + 1e-8)

        df["src_tokens"] = df["src_clean"].str.split().str.len()
        df["mt_tokens"] = df["mt_clean"].str.split().str.len()
        df["token_ratio"] = df["mt_tokens"] / (df["src_tokens"] + 1e-8)

        df["len_diff"] = np.abs(df["src_len"] - df["mt_len"])
        df["token_diff"] = np.abs(df["src_tokens"] - df["mt_tokens"])

        return df

    def plot_1_overview(self, df):
        """Figure 1: Dataset Overview"""

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Sample counts by language pair
        ax1 = fig.add_subplot(gs[0, 0])
        counts = df["lang_pair"].value_counts()
        bars = ax1.bar(
            range(len(counts)),
            counts.values,
            color=[self.colors[lp] for lp in counts.index],
        )
        ax1.set_xticks(range(len(counts)))
        ax1.set_xticklabels(counts.index, rotation=0)
        ax1.set_ylabel("Number of Samples", fontweight="bold")
        ax1.set_title("(A) Samples per Language Pair", fontweight="bold", pad=10)
        ax1.grid(axis="y", alpha=0.3)

        for i, (lp, count) in enumerate(counts.items()):
            ax1.text(
                i, count + 50, f"{count:,}", ha="center", va="bottom", fontweight="bold"
            )

        # 2. Resource level distribution
        ax2 = fig.add_subplot(gs[0, 1])
        resource_counts = df["resource_level"].value_counts()
        colors_resource = ["#2E86AB", "#A23B72", "#F18F01"]
        wedges, texts, autotexts = ax2.pie(
            resource_counts.values,
            labels=resource_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors_resource,
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
        ax2.set_title("(B) Resource Level Distribution", fontweight="bold", pad=10)

        # # 3. Score distribution - overall
        # ax3 = fig.add_subplot(gs[0, 2])
        # ax3.hist(df["score"], bins=50, edgecolor="black", alpha=0.7, color="#2E86AB")
        # ax3.axvline(
        #     df["score"].mean(),
        #     color="red",
        #     linestyle="--",
        #     linewidth=2,
        #     label=f'Mean: {df["score"].mean():.3f}',
        # )
        # ax3.axvline(
        #     df["score"].median(),
        #     color="green",
        #     linestyle="--",
        #     linewidth=2,
        #     label=f'Median: {df["score"].median():.3f}',
        # )
        # ax3.set_xlabel("Quality Score (z-mean)", fontweight="bold")
        # ax3.set_ylabel("Frequency", fontweight="bold")
        # ax3.set_title("(C) Overall Score Distribution", fontweight="bold", pad=10)
        # ax3.legend()
        # ax3.grid(axis="y", alpha=0.3)

        # 3. Score distribution - overall
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(df["score"], bins=50, edgecolor="black", alpha=0.7, color="#2E86AB")
        ax3.axvline(
            df["score"].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f'Mean: {df["score"].mean():.3f}',
        )
        ax3.axvline(
            df["score"].median(),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f'Median: {df["score"].median():.3f}',
        )
        ax3.set_xlabel("Quality Score (z-mean)", fontweight="bold")
        ax3.set_ylabel("Frequency", fontweight="bold")
        ax3.set_title("(C) Overall Score Distribution", fontweight="bold", pad=10)
        ax3.legend(fontsize=8)  # Added fontsize to make legend smaller
        ax3.grid(axis="y", alpha=0.3)
        ax3.set_ylim(0, ax3.get_ylim()[1] * 1.1)  # Add 10% space at top

        # 4. Score distribution by language pair
        ax4 = fig.add_subplot(gs[1, :2])
        for lp in df["lang_pair"].unique():
            subset = df[df["lang_pair"] == lp]
            ax4.hist(
                subset["score"],
                bins=30,
                alpha=0.5,
                label=lp,
                color=self.colors[lp],
                edgecolor="black",
            )
        ax4.set_xlabel("Quality Score (z-mean)", fontweight="bold")
        ax4.set_ylabel("Frequency", fontweight="bold")
        ax4.set_title(
            "(D) Score Distribution by Language Pair", fontweight="bold", pad=10
        )
        ax4.legend(title="Language Pair", fontsize=10)
        ax4.grid(axis="y", alpha=0.3)

        # # 5. Box plot of scores
        # ax5 = fig.add_subplot(gs[1, 2])
        # lang_pairs = df["lang_pair"].unique()
        # positions = range(len(lang_pairs))
        # box_data = [df[df["lang_pair"] == lp]["score"].values for lp in lang_pairs]
        # bp = ax5.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)

        # for patch, lp in zip(bp["boxes"], lang_pairs):
        #     patch.set_facecolor(self.colors[lp])
        #     patch.set_alpha(0.7)

        # ax5.set_xticks(positions)
        # ax5.set_xticklabels(lang_pairs, rotation=0)
        # ax5.set_ylabel("Quality Score", fontweight="bold")
        # ax5.set_title("(E) Score Distribution (Box Plot)", fontweight="bold", pad=10)
        # ax5.grid(axis="y", alpha=0.3)

        # 5. Box plot of scores
        ax5 = fig.add_subplot(gs[1, 2])
        lang_pairs = df["lang_pair"].unique()
        positions = range(len(lang_pairs))
        box_data = [df[df["lang_pair"] == lp]["score"].values for lp in lang_pairs]
        bp = ax5.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)

        for patch, lp in zip(bp["boxes"], lang_pairs):
            patch.set_facecolor(self.colors[lp])
            patch.set_alpha(0.7)

        ax5.set_xticks(positions)
        ax5.set_xticklabels(lang_pairs, rotation=0)
        ax5.set_ylabel("Quality Score", fontweight="bold")
        # ax5.set_title(
        #     # "(E) Score Distribution (Box Plot)", fontweight="bold", pad=15
        # )  # Increased pad from 10 to 15
        ax5.grid(axis="y", alpha=0.3)

        # 6. Score statistics table
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis("off")

        stats_data = []
        for lp in df["lang_pair"].unique():
            subset = df[df["lang_pair"] == lp]
            stats_data.append(
                [
                    lp,
                    f"{len(subset):,}",
                    f"{subset['score'].mean():.4f}",
                    f"{subset['score'].std():.4f}",
                    f"{subset['score'].min():.4f}",
                    f"{subset['score'].quantile(0.25):.4f}",
                    f"{subset['score'].median():.4f}",
                    f"{subset['score'].quantile(0.75):.4f}",
                    f"{subset['score'].max():.4f}",
                ]
            )

        headers = [
            "Lang Pair",
            "Count",
            "Mean",
            "Std",
            "Min",
            "Q1",
            "Median",
            "Q3",
            "Max",
        ]
        table = ax6.table(
            cellText=stats_data,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Color header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor("#2E86AB")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # Color rows alternately
        for i in range(1, len(stats_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor("#f0f0f0")

        plt.suptitle(
            "MT Quality Estimation: Dataset Overview",
            fontsize=18,
            fontweight="bold",
            y=0.995,
        )

        plt.savefig(
            self.figures_dir / "01_dataset_overview.png", dpi=300, bbox_inches="tight"
        )
        print(f"✓ Saved: 01_dataset_overview.png")
        plt.close()

    def plot_2_length_analysis(self, df):
        """Figure 2: Length Analysis"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Length Analysis Across Language Pairs",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        # Source length distribution
        ax = axes[0, 0]
        for lp in df["lang_pair"].unique():
            subset = df[df["lang_pair"] == lp]
            ax.hist(
                subset["src_len"], bins=30, alpha=0.5, label=lp, color=self.colors[lp]
            )
        ax.set_xlabel("Source Length (characters)", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title("(A) Source Text Length", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # MT length distribution
        ax = axes[0, 1]
        for lp in df["lang_pair"].unique():
            subset = df[df["lang_pair"] == lp]
            ax.hist(
                subset["mt_len"], bins=30, alpha=0.5, label=lp, color=self.colors[lp]
            )
        ax.set_xlabel("Translation Length (characters)", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title("(B) Translation Length", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Length ratio distribution
        ax = axes[0, 2]
        for lp in df["lang_pair"].unique():
            subset = df[df["lang_pair"] == lp]
            # Filter extreme outliers for better visualization
            ratios = subset["len_ratio"]
            ratios = ratios[(ratios > 0.2) & (ratios < 5)]
            ax.hist(ratios, bins=30, alpha=0.5, label=lp, color=self.colors[lp])
        ax.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Perfect (1.0)")
        ax.set_xlabel("Length Ratio (MT/Source)", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title("(C) Length Ratio Distribution", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Source vs MT length scatter
        ax = axes[1, 0]
        for lp in df["lang_pair"].unique():
            subset = df[df["lang_pair"] == lp].sample(
                min(1000, len(df[df["lang_pair"] == lp]))
            )
            ax.scatter(
                subset["src_len"],
                subset["mt_len"],
                alpha=0.4,
                s=10,
                label=lp,
                color=self.colors[lp],
            )
        max_len = max(df["src_len"].max(), df["mt_len"].max())
        ax.plot(
            [0, max_len], [0, max_len], "r--", linewidth=2, label="Perfect alignment"
        )
        ax.set_xlabel("Source Length", fontweight="bold")
        ax.set_ylabel("Translation Length", fontweight="bold")
        ax.set_title("(D) Source vs Translation Length", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        # Token count comparison
        ax = axes[1, 1]
        lang_pairs = df["lang_pair"].unique()
        x = np.arange(len(lang_pairs))
        width = 0.35

        src_means = [
            df[df["lang_pair"] == lp]["src_tokens"].mean() for lp in lang_pairs
        ]
        mt_means = [df[df["lang_pair"] == lp]["mt_tokens"].mean() for lp in lang_pairs]

        ax.bar(
            x - width / 2, src_means, width, label="Source", alpha=0.8, color="#2E86AB"
        )
        ax.bar(
            x + width / 2,
            mt_means,
            width,
            label="Translation",
            alpha=0.8,
            color="#F18F01",
        )

        ax.set_xlabel("Language Pair", fontweight="bold")
        ax.set_ylabel("Average Token Count", fontweight="bold")
        ax.set_title("(E) Average Tokens per Language Pair", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(lang_pairs)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Length ratio by language pair
        ax = axes[1, 2]
        box_data = []
        for lp in df["lang_pair"].unique():
            ratios = df[df["lang_pair"] == lp]["len_ratio"]
            # Filter extreme outliers
            ratios = ratios[(ratios > 0.2) & (ratios < 5)]
            box_data.append(ratios.values)

        bp = ax.boxplot(box_data, labels=lang_pairs, patch_artist=True)
        for patch, lp in zip(bp["boxes"], lang_pairs):
            patch.set_facecolor(self.colors[lp])
            patch.set_alpha(0.7)

        ax.axhline(1.0, color="red", linestyle="--", linewidth=2, alpha=0.5)
        ax.set_xlabel("Language Pair", fontweight="bold")
        ax.set_ylabel("Length Ratio", fontweight="bold")
        ax.set_title("(F) Length Ratio by Language Pair", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "02_length_analysis.png", dpi=300, bbox_inches="tight"
        )
        print(f"✓ Saved: 02_length_analysis.png")
        plt.close()

    def plot_3_score_analysis(self, df):
        """Figure 3: Quality Score Analysis"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Quality Score Analysis", fontsize=16, fontweight="bold", y=0.995)

        # Score vs source length
        # ax = axes[0, 0]
        # for lp in df['lang_pair'].unique():
        #     subset = df[df['lang_pair'] == lp].sample(min(1000, len(df[df=='lang_pair'] == lp])))
        #     ax.scatter(subset['src_tokens'], subset['score'], alpha=0.4, s=15,
        #               label=lp, color=self.colors[lp])
        ax = axes[0, 0]
        for lp in df["lang_pair"].unique():
            subset = df[df["lang_pair"] == lp].sample(
                min(1000, len(df[df["lang_pair"] == lp]))
            )
            ax.scatter(
                subset["src_tokens"],
                subset["score"],
                alpha=0.4,
                s=15,
                label=lp,
                color=self.colors[lp],
            )

        ax.set_xlabel("Source Token Count", fontweight="bold")
        ax.set_ylabel("Quality Score", fontweight="bold")
        ax.set_title("(A) Score vs Source Length", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        # Score vs length ratio
        ax = axes[0, 1]
        for lp in df["lang_pair"].unique():
            subset = df[df["lang_pair"] == lp].sample(
                min(1000, len(df[df["lang_pair"] == lp]))
            )
            # Filter ratios
            subset = subset[(subset["len_ratio"] > 0.2) & (subset["len_ratio"] < 5)]
            ax.scatter(
                subset["len_ratio"],
                subset["score"],
                alpha=0.4,
                s=15,
                label=lp,
                color=self.colors[lp],
            )
        ax.axvline(1.0, color="red", linestyle="--", alpha=0.3, linewidth=2)
        ax.set_xlabel("Length Ratio (MT/Source)", fontweight="bold")
        ax.set_ylabel("Quality Score", fontweight="bold")
        ax.set_title("(B) Score vs Length Ratio", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        # Score vs length difference
        ax = axes[0, 2]
        for lp in df["lang_pair"].unique():
            subset = df[df["lang_pair"] == lp].sample(
                min(1000, len(df[df["lang_pair"] == lp]))
            )
            ax.scatter(
                subset["len_diff"],
                subset["score"],
                alpha=0.4,
                s=15,
                label=lp,
                color=self.colors[lp],
            )
        ax.set_xlabel("Absolute Length Difference", fontweight="bold")
        ax.set_ylabel("Quality Score", fontweight="bold")
        ax.set_title("(C) Score vs Length Difference", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        # Score distribution by bins
        ax = axes[1, 0]
        score_bins = pd.cut(df["score"], bins=5)
        bin_counts = score_bins.value_counts().sort_index()
        bars = ax.bar(
            range(len(bin_counts)),
            bin_counts.values,
            color="#2E86AB",
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xticks(range(len(bin_counts)))
        ax.set_xticklabels(
            [
                f"{interval.left:.2f}-{interval.right:.2f}"
                for interval in bin_counts.index
            ],
            rotation=45,
            ha="right",
        )
        ax.set_xlabel("Score Range", fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title("(D) Score Distribution by Bins", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for i, count in enumerate(bin_counts.values):
            ax.text(
                i, count + 50, f"{count}", ha="center", va="bottom", fontweight="bold"
            )

        # Violin plot by language pair
        ax = axes[1, 1]
        parts = ax.violinplot(
            [
                df[df["lang_pair"] == lp]["score"].values
                for lp in df["lang_pair"].unique()
            ],
            positions=range(len(df["lang_pair"].unique())),
            showmeans=True,
            showmedians=True,
        )

        for i, pc in enumerate(parts["bodies"]):
            lp = list(df["lang_pair"].unique())[i]
            pc.set_facecolor(self.colors[lp])
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(df["lang_pair"].unique())))
        ax.set_xticklabels(df["lang_pair"].unique())
        ax.set_xlabel("Language Pair", fontweight="bold")
        ax.set_ylabel("Quality Score", fontweight="bold")
        ax.set_title("(E) Score Distribution (Violin Plot)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Resource level comparison
        ax = axes[1, 2]
        resource_levels = ["high", "medium", "low"]
        means = [
            df[df["resource_level"] == rl]["score"].mean() for rl in resource_levels
        ]
        stds = [df[df["resource_level"] == rl]["score"].std() for rl in resource_levels]

        bars = ax.bar(
            resource_levels,
            means,
            yerr=stds,
            capsize=10,
            alpha=0.7,
            color=["#2E86AB", "#A23B72", "#F18F01"],
            edgecolor="black",
        )
        ax.set_xlabel("Resource Level", fontweight="bold")
        ax.set_ylabel("Mean Quality Score", fontweight="bold")
        ax.set_title("(F) Score by Resource Level", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(
                i,
                mean + std + 0.02,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "03_score_analysis.png", dpi=300, bbox_inches="tight"
        )
        print(f"✓ Saved: 03_score_analysis.png")
        plt.close()

    def plot_4_correlation_analysis(self, df):
        """Figure 4: Correlation Analysis"""

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            "Feature Correlation Analysis", fontsize=16, fontweight="bold", y=1.0
        )

        # Correlation matrix
        feature_cols = [
            "score",
            "src_len",
            "mt_len",
            "len_ratio",
            "src_tokens",
            "mt_tokens",
            "token_ratio",
            "len_diff",
        ]
        corr_matrix = df[feature_cols].corr()

        ax = axes[0]
        im = ax.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(feature_cols)))
        ax.set_yticks(range(len(feature_cols)))
        ax.set_xticklabels(feature_cols, rotation=45, ha="right")
        ax.set_yticklabels(feature_cols)

        # Add correlation values
        for i in range(len(feature_cols)):
            for j in range(len(feature_cols)):
                text = ax.text(
                    j,
                    i,
                    f"{corr_matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        plt.colorbar(im, ax=ax)
        ax.set_title("(A) Feature Correlation Matrix", fontweight="bold", pad=10)

        # Correlation with score (bar plot)
        ax = axes[1]
        score_corr = (
            corr_matrix["score"].drop("score").sort_values(key=abs, ascending=False)
        )
        colors_bar = ["#2E86AB" if x > 0 else "#F18F01" for x in score_corr.values]
        bars = ax.barh(
            range(len(score_corr)),
            score_corr.values,
            color=colors_bar,
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_yticks(range(len(score_corr)))
        ax.set_yticklabels(score_corr.index)
        ax.set_xlabel("Correlation with Quality Score", fontweight="bold")
        ax.set_title(
            "(B) Feature Correlation with Quality Score", fontweight="bold", pad=10
        )
        ax.axvline(0, color="black", linestyle="-", linewidth=1)
        ax.grid(axis="x", alpha=0.3)

        for i, (feat, val) in enumerate(score_corr.items()):
            ax.text(
                val + 0.01 if val > 0 else val - 0.01,
                i,
                f"{val:.3f}",
                va="center",
                ha="left" if val > 0 else "right",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "04_correlation_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"✓ Saved: 04_correlation_analysis.png")
        plt.close()

    def generate_summary_stats(self, df):
        """Generate summary statistics table"""

        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        print("\n1. Overall Statistics:")
        print(f"   Total samples: {len(df):,}")
        print(f"   Score range: [{df['score'].min():.4f}, {df['score'].max():.4f}]")
        print(f"   Score mean: {df['score'].mean():.4f} ± {df['score'].std():.4f}")

        print("\n2. By Language Pair:")
        for lp in df["lang_pair"].unique():
            subset = df[df["lang_pair"] == lp]
            print(f"\n   {lp}:")
            print(f"      Samples: {len(subset):,}")
            print(
                f"      Score: {subset['score'].mean():.4f} ± {subset['score'].std():.4f}"
            )
            print(f"      Avg source length: {subset['src_tokens'].mean():.1f} tokens")
            print(f"      Avg MT length: {subset['mt_tokens'].mean():.1f} tokens")
            print(f"      Avg length ratio: {subset['len_ratio'].median():.3f}")

        print("\n3. By Resource Level:")
        for rl in ["high", "medium", "low"]:
            subset = df[df["resource_level"] == rl]
            print(f"\n   {rl.capitalize()}-resource:")
            print(f"      Samples: {len(subset):,}")
            print(
                f"      Score: {subset['score'].mean():.4f} ± {subset['score'].std():.4f}"
            )

        print("\n" + "=" * 80)

    def run_complete_eda(self):
        """Run the complete EDA pipeline"""

        print("\n" + "=" * 80)
        print("COMPREHENSIVE EDA FOR MT QUALITY ESTIMATION")
        print("=" * 80)

        # Load data
        df = self.load_data()
        if df is None:
            return

        # Compute features
        print("\nComputing linguistic features...")
        df = self.compute_features(df)

        # Generate visualizations
        print("\nGenerating visualizations...")
        print("  Creating Figure 1: Dataset Overview...")
        self.plot_1_overview(df)

        print("  Creating Figure 2: Length Analysis...")
        self.plot_2_length_analysis(df)

        print("  Creating Figure 3: Score Analysis...")
        self.plot_3_score_analysis(df)

        print("  Creating Figure 4: Correlation Analysis...")
        self.plot_4_correlation_analysis(df)

        # Generate summary statistics
        self.generate_summary_stats(df)

        # Save enhanced data with features
        output_file = self.data_dir / "train_with_features.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved enhanced data to {output_file}")

        print("\n" + "=" * 80)
        print("EDA COMPLETE!")
        print(f"All visualizations saved to: {self.figures_dir}")
        print("=" * 80)


def main():
    """Main execution"""
    eda = MTQualityEDA(data_dir="data")
    eda.run_complete_eda()


if __name__ == "__main__":
    main()
