import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, precision_recall_curve, auc)
import xgboost as xgb
import warnings
import ast
from pathlib import Path
import sys

warnings.filterwarnings('ignore')


class NeoRankingEnsemble:

    def __init__(self):
        # Path Configuration
        self.project_root = Path(__file__).resolve().parent.parent.parent

        # Input Data Path
        self.data_path = self.project_root / 'data' / 'raw' / 'NeoPepBase.csv'

        # Optimization Results Path
        self.optuna_path = self.project_root / 'results' / 'reports' / 'Optuna_Results' / 'optuna_results_general.csv'

    def load_data(self):
        """Loads and splits the dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")

        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)

        # Prepare variables
        X = df.select_dtypes(include=['float64', 'int64']).drop(columns=['immunogenicity'], errors='ignore')
        y = df['immunogenicity']

        print(f"Dataset shape: {df.shape}")
        print(f"Features: {X.shape[1]}")
        print(f"Class distribution: {y.value_counts().to_dict()}")

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def load_best_params(self, model_name):
        """Loads best parameters for a specific model from the Optuna results CSV."""
        if not self.optuna_path.exists():
            print(f"Warning: Optuna results not found at {self.optuna_path}. Using default parameters.")
            return {}

        try:
            df_results = pd.read_csv(self.optuna_path)
            # Filter by model name (assuming column name is 'model' or similar based on previous context)
            # If your CSV has different column names, adjust 'model' and 'result' below.
            if 'model' not in df_results.columns:
                # Fallback if column names differ, trying to adapt to standard Optuna output
                return {}

            model_results = df_results[df_results['model'] == model_name].copy()

            if len(model_results) == 0:
                print(f"No results found for {model_name}, using default parameters.")
                return {}

            # Sort by result (descending for AUC/Score) and take the best
            model_results = model_results.sort_values('result', ascending=False)
            best_result = model_results.iloc[0]

            print(f"Loading best parameters for {model_name}: Score = {best_result['result']:.4f}")
            return ast.literal_eval(best_result['best_params'])

        except Exception as e:
            print(f"Error loading parameters for {model_name}: {e}")
            return {}

    def create_balanced_rf(self):
        """Creates BalancedRandomForest with optimized parameters."""
        params = self.load_best_params('BalancedRandomForestClassifier')

        rf_params = {
            'random_state': 42,
            'n_jobs': -1
        }
        rf_params.update(params)

        return BalancedRandomForestClassifier(**rf_params)

    def create_xgboost(self):
        """Creates XGBoost with optimized parameters."""
        params = self.load_best_params('XGBoost')

        xgb_params = {
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        xgb_params.update(params)

        return xgb.XGBClassifier(**xgb_params)

    # ==========================================================
    # Custom Metrics
    # ==========================================================

    @staticmethod
    def fraction_ranked(y_true, y_scores):
        """Fraction Ranked (FR): proportion of positives with a valid score."""
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        positives = y_true == 1
        if positives.sum() == 0:
            return np.nan
        ranked = ~np.isnan(y_scores[positives])
        return ranked.sum() / positives.sum()

    @staticmethod
    def top_n_recall(y_true, y_scores, n=20):
        """Percentage of positives present in the top-N scores."""
        df_metrics = pd.DataFrame({'y_true': y_true, 'y_score': y_scores})
        df_metrics = df_metrics.sort_values('y_score', ascending=False).reset_index(drop=True)
        top_n = df_metrics.head(n)
        if df_metrics['y_true'].sum() == 0:
            return np.nan
        return top_n['y_true'].sum() / df_metrics['y_true'].sum()

    @staticmethod
    def ttif(y_true, y_scores, n=20):
        """Top-N Immunogenic Fraction (TTIF): fraction of top-N that are actually positive."""
        df_metrics = pd.DataFrame({'y_true': y_true, 'y_score': y_scores})
        df_metrics = df_metrics.sort_values('y_score', ascending=False).reset_index(drop=True)
        top_n = df_metrics.head(n)
        return top_n['y_true'].sum() / len(top_n)

    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluates a model using standard and custom metrics."""
        print(f"\n{'-' * 60}")
        print(f"Results: {model_name}")
        print(f"{'-' * 60}")

        y_pred = model.predict(X_test)

        # Handle probability prediction for different model types
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            # Fallback for models without probability (rare in this context)
            y_pred_proba = y_pred

        auc_roc = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC-ROC: {auc_roc:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # NeoRanking Metrics
        fr_value = self.fraction_ranked(y_test, y_pred_proba)
        ttif_20 = self.ttif(y_test, y_pred_proba, n=20)
        recall_20 = self.top_n_recall(y_test, y_pred_proba, n=20)
        recall_50 = self.top_n_recall(y_test, y_pred_proba, n=50)
        recall_100 = self.top_n_recall(y_test, y_pred_proba, n=100)

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auprc = auc(recall, precision)

        print(f"\nCustom Metrics:")
        print(f"  Fraction Ranked (FR): {fr_value:.3f}")
        print(f"  Top-20 Immunogenic Fraction (TTIF): {ttif_20:.3f}")
        print(f"  Top-20 Recall: {recall_20:.3f}")
        print(f"  Top-50 Recall: {recall_50:.3f}")
        print(f"  Top-100 Recall: {recall_100:.3f}")
        print(f"  AUPRC: {auprc:.3f}")

        return auc_roc

    def run(self):
        """Main execution flow."""
        print("Starting Ensemble Balanced RF + XGBoost...")

        X_train, X_test, y_train, y_test = self.load_data()

        # 1. Create Individual Models
        print("\nInitializing models...")
        brf_model = self.create_balanced_rf()
        xgb_model = self.create_xgboost()

        # 2. Create Ensemble
        print("Creating Voting Classifier (Soft Voting)...")
        ensemble_model = VotingClassifier(
            estimators=[
                ('balanced_rf', brf_model),
                ('xgboost', xgb_model)
            ],
            voting='soft'
        )

        # 3. Train Ensemble
        print("Training Ensemble...")
        ensemble_model.fit(X_train, y_train)

        # 4. Evaluate Ensemble
        ensemble_auc = self.evaluate_model(ensemble_model, X_test, y_test, "Ensemble (BRF + XGB)")

        # 5. Compare with Individual Models
        print(f"\n{'=' * 60}")
        print("Comparison: Ensemble vs Individual Models")
        print(f"{'=' * 60}")

        print("Evaluating individual models for comparison...")

        brf_model.fit(X_train, y_train)
        brf_auc = roc_auc_score(y_test, brf_model.predict_proba(X_test)[:, 1])

        xgb_model.fit(X_train, y_train)
        xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

        print(f"\nAUC-ROC Comparison:")
        print(f"  Balanced Random Forest: {brf_auc:.4f}")
        print(f"  XGBoost:                {xgb_auc:.4f}")
        print(f"  Ensemble:               {ensemble_auc:.4f}")

        improvement_brf = ensemble_auc - brf_auc
        improvement_xgb = ensemble_auc - xgb_auc

        print(f"\nEnsemble Improvement:")
        print(f"  vs Balanced RF: {improvement_brf:+.4f}")
        print(f"  vs XGBoost:     {improvement_xgb:+.4f}")

        if improvement_brf > 0 and improvement_xgb > 0:
            print("Conclusion: The ensemble improves upon both individual models.")
        elif improvement_brf > 0 or improvement_xgb > 0:
            print("Conclusion: The ensemble improves upon at least one individual model.")
        else:
            print("Conclusion: The ensemble does not improve individual models.")


if __name__ == "__main__":
    predictor = NeoRankingEnsemble()
    predictor.run()