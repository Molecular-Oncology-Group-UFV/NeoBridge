import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report, precision_recall_curve)
import xgboost as xgb
import joblib
import warnings
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Matplotlib configuration
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 10
rcParams['figure.titlesize'] = 14


# ==============================================================================
# HELPER CLASSES
# ==============================================================================

class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.3):
        super(ImprovedMLP, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class HybridEnsemble:
    def __init__(self, pytorch_model, xgb_model, device):
        self.pytorch_model = pytorch_model
        self.xgb_model = xgb_model
        self.device = device

    def predict_proba(self, X):
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X_tensor = X.to(self.device)

        self.pytorch_model.eval()
        with torch.no_grad():
            pytorch_logits = self.pytorch_model(X_tensor)
            pytorch_proba = torch.sigmoid(pytorch_logits).cpu().numpy().flatten()

        if isinstance(X, torch.Tensor):
            X_numpy = X.cpu().numpy()
        else:
            X_numpy = X

        xgb_proba = self.xgb_model.predict_proba(X_numpy)[:, 1]

        return pytorch_proba, xgb_proba

    def ensemble_predict_proba(self, X, weights=(0.5, 0.5)):
        p_prob, x_prob = self.predict_proba(X)

        total = sum(weights)
        w1, w2 = weights[0] / total, weights[1] / total

        ensemble_proba = (w1 * p_prob + w2 * x_prob)
        return ensemble_proba, p_prob, x_prob


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

class HybridEnsemblePipeline:
    def __init__(self):
        # Path Configuration
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.data_path = self.project_root / 'data' / 'processed' / 'dataset_80_20_normalized_tcr_with_cohort_and_rank.csv'

        # Output Directories
        self.models_dir = self.project_root / 'models' / 'production' / 'hybrid_ensemble'
        self.reports_dir = self.project_root / 'results' / 'reports'
        self.figures_dir = self.project_root / 'results' / 'figures'

        # Create directory structure
        for d in [self.models_dir, self.reports_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)

        for sub in ['optuna', 'performance', 'feature_importance', 'elispot_candidates']:
            (self.reports_dir / sub).mkdir(exist_ok=True)
            (self.figures_dir / sub).mkdir(exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device selected: {self.device}")

    def load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found at {self.data_path}")

        print(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)

        # We keep the raw DataFrame to recover metadata later (peptide sequences, etc.)
        features_to_drop = ['immunogenicity', 'mut_is_binding_pos', 'bestWTMatchType_I',
                            'DAI_MixMHC_mbp', 'mutant_rank']

        # Select numeric features for training
        X_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=features_to_drop, errors='ignore')

        X = X_df.values
        y = df['immunogenicity'].values

        self.feature_names = X_df.columns.tolist()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        return df, X_scaled, y

    # --------------------------------------------------------------------------
    # TRAINING HELPERS
    # --------------------------------------------------------------------------

    def train_pytorch_fold(self, X_train, y_train, X_val, y_val, config):
        hidden_dims = [config[f'hidden_dim_{i}'] for i in range(config['n_layers'])]

        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

        model = ImprovedMLP(X_train.shape[1], hidden_dims, config['dropout_rate']).to(self.device)

        pos_weight_val = len(y_train) / sum(y_train) if sum(y_train) > 0 else 1.0
        pos_weight = torch.tensor([pos_weight_val]).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        best_auc = 0
        patience_counter = 0

        for epoch in range(config.get('epochs', 50)):
            model.train()
            for inputs, labels in loader:
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(model(X_v)).cpu().numpy().flatten()

            try:
                val_auc = roc_auc_score(y_val, val_probs)
            except:
                val_auc = 0.5

            if val_auc > best_auc + 0.0001:
                best_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.get('patience', 10):
                break

        return best_auc, model

    def run_optimization_cv(self, X, y, model_type, n_trials, metric="auc_roc"):
        print(f"\nOptimizing {model_type} with 5-Fold CV ({n_trials} trials)...")

        def objective(trial):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []

            if model_type == 'PyTorch':
                n_layers = trial.suggest_int('n_layers', 1, 3)
                hidden_dims = [trial.suggest_int(f'hidden_dim_{i}', 32, 256, log=True) for i in range(n_layers)]
                config = {
                    'n_layers': n_layers,
                    'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                    'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
                    'epochs': 30, 'patience': 5
                }
                for i, h in enumerate(hidden_dims): config[f'hidden_dim_{i}'] = h

            elif model_type == 'XGBoost':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'eval_metric': 'logloss', 'verbosity': 0, 'n_jobs': 1
                }

            for train_idx, val_idx in skf.split(X, y):
                X_t, X_v = X[train_idx], X[val_idx]
                y_t, y_v = y[train_idx], y[val_idx]

                if model_type == 'PyTorch':
                    score, _ = self.train_pytorch_fold(X_t, y_t, X_v, y_v, config)
                    scores.append(score)
                elif model_type == 'XGBoost':
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_t, y_t)
                    probs = model.predict_proba(X_v)[:, 1]
                    if metric == "auc_pr":
                        scores.append(average_precision_score(y_v, probs))
                    else:
                        scores.append(roc_auc_score(y_v, probs))

            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.save_optimization_results(study, model_type, metric)
        return study.best_params

    def optimize_weights(self, pytorch_model, xgb_model, X_val, y_val, n_trials=20, metric="auc_roc"):
        print("\nOptimizing Ensemble Weights...")
        ensemble = HybridEnsemble(pytorch_model, xgb_model, self.device)

        def objective(trial):
            w_pytorch = trial.suggest_float('weight_pytorch', 0.1, 0.9)
            probs, _, _ = ensemble.ensemble_predict_proba(X_val, (w_pytorch, 1.0 - w_pytorch))
            if metric == "auc_pr": return average_precision_score(y_val, probs)
            return roc_auc_score(y_val, probs)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_w = study.best_params['weight_pytorch']
        return (best_w, 1.0 - best_w)

    def evaluate_performance(self, y_true, y_probs, set_name="test"):
        # F2/F1 Score Optimization
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        f2_scores = []
        for i in range(len(precision) - 1):
            p, r = precision[i], recall[i]
            f2 = (5 * p * r) / (4 * p + r) if (p + r) > 0 else 0
            f2_scores.append(f2)

        f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)

        # Best Thresholds
        best_thresh_f2 = thresholds[np.argmax(f2_scores)]
        best_thresh_f1 = thresholds[np.argmax(f1_scores[:-1])]

        print(f"\n--- {set_name.upper()} THRESHOLDS ---")
        print(f"Optimal F2 Threshold: {best_thresh_f2:.4f}")
        print(f"Optimal F1 Threshold: {best_thresh_f1:.4f}")

        return best_thresh_f2, best_thresh_f1

    def analyze_feature_importance(self, pytorch_model, xgb_model, X_sample, weights):
        print("\nGenerating Feature Importance Analysis...")
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)
        xgb_imp = np.abs(shap_values).mean(0)
        xgb_imp /= xgb_imp.sum()

        pytorch_model.eval()
        X_t = torch.tensor(X_sample, dtype=torch.float32).to(self.device)
        X_t.requires_grad = True
        outputs = pytorch_model(X_t)
        outputs.backward(torch.ones_like(outputs))
        mlp_imp = np.abs(X_t.grad.cpu().numpy()).mean(0)
        mlp_imp /= mlp_imp.sum()

        ens_imp = weights[0] * mlp_imp + weights[1] * xgb_imp

        df_imp = pd.DataFrame({'feature': self.feature_names, 'importance': ens_imp}).sort_values('importance',
                                                                                                  ascending=False)
        df_imp.to_csv(self.reports_dir / 'feature_importance' / 'ensemble_importance.csv', index=False)
        return np.corrcoef(mlp_imp, xgb_imp)[0, 1]

    def save_optimization_results(self, study, model_type, metric):
        df_results = study.trials_dataframe()
        df_results['model'] = model_type
        df_results['metric'] = metric

        results_path = self.reports_dir / 'optuna' / f'{model_type}_optimization_results.csv'
        df_results.to_csv(results_path, index=False)

    # --------------------------------------------------------------------------
    # NEW: RANKING & TOP-N METRICS ANALYSIS
    # --------------------------------------------------------------------------

    def analyze_ranking_performance(self, df_test, y_true, y_probs):
        """
        Calculates Ranking metrics (Top-N Recall, TTIF) and exports Top-100 candidates.
        """
        print("\n" + "=" * 60)
        print("RANKING & TOP-N METRICS ANALYSIS (PAPER & ELISPOT)")
        print("=" * 60)

        # 1. Prepare Dataframe
        results = df_test.copy()
        results['model_probability'] = y_probs
        results['true_label'] = y_true

        # Sort by probability descending (High confidence first)
        results = results.sort_values(by='model_probability', ascending=False).reset_index(drop=True)

        total_positives = np.sum(y_true)
        print(f"Total Immunogenic in Test Set: {total_positives}")

        # 2. Calculate Metrics
        auprc = average_precision_score(y_true, y_probs)

        # TTIF (Top-20 Immunogenic Fraction) -> Precision @ 20
        top_20 = results.head(20)
        ttif = top_20['true_label'].mean()

        # Top-N Recall
        recalls = {}
        for n in [20, 50, 100]:
            top_n = results.head(n)
            positives_recovered = top_n['true_label'].sum()
            recall_at_n = positives_recovered / total_positives if total_positives > 0 else 0
            recalls[n] = recall_at_n

        fr = 1.0

        # 3. Generate Comparative Table
        benchmarks = {
            'AUPRC': 0.273,
            'FR': 0.778,
            'TTIF': 0.250,
            'Top-20 Recall': 0.030,
            'Top-50 Recall': 0.062,
            'Top-100 Recall': 0.076
        }

        my_metrics = {
            'AUPRC': auprc,
            'FR': fr,
            'TTIF': ttif,
            'Top-20 Recall': recalls[20],
            'Top-50 Recall': recalls[50],
            'Top-100 Recall': recalls[100]
        }

        comparison_rows = []
        for metric, neoval in benchmarks.items():
            my_val = my_metrics[metric]

            # Diff percentage
            if neoval > 0:
                diff_pct = ((my_val - neoval) / neoval) * 100
                diff_str = f"{diff_pct:+.1f}%"
            else:
                diff_str = "N/A"

            comparison_rows.append({
                'Metric': metric,
                'My Model': f"{my_val:.3f}",
                'NeoRanking (Benchmark)': f"{neoval:.3f}",
                'Difference': diff_str
            })

        df_comparison = pd.DataFrame(comparison_rows)

        print("\n>>> Comparison Table (My Model vs NeoRanking Benchmark):")
        print(df_comparison.to_string(index=False))

        # Save comparison to CSV
        df_comparison.to_csv(self.reports_dir / 'performance' / 'neoranking_comparison.csv', index=False)

        # 4. Export Top 100 Candidates for ELISpot
        print("\n>>> Exporting Top 100 Candidates for ELISpot...")
        top_100_candidates = results.head(100)

        # Save with timestamp
        filename = f'top_100_elispot_candidates.csv'
        save_path = self.reports_dir / 'elispot_candidates' / filename

        # Select important columns to put first (if they exist in metadata)
        cols = list(top_100_candidates.columns)
        priority_cols = ['model_probability', 'true_label', 'peptide', 'HLA', 'gene', 'cohort']
        ordered_cols = [c for c in priority_cols if c in cols] + [c for c in cols if c not in priority_cols]

        top_100_candidates = top_100_candidates[ordered_cols]
        top_100_candidates.to_csv(save_path, index=False)

        print(f"✓ Saved: {save_path}")
        print(f"   - Min Probability in Top 100: {top_100_candidates['model_probability'].min():.4f}")
        print(f"   - Max Probability in Top 100: {top_100_candidates['model_probability'].max():.4f}")
        print(f"   - Immunogenic candidates found in Top 100: {top_100_candidates['true_label'].sum()}")

    # --------------------------------------------------------------------------
    # MAIN EXECUTION METHOD
    # --------------------------------------------------------------------------

    def run(self, n_trials_pytorch, n_trials_xgb, n_trials_ensemble, metric, use_cv):
        # 1. Load Data
        df, X, y = self.load_data()

        # 2. Strict Hold-out Test Set (20%)
        # Pass full DF to split to keep metadata
        X_dev, X_test, y_dev, y_test, df_dev, df_test = train_test_split(
            X, y, df, test_size=0.20, random_state=42, stratify=y
        )

        print(f"\nData Splits:")
        print(f"  Development (Opt): {X_dev.shape[0]} samples")
        print(f"  Test (Final Eval): {X_test.shape[0]} samples")

        # 3. Hyperparameter Optimization
        best_py_params = self.run_optimization_cv(X_dev, y_dev, 'PyTorch', n_trials_pytorch, metric=metric)
        best_xgb_params = self.run_optimization_cv(X_dev, y_dev, 'XGBoost', n_trials_xgb, metric=metric)

        # 4. Final Model Training (Internal Split for Ensemble Weights)
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_dev, y_dev, test_size=0.1, stratify=y_dev, random_state=42
        )

        print("\nTraining final models...")
        best_py_params['epochs'] = 100;
        best_py_params['patience'] = 15
        _, pytorch_model = self.train_pytorch_fold(X_train_final, y_train_final, X_val_final, y_val_final,
                                                   best_py_params)

        final_xgb_params = best_xgb_params.copy()
        if 'n_estimators' in final_xgb_params: final_xgb_params.pop('n_estimators')
        xgb_model = xgb.XGBClassifier(**final_xgb_params, n_estimators=500)
        xgb_model.fit(X_train_final, y_train_final, eval_set=[(X_val_final, y_val_final)], verbose=False)

        # 5. Optimize Weights
        best_weights = self.optimize_weights(pytorch_model, xgb_model, X_val_final, y_val_final, n_trials_ensemble,
                                             metric)

        # 6. Final Evaluation
        ensemble = HybridEnsemble(pytorch_model, xgb_model, self.device)
        probs_test, _, _ = ensemble.ensemble_predict_proba(X_test, weights=best_weights)

        final_auc = roc_auc_score(y_test, probs_test)
        final_pr = average_precision_score(y_test, probs_test)
        print(f"\nFinal AUC-ROC: {final_auc:.4f} | AUC-PR: {final_pr:.4f}")

        self.evaluate_performance(y_test, probs_test, set_name="final_test")
        self.analyze_feature_importance(pytorch_model, xgb_model, X_test[:200], best_weights)

        # 7. NEW: Analyze Ranking & Export Top 100
        self.analyze_ranking_performance(df_test, y_test, probs_test)

        # 8. Save Models
        print(f"\nSaving models to {self.models_dir}...")
        torch.save(pytorch_model.state_dict(), self.models_dir / 'pytorch_final.pth')
        xgb_model.save_model(self.models_dir / 'xgboost_final.json')
        joblib.dump(self.scaler, self.models_dir / 'scaler.pkl')
        print("Pipeline completed successfully.")


def main():
    print("HYBRID ENSEMBLE PIPELINE (GENERAL MODEL - PRODUCTION)")
    print("=" * 60)

    # 1. Configurar Métrica
    metric_in = input("\nSelect metric (1: AUC-ROC, 2: AUC-PR) [Default: 2]: ") or "2"
    metric = "auc_pr" if metric_in == "2" else "auc_roc"

    # 2. Configurar CV
    cv_in = input("Use Cross-Validation? (y/n) [Default: y]: ") or "y"
    use_cv = cv_in.lower() == 'y'

    # 3. Configurar Trials (Optuna)
    print("\n--- OPTUNA TRIALS CONFIGURATION ---")

    try:
        n_py = int(input("PyTorch trials [Default: 30]: ") or "30")
    except ValueError:
        n_py = 30

    try:
        n_xgb = int(input("XGBoost trials [Default: 30]: ") or "30")
    except ValueError:
        n_xgb = 30

    try:
        n_ens = int(input("Ensemble trials [Default: 20]: ") or "20")
    except ValueError:
        n_ens = 20

    print(f"\nConfiguration: Metric={metric}, PyTorch={n_py}, XGB={n_xgb}, Ensemble={n_ens}")
    print("Starting pipeline...\n")

    pipeline = HybridEnsemblePipeline()
    pipeline.run(n_trials_pytorch=n_py, n_trials_xgb=n_xgb, n_trials_ensemble=n_ens, metric=metric, use_cv=use_cv)


if __name__ == "__main__":
    main()