import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score
import xgboost as xgb
import joblib
import warnings
import os
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

# Matplotlib configuration for publication style
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 12


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


class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_auc):
        if self.best_score is None:
            self.best_score = val_auc
        elif val_auc < self.best_score + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_auc
            self.epochs_no_improve = 0
        return self.early_stop


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
            pytorch_proba = torch.sigmoid(pytorch_logits).cpu().numpy()

        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]

        return pytorch_proba.flatten(), xgb_proba

    def ensemble_predict_proba(self, X, weights=(0.5, 0.5)):
        pytorch_proba, xgb_proba = self.predict_proba(X)

        # Normalize weights
        total_weight = weights[0] + weights[1]
        w1 = weights[0] / total_weight
        w2 = weights[1] / total_weight

        ensemble_proba = (w1 * pytorch_proba + w2 * xgb_proba)
        return ensemble_proba, pytorch_proba, xgb_proba


# ==============================================================================
# MAIN PIPELINE CLASS
# ==============================================================================

class HybridEnsemblePipeline:
    def __init__(self):
        # Path Configuration
        self.project_root = Path(__file__).resolve().parent.parent.parent

        # Inputs
        self.data_path = self.project_root / 'data' / 'processed' / 'dataset_80_20_normalized_tcr.csv'

        # Outputs (Models)
        self.models_dir = self.project_root / 'models' / 'production' / 'hybrid_ensemble'

        # Outputs (Reports - CSVs)
        self.reports_dir = self.project_root / 'results' / 'reports'
        self.optuna_dir = self.reports_dir / 'Optuna_Results'
        self.feat_imp_report_dir = self.reports_dir / 'feature_importance'

        # Outputs (Figures - PNGs)
        self.figures_dir = self.project_root / 'results' / 'figures' / 'feature_importance'

        # Create directories structure
        for d in [self.models_dir, self.optuna_dir, self.feat_imp_report_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for analysis
        for sub in ['xgboost_analysis', 'mlp_analysis', 'ensemble_analysis']:
            (self.figures_dir / sub).mkdir(exist_ok=True)
            (self.feat_imp_report_dir / sub).mkdir(exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found at {self.data_path}")

        print(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)

        # Drop metadata columns
        X = df.select_dtypes(include=['float64', 'int64']).drop(
            columns=['immunogenicity', 'mut_is_binding_pos', 'bestWTMatchType_I', 'DAI_MixMHC_mbp', 'mutant_rank'],
            errors='ignore'
        )
        y = df['immunogenicity']

        self.feature_names = X.columns.tolist()

        # Scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y.values

    # --------------------------------------------------------------------------
    # Training Helpers
    # --------------------------------------------------------------------------

    def train_pytorch_model(self, X_train, y_train, X_val, y_val, config, save_temp=False):
        hidden_dims = [config[f'hidden_dim_{i}'] for i in range(config['n_layers'])]

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

        model = ImprovedMLP(X_train.shape[1], hidden_dims, config['dropout_rate']).to(self.device)

        pos_weight_value = len(y_train) / sum(y_train) if sum(y_train) > 0 else 8.0
        pos_weight = torch.tensor([pos_weight_value]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        early_stopping = EarlyStopping(patience=config.get('patience', 15))
        best_auc = 0.0

        for epoch in range(config.get('epochs', 50)):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_labels, val_outputs = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = torch.sigmoid(model(inputs))
                    val_labels.extend(labels.cpu().numpy())
                    val_outputs.extend(outputs.cpu().numpy())

            try:
                val_auc = roc_auc_score(val_labels, val_outputs)
            except:
                val_auc = 0.0

            if val_auc > best_auc:
                best_auc = val_auc
                if save_temp:
                    torch.save(model.state_dict(), self.models_dir / 'temp_pytorch_model.pth')

            if early_stopping(val_auc):
                break

        if save_temp and (self.models_dir / 'temp_pytorch_model.pth').exists():
            model.load_state_dict(torch.load(self.models_dir / 'temp_pytorch_model.pth', map_location=self.device))

        model.eval()
        return model, best_auc

    def train_xgboost_model(self, X_train, y_train, params):
        if params is None:
            params = {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 100}

        if 'scale_pos_weight' not in params:
            params['scale_pos_weight'] = len(y_train) / sum(y_train) if sum(y_train) > 0 else 8.0

        params.update({'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0})

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        return model

    # --------------------------------------------------------------------------
    # Optimization Logic
    # --------------------------------------------------------------------------

    def run_optimization(self, X_train, y_train, X_test, y_test, model_type, n_trials, metric="auc_roc", use_cv=False):
        print(f"\nOptimizing {model_type} ({n_trials} trials)...")

        X_full = np.vstack([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])

        def objective(trial):
            if model_type == 'PyTorch':
                n_layers = trial.suggest_int('n_layers', 1, 4)
                hidden_dims = [trial.suggest_int(f'hidden_dim_{i}', 32, 512, log=True) for i in range(n_layers)]
                config = {
                    'n_layers': n_layers,
                    'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                    'epochs': 50,
                    'patience': 10
                }
                for i, hdim in enumerate(hidden_dims): config[f'hidden_dim_{i}'] = hdim

                if use_cv:
                    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    scores = []
                    for t_idx, v_idx in skf.split(X_full, y_full):
                        model, best_auc = self.train_pytorch_model(X_full[t_idx], y_full[t_idx], X_full[v_idx],
                                                                   y_full[v_idx], config)
                        scores.append(best_auc)
                    return np.mean(scores)
                else:
                    model, best_auc = self.train_pytorch_model(X_train, y_train, X_test, y_test, config)
                    return best_auc

            elif model_type == 'XGBoost':
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 4.0, 12.0)
                }

                model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss', verbosity=0)

                if use_cv:
                    scoring = 'roc_auc' if metric == "auc_roc" else 'average_precision'
                    scores = cross_val_score(model, X_full, y_full, cv=5, scoring=scoring, n_jobs=-1)
                    return np.mean(scores)
                else:
                    model.fit(X_train, y_train)
                    probs = model.predict_proba(X_test)[:, 1]
                    if metric == "auc_pr": return average_precision_score(y_test, probs)
                    return roc_auc_score(y_test, probs)

        study = optuna.create_study(direction='maximize', study_name=f"{model_type}_opt")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.save_optuna_csv(model_type, study.best_params, study.best_value, metric)
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
        weights = (best_w, 1.0 - best_w)

        self.save_optuna_csv("Ensemble_Weights", {'weights': weights}, study.best_value, metric)
        return weights

    def save_optuna_csv(self, model_name, params, score, metric):
        file_path = self.optuna_dir / 'optuna_results_general.csv'
        record = {
            'model': model_name,
            'best_params': str(params),
            'result': score,
            'metric': metric,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        df = pd.DataFrame([record])
        if file_path.exists():
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)

    # --------------------------------------------------------------------------
    # COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS
    # --------------------------------------------------------------------------

    def analyze_feature_importance(self, pytorch_model, xgb_model, X_sample, weights):
        print("\nGenerating Detailed Feature Importance Analysis...")

        # --- 1. XGBoost Analysis (SHAP) ---
        explainer_xgb = shap.TreeExplainer(xgb_model)
        shap_values_xgb = explainer_xgb.shap_values(X_sample)
        xgb_imp = np.abs(shap_values_xgb).mean(0)
        xgb_imp_norm = xgb_imp / xgb_imp.sum()

        # Save XGBoost Plots (Bar & Beeswarm)
        # Bar Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_xgb, X_sample, feature_names=self.feature_names, plot_type="bar", show=False)
        plt.title('XGBoost SHAP Importance')
        plt.savefig(self.figures_dir / 'xgboost_analysis' / 'xgboost_shap_bar.png', bbox_inches='tight')
        plt.close()

        # Beeswarm Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_xgb, X_sample, feature_names=self.feature_names, show=False)
        plt.title('XGBoost SHAP Distribution')
        plt.savefig(self.figures_dir / 'xgboost_analysis' / 'xgboost_shap_beeswarm.png', bbox_inches='tight')
        plt.close()

        # Save Top 10 XGBoost
        self._plot_top_features(xgb_imp_norm, 'XGBoost (SHAP)',
                                self.figures_dir / 'xgboost_analysis' / 'xgboost_top10.png', color='skyblue')

        # --- 2. MLP Analysis (Gradients) ---
        pytorch_model.eval()
        X_t = torch.tensor(X_sample, dtype=torch.float32).to(self.device)
        X_t.requires_grad = True
        outputs = pytorch_model(X_t)
        outputs.backward(torch.ones_like(outputs))
        grads = X_t.grad.cpu().numpy()

        mlp_imp = np.abs(grads).mean(0)
        mlp_imp_norm = mlp_imp / mlp_imp.sum()

        # Save Top 10 MLP
        self._plot_top_features(mlp_imp_norm, 'MLP (Gradients)',
                                self.figures_dir / 'mlp_analysis' / 'mlp_top10_gradients.png', color='salmon')

        # Save MLP Heatmap (First 50 samples)
        plt.figure(figsize=(12, 8))
        sns.heatmap(np.abs(grads[:50]), cmap='viridis', xticklabels=False, yticklabels=False)
        plt.title("MLP Gradients Heatmap (First 50 samples)")
        plt.xlabel("Features")
        plt.ylabel("Samples")
        plt.savefig(self.figures_dir / 'mlp_analysis' / 'mlp_gradients_heatmap.png', bbox_inches='tight')
        plt.close()

        # --- 3. Ensemble Analysis ---
        ensemble_imp_norm = (weights[0] * mlp_imp_norm) + (weights[1] * xgb_imp_norm)
        ensemble_imp_norm = ensemble_imp_norm / ensemble_imp_norm.sum()

        # Comparison Plot
        top_n = 15
        idx = np.argsort(ensemble_imp_norm)[::-1][:top_n]
        top_feats = [self.feature_names[i] for i in idx]

        plt.figure(figsize=(14, 8))
        x = np.arange(top_n)
        width = 0.25
        plt.bar(x - width, mlp_imp_norm[idx], width, label='MLP', alpha=0.8, color='salmon')
        plt.bar(x, xgb_imp_norm[idx], width, label='XGBoost', alpha=0.8, color='skyblue')
        plt.bar(x + width, ensemble_imp_norm[idx], width, label='Ensemble', alpha=0.8, color='lightgreen')
        plt.xticks(x, top_feats, rotation=45, ha='right')
        plt.legend()
        plt.title('Ensemble Feature Importance Comparison')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'ensemble_analysis' / 'ensemble_comparison.png', bbox_inches='tight')
        plt.close()

        # Correlation Scatter Plot
        plt.figure(figsize=(8, 8))
        plt.scatter(mlp_imp_norm, xgb_imp_norm, alpha=0.6, color='purple')
        plt.xlabel('MLP Importance')
        plt.ylabel('XGBoost Importance')
        plt.title(f'Importance Correlation (r={np.corrcoef(mlp_imp_norm, xgb_imp_norm)[0, 1]:.2f})')
        plt.plot([0, max(mlp_imp_norm)], [0, max(mlp_imp_norm)], 'k--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'ensemble_analysis' / 'ensemble_correlation_scatter.png', bbox_inches='tight')
        plt.close()

        # Save CSV Data to REPORTS directory (Corrected)
        df_imp = pd.DataFrame({
            'feature': self.feature_names,
            'mlp_importance': mlp_imp_norm,
            'xgb_importance': xgb_imp_norm,
            'ensemble_importance': ensemble_imp_norm
        }).sort_values('ensemble_importance', ascending=False)

        df_imp.to_csv(self.feat_imp_report_dir / 'ensemble_analysis' / 'ensemble_importance_data.csv', index=False)

        return np.corrcoef(mlp_imp_norm, xgb_imp_norm)[0, 1]

    def _plot_top_features(self, importance_values, title, save_path, color='blue', n=10):
        """Helper to plot horizontal bar chart of top N features."""
        idx = np.argsort(importance_values)[::-1][:n]
        top_feats = [self.feature_names[i] for i in idx]
        top_vals = importance_values[idx]

        plt.figure(figsize=(10, 6))
        plt.barh(top_feats[::-1], top_vals[::-1], color=color, alpha=0.8, edgecolor='black')
        plt.title(f'Top {n} Features - {title}')
        plt.xlabel('Normalized Importance')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    # --------------------------------------------------------------------------
    # Main Execution Method
    # --------------------------------------------------------------------------

    def run(self, n_trials_pytorch=30, n_trials_xgb=30, n_trials_ensemble=20, metric="auc_roc", use_cv=False):
        X, y = self.load_data()

        # Stratified Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 1. Optimize Hyperparameters
        best_pytorch_params = self.run_optimization(X_train, y_train, X_test, y_test, 'PyTorch', n_trials_pytorch,
                                                    metric, use_cv)
        best_xgb_params = self.run_optimization(X_train, y_train, X_test, y_test, 'XGBoost', n_trials_xgb, metric,
                                                use_cv)

        # 2. Train Optimized Models
        print("\nTraining final models with optimized parameters...")
        best_pytorch_params['epochs'] = 100
        best_pytorch_params['patience'] = 15
        pytorch_model, _ = self.train_pytorch_model(X_train, y_train, X_test, y_test, best_pytorch_params,
                                                    save_temp=True)

        xgb_model = self.train_xgboost_model(X_train, y_train, best_xgb_params)

        # 3. Optimize Weights
        best_weights = self.optimize_weights(pytorch_model, xgb_model, X_test, y_test, n_trials_ensemble, metric)

        # 4. Final Evaluation
        ensemble = HybridEnsemble(pytorch_model, xgb_model, self.device)
        probs, _, _ = ensemble.ensemble_predict_proba(X_test, best_weights)

        final_auc = roc_auc_score(y_test, probs)
        final_pr = average_precision_score(y_test, probs)

        print(f"\nFinal Results on Test Set:")
        print(f"  AUC-ROC: {final_auc:.4f}")
        print(f"  AUC-PR:  {final_pr:.4f}")
        print(f"  Optimal Weights: PyTorch={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}")

        # 5. Feature Analysis (Restored Full Analysis)
        corr = self.analyze_feature_importance(pytorch_model, xgb_model, X_test[:200], best_weights)
        print(f"  Feature Importance Correlation: {corr:.4f}")

        # 6. Save Artifacts
        print("\nSaving artifacts...")
        torch.save(pytorch_model.state_dict(), self.models_dir / 'pytorch_model_optimized.pth')
        xgb_model.save_model(self.models_dir / 'xgboost_model_optimized.json')
        joblib.dump(ensemble, self.models_dir / 'hybrid_ensemble_optimized.pkl')
        joblib.dump(self.scaler, self.models_dir / 'scaler_optimized.pkl')

        config = {
            'pytorch_params': best_pytorch_params,
            'xgb_params': best_xgb_params,
            'weights': best_weights,
            'metrics': {'auc': final_auc, 'pr': final_pr}
        }
        joblib.dump(config, self.models_dir / 'optimal_configuration.pkl')

        print(f"Done! Models saved to {self.models_dir}")


def main():
    print("HYBRID ENSEMBLE OPTIMIZATION PIPELINE")

    metric = input("\nSelect metric (1: AUC-ROC, 2: AUC-PR) [2]: ") or "2"
    metric = "auc_pr" if metric == "2" else "auc_roc"

    cv_in = input("\nUse Cross-Validation? (y/n) [y]: ") or "y"
    use_cv = cv_in.lower() == "y"

    n_py = int(input("PyTorch trials [30]: ") or "30")
    n_xgb = int(input("XGBoost trials [30]: ") or "30")
    n_ens = int(input("Ensemble trials [20]: ") or "20")

    pipeline = HybridEnsemblePipeline()
    pipeline.run(n_py, n_xgb, n_ens, metric, use_cv)


if __name__ == "__main__":
    main()