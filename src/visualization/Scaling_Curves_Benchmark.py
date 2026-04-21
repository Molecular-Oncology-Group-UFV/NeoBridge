import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score
import xgboost as xgb
import joblib
import os
import optuna
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
from tqdm import tqdm
import time
from scipy.optimize import curve_fit
from pathlib import Path
import sys
import seaborn as sns

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


class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.3):
        super(ImprovedMLP, self).__init__()
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
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

        total_weight = weights[0] + weights[1]
        normalized_weights = [w / total_weight for w in weights]

        ensemble_proba = (normalized_weights[0] * pytorch_proba +
                          normalized_weights[1] * xgb_proba)

        return ensemble_proba, pytorch_proba, xgb_proba


class ScalingCurveBenchmark:
    def __init__(self):
        # Path Configuration
        self.project_root = Path(__file__).resolve().parent.parent.parent

        # Input Data Path
        self.data_path = self.project_root / 'data' / 'processed' / 'dataset_80_20_normalized_tcr_with_cohort_and_rank.csv'

        # Output Paths
        self.reports_dir = self.project_root / 'results' / 'reports' / 'scaling_curves'
        self.figures_dir = self.project_root / 'results' / 'figures' / 'scaling_curves'

        # Create directories
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def train_pytorch_model(self, X_train, y_train, X_val, y_val, X_test, y_test, pytorch_params, device):
        """Train PyTorch model with separate validation and test sets."""
        hidden_dims = [pytorch_params[f'hidden_dim_{i}'] for i in range(pytorch_params['n_layers'])]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        # --- CORRECCIÓN AQUÍ: drop_last=True ---
        # Esto evita que el último batch tenga tamaño 1 y rompa el BatchNorm
        train_loader = DataLoader(
            train_dataset,
            batch_size=pytorch_params['batch_size'],
            shuffle=True,
            drop_last=True
        )

        # El validation loader no necesita drop_last porque corre en mode.eval()
        val_loader = DataLoader(val_dataset, batch_size=pytorch_params['batch_size'])

        model = ImprovedMLP(X_train.shape[1], hidden_dims, pytorch_params['dropout_rate']).to(device)

        pos_weight_value = len(y_train) / sum(y_train) if sum(y_train) > 0 else 8.0
        pos_weight = torch.tensor([pos_weight_value]).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(
            model.parameters(),
            lr=pytorch_params['lr'],
            weight_decay=pytorch_params['weight_decay']
        )

        early_stopping = EarlyStopping(patience=pytorch_params.get('patience', 15))
        best_val_auc = 0
        best_model_state = None

        epochs = pytorch_params.get('epochs', 100)
        for epoch in range(epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_labels = []
            val_outputs = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = torch.sigmoid(model(inputs))
                    val_labels.extend(labels.cpu().numpy())
                    val_outputs.extend(outputs.cpu().numpy())

            try:
                val_auc = roc_auc_score(val_labels, val_outputs)
            except ValueError:
                val_auc = 0.0

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()

            if early_stopping(val_auc):
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Final evaluation
        model.eval()
        X_test_tensor_device = X_test_tensor.to(device)
        with torch.no_grad():
            test_logits = model(X_test_tensor_device)
            test_proba = torch.sigmoid(test_logits).cpu().numpy().flatten()
        final_test_auc = roc_auc_score(y_test, test_proba)

        return model, device, best_val_auc, final_test_auc

    def train_xgboost_model(self, X_train, y_train, X_val, y_val, X_test, y_test, xgb_params):
        """Train XGBoost model with validation and test evaluation."""
        if 'scale_pos_weight' not in xgb_params:
            scale_pos_weight = len(y_train) / sum(y_train) if sum(y_train) > 0 else 8.0
            xgb_params['scale_pos_weight'] = scale_pos_weight

        xgb_params.update({
            'random_state': 42,
            'eval_metric': 'logloss',
            'verbosity': 0,
            'use_label_encoder': False
        })

        eval_set = [(X_val, y_val)]

        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        val_proba = xgb_model.predict_proba(X_val)[:, 1]
        test_proba = xgb_model.predict_proba(X_test)[:, 1]

        val_auc = roc_auc_score(y_val, val_proba)
        test_auc = roc_auc_score(y_test, test_proba)

        return xgb_model, val_auc, test_auc

    def optimize_hyperparameters(self, X_full, y_full, optimization_metric="auc_roc", n_trials=50):
        """Optimize hyperparameters using train/validation/test split."""
        print("\n" + "=" * 70)
        print("HYPERPARAMETER OPTIMIZATION (60/20/20 Split)")
        print("=" * 70)

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
        )

        print(f"Total data: {X_full.shape[0]}")
        print(f"Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")

        print("\nOptimizing PyTorch hyperparameters...")

        def objective_pytorch(trial):
            n_layers = trial.suggest_int('n_layers', 1, 4)
            hidden_dims = []
            for i in range(n_layers):
                hidden_dims.append(trial.suggest_int(f'hidden_dim_{i}', 32, 512, log=True))

            config = {
                'n_layers': n_layers,
                'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'epochs': 100,
                'patience': 15
            }
            for i, hdim in enumerate(hidden_dims):
                config[f'hidden_dim_{i}'] = hdim

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, device, val_auc, _ = self.train_pytorch_model(
                X_train, y_train, X_val, y_val, X_test, y_test, config, device
            )

            if optimization_metric == "auc_pr":
                model.eval()
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
                with torch.no_grad():
                    probabilities = torch.sigmoid(model(X_val_tensor)).cpu().numpy().flatten()
                score = average_precision_score(y_val, probabilities)
            else:
                score = val_auc
            return score

        study_pytorch = optuna.create_study(direction='maximize', study_name='pytorch_scaling')
        study_pytorch.optimize(objective_pytorch, n_trials=n_trials, show_progress_bar=True)
        best_pytorch_params = study_pytorch.best_params

        print("\nOptimizing XGBoost hyperparameters...")

        def objective_xgboost(trial):
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
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 4.0, 12.0),
                'random_state': 42,
                'verbosity': 0,
                'use_label_encoder': False
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_val_pred = model.predict_proba(X_val)[:, 1]

            if optimization_metric == "auc_pr":
                return average_precision_score(y_val, y_val_pred)
            return roc_auc_score(y_val, y_val_pred)

        study_xgb = optuna.create_study(direction='maximize', study_name='xgboost_scaling')
        study_xgb.optimize(objective_xgboost, n_trials=n_trials, show_progress_bar=True)
        best_xgb_params = study_xgb.best_params

        print("\nOptimizing ensemble weights...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pytorch_model, device, _, _ = self.train_pytorch_model(
            X_train, y_train, X_val, y_val, X_test, y_test, best_pytorch_params, device
        )
        xgb_model, _, _ = self.train_xgboost_model(
            X_train, y_train, X_val, y_val, X_test, y_test, best_xgb_params
        )
        ensemble = HybridEnsemble(pytorch_model, xgb_model, device)

        def objective_weights(trial):
            w_pytorch = trial.suggest_float('weight_pytorch', 0.1, 0.9)
            ensemble_proba, _, _ = ensemble.ensemble_predict_proba(X_val, (w_pytorch, 1.0 - w_pytorch))
            if optimization_metric == "auc_pr":
                return average_precision_score(y_val, ensemble_proba)
            return roc_auc_score(y_val, ensemble_proba)

        study_weights = optuna.create_study(direction='maximize', study_name='ensemble_weights')
        study_weights.optimize(objective_weights, n_trials=20, show_progress_bar=True)

        best_w_pytorch = study_weights.best_params['weight_pytorch']
        best_weights_score = study_weights.best_value

        # Final Test Evaluation
        ensemble_test_proba, _, _ = ensemble.ensemble_predict_proba(
            X_test, (best_w_pytorch, 1.0 - best_w_pytorch)
        )
        if optimization_metric == "auc_pr":
            test_score = average_precision_score(y_test, ensemble_test_proba)
        else:
            test_score = roc_auc_score(y_test, ensemble_test_proba)

        return {
            'pytorch_params': best_pytorch_params,
            'xgb_params': best_xgb_params,
            'ensemble_weights': (best_w_pytorch, 1.0 - best_w_pytorch),
            'validation_score': best_weights_score,
            'test_score': test_score,
            'optimization_metric': optimization_metric
        }

    def calculate_scaling_curve(self, X_full, y_full, optimal_params, scaling_points):
        """Calculate scaling curve with train/validation/test split."""

        scaling_points = [p for p in scaling_points if p <= X_full.shape[0]]
        if X_full.shape[0] not in scaling_points:
            scaling_points.append(X_full.shape[0])

        # Ordenar y eliminar duplicados por si acaso
        scaling_points = sorted(list(set(scaling_points)))

        print(f"\n" + "=" * 70)
        print("CALCULATING SCALING CURVE")
        print(f"Points ({len(scaling_points)} steps): {scaling_points}")
        print("=" * 70)

        results = {
            'data_size': [],
            'pytorch_val_auc': [], 'pytorch_test_auc': [],
            'xgb_val_auc': [], 'xgb_test_auc': [],
            'ensemble_val_auc': [], 'ensemble_test_auc': [],
            'ensemble_test_auc_pr': [],
            'ensemble_test_f2': [],
            'train_time': []
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for data_size in tqdm(scaling_points, desc="Calculating points"):
            if data_size < X_full.shape[0]:
                X_train_val, _, y_train_val, _ = train_test_split(
                    X_full, y_full, train_size=data_size, random_state=42, stratify=y_full
                )
            else:
                X_train_val, y_train_val = X_full, y_full

            X_train_val_small, X_test, y_train_val_small, y_test = train_test_split(
                X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val_small, y_train_val_small, test_size=0.25, random_state=42, stratify=y_train_val_small
            )

            start_time = time.time()

            pytorch_model, device, pytorch_val_auc, pytorch_test_auc = self.train_pytorch_model(
                X_train, y_train, X_val, y_val, X_test, y_test, optimal_params['pytorch_params'], device
            )

            xgb_model, xgb_val_auc, xgb_test_auc = self.train_xgboost_model(
                X_train, y_train, X_val, y_val, X_test, y_test, optimal_params['xgb_params']
            )

            ensemble = HybridEnsemble(pytorch_model, xgb_model, device)

            ensemble_val_proba, _, _ = ensemble.ensemble_predict_proba(X_val, optimal_params['ensemble_weights'])
            ensemble_val_auc = roc_auc_score(y_val, ensemble_val_proba)

            ensemble_test_proba, _, _ = ensemble.ensemble_predict_proba(X_test, optimal_params['ensemble_weights'])
            ensemble_test_auc = roc_auc_score(y_test, ensemble_test_proba)
            ensemble_test_auc_pr = average_precision_score(y_test, ensemble_test_proba)

            ensemble_test_preds = (ensemble_test_proba > 0.5).astype(int)
            f2_val = fbeta_score(y_test, ensemble_test_preds, beta=2)

            train_time = time.time() - start_time

            results['data_size'].append(data_size)
            results['pytorch_val_auc'].append(pytorch_val_auc)
            results['pytorch_test_auc'].append(pytorch_test_auc)
            results['xgb_val_auc'].append(xgb_val_auc)
            results['xgb_test_auc'].append(xgb_test_auc)
            results['ensemble_val_auc'].append(ensemble_val_auc)
            results['ensemble_test_auc'].append(ensemble_test_auc)
            results['ensemble_test_auc_pr'].append(ensemble_test_auc_pr)
            results['ensemble_test_f2'].append(f2_val)
            results['train_time'].append(train_time)

        return pd.DataFrame(results)

    def plot_results(self, results_df, optimal_params):
        """Generate and save publication-ready scaling curve plots."""
        print("Generating scientific scaling curves...")

        # Función de Ley de Potencia para el ajuste teórico (Learning Curve Theory)
        # y = a - b * x^(-c)
        # a: asíntota (rendimiento máximo teórico)
        def power_law(x, a, b, c):
            return a - b * np.power(x, -c)

        # Preparar datos para el ajuste
        x_data = results_df['data_size'].values
        y_data = results_df['ensemble_test_auc'].values

        # Intentar ajustar la curva teórica (puede fallar si los datos son muy ruidosos)
        popt = None
        try:
            # Límites razonables para a (0.5 a 1.0), b (>0), c (>0)
            popt, _ = curve_fit(power_law, x_data, y_data,
                                p0=[0.9, 0.5, 0.3],
                                bounds=([0.5, 0, 0], [1.0, np.inf, np.inf]),
                                maxfev=5000)
        except Exception as e:
            print(f"No se pudo ajustar la curva teórica: {e}")

        # --- 1. MAIN PLOT: Scientific Scaling Curve ---
        fig, ax = plt.subplots(figsize=(10, 7))

        # Colores científicos
        color_test = '#08519c'  # Azul oscuro fuerte
        color_val = '#9ecae1'  # Azul claro suave
        color_fit = '#cb181d'  # Rojo oscuro para la teoría

        # Plot Datos Reales (Con marcadores más pequeños y elegantes)
        ax.plot(results_df['data_size'], results_df['ensemble_test_auc'],
                'o-', color=color_test, linewidth=1.5, markersize=5,
                label='Test AUC (Empirical)', alpha=0.9, zorder=3)

        ax.plot(results_df['data_size'], results_df['ensemble_val_auc'],
                's--', color=color_val, linewidth=1.5, markersize=4,
                label='Validation AUC', alpha=0.7, zorder=2)

        # Plot Curva Teórica (Si se calculó)
        if popt is not None:
            x_smooth = np.linspace(x_data.min(), x_data.max(), 500)
            y_smooth = power_law(x_smooth, *popt)
            ax.plot(x_smooth, y_smooth, '--', color=color_fit, linewidth=2,
                    label=f'Power Law Fit (Asymptote: {popt[0]:.3f})', zorder=1)

            # Marcar la asíntota en el texto
            ax.text(x_data[-1], popt[0] + 0.005, f"Limit: {popt[0]:.3f}",
                    color=color_fit, fontsize=9, fontweight='bold', va='bottom', ha='right')

        # --- MEJORA DEL EJE X (EVITAR SUPERPOSICIÓN AL FINAL) ---
        xticks = results_df['data_size'].tolist()

        # Si los dos últimos ticks están a menos de 300 muestras de distancia,
        # quitamos el penúltimo solo de las etiquetas (el punto se sigue graficando)
        if len(xticks) >= 2 and (xticks[-1] - xticks[-2] < 300):
            ticks_to_show = xticks[:-2] + [xticks[-1]]
        else:
            ticks_to_show = xticks

        ax.set_xticks(ticks_to_show)
        ax.set_xticklabels(ticks_to_show, rotation=45, fontsize=9)

        # Estética General
        ax.set_xlabel('Training Set Size (Samples)', fontweight='bold', fontsize=11)
        ax.set_ylabel('AUC-ROC', fontweight='bold', fontsize=11)
        ax.set_title('Learning Curve Analysis: Data Scaling', fontweight='bold', fontsize=13, pad=15)

        # Grid suave y limpieza de bordes (Despine)
        ax.grid(True, linestyle=':', alpha=0.6)
        sns.despine(ax=ax, trim=True)  # Quita bordes arriba y derecha y recorta los ejes

        # Leyenda limpia
        ax.legend(frameon=True, framealpha=0.9, loc='lower right', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'scaling_curve_test_vs_validation.png', dpi=300, bbox_inches='tight')
        plt.close()

        # --- 2. Generalization Gap (Simplified) ---
        plt.figure(figsize=(10, 6))
        gap = results_df['ensemble_val_auc'] - results_df['ensemble_test_auc']

        plt.fill_between(results_df['data_size'], gap, color='#de2d26', alpha=0.1)
        plt.plot(results_df['data_size'], gap, 'o-', color='#de2d26', linewidth=1.5, markersize=5)

        plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

        # Eje X corregido
        plt.xticks(results_df['data_size'], rotation=45, fontsize=9)

        plt.xlabel('Training Set Size', fontweight='bold')
        plt.ylabel('Generalization Gap (Val - Test)', fontweight='bold')
        plt.title('Overfitting Analysis (Generalization Gap)', fontweight='bold')

        sns.despine()
        plt.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'generalization_gap_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # --- 3. Comprehensive Analysis (4-panel) - Actualizado ejes X ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # A: Ensemble Raw
        axes[0, 0].plot(results_df['data_size'], results_df['ensemble_test_auc'], 'o-', color=color_test)
        axes[0, 0].set_title('Ensemble Test Performance')
        axes[0, 0].set_xticks(results_df['data_size'][::2])  # Mostrar uno sí uno no si hay muchos
        axes[0, 0].tick_params(axis='x', rotation=45)

        # B: Model Comparison
        axes[0, 1].plot(results_df['data_size'], results_df['ensemble_test_auc'], 'o-', label='Ensemble',
                        color=color_test)
        axes[0, 1].plot(results_df['data_size'], results_df['pytorch_test_auc'], 's--', label='PyTorch',
                        color='#74c476', alpha=0.7)
        axes[0, 1].plot(results_df['data_size'], results_df['xgb_test_auc'], '^--', label='XGBoost', color='#fd8d3c',
                        alpha=0.7)
        axes[0, 1].set_title('Architecture Comparison')
        axes[0, 1].legend()
        axes[0, 1].set_xticks(results_df['data_size'][::2])
        axes[0, 1].tick_params(axis='x', rotation=45)

        # C: AUC-PR
        axes[1, 0].plot(results_df['data_size'], results_df['ensemble_test_auc_pr'], 'o-', color='#6a51a3',
                        label='AUC-PR')
        axes[1, 0].set_title('Precision-Recall Area (AUC-PR)')
        axes[1, 0].set_xticks(results_df['data_size'][::2])
        axes[1, 0].tick_params(axis='x', rotation=45)

        # D: Saturation (Normalized)
        max_auc = results_df['ensemble_test_auc'].max()
        if max_auc > 0:
            norm_perf = results_df['ensemble_test_auc'] / max_auc
            axes[1, 1].plot(results_df['data_size'], norm_perf, 'o-', color='black')
            axes[1, 1].axhline(0.99, linestyle='--', color='red', label='99% of Max')
            axes[1, 1].set_title('Performance Saturation (% of Max)')
            axes[1, 1].legend()
            axes[1, 1].set_xticks(results_df['data_size'][::2])
            axes[1, 1].tick_params(axis='x', rotation=45)

        for ax in axes.flatten():
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax)

        plt.suptitle('Comprehensive Generalization Benchmark', fontweight='bold', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'comprehensive_generalization_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n Scientific plots saved in: {self.figures_dir}")

    def run(self):
        """Main execution method."""
        print("=" * 80)
        print("SCALING CURVE ANALYSIS")
        print("=" * 80)

        if not self.data_path.exists():
            print(f"Error: Dataset not found at {self.data_path}")
            return

        print("\nLoading dataset...")
        df = pd.read_csv(self.data_path)
        X = df.select_dtypes(include=['float64', 'int64']).drop(
            columns=['immunogenicity', 'mut_is_binding_pos', 'bestWTMatchType_I', 'DAI_MixMHC_mbp', 'mutant_rank'],
            errors='ignore'
        )
        y = df['immunogenicity']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(f"Data loaded: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")

        # Configuration Input
        print("\nSelect optimization metric:")
        print("1. AUC-ROC")
        print("2. AUC-PR")
        metric_choice = input("\nSelect (1 or 2) [2]: ") or "2"
        optimization_metric = "auc_pr" if metric_choice == "2" else "auc_roc"

        n_trials_input = input("\nNumber of optimization trials [30]: ") or "30"
        n_trials = int(n_trials_input)

        # --- CAMBIO REALIZADO AQUÍ ---
        # Generar puntos de escalado cada 500 muestras automáticamente hasta el total
        max_samples = X_scaled.shape[0]
        # range(start, stop, step). Se usa max_samples + 1 para incluir el último rango si es exacto
        scaling_points = list(range(500, max_samples + 1, 500))

        # Asegurar que el último punto sea el total exacto si no es múltiplo de 500
        # (para aprovechar todos los datos disponibles)
        if scaling_points[-1] != max_samples:
            scaling_points.append(max_samples)

        print(f"\nScaling points generated (every 500): {scaling_points}")
        # -----------------------------

        # Execution
        print("\nPHASE 1: HYPERPARAMETER OPTIMIZATION")
        optimal_params = self.optimize_hyperparameters(X_scaled, y.values, optimization_metric, n_trials)
        joblib.dump(optimal_params, self.reports_dir / 'optimal_hyperparameters.pkl')

        print("\nPHASE 2: SCALING CURVE CALCULATION")
        results_df = self.calculate_scaling_curve(X_scaled, y.values, optimal_params, scaling_points)
        results_df.to_csv(self.reports_dir / 'scaling_curve_results.csv', index=False)

        print("\nPHASE 3: VISUALIZATION")
        self.plot_results(results_df, optimal_params)

        print("\nDone! Results saved to:")
        print(f"Reports: {self.reports_dir}")
        print(f"Figures: {self.figures_dir}")


if __name__ == "__main__":
    benchmark = ScalingCurveBenchmark()
    benchmark.run()