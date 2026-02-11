import sys
import time
import warnings
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import signal
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to activate scikit-learn-intelex for acceleration
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass

# ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, fbeta_score,
    precision_score, recall_score, balanced_accuracy_score,
    confusion_matrix, classification_report, accuracy_score
)

# Parallelization imports
from joblib import Parallel, delayed, parallel_backend, Memory
import multiprocessing as mp

# Bayesian optimization imports
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Base model imports
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

# Import imbalanced-learn if available
try:
    from imblearn.ensemble import (
        BalancedRandomForestClassifier, EasyEnsembleClassifier,
        RUSBoostClassifier, BalancedBaggingClassifier
    )
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False


# ============================================================================
# UTILITIES AND TIMEOUT HANDLERS
# ============================================================================

class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager for execution time limitation"""

    def signal_handler(signum, frame):
        raise TimeoutException(f"Timeout after {seconds} seconds")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


# ============================================================================
# CONFIGURATION AND DATA STRUCTURES
# ============================================================================

class ExecutionMode(Enum):
    QUICK = "quick"
    STANDARD = "standard"
    FULL = "full"


@dataclass
class ModelConfig:
    """Configuration for an individual model"""
    name: str
    class_path: str
    priority: int
    imbalance_strategy: str
    module_path: str = ""
    class_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if '.' in self.class_path:
            parts = self.class_path.rsplit('.', 1)
            self.module_path = parts[0]
            self.class_name = parts[1]


@dataclass
class BenchmarkResults:
    """Benchmarking results for a model"""
    model_name: str
    execution_time: float
    cv_scores: Dict[str, List[float]]
    test_scores: Dict[str, float]
    best_params: Dict[str, Any]
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    feature_importance: Optional[np.ndarray] = None

    def get_mean_cv_scores(self) -> Dict[str, float]:
        return {metric: np.mean(scores) for metric, scores in self.cv_scores.items()}

    def get_std_cv_scores(self) -> Dict[str, float]:
        return {metric: np.std(scores) for metric, scores in self.cv_scores.items()}


class ProgressTracker:
    """Execution progress tracking"""

    def __init__(self, total_models: int):
        self.total_models = total_models
        self.completed_models = 0
        self.start_time = time.time()
        self.model_times = {}
        self.failed_models = []

    def start_model(self, model_name: str):
        self.current_model = model_name
        self.model_start_time = time.time()

    def end_model(self, success: bool = True, error_msg: str = None):
        model_time = time.time() - self.model_start_time
        self.model_times[self.current_model] = model_time

        if success:
            self.completed_models += 1
        else:
            self.failed_models.append(self.current_model)

        elapsed = time.time() - self.start_time
        if self.completed_models > 0:
            avg_time = elapsed / self.completed_models
            remaining = avg_time * (self.total_models - self.completed_models)
        else:
            remaining = 0


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

class ConfigLoader:

    @staticmethod
    def load_configs() -> Tuple[Dict, Dict, Dict]:
        config_dir = Path(__file__).parent / "Benchmarking_Configs"

        if not config_dir.exists():
            raise FileNotFoundError(f"Directory {config_dir} not found")

        configs = []
        for config_file in ["models_to_benchmark.yaml",
                            "hyperparameter_grids.yaml",
                            "parallelization_config.yaml"]:
            file_path = config_dir / config_file
            if not file_path.exists():
                raise FileNotFoundError(f"File {file_path} not found")

            with open(file_path, 'r') as f:
                configs.append(yaml.safe_load(f))

        return tuple(configs)

    @staticmethod
    def create_model_configs(models_config: Dict, hyperparams_config: Dict) -> List[ModelConfig]:
        model_configs = []

        for model_info in models_config['models']:
            if not IMBALANCED_LEARN_AVAILABLE and model_info['class'].startswith('imblearn'):
                continue

            model_params = hyperparams_config.get(model_info['name'], {})

            config = ModelConfig(
                name=model_info['name'],
                class_path=model_info['class'],
                priority=model_info['priority'],
                imbalance_strategy=model_info.get('imbalance_strategy', 'none'),
                params=model_params
            )
            model_configs.append(config)

        return model_configs


# ============================================================================
# DATA HANDLER
# ============================================================================

class DataHandler:

    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        dataset_path = self.config['dataset']['path']
        if not Path(dataset_path).exists():
            alt_paths = [
                dataset_path,
                f"data/processed/{Path(dataset_path).name}",  # Busca en la carpeta correcta
                f"../data/processed/{Path(dataset_path).name}",
                f"../../data/processed/{Path(dataset_path).name}"
            ]
            for path in alt_paths:
                if Path(path).exists():
                    dataset_path = path
                    break

        df = pd.read_csv(dataset_path)

        target_col = self.config['dataset']['target_column']
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        y = df[target_col].values

        X = df.select_dtypes(include=['float64', 'int64']).copy()
        columns_to_drop = [target_col, 'immunogenicity', 'mut_is_binding_pos',
                           'bestWTMatchType_I', 'DAI_MixMHC_mbp', 'mutant_rank']

        for col in columns_to_drop:
            if col in X.columns:
                X = X.drop(columns=[col])

        feature_names = X.columns.tolist()

        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)

        return X, y, feature_names

    def prepare_splits(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        test_size = self.config['dataset']['test_size']
        random_state = self.config['dataset']['random_state']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        return X_train_scaled, X_test_scaled, y_train, y_test


# ============================================================================
# DYNAMIC MODEL LOADER
# ============================================================================

class ModelLoader:
    # Class name to actual class mapping
    CLASS_MAP = {
        # Tree-based
        'XGBClassifier': ('xgboost', 'XGBClassifier'),
        'LGBMClassifier': ('lightgbm', 'LGBMClassifier'),
        'CatBoostClassifier': ('catboost', 'CatBoostClassifier'),

        # Sklearn ensembles
        'RandomForestClassifier': ('sklearn.ensemble', 'RandomForestClassifier'),
        'ExtraTreesClassifier': ('sklearn.ensemble', 'ExtraTreesClassifier'),
        'GradientBoostingClassifier': ('sklearn.ensemble', 'GradientBoostingClassifier'),
        'HistGradientBoostingClassifier': ('sklearn.ensemble', 'HistGradientBoostingClassifier'),
        'AdaBoostClassifier': ('sklearn.ensemble', 'AdaBoostClassifier'),
        'BaggingClassifier': ('sklearn.ensemble', 'BaggingClassifier'),

        # Linear models
        'LogisticRegression': ('sklearn.linear_model', 'LogisticRegression'),
        'RidgeClassifier': ('sklearn.linear_model', 'RidgeClassifier'),
        'SGDClassifier': ('sklearn.linear_model', 'SGDClassifier'),

        # SVM
        'LinearSVC': ('sklearn.svm', 'LinearSVC'),
        'SVC': ('sklearn.svm', 'SVC'),

        # Neural networks
        'MLPClassifier': ('sklearn.neural_network', 'MLPClassifier'),

        # Naive Bayes
        'GaussianNB': ('sklearn.naive_bayes', 'GaussianNB'),
        'ComplementNB': ('sklearn.naive_bayes', 'ComplementNB'),

        # Neighbors
        'KNeighborsClassifier': ('sklearn.neighbors', 'KNeighborsClassifier'),
        'RadiusNeighborsClassifier': ('sklearn.neighbors', 'RadiusNeighborsClassifier'),

        # Imbalanced-learn (if available)
        'BalancedRandomForestClassifier': ('imblearn.ensemble', 'BalancedRandomForestClassifier'),
        'EasyEnsembleClassifier': ('imblearn.ensemble', 'EasyEnsembleClassifier'),
        'RUSBoostClassifier': ('imblearn.ensemble', 'RUSBoostClassifier'),
        'BalancedBaggingClassifier': ('imblearn.ensemble', 'BalancedBaggingClassifier'),
    }

    @staticmethod
    def load_model(model_config: ModelConfig) -> Any:
        try:
            # 1. Cargar clase dinámicamente
            if model_config.class_name in ModelLoader.CLASS_MAP:
                module_name, class_name = ModelLoader.CLASS_MAP[model_config.class_name]
                if module_name.startswith('imblearn') and not IMBALANCED_LEARN_AVAILABLE:
                    return None
                module = __import__(module_name, fromlist=[class_name])
                model_class = getattr(module, class_name)
            else:
                module = __import__(model_config.module_path, fromlist=[model_config.class_name])
                model_class = getattr(module, model_config.class_name)

            fixed_params = model_config.params.get('fixed_params', {}).copy()

            for param in ['n_jobs', 'nthread', 'num_threads', 'thread_count']:
                if param in fixed_params: fixed_params[param] = 1

            model = model_class(**fixed_params)

            if model_config.name in ['RidgeClassifier', 'LinearSVC']:
                model = CalibratedClassifierCV(model, method='sigmoid', cv=3)

            return model

        except Exception as e:
            print(f"Error loading {model_config.name}: {e}")
            return None


# ============================================================================
# SIMPLIFIED BAYESIAN OPTIMIZER
# ============================================================================

class BayesianOptimizer:
    def __init__(self, model_config: ModelConfig, X: np.ndarray, y: np.ndarray,
                 n_iter: int = 50, cv_folds: int = 5, timeout: int = 600):
        self.model_config = model_config
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.timeout = timeout
        self.best_params = {}
        self.best_score = 0

    def optimize(self) -> Tuple[Dict, float]:
        """Exhaustive optimization with more parameters"""
        opt_config = self.model_config.params.get('optimization_settings', {})
        n_iter = opt_config.get('n_iter', self.n_iter)

        if self.n_iter > 40:
            n_iter = min(100, n_iter * 2)

        param_space = self.model_config.params.get('param_space', {})
        if not param_space:
            return {}, 0

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42, n_startup_trials=10),
            pruner=MedianPruner(n_warmup_steps=5, n_min_trials=3),
            study_name=f"{self.model_config.name}_optimization"
        )

        study.optimize(
            lambda trial: self._objective(trial),
            n_trials=n_iter,
            timeout=self.timeout,
            show_progress_bar=False,
            callbacks=[self._callback]
        )

        if study.best_trials:
            self.best_params = study.best_params
            self.best_score = study.best_value

        return self.best_params, self.best_score

    def _objective(self, trial):
        """Improved objective function"""
        params = {}

        for param_name, param_config in self.model_config.params.get('param_space', {}).items():
            try:
                if param_name == 'scale_pos_weight' and self.model_config.imbalance_strategy == 'scale_pos_weight':
                    continue

                param_type = param_config.get('type', 'float')

                if param_name == 'hidden_layer_sizes_idx':
                    hidden_options = {
                        0: (50,),
                        1: (100,),
                        2: (100, 50),
                        3: (200, 100),
                        4: (200, 100, 50)
                    }
                    idx = trial.suggest_categorical(param_name, param_config['values'])
                    params['hidden_layer_sizes'] = hidden_options.get(idx, (100,))
                    continue

                if param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        low=param_config['low'],
                        high=param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        low=param_config['low'],
                        high=param_config['high']
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        choices=param_config['values']
                    )
            except:
                continue

        if self.model_config.imbalance_strategy == 'scale_pos_weight':
            pos_weight = np.sum(self.y == 0) / np.sum(self.y == 1)
            if 'scale_pos_weight' not in params:
                params['scale_pos_weight'] = trial.suggest_float(
                    'scale_pos_weight',
                    low=max(1.0, pos_weight * 0.5),
                    high=pos_weight * 2.0
                )
        elif self.model_config.imbalance_strategy == 'class_weight' and 'class_weight' not in params:
            params['class_weight'] = 'balanced'

        model = ModelLoader.load_model(self.model_config)
        if model is None:
            return 0.0

        try:
            model.set_params(**params)
        except:
            valid_params = {}
            for k, v in params.items():
                try:
                    model.set_params(**{k: v})
                    valid_params[k] = v
                except:
                    pass
            params = valid_params
            model.set_params(**params)

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        try:
            scores = cross_val_score(
                model, self.X, self.y,
                cv=cv,
                scoring='average_precision',
                n_jobs=1,
                error_score='raise'
            )
            return np.mean(scores)
        except:
            return 0.0

    def _callback(self, study, trial):
        """Callback for progress display"""
        pass


# ============================================================================
# ROBUST MODEL EVALUATOR
# ============================================================================

class ModelEvaluator:

    def __init__(self, config: Dict, mode: ExecutionMode):
        self.config = config
        self.mode = mode

        if self.mode == ExecutionMode.QUICK:
            self.n_iter = 10
            self.cv_folds = 3
            self.timeout_per_model = 300
        elif self.mode == ExecutionMode.STANDARD:
            self.n_iter = 30
            self.cv_folds = 5
            self.timeout_per_model = 600
        else:
            self.n_iter = 50
            self.cv_folds = 5
            self.timeout_per_model = 900

        self.metrics = ['roc_auc', 'average_precision', 'f1', 'f2', 'precision', 'recall', 'accuracy']
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

    def evaluate_model(self, model_config: ModelConfig,
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> Optional[BenchmarkResults]:
        """Evaluate model with robust error handling"""

        try:
            start_time = time.time()

            n_iter = self.n_iter
            cv_folds = self.cv_folds

            best_params = {}
            best_score = 0.0

            if n_iter > 0:
                optimizer = BayesianOptimizer(
                    model_config, X_train, y_train,
                    n_iter=n_iter,
                    cv_folds=cv_folds,
                    timeout=self.timeout_per_model
                )
                best_params, best_score = optimizer.optimize()

            model = ModelLoader.load_model(model_config)
            if model is None:
                return None

            applied_params = {}
            if best_params:
                corrected_params = self._correct_model_params(model_config.name, best_params.copy())
                for param_name, param_value in corrected_params.items():
                    try:
                        model.set_params(**{param_name: param_value})
                        applied_params[param_name] = param_value
                    except Exception as e:
                        pass

            try:
                model.fit(X_train, y_train)
            except Exception as e:
                return None

            cv_scores = self._run_cross_validation(model, X_train, y_train, cv_folds)
            test_scores, predictions, probabilities = self._evaluate_test_set(model, X_test, y_test)

            exec_time = time.time() - start_time

            feature_importance = None
            try:
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    feature_importance = np.abs(model.coef_[0])
            except:
                pass

            results = BenchmarkResults(
                model_name=model_config.name,
                execution_time=exec_time,
                cv_scores=cv_scores,
                test_scores=test_scores,
                best_params=applied_params,
                predictions=predictions,
                probabilities=probabilities,
                feature_importance=feature_importance
            )

            return results

        except Exception as e:
            return None

    def _correct_model_params(self, model_name: str, params: Dict) -> Dict:
        """Correct model-specific parameters"""
        corrected = params.copy()

        if model_name == 'MLP' and 'class_weight' in corrected:
            corrected.pop('class_weight')

        if model_name == 'AdaBoost' and 'algorithm' in corrected:
            corrected.pop('algorithm')

        if model_name == 'ComplementNB':
            if 'alpha' in corrected and corrected['alpha'] <= 0:
                corrected['alpha'] = 1.0

        if model_name in ['XGBoost', 'LightGBM']:
            if 'scale_pos_weight' in corrected:
                if corrected['scale_pos_weight'] < 1:
                    corrected['scale_pos_weight'] = 4.0

        return corrected

    def _run_cross_validation(self, model, X: np.ndarray, y: np.ndarray,
                              cv_folds: int) -> Dict[str, List[float]]:
        """Run CV for multiple metrics"""
        cv_scores = {metric: [] for metric in self.metrics}

        if cv_folds < 2:
            return cv_scores

        try:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            fold_times = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                fold_start = time.time()

                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                try:
                    model_clone = model.__class__(**model.get_params())

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model_clone.fit(X_train_fold, y_train_fold)

                    y_pred = model_clone.predict(X_val_fold)
                    y_proba = None
                    if hasattr(model_clone, 'predict_proba'):
                        try:
                            y_proba = model_clone.predict_proba(X_val_fold)[:, 1]
                        except:
                            y_proba = None

                    if y_proba is not None:
                        try:
                            cv_scores['roc_auc'].append(roc_auc_score(y_val_fold, y_proba))
                        except:
                            cv_scores['roc_auc'].append(0.5)

                        try:
                            cv_scores['average_precision'].append(average_precision_score(y_val_fold, y_proba))
                        except:
                            cv_scores['average_precision'].append(0.0)
                    else:
                        cv_scores['roc_auc'].append(0.5)
                        cv_scores['average_precision'].append(0.0)

                    try:
                        cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred))
                        cv_scores['f1'].append(f1_score(y_val_fold, y_pred, zero_division=0))
                        cv_scores['f2'].append(fbeta_score(y_val_fold, y_pred, beta=2, zero_division=0))
                        cv_scores['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
                        cv_scores['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
                    except:
                        for metric in ['accuracy', 'f1', 'f2', 'precision', 'recall']:
                            cv_scores[metric].append(0.0)

                    fold_time = time.time() - fold_start
                    fold_times.append(fold_time)

                except Exception as e:
                    for metric in self.metrics:
                        cv_scores[metric].append(0.0 if 'auc' not in metric else 0.5)

        except Exception as e:
            for metric in self.metrics:
                cv_scores[metric] = [0.0] * cv_folds if 'auc' not in metric else [0.5] * cv_folds

        return cv_scores

    def _evaluate_test_set(self, model, X_test: np.ndarray,
                           y_test: np.ndarray) -> Tuple[Dict, np.ndarray, Optional[np.ndarray]]:
        """Evaluate on test set"""
        test_scores = {}
        predictions = None
        probabilities = None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predictions = model.predict(X_test)

                if hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(X_test)[:, 1]
                    except:
                        probabilities = None

            try:
                test_scores['accuracy'] = accuracy_score(y_test, predictions)
                test_scores['f1'] = f1_score(y_test, predictions, zero_division=0)
                test_scores['f2'] = fbeta_score(y_test, predictions, beta=2, zero_division=0)
                test_scores['precision'] = precision_score(y_test, predictions, zero_division=0)
                test_scores['recall'] = recall_score(y_test, predictions, zero_division=0)
            except:
                for metric in ['accuracy', 'f1', 'f2', 'precision', 'recall']:
                    test_scores[metric] = 0.0

            if probabilities is not None:
                try:
                    test_scores['roc_auc'] = roc_auc_score(y_test, probabilities)
                except:
                    test_scores['roc_auc'] = 0.5

                try:
                    test_scores['average_precision'] = average_precision_score(y_test, probabilities)
                except:
                    test_scores['average_precision'] = 0.0
            else:
                test_scores['roc_auc'] = 0.5
                test_scores['average_precision'] = 0.0

        except Exception as e:
            for metric in self.metrics:
                test_scores[metric] = 0.0 if 'auc' not in metric else 0.5
            predictions = np.zeros(len(y_test))
            probabilities = None

        return test_scores, predictions, probabilities


# ============================================================================
# RESULTS VISUALIZER (PAPER-READY VERSION)
# ============================================================================

class ResultsVisualizer:

    def __init__(self):
        self.figures_dir = Path("results/figures/benchmarking")
        self.reports_dir = Path("results/reports/benchmarking")
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Configuración de estilo global para figuras académicas
        plt.rcParams.update({
            'font.family': 'serif',  # Tipografía con serifa (estilo LaTeX/Paper)
            'font.serif': ['DejaVu Serif', 'Times New Roman'],
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--'
        })
        # Paleta de colores profesional (Colorblind friendly)
        self.colors = sns.color_palette("colorblind")

    def plot_performance_comparison(self, results: Dict[str, BenchmarkResults]):
        """
        Genera una figura compuesta profesional comparando las métricas principales.
        Omite tiempos de ejecución.
        """
        if not results:
            return

        # Preparar datos
        data = []
        for model, res in results.items():
            # Usamos las medias de CV si están disponibles para mayor robustez,
            # o los scores de test. Aquí uso test scores para consistencia con tu código anterior.
            data.append({
                'Model': model,
                'AUC-ROC': res.test_scores.get('roc_auc', 0),
                'AUC-PR': res.test_scores.get('average_precision', 0),
                'F1-Score': res.test_scores.get('f1', 0),
                'F2-Score': res.test_scores.get('f2', 0)
            })

        df = pd.DataFrame(data)
        # Ordenar por AUC-PR para que el mejor modelo quede arriba
        df = df.sort_values('AUC-PR', ascending=True)

        # Configurar la figura (2 columnas: Discriminación vs Métricas de Umbral)
        fig, axes = plt.subplots(1, 2, figsize=(14, len(df) * 0.5 + 2))
        fig.suptitle('Comparative Model Performance', fontweight='bold', y=0.98)

        # --- Subplot 1: Métricas de Área (AUC) ---
        # Transformar datos para Seaborn (Long format)
        df_auc = df.melt(id_vars='Model', value_vars=['AUC-ROC', 'AUC-PR'],
                         var_name='Metric', value_name='Score')

        sns.barplot(data=df_auc, x='Score', y='Model', hue='Metric',
                    ax=axes[0], palette=[self.colors[0], self.colors[1]],
                    edgecolor='black', linewidth=0.5)

        axes[0].set_title('Discrimination Metrics (AUC)', fontweight='bold')
        axes[0].set_xlabel('Score')
        axes[0].set_ylabel('')  # Ocultar label Y redundante
        axes[0].set_xlim(0, 1.05)
        axes[0].legend(loc='lower right')

        # Añadir valores numéricos al final de las barras
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.3f', padding=3, fontsize=8)

        # --- Subplot 2: Métricas F (Balance Precisión/Recall) ---
        df_f = df.melt(id_vars='Model', value_vars=['F1-Score', 'F2-Score'],
                       var_name='Metric', value_name='Score')

        sns.barplot(data=df_f, x='Score', y='Model', hue='Metric',
                    ax=axes[1], palette=[self.colors[2], self.colors[3]],
                    edgecolor='black', linewidth=0.5)

        axes[1].set_title('Threshold Metrics (F-Scores)', fontweight='bold')
        axes[1].set_xlabel('Score')
        axes[1].set_ylabel('')
        axes[1].set_yticks([])  # Quitar nombres de modelos en el segundo gráfico para limpiar
        axes[1].set_xlim(0, 1.05)
        axes[1].legend(loc='lower right')

        for container in axes[1].containers:
            axes[1].bar_label(container, fmt='%.3f', padding=3, fontsize=8)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Generar Heatmap adicional (Muy útil para Papers)
        self._plot_metrics_heatmap(df)

    def _plot_metrics_heatmap(self, df: pd.DataFrame):
        """Genera un heatmap resumen de todas las métricas para el paper"""
        # Ordenar descendente para el heatmap
        df_heat = df.set_index('Model').sort_values('AUC-PR', ascending=False)

        plt.figure(figsize=(8, len(df) * 0.5 + 1))
        sns.heatmap(df_heat, annot=True, cmap='viridis', fmt='.3f',
                    linewidths=.5, cbar_kws={'label': 'Score'})
        plt.title('Performance Metrics Heatmap', fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrices(self, results: Dict[str, BenchmarkResults],
                                X_test: np.ndarray, y_test: np.ndarray):
        """
        Matrices de confusión estilizadas para publicación.
        Muestra los Top 3 modelos.
        """
        valid_results = [(name, res) for name, res in results.items()
                         if res.predictions is not None]

        if not valid_results:
            return

        # Ordenar por AUC-PR
        valid_results.sort(key=lambda x: x[1].test_scores.get('average_precision', 0),
                           reverse=True)
        top_models = valid_results[:3]

        if not top_models:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        # Título general eliminado para que sea más fácil de insertar en documento LaTeX/Word

        for idx, (model_name, result) in enumerate(top_models):
            ax = axes[idx]

            try:
                cm = confusion_matrix(y_test, result.predictions)
                # Normalizar por filas (Recall por clase) para que sea más informativo en desbalanceo
                # O usar raw counts. Para paper de desbalanceo, raw counts + annotations suele ser mejor.

                sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', ax=ax,
                            cbar=False, square=True, annot_kws={"size": 14, "weight": "bold"},
                            linewidths=1, linecolor='black')

                ax.set_xlabel('Predicted Label', fontweight='bold')
                if idx == 0:
                    ax.set_ylabel('True Label', fontweight='bold')
                else:
                    ax.set_ylabel('')

                auc_pr = result.test_scores.get('average_precision', 0)
                # Título limpio
                ax.set_title(f'{model_name}\n(AUC-PR: {auc_pr:.3f})', fontsize=12)

                # Labels de clases (asumiendo binario 0/1)
                ax.set_xticklabels(['Neg', 'Pos'])
                ax.set_yticklabels(['Neg', 'Pos'], rotation=0)

            except Exception as e:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'top_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_results_table(self, results: Dict[str, BenchmarkResults]) -> pd.DataFrame:
        """Guarda tabla CSV limpia sin tiempos"""
        rows = []
        for model_name, result in results.items():
            # Base metrics
            row = {
                'Model': model_name,
                'AUC_ROC': result.test_scores.get('roc_auc', 0),
                'AUC_PR': result.test_scores.get('average_precision', 0),
                'F1': result.test_scores.get('f1', 0),
                'F2': result.test_scores.get('f2', 0),
                'Precision': result.test_scores.get('precision', 0),
                'Recall': result.test_scores.get('recall', 0),
                'Balanced_Acc': result.test_scores.get('balanced_accuracy', 0)
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        # Ordenar y redondear
        df = df.sort_values('AUC_PR', ascending=False)
        df = df.round(4)

        csv_path = self.reports_dir / 'benchmark_metrics.csv'
        df.to_csv(csv_path, index=False)

        # Generar versión LaTeX para copiar y pegar en el paper
        latex_path = self.reports_dir / 'benchmark_table_latex.txt'
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, caption="Model Performance Benchmarking", label="tab:benchmarking"))

        return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class BenchmarkingPipeline:

    def __init__(self, mode: ExecutionMode = ExecutionMode.STANDARD):
        self.mode = mode
        self.results = {}

        try:
            self.models_config, self.hyperparams_config, self.parallel_config = ConfigLoader.load_configs()
        except Exception as e:
            print(f"Error loading configurations: {e}")
            sys.exit(1)

        self.model_configs = ConfigLoader.create_model_configs(
            self.models_config, self.hyperparams_config
        )
        self.model_configs = self._filter_models_by_mode(self.model_configs)

    def _filter_models_by_mode(self, model_configs: List[ModelConfig]) -> List[ModelConfig]:
        """Filter models based on execution mode"""
        slow_models = ['SVC', 'MLP', 'GradientBoosting']

        if self.mode == ExecutionMode.QUICK:
            filtered = [m for m in model_configs if m.name not in slow_models]
            return filtered[:10]
        elif self.mode == ExecutionMode.STANDARD:
            filtered = [m for m in model_configs if m.name != 'SVC']
            return filtered
        else:
            return model_configs

    def run(self):
        """Execute complete pipeline"""
        total_start = time.time()

        try:
            data_handler = DataHandler(self.models_config)
            X, y, feature_names = data_handler.load_data()
            X_train, X_test, y_train, y_test = data_handler.prepare_splits(X, y)

            progress = ProgressTracker(len(self.model_configs))

            n_jobs = min(
                self.parallel_config['hardware'].get('max_models_simultaneous', 4),
                len(self.model_configs),
                mp.cpu_count()
            )

            evaluator = ModelEvaluator(self.models_config, self.mode)

            def evaluate_single_wrapper(model_config):
                try:
                    return evaluator.evaluate_model(
                        model_config, X_train, y_train, X_test, y_test
                    )
                except Exception as e:
                    return None

            with parallel_backend('loky', n_jobs=n_jobs):
                parallel_results = Parallel(
                    verbose=50,
                    timeout=1800,
                    prefer='processes',
                    batch_size=1
                )(
                    delayed(evaluate_single_wrapper)(model_config)
                    for model_config in self.model_configs
                )

            successful = 0
            for i, result in enumerate(parallel_results):
                if result is not None:
                    self.results[result.model_name] = result
                    successful += 1
                    progress.completed_models = successful
                    progress.model_times[result.model_name] = result.execution_time

            if successful == 0:
                print("No models could be evaluated. Check errors.")
                return

            visualizer = ResultsVisualizer()
            visualizer.plot_performance_comparison(self.results)
            visualizer.plot_confusion_matrices(self.results, X_test, y_test)
            df_results = visualizer.save_results_table(self.results)
            self._print_final_summary(total_start, df_results)

        except KeyboardInterrupt:
            if self.results:
                visualizer = ResultsVisualizer()
                visualizer.save_results_table(self.results)

        except Exception as e:
            print(f"Pipeline error: {e}")

    def _print_final_summary(self, total_start: float, df_results: pd.DataFrame):
        """Print final summary"""
        total_time = time.time() - total_start

        if not df_results.empty:
            best_model = df_results.iloc[0]['Model']
            best_auc_pr = df_results.iloc[0]['AUC_PR']


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='ML model benchmarking for 80/20 imbalanced datasets'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['quick', 'standard', 'full'],
        default='standard',
        help='Execution mode (default: standard)'
    )

    args = parser.parse_args()

    mode_map = {
        'quick': ExecutionMode.QUICK,
        'standard': ExecutionMode.STANDARD,
        'full': ExecutionMode.FULL
    }

    mode = mode_map[args.mode]
    pipeline = BenchmarkingPipeline(mode=mode)
    pipeline.run()


if __name__ == "__main__":
    main()