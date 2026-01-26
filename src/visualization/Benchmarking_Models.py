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
        """Load model robustly"""
        try:
            if model_config.class_name in ModelLoader.CLASS_MAP:
                module_name, class_name = ModelLoader.CLASS_MAP[model_config.class_name]

                if module_name.startswith('imblearn') and not IMBALANCED_LEARN_AVAILABLE:
                    return None

                module = __import__(module_name, fromlist=[class_name])
                model_class = getattr(module, class_name)
            else:
                module = __import__(model_config.module_path, fromlist=[model_config.class_name])
                model_class = getattr(module, model_config.class_name)

            fixed_params = model_config.params.get('fixed_params', {})
            fixed_params = fixed_params.copy()

            # Adjust parallelization parameters
            for param in ['n_jobs', 'nthread', 'num_threads', 'thread_count']:
                if param in fixed_params:
                    fixed_params[param] = 1

            if model_config.imbalance_strategy == 'class_weight' and 'class_weight' not in fixed_params:
                fixed_params['class_weight'] = 'balanced'

            return model_class(**fixed_params)

        except (ImportError, AttributeError) as e:
            return None
        except Exception as e:
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
# RESULTS VISUALIZER
# ============================================================================

class ResultsVisualizer:

    def __init__(self):
        # Definimos las dos rutas de salida según la nueva estructura
        self.figures_dir = Path("results/figures/benchmarking")
        self.reports_dir = Path("results/reports/benchmarking")

        # Creamos ambas carpetas
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def plot_performance_comparison(self, results: Dict[str, BenchmarkResults]):
        """Create main performance comparison plot"""

        if not results:
            return

        model_names = []
        auc_roc_scores = []
        auc_pr_scores = []
        f1_scores = []
        f2_scores = []
        exec_times = []

        for model_name, result in results.items():
            model_names.append(model_name)
            auc_roc_scores.append(result.test_scores.get('roc_auc', 0))
            auc_pr_scores.append(result.test_scores.get('average_precision', 0))
            f1_scores.append(result.test_scores.get('f1', 0))
            f2_scores.append(result.test_scores.get('f2', 0))
            exec_times.append(result.execution_time)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ML Models Benchmarking - Performance Comparison',
                     fontsize=16, fontweight='bold')

        ax = axes[0, 0]
        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax.bar(x - width / 2, auc_roc_scores, width, label='AUC-ROC',
                       alpha=0.8, color='skyblue', edgecolor='black')
        bars2 = ax.bar(x + width / 2, auc_pr_scores, width, label='AUC-PR',
                       alpha=0.8, color='lightcoral', edgecolor='black')

        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('AUC-ROC vs AUC-PR', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        ax = axes[0, 1]
        bars3 = ax.bar(x - width / 2, f1_scores, width, label='F1-Score',
                       alpha=0.8, color='lightgreen', edgecolor='black')
        bars4 = ax.bar(x + width / 2, f2_scores, width, label='F2-Score',
                       alpha=0.8, color='gold', edgecolor='black')

        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('F1-Score vs F2-Score', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        ax = axes[1, 0]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names)))
        bars_time = ax.bar(x, exec_times, width=0.8, color=colors, edgecolor='black')

        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontweight='bold')
        ax.set_title('Execution Time', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        ax = axes[1, 1]
        scatter = ax.scatter(exec_times, auc_pr_scores, s=100, alpha=0.7,
                             c=range(len(model_names)), cmap='viridis')

        for i, (model, time_val, auc_pr) in enumerate(zip(model_names, exec_times, auc_pr_scores)):
            ax.annotate(model, (time_val, auc_pr), xytext=(5, 5),
                        textcoords='offset points', fontsize=8)

        ax.set_xlabel('Execution Time (s)', fontweight='bold')
        ax.set_ylabel('AUC-PR', fontweight='bold')
        ax.set_title('Time vs Performance (AUC-PR)', fontweight='bold')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrices(self, results: Dict[str, BenchmarkResults],
                                X_test: np.ndarray, y_test: np.ndarray):
        """Create confusion matrices for top models"""
        valid_results = [(name, res) for name, res in results.items()
                         if res.predictions is not None]

        if not valid_results:
            return

        valid_results.sort(key=lambda x: x[1].test_scores.get('average_precision', 0),
                           reverse=True)
        top_models = valid_results[:3]

        if not top_models:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Confusion Matrices - Top 3 Models',
                     fontsize=16, fontweight='bold')

        for idx, (model_name, result) in enumerate(top_models):
            ax = axes[idx]

            try:
                cm = confusion_matrix(y_test, result.predictions)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            cbar_kws={'label': 'Count'})

                ax.set_xlabel('Prediction', fontweight='bold')
                ax.set_ylabel('Actual', fontweight='bold')

                auc_pr = result.test_scores.get('average_precision', 0)
                ax.set_title(f'{model_name}\nAUC-PR: {auc_pr:.3f}')

            except Exception as e:
                ax.text(0.5, 0.5, f'Error\n{e}', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(model_name)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_results_table(self, results: Dict[str, BenchmarkResults]) -> pd.DataFrame:
        """Save results to CSV and LaTeX"""
        rows = []
        for model_name, result in results.items():
            row = {
                'Model': model_name,
                'Time_s': round(result.execution_time, 2),
                'AUC_ROC': round(result.test_scores.get('roc_auc', 0), 4),
                'AUC_PR': round(result.test_scores.get('average_precision', 0), 4),
                'F1': round(result.test_scores.get('f1', 0), 4),
                'F2': round(result.test_scores.get('f2', 0), 4),
                'Precision': round(result.test_scores.get('precision', 0), 4),
                'Recall': round(result.test_scores.get('recall', 0), 4),
                'Accuracy': round(result.test_scores.get('accuracy', 0), 4),
            }

            cv_means = result.get_mean_cv_scores()
            for metric, mean_score in cv_means.items():
                row[f'CV_{metric}_mean'] = round(mean_score, 4)

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values('AUC_PR', ascending=False)

        csv_path = self.reports_dir / 'benchmark_results.csv'
        df.to_csv(csv_path, index=False)

        txt_path = self.reports_dir / 'benchmark_summary.txt'
        with open(txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BENCHMARKING RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write("Top 5 Models by AUC-PR:\n")
            f.write("-" * 40 + "\n")
            for i, row in df.head().iterrows():
                f.write(f"{i + 1}. {row['Model']}:\n")
                f.write(f"   AUC-PR: {row['AUC_PR']:.4f} | AUC-ROC: {row['AUC_ROC']:.4f}\n")
                f.write(f"   F2-Score: {row['F2']:.4f} | Time: {row['Time_s']:.1f}s\n")
                f.write("\n")

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