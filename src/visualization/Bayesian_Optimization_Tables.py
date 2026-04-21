import yaml
import pandas as pd
from openpyxl.styles import Font, Alignment


def load_yaml(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def generate_supplementary_tables():
    # 1. Cargar los archivos YAML (Asegúrate de que las rutas sean correctas)
    models_config = load_yaml('src/visualization/Benchmarking_Configs/models_to_benchmark.yaml')
    hyper_config = load_yaml('src/visualization/Benchmarking_Configs/hyperparameter_grids.yaml')  # Ajusta la ruta si es necesario
    parallel_config = load_yaml('src/visualization/Benchmarking_Configs/parallelization_config.yaml')

    # ==========================================
    # TABLA S1: Modelos y Estrategias de Desbalance
    # ==========================================
    category_map = {
        "XGBoost": "Tree-Based (State-of-the-Art)", "LightGBM": "Tree-Based (State-of-the-Art)",
        "CatBoost": "Tree-Based (State-of-the-Art)",
        "RandomForest": "Random Forest Variants", "ExtraTrees": "Random Forest Variants",
        "BalancedRandomForest": "Random Forest Variants",
        "GradientBoosting": "Gradient Boosting Variants", "HistGradientBoosting": "Gradient Boosting Variants",
        "EasyEnsemble": "Imbalanced-Learn Ensembles", "RUSBoost": "Imbalanced-Learn Ensembles",
        "BalancedBagging": "Imbalanced-Learn Ensembles",
        "LogisticRegression": "Linear Models", "RidgeClassifier": "Linear Models", "SGDClassifier": "Linear Models",
        "LinearSVC": "Support Vector Machines", "SVC": "Support Vector Machines",
        "MLP": "Neural Networks",
        "GaussianNB": "Bayesian & Neighbors", "ComplementNB": "Bayesian & Neighbors",
        "KNeighbors": "Bayesian & Neighbors", "RadiusNeighbors": "Bayesian & Neighbors",
        "Voting_Ensemble": "Meta-Ensembles", "Stacking_Ensemble": "Meta-Ensembles", "AdaBoost": "Meta-Ensembles",
        "Bagging": "Meta-Ensembles"
    }

    t1_data = []
    for model in models_config.get('models', []):
        t1_data.append({
            "Category": category_map.get(model['name'], "Other"),
            "Model Name": model['name'],
            "Class Imbalance Strategy": model.get('imbalance_strategy', 'None / Custom Weight'),
            "Priority": "High" if model.get('priority') == 1 else ("Medium" if model.get('priority') == 2 else "Low")
        })
    df_s1 = pd.DataFrame(t1_data)

    # ==========================================
    # TABLA S2: Hiperparámetros y Espacio de Búsqueda
    # ==========================================
    t2_data = []
    # Filtramos para mostrar solo los modelos más relevantes y no saturar la tabla
    target_models = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'MLP']

    for model_name, config in hyper_config.items():
        if model_name in target_models and 'param_space' in config:
            for param, bounds in config['param_space'].items():
                ptype = bounds.get('type')
                if ptype in ['int', 'float']:
                    val_str = f"[{bounds.get('low')}, {bounds.get('high')}]"
                    scale = "Logarithmic" if bounds.get('log') else "Linear"
                elif ptype == 'categorical':
                    val_str = ", ".join(map(str, bounds.get('values', [])))
                    scale = "Categorical"
                else:
                    continue

                # Excepción para hacer legible la arquitectura del MLP
                if model_name == 'MLP' and param == 'hidden_layer_sizes_idx':
                    param = 'hidden_layer_sizes'
                    val_str = "(50,), (100,), (100,50), (200,100), (200,100,50)"

                t2_data.append({
                    "Model": model_name,
                    "Hyperparameter": param,
                    "Search Space / Values": val_str,
                    "Sampling Scale": scale
                })
    df_s2 = pd.DataFrame(t2_data)

    # ==========================================
    # TABLA S3: Configuración de Ejecución
    # ==========================================
    t3_data = [
        {"Parameter Category": "Optimization Strategy",
         "Configuration Detail": "Bayesian Optimization (Tree-structured Parzen Estimator)"},
        {"Parameter Category": "Primary Optimization Metric",
         "Configuration Detail": str(models_config['evaluation']['metrics']['primary'])},
        {"Parameter Category": "Secondary Evaluation Metrics",
         "Configuration Detail": ", ".join(models_config['evaluation']['metrics']['secondary'])},
        {"Parameter Category": "Cross-Validation Scheme",
         "Configuration Detail": f"Stratified {models_config['evaluation']['cv_folds']}-Fold Cross-Validation"},
        {"Parameter Category": "Hyperparameter Iterations",
         "Configuration Detail": f"{models_config['evaluation']['n_iter_optimization']} iterations per model"},
        {"Parameter Category": "Timeout per Model",
         "Configuration Detail": f"{parallel_config['resource_limits']['timeout_per_model_minutes']} minutes"},
        {"Parameter Category": "Parallelization Backend",
         "Configuration Detail": f"{parallel_config['parallel_strategy']['model_level']['backend']} (Python {parallel_config['hardware']['python_version']} optimized)"},
        {"Parameter Category": "Hardware Environment",
         "Configuration Detail": f"Max {parallel_config['resource_limits']['max_models_running']} models running simultaneously (Batch-Level Parallelism)"}
    ]
    df_s3 = pd.DataFrame(t3_data)

    # ==========================================
    # GUARDAR Y FORMATEAR EXCEL
    # ==========================================
    output_filename = "Supplementary_Tables_Benchmarking.xlsx"
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        df_s1.to_excel(writer, sheet_name='Table_S1_Models', index=False)
        df_s2.to_excel(writer, sheet_name='Table_S2_Hyperparameters', index=False)
        df_s3.to_excel(writer, sheet_name='Table_S3_Setup', index=False)

        # Dar formato de publicación (cabeceras en negrita y ajustar columnas)
        for sheetname in writer.sheets:
            worksheet = writer.sheets[sheetname]
            # Negrita en cabeceras
            for cell in worksheet[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal='center')

            # Auto-ajustar el ancho de las columnas
            for col in worksheet.columns:
                max_length = 0
                column = col[0].column_letter  # Obtener la letra de la columna
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column].width = adjusted_width

    print(f"¡Éxito! El archivo '{output_filename}' se ha generado correctamente con formato de publicación.")


if __name__ == "__main__":
    generate_supplementary_tables()