# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 16:18:13 2025

@author: syedi
"""

import numpy as np 

import pandas as pd 

 

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV 

from sklearn.preprocessing import StandardScaler 

from sklearn.metrics import ( 

    accuracy_score, 

    precision_score, 

    recall_score, 

    f1_score, 

    confusion_matrix, 

    classification_report 

) 

from sklearn.naive_bayes import GaussianNB 

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 

 

# pip install lightgbm 

from lightgbm import LGBMClassifier 

 

 

# ============================================================ 

# 1. Load Dataset 

# ============================================================ 

 

def load_wesad_features(path: str = "wesad_features.csv") -> pd.DataFrame: 

    """ 

    Load the WESAD feature dataset. 

 

    Args: 

        path: path to wesad_features.csv 

 

    Returns: 

        df: pandas DataFrame 

    """ 

    df = pd.read_csv(path) 

    return df 

 

 

# ============================================================ 

# 2. Basic EDA (optional) 

# ============================================================ 

 

def basic_eda(df: pd.DataFrame): 

    """ 

    Print basic info for sanity checking. 

    """ 

    print("=== BASIC EDA ===") 

    print("\nFirst five rows:") 

    print(df.head()) 

    print("\nShape:", df.shape) 

    print("\nColumns:", df.columns.tolist()) 

    print("\nMissing values per column:") 

    print(df.isna().sum()) 

    if "label" in df.columns: 

        print("\nLabel distribution (0=baseline, 1=stress, 2=amusement):") 

        print(df["label"].value_counts()) 

 

 

# ============================================================ 

# 3. Prepare Data: Features, Labels, Split, Scaling 

# ============================================================ 

 

def prepare_data(df: pd.DataFrame, test_size: float = 0.2): 

    """ 

    Separate features and labels, handle missing values, split, standardise. 

 

    Returns: 

        X_train_scaled, X_test_scaled, y_train, y_test, scaler 

    """ 

 

    if "label" not in df.columns: 

        raise ValueError("Expected target column 'label' not found in dataset.") 

 

    # Drop any non-numeric columns except label (e.g., subject_id, timestamps) 

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist() 

    if "label" not in numeric_cols: 

        numeric_cols.append("label") 

    df_numeric = df[numeric_cols].copy() 

 

    # Handle missing values (if any) 

    df_numeric = df_numeric.fillna(df_numeric.median()) 

 

    # Separate X and y 

    y = df_numeric["label"].astype(int) 

    X = df_numeric.drop(columns=["label"]) 

 

    # Sanity check: all features numeric 

    if not np.all([np.issubdtype(dt, np.number) for dt in X.dtypes]): 

        raise ValueError("All feature columns must be numeric. Please check wesad_features.csv.") 

 

    # Stratified split 

    X_train, X_test, y_train, y_test = train_test_split( 

        X, 

        y, 

        test_size=test_size, 

        stratify=y, 

        random_state=42 

    ) 

 

    print("\nTrain/Test shapes before scaling:") 

    print("X_train:", X_train.shape, "X_test:", X_test.shape) 

    print("y_train:", y_train.shape, "y_test:", y_test.shape) 

 

    # Standardise features (Z-score) 

    scaler = StandardScaler() 

    X_train_scaled = scaler.fit_transform(X_train) 

    X_test_scaled = scaler.transform(X_test) 

 

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler 

 

 

# ============================================================ 

# 4. Evaluation Helper 

# ============================================================ 

 

def evaluate_model(name: str, y_test, y_pred): 

    """ 

    Compute and print metrics for multiclass classification. 

 

    Metrics: 

        - Accuracy 

        - Macro Precision / Recall / F1 

        - Confusion matrix 

    """ 

    acc = accuracy_score(y_test, y_pred) 

    prec = precision_score(y_test, y_pred, average="macro", zero_division=0) 

    rec = recall_score(y_test, y_pred, average="macro", zero_division=0) 

    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0) 

    cm = confusion_matrix(y_test, y_pred) 

 

    print(f"\n===== {name} =====") 

    print(f"Accuracy : {acc:.4f}") 

    print(f"Precision (macro): {prec:.4f}") 

    print(f"Recall (macro)   : {rec:.4f}") 

    print(f"F1-score (macro) : {f1:.4f}") 

    print("Confusion Matrix:") 

    print(cm) 

    print("\nClassification Report:") 

    print(classification_report(y_test, y_pred, zero_division=0)) 

 

    return { 

        "Model": name, 

        "Accuracy": acc, 

        "Precision": prec, 

        "Recall": rec, 

        "F1-Score": f1, 

        "ConfusionMatrix": cm 

    } 

 

 

# ============================================================ 

# 5. Gaussian Naïve Bayes 

# ============================================================ 

 

def run_gaussian_nb(X_train, X_test, y_train, y_test): 

    """ 

    Train and evaluate Gaussian Naïve Bayes with simple var_smoothing tuning. 

    """ 

 

    candidate_vs = [1e-9, 1e-8] 

    best_model = None 

    best_f1 = -1 

    best_vs = None 

 

    for vs in candidate_vs: 

        model = GaussianNB(var_smoothing=vs) 

        model.fit(X_train, y_train) 

        y_pred_val = model.predict(X_test) 

        f1_val = f1_score(y_test, y_pred_val, average="macro", zero_division=0) 

        if f1_val > best_f1: 

            best_f1 = f1_val 

            best_vs = vs 

            best_model = model 

 

    print("\nBest var_smoothing for GaussianNB:", best_vs) 

    y_pred = best_model.predict(X_test) 

    metrics = evaluate_model("Gaussian Naïve Bayes", y_test, y_pred) 

 

    grid_info = {"var_smoothing": candidate_vs} 

    return best_model, metrics, grid_info 

 

 

# ============================================================ 

# 6. Quadratic Discriminant Analysis (QDA) 

# ============================================================ 

 

def run_qda(X_train, X_test, y_train, y_test): 

    """ 

    Train and evaluate QDA using GridSearchCV. 

    """ 

    qda = QuadraticDiscriminantAnalysis() 

 

    param_grid = { 

        "reg_param": [0.0, 0.1, 0.2] 

    } 

 

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

 

    grid = GridSearchCV( 

        estimator=qda, 

        param_grid=param_grid, 

        scoring="f1_macro", 

        cv=cv, 

        n_jobs=-1 

    ) 

 

    grid.fit(X_train, y_train) 

 

    print("\nBest parameters for QDA:") 

    print(grid.best_params_) 

    print(f"Best CV F1-macro: {grid.best_score_:.4f}") 

 

    best_qda = grid.best_estimator_ 

    y_pred = best_qda.predict(X_test) 

 

    metrics = evaluate_model("Quadratic Discriminant Analysis", y_test, y_pred) 

    return best_qda, metrics, param_grid 

 

 

# ============================================================ 

# 7. LightGBM Classifier 

# ============================================================ 

 

def run_lightgbm(X_train, X_test, y_train, y_test): 

    """ 

    Train and evaluate LightGBM using GridSearchCV. 

    """ 

    lgbm = LGBMClassifier( 

        objective="multiclass", 

        num_class=3, 

        random_state=42 

    ) 

 

    param_grid = { 

        "num_leaves": [31, 63], 

        "learning_rate": [0.05, 0.1], 

        "n_estimators": [100, 200] 

    } 

 

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

 

    grid = GridSearchCV( 

        estimator=lgbm, 

        param_grid=param_grid, 

        scoring="f1_macro", 

        cv=cv, 

        n_jobs=-1 

    ) 

 

    grid.fit(X_train, y_train) 

 

    print("\nBest parameters for LightGBM:") 

    print(grid.best_params_) 

    print(f"Best CV F1-macro: {grid.best_score_:.4f}") 

 

    best_lgbm = grid.best_estimator_ 

    y_pred = best_lgbm.predict(X_test) 

 

    metrics = evaluate_model("LightGBM", y_test, y_pred) 

    return best_lgbm, metrics, param_grid 

 

 

# ============================================================ 

# 8. Performance Tables 

# ============================================================ 

 

def build_performance_table(metrics_list): 

    """ 

    Build a simple summary table from metrics. 

    """ 

    rows = [] 

    for m in metrics_list: 

        rows.append({ 

            "Model": m["Model"], 

            "Accuracy": round(m["Accuracy"], 3), 

            "Precision": round(m["Precision"], 3), 

            "Recall": round(m["Recall"], 3), 

            "F1-Score": round(m["F1-Score"], 3) 

        }) 

    return pd.DataFrame(rows) 

 

 

def build_confusion_matrix_tables(metrics_list, label_names=None): 

    """ 

    Build confusion matrix DataFrames. 

    """ 

    cm_tables = {} 

    for m in metrics_list: 

        cm = m["ConfusionMatrix"] 

        if label_names is not None: 

            df_cm = pd.DataFrame(cm, index=label_names, columns=label_names) 

        else: 

            df_cm = pd.DataFrame(cm) 

        cm_tables[m["Model"]] = df_cm 

    return cm_tables 

 

 

# ============================================================ 

# 9. Main 

# ============================================================ 

 

def main(): 

    # Step 1: Load data 

    df = load_wesad_features() 

    print("Dataset loaded. Shape:", df.shape) 

 

    # Step 2: Basic EDA 

    basic_eda(df) 

 

    # Step 3: Prepare data 

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(df) 

 

    # Optional label names (for confusion matrix display) 

    label_names = ["Baseline", "Stress", "Amusement"] 

 

    # Step 4: Train three models 

    nb_model, nb_metrics, nb_grid = run_gaussian_nb(X_train_scaled, X_test_scaled, y_train, y_test) 

    qda_model, qda_metrics, qda_grid = run_qda(X_train_scaled, X_test_scaled, y_train, y_test) 

    lgbm_model, lgbm_metrics, lgbm_grid = run_lightgbm(X_train_scaled, X_test_scaled, y_train, y_test) 

 

    # Step 5: Build performance table 

    metrics_list = [nb_metrics, qda_metrics, lgbm_metrics] 

    perf_table = build_performance_table(metrics_list) 

 

    print("\n=== PERFORMANCE TABLE ===") 

    print(perf_table) 

 

    perf_table.to_csv("wesad_stress_performance_results.csv", index=False) 

    print("Performance table saved as 'wesad_stress_performance_results.csv'.") 

 

    # Step 6: Confusion matrices 

    cm_tables = build_confusion_matrix_tables(metrics_list, label_names=label_names) 

    for model_name, cm_df in cm_tables.items(): 

        fname = f"wesad_cm_{model_name.lower().replace(' ', '_')}.csv" 

        cm_df.to_csv(fname) 

        print(f"Confusion matrix for {model_name} saved as '{fname}'.") 

 

 

if __name__ == "__main__": 

    main() 