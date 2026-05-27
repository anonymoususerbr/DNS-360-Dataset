import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.svm import SVC


# ===========================
# CONFIG (edite SOMENTE aqui)
# ===========================


TRAIN_DIR = r"C:\ML\MyDataset_SPLIT\Train"  # pasta com CSVs (recursivo)
TEST_DIR  = r"C:\ML\MyDataset_SPLIT\Test"        # pasta com CSVs (recursivo)
OUT_DIR   = r"C:\ML\Results\Training and Testing (My_SPLIT 80-20) - SVM"  # saída

TARGET_COLUMN = "label"

# SVM
SVM_KERNEL = "rbf"
SVM_C = 1.0
SVM_GAMMA = "scale"

# salvar predição com label original
SAVE_PREDICTION_LABEL = True

CSV_READ_KWARGS = dict(low_memory=False)
# ===========================


def log(msg: str):
    print(msg, flush=True)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def load_dataset(folder_path: str):
    csv_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    csv_files.sort()
    return csv_files


def read_concat_csv(files):
    return pd.concat((pd.read_csv(f, **CSV_READ_KWARGS) for f in files), ignore_index=True)


def fit_target_encoder(train_data: pd.DataFrame, target_column: str) -> LabelEncoder:
    if target_column not in train_data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in TRAIN data. "
            f"Columns sample: {list(train_data.columns)[:30]}"
        )
    y_train_raw = train_data[target_column].fillna("NA").astype(str)
    le = LabelEncoder()
    le.fit(y_train_raw)
    return le


def transform_target(data: pd.DataFrame, target_column: str, le: LabelEncoder) -> np.ndarray:
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")
    y_raw = data[target_column].fillna("NA").astype(str)

    unknown = set(pd.unique(y_raw)) - set(le.classes_)
    if unknown:
        raise ValueError(
            f"Unknown labels found in '{target_column}' (test has labels not seen in training). "
            f"Examples: {list(unknown)[:20]}"
        )
    return le.transform(y_raw).astype(int)


def build_preprocessor_from_train(X_train_raw: pd.DataFrame):
    """
    Define colunas numéricas e categóricas usando SOMENTE o treino.
    Para SVM, as variáveis numéricas são padronizadas.
    """
    num_cols = X_train_raw.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = X_train_raw.select_dtypes(include=["object", "string", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )

    return preprocessor, num_cols, cat_cols


def align_test_to_train_schema(X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame, out_dir: str):
    """
    Garante que o teste tenha todas as colunas do treino, na mesma ordem.
    Colunas faltantes no teste são criadas como NaN.
    Colunas extras do teste são removidas.
    Também salva diagnósticos de schema.
    """
    train_cols = list(X_train_raw.columns)
    test_cols = list(X_test_raw.columns)

    missing_in_test = sorted(set(train_cols) - set(test_cols))
    extra_in_test = sorted(set(test_cols) - set(train_cols))

    pd.DataFrame({"train_columns": train_cols}).to_csv(os.path.join(out_dir, "train_schema.csv"), index=False)
    pd.DataFrame({"test_columns": test_cols}).to_csv(os.path.join(out_dir, "test_schema.csv"), index=False)

    pd.DataFrame({"missing_in_test": missing_in_test}).to_csv(
        os.path.join(out_dir, "schema_missing_in_test.csv"), index=False
    )
    pd.DataFrame({"extra_in_test": extra_in_test}).to_csv(
        os.path.join(out_dir, "schema_extra_in_test.csv"), index=False
    )

    if missing_in_test:
        log(f"[WARN] Test is missing {len(missing_in_test)} train columns. Adding them as NaN.")
        log(f"[WARN] Missing examples: {missing_in_test[:25]}")
    if extra_in_test:
        log(f"[WARN] Test has {len(extra_in_test)} extra columns not in train. Dropping them.")
        log(f"[WARN] Extra examples: {extra_in_test[:25]}")

    X_test_aligned = X_test_raw.reindex(columns=train_cols, fill_value=np.nan)

    missing_pct = (X_test_aligned.isna().mean() * 100.0).sort_values(ascending=False)
    missing_pct.to_csv(
        os.path.join(out_dir, "test_missing_percent_after_align.csv"),
        header=["missing_percent"]
    )

    return X_test_aligned


def save_predictions(y_pred_enc: np.ndarray, le: LabelEncoder, out_dir: str):
    path = os.path.join(out_dir, "predictions.csv")
    out = {"PredictionEncoded": y_pred_enc.astype(int)}
    if SAVE_PREDICTION_LABEL:
        out["PredictionLabel"] = le.inverse_transform(y_pred_enc.astype(int))
    pd.DataFrame(out).to_csv(path, index=False)
    return path


def save_class_distributions(y_true, y_pred, le: LabelEncoder, out_dir: str):
    true_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()

    df = pd.DataFrame({
        "ClassEncoded": np.arange(len(le.classes_)),
        "ClassLabel": le.classes_,
        "y_true_count": [int(true_counts.get(i, 0)) for i in range(len(le.classes_))],
        "y_pred_count": [int(pred_counts.get(i, 0)) for i in range(len(le.classes_))],
    })
    path = os.path.join(out_dir, "class_distributions.csv")
    df.to_csv(path, index=False)
    return path


def save_confusion_matrices(y_true, y_pred, le: LabelEncoder, out_dir: str):
    labels = np.arange(len(le.classes_))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_path = os.path.join(out_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_path, index=True)

    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cmn_df = pd.DataFrame(cm_norm, index=le.classes_, columns=le.classes_)
    cmn_path = os.path.join(out_dir, "confusion_matrix_normalized.csv")
    cmn_df.to_csv(cmn_path, index=True)

    return cm_path, cmn_path


def save_metrics(y_true, y_pred, le: LabelEncoder, out_dir: str):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics_df = pd.DataFrame([{
        "Accuracy": acc,
        "BalancedAccuracy": bal_acc,
        "PrecisionMacro": prec_macro,
        "RecallMacro": rec_macro,
        "F1Macro": f1_macro,
        "PrecisionWeighted": prec_weighted,
        "RecallWeighted": rec_weighted,
        "F1Weighted": f1_weighted,
    }])

    metrics_path = os.path.join(out_dir, "report.csv")
    metrics_df.to_csv(metrics_path, index=False)

    report_dict = classification_report(
        y_true, y_pred,
        labels=np.arange(len(le.classes_)),
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    classrep_path = os.path.join(out_dir, "classification_report.csv")
    report_df.to_csv(classrep_path, index=True)

    return metrics_path, classrep_path


def run_model(train_files, test_files, output_path: str):
    safe_mkdir(output_path)

    log(f"Train CSV files: {len(train_files)}")
    log(f"Test  CSV files: {len(test_files)}")

    log("Loading training data...")
    train_data = read_concat_csv(train_files)
    log(f"Training rows: {len(train_data)}")

    log("Loading testing data...")
    test_data = read_concat_csv(test_files)
    log(f"Testing rows: {len(test_data)}")

    le = fit_target_encoder(train_data, TARGET_COLUMN)

    y_train = transform_target(train_data, TARGET_COLUMN, le)
    y_test = transform_target(test_data, TARGET_COLUMN, le)

    X_train_raw = train_data.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    X_test_raw = test_data.drop(columns=[TARGET_COLUMN], errors="ignore").copy()

    X_test_raw = align_test_to_train_schema(X_train_raw, X_test_raw, output_path)

    preprocessor, num_cols, cat_cols = build_preprocessor_from_train(X_train_raw)

    model = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        class_weight="balanced",
        random_state=50
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    log(
        f"Training SVM model "
        f"(kernel={SVM_KERNEL}, C={SVM_C}, gamma={SVM_GAMMA})..."
    )
    clf.fit(X_train_raw, y_train)
    log("SVM model trained.")

    log("Predicting...")
    y_pred = clf.predict(X_test_raw).astype(int)
    log(f"Predictions: {len(y_pred)}")

    pred_path = save_predictions(y_pred, le, output_path)
    dist_path = save_class_distributions(y_test, y_pred, le, output_path)

    log("Saving metrics...")
    metrics_path, classrep_path = save_metrics(y_test, y_pred, le, output_path)

    cm_path, cmn_path = save_confusion_matrices(y_test, y_pred, le, output_path)

    log("DONE")
    log("Saved:")
    log(f"- {metrics_path}")
    log(f"- {classrep_path}")
    log(f"- {cm_path}")
    log(f"- {cmn_path}")
    log(f"- {pred_path}")
    log(f"- {dist_path}")
    log("- train_schema.csv / test_schema.csv / schema_missing_in_test.csv / schema_extra_in_test.csv / test_missing_percent_after_align.csv")


def main():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"TRAIN_DIR não existe: {TRAIN_DIR}")
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"TEST_DIR não existe: {TEST_DIR}")

    train_files = load_dataset(TRAIN_DIR)
    test_files = load_dataset(TEST_DIR)

    if not train_files:
        raise RuntimeError(f"No CSV files found in train folder: {TRAIN_DIR}")
    if not test_files:
        raise RuntimeError(f"No CSV files found in test folder: {TEST_DIR}")

    run_model(train_files, test_files, OUT_DIR)


if __name__ == "__main__":
    main()