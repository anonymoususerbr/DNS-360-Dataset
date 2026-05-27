import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sklearn.metrics

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(
        "O pacote xgboost não está instalado. Instale com: pip install xgboost"
    ) from e


# ===========================
# CONFIG
# ===========================
DATASET_PATH = r"C:\Users\cborges\OneDrive - Fortinet\Documents\Studies\Doctorade UFU\Research\Development\Dataset\My Dataset\CSVs\DoH\Versao_Final\dataset_DoH_normalized.csv"

OUT_DIR = r"C:\Users\cborges\OneDrive - Fortinet\Documents\Studies\Doctorade UFU\Research\Development\Dataset\My Dataset\CSVs\Results\XGBoost_DoH - Train (My) and Test (My)"

TARGET_COLUMN = "label"
TEST_SIZE = 0.20
RANDOM_STATE = 42

XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8

SAVE_PREDICTIONS = True
CSV_READ_KWARGS = dict(low_memory=False)
# ===========================


def log(msg: str):
    print(msg, flush=True)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def infer_feature_types(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object", "string", "bool"]
    ).columns.tolist()

    return numeric_cols, categorical_cols


def build_preprocessor(X_train: pd.DataFrame):
    numeric_cols, categorical_cols = infer_feature_types(X_train)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float64
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    return preprocessor, numeric_cols, categorical_cols


def save_split_metadata(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_raw: pd.Series,
    y_test_raw: pd.Series,
    out_dir: str,
    scale_pos_weight: float
):
    summary = {
        "dataset_rows_total": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_ratio": float(len(X_train) / len(df)),
        "test_ratio": float(len(X_test) / len(df)),
        "target_column": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "class_distribution_full": df[TARGET_COLUMN].value_counts(dropna=False).to_dict(),
        "class_distribution_train": y_train_raw.value_counts(dropna=False).to_dict(),
        "class_distribution_test": y_test_raw.value_counts(dropna=False).to_dict(),
        "model": "XGBoost",
        "xgb_n_estimators": XGB_N_ESTIMATORS,
        "xgb_max_depth": XGB_MAX_DEPTH,
        "xgb_learning_rate": XGB_LEARNING_RATE,
        "xgb_subsample": XGB_SUBSAMPLE,
        "xgb_colsample_bytree": XGB_COLSAMPLE_BYTREE,
        "scale_pos_weight": scale_pos_weight,
    }

    with open(os.path.join(out_dir, "split_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def save_feature_inventory(X: pd.DataFrame, numeric_cols, categorical_cols, out_dir: str):
    rows = []

    for col in numeric_cols:
        rows.append({"feature": col, "type": "numeric"})

    for col in categorical_cols:
        rows.append({"feature": col, "type": "categorical"})

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "feature_inventory.csv"),
        index=False
    )

    pd.DataFrame({"train_columns": X.columns.tolist()}).to_csv(
        os.path.join(out_dir, "train_schema.csv"),
        index=False
    )


def save_metrics(y_true, y_pred, class_names, out_dir: str):
    metrics = {
        "Accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
        "BalancedAccuracy": sklearn.metrics.balanced_accuracy_score(y_true, y_pred),
        "PrecisionMacro": sklearn.metrics.precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "RecallMacro": sklearn.metrics.recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "F1Macro": sklearn.metrics.f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "PrecisionWeighted": sklearn.metrics.precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "RecallWeighted": sklearn.metrics.recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "F1Weighted": sklearn.metrics.f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }

    pd.DataFrame([metrics]).to_csv(
        os.path.join(out_dir, "report.csv"),
        index=False
    )

    report_dict = sklearn.metrics.classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(out_dir, "classification_report.csv"),
        index=True
    )


def save_confusion(y_true, y_pred, class_names, out_dir: str):
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    cmn_df = pd.DataFrame(
        cm_norm,
        index=class_names,
        columns=class_names
    )

    cmn_df.to_csv(os.path.join(out_dir, "confusion_matrix_normalized.csv"))


def save_predictions_file(y_test_enc, y_pred_enc, le: LabelEncoder, out_dir: str):
    df_pred = pd.DataFrame({
        "y_true_encoded": y_test_enc.astype(int),
        "y_pred_encoded": y_pred_enc.astype(int),
        "y_true_label": le.inverse_transform(y_test_enc.astype(int)),
        "y_pred_label": le.inverse_transform(y_pred_enc.astype(int)),
    })

    df_pred.to_csv(
        os.path.join(out_dir, "predictions.csv"),
        index=False
    )


def extract_feature_names(preprocessor, numeric_cols, categorical_cols):
    cat_names = []

    if categorical_cols:
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
        except Exception:
            cat_names = []

    return numeric_cols + cat_names


def save_feature_importance(model, feature_names, out_dir: str):
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_

    if len(importances) != len(feature_names):
        log(
            f"Warning: feature importance length mismatch. "
            f"Importances={len(importances)}, feature_names={len(feature_names)}"
        )
        min_len = min(len(importances), len(feature_names))
        importances = importances[:min_len]
        feature_names = feature_names[:min_len]

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    fi.to_csv(
        os.path.join(out_dir, "feature_importance.csv"),
        index=False
    )


def calculate_scale_pos_weight(y_train: np.ndarray) -> float:
    values, counts = np.unique(y_train, return_counts=True)
    count_dict = dict(zip(values, counts))

    if len(count_dict) != 2:
        log("Warning: scale_pos_weight set to 1.0 because this is not a binary problem.")
        return 1.0

    negative_count = count_dict.get(0, 0)
    positive_count = count_dict.get(1, 0)

    if positive_count == 0:
        log("Warning: positive class count is zero. scale_pos_weight set to 1.0.")
        return 1.0

    return float(negative_count / positive_count)


def main():
    safe_mkdir(OUT_DIR)

    log("Loading dataset...")
    df = pd.read_csv(DATASET_PATH, **CSV_READ_KWARGS)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)

    log(f"Rows before dedup: {before}")
    log(f"Rows after dedup : {after}")

    y_raw = df[TARGET_COLUMN].fillna("NA").astype(str)
    X = df.drop(columns=[TARGET_COLUMN], errors="ignore").copy()

    log("Performing stratified train/test split (80/20)...")
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X,
        y_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_raw,
    )

    le = LabelEncoder()
    le.fit(y_train_raw)

    unknown_test_labels = set(pd.unique(y_test_raw)) - set(le.classes_)

    if unknown_test_labels:
        raise ValueError(
            f"Test split has unseen labels not present in training: "
            f"{list(unknown_test_labels)}"
        )

    y_train = le.transform(y_train_raw).astype(int)
    y_test = le.transform(y_test_raw).astype(int)

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    save_split_metadata(
        df=df,
        X_train=X_train,
        X_test=X_test,
        y_train_raw=y_train_raw,
        y_test_raw=y_test_raw,
        out_dir=OUT_DIR,
        scale_pos_weight=scale_pos_weight
    )

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    save_feature_inventory(
        X=X_train,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        out_dir=OUT_DIR
    )

    log(f"Numeric features     : {len(numeric_cols)}")
    log(f"Categorical features : {len(categorical_cols)}")
    log(f"Classes              : {list(le.classes_)}")
    log(f"scale_pos_weight     : {scale_pos_weight:.6f}")

    model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    log("Training XGBoost...")
    clf.fit(X_train, y_train)
    log("Model trained.")

    log("Predicting test set...")
    y_pred = clf.predict(X_test).astype(int)

    log("Saving evaluation files...")
    save_metrics(y_test, y_pred, le.classes_, OUT_DIR)
    save_confusion(y_test, y_pred, le.classes_, OUT_DIR)

    if SAVE_PREDICTIONS:
        save_predictions_file(y_test, y_pred, le, OUT_DIR)

    feature_names = extract_feature_names(
        clf.named_steps["preprocess"],
        numeric_cols,
        categorical_cols
    )

    save_feature_importance(
        clf.named_steps["model"],
        feature_names,
        OUT_DIR
    )

    acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    bal_acc = sklearn.metrics.balanced_accuracy_score(y_test, y_pred)
    f1_macro = sklearn.metrics.f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )
    f1_weighted = sklearn.metrics.f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    summary_txt = [
        "=== XGBoost - UFU-DoH-EXF (80/20) ===",
        f"Dataset: {DATASET_PATH}",
        f"Rows total: {len(df)}",
        f"Train rows: {len(X_train)}",
        f"Test rows: {len(X_test)}",
        f"Classes: {list(le.classes_)}",
        "",
        "Model: XGBoost",
        f"n_estimators: {XGB_N_ESTIMATORS}",
        f"max_depth: {XGB_MAX_DEPTH}",
        f"learning_rate: {XGB_LEARNING_RATE}",
        f"subsample: {XGB_SUBSAMPLE}",
        f"colsample_bytree: {XGB_COLSAMPLE_BYTREE}",
        f"scale_pos_weight: {scale_pos_weight:.6f}",
        "",
        f"Accuracy: {acc:.6f}",
        f"Balanced Accuracy: {bal_acc:.6f}",
        f"F1 Macro: {f1_macro:.6f}",
        f"F1 Weighted: {f1_weighted:.6f}",
        "",
        f"Numeric features: {len(numeric_cols)}",
        f"Categorical features: {len(categorical_cols)}",
        "",
        "Top output files:",
        "- report.csv",
        "- classification_report.csv",
        "- confusion_matrix.csv",
        "- confusion_matrix_normalized.csv",
        "- predictions.csv",
        "- feature_importance.csv",
        "- split_metadata.json",
        "- feature_inventory.csv",
    ]

    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_txt))

    log("\n".join(summary_txt))
    log("Done.")


if __name__ == "__main__":
    main()