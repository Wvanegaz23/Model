"""Evaluación ilustrativa de un modelo de retiro con datos históricos."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
NUMERIC = [
    "Saldos.Mes.Ant", "Pago.Minimo", "Vr.Cuota.Manejo", "Total.Intereses",
    "Edad.Mora", "Vr.Mora", "Pagos.Mes.Ant", "Pago.del.Mes",
    "Disponible.Avances",
]
CATEGORICAL = ["categoria", "segmento", "nivel_educativo", "estado_civil", "Genero"]


def load_data(root=ROOT):
    data = pd.read_csv(root / "train.csv", sep=";", encoding="ISO-8859-1")
    for filename in ("train_test_demograficas.xlsx", "train_test_subsidios.xlsx"):
        extra = pd.read_excel(root / filename)
        data = data.merge(extra, on="id", how="left", validate="one_to_one")

    y = pd.to_numeric(data["Target"], errors="raise")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Target debe contener únicamente las clases 0 y 1, sin nulos.")
    X = data[NUMERIC + CATEGORICAL].copy()
    for col in NUMERIC:
        cleaned = X[col].astype("string").str.replace(r"[$,\s]", "", regex=True)
        X[col] = pd.to_numeric(cleaned, errors="raise")
    X[CATEGORICAL] = X[CATEGORICAL].astype(object).where(
        X[CATEGORICAL].notna(), float("nan")
    )
    return X, y.astype(int)


def make_model():
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Prevalencia de retiro en evaluación: {y_test.mean():.3f}")
    for name, model in (
        ("Línea base (clase mayoritaria)", DummyClassifier(strategy="most_frequent")),
        ("Regresión logística balanceada", make_model()),
    ):
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]
        print(f"\n{name}")
        print(classification_report(y_test, predicted, digits=3, zero_division=0))
        print("Matriz de confusión (filas reales, columnas predichas):")
        print(confusion_matrix(y_test, predicted))
        print(f"Average precision: {average_precision_score(y_test, probability):.3f}")
        print(f"ROC-AUC: {roc_auc_score(y_test, probability):.3f}")


if __name__ == "__main__":
    main()
