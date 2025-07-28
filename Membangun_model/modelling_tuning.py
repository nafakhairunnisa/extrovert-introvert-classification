import mlflow
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from dotenv import load_dotenv
import os
import joblib
from dagshub import auth, init

# Setup Tracking URI
load_dotenv()
USE_DAGSHUB = False
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

try:
    if DAGSHUB_TOKEN:
        auth.add_app_token(DAGSHUB_TOKEN)
        mlflow.set_tracking_uri("https://dagshub.com/nafakhairunnisa/Membangun_model.mlflow")
        init(repo_owner='nafakhairunnisa', repo_name='Membangun_model', mlflow=True)
        USE_DAGSHUB = True
        print("Tracking ke Dagshub...")
    else:
        print("No DAGSHUB_TOKEN found, tracking local...")
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
except Exception as e:
    print(f"Fallback ke lokal MLflow tracking. Reason: {e}")
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")

mlflow.set_experiment("behavior-classifier")

def load_data():
    data_dir = Path('personality_preprocessing')
    return (
        pd.read_csv(data_dir/'X_train.csv'),
        pd.read_csv(data_dir/'X_test.csv'),
        pd.read_csv(data_dir/'y_train.csv').squeeze("columns"),
        pd.read_csv(data_dir/'y_test.csv').squeeze("columns")
    )

def log_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = {
        'accuracy': report['accuracy'],
        'f1_weighted': report['weighted avg']['f1-score'],
        'roc_auc': roc_auc_score(y_true, y_pred),
        'precision_class0': report['0']['precision'],
        'recall_class1': report['1']['recall'],
        'log_loss': log_loss(y_true, y_pred, labels=[0, 1])
    }
    mlflow.log_metrics(metrics)

    # Simpan report ke file .json
    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Log file json sebagai artefak
    mlflow.log_artifact("classification_report.json")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    with mlflow.start_run(run_name="RF_Tuned"):
        mlflow.log_params(grid_search.best_params_)
        y_pred = best_model.predict(X_test)
        log_metrics(y_test, y_pred)

        # Log model via mlflow
        mlflow.sklearn.log_model(best_model, "model")

        # Save model locally and log as artifact
        model_path = "rf_model_tuned.joblib"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        print(f"Tuned RF - F1 Weighted: {f1_score(y_test, y_pred, average='weighted'):.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')

        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        print("Confusion matrix & model logged.")
