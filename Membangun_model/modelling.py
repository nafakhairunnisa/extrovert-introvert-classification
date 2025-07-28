import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("behavior-classifier")

mlflow.autolog()

def load_data():
    data_dir = Path('personality_preprocessing')
    return (
        pd.read_csv(data_dir / 'X_train.csv'),
        pd.read_csv(data_dir / 'X_test.csv'),
        pd.read_csv(data_dir / 'y_train.csv').squeeze("columns"),
        pd.read_csv(data_dir / 'y_test.csv').squeeze("columns")
    )

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model = RandomForestClassifier(random_state=42)

    with mlflow.start_run(run_name="basic_rf"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Evaluating...")

        # Evaluasi
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')

        print(f"F1 Weighted: {f1:.4f}")
        print(f"Accuracy    : {acc:.4f}")
        print(f"Precision   : {prec:.4f}")
        print(f"Recall      : {rec:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.show()
        plt.close()