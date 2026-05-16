import mlflow
import mlflow.sklearn
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=10)
args = parser.parse_args()

X_train = pd.read_csv('diabetes_preprocessing/X_train.csv')
X_test = pd.read_csv('diabetes_preprocessing/X_test.csv')
y_train = pd.read_csv('diabetes_preprocessing/y_train.csv').values.ravel()
y_test = pd.read_csv('diabetes_preprocessing/y_test.csv').values.ravel()

mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("test_f1", f1_score(y_test, preds))
    mlflow.log_metric("test_precision", precision_score(y_test, preds))
    mlflow.log_metric("test_recall", recall_score(y_test, preds))
    
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")