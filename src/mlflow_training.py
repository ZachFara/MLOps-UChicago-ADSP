import kagglehub
from pathlib import Path
import shutil
import os
import glob
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
# Required imports
import seaborn as sns
import shap
import mlflow
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
import pickle
import random

mlflow.set_experiment("Final Classification Experiment")

# Function to evaluate classification metrics
def classification_metrics(y_true, y_pred):
    y_pred = (y_pred > .5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    return accuracy, precision, recall, f1, roc_auc

# Flexible experiment runner
def run_experiment(train, test, model, hyperparams=None):
    try:
        X_train, y_train = train
        X_test, y_test = test
        
        scaler = StandardScaler()
        continuous_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
        X_train[continuous_columns] = scaler.fit_transform(X_train[continuous_columns])
        X_test[continuous_columns] = scaler.transform(X_test[continuous_columns])

        if hyperparams:
            model.set_params(**hyperparams)

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        metrics_train = classification_metrics(y_train, y_pred_train)
        metrics_test = classification_metrics(y_test, y_pred_test)
        
        input_example = X_test.iloc[:5]

        mlflow.start_run()
        mlflow.log_param("model", type(model).__name__)
        if hyperparams:
            mlflow.log_params(hyperparams)
        mlflow.log_metric("accuracy_train", metrics_train[0])
        mlflow.log_metric("accuracy_test", metrics_test[0])
        mlflow.log_metric("precision_train", metrics_train[1])
        mlflow.log_metric("precision_test", metrics_test[1])
        mlflow.log_metric("recall_train", metrics_train[2])
        mlflow.log_metric("recall_test", metrics_test[2])
        mlflow.log_metric("f1_train", metrics_train[3])
        mlflow.log_metric("f1_test", metrics_test[3])
        mlflow.log_metric("roc_auc_train", metrics_train[4])
        mlflow.log_metric("roc_auc_test", metrics_test[4])
        
        if isinstance(model, (RandomForestClassifier, XGBClassifier, MLPClassifier)):
            mlflow.sklearn.log_model(model, "model", input_example=input_example)
        else:
            print("Model not supported for logging")

        generate_model_explanations(model, X_train, X_test, y_test)
        generate_confusion_matrix(y_test, y_pred_test)
        generate_pr_roc_curves(model, X_test, y_test)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

    finally:
        mlflow.end_run()
        return scaler

def generate_model_explanations(model, X_train, X_test, y_test, show = False):
    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, show = show)
        mlflow.log_figure(plt.gcf(), "shap_summary_plot.png")
        if not show:
            plt.close()
    elif isinstance(model, MLPClassifier):
        background = X_train.sample(n=100, random_state=0)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_test, nsamples=100)
        shap.summary_plot(shap_values, X_test, show = show)
        mlflow.log_figure(plt.gcf(), "shap_summary_plot.png")
        if not show:
            plt.close()

def generate_confusion_matrix(y_test, y_pred_test, show = False):
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if show:
        plt.show()
    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")

def generate_pr_roc_curves(model, X_test, y_test, show = False):
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    if show:
        plt.show()
    mlflow.log_figure(plt.gcf(), "precision_recall_curve.png")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    if show:
        plt.show()    
    mlflow.log_figure(plt.gcf(), "roc_curve.png")

# Example models and parameter grids


# Hyperparameter tuning for XGBClassifier
xgb_param_grid = {
    'n_estimators': list(range(50, 1000, 50)),
    'max_depth': [2, 3, 5, 7, 9, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [1, 1.5, 2, 5]
}





def main():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    assert "User ID" not in train.columns
    assert "UserID" not in train.columns

    
    x_train = train.drop(columns=['Productivity Lost'])
    y_train = train['Productivity Lost']
    x_test = test.drop(columns=['Productivity Lost'])
    y_test = test['Productivity Lost']

    training_data = (x_train, y_train)
    testing_data = (x_test, y_test)
    
    xgb_model = XGBClassifier(random_state=0)
    rf_model = RandomForestClassifier(random_state=0)
    mlp_model = MLPClassifier(random_state=0)
    
        # Example hyperparameter tuning iterations
    for i in range(5):
        hyperparams = {k: np.random.choice(v) for k, v in xgb_param_grid.items()}
        print(f"XGBClassifier Iteration {i+1}/2")
        print("Hyperparameters:", hyperparams)
        scaler = run_experiment(train=training_data, test=testing_data, model=xgb_model, hyperparams=hyperparams)
        
    param_grid = {
    'n_estimators': list(range(50, 1000, 50)),  # Number of trees
    'max_depth': [2, 3, 5, 7, 9, 12],  # Maximum depth of trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': [None, 'sqrt', 'log2']  # Number of features to consider when looking for the best split
}
    # Save scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    num_iterations = 5
    for i in range(num_iterations):
        # Randomly sample hyperparameters
        hyperparams = {k: np.random.choice(v) for k, v in param_grid.items()}
        print(f"Iteration {i+1}/{num_iterations}")
        print("Hyperparameters:", hyperparams)
        scaler = run_experiment(train=training_data, test=testing_data, model=rf_model, hyperparams=hyperparams)
        
    param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': np.linspace(0.001, 0.1, 50),  # Fix learning rate range
    'max_iter': [200, 300, 400]
}

    import random

    num_iterations = 2
    for i in range(num_iterations):
        hyperparams = {k: random.choice(v) for k, v in param_grid.items()}
        print(f"Iteration {i+1}/{num_iterations}")
        print("Hyperparameters:", hyperparams)
        scaler = run_experiment(train=training_data, test=testing_data, model=mlp_model, hyperparams=hyperparams)

if __name__ == '__main__':
    main()