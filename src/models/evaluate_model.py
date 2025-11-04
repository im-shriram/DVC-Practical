# Importing necessary libraries
import numpy as np
import pandas as pd
import pathlib
import logging
import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import yaml
from dvclive import Live
# import mlflow
from sklearn.metrics import ConfusionMatrixDisplay


# Loading Data - Train and Test
def load_data(data_dir: str, live: Live) -> pd.DataFrame:
    logger = logging.getLogger(__name__)

    # Forming file paths
    train_path = data_dir / "features" / "train.csv"
    test_path = data_dir / "features" / "test.csv"
    logger.info(f"Loading dataset from {data_dir / "features"}")

    # Loading datasets
    train_df = pd.read_csv(filepath_or_buffer=train_path)
    test_df = pd.read_csv(filepath_or_buffer=test_path)

    # Logging both the datasets using dvc
    logger.info(f"Logging Train and Test Datasets using DVC")
    live.log_artifact(
        path=train_path,
        type="dataset",
        name="train_dataset"
    )
    live.log_artifact(
        path=test_path,
        type="dataset",
        name="test_dataset"
    )
    
    # # Logging both the datasets using mlflow
    # logger.info(f"Logging Train and Test Datasets using MLflow")
    # # Convert pandas DataFrames to mlflow compatible datasets
    # train_dataset = mlflow.data.from_pandas(train_df)
    # test_dataset = mlflow.data.from_pandas(test_df)
    
    # # Log the datasets using mlflow
    # mlflow.log_input(train_dataset, context="training")
    # mlflow.log_input(test_dataset, context="testing")

    # Returning datasets
    return train_df.dropna(), test_df.dropna()

# Loading Parameters
def load_params(params_path: pathlib.Path, live:Live) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Loading parameters from {params_path}")
    with open(file=params_path, mode='r') as f:
        params = yaml.safe_load(f)
        
        # Check if 'make_dataset' key exists in params
        if 'model_training' not in params:
            logger.error("model_training section not found in parameters file")
            raise KeyError("model_training section not found in parameters file")

        # Logging all the parameters using dvc
        logger.info(f"Logging all the prameters using DVC")
        live.log_params(params["data_ingestion"])
        live.log_params(params["feature_engineering"])
        live.log_params(params["model_training"]["estimator"])
        live.log_params(params["model_training"]["bagging"])

        # Logging all the parameters using mlflow
        # logger.info(f"Logging all the prameters using MLflow")
        # mlflow.log_params(params["data_ingestion"])
        # mlflow.log_params(params["feature_engineering"])
        # mlflow.log_params(params["model_training"]["estimator"])
        # mlflow.log_params(params["model_training"]["bagging"])

# Loading Model
def load_model(model_dir: str, live: Live) -> BaggingClassifier:
    logger = logging.getLogger(__name__)

    # Loading Model
    model_path = model_dir / "models" / "bagging_classifier.joblib"
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(filename=model_path)

    # Logging model using DVC
    logger.info(f"Logging model using DVC from {model_path}")
    live.log_artifact(
        path=model_path,
        type="model",
        name="BaggingClassifier"
    )

    # Logging model using MLflow
    # logger.info(f"Logging model using MLflow from {model_path}")
    # mlflow.sklearn.log_model(model)

    return model

# Evaluating Model
def evaluate_model(df: pd.DataFrame, model: BaggingClassifier, live: Live, split: str) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Evaluating the model")

    X = df.drop(columns=["sentiment", "content"])
    y = df["sentiment"]

    y_pred = model.predict(X=X)
    y_probab = model.predict_proba(X=X)
    predictions = y_probab[:, 1]

    report_dict = classification_report(y_true=y, y_pred=y_pred, output_dict=True)
    clf_matrix = confusion_matrix(y_true=y, y_pred=y_pred)

    # Log metrics for each class
    for class_label in report_dict.keys():
        if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics_prefix = f"{split}_class_{class_label}"
            live.log_metric(f"{metrics_prefix}_precision", report_dict[class_label]['precision'])
            live.log_metric(f"{metrics_prefix}_recall", report_dict[class_label]['recall'])
            live.log_metric(f"{metrics_prefix}_f1", report_dict[class_label]['f1-score'])
            live.log_metric(f"{metrics_prefix}_support", report_dict[class_label]['support'])

            # mlflow.log_metric(key=f"{metrics_prefix}_precision", value=report_dict[class_label]['precision'])
            # mlflow.log_metric(key=f"{metrics_prefix}_recall", value=report_dict[class_label]['recall'])
            # mlflow.log_metric(key=f"{metrics_prefix}_f1", value=report_dict[class_label]['f1-score'])
            # mlflow.log_metric(key=f"{metrics_prefix}_support", value=report_dict[class_label]['support'])
    
    # Log overall metrics
    live.log_metric(f"{split}_accuracy", report_dict['accuracy'])
    # mlflow.log_metric(f"{split}_accuracy", report_dict['accuracy'])
    
    # Log macro averages
    live.log_metric(f"{split}_macro_precision", report_dict['macro avg']['precision'])
    live.log_metric(f"{split}_macro_recall", report_dict['macro avg']['recall'])
    live.log_metric(f"{split}_macro_f1", report_dict['macro avg']['f1-score'])
    # mlflow.log_metric(f"{split}_macro_precision", report_dict['macro avg']['precision'])
    # mlflow.log_metric(f"{split}_macro_recall", report_dict['macro avg']['recall'])
    # mlflow.log_metric(f"{split}_macro_f1", report_dict['macro avg']['f1-score'])
    
    # Log weighted averages
    live.log_metric(f"{split}_weighted_precision", report_dict['weighted avg']['precision'])
    live.log_metric(f"{split}_weighted_recall", report_dict['weighted avg']['recall'])
    live.log_metric(f"{split}_weighted_f1", report_dict['weighted avg']['f1-score'])
    # mlflow.log_metric(f"{split}_weighted_precision", report_dict['weighted avg']['precision'])
    # mlflow.log_metric(f"{split}_weighted_recall", report_dict['weighted avg']['recall'])
    # mlflow.log_metric(f"{split}_weighted_f1", report_dict['weighted avg']['f1-score'])

    # Logging the evaluation plots
    live.log_sklearn_plot("confusion_matrix", y, y_pred)
    live.log_sklearn_plot("precision_recall", y, predictions)
    # Log confusion matrix plot using MLflow
    # fig, ax = plt.subplots(figsize=(10, 8))
    # disp = ConfusionMatrixDisplay(confusion_matrix=clf_matrix, display_labels=model.classes_)
    # disp.plot(ax=ax)
    # plt.title(f'Confusion Matrix - {split} set')
    # mlflow.log_figure(fig, f"{split}_confusion_matrix.png")
    # plt.close()

# Forming Logger
def form_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger


# Main function
def main() -> None:
    # Forming logger
    logger = form_logger()
    logger.info(msg="Started model evaluation pipeline")

    # Setting mlflow tracking uri
    # mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Forming directory paths
    home_dir = pathlib.Path(__file__).parent.parent.parent
    data_dir = home_dir / "data"
    params_path = home_dir / "params.yaml"
    model_dir = home_dir / "models" 
    logger.info(f"Working directory: {home_dir}")

    # dvclive storing path
    dvclive_path = home_dir / "dvclive"
    # with mlflow.start_run() as run:
    with Live(dir=dvclive_path, save_dvc_exp=True, exp_name="dummy_exp") as live:
        # Loading data
        train_df, test_df = load_data(data_dir=data_dir, live=live)

        # Loading Parameters
        load_params(params_path=params_path, live=live)

        # Loading model
        model = load_model(model_dir=model_dir, live=live)

        # Evaluating model
        evaluate_model(df=train_df, model=model, live=live, split="train")
        evaluate_model(df=test_df, model=model, live=live, split="test")

        logger.info("Model Evaluation completed successfully")

if __name__ == "__main__":
    main()