from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelEvaluator:
    """
    Evaluate a sentiment analysis model.

    Args:
        model_path (str): Path to .pkl model.
        data_path (str): Path to test dataset.
    """

    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        try:
            self.model, self.vectorizer = joblib.load(model_path)
            self.df = pd.read_csv(data_path)
            logging.info(f"Loaded model from {model_path} and data from {data_path}")
        except Exception as e:
            logging.error(f"Error loading model or data: {e}")
            raise

    def evaluate(self):
        """
        Evaluate model on test data.

        Returns:
            dict: Evaluation metrics.
        """
        try:
            X = self.df['processed_text']
            y = self.df['sentiment']
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            X_test_vec = self.vectorizer.transform(X_test)
            y_pred = self.model.predict(X_test_vec)

            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            metrics = {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }

            logging.info(f"Evaluation results: {metrics}")
            return metrics
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            raise


if __name__ == '__main__':
    evaluator = ModelEvaluator(
        model_path='models/logistic_model.pkl',
        data_path='data/processed/processed_imdb.csv'
    )
    metrics = evaluator.evaluate()
    print(metrics)