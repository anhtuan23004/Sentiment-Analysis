from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_model(data_path, output_dir='models', model_name='logistic_model', display_name=None):
    """
    Train a sentiment analysis model and save as .pkl.

    Args:
        data_path (str): Path to processed CSV.
        output_dir (str): Directory to save model.
        model_name (str): Name of the model file (without .pkl).
        display_name (str): Display name for UI (optional).

    Returns:
        tuple: Trained model and vectorizer.
    """
    try:
        # Load data
        df = pd.read_csv(data_path)
        X = df['processed_text']
        y = df['sentiment']

        # Split data
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize vectorizer and model
        vectorizer = TfidfVectorizer(max_features=5000)
        model = LogisticRegression(max_iter=1000)

        # Transform and train
        X_train_vec = vectorizer.fit_transform(X_train)
        model.fit(X_train_vec, y_train)

        # Save model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f'{model_name}.pkl')
        joblib.dump((model, vectorizer), model_path)
        logging.info(f"Model saved to {model_path}")

        # Update models_config.json
        config_path = os.path.join(output_dir, 'models_config.json')
        config = {'models': []}
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

        config['models'] = [m for m in config['models'] if m['file'] != f'{model_name}.pkl']
        config['models'].append({
            'file': f'{model_name}.pkl',
            'display_name': display_name or model_name.replace('_', ' ').title()
        })

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        logging.info(f"Updated {config_path}")

        return model, vectorizer
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise


if __name__ == '__main__':
    train_model(
        data_path='data/processed/processed_imdb.csv',
        model_name='logistic_model',
        display_name='Logistic Sentiment'
    )