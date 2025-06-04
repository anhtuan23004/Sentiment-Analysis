import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def predict_text(model_path, text):
    """
    Predict sentiment for a single text.

    Args:
        model_path (str): Path to .pkl model.
        text (str): Input text.

    Returns:
        tuple: Prediction and confidence.
    """
    try:
        model, vectorizer = joblib.load(model_path)
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X)[0].max()
        return 'positive' if prediction == 1 else 'negative', confidence
    except Exception as e:
        logging.error(f"Error predicting text: {e}")
        return None, None


def predict_batch(model_path, texts):
    """
    Predict sentiment for a batch of texts.

    Args:
        model_path: Path to .pkl model.
        texts: List of texts.

    Returns:
        list: List of predictions and confidences.
    """
    try:
        model, vectorizer = joblib.load(model_path)
        X = []
        for text in texts:
            X.append(vectorizer.transform([text]))
        predictions = []
        for x in X:
            predictions.append(model.predict(x))
        confidences = []
        for x in X:
            confidences.append(model.predict_proba(x).max(axis=1))
        results = []
        for p, c in zip(predictions, confidences):
            results.append({
                'prediction': 'positive' if p == 1 else 'negative',
                'confidence': c
            })
        return results
    except Exception as e:
        logging.error(f"Error predicting batch: {e}")
        return []


if __name__ == '__main__':
    result = predict_text('models/logistic_model.pkl', 'This movie is fantastic!')
    print(result)
    batch_results = predict_batch('models/logistic_model.pkl', ['Great film!', 'Terrible movie'])
    print(batch_results)