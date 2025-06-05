from flask import Flask, request, jsonify, render_template, send_file, make_response
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import chardet
import pandas as pd
import os
import json
from datetime import datetime
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import string
import logging
from langdetect import detect, LangDetectException, DetectorFactory
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__, static_folder='static', template_folder='templates')

output_dir = 'static/output'
models_dir = 'scripts/models'

# Create directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load models
try:
    # Hugging Face DistilBERT
    hf_classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english',
                             truncation=True, max_length=512)
    logger.info("Successfully loaded HuggingFace DistilBERT model")
except Exception as e:
    logger.error(f"Error loading HuggingFace model: {str(e)}")
    hf_classifier = None

try:
    # Load finetuned RoBERTa model
    roberta_model_path = os.path.join('scripts/models/roberta_model')
    if os.path.exists(roberta_model_path):
        roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_path)
        roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_path)
        roberta_model.to(device)
        logger.info("Successfully loaded RoBERTa model")
    else:
        logger.warning(f"RoBERTa model path not found: {roberta_model_path}")
        roberta_model = None
        roberta_tokenizer = None
except Exception as e:
    logger.error(f"Error loading RoBERTa model: {str(e)}")
    roberta_model = None
    roberta_tokenizer = None

try:
    # mBERT multilingual model
    m_bert_classifier = pipeline('sentiment-analysis',
                               model='nlptown/bert-base-multilingual-uncased-sentiment',
                               truncation=True, max_length=512)
    logger.info("Successfully loaded mBERT multilingual model")
except Exception as e:
    logger.error(f"Error loading mBERT model: {str(e)}")
    m_bert_classifier = None

try:
    # XLM-RoBERTa multilingual model
    xlm_roberta_classifier = pipeline('sentiment-analysis',
                                     model='xlm-roberta-base',
                                     truncation=True, max_length=512)
    logger.info("Successfully loaded XLM-RoBERTa multilingual model")
except Exception as e:
    logger.error(f"Error loading XLM-RoBERTa model: {str(e)}")
    xlm_roberta_classifier = None

# Define max_length
max_length = 512

LANGUAGE_MODELS = {
    'vi': 'NlpHUST/vibert4news-sentiment',  # Vietnamese
    'en': 'distilbert-base-uncased-finetuned-sst-2-english',  # English
    'zh': 'uer/roberta-base-finetuned-jd-binary-chinese',  # Chinese
}

@lru_cache(maxsize=5)
def get_language_model(language_code):
    """Load a language-specific model on demand and cache it"""
    model_name = LANGUAGE_MODELS.get(language_code)
    if not model_name:
        return None

    try:
        model = pipeline('sentiment-analysis', model=model_name, truncation=True, max_length=512)
        logger.info(f"Loaded language-specific model for {language_code}: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Error loading model for {language_code}: {str(e)}")
        return None

def ensemble_prediction(text, language):
    """Combine predictions from multiple models for better accuracy."""
    predictions = []

    # Get language code
    lang_code = language.split('-')[0] if '-' in language else language

    # 1. Language-specific model
    lang_model = get_language_model(lang_code)
    if lang_model:
        try:
            result = lang_model(text)[0]
            pred = 'positive' if result['label'] in ['LABEL_1', 'positive', 'POSITIVE'] else 'negative'
            conf = float(result['score'])
            predictions.append({'model': f'lang-specific-{lang_code}', 'prediction': pred, 'confidence': conf * 1.2})
        except Exception as e:
            logger.error(f"Error with language-specific model: {str(e)}")

    # 2. Multilingual model (mBERT)
    if m_bert_classifier:
        try:
            result = m_bert_classifier(text)[0]
            stars = int(result['label'].split()[0])
            pred = 'positive' if stars > 3 else 'negative'
            conf = float(result['score'])
            predictions.append({'model': 'mbert', 'prediction': pred, 'confidence': conf})
        except Exception as e:
            logger.error(f"Error with mBERT: {str(e)}")
    # XLM-RoBERTa multilingual model
    if xlm_roberta_classifier:
        try:
            result = xlm_roberta_classifier(text)[0]
            pred = 'positive' if result['label'] == 'POSITIVE' else 'negative'
            predictions.append({
                'model': 'xlm-roberta',
                'prediction': pred,
                'confidence': float(result['score']) * 1.1  # Higher weight for XLM-RoBERTa
            })
        except Exception as e:
            logger.error(f"Error with XLM-RoBERTa: {str(e)}")

    # 4. Fallback to default English model
    if not predictions and hf_classifier:
        try:
            result = hf_classifier(text)[0]
            pred = 'positive' if result['label'] == 'POSITIVE' else 'negative'
            conf = float(result['score'])
            predictions.append({'model': 'distilbert-fallback', 'prediction': pred, 'confidence': conf})
        except Exception as e:
            logger.error(f"Error with fallback model: {str(e)}")

    # Aggregate predictions
    if not predictions:
        return None, None

    pos_score = sum(p['confidence'] for p in predictions if p['prediction'] == 'positive')
    neg_score = sum(p['confidence'] for p in predictions if p['prediction'] == 'negative')

    final_prediction = 'positive' if pos_score > neg_score else 'negative'
    total_confidence = pos_score + neg_score
    norm_confidence = (pos_score / total_confidence) if final_prediction == 'positive' else (neg_score / total_confidence)

    logger.info(f"Ensemble prediction: {final_prediction} with confidence {norm_confidence:.2f}")
    return final_prediction, norm_confidence

def preprocess_text(text):
    # Basic preprocessing: lowercase, remove punctuation, strip whitespace
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

def chunk_text(text, max_words=100, overlap=10):
    """Split text into smaller chunks with some overlap for context."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words - overlap):
        chunk = ' '.join(words[i:i + max_words])
        if chunk:  # Ensure we don't add empty chunks
            chunks.append(chunk)

    return chunks

DetectorFactory.seed = 0


# Add this function to detect language
def detect_language(text):
    """Detect the language of the input text."""
    try:
        if not text or len(text.strip()) < 3:
            return "unknown"
        return detect(text)
    except LangDetectException:
        return "unknown"


# Update the analyze_long_text function to include language detection
# Fix analyze_long_text function which is returning only 2 values but 3 are expected
def analyze_long_text(text, model_name, max_chunk_words=100):
    """Process long text by chunking and aggregating results."""
    try:
        # Detect language first
        language = detect_language(text)
        logger.info(f"Detected language: {language}")

        # First check if text is already within limits
        if count_words(text) <= 500 and estimate_tokens(text) <= 512:
            # Process normally if text is short enough
            prediction, confidence = predict_single_chunk(text, model_name)
            return prediction, confidence, language

        # For longer texts, break into chunks
        chunks = chunk_text(text, max_words=max_chunk_words)
        logger.info(f"Text split into {len(chunks)} chunks for processing")

        results = []
        for chunk in chunks:
            # Skip chunks that are too large (shouldn't happen with our chunking function)
            if estimate_tokens(chunk) > 512:
                logger.warning(f"Chunk exceeds token limit, skipping")
                continue

            prediction, confidence = predict_single_chunk(chunk, model_name)
            if prediction:
                results.append({
                    'prediction': prediction,
                    'confidence': confidence
                })

        if not results:
            return None, None, language

        # Aggregate results (weighted by confidence)
        pos_score = sum(r['confidence'] for r in results if r['prediction'] == 'positive')
        neg_score = sum(r['confidence'] for r in results if r['prediction'] == 'negative')

        # Final prediction is the sentiment with the higher aggregate score
        prediction = 'positive' if pos_score >= neg_score else 'negative'
        # Confidence is the average of chunk confidences
        avg_confidence = sum(r['confidence'] for r in results) / len(results)

        return prediction, avg_confidence, language

    except Exception as e:
        logger.error(f"Error in analyze_long_text: {str(e)}")
        return None, None, "unknown"


# Update the predict_single_chunk function to handle language
def predict_single_chunk(text, model_name, language="en"):
    """Process a single text chunk with the selected model."""
    prediction = None
    confidence = None

    # Log the language for debugging
    logger.info(f"Processing text in language: {language}")

    # Use ensemble prediction for multilingual model
    if model_name == 'multilingual':
        return ensemble_prediction(text, language)

    # Default models
    if model_name == 'huggingface':
        if hf_classifier is None:
            return None, None
        result = hf_classifier(text)[0]
        prediction = 'positive' if result['label'] == 'POSITIVE' else 'negative'
        confidence = float(result['score'])

    elif model_name == 'mbert':
        if m_bert_classifier is None:
            return None, None
        result = m_bert_classifier(text)[0]
        stars = int(result['label'].split()[0])
        prediction = 'positive' if stars > 3 else 'negative'
        confidence = float(result['score'])

    elif model_name == 'xlm-roberta':
        if xlm_roberta_classifier is None:
            return None, None
        result = xlm_roberta_classifier(text)[0]
        prediction = 'positive' if result['label'] == 'POSITIVE' else 'negative'
        confidence = float(result['score'])

    elif model_name == 'roberta':
        if roberta_model is None or roberta_tokenizer is None:
            return None, None
        prediction, confidence = predict_sentiment(text, roberta_model, roberta_tokenizer, device)

    else:
        model_path = os.path.join(models_dir, model_name)
        prediction, confidence = predict_text(model_path, text)

    return prediction, confidence


def predict_sentiment(text, model, tokenizer, device):
    if model is None or tokenizer is None:
        return None, None

    try:
        # Preprocess the text
        processed_text = preprocess_text(text)
        # Tokenize
        encoded_dict = tokenizer.encode_plus(
            processed_text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # Move to device
        input_ids = encoded_dict['input_ids'].to(device)
        attention_mask = encoded_dict['attention_mask'].to(device)
        # Set model to evaluation mode
        model.eval()
        # Get prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        # Get prediction class
        prediction = torch.argmax(logits, dim=1).item()
        # Map prediction to sentiment
        sentiment_map = {0: 'negative', 1: 'positive'}
        sentiment = sentiment_map[prediction]
        # Compute confidence (softmax on logits)
        probs = torch.softmax(logits, dim=1)
        confidence = float(probs[0][prediction].item())  # Convert to float for JSON
        return sentiment, confidence
    except Exception as e:
        logger.error(f"Error in predict_sentiment: {str(e)}")
        return None, None


def count_words(text):
    return len(text.strip().split())


def load_model_config():
    config_path = os.path.join(models_dir, 'models_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return {item['file']: item['display_name'] for item in config.get('models', [])}
        except Exception as e:
            logger.error(f"Error loading models_config.json: {str(e)}")
            return {}
    return {}


# Load vectorizer and model for custom .pkl models
def load_custom_model(model_path):
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None, None

    try:
        model_data = joblib.load(model_path)
        if isinstance(model_data, tuple) and len(model_data) == 2:
            model, vectorizer = model_data
            return model, vectorizer
        else:
            model = model_data
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            return model, vectorizer
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {str(e)}")
        return None, None


def predict_text(model_path, text):
    model, vectorizer = load_custom_model(model_path)
    if model is None or vectorizer is None:
        return None, None
    try:
        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)[0]
        confidence = float(model.predict_proba(text_vector)[0].max()) if hasattr(model, 'predict_proba') else 0.5
        prediction_label = 'positive' if prediction == 1 else 'negative'
        return prediction_label, confidence
    except Exception as e:
        logger.error(f"Error predicting with model {model_path}: {str(e)}")
        return None, None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    # Simple health check endpoint
    return jsonify({'status': 'ok'})


@app.route('/models', methods=['GET'])
def get_models():
    model_config = load_model_config()
    pkl_models = []
    try:
        pkl_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    except FileNotFoundError:
        logger.warning(f"Models directory '{models_dir}' not found. Creating it.")
        os.makedirs(models_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Error accessing models directory: {str(e)}")

    models = []

    # Add multilingual ensemble model as the first option
    models.append({
        'value': 'multilingual',
        'display_name': 'Multilingual Ensemble (Recommended for Non-English)'
    })

    # Add individual multilingual models
    if xlm_roberta_classifier is not None:
        models.append({'value': 'xlm-roberta', 'display_name': 'XLM-RoBERTa (Best for Multilingual)'})

    if m_bert_classifier is not None:
        models.append({'value': 'mbert', 'display_name': 'mBERT (Multilingual Sentiment)'})

    # Only add classic models that are available
    if hf_classifier is not None:
        models.append({'value': 'huggingface', 'display_name': 'DistilBERT (English)'})

    if roberta_model is not None and roberta_tokenizer is not None:
        models.append({'value': 'roberta', 'display_name': 'RoBERTa (English)'})

    for model_file in pkl_models:
        display_name = model_config.get(model_file, model_file.replace('.pkl', '').replace('_', ' ').title())
        models.append({'value': model_file, 'display_name': display_name})

    logger.info(f"Available models: {[m['value'] for m in models]}")
    return jsonify(models)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400

        text = data.get('text', '')
        model_name = data.get('model', 'huggingface')

        logger.info(f"Predict request with model: {model_name}, text length: {len(text)}")

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # For very large texts (over 5000 words), return an error
        word_count = count_words(text)
        if word_count > 5000:
            return jsonify({'error': 'Text too large (exceeds 5000 words)'}), 400

        # Use new long text processing function with language detection
        prediction, confidence, language = analyze_long_text(text, model_name)

        if prediction is None:
            return jsonify({'error': f'Failed to predict with model: {model_name}'}), 500

        # Return the prediction with language information
        response = make_response(jsonify({
            'prediction': prediction,
            'confidence': float(confidence),
            'language': language,
            'processed': 'chunked' if word_count > 500 else 'direct'
        }))
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response

    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch', methods=['POST'])
def batch_process():
    try:
        if 'files[]' not in request.files:
            logger.error("No files found in request")
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('files[]')
        if not files or len(files) == 0:
            return jsonify({'error': 'No valid files uploaded'}), 400

        model_name = request.form.get('model', 'huggingface')
        logger.info(f"Batch request with model: {model_name}, files: {[f.filename for f in files]}")

        all_results = []

        for file in files:
            if not file or not file.filename:
                continue

            if not file.filename.endswith(('.csv', '.txt')):
                logger.warning(f"Skipping unsupported file: {file.filename}")
                continue

            try:
                file.seek(0)
                raw_data = file.read()
                encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
                content = raw_data.decode(encoding)
            except Exception as e:
                logger.error(f"Error decoding file {file.filename}: {str(e)}")
                continue

            texts = []
            if file.filename.endswith('.csv'):
                try:
                    file.seek(0)
                    df = pd.read_csv(file)
                    if 'text' not in df.columns:
                        logger.warning(f"CSV file {file.filename} missing 'text' column")
                        continue
                    texts = df['text'].astype(str).tolist()
                except Exception as e:
                    logger.error(f"Error reading CSV {file.filename}: {str(e)}")
                    continue
            else:
                texts = [line.strip() for line in content.splitlines() if line.strip()]

            if not texts:
                logger.warning(f"No valid texts found in file {file.filename}")
                continue

            file_results = []
            for text in texts:
                word_count = count_words(text)
                if word_count > 5000:
                    logger.warning(f"Text exceeds 5000 words, skipping")
                    continue

                prediction, confidence, language = analyze_long_text(text, model_name)

                if prediction is None:
                    continue

                file_results.append({
                    'text': text,
                    'prediction': prediction,
                    'confidence': float(confidence),
                    'language': language,
                    'filename': file.filename,
                    'processed': 'chunked' if word_count > 500 else 'direct'
                })

            all_results.extend(file_results)

        if not all_results:
            return jsonify({'error': 'No valid texts processed'}), 400

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f'batch_results_{timestamp}.csv')

        pd.DataFrame(all_results).to_csv(output_file, index=False)

        response = make_response(jsonify({
            'results': all_results,
            'download_url': f'/download/{os.path.basename(output_file)}'
        }))
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response

    except Exception as e:
        logger.error(f"Error in batch endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download(filename):
    try:
        file_path = os.path.join(output_dir, filename)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404

        return send_file(file_path, as_attachment=True, mimetype='text/csv')
    except Exception as e:
        logger.error(f"Error in download endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


def estimate_tokens(text):
    return len(re.findall(r'\w+|[^\w\s]', text)) + 10


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)
