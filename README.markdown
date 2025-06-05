# Sentiment Analysis Web Application

## Overview

<div>
    <img src="./img1.png" width="60%" height="50%">
</div>

This web application performs sentiment analysis on English text using Hugging Face's DistilBERT or RoBert-finetune. It supports single text input and batch processing via CSV uploads, with confidence score visualizations using Chart.js.

## Features

- Single text sentiment analysis (positive/negative).
- Batch processing for CSV files with a "text" column.
- Confidence score visualization with bar charts.
- Model selection: Hugging Face (DistilBERT) or TextBlob.

## Requirements

- Python 3.10+
- Libraries listed in `requirements.txt`
- Docker (optional for containerized deployment)

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd sentiment-analysis-app
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

## Running Locally

1. Start the Flask server:

   ```bash
   python app.py
   ```

2. Open a browser and navigate to `http://localhost:5000`.

## Usage

- **Single Text:** Enter text, select a model (Hugging Face or TextBlob), and click "Analyze" to view sentiment and confidence.
- **Batch Processing:** Upload a CSV with a "text" column, select a model, and click "Analyze Batch" to view and download results.
- **Charts:** Confidence scores are visualized as bar charts.

## Batch Input Format

Create a CSV file (e.g., `input.csv`):

```
text
"This movie is great!"
"I hated this film."
"Neutral opinion here."
```

## Deployment

### Heroku

1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli

2. Install gunicorn: `pip install gunicorn`

3. Create Heroku app:

   ```bash
   heroku create sentiment-analysis-app
   ```

4. Deploy:

   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   heroku git:remote -a sentiment-analysis-app
   git push heroku main
   ```

5. Scale dynos: `heroku ps:scale web=1`

### Docker

1. Install Docker: https://docs.docker.com/get-docker/

2. Build image:

   ```bash
   docker build -t sentiment-analysis-app .
   ```

3. Run container:

   ```bash
   docker run -p 5000:5000 sentiment-analysis-app
   ```

## Project Structure

- `models/`: Empty (Hugging Face models downloaded at runtime).
- `static/`: CSS, JavaScript, and batch output files.
- `templates/`: HTML templates (`index.html`, `batch.html`).
- `app.py`: Flask backend.
- `requirements.txt`: Dependencies.
- `Dockerfile`: Docker configuration.
- `Procfile`: Heroku configuration.
- `README.md`: This file.

## Troubleshooting

- Ensure sufficient memory (2GB+) for Hugging Face models.
- Verify CSV files for batch processing have a "text" column.
- Check Heroku logs with `heroku logs --tail` or Docker logs with `docker logs <container-id>`.

[//]: # (## Future Improvements)

[//]: # ()
[//]: # (- Add support for non-English text.)

[//]: # (- Implement real-time progress indicators for batch processing.)

[//]: # (- Optimize Hugging Face model loading for faster startup.)
