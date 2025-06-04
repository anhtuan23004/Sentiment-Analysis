import os
import requests
import tarfile
import pandas as pd
from urllib.request import urlretrieve
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_imdb_dataset(output_dir='data/raw', url='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'):
    """
    Download and extract IMDb sentiment dataset.

    Args:
        output_dir (str): Directory to save raw data.
        url (str): URL of the dataset.

    Returns:
        str: Path to extracted dataset.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        tar_path = os.path.join(output_dir, 'aclImdb_v1.tar.gz')
        extract_path = os.path.join(output_dir, 'aclImdb')

        if not os.path.exists(tar_path):
            logging.info(f"Downloading dataset from {url}")
            urlretrieve(url, tar_path)
        else:
            logging.info("Dataset already downloaded")

        if not os.path.exists(extract_path):
            logging.info("Extracting dataset")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(output_dir)

        # Convert to CSV
        csv_path = os.path.join(output_dir, 'imdb_dataset.csv')
        if not os.path.exists(csv_path):
            logging.info("Converting to CSV")
            reviews = []
            labels = []
            for split in ['train', 'test']:
                for sentiment in ['pos', 'neg']:
                    folder = os.path.join(extract_path, split, sentiment)
                    label = 1 if sentiment == 'pos' else 0
                    for filename in os.listdir(folder):
                        with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                            reviews.append(f.read())
                            labels.append(label)
            df = pd.DataFrame({'text': reviews, 'sentiment': labels})
            df.to_csv(csv_path, index=False)
            logging.info(f"Dataset saved to {csv_path}")

        return csv_path
    except Exception as e:
        logging.error(f"Error downloading dataset: {e}")
        raise


if __name__ == '__main__':
    download_imdb_dataset()