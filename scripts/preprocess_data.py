import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




def clean_text(text):
    """
        Clean text by removing HTML tags, punctuation, and optionally stopwords.

        Args:
            text (str): Input text.
            remove_stopwords (bool): Whether to remove stopwords.

        Returns:
            str: Cleaned text.
    """
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove HTML tags (e.g., <br />)
    text = re.sub(r'<.*?>', '', text)

    # 3. Remove punctuation and numbers
    # Keep only alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # 4. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # 5. Tokenize and remove stopwords, then stem
    words = text.split()
    processed_words = []
    for word in words:
        if word not in stop_words:
            processed_words.append(ps.stem(word))
    return ' '.join(processed_words)


class BatchLoader:
    """
    Batch loader for text data.

    Args:
        data_path (str): Path to CSV file.
        text_column (str): Column name for text.
        label_column (str): Column name for labels (optional).
        batch_size (int): Size of each batch.
    """

    def __init__(self, data_path, text_column='text', label_column=None, batch_size=32):
        self.data_path = data_path
        self.text_column = text_column
        self.label_column = label_column
        self.batch_size = batch_size
        try:
            self.df = pd.read_csv(data_path)
            logging.info(f"Loaded dataset from {data_path}")
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise

    def preprocess(self, output_path=None):
        """
        Preprocess dataset and optionally save to file.

        Args:
            output_path (str): Path to save processed data.

        Returns:
            pd.DataFrame: Preprocessed dataframe.
        """
        try:
            self.df['processed_text'] = self.df[self.text_column].apply(clean_text)
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                self.df.to_csv(output_path, index=False)
                logging.info(f"Preprocessed data saved to {output_path}")
            return self.df
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            raise

    def __iter__(self):
        for i in range(0, len(self.df), self.batch_size):
            batch = self.df.iloc[i:i + self.batch_size]
            texts = batch['processed_text'].tolist()
            labels = batch[self.label_column].tolist() if self.label_column else None
            yield texts, labels


if __name__ == '__main__':
    loader = BatchLoader('data/raw/imdb_dataset.csv', text_column='text', label_column='sentiment', batch_size=32)
    loader.preprocess('data/processed/processed_imdb.csv')
    for texts, labels in loader:
        print(f"Batch: {len(texts)} samples")
        break