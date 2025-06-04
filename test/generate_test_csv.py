import pandas as pd
import os
import argparse

def create_test_csv(input_csv='IMDB Dataset.csv', output_csv='data/test_data.csv', num_samples=10):
    try:
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input file {input_csv} not found. Please download from Kaggle.")

        print(f"Reading dataset from {input_csv}...")
        df = pd.read_csv(input_csv)

        if 'review' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("Input CSV must have 'review' and 'sentiment' columns.")

        print(f"Selecting {num_samples} truly random samples...")
        test_df = df[['review', 'sentiment']].sample(n=num_samples).reset_index(drop=True)

        test_df = test_df.rename(columns={'review': 'text'})

        test_df['sentiment'] = test_df['sentiment'].str.lower()
        if not test_df['sentiment'].isin(['positive', 'negative']).all():
            raise ValueError("Sentiment column must contain only 'positive' or 'negative' values.")

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        # Lưu file CSV
        test_df.to_csv(output_csv, index=False)
        print(f"Test CSV saved to {output_csv} with {num_samples} samples, including 'text' and 'sentiment' columns.")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a random test CSV from IMDB dataset with sentiment.")
    parser.add_argument('--input', default='IMDB Dataset.csv', help="Path to input IMDB CSV file")
    parser.add_argument('--output', default='data/test_data_with_sentiment.csv', help="Path to output test CSV file")
    parser.add_argument('--num_samples', type=int, default=4, help="Number of samples to select")
    args = parser.parse_args()

    create_test_csv(args.input, args.output, args.num_samples)
