"""
Simple script to test the API endpoints.
Run this script to check if the API is working correctly.
"""
import requests
import json
import os

BASE_URL = 'http://localhost:5000'  # Change if your server runs on a different port

def test_health():
    """Test the health endpoint."""
    try:
        response = requests.get(f'{BASE_URL}/health')
        print(f"Health endpoint: {response.status_code}")
        print(response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
        return False

def test_models():
    """Test the models endpoint."""
    try:
        response = requests.get(f'{BASE_URL}/models')
        print(f"Models endpoint: {response.status_code}")
        models = response.json()
        print(f"Available models: {models}")
        return response.status_code == 200 and len(models) > 0
    except Exception as e:
        print(f"Error testing models endpoint: {e}")
        return False

def test_predict():
    """Test the predict endpoint."""
    try:
        data = {
            'text': 'This is a test text. I am very happy today.',
            'model': 'huggingface'  # Change to a model that's available
        }
        response = requests.post(
            f'{BASE_URL}/predict',
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Predict endpoint: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing predict endpoint: {e}")
        return False

def create_test_file(filename, content_list):
    """Create a test file with the given content."""
    try:
        if filename.endswith('.csv'):
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("text\n")  # Header
                for item in content_list:
                    f.write(f"{item}\n")
        else:
            with open(filename, 'w', encoding='utf-8') as f:
                for item in content_list:
                    f.write(f"{item}\n")
        return True
    except Exception as e:
        print(f"Error creating test file: {e}")
        return False

def test_batch():
    """Test the batch endpoint."""
    try:
        # Create test files
        test_texts = [
            "I am very happy today.",
            "This is a terrible experience.",
            "The product works as expected."
        ]
        
        # Create both CSV and TXT test files
        csv_file = "test_batch.csv"
        txt_file = "test_batch.txt"
        
        if not create_test_file(csv_file, test_texts) or not create_test_file(txt_file, test_texts):
            return False
        
        print("\nTesting batch endpoint with CSV file...")
        # Test with CSV file
        with open(csv_file, 'rb') as f:
            files = {'file': (csv_file, f, 'text/csv')}
            data = {'model': 'huggingface'}
            response = requests.post(
                f'{BASE_URL}/batch',
                files=files,
                data=data
            )
        
        print(f"Batch CSV endpoint: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"CSV Results count: {len(result.get('results', []))}")
            print(f"CSV Download URL: {result.get('download_url')}")
        else:
            print(f"CSV Response error: {response.text}")
        
        print("\nTesting batch endpoint with TXT file...")
        # Test with TXT file
        with open(txt_file, 'rb') as f:
            files = {'file': (txt_file, f, 'text/plain')}
            data = {'model': 'huggingface'}
            response = requests.post(
                f'{BASE_URL}/batch',
                files=files,
                data=data
            )
        
        print(f"Batch TXT endpoint: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"TXT Results count: {len(result.get('results', []))}")
            print(f"TXT Download URL: {result.get('download_url')}")
        else:
            print(f"TXT Response error: {response.text}")
        
        # Clean up test files
        try:
            os.remove(csv_file)
            os.remove(txt_file)
        except Exception as e:
            print(f"Error removing test files: {e}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing batch endpoint: {e}")
        return False

if __name__ == '__main__':
    print("Testing API endpoints...\n")
    
    if test_health():
        print("✓ Health endpoint working correctly.")
    else:
        print("✗ Health endpoint not working.")
        
    if test_models():
        print("✓ Models endpoint working correctly.")
    else:
        print("✗ Models endpoint not working.")
        
    if test_predict():
        print("✓ Predict endpoint working correctly.")
    else:
        print("✗ Predict endpoint not working.")
        
    if test_batch():
        print("✓ Batch endpoint working correctly.")
    else:
        print("✗ Batch endpoint not working.")
