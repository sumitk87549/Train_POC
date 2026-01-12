#!/usr/bin/env python3
"""
Text Cleaning and Segregation Inference Script
Uses trained Ollama model to clean and segregate book text.
"""

import argparse
import json
import os
import sys
import requests
from pathlib import Path

class TextCleaner:
    """Handles text cleaning using trained Ollama model."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.ollama_base_url = "http://localhost:11434"
    
    def check_ollama_running(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def check_model_available(self) -> bool:
        """Check if the trained model is available."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(self.model_name in model.get('name', '') for model in models)
            return False
        except:
            return False
    
    def clean_text(self, input_text: str) -> str:
        """Clean and segregate the input text using the trained model."""
        prompt = f"""Clean and segregate this book text. Follow these instructions:
1. Remove publisher/printer information, page numbers, copyright notices, ISBN
2. Remove useless characters and fix formatting issues
3. Segregate content into marked sections:
   - ***MAIN_CONTENT*** for story content
   - ***ACKNOWLEDGEMENTS*** for acknowledgements
   - ***GLOSSARY*** for glossary
   - ***APPENDIX*** for appendix
   - ***REFERENCES*** for references
   - ***INDEX*** for index
   - ***PREFACE*** for preface
4. Preserve the actual content while removing metadata
5. Use clear section markers with special characters

Text to clean and segregate:
{input_text}"""
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "max_tokens": 4000
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                print(f"Error: {response.text}")
                return ""
                
        except Exception as e:
            print(f"Error cleaning text: {str(e)}")
            return ""
    
    def clean_file(self, input_file: str, output_file: str):
        """Clean text from file and save to output file."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                input_text = f.read()
            
            print(f"Cleaning text from: {input_file}")
            print(f"Using model: {self.model_name}")
            
            cleaned_text = self.clean_text(input_text)
            
            if cleaned_text:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                print(f"Cleaned text saved to: {output_file}")
                return True
            else:
                print("Failed to clean text")
                return False
                
        except Exception as e:
            print(f"Error processing files: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Clean and segregate book text using trained model")
    parser.add_argument("--model", required=True, help="Trained model name (e.g., deepseek-r1-cleaner)")
    parser.add_argument("-i", "--input", required=True, help="Input text file to clean")
    parser.add_argument("-o", "--output", required=True, help="Output file for cleaned text")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Initialize cleaner
    cleaner = TextCleaner(args.model)
    
    # Check if Ollama is running
    if not cleaner.check_ollama_running():
        print("Error: Ollama is not running. Please start Ollama first.")
        print("To start Ollama: ollama serve")
        sys.exit(1)
    
    # Check if model is available
    if not cleaner.check_model_available():
        print(f"Error: Model {args.model} is not available.")
        print("Available models:")
        try:
            response = requests.get(f"{cleaner.ollama_base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                for model in models:
                    print(f"  - {model.get('name')}")
        except:
            pass
        sys.exit(1)
    
    # Clean the text
    success = cleaner.clean_file(args.input, args.output)
    
    if success:
        print("=" * 50)
        print("TEXT CLEANING COMPLETED SUCCESSFULLY!")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Model: {args.model}")
        print("=" * 50)
    else:
        print("TEXT CLEANING FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
