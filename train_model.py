#!/usr/bin/env python3
"""
Text Cleaning and Segregation Training Script
Trains Ollama models to clean and segregate book text content.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class OllamaTrainer:
    """Handles Ollama model training and inference for text cleaning."""
    
    def __init__(self, model_name: str = "deepseek-r1:1.5b"):
        self.model_name = model_name
        self.ollama_base_url = "http://localhost:11434"
        self.trained_model_path = Path("./trained_models")
        self.trained_model_path.mkdir(exist_ok=True)
        
    def check_ollama_running(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def pull_model(self) -> bool:
        """Pull the specified model from Ollama if not available."""
        logger.info(f"Checking if model {self.model_name} is available...")
        
        try:
            # Check if model exists
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_exists = any(model.get('name') == self.model_name for model in models)
                
                if model_exists:
                    logger.info(f"Model {self.model_name} is already available")
                    return True
            
            # Pull the model
            logger.info(f"Pulling model {self.model_name}...")
            pull_data = {"name": self.model_name}
            response = requests.post(
                f"{self.ollama_base_url}/api/pull",
                json=pull_data,
                stream=True
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if 'status' in data:
                            logger.info(f"Pull status: {data['status']}")
                        if 'error' in data:
                            logger.error(f"Pull error: {data['error']}")
                            return False
                logger.info(f"Successfully pulled model {self.model_name}")
                return True
            else:
                logger.error(f"Failed to pull model: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model: {str(e)}")
            return False
    
    def create_training_dataset(self, extracted_file: str, final_file: str) -> List[Dict]:
        """Create training dataset from extracted and final files."""
        logger.info("Creating training dataset...")
        
        try:
            with open(extracted_file, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
            
            with open(final_file, 'r', encoding='utf-8') as f:
                final_text = f.read()
            
            # Create training examples
            training_examples = []
            
            # Example 1: Full text cleaning and segregation
            example1 = {
                "instruction": """Clean and segregate the following book text. Remove:
- Publisher and printer information
- Page numbers
- Copyright notices
- ISBN information
- Useless characters

Segregate the content into sections marked with special characters:
- ***MAIN_CONTENT*** for the story/main content
- ***ACKNOWLEDGEMENTS*** for acknowledgements
- ***GLOSSARY*** for glossary
- ***APPENDIX*** for appendix
- ***REFERENCES*** for references
- ***INDEX*** for index
- ***PREFACE*** for preface

Only keep the actual content, remove all metadata.""",
                "input": extracted_text,
                "output": final_text
            }
            training_examples.append(example1)
            
            # Example 2: Section-specific cleaning
            chapters = self.extract_chapters(extracted_text)
            for i, chapter in enumerate(chapters[:3]):  # Limit to first 3 chapters
                example = {
                    "instruction": "Clean this chapter content by removing page numbers and formatting issues.",
                    "input": chapter,
                    "output": self.clean_chapter_content(chapter)
                }
                training_examples.append(example)
            
            logger.info(f"Created {len(training_examples)} training examples")
            return training_examples
            
        except Exception as e:
            logger.error(f"Error creating training dataset: {str(e)}")
            return []
    
    def extract_chapters(self, text: str) -> List[str]:
        """Extract individual chapters from text."""
        import re
        chapters = []
        chapter_pattern = r'Chapter \d+:.*?(?=Chapter \d+:|Page \d+|$)'
        matches = re.findall(chapter_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            # Remove page numbers
            clean_chapter = re.sub(r'Page \d+', '', match)
            chapters.append(clean_chapter.strip())
        
        return chapters
    
    def clean_chapter_content(self, chapter: str) -> str:
        """Clean individual chapter content."""
        import re
        # Remove page numbers and extra whitespace
        cleaned = re.sub(r'Page \d+', '', chapter)
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        return cleaned.strip()
    
    def create_modelfile(self, training_examples: List[Dict]) -> str:
        """Create a Modelfile for fine-tuning."""
        logger.info("Creating Modelfile...")
        
        modelfile_path = self.trained_model_path / "Modelfile"
        
        with open(modelfile_path, 'w') as f:
            f.write(f"FROM {self.model_name}\n\n")
            f.write("# Text Cleaning and Segregation Instructions\n")
            f.write('SYSTEM """You are an expert text cleaner and segregator for books. Your task is to:\n')
            f.write('1. Remove publisher/printer information, page numbers, copyright notices, ISBN\n')
            f.write('2. Clean useless characters and formatting issues\n')
            f.write('3. Segregate content into marked sections:\n')
            f.write('   - ***MAIN_CONTENT*** for story content\n')
            f.write('   - ***ACKNOWLEDGEMENTS*** for acknowledgements\n')
            f.write('   - ***GLOSSARY*** for glossary\n')
            f.write('   - ***APPENDIX*** for appendix\n')
            f.write('   - ***REFERENCES*** for references\n')
            f.write('   - ***INDEX*** for index\n')
            f.write('   - ***PREFACE*** for preface\n')
            f.write('4. Preserve the actual content while removing metadata\n')
            f.write('5. Use clear section markers with special characters\n"""\n\n')
            
            # Add parameters for better control
            f.write("# Parameters for the model\n")
            f.write("PARAMETER temperature 0.1\n")
            f.write("PARAMETER top_p 0.9\n")
            f.write("PARAMETER num_ctx 4096\n")
            f.write("PARAMETER stop \"<|im_end|>\"\n")
            f.write("PARAMETER stop \"<|im_start|>\"\n\n")
            
        return str(modelfile_path)
    
    def train_model(self, modelfile_path: str, model_name_suffix: str = "cleaner", training_examples_count: int = 0) -> str:
        """Train the model using the Modelfile."""
        trained_model_name = f"{self.model_name.split(':')[0]}-{model_name_suffix}"
        
        logger.info(f"Training model: {trained_model_name}")
        
        try:
            # Create the model using ollama create with absolute path
            # Use the current working directory to avoid path issues
            cmd = ["ollama", "create", trained_model_name, "-f", str(modelfile_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully created trained model: {trained_model_name}")
                logger.info(f"Model saved in: {self.trained_model_path}")
                
                # Save model info
                model_info = {
                    "base_model": self.model_name,
                    "trained_model": trained_model_name,
                    "modelfile": modelfile_path,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "training_examples_count": training_examples_count
                }
                
                with open(self.trained_model_path / "model_info.json", 'w') as f:
                    json.dump(model_info, f, indent=2)
                
                return trained_model_name
            else:
                logger.error(f"Failed to create model: {result.stderr}")
                logger.error(f"Command: {' '.join(cmd)}")
                return ""
                
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return ""
    
    def test_model(self, trained_model_name: str, test_text: str) -> str:
        """Test the trained model with sample text."""
        logger.info("Testing trained model...")
        
        try:
            test_prompt = f"""Clean and segregate this book text. Remove publisher info, page numbers, and metadata. Mark sections appropriately.

Text to clean:
{test_text[:1000]}..."""

            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": trained_model_name,
                    "prompt": test_prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                cleaned_text = result.get('response', '')
                logger.info("Model test completed successfully")
                return cleaned_text
            else:
                logger.error(f"Model test failed: {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error testing model: {str(e)}")
            return ""
    
    def save_cleaned_output(self, text: str, output_file: str):
        """Save the cleaned and segregated text to file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Cleaned output saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving output: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Train Ollama model for text cleaning and segregation")
    parser.add_argument("--model", default="deepseek-r1:1.5b", help="Base model name from Ollama")
    parser.add_argument("-e", "--extracted", required=True, help="Path to extracted text file")
    parser.add_argument("-f", "--final", required=True, help="Path to final cleaned text file")
    parser.add_argument("--model-suffix", default="cleaner", help="Suffix for trained model name")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.extracted):
        logger.error(f"Extracted file not found: {args.extracted}")
        sys.exit(1)
    
    if not os.path.exists(args.final):
        logger.error(f"Final file not found: {args.final}")
        sys.exit(1)
    
    # Initialize trainer
    trainer = OllamaTrainer(args.model)
    
    # Check if Ollama is running
    if not trainer.check_ollama_running():
        logger.error("Ollama is not running. Please start Ollama first.")
        logger.info("To start Ollama: ollama serve")
        sys.exit(1)
    
    # Pull base model if needed
    if not trainer.pull_model():
        logger.error("Failed to pull base model")
        sys.exit(1)
    
    # Create training dataset
    training_examples = trainer.create_training_dataset(args.extracted, args.final)
    if not training_examples:
        logger.error("Failed to create training dataset")
        sys.exit(1)
    
    # Create Modelfile
    modelfile_path = trainer.create_modelfile(training_examples)
    
    # Train model
    trained_model_name = trainer.train_model(modelfile_path, args.model_suffix, len(training_examples))
    if not trained_model_name:
        logger.error("Failed to train model")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info(f"Base model: {args.model}")
    logger.info(f"Trained model: {trained_model_name}")
    logger.info(f"Model files saved in: {trainer.trained_model_path}")
    logger.info("To test the trained model:")
    logger.info(f"  python test_model.py --model {trained_model_name}")
    logger.info("To use the trained model:")
    logger.info(f"  python clean_text.py --model {trained_model_name} -i input.txt -o output.txt")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
