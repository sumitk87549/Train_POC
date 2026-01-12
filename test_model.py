#!/usr/bin/env python3
"""
Model Testing Script
Tests trained Ollama models for text cleaning and segregation performance.
"""

import argparse
import json
import os
import sys
import requests
from pathlib import Path
import logging
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('testing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ModelTester:
    """Handles testing of trained text cleaning models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.ollama_base_url = "http://localhost:11434"
        self.test_results = []
    
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
    
    def test_model_with_sample(self, sample_text: str, expected_output: str = None) -> dict:
        """Test the model with a sample text and return results."""
        logger.info("Testing model with sample text...")
        
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
{sample_text}"""
        
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
                cleaned_text = result.get('response', '')
                
                # Calculate similarity if expected output is provided
                similarity_score = None
                if expected_output:
                    similarity_score = SequenceMatcher(None, cleaned_text, expected_output).ratio()
                
                test_result = {
                    "input_length": len(sample_text),
                    "output_length": len(cleaned_text),
                    "similarity_score": similarity_score,
                    "input_preview": sample_text[:200] + "..." if len(sample_text) > 200 else sample_text,
                    "output_preview": cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text,
                    "full_output": cleaned_text
                }
                
                return test_result
            else:
                logger.error(f"Model test failed: {response.text}")
                return {"error": response.text}
                
        except Exception as e:
            logger.error(f"Error testing model: {str(e)}")
            return {"error": str(e)}
    
    def run_comprehensive_test(self, test_file: str = None, expected_file: str = None):
        """Run comprehensive tests on the model."""
        logger.info(f"Running comprehensive test for model: {self.model_name}")
        
        # Default test samples if no file provided
        if not test_file:
            test_samples = self.get_default_test_samples()
            results = []
            
            for i, sample in enumerate(test_samples):
                logger.info(f"Running test {i+1}/{len(test_samples)}")
                result = self.test_model_with_sample(sample["input"], sample.get("expected"))
                result["test_name"] = sample["name"]
                results.append(result)
                
        else:
            # Test with provided file
            if not os.path.exists(test_file):
                logger.error(f"Test file not found: {test_file}")
                return
            
            with open(test_file, 'r', encoding='utf-8') as f:
                test_text = f.read()
            
            expected_output = None
            if expected_file and os.path.exists(expected_file):
                with open(expected_file, 'r', encoding='utf-8') as f:
                    expected_output = f.read()
            
            result = self.test_model_with_sample(test_text, expected_output)
            result["test_name"] = f"File test: {test_file}"
            results = [result]
        
        self.test_results = results
        self.display_results()
        self.save_results()
        
        return results
    
    def get_default_test_samples(self) -> list:
        """Get default test samples for comprehensive testing."""
        return [
            {
                "name": "Publisher Info Removal",
                "input": """PUBLISHED BY HARPERCOLLINS PUBLISHERS
1 London Bridge Street
London SE1 9GF

© 2023 by John Doe
All rights reserved.

Page 1

THE GREAT ADVENTURE

Chapter 1: The Beginning

It was a dark and stormy night when Sarah first realized her life would never be the same again.

Page 2

"Where are you going?" asked her friend Mark.""",
                "expected": """***MAIN_CONTENT***

THE GREAT ADVENTURE

Chapter 1: The Beginning

It was a dark and stormy night when Sarah first realized her life would never be the same again.

"Where are you going?" asked her friend Mark."""
            },
            {
                "name": "Section Segregation",
                "input": """Page 3

ACKNOWLEDGEMENTS

I would like to thank my family for their unwavering support during the writing of this book.

Page 4

GLOSSARY

Adventure: An exciting or unusual experience.
Journey: An act of traveling from one place to another.

Page 5

Chapter 2: The Journey Continues

As the sun rose, Sarah and Mark found themselves at the edge of a vast desert.""",
                "expected": """***ACKNOWLEDGEMENTS***

I would like to thank my family for their unwavering support during the writing of this book.

***GLOSSARY***

Adventure: An exciting or unusual experience.
Journey: An act of traveling from one place to another.

***MAIN_CONTENT***

Chapter 2: The Journey Continues

As the sun rose, Sarah and Mark found themselves at the edge of a vast desert."""
            },
            {
                "name": "Complex Text Cleaning",
                "input": """PRINTED BY GLOBAL PRINT SOLUTIONS
123 Printing Avenue
New York, NY 10001

ISBN: 978-1-23456-789-0

Page 10

REFERENCES

Doe, John. (2022). The Art of Storytelling. New York: HarperCollins.
Smith, Jane. (2021). Writing Techniques. London: Penguin Books.

Page 11

INDEX

Adventure, 1-2
Desert, 5-6
Journey, 1-9

Page 12

The desert proved to be more treacherous than they imagined. Sandstorms raged for days."""
            }
        ]
    
    def display_results(self):
        """Display test results in a formatted way."""
        logger.info("=" * 80)
        logger.info("MODEL TESTING RESULTS")
        logger.info("=" * 80)
        
        for i, result in enumerate(self.test_results):
            if "error" in result:
                logger.error(f"Test {i+1}: {result.get('test_name', 'Unknown')} - ERROR: {result['error']}")
                continue
            
            logger.info(f"\nTest {i+1}: {result.get('test_name', 'Unknown')}")
            logger.info("-" * 40)
            logger.info(f"Input length: {result['input_length']} characters")
            logger.info(f"Output length: {result['output_length']} characters")
            
            if result['similarity_score'] is not None:
                logger.info(f"Similarity score: {result['similarity_score']:.2%}")
            
            logger.info(f"Input preview: {result['input_preview']}")
            logger.info(f"Output preview: {result['output_preview']}")
            
            # Check for key cleaning indicators
            output = result.get('full_output', '')
            has_section_markers = '***' in output
            has_page_numbers = 'Page' in output
            has_publisher_info = any(word in output.lower() for word in ['published by', 'printed by', 'isbn'])
            
            logger.info(f"Quality indicators:")
            logger.info(f"  ✓ Has section markers: {has_section_markers}")
            logger.info(f"  ✓ Removed page numbers: {not has_page_numbers}")
            logger.info(f"  ✓ Removed publisher info: {not has_publisher_info}")
        
        logger.info("\n" + "=" * 80)
    
    def save_results(self):
        """Save test results to a file."""
        results_file = f"test_results_{self.model_name.replace(':', '_')}.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "model": self.model_name,
                    "timestamp": str(Path.cwd()),
                    "results": self.test_results
                }, f, indent=2)
            
            logger.info(f"Test results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def interactive_test(self):
        """Run an interactive test session."""
        logger.info("Starting interactive test session...")
        logger.info("Type 'quit' to exit, 'help' for commands")
        
        while True:
            try:
                user_input = input("\nEnter text to test (or command): ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    print("Commands:")
                    print("  help - Show this help")
                    print("  quit - Exit interactive mode")
                    print("  Any other text - Test the model with that text")
                    continue
                elif not user_input:
                    continue
                
                logger.info("Testing with user input...")
                result = self.test_model_with_sample(user_input)
                
                if "error" in result:
                    logger.error(f"Error: {result['error']}")
                else:
                    logger.info("Model output:")
                    print("-" * 40)
                    print(result.get('full_output', 'No output'))
                    print("-" * 40)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {str(e)}")
        
        logger.info("Interactive test session ended.")

def main():
    parser = argparse.ArgumentParser(description="Test trained Ollama model for text cleaning")
    parser.add_argument("--model", required=True, help="Trained model name to test")
    parser.add_argument("-i", "--input", help="Input file for testing")
    parser.add_argument("-e", "--expected", help="Expected output file for comparison")
    parser.add_argument("--interactive", action="store_true", help="Run interactive test mode")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive default tests")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ModelTester(args.model)
    
    # Check if Ollama is running
    if not tester.check_ollama_running():
        logger.error("Ollama is not running. Please start Ollama first.")
        logger.info("To start Ollama: ollama serve")
        sys.exit(1)
    
    # Check if model is available
    if not tester.check_model_available():
        logger.error(f"Model {args.model} is not available.")
        logger.info("Available models:")
        try:
            response = requests.get(f"{tester.ollama_base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                for model in models:
                    logger.info(f"  - {model.get('name')}")
        except:
            pass
        sys.exit(1)
    
    # Run tests based on arguments
    if args.interactive:
        tester.interactive_test()
    elif args.comprehensive or not args.input:
        tester.run_comprehensive_test()
    else:
        tester.run_comprehensive_test(args.input, args.expected)
    
    logger.info("Testing completed!")

if __name__ == "__main__":
    main()
