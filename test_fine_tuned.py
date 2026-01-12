#!/usr/bin/env python3
"""
Test script for fine-tuned models
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fine_tuned_model(model_path: str, test_text: str):
    """Test a fine-tuned model with sample text."""
    
    logger.info(f"Loading fine-tuned model from: {model_path}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if not torch.cuda.is_available():
            model = model.to("cpu")
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare input
        instruction = "Clean and segregate this book text by removing publisher info, page numbers, and organizing into sections."
        input_text = f"<|instruction|>{instruction}<|input|>{test_text}<|output|>"
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the output part
        if "<|output|>" in generated_text:
            output = generated_text.split("<|output|>")[1].split("<|end|>")[0]
        else:
            output = generated_text
        
        logger.info("Generated output:")
        print("=" * 50)
        print(output)
        print("=" * 50)
        
        return output
        
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned model")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model")
    parser.add_argument("-i", "--input", help="Input text file")
    parser.add_argument("--text", help="Direct text input")
    
    args = parser.parse_args()
    
    # Get test text
    if args.text:
        test_text = args.text
    elif args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            test_text = f.read()
    else:
        # Default test text
        test_text = """PUBLISHED BY HARPERCOLLINS PUBLISHERS
1 London Bridge Street
London SE1 9GF

© 2023 by John Doe
All rights reserved.

Page 1

THE GREAT ADVENTURE

Chapter 1: The Beginning

It was a dark and stormy night when Sarah first realized her life would never be the same again.

Page 2

"Where are you going?" asked her friend Mark."""
    
    test_fine_tuned_model(args.model, test_text)

if __name__ == "__main__":
    main()
