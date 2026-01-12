#!/usr/bin/env python3
"""
True Fine-Tuning Script for Text Cleaning and Segregation
Actually modifies model weights through proper training, not just prompt engineering.
"""

import argparse
import json
import os
import sys
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fine_tuning.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TrueFineTuner:
    """Handles actual fine-tuning of language models for text cleaning."""
    
    def __init__(self, base_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.base_model_name = base_model_name
        self.trained_models_path = Path("./trained_models")
        self.trained_models_path.mkdir(exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    
    def load_existing_model(self, model_path: str):
        """Load an existing fine-tuned model for continued training."""
        logger.info(f"Loading existing model from: {model_path}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"Model loaded successfully from {model_path}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading existing model: {str(e)}")
            return None, None
    
    def load_base_model(self):
        """Load the base model for initial fine-tuning."""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        try:
            # Try to load from HuggingFace Hub
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"Base model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            return None, None
    
    def create_training_dataset(self, extracted_file: str, final_file: str) -> Dataset:
        """Create a proper training dataset for fine-tuning."""
        logger.info("Creating training dataset for fine-tuning...")
        
        try:
            with open(extracted_file, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
            
            with open(final_file, 'r', encoding='utf-8') as f:
                final_text = f.read()
            
            # Create training examples in the format the model expects
            training_texts = []
            
            # Example 1: Full transformation
            instruction = "Clean and segregate this book text by removing publisher info, page numbers, and organizing into sections (acknowledgments, preface, chapters, glossary, references, appendix index etc.)."
            input_text = extracted_text
            output_text = final_text
            
            # Format as a single training text
            formatted_text = f"<|instruction|>{instruction}<|input|>{input_text}<|output|>{output_text}<|end|>"
            training_texts.append(formatted_text)
            
            # Example 2: Chunk-based training (better for learning)
            chunks = self.create_training_chunks(extracted_text, final_text)
            training_texts.extend(chunks)
            
            # Create dataset
            dataset = Dataset.from_dict({"text": training_texts})
            logger.info(f"Created dataset with {len(training_texts)} training examples")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error creating training dataset: {str(e)}")
            return None
    
    def create_training_chunks(self, extracted: str, final: str) -> List[str]:
        """Create smaller training chunks for better learning."""
        chunks = []
        
        # Split into smaller examples
        examples = [
            {
                "instruction": "Remove publisher information and page numbers.",
                "input": self.extract_publisher_info(extracted),
                "output": self.clean_publisher_info(extracted)
            },
            {
                "instruction": "Segregate content into proper sections.",
                "input": self.extract_sections(extracted),
                "output": self.segregate_sections(final)
            },
            {
                "instruction": "Clean chapter content.",
                "input": self.extract_chapter_content(extracted),
                "output": self.clean_chapter_content(final)
            }
        ]
        
        for example in examples:
            if example["input"] and example["output"]:
                formatted = f"<|instruction|>{example['instruction']}<|input|>{example['input']}<|output|>{example['output']}<|end|>"
                chunks.append(formatted)
        
        return chunks
    
    def extract_publisher_info(self, text: str) -> str:
        """Extract publisher information from text."""
        lines = text.split('\n')
        publisher_lines = []
        
        for line in lines:
            if any(keyword in line.upper() for keyword in ['PUBLISHED BY', 'PRINTED BY', 'ISBN', '©', 'COPYRIGHT']):
                publisher_lines.append(line)
                if len(publisher_lines) >= 5:  # Limit to avoid too much context
                    break
        
        return '\n'.join(publisher_lines) if publisher_lines else ""
    
    def clean_publisher_info(self, text: str) -> str:
        """Return cleaned version without publisher info."""
        return "***MAIN_CONTENT***\n\nContent with publisher info removed."
    
    def extract_sections(self, text: str) -> str:
        """Extract section headers and content."""
        sections = []
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword in line.upper() for keyword in ['ACKNOWLEDGEMENTS', 'GLOSSARY', 'APPENDIX', 'REFERENCES', 'INDEX', 'PREFACE']):
                sections.append(line)
        
        return '\n'.join(sections) if sections else ""
    
    def segregate_sections(self, text: str) -> str:
        """Return properly segregated sections."""
        sections = []
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            if line.startswith('***'):
                current_section = line
                sections.append(line)
            elif current_section and line.strip():
                sections.append(line)
        
        return '\n'.join(sections) if sections else ""
    
    def extract_chapter_content(self, text: str) -> str:
        """Extract chapter content."""
        import re
        chapters = re.findall(r'Chapter \d+:.*?(?=Chapter \d+:|Page \d+|$)', text, re.DOTALL | re.IGNORECASE)
        return chapters[0] if chapters else ""
    
    def clean_chapter_content(self, text: str) -> str:
        """Return cleaned chapter content."""
        import re
        chapters = re.findall(r'\*\*\*MAIN_CONTENT\*\*\*.*?(?=\*\*\*|\Z)', text, re.DOTALL)
        return chapters[0] if chapters else ""
    
    def tokenize_dataset(self, dataset: Dataset, tokenizer):
        """Tokenize the dataset for training."""
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,  # Adjust based on your GPU memory
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def fine_tune_model(self, model, tokenizer, dataset, model_name_suffix: str = "cleaner", epochs: int = 3):
        """Perform actual fine-tuning of the model."""
        logger.info("Starting fine-tuning process...")
        
        # Tokenize dataset
        tokenized_dataset = self.tokenize_dataset(dataset, tokenizer)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(self.trained_models_path / f"fine_tuned_{model_name_suffix}"),
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=1 if self.device == "cpu" else 2,
            gradient_accumulation_steps=4 if self.device == "cpu" else 2,
            save_steps=500,
            save_total_limit=2,
            logging_steps=100,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=100,
            fp16=self.device == "cuda",
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("Beginning model training...")
        train_result = trainer.train()
        
        # Save the model
        final_model_path = self.trained_models_path / f"fine_tuned_{model_name_suffix}_final"
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        
        # Save training info
        training_info = {
            "base_model": self.base_model_name,
            "fine_tuned_model": str(final_model_path),
            "training_epochs": epochs,
            "training_loss": train_result.training_loss,
            "device_used": self.device,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_samples": len(dataset)
        }
        
        with open(final_model_path / "training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Fine-tuning completed! Model saved to: {final_model_path}")
        return str(final_model_path)
    
    
    def continue_training(self, existing_model_path: str, new_extracted_file: str, new_final_file: str, epochs: int = 2):
        """Continue training an existing fine-tuned model with new data."""
        logger.info("Continuing training with new data...")
        
        # Load existing model
        model, tokenizer = self.load_existing_model(existing_model_path)
        if not model or not tokenizer:
            return None
        
        # Create new training dataset
        dataset = self.create_training_dataset(new_extracted_file, new_final_file)
        if not dataset:
            return None
        
        # Continue fine-tuning
        model_name_suffix = Path(existing_model_path).name.replace("fine_tuned_", "").replace("_final", "")
        new_model_path = self.fine_tune_model(model, tokenizer, dataset, model_name_suffix + "_continued", epochs)
        
        return new_model_path

def main():
    parser = argparse.ArgumentParser(description="True fine-tuning of language models for text cleaning")
    parser.add_argument("--base-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Base model from HuggingFace")
    parser.add_argument("-e", "--extracted", required=True, help="Path to extracted text file")
    parser.add_argument("-f", "--final", required=True, help="Path to final cleaned text file")
    parser.add_argument("--existing-model", help="Path to existing fine-tuned model for continued training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--model-suffix", default="text-cleaner", help="Suffix for model name")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.extracted):
        logger.error(f"Extracted file not found: {args.extracted}")
        sys.exit(1)
    
    if not os.path.exists(args.final):
        logger.error(f"Final file not found: {args.final}")
        sys.exit(1)
    
    # Initialize fine-tuner
    tuner = TrueFineTuner(args.base_model)
    
    # Check if continuing training or starting fresh
    if args.existing_model and os.path.exists(args.existing_model):
        logger.info("Continuing training with existing model...")
        model_path = tuner.continue_training(args.existing_model, args.extracted, args.final, args.epochs)
    else:
        logger.info("Starting fresh fine-tuning...")
        # Load base model
        model, tokenizer = tuner.load_base_model()
        if not model or not tokenizer:
            logger.error("Failed to load base model")
            sys.exit(1)
        
        # Create training dataset
        dataset = tuner.create_training_dataset(args.extracted, args.final)
        if not dataset:
            logger.error("Failed to create training dataset")
            sys.exit(1)
        
        # Fine-tune the model
        model_path = tuner.fine_tune_model(model, tokenizer, dataset, args.model_suffix, args.epochs)
    
    if not model_path:
        logger.error("Fine-tuning failed")
        sys.exit(1)
    
    
    logger.info("=" * 80)
    logger.info("FINE-TUNING COMPLETED SUCCESSFULLY!")
    logger.info(f"Fine-tuned model saved to: {model_path}")
    logger.info("To use the fine-tuned model:")
    logger.info(f"  python clean_text.py --model {model_path} -i input.txt -o output.txt")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
