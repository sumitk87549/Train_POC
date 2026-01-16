#!/usr/bin/env python3
"""
Book Text Cleaning and Segregation Model
Trains/tests DeepSeek-R1 model to clean and segregate book sections
"""

import os
import sys
import argparse
import torch
from datetime import datetime
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
from typing import Optional, Dict, List
import re

# Configuration
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TRAINED_MODEL_PATH = "./trained_model"
OUTPUT_BASE_PATH = "./output"

class BookSegregationModel:
    """Handles model loading, training, and inference for book text segregation"""

    def __init__(self, model_name: str = MODEL_ID, verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._print(f"🖥️  Using device: {self.device}")

    def _print(self, message: str):
        """Print verbose messages"""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
            sys.stdout.flush()

    def check_and_load_model(self, use_trained: bool = True) -> bool:
        """Check if model exists locally and load it"""
        self._print("🔍 Checking for model...")

        # Check for trained model first
        if use_trained and os.path.exists(TRAINED_MODEL_PATH):
            self._print(f"✅ Found trained model at {TRAINED_MODEL_PATH}")
            try:
                self._print("📥 Loading trained model...")
                self.tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH)
                self.model = AutoModelForCausalLM.from_pretrained(
                    TRAINED_MODEL_PATH,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                self._print("✅ Trained model loaded successfully!")
                return True
            except Exception as e:
                self._print(f"⚠️  Error loading trained model: {e}")
                self._print("📥 Will download base model instead...")

        # Download base model from HuggingFace
        self._print(f"📥 Downloading model from HuggingFace: {self.model_name}")
        self._print("⏳ This may take a few minutes on first run...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self._print("✅ Model downloaded and loaded successfully!")
            return True
        except Exception as e:
            self._print(f"❌ Error downloading model: {e}")
            return False

    def prepare_training_data(self, training_file: str, input_text: str) -> Dataset:
        """Prepare training dataset from file and input"""
        self._print(f"📚 Preparing training data from: {training_file}")

        # Read training examples
        with open(training_file, 'r', encoding='utf-8') as f:
            training_content = f.read()

        # Create training prompt with readlyte cleaning instructions
        system_prompt = """You are a specialized text processing model trained for 'readlyte' cleaning and segregation. Your task is to:
1. Clean extracted text from EPUB/PDF files
2. Identify and mark different sections with clear markers
3. Preserve main content exactly as-is
4. Add section markers in format: [SECTION_START:section_name] ... [SECTION_END:section_name]

Common sections to identify:
- ACKNOWLEDGMENT
- FOREWORD
- PREFACE
- TABLE_OF_CONTENTS
- CHAPTER_N (where N is chapter number)
- GLOSSARY
- APPENDIX
- REFERENCES
- BIBLIOGRAPHY
- ABOUT_AUTHOR
- ABOUT_PUBLISHER
- ABOUT_DISTRIBUTOR
- PRINTING_INFO
- COPYRIGHT
- INDEX

Guidelines:
- Keep text clean and readable
- Remove unnecessary formatting artifacts
- Preserve paragraph structure
- Add clear section boundaries
- Main content stays unchanged
"""

        # Create training examples
        training_examples = []

        # Example format: input -> expected output
        example = {
            "input": f"{system_prompt}\n\nProcess this text:\n{input_text}",
            "output": training_content
        }
        training_examples.append(example)

        # Create formatted training texts
        texts = []
        for ex in training_examples:
            text = f"<|user|>\n{ex['input']}\n<|assistant|>\n{ex['output']}<|end|>"
            texts.append(text)

        self._print(f"✅ Created {len(texts)} training examples")

        # Tokenize
        self._print("🔤 Tokenizing training data...")
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=2048,
            return_tensors="pt"
        )

        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"]
        })

        self._print(f"✅ Training dataset prepared with {len(dataset)} samples")
        return dataset

    def train_model(self, training_file: str, input_text: str):
        """Fine-tune the model on training data"""
        self._print("🎓 Starting model training (fine-tuning)...")

        # Prepare dataset
        train_dataset = self.prepare_training_data(training_file, input_text)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=TRAINED_MODEL_PATH,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            fp16=self.device == "cuda",
            report_to="none",
            remove_unused_columns=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Create trainer
        self._print("⚙️  Configuring trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # Train
        self._print("🚀 Training started...")
        self._print("📊 Progress will be shown below:")
        print("-" * 80)

        trainer.train()

        print("-" * 80)
        self._print("✅ Training completed!")

        # Save model
        self._print(f"💾 Saving trained model to {TRAINED_MODEL_PATH}...")
        trainer.save_model(TRAINED_MODEL_PATH)
        self.tokenizer.save_pretrained(TRAINED_MODEL_PATH)

        self._print("✅ Model saved successfully!")

    def process_text(self, input_text: str, max_length: int = 2048) -> str:
        """Process text with the trained model"""
        self._print("🤖 Processing text with model...")

        system_prompt = """You are a specialized text processing model trained for 'readlyte' cleaning and segregation. Process the following text by:
1. Cleaning extracted text artifacts
2. Identifying and marking sections with [SECTION_START:name] and [SECTION_END:name]
3. Preserving main content exactly
4. Adding clear section boundaries

Process this text:
"""

        # Prepare input
        prompt = f"<|user|>\n{system_prompt}{input_text}\n<|assistant|>\n"

        self._print("🔤 Tokenizing input...")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self._print("💭 Generating output (model thinking process)...")
        self._print("-" * 80)

        # Generate with streaming
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "<|assistant|>" in generated_text:
            result = generated_text.split("<|assistant|>")[-1].strip()
        else:
            result = generated_text

        print("-" * 80)
        self._print("✅ Text processing completed!")

        # Show preview of thinking process
        self._print("\n📝 Model Output Preview:")
        preview = result[:500] + "..." if len(result) > 500 else result
        print(preview)

        return result

    def save_output(self, content: str, output_file: Optional[str], book_name: str = "book"):
        """Save processed content to output file"""
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(OUTPUT_BASE_PATH) / f"{book_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine output filename
        if output_file:
            output_path = output_dir / output_file
        else:
            output_path = output_dir / "processed_output.txt"

        self._print(f"💾 Saving output to: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        self._print(f"✅ Output saved successfully!")
        self._print(f"📁 Output location: {output_path}")

        # Also save section markers guide
        guide_path = output_dir / "section_markers_guide.txt"
        guide_content = """Section Markers Guide
=====================

This file contains section markers that can be used to split the book into separate files.

Format: [SECTION_START:name] ... [SECTION_END:name]

You can use a simple script to split this file based on these markers.

Example Python script to split:
--------------------------------
import re

with open('processed_output.txt', 'r') as f:
    content = f.read()

sections = re.findall(r'\[SECTION_START:(\w+)\](.*?)\[SECTION_END:\1\]', content, re.DOTALL)

for section_name, section_content in sections:
    with open(f'{section_name}.txt', 'w') as f:
        f.write(section_content.strip())
"""
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)

        self._print(f"📖 Section markers guide saved to: {guide_path}")

        return output_path


def main():
    """Main function to handle CLI arguments"""
    parser = argparse.ArgumentParser(
        description="Book Text Cleaning and Segregation using DeepSeek-R1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training mode
  python script.py --train -i input.txt -o training_data.txt

  # Testing mode
  python script.py --test -i book_text.txt -o cleaned_book.txt

  # Custom model and book name
  python script.py --test -i mybook.txt --model_name custom_model --book_name "My Novel"
        """
    )

    parser.add_argument('--train', action='store_true', help='Training mode')
    parser.add_argument('--test', action='store_true', help='Testing mode')
    parser.add_argument('-i', '--input', required=True, help='Input text file')
    parser.add_argument('-o', '--output_file', help='Output file name (optional)')
    parser.add_argument('--model_name', default=MODEL_ID, help='Model name or path')
    parser.add_argument('--book_name', default='book', help='Book name for output folder')

    args = parser.parse_args()

    # Validate arguments
    if not (args.train or args.test):
        print("❌ Error: Must specify either --train or --test mode")
        parser.print_help()
        sys.exit(1)

    if args.train and args.test:
        print("❌ Error: Cannot use both --train and --test simultaneously")
        sys.exit(1)

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)

    # Read input
    print("\n" + "="*80)
    print("📚 BOOK TEXT CLEANING AND SEGREGATION MODEL")
    print("="*80 + "\n")

    with open(args.input, 'r', encoding='utf-8') as f:
        input_text = f.read()

    print(f"📄 Input file: {args.input}")
    print(f"📏 Input size: {len(input_text)} characters")
    print(f"🎯 Mode: {'Training' if args.train else 'Testing'}")
    print("\n" + "-"*80 + "\n")

    # Initialize model
    model_handler = BookSegregationModel(model_name=args.model_name, verbose=True)

    # Training mode
    if args.train:
        if not args.output_file:
            print("❌ Error: --output_file (-o) required for training mode")
            sys.exit(1)

        if not os.path.exists(args.output_file):
            print(f"❌ Error: Training data file not found: {args.output_file}")
            sys.exit(1)

        # Load base model
        if not model_handler.check_and_load_model(use_trained=False):
            print("❌ Failed to load model")
            sys.exit(1)

        # Train
        model_handler.train_model(args.output_file, input_text)

    # Testing mode
    elif args.test:
        # Load trained model (or base if not available)
        if not model_handler.check_and_load_model(use_trained=True):
            print("❌ Failed to load model")
            sys.exit(1)

        # Process text
        processed_text = model_handler.process_text(input_text)

        # Save output
        output_path = model_handler.save_output(
            processed_text,
            args.output_file,
            args.book_name
        )

    print("\n" + "="*80)
    print("✅ PROCESS COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()