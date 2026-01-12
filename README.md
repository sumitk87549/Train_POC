# Text Cleaning and Segregation Training System

This system offers **two approaches** for training models to automatically clean and segregate book text content:

1. **True Fine-Tuning** (Recommended) - Actually modifies model weights for real learning
2. **Prompt Engineering** - Uses system prompts (faster but less effective)

## Features

- **True Fine-Tuning**: Modifies model weights using actual training data
- **Continuous Learning**: Can continue training existing models with new data
- **Automatic Text Cleaning**: Removes publisher/printer data, page numbers, copyright notices, ISBN
- **Content Segregation**: Organizes content into marked sections
- **Ollama Integration**: Uses local Ollama models for privacy and control
- **Model Persistence**: Saves trained models locally for reuse

## Prerequisites

1. **Install Ollama**: Follow instructions at https://ollama.ai
2. **Start Ollama**: Run `ollama serve` in terminal
3. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **GPU Recommended**: Fine-tuning works best with CUDA-enabled GPU

## Approaches

### Approach 1: True Fine-Tuning (Recommended)

**Actually modifies model weights** for real learning and better performance.

```bash
# Initial fine-tuning
python fine_tune_model.py --base-model deepseek-coder-1.3b-base -e PROCESSED/extracted_text.txt -f PROCESSED/Cleaned_segregated_final.txt --convert-to-ollama

# Continue training with new data
python fine_tune_model.py --existing-model trained_models/fine_tuned_text-cleaner_final -e new_data.txt -f new_cleaned.txt --convert-to-ollama
```

**Fine-Tuning Options:**
- `--base-model`: Base model from HuggingFace (default: deepseek-coder-1.3b-base)
- `-e, --extracted`: Path to raw extracted text file
- `-f, --final`: Path to cleaned/segregated target file
- `--existing-model`: Path to existing fine-tuned model for continued training
- `--epochs`: Number of training epochs (default: 3)
- `--model-suffix`: Suffix for model name (default: text-cleaner)
- `--convert-to-ollama`: Convert to Ollama format after training

### Approach 2: Prompt Engineering (Legacy)

**Uses system prompts only** - faster but less effective learning.

```bash
python train_model.py --model deepseek-r1:1.5b -e PROCESSED/extracted_text.txt -f PROCESSED/Cleaned_segregated_final.txt
```

## Testing Models

### Test Fine-Tuned Models
```bash
# Test fine-tuned model directly
python test_fine_tuned.py --model trained_models/fine_tuned_text-cleaner_final

# Test with custom input
python test_fine_tuned.py --model trained_models/fine_tuned_text-cleaner_final -i test_input.txt
```

### Test Ollama Models
```bash
# Run comprehensive default tests
python test_model.py --model deepseek-coder-text-cleaner --comprehensive

# Interactive testing mode
python test_model.py --model deepseek-coder-text-cleaner --interactive
```

## Using Trained Models

### Use Fine-Tuned Models
```bash
python clean_text.py --model deepseek-coder-text-cleaner -i new_book_extracted.txt -o cleaned_output.txt
```

### Use Fine-Tuned Models Directly (HuggingFace)
```bash
python test_fine_tuned.py --model trained_models/fine_tuned_text-cleaner_final -i input.txt
```

## Performance Comparison

| Aspect | True Fine-Tuning | Prompt Engineering |
|--------|------------------|-------------------|
| **Learning** | ✅ Actual weight modification | ❌ No weight changes |
| **Consistency** | ✅ Consistent patterns | ⚠️ Varies with prompt |
| **Quality** | 📈 Improves with data | 🔄 Depends on prompt |
| **Reliability** | ✅ Learned behavior | ⚠️ May ignore instructions |
| **Speed** | 🐢 Slower (requires training) | ⚡ Fast (no training) |
| **Memory** | 💾 Higher (model weights) | 💾 Lower (just prompts) |

## Continuous Learning Workflow

1. **Initial Training**: Fine-tune base model with initial dataset
2. **Test Performance**: Verify cleaning quality
3. **Collect New Data**: Add more examples of problematic cases
4. **Continue Training**: Use `--existing-model` to improve further
5. **Repeat**: Continuously improve with more data

```bash
# Step 1: Initial training
python fine_tune_model.py --base-model deepseek-coder-1.3b-base -e data1.txt -f clean1.txt --convert-to-ollama

# Step 2: Test
python test_model.py --model deepseek-coder-text-cleaner --comprehensive

# Step 3: Continue training with new data
python fine_tune_model.py --existing-model trained_models/fine_tuned_text-cleaner_final -e data2.txt -f clean2.txt --convert-to-ollama

# Step 4: Test improved model
python test_model.py --model deepseek-coder-text-cleaner --comprehensive
```

**Options:**
- `--model`: Your trained model name (e.g., deepseek-r1-cleaner)
- `-i, --input`: Input text file to clean
- `-o, --output`: Output file for cleaned text

## Training Process

The training script:

1. **Checks Ollama**: Ensures Ollama is running and accessible
2. **Pulls Base Model**: Downloads the specified model if not available
3. **Creates Training Dataset**: Generates training examples from your input files
4. **Builds Modelfile**: Creates a custom Modelfile with fine-tuning instructions
5. **Trains Model**: Uses `ollama create` to build the custom model
6. **Tests Model**: Validates the trained model with sample text
7. **Saves Model**: Stores model files in `./trained_models/`

## File Structure

```
Train_POC/
├── fine_tune_model.py       # True fine-tuning script (RECOMMENDED)
├── train_model.py           # Legacy prompt engineering script
├── test_model.py            # Model testing script
├── test_fine_tuned.py       # Fine-tuned model testing
├── clean_text.py            # Inference script for cleaning text
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── training.log            # Training logs
├── fine_tuning.log         # Fine-tuning logs
├── testing.log             # Testing logs
├── trained_models/         # Directory for trained models
│   ├── fine_tuned_*/       # Fine-tuned models
│   ├── Modelfile           # Generated modelfile
│   ├── model_info.json     # Model metadata
│   └── training_info.json # Fine-tuning metadata
└── PROCESSED/              # Training data
    ├── extracted_text.txt
    └── Cleaned_segregated_final.txt
```

## Training Data Format

### Input (extracted_text.txt)
Raw text with publisher info, page numbers, and unstructured content.

### Target (Cleaned_segregated_final.txt)
Cleaned and segregated text with section markers:

```
***MAIN_CONTENT***

[Story content here]

***ACKNOWLEDGEMENTS***

[Acknowledgements here]

***GLOSSARY***

[Glossary terms here]

[Other sections...]
```

## Model Management

### List Available Models
```bash
ollama list
```

### Use Trained Model Interactively
```bash
ollama run deepseek-r1-cleaner
```

### Delete Trained Model
```bash
ollama rm deepseek-r1-cleaner
```

## Troubleshooting

### Ollama Not Running
```bash
ollama serve
```

### Model Pull Fails
Check internet connection and model name spelling.

### Training Fails
- Verify input files exist and contain text
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Review `training.log` for detailed error messages

### Poor Cleaning Results
- Ensure training data is high quality and representative
- Try different base models (llama2, mistral, etc.)
- Add more training examples

## Advanced Usage

### Custom Base Models
```bash
python train_model.py --model llama2:7b -e data.txt -f target.txt
```

### Batch Processing
Create a shell script to process multiple files:

```bash
#!/bin/bash
for file in *.txt; do
    python clean_text.py --model deepseek-r1-cleaner -i "$file" -o "cleaned_$file"
done
```

## Logging

Training progress and errors are logged to:
- Console output
- `training.log` file

## Model Persistence

Trained models are saved in:
- Ollama's model directory
- `./trained_models/` with metadata and Modelfile

Models persist across Ollama restarts and can be reused indefinitely.