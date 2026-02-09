#!/usr/bin/env python3
"""
Script to add English variants to LANGUAGE_OPTIONS.
"""

import json
import re

# Read the notebook
with open("Translation_Colab_Generator (1).ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Find and modify the settings cell
for i, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    
    source = "".join(cell["source"])
    
    # Update LANGUAGE_OPTIONS in settings cell
    if "LANGUAGE_OPTIONS = {" in source and '"hi"' in source:
        print(f"Found settings cell at index {i}")
        
        # Replace LANGUAGE_OPTIONS with updated version including English variants
        old_options = '''# TranslateGemma:27b supported Indian languages
# Using ISO 639-1 codes for translategemma compatibility
LANGUAGE_OPTIONS = {
    "Hindi (हिन्दी)": "hi",
    "Bengali (বাংলা)": "bn",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Marathi (मराठी)": "mr",
    "Gujarati (ગુજરાતી)": "gu",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Odia (ଓଡ଼ିଆ)": "or",
    "Urdu (اردو)": "ur",
    "Assamese (অসমীয়া)": "as",
    "Nepali (नेपाली)": "ne",
    "Sindhi (سنڌي)": "sd",
}'''
        
        new_options = '''# TranslateGemma:27b supported languages
# Indian languages + English variants commonly used in India
LANGUAGE_OPTIONS = {
    "Hindi (हिन्दी)": "hi",
    "Bengali (বাংলা)": "bn",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Marathi (मराठी)": "mr",
    "Gujarati (ગુજરાતી)": "gu",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Odia (ଓଡ଼ିଆ)": "or",
    "Urdu (اردو)": "ur",
    "Assamese (অসমীয়া)": "as",
    "Nepali (नेपाली)": "ne",
    "Sindhi (سنڌي)": "sd",
    "English (Indian)": "en-IN",
    "English (US)": "en-US",
    "English (UK)": "en-GB",
    "English (Australia)": "en-AU",
    "English (Canada)": "en-CA",
}'''
        
        source = source.replace(old_options, new_options)
        cell["source"] = source.split("\n")
        cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]
        break

# Update LANG_NAMES in engine cell
for i, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    
    source = "".join(cell["source"])
    
    if "LANG_NAMES = {" in source and "'hi': 'Hindi'" in source:
        print(f"Found LANG_NAMES in cell at index {i}")
        
        old_lang_names = '''LANG_NAMES = {
    'hi': 'Hindi', 'bn': 'Bengali', 'ta': 'Tamil',
    'te': 'Telugu', 'mr': 'Marathi', 'gu': 'Gujarati',
    'kn': 'Kannada', 'ml': 'Malayalam', 'pa': 'Punjabi',
    'or': 'Odia', 'ur': 'Urdu', 'as': 'Assamese',
    'ne': 'Nepali', 'sd': 'Sindhi',
    # Keep NLLB codes for compatibility
    'hin_Deva': 'Hindi', 'ben_Beng': 'Bengali', 'tam_Taml': 'Tamil',
    'tel_Telu': 'Telugu', 'mar_Deva': 'Marathi', 'guj_Gujr': 'Gujarati',
    'kan_Knda': 'Kannada', 'mal_Mlym': 'Malayalam', 'pan_Guru': 'Punjabi',
    'ory_Orya': 'Odia', 'urd_Arab': 'Urdu'
}'''
        
        new_lang_names = '''LANG_NAMES = {
    'hi': 'Hindi', 'bn': 'Bengali', 'ta': 'Tamil',
    'te': 'Telugu', 'mr': 'Marathi', 'gu': 'Gujarati',
    'kn': 'Kannada', 'ml': 'Malayalam', 'pa': 'Punjabi',
    'or': 'Odia', 'ur': 'Urdu', 'as': 'Assamese',
    'ne': 'Nepali', 'sd': 'Sindhi',
    # English variants
    'en-IN': 'Indian English', 'en-US': 'American English',
    'en-GB': 'British English', 'en-AU': 'Australian English',
    'en-CA': 'Canadian English',
    # Keep NLLB codes for compatibility
    'hin_Deva': 'Hindi', 'ben_Beng': 'Bengali', 'tam_Taml': 'Tamil',
    'tel_Telu': 'Telugu', 'mar_Deva': 'Marathi', 'guj_Gujr': 'Gujarati',
    'kan_Knda': 'Kannada', 'mal_Mlym': 'Malayalam', 'pan_Guru': 'Punjabi',
    'ory_Orya': 'Odia', 'urd_Arab': 'Urdu'
}'''
        
        source = source.replace(old_lang_names, new_lang_names)
        cell["source"] = source.split("\n")
        cell["source"] = [line + "\n" for line in cell["source"][:-1]] + [cell["source"][-1]]
        break

# Save the modified notebook
with open("Translation_Colab_Generator (1).ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("\n✅ Added English variants to LANGUAGE_OPTIONS!")
print("   - English (Indian)")
print("   - English (US)")
print("   - English (UK)")
print("   - English (Australia)")
print("   - English (Canada)")
