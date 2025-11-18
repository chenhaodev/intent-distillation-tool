#!/bin/bash

# Multi-Turn Medical Dialog Intent Classification Example
# This script demonstrates how to import real-world medical dialogs
# and generate intent tags for training intent classification models

set -e

echo "=============================================="
echo "Medical Dialog Intent Tagging Pipeline"
echo "=============================================="
echo ""

# Configuration
INPUT_CSV="examples/data/medical_dialogues.csv"
OUTPUT_TAGGED="examples/data/medical_dialogues_tagged.jsonl"
OUTPUT_TRAINING="examples/data/medical_intent_training.json"
OUTPUT_TRAINING_SPLIT_TRAIN="examples/data/medical_intent_training_train.json"
OUTPUT_TRAINING_SPLIT_TEST="examples/data/medical_intent_training_test.json"

# Check if input file exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "Error: Input file not found: $INPUT_CSV"
    echo "Please ensure the medical dialogues CSV is in examples/data/"
    exit 1
fi

# Check if API key is set
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "Warning: DEEPSEEK_API_KEY environment variable is not set"
    echo "Please set it with: export DEEPSEEK_API_KEY='your_key_here'"
    exit 1
fi

echo "Step 1: Import medical dialogs and generate intent tags"
echo "Input: $INPUT_CSV"
echo "Output: $OUTPUT_TAGGED"
echo ""
echo "This step performs 3 stages:"
echo "  1. Parse medical dialogues from CSV"
echo "  2. Build domain-specific intent taxonomy from sample conversations"
echo "  3. Generate intent tags for all conversations using the taxonomy"
echo ""

# Process a limited number for demonstration (5 conversations)
# Remove --limit to process all conversations
python cli.py import-medical-dialogs \
  --input "$INPUT_CSV" \
  --output "$OUTPUT_TAGGED" \
  --limit 5 \
  --model deepseek

echo ""
echo "=============================================="
echo "Step 2: Export to Alpaca format for training"
echo "=============================================="
echo ""

# Export to Alpaca format (default for intent classification)
python cli.py export \
  -i "$OUTPUT_TAGGED" \
  -o "$OUTPUT_TRAINING" \
  --format alpaca \
  --mode intent-classification

echo ""
echo "Exported to: $OUTPUT_TRAINING"
echo ""

echo "=============================================="
echo "Step 3 (Optional): Export with train/test split"
echo "=============================================="
echo ""

# Export with 80/20 train/test split
python cli.py export \
  -i "$OUTPUT_TAGGED" \
  -o "$OUTPUT_TRAINING" \
  --format alpaca \
  --mode intent-classification \
  --split 0.8

echo ""
echo "Training set: $OUTPUT_TRAINING_SPLIT_TRAIN"
echo "Test set: $OUTPUT_TRAINING_SPLIT_TEST"
echo ""

echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Files generated:"
echo "  1. Tagged conversations (JSONL): $OUTPUT_TAGGED"
echo "  2. Training data (Alpaca JSON): $OUTPUT_TRAINING"
echo "  3. Training split: $OUTPUT_TRAINING_SPLIT_TRAIN"
echo "  4. Test split: $OUTPUT_TRAINING_SPLIT_TEST"
echo ""
echo "You can now use the Alpaca format files to train your intent classification model!"
echo ""
echo "Example Alpaca format entry:"
echo '{'
echo '  "instruction": "Given the conversation context, classify the intent of the user'\''s current message.",'
echo '  "input": "Previous conversation: [Doctor] How are you?... Current message: I have pain",'
echo '  "output": "Symptom Reporting"'
echo '}'
echo ""
