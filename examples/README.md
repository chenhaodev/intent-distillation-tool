# Examples

This directory contains example scripts and data for the Intent Distillation Tool.

## Files

### Scripts

- **`multi-turn-med-dialog.sh`**: Complete pipeline for importing and tagging medical dialogs
  - Imports medical conversations from CSV
  - Generates intent tags using LLM
  - Exports to Alpaca format for training

### Data

- **`medical_dialogues.csv`**: Sample medical doctor-patient conversations from MedVAL-Bench
  - 85 real-world medical consultation dialogues
  - Dialogue2note task format
  - Source: [MedVAL-Bench](https://physionet.org/) dataset

## Running Examples

### Medical Dialog Intent Tagging

Run the complete medical dialog tagging pipeline:

```bash
# Make sure you have set your API key
export DEEPSEEK_API_KEY="your_key_here"

# Run the example script
bash examples/multi-turn-med-dialog.sh
```

This will:
1. Parse medical dialogues from CSV
2. Generate intent tags for each patient turn
3. Export to Alpaca format for training intent classification models

### Output Files

After running `multi-turn-med-dialog.sh`, you'll get:

- `medical_dialogues_tagged.jsonl`: Tagged conversations with intent labels
- `medical_intent_training.json`: Alpaca format training data
- `medical_intent_training_train.json`: Training split (80%)
- `medical_intent_training_test.json`: Test split (20%)

## Data Format

### Input (CSV)
Medical dialogues in MedVAL-Bench format with columns:
- `#`: Record ID
- `id`: Dialogue ID
- `task`: Task type (dialogue2note)
- `input`: Raw dialogue transcript with [doctor]/[patient] markers
- Other metadata columns

### Output (JSONL)
Tagged conversations with structure:
```json
{
  "conversation_id": "medval_1_20251114",
  "domain": "medical",
  "turns": [
    {"role": "user", "content": "...", "intent": "Symptom Reporting", "intent_path": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "primary_intent": "Medical Consultation",
  "all_intents": ["Symptom Reporting", "Treatment Planning", ...]
}
```

### Training Format (Alpaca JSON)
```json
{
  "instruction": "Given the conversation context, classify the intent of the user's current message.",
  "input": "Previous conversation:\n...\n\nCurrent message:\nI have chest pain",
  "output": "Symptom Reporting"
}
```

## Customization

You can modify the scripts to:
- Process different numbers of conversations (`--limit` parameter)
- Use different LLM models (`--model` parameter)
- Change language for intent tags (`--language en` or `zh`)
- Adjust train/test split ratio (`--split 0.8`)
