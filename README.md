# Intent Distillation Tool

> Generate high-quality training data for intent classification models from synthetic questions or real-world conversations.

## Quick Start

```bash
# Install
pip install -r requirements.txt
export DEEPSEEK_API_KEY="your_key_here"

# Generate synthetic questions
python cli.py distill-auto --topic "Customer Support" --output dataset.jsonl

# Import & tag real medical dialogs
bash examples/multi-turn-med-dialog.sh

# Export for training
python cli.py export -i dataset.jsonl -o training.json --format alpaca
```

## Features

- **Synthetic Data Generation**: Auto-generate questions with hierarchical intent taxonomy
- **Real-World Dialog Tagging**: Import and tag multi-turn conversations with LLM-powered intent labels
- **Domain-Specific Taxonomy**: Build custom intent hierarchies from real conversations
- **Multiple Export Formats**: Alpaca, ShareGPT, JSON, JSONL, CSV
- **Bilingual Support**: English and Chinese

## Main Commands

### 1. Generate Synthetic Questions

```bash
python cli.py distill-auto \
  --topic "Customer Support" \
  --levels 3 \
  --tags-per-level 5 \
  --questions-per-tag 20 \
  --output dataset.jsonl
```

### 2. Import & Tag Real-World Dialogs

```bash
python cli.py import-medical-dialogs \
  --input examples/data/medical_dialogues.csv \
  --output tagged.jsonl \
  --limit 10
```

**How it works** (3-stage pipeline):
1. **Parse** - Extract conversations from source format
2. **Build Taxonomy** - LLM analyzes conversations to create domain-specific intent hierarchy
3. **Tag Intents** - Apply taxonomy to label each turn with hierarchical intent tags

**Output**:
```json
{
  "conversation_id": "medval_1",
  "turns": [
    {"role": "assistant", "content": "What brings you in?"},
    {"role": "user", "content": "I have chest pain",
     "intent": "Symptom Reporting",
     "intent_path": "Medical Consultation -> Chief Complaint -> Symptom Reporting"}
  ]
}
```

### 3. Export for Training

```bash
# Intent classification (with conversation context)
python cli.py export -i tagged.jsonl -o training.json --format alpaca

# Full conversations
python cli.py export -i tagged.jsonl -o training.json --format sharegpt

# Train/test split
python cli.py export -i tagged.jsonl -o training.json --format alpaca --split 0.8
```

## Dataset Integration Roadmap

### Current Support
- ✅ **MedVAL-Bench** - Expert-annotated medical dialogues (Physionet)
  - Format: `dialogue2note` CSV with doctor-patient conversations
  - Parser: `src/parsers/medical_dialog_parser.py`

### Planned Datasets

#### Phase 1: Medical Domain
- 🔄 **ACI-Bench** - AI-patient communication benchmark
  - Multi-turn clinical conversations
  - Focus: Patient education, symptom assessment

- 🔄 **MIMIC-IV Discharge Notes** - Clinical notes from ICU patients
  - Format: Unstructured discharge summaries
  - Focus: Clinical reasoning, treatment planning

#### Phase 2: General Dialog
- 📋 **MultiWOZ** - Multi-domain task-oriented dialogs
- 📋 **SGD** - Schema-Guided Dialog dataset
- 📋 **DSTC** - Dialog State Tracking Challenge datasets

### Adding New Datasets

1. Create parser in `src/parsers/` (see `medical_dialog_parser.py` as template)
2. Implement `parse_csv()` or `parse_json()` returning conversation dict
3. Add CLI command in `cli.py`
4. Add example script in `examples/`

**Conversation format**:
```python
{
    "conversation_id": "unique_id",
    "source": "dataset_name",
    "domain": "medical|general|...",
    "turns": [
        {"role": "user|assistant", "content": "..."},
        ...
    ]
}
```

## Configuration

```yaml
# config.yaml
llm:
  deepseek:
    api_key: "${DEEPSEEK_API_KEY}"
    model: "deepseek-chat"
```

## Use Cases

- **Medical AI**: Train intent classifiers on doctor-patient conversations
- **Chatbots**: Generate training data for conversational agents
- **Task Routing**: Classify user queries to route to specialized systems
- **Context-Aware Models**: Train on multi-turn dialogs with conversation history

## Examples

See `examples/` directory:
- `multi-turn-med-dialog.sh` - Complete pipeline for medical dialog tagging
- `example_inputs.txt` - Sample queries for synthetic generation

## License

MIT License

## Credits

- Methodology from [EasyDataSet](https://github.com/ConardLi/easy-dataset)
- Built with Click, Rich, and OpenAI
