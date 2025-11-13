# Intent Distillation Tool

> A knowledge distillation tool & data-generator for intent classification. Core idea derived from [EasyDataSet](https://github.com/ConardLi/easy-dataset).

Generate thousands of high-quality training samples for intent classification models through automated knowledge distillation - no manual data collection needed.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export DEEPSEEK_API_KEY="your_key_here"

# Generate complete intent classification dataset
python cli.py distill-auto \
  --topic "Customer Support" \
  --levels 3 \
  --tags-per-level 5 \
  --questions-per-tag 20 \
  --output dataset.jsonl

# Export for training
python cli.py export \
  -i dataset.jsonl \
  -o training.json \
  --format alpaca
```

**Result**: 2,500 training samples ready for fine-tuning your SLM!

## What It Does

This tool uses **knowledge distillation** to automatically generate training datasets for intent classification:

```
Root Topic (e.g., "Customer Support")
    ↓
[Tag Distiller] → Build Intent Taxonomy Tree
    ↓
[Question Distiller] → Generate Diverse Questions per Intent
    ↓
Training Dataset (thousands of labeled samples)
```

### Example Output

**Input**: Topic "Customer Support"

**Stage 1 - Intent Taxonomy**:
```
Customer Support
├── 1 Technical Support
│   ├── 1.1 Troubleshooting Assistance
│   ├── 1.2 Installation & Setup
│   └── 1.3 Feature Usage Guidance
├── 2 Account Management
│   ├── 2.1 Password Reset
│   ├── 2.2 Profile Updates
│   └── 2.3 Subscription Management
└── 3 Billing & Payments
    ├── 3.1 Payment Issues
    └── 3.2 Refund Requests
```

**Stage 2 - Generated Questions** (for intent "2.1 Password Reset"):
```
- "How do I reset my password?"
- "I forgot my password, can you help?"
- "Password reset not working"
- "Send me a password recovery link"
... (×20 per intent)
```

**Final Output** (JSONL format):
```json
{"question": "How do I reset my password?", "intent": "Password Reset", "intent_path": "Customer Support -> Account Management -> Password Reset", "intent_hierarchy": ["Customer Support", "Account Management", "Password Reset"]}
{"question": "I forgot my password, can you help?", "intent": "Password Reset", ...}
```

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/intent-distillation-tool.git
cd intent-distillation-tool

# Install dependencies
pip install -r requirements.txt
# or using poetry
poetry install

# Configure API keys
cp config.yaml.example config.yaml
# Edit config.yaml with your API keys
```

### Configuration

Edit `config.yaml`:

```yaml
llm:
  deepseek:
    api_key: "${DEEPSEEK_API_KEY}"
    base_url: "https://api.deepseek.com/v1"
    model: "deepseek-chat"

  openrouter:
    api_key: "${OPENROUTER_API_KEY}"
    base_url: "https://openrouter.ai/api/v1"
    model: "opengvlab/internvl3-78b"
```

Or set environment variables:
```bash
export DEEPSEEK_API_KEY="your_key_here"
export OPENROUTER_API_KEY="your_key_here"
```

## CLI Commands

### 1. Auto-Distillation (Recommended)

Fully automated end-to-end distillation:

```bash
python cli.py distill-auto \
  --topic "E-commerce Customer Service" \
  --levels 3 \
  --tags-per-level 5 \
  --questions-per-tag 25 \
  --output ecommerce_dataset.jsonl
```

**Parameters**:
- `--topic`: Root topic/domain
- `--levels`: Taxonomy depth (recommended: 2-3)
- `--tags-per-level`: Intents per level (recommended: 4-8)
- `--questions-per-tag`: Questions per intent (recommended: 20-50)
- `--leaf-only`: Generate questions only for leaf intents (default: true)
- `--language`: Language (en/zh, default: en)
- `--model`: LLM model (deepseek/openrouter, default: deepseek)
- `--output`: Output file path
- `--export-taxonomy`: Save taxonomy tree to file (optional)

**Output**:
- ~125 intent categories (5³ tree structure)
- ~3,125 training samples (125 × 25 questions)

### 2. Manual Tag Distillation

Generate sub-intents for a parent intent:

```bash
python cli.py distill-tags \
  --parent "Customer Support" \
  --count 10 \
  --output intents.json
```

**Parameters**:
- `--parent`: Parent intent name
- `--count`: Number of sub-intents to generate
- `--intent-path`: Full intent path (optional)
- `--existing`: Existing sibling intents to avoid (multiple)
- `--output`: Output file path
- `--language`: Language (en/zh)
- `--model`: LLM model to use

### 3. Manual Question Distillation

Generate questions for a specific intent:

```bash
python cli.py distill-questions \
  --intent "Password Reset" \
  --intent-path "Account -> Password Reset" \
  --count 50 \
  --output questions.jsonl
```

**Parameters**:
- `--intent`: Intent name
- `--intent-path`: Full intent path
- `--count`: Number of questions to generate
- `--existing`: File with existing questions to avoid duplicates
- `--output`: Output file path
- `--language`: Language (en/zh)
- `--model`: LLM model to use

### 4. Export for Training

Convert distillation results to training formats:

```bash
# Alpaca format
python cli.py export \
  -i dataset.jsonl \
  -o training.json \
  --format alpaca

# ShareGPT format
python cli.py export \
  -i dataset.jsonl \
  -o training_sharegpt.json \
  --format sharegpt

# With train/test split
python cli.py export \
  -i dataset.jsonl \
  -o training.json \
  --format alpaca \
  --split 0.8
```

**Export Formats**:
- `alpaca`: Alpaca instruction format
- `sharegpt`: ShareGPT conversation format
- `json`: Raw JSON
- `jsonl`: JSON Lines
- `csv`: CSV format

**Options**:
- `--split`: Train/test split ratio (e.g., 0.8)
- `--system-prompt`: Custom system prompt

## Output Formats

### Distillation Output (JSONL)

```json
{
  "question": "How do I reset my password?",
  "intent": "Password Reset",
  "intent_number": "2.1",
  "intent_full_name": "2.1 Password Reset",
  "intent_path": "Customer Support -> Account Management -> Password Reset",
  "intent_numbered_path": "Customer Support -> 2 Account Management -> 2.1 Password Reset",
  "intent_hierarchy": ["Customer Support", "Account Management", "Password Reset"],
  "question_index": 1,
  "timestamp": "2025-11-13T14:20:26.756763Z"
}
```

### Alpaca Format

```json
{
  "instruction": "Classify the intent of the following user query.",
  "input": "How do I reset my password?",
  "output": "Password Reset"
}
```

### ShareGPT Format

```json
{
  "messages": [
    {
      "role": "user",
      "content": "How do I reset my password?"
    },
    {
      "role": "assistant",
      "content": "Password Reset"
    }
  ]
}
```

## Programmatic Usage

```python
from src.llm.client import LLMClient
from src.distillers.intent_tag_distiller import IntentTagDistiller
from src.distillers.intent_question_distiller import IntentQuestionDistiller
from src.exporters.dataset_exporter import DatasetExporter
from src.utils.config_loader import load_config

# Load configuration
config = load_config("config.yaml")
llm_client = LLMClient(config["llm"]["deepseek"])

# Stage 1: Build intent taxonomy
tag_distiller = IntentTagDistiller(llm_client, language="en")
root = tag_distiller.build_taxonomy(
    root_topic="Customer Support",
    levels=3,
    tags_per_level=5
)

# Stage 2: Generate questions
question_distiller = IntentQuestionDistiller(llm_client, language="en")
leaf_intents = tag_distiller.get_leaf_intents(root)

all_questions = []
for intent_node in leaf_intents:
    questions = question_distiller.distill_questions(
        intent_node=intent_node,
        count=20
    )
    all_questions.extend(questions)

# Stage 3: Export
DatasetExporter.export(
    all_questions,
    "dataset.jsonl",
    "jsonl"
)

# Export to Alpaca format
DatasetExporter.export(
    all_questions,
    "training.json",
    "alpaca",
    system_prompt="You are an intent classification assistant."
)
```

## Architecture

### Core Components

1. **IntentTagDistiller** (`src/distillers/intent_tag_distiller.py`)
   - Builds hierarchical intent taxonomies
   - Iterative distillation with automatic numbering
   - Tree structure with parent-child relationships

2. **IntentQuestionDistiller** (`src/distillers/intent_question_distiller.py`)
   - Generates diverse questions for each intent
   - Multiple expression styles and difficulty levels
   - Metadata enrichment (path, hierarchy, etc.)

3. **LLMClient** (`src/llm/client.py`)
   - Unified client for DeepSeek/OpenRouter APIs
   - JSON response parsing with retry logic
   - OpenAI-compatible interface

4. **DatasetExporter** (`src/exporters/dataset_exporter.py`)
   - Multiple format support (Alpaca, ShareGPT, JSON, JSONL, CSV)
   - Train/test splitting
   - Custom system prompts

### Distillation Workflow

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT: Root Topic                     │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│           STAGE 1: Intent Tag Distillation              │
│                  (IntentTagDistiller)                    │
│  • Generate sub-intents level by level                  │
│  • Build hierarchical tree (1, 1.1, 1.1.1, ...)         │
│  • Avoid duplicates                                      │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              STAGE 2: Question Distillation             │
│              (IntentQuestionDistiller)                   │
│  • Generate diverse questions per intent                │
│  • Multiple expression styles                           │
│  • Enrich with metadata (path, hierarchy)               │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│             STAGE 3: Export to Training Format           │
│                  (DatasetExporter)                       │
│  • Alpaca / ShareGPT / Custom formats                   │
│  • Train/test splitting                                  │
└─────────────────────────────────────────────────────────┘
```

## Methodology: Based on EasyDataSet

This tool adapts the **knowledge distillation methodology** from [EasyDataSet](https://github.com/ConardLi/easy-dataset):

| EasyDataSet | Intent Distillation Tool |
|-------------|--------------------------|
| Document Chunks | Root Topic |
| Tag Distillation | Intent Tag Distillation |
| Question Generation | Question Distillation |
| QA Pairs | Intent Classification Dataset |

**Core Principles**:
- ✅ Hierarchical taxonomy building
- ✅ Iterative distillation with LLMs
- ✅ Diverse question generation
- ✅ Automated end-to-end workflow

**Key Difference**:
- EasyDataSet: Documents → Knowledge extraction
- This tool: Topic → Intent classification data generation

## Use Cases

### 1. Train Intent Classification Models

```bash
python cli.py distill-auto \
  --topic "Customer Support" \
  --levels 3 \
  --tags-per-level 5 \
  --questions-per-tag 30 \
  --output support_dataset.jsonl

python cli.py export \
  -i support_dataset.jsonl \
  -o training.json \
  --format alpaca \
  --split 0.8
```

Fine-tune your SLM (e.g., Llama 3, Qwen) on the generated training data.

### 2. Task Routing for Agent Systems

Generate training data for routing user queries to specialized agents:

```bash
python cli.py distill-auto \
  --topic "AI Agent Capabilities" \
  --levels 2 \
  --tags-per-level 8 \
  --questions-per-tag 25 \
  --output agent_routing.jsonl
```

### 3. Multi-Domain Intent Classification

```bash
# E-commerce
python cli.py distill-auto --topic "E-commerce" --output ecommerce.jsonl

# Healthcare
python cli.py distill-auto --topic "Healthcare Support" --output healthcare.jsonl

# Finance
python cli.py distill-auto --topic "Banking Services" --output banking.jsonl

# Merge datasets
cat ecommerce.jsonl healthcare.jsonl banking.jsonl > multi_domain.jsonl
```

### 4. Bilingual Dataset Generation

```bash
# English
python cli.py distill-auto --topic "Customer Service" --language en --output en_dataset.jsonl

# Chinese
python cli.py distill-auto --topic "客户服务" --language zh --output zh_dataset.jsonl
```

## Performance Tips

1. **Optimal Parameters**:
   - Levels: 2-3 (deeper = more specific but slower)
   - Tags per level: 4-8 (balanced coverage)
   - Questions per tag: 20-50 (enough diversity)

2. **Cost Optimization**:
   - Use `--leaf-only` to generate questions only for leaf intents
   - Start with small taxonomy (levels=2, tags=3) to test
   - DeepSeek is more cost-effective than OpenRouter

3. **Quality Improvement**:
   - Use specific, well-defined topics
   - Increase `questions-per-tag` for more diversity
   - Review and filter generated questions

4. **Speed**:
   - Parallel processing is automatic
   - Typical speed: ~5-10 intents/minute, ~20-30 questions/minute
   - Full 3-level taxonomy (125 intents): ~30-40 minutes

## Troubleshooting

### API Key Issues

```bash
# Check if API key is set
echo $DEEPSEEK_API_KEY

# Set temporarily
export DEEPSEEK_API_KEY="your_key_here"

# Set permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export DEEPSEEK_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### Insufficient Balance

Error: `402 - Insufficient Balance`

**Solution**: Top up your DeepSeek or OpenRouter account, or switch models:

```bash
python cli.py distill-auto --model openrouter ...
```

### Rate Limiting

The tool includes automatic retry logic with exponential backoff. If you hit rate limits frequently:

1. Reduce `tags-per-level` or `questions-per-tag`
2. Add delays in `config.yaml`:
   ```yaml
   processing:
     retry_delay: 2.0  # Increase delay
     retry_attempts: 5  # More retries
   ```

### Empty Output

If taxonomy building fails, check:
1. API key is valid
2. Model is accessible
3. Topic is well-defined (not too vague)

## Project Structure

```
intent-distillation-tool/
├── cli.py                              # CLI entry point
├── config.yaml                         # Configuration
├── requirements.txt                    # Dependencies
├── README.md                           # This file
│
├── src/
│   ├── distillers/
│   │   ├── intent_tag_distiller.py     # Build intent taxonomies
│   │   └── intent_question_distiller.py # Generate questions
│   │
│   ├── llm/
│   │   ├── client.py                   # LLM client
│   │   └── prompts/
│   │       ├── distill_intent_tags.py       # Tag generation prompts
│   │       └── distill_intent_questions.py  # Question generation prompts
│   │
│   ├── exporters/
│   │   └── dataset_exporter.py         # Export utilities
│   │
│   └── utils/
│       └── config_loader.py            # Config loader
│
└── output/                             # Generated datasets
```

## Requirements

```
click>=8.1.0
pydantic>=2.0.0
openai>=1.0.0
PyYAML>=6.0
rich>=13.0.0
tenacity>=8.0.0
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Credits

- Core methodology derived from [EasyDataSet](https://github.com/ConardLi/easy-dataset)
- Built with [Click](https://click.palletsprojects.com/), [Rich](https://rich.readthedocs.io/), and [OpenAI](https://platform.openai.com/)

## Support

- Issues: [GitHub Issues](https://github.com/your-org/intent-distillation-tool/issues)
- Discussions: [GitHub Discussions](https://github.com/your-org/intent-distillation-tool/discussions)
