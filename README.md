# Intent Distillation Tool

> A knowledge distillation tool & data-generator for intent classification. Core idea derived from [EasyDataSet](https://github.com/ConardLi/easy-dataset).

Generate thousands of high-quality training samples for intent classification models - no manual data collection needed.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set API key
export DEEPSEEK_API_KEY="your_key_here"

# Generate single-turn questions
python cli.py distill-auto \
  --topic "Customer Support" \
  --levels 3 \
  --tags-per-level 5 \
  --questions-per-tag 20 \
  --output dataset.jsonl

# OR generate multi-turn conversations
python cli.py distill-conversations \
  --topic "Customer Support" \
  --levels 2 \
  --tags-per-level 5 \
  --conversations-per-tag 5 \
  --turns-per-conversation 4 \
  --output conversations.jsonl

# Export for training
python cli.py export -i dataset.jsonl -o training.json --format alpaca
```

## Features

- **Single-turn questions**: Generate diverse questions for each intent
- **Multi-turn conversations**: Generate realistic conversations with intent transitions
- **Auto taxonomy building**: Hierarchical intent classification trees
- **Multiple export formats**: Alpaca, ShareGPT, JSON, JSONL, CSV
- **Bilingual support**: English and Chinese
- **Intent transitions**: Natural topic changes in conversations

## Commands

### 1. Single-Turn Questions

Generate individual questions for each intent:

```bash
python cli.py distill-auto \
  --topic "Customer Support" \
  --levels 3 \
  --tags-per-level 5 \
  --questions-per-tag 20 \
  --output dataset.jsonl
```

**Output**: 2,500 labeled questions (125 intents × 20 questions)

### 2. Multi-Turn Conversations (NEW!)

Generate realistic multi-turn conversations with intent transitions:

```bash
python cli.py distill-conversations \
  --topic "Customer Support" \
  --levels 2 \
  --tags-per-level 5 \
  --conversations-per-tag 5 \
  --turns-per-conversation 4 \
  --transition-rate 0.3 \
  --output conversations.jsonl
```

**Key Parameters**:
- `--turns-per-conversation`: Number of back-and-forth exchanges (default: 4)
- `--transition-rate`: Probability of switching to related intent (0-1, default: 0.3)
- `--conversations-per-tag`: How many conversations per intent (default: 5)

**Output**: 125 conversations with natural intent transitions

**Example conversation**:
```
User: How do I reset my password? [Intent: Password Reset]
Assistant: Click 'Forgot Password' on the login page...
User: Is my account secure after reset? [Intent: Account Security] ← transition
Assistant: Yes, we'll send a verification email...
User: Can I enable two-factor authentication? [Intent: 2FA Setup] ← transition
Assistant: Absolutely! Go to Settings > Security...
```

### 3. Export

Export to various training formats:

```bash
# For intent classification (single-turn)
python cli.py export -i dataset.jsonl -o training.json --format alpaca

# For conversational AI (multi-turn)
python cli.py export -i conversations.jsonl -o chatbot.json --format sharegpt

# Intent classification from conversations (with context)
python cli.py export \
  -i conversations.jsonl \
  -o intent_training.json \
  --format alpaca \
  --mode intent-classification

# Train/test split
python cli.py export -i dataset.jsonl -o training.json --format alpaca --split 0.8
```

**Export Modes** (for conversations):
- `intent-classification`: Extract each user turn as separate intent sample with conversation context
- `conversation`: Full conversation as single training sample

**Formats**: `alpaca`, `sharegpt`, `json`, `jsonl`, `csv`

## Configuration

```bash
# Option 1: Environment variable
export DEEPSEEK_API_KEY="your_key_here"

# Option 2: Edit config.yaml
llm:
  deepseek:
    api_key: "${DEEPSEEK_API_KEY}"
    model: "deepseek-chat"
```

## Output Examples

### Single-Turn Question

**Raw Output**:
```json
{
  "question": "How do I reset my password?",
  "intent": "Password Reset",
  "intent_path": "Customer Support -> Account Management -> Password Reset"
}
```

**Alpaca Format**:
```json
{
  "instruction": "Classify the intent of the following user query.",
  "input": "How do I reset my password?",
  "output": "Password Reset"
}
```

### Multi-Turn Conversation

**Raw Output**:
```json
{
  "conversation_id": "conv_20251113_1234",
  "primary_intent": "Password Reset",
  "all_intents": ["Password Reset", "Account Security"],
  "turns": [
    {"role": "user", "content": "How do I reset my password?", "intent": "Password Reset"},
    {"role": "assistant", "content": "Click 'Forgot Password' on the login page..."},
    {"role": "user", "content": "Is my account secure?", "intent": "Account Security"},
    {"role": "assistant", "content": "Yes, we'll send verification..."}
  ],
  "transition_points": [3]
}
```

**ShareGPT Format**:
```json
{
  "messages": [
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "Click 'Forgot Password'..."},
    {"role": "user", "content": "Is my account secure?"},
    {"role": "assistant", "content": "Yes, we'll send verification..."}
  ]
}
```

**Alpaca Format (intent-classification mode)**:
```json
{
  "instruction": "Given the conversation context, classify the intent of the user's current message.",
  "input": "Previous conversation:\nUser: How do I reset my password?\nAssistant: Click 'Forgot Password'...\n\nCurrent message:\nIs my account secure?",
  "output": "Account Security"
}
```

## Use Cases

- **Intent Classification**: Train SLMs to classify user queries
- **Conversational AI**: Train chatbots with multi-turn dialogue data
- **Task Routing**: Route queries to specialized agents based on intent
- **Context-Aware Classification**: Train models that understand conversation history
- **Multi-Domain**: Generate datasets for multiple domains
- **Bilingual**: Support both English and Chinese

## Troubleshooting

**API Key Error**:
```bash
export DEEPSEEK_API_KEY="your_key_here"
```

**Insufficient Balance**:
- Top up your DeepSeek account, or
- Switch to OpenRouter: `--model openrouter`

**Empty Output**:
- Check API key is valid
- Ensure topic is well-defined

## License

MIT License

## Credits

- Core methodology from [EasyDataSet](https://github.com/ConardLi/easy-dataset)
- Multi-turn conversation prompts adapted from EasyDataSet's multiTurnConversation.js
- Built with Click, Rich, and OpenAI
