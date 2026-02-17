# Multi-Intent Confusing Question Generator

An intelligent question generation pipeline that creates **non-repetitive, multi-intent, confusing test questions** (in English) for evaluating agricultural chatbots.

## Features

- **28-Intent Taxonomy** — Covers weather, prices, cultivation, subsidies, livestock, and more
- **Dynamic Intent Mixing** — Randomly blends 2-3 intents per question with evolving weights
- **Semantic Deduplication** — Uses sentence-transformer embeddings to prevent duplicate questions against existing 10K+ question dataset
- **Difficulty Levels** — Medium, Hard, Expert with escalating confusion techniques
- **Claude API** — Generates questions via Anthropic's Claude model
- **Quality Metrics** — Tracks diversity, intent coverage, and duplication rates

## Quick Start

### 1. Install Dependencies

```bash
cd multi-intent-question-generator
pip install -r requirements.txt
```

### 2. Set API Key

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your-api-key-here
```

### 3. Run

```bash
# Dry run (3 questions)
python main.py --dry-run

# Generate 100 questions
python main.py --total 100

# Full run (500 questions, default)
python main.py

# Custom settings
python main.py --total 200 --batch-size 5 --strategy coverage_based
```

## Project Structure

```
multi-intent-question-generator/
├── data/
│   ├── intents/intent_taxonomy.json     # 28-intent taxonomy
│   ├── prompts/agent_system_prompt.txt  # Chatbot system prompt
│   └── csv/generated_questions.csv      # Output
├── src/
│   ├── config.py                        # Configuration
│   ├── intent_manager.py               # Intent mixing & weight evolution
│   ├── similarity_checker.py           # Deduplication via embeddings
│   ├── prompt_builder.py               # Dynamic prompt construction
│   ├── question_generator.py           # Claude API integration
│   └── evaluation_metrics.py           # Quality scoring
├── outputs/
│   ├── generated_batches/              # Batch checkpoint JSONs
│   ├── generation_metrics.json         # Quality metrics
│   └── intent_evolution_log.json       # Weight evolution history
├── main.py                             # Entry point
├── requirements.txt
└── README.md
```

## Weight Evolution Strategies

| Strategy | Description |
|----------|-------------|
| `adaptive` | Boosts underrepresented intents, dampens overrepresented ones |
| `random_walk` | Applies small random perturbations to weights |
| `coverage_based` | Heavily boosts intents that have never been used |

## Output Format

Generated questions CSV contains:

| Column | Description |
|--------|-------------|
| `question` | The generated English question |
| `intents` | List of `[intent_id, weight]` pairs |
| `expected_intents` | Intent IDs the question targets |
| `difficulty` | medium / hard / expert |
| `confusion_points` | Why this question is confusing for classifiers |
| `similarity_score` | Max similarity to existing questions |
