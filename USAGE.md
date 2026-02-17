# Question Generation Pipeline - Usage Guide

This project generates multi-intent, confusing questions for agricultural chatbots using a multi-agent system.

## ✨ Features
- **Multi-Provider Support**: Automatically falls back between Groq, Gemini, HuggingFace, and OpenRouter.
- **Persona-Driven**: Generates questions from the perspective of specific farmer personas (e.g., "An angry cotton farmer from Punjab").
- **Intent Mixing**: Combines multiple intents to create challenging test cases.
- **Deduplication**: Checks against existing questions to ensure variety.
- **MongoDB Integration**: Autosaves generated questions.

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd multi-intent-question-generator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file based on `.env.example`:
   ```env
   # API Keys (comma-separated for rotation)
   GROQ_API_KEYS=gsk_...
   GEMINI_API_KEYS=...
   HF_API_KEYS=hf_...
   OPENROUTER_API_KEYS=sk-or-...

   # MongoDB
   MONGO_URI=mongodb://localhost:27017/
   MONGO_DB_NAME=questions
    ```

## 🛠️ Configuration
Edit `src/config.py` to change models or priority:
- **Groq**: `llama-3.3-70b-versatile`
- **Gemini**: `gemini-2.5-flash`
- **OpenRouter**: `google/gemini-2.0-flash-lite-preview-02-05:free`

## 💻 CLI Usage

The main entry point is `main.py`.

### 1. Basic Generation (Random Personas)
Generate 50 questions using random personas:
```bash
python main.py --total 50
```

### 2. Specific Persona
Generate questions from a specific persona's perspective:
```bash
python main.py --total 100 --persona "An angry cotton farmer from Punjab dealing with pink bollworm"
```
The **Persona Agent** will first flesh out this description into a full profile (Name, Age, Backstory) before generating questions.

### 3. Dry Run (Testing)
Test the pipeline without making actual LLM calls (uses mock responses):
```bash
python main.py --dry-run
```

### 4. Adjust Batch Size & Strategy
```bash
python main.py --total 500 --batch-size 10 --strategy coverage_based
```

### 5. Granular Intent Control
Force specific intents or set intent mix size:

**Single Intent Questions:**
```bash
python main.py --total 50 --mix-size 1
```

**Specific Intents:**
Generate questions combining exactly "crop_disease" and "market_prices":
```bash
python main.py --total 20 --intents "crop_disease,market_prices"
```

## 🧩 Architecture

1.  **Persona Agent** (`src/persona_manager.py`): Generates a detailed user profile.
2.  **Question Generator** (`src/question_generator.py`):
    *   Builds a prompt using the mock persona.
    *   Selects a mix of intents (e.g., "crop_disease" + "market_prices").
    *   Calls the LLM (via `src/agent.py`).
    *   Tools: `check_duplicate`, `save_to_mongo`.
3.  **Fallback Mechanism** (`src/agent.py`): Tries Groq → Gemini → HuggingFace → OpenRouter until successful.
