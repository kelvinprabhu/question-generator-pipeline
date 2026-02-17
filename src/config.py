"""
Configuration and hyperparameters for the Multi-Intent Question Generator.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _parse_keys(env_var: str) -> list:
    """Parse comma-separated API keys from an env variable."""
    raw = os.getenv(env_var, "")
    return [k.strip() for k in raw.split(",") if k.strip()]


class Config:
    """Central configuration for the question generation pipeline."""

    # ── Paths ───────────────────────────────────────────────────────────
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    INTENT_TAXONOMY_PATH = DATA_DIR / "intents" / "intent_taxonomy.json"
    AGENT_PROMPT_PATH = DATA_DIR / "prompts" / "agent_system_prompt.txt"

    # Existing data (one level up from project root)
    EXISTING_DATA_DIR = PROJECT_ROOT.parent / "csv"
    QUESTIONS_CSV_PATH = EXISTING_DATA_DIR / "questions.csv"
    EMBEDDINGS_CSV_PATH = EXISTING_DATA_DIR / "embeddings.csv"

    # Outputs
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    GENERATED_BATCHES_DIR = OUTPUT_DIR / "generated_batches"
    GENERATED_QUESTIONS_CSV = DATA_DIR / "csv" / "generated_questions.csv"

    # ── Embedding Model ─────────────────────────────────────────────────
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # ── Multi-Provider API Keys ─────────────────────────────────────────
    # Set comma-separated keys in .env, e.g.:  GROQ_API_KEYS=key1,key2,key3
    GROQ_API_KEYS = _parse_keys("GROQ_API_KEYS")
    GEMINI_API_KEYS = _parse_keys("GEMINI_API_KEYS")
    HF_API_KEYS = _parse_keys("HF_API_KEYS")
    OPENROUTER_API_KEYS = _parse_keys("OPENROUTER_API_KEYS")

    # ── Provider Models ─────────────────────────────────────────────────
    GROQ_MODEL = "llama-3.3-70b-versatile"          # Groq model
    GEMINI_MODEL = "gemini-2.5-flash"               # Google Gemini model
    HF_MODEL = "Qwen/Qwen3-32B"  # HuggingFace model
    OPENROUTER_MODEL = "arcee-ai/trinity-large-preview:free"

    # Provider priority order (tried in this sequence)
    PROVIDER_PRIORITY = ["groq", "gemini", "huggingface", "openrouter"]

    # ── Rate Limiting ───────────────────────────────────────────────────
    RATE_LIMIT_COOLDOWN = 60*2          # Seconds to cooldown a key after 429

    # ── Generation Parameters ───────────────────────────────────────────
    BATCH_SIZE = 50
    TOTAL_QUESTIONS = 500
    INTENT_MIX_SIZES = [2, 6]          # Number of intents to mix per question
    DIFFICULTY_LEVELS = ["medium", "hard", "expert"]
    LANGUAGE = "english"              # Output language

    # ── Similarity Thresholds ───────────────────────────────────────────
    DUPLICATE_THRESHOLD = 0.85        # Reject if similarity >= this
    SIMILAR_REFERENCE_THRESHOLD = 0.70  # Retrieve references above this

    # ── Intent Evolution ────────────────────────────────────────────────
    EVOLUTION_FREQUENCY = 50          # Update weights every N questions
    EVOLUTION_STRATEGY = "adaptive"   # 'adaptive' | 'random_walk' | 'coverage_based'

    # Weight evolution bounds
    WEIGHT_DECAY = 0.95               # Decay for overrepresented intents
    WEIGHT_BOOST = 1.1                # Boost for underrepresented intents
    MIN_WEIGHT = 0.05
    MAX_WEIGHT = 0.30

    # ── LLM Generation ──────────────────────────────────────────────────
    MAX_TOKENS = 2048
    TEMPERATURE = 0.5
    MAX_RETRIES = 3                   # Retries per question on failure

    # ── MongoDB ─────────────────────────────────────────────────────────
    USE_MONGO = os.getenv("USE_MONGO", "true").lower() == "true"
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB = "questions"
    MONGO_COLLECTION = "generated_questions"

    # ── Cron / Scheduler ────────────────────────────────────────────────
    CRON_INTERVAL_MINUTES = 20
    CRON_QUESTIONS_PER_RUN = 500

    def __init__(self, **overrides):
        """Allow runtime overrides via keyword arguments."""
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")

        # Ensure output directories exist
        self.GENERATED_BATCHES_DIR.mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "csv").mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        attrs = {k: v for k, v in vars(type(self)).items()
                 if not k.startswith("_") and k.isupper()}
        return f"Config({attrs})"
