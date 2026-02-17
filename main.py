"""
Multi-Intent Question Generator — Entry Point (Refactored for Pydantic AI)

Orchestrates the generation pipeline using:
- IntentManager (taxonomy)
- SimilarityChecker (deduplication)
- PromptBuilder (templates)
- Agent (LLM interaction via pydantic-ai)
- MongoStore (persistence)
"""

import sys
import json
import uuid
import logging
import argparse
from pathlib import Path
from datetime import datetime
import random
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.intent_manager import IntentManager
from src.similarity_checker import SimilarityChecker
from src.prompt_builder import PromptBuilder
from src.question_generator import QuestionGenerator
from src.agent import create_agent, PipelineDeps
from src.evaluation_metrics import EvaluationMetrics
from src.persona_manager import PersonaManager


# ── Logging Setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("generation.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def save_results_csv(questions, filepath):
    """Save generated questions to CSV."""
    import pandas as pd
    if not questions:
        return
    
    df = pd.DataFrame(questions)
    # Flatten intents for CSV readability
    df["intent_ids"] = df["intents"].apply(lambda x: [i[0] for i in x])
    df["intent_weights"] = df["intents"].apply(lambda x: [i[1] for i in x])
    
    # Ensure provider columns exist
    if "provider" not in df.columns:
        df["provider"] = "unknown"
    if "model" not in df.columns:
        df["model"] = "unknown"
        
    cols = [
        "question", "intent_ids", "intent_weights", "difficulty", 
        "similarity_score", "provider", "model", "confusion_points"
    ]
    # Keep only columns that exist
    cols = [c for c in cols if c in df.columns]
    
    df[cols].to_csv(filepath, index=False, encoding="utf-8")
    logger.info("Saved %d questions to %s", len(df), filepath)


def save_metrics(metrics, filepath):
    """Save generation metrics to JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Saved metrics to %s", filepath)


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-Intent Question Generator")
    parser.add_argument("--batch-size", type=int, default=10, help="Questions per batch")
    parser.add_argument("--batches", type=int, default=1, help="Number of batches")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--mongo", action="store_true", help="Save to MongoDB")
    parser.add_argument("--dry-run", action="store_true", help="Run a small test generation")
    
    parser.add_argument("--total", type=int, default=None, help="Total number of questions (overrides --batches)")
    parser.add_argument("--strategy", choices=["adaptive", "coverage_based", "random_walk"], default=None, help="Evolution strategy")
    parser.add_argument("--persona", type=str, default=None, help="Context for persona generation (e.g., 'An angry cotton farmer')")
    parser.add_argument("--mix-size", type=int, default=None, help="Number of intents per question (overrides random mix)")
    parser.add_argument("--intents", type=str, default=None, help="Specific intent IDs to check (comma-separated)")
    
    args = parser.parse_args()

    # Calculate batches from total if provided
    if args.total:
        args.batches = (args.total + args.batch_size - 1) // args.batch_size

    # ── 1. Configuration ─────────────────────────────────────────────────
    load_dotenv()
    config = Config()
    
    if args.strategy:
        config.EVOLUTION_STRATEGY = args.strategy
    
    # Ensure output directories exist
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "csv").mkdir(parents=True, exist_ok=True)

    # Dry run overrides
    if args.dry_run:
        logger.info("--- DRY RUN MODE ---")
        args.batch_size = 3
        args.batches = 1
        config.RATE_LIMIT_DELAY = 1.0

    # ── 2. Initialize Components ─────────────────────────────────────────
    logger.info("Initializing components...")

    # Intent Manager
    intent_manager = IntentManager(
        str(config.INTENT_TAXONOMY_PATH), config=config,
    )
    logger.info("IntentManager: %d active intents", len(intent_manager.active_intent_ids))

    # Embedding model
    logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)

    # Similarity Checker
    similarity_checker = SimilarityChecker(config, embedding_model=embedding_model)

    # Prompt Builder
    agent_prompt = config.AGENT_PROMPT_PATH.read_text(encoding="utf-8")
    prompt_builder = PromptBuilder(agent_prompt, config=config)

    # MongoDB (optional)
    mongo = None
    use_mongo = args.mongo or config.USE_MONGO
    if use_mongo:
        from src.mongo_store import MongoStore
        mongo = MongoStore(
            uri=config.MONGO_URI,
            db_name=config.MONGO_DB,
            collection_name=config.MONGO_COLLECTION,
        )
        if not mongo.check_connection():
            logger.warning("MongoDB not available, falling back to CSV only")
            mongo = None
        else:
            logger.info("MongoDB connected: %s", config.MONGO_DB)

    # Persona Manager
    persona_manager = PersonaManager(config)
    
    # Generate/Load Persona
    logger.info("Initializing Persona...")
    current_persona = persona_manager.generate_persona(context=args.persona)
    logger.info("Using Persona: %s (%s, %s)", current_persona.name, current_persona.age, current_persona.region)

    # Pydantic AI Agent & Dependencies
    # dependencies passed to tools via ctx.deps
    deps = PipelineDeps(
        intent_manager=intent_manager,
        similarity_checker=similarity_checker,
        prompt_builder=prompt_builder,
        config=config,
        mongo_store=mongo,
    )

    try:
        agent = create_agent(config)
    except Exception as e:
        logger.error("Failed to create agent: %s", e)
        sys.exit(1)
    
    # Question Generator
    generator = QuestionGenerator(
        intent_manager=intent_manager,
        similarity_checker=similarity_checker,
        prompt_builder=prompt_builder,
        agent=agent,
        deps=deps,
        config=config,
    )

    # Evaluator
    evaluator = EvaluationMetrics(embedding_model=embedding_model)

    # ── 3. Generation Loop ───────────────────────────────────────────────
    
    total_new_questions = []
    start_time = datetime.now()
    batch_id = start_time.strftime("%Y%m%d_%H%M%S")

    logger.info("Starting generation: %d batches of %d", args.batches, args.batch_size)

    try:
        for i in range(args.batches):
            logger.info("Batch %d/%d", i + 1, args.batches)
            
            # Generate
            batch_questions = generator.generate_batch(
                batch_size=args.batch_size,
                difficulty=args.difficulty,
                intent_mix_size=args.mix_size if args.mix_size else random.choice(config.INTENT_MIX_SIZES),
                persona=current_persona,
                target_intents=args.intents.split(",") if args.intents else None,
            )
            
            if batch_questions:
                total_new_questions.extend(batch_questions)
                
                # Save intermediate CSV
                save_results_csv(
                    total_new_questions, 
                    str(config.OUTPUT_DIR / "csv" / f"generated_questions_{batch_id}.csv")
                )
                
                # Save to Mongo immediately if enabled (handled via tool call in agent? 
                # No, agent tool handles it independently if invoked. 
                # But we might want to ensure saving here too if agent didn't save it.)
                # Actually, the agent tool `save_to_mongo` is available for the LLM to call, 
                # but typically question generation logic calls LLM to *get* questions, 
                # then we save them. 
                # The `save_to_mongo` tool is more for "task-based" agents. 
                # Here we are just using the agent to generate text.
                # So we should explicit save here.
                if mongo:
                    mongo.insert_questions(batch_questions, batch_id=batch_id)

            # Evolve intent weights
            if not args.dry_run and config.EVOLUTION_FREQUENCY > 0:
                if (i + 1) % (config.EVOLUTION_FREQUENCY // config.BATCH_SIZE) == 0:
                    intent_manager.evolve_weights(
                        strategy=config.EVOLUTION_STRATEGY
                    )

    except KeyboardInterrupt:
        logger.warning("Generation interrupted by user.")
    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
    finally:
        # ── 4. Finalize & Metrics ────────────────────────────────────────
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("Generation finished. Total questions: %d", len(total_new_questions))
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(total_new_questions)
        metrics["meta"] = {
            "batch_id": batch_id,
            "duration_seconds": duration,
            "batches_attempted": args.batches,
            "difficulty": args.difficulty,
            "timestamp": end_time.isoformat(),
        }
        
        # Save metrics
        metrics_combined = {
            "metrics": metrics,
            "intent_evolution": intent_manager.get_evolution_log(),
            "generator_stats": generator.stats,
        }
        save_metrics(metrics_combined, str(config.OUTPUT_DIR / "generation_metrics.json"))
        save_metrics(
            intent_manager.get_evolution_log(),
            str(config.OUTPUT_DIR / "intent_evolution_log.json"),
        )
        
        logger.info("Done.")

if __name__ == "__main__":
    main()
