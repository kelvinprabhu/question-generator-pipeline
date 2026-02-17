"""
Scheduler logic using pydantic-ai pipeline.

Provides:
- run_question_cron(): General question generation
- run_intent_cron(): Targeted intent generation (mixes)

Uses agent-based QuestionGenerator for robust multi-provider handling.
"""

import sys
import logging
import argparse
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Imports
from src.config import Config
from src.intent_manager import IntentManager
from src.similarity_checker import SimilarityChecker
from src.prompt_builder import PromptBuilder
from src.question_generator import QuestionGenerator
from src.evaluation_metrics import EvaluationMetrics
from src.agent import create_agent, PipelineDeps

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scheduler.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

# State
scheduler = BlockingScheduler()
_components = {}


def init_components():
    """Lazy initialization of components."""
    if _components:
        return _components

    load_dotenv()
    config = Config()

    logger.info("Initializing Scheduler Components...")
    
    # Core logic
    intent_mgr = IntentManager(str(config.INTENT_TAXONOMY_PATH), config=config)
    
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    sim_checker = SimilarityChecker(config, embedding_model=embedding_model)

    prompt_content = config.AGENT_PROMPT_PATH.read_text(encoding="utf-8")
    prompt_builder = PromptBuilder(prompt_content, config=config)

    # Mongo
    mongo = None
    if config.USE_MONGO:
        try:
            from src.mongo_store import MongoStore
            mongo = MongoStore(
                uri=config.MONGO_URI,
                db_name=config.MONGO_DB,
                collection_name=config.MONGO_COLLECTION
            )
            if not mongo.check_connection():
                logger.warning("MongoDB unavailable for scheduler.")
                mongo = None
        except ImportError:
            logger.warning("pymongo not installed, skipping MongoStore.")

    # Agent
    deps = PipelineDeps(
        intent_manager=intent_mgr,
        similarity_checker=sim_checker,
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
        intent_manager=intent_mgr,
        similarity_checker=sim_checker,
        prompt_builder=prompt_builder,
        agent=agent,
        deps=deps,
        config=config,
    )

    evaluator = EvaluationMetrics(embedding_model=embedding_model)

    _components.update({
        "config": config,
        "intent_manager": intent_mgr,
        "similarity_checker": sim_checker,
        "generator": generator,
        "prompt_builder": prompt_builder,
        "evaluator": evaluator,
        "mongo": mongo,
        "agent": agent,
    })
    return _components


def run_question_cron(batch_size=50, difficulty="hard"):
    """
    Main scheduled job: Generate batches of questions.
    """
    logger.info("--- Starting Question Cron ---")
    c = init_components()
    
    try:
        questions = c["generator"].generate_batch(
            batch_size=batch_size,
            difficulty=difficulty
        )
        
        if questions:
            # Save metrics
            metrics = c["evaluator"].calculate_metrics(questions)
            logger.info("Metrics: %s", metrics)
            
            # Save to Mongo (if configured)
            if c["mongo"]:
                count = c["mongo"].insert_questions(questions, batch_id="cron_general")
                logger.info("Saved %d questions to Mongo.", count)
                
    except Exception as e:
        logger.error("Error in Question Cron: %s", e, exc_info=True)
    
    logger.info("--- Question Cron Finished ---")


def run_intent_cron(intent_ids=None, batch_size=20):
    """
    Scheduled job: Target specific intents (or under-represented ones).
    """
    logger.info("--- Starting Intent Cron ---")
    c = init_components()
    
    if not intent_ids:
        # Auto-select under-represented intents logic can go here
        # For now, sample randomly
        pass

    try:
        # Generate using standard batch logic, relying on intent manager sampling
        # TODO: Add specific intent targeting to generator if needed
        questions = c["generator"].generate_batch(
            batch_size=batch_size,
            difficulty="medium", # simpler for targeted learning?
            intent_mix_size=2
        )
        
        if questions and c["mongo"]:
            c["mongo"].insert_questions(questions, batch_id="cron_intent")

    except Exception as e:
        logger.error("Error in Intent Cron: %s", e, exc_info=True)

    logger.info("--- Intent Cron Finished ---")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["questions", "intents", "both"], default="both")
    parser.add_argument("--interval-minutes", type=int, default=20)
    parser.add_argument("--run-once", action="store_true", help="Run immediately and exit")
    args = parser.parse_args()
    
    # Initialize components once logic check
    init_components()
    
    if args.run_once:
        if args.mode in ["questions", "both"]:
            run_question_cron(batch_size=10) # smaller batch for run-once
        if args.mode in ["intents", "both"]:
            run_intent_cron(batch_size=5)
        sys.exit(0)

    # Schedule jobs
    if args.mode in ["questions", "both"]:
        scheduler.add_job(
            run_question_cron, 
            "interval", 
            minutes=args.interval_minutes,
            args=[50, "hard"]
        )
        logger.info("Scheduled Question Cron every %d min", args.interval_minutes)

    if args.mode in ["intents", "both"]:
        # Potentially different interval or same
        scheduler.add_job(
            run_intent_cron, 
            "interval", 
            minutes=args.interval_minutes, 
            args=[None, 20]
        )
        logger.info("Scheduled Intent Cron every %d min", args.interval_minutes)

    try:
        logger.info("Scheduler started via BlockingScheduler. Press Ctrl+C to stop.")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass

if __name__ == "__main__":
    main()
