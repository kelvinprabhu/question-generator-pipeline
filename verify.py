"""Quick verification script for the project (Phase 2)."""
import ast
import json
import pathlib
import sys

def check_syntax():
    """Check all Python files parse correctly."""
    files = (
        list(pathlib.Path("src").glob("*.py"))
        + [pathlib.Path("main.py"), pathlib.Path("scheduler.py")]
    )
    ok = True
    for f in sorted(files):
        try:
            ast.parse(f.read_text(encoding="utf-8"))
            print(f"  OK: {f}")
        except SyntaxError as e:
            print(f"  FAIL: {f} -> {e}")
            ok = False
    return ok

def check_taxonomy():
    """Check intent taxonomy loads."""
    path = pathlib.Path("data/intents/intent_taxonomy.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"  Loaded {len(data)} intents from taxonomy")
    return len(data) == 28

def check_core_imports():
    """Check core imports work."""
    sys.path.insert(0, ".")
    from src.config import Config
    from src.intent_manager import IntentManager
    from src.prompt_builder import PromptBuilder
    from src.evaluation_metrics import EvaluationMetrics
    from src.llm_provider import ProviderPool, AnthropicProvider, GroqProvider
    from src.llm_provider import GeminiProvider, HuggingFaceProvider
    print("  All core imports OK")

    config = Config()
    im = IntentManager(str(config.INTENT_TAXONOMY_PATH), config=config)
    print(f"  IntentManager: {len(im.active_intent_ids)} active intents")

    mix = im.sample_intent_mix(n_intents=2)
    print(f"  Sample intent mix: {mix}")

    # Provider pool (won't have keys but should initialise)
    pool = ProviderPool(config)
    print(f"  ProviderPool: {len(pool.keys)} keys registered")
    return True

def check_mongo_import():
    """Check MongoDB module imports."""
    try:
        from src.mongo_store import MongoStore
        print("  MongoStore import OK")
        return True
    except ImportError as e:
        print(f"  MongoStore import FAIL (pymongo missing?): {e}")
        return False

if __name__ == "__main__":
    print("\n=== Syntax Check ===")
    ok1 = check_syntax()

    print("\n=== Taxonomy Check ===")
    ok2 = check_taxonomy()

    print("\n=== Core Import Check ===")
    try:
        ok3 = check_core_imports()
    except Exception as e:
        print(f"  Import error: {e}")
        ok3 = False

    print("\n=== MongoDB Import Check ===")
    try:
        ok4 = check_mongo_import()
    except Exception as e:
        print(f"  Error: {e}")
        ok4 = False

    print("\n=== Result ===")
    all_ok = ok1 and ok2 and ok3 and ok4
    if all_ok:
        print("  ALL CHECKS PASSED")
    else:
        print(f"  syntax={ok1}, taxonomy={ok2}, imports={ok3}, mongo={ok4}")
