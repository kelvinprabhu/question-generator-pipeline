"""
MongoDB Store — persistence layer for generated questions.

Connects to localhost:27017, database 'questions'.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Optional

from pymongo import MongoClient, errors as mongo_errors

logger = logging.getLogger(__name__)


class MongoStore:
    """
    Stores and retrieves generated questions from MongoDB.

    Default connection: mongodb://localhost:27017
    Database: questions
    Collection: generated_questions
    """

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "questions",
        collection_name: str = "generated_questions",
    ):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name

        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        # Create indexes for efficient querying
        self._ensure_indexes()
        logger.info(
            "MongoStore connected: %s / %s.%s",
            uri, db_name, collection_name,
        )

    def _ensure_indexes(self):
        """Create useful indexes."""
        try:
            self.collection.create_index("generated_at")
            self.collection.create_index("difficulty")
            self.collection.create_index("provider")
            self.collection.create_index("batch_id")
            self.collection.create_index("cron_run_id")
            self.collection.create_index([("expected_intents", 1)])
        except Exception as e:
            logger.warning("Could not create indexes: %s", e)

    # ── Insert ───────────────────────────────────────────────────────────

    def insert_questions(
        self,
        questions: List[Dict],
        batch_id: Optional[str] = None,
        cron_run_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> int:
        """
        Insert a list of generated questions into MongoDB.

        Adds metadata fields:  generated_at, batch_id, cron_run_id, provider, model.
        Returns the number of inserted documents.
        """
        if not questions:
            return 0

        batch_id = batch_id or str(uuid.uuid4())
        now = datetime.utcnow()

        docs = []
        for q in questions:
            doc = {
                "question": q.get("question", ""),
                "intents": [
                    [int(iid), float(w)]
                    for iid, w in q.get("intents", [])
                ],
                "expected_intents": q.get("expected_intents", []),
                "difficulty": q.get("difficulty", ""),
                "confusion_points": q.get("confusion_points", []),
                "similarity_score": float(q.get("similarity_score", 0.0)),
                "provider": provider or q.get("provider", "unknown"),
                "model": model or q.get("model", "unknown"),
                "generated_at": now,
                "batch_id": batch_id,
                "cron_run_id": cron_run_id or "",
            }
            docs.append(doc)

        try:
            result = self.collection.insert_many(docs)
            count = len(result.inserted_ids)
            logger.info(
                "Inserted %d questions into MongoDB (batch_id=%s)",
                count, batch_id[:8],
            )
            return count
        except Exception as e:
            logger.error("MongoDB insert error: %s", e)
            return 0

    # ── Query ────────────────────────────────────────────────────────────

    def get_question_count(self) -> int:
        """Total number of questions in the collection."""
        return self.collection.count_documents({})

    def get_questions_by_intent(self, intent_id: int, limit: int = 100) -> List[Dict]:
        """Retrieve questions that contain a specific intent."""
        cursor = self.collection.find(
            {"expected_intents": intent_id},
            {"_id": 0},
        ).sort("generated_at", -1).limit(limit)
        return list(cursor)

    def get_questions_by_difficulty(self, difficulty: str, limit: int = 100) -> List[Dict]:
        """Retrieve questions by difficulty level."""
        cursor = self.collection.find(
            {"difficulty": difficulty},
            {"_id": 0},
        ).sort("generated_at", -1).limit(limit)
        return list(cursor)

    def get_recent_questions(self, limit: int = 50) -> List[Dict]:
        """Retrieve the most recently generated questions."""
        cursor = self.collection.find(
            {}, {"_id": 0}
        ).sort("generated_at", -1).limit(limit)
        return list(cursor)

    def get_all_questions(self) -> List[Dict]:
        """Retrieve all questions (use with caution on large collections)."""
        cursor = self.collection.find({}, {"_id": 0})
        return list(cursor)

    def get_provider_stats(self) -> Dict:
        """Get count of questions grouped by provider."""
        pipeline = [
            {"$group": {"_id": "$provider", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        results = list(self.collection.aggregate(pipeline))
        return {r["_id"]: r["count"] for r in results}

    # ── Health ───────────────────────────────────────────────────────────

    def check_connection(self) -> bool:
        """Verify MongoDB connectivity."""
        try:
            self.client.admin.command("ping")
            count = self.get_question_count()
            logger.info("MongoDB OK: %d questions in collection", count)
            return True
        except mongo_errors.ServerSelectionTimeoutError:
            logger.error("MongoDB connection failed: %s", self.uri)
            return False
        except Exception as e:
            logger.error("MongoDB health check failed: %s", e)
            return False

    def close(self):
        """Close the MongoDB client."""
        self.client.close()
