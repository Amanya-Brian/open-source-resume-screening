"""MongoDB service for database operations."""

import logging
from datetime import datetime
from typing import Any, Optional, Type, TypeVar

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pydantic import BaseModel
from pymongo import UpdateOne
from pymongo.errors import ConnectionFailure, DuplicateKeyError

from src.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# Collection configurations with indexes
COLLECTION_CONFIGS = {
    "students": {
        "indexes": [
            {"keys": [("email", 1)], "unique": True},
            {"keys": [("skills", 1)]},
            {"keys": [("university", 1), ("graduation_year", -1)]},
        ]
    },
    "job_listings": {
        "indexes": [
            {"keys": [("company", 1)]},
            {"keys": [("posted_at", -1)]},
            {"keys": [("is_active", 1)]},
        ]
    },
    "applications": {
        "indexes": [
            {"keys": [("job_id", 1), ("student_id", 1)], "unique": True},
            {"keys": [("status", 1)]},
            {"keys": [("applied_at", -1)]},
        ]
    },
    "resumes": {
        "indexes": [
            {"keys": [("student_id", 1)], "unique": True},
        ]
    },
    "screening_sessions": {
        "indexes": [
            {"keys": [("job_id", 1)]},
            {"keys": [("created_at", -1)]},
            {"keys": [("status", 1)]},
        ]
    },
    "screening_scores": {
        "indexes": [
            {"keys": [("job_id", 1), ("candidate_id", 1)]},
            {"keys": [("overall_score", -1)]},
        ]
    },
    "rankings": {
        "indexes": [
            {"keys": [("job_id", 1)]},
            {"keys": [("created_at", -1)]},
        ]
    },
    "fairness_reports": {
        "indexes": [
            {"keys": [("job_id", 1)]},
            {"keys": [("session_id", 1)]},
            {"keys": [("is_compliant", 1)]},
        ]
    },
    "explanations": {
        "indexes": [
            {"keys": [("job_id", 1), ("candidate_id", 1)]},
            {"keys": [("generated_at", -1)]},
        ]
    },
    "historical_decisions": {
        "indexes": [
            {"keys": [("job_id", 1)]},
            {"keys": [("was_hired", 1)]},
        ]
    },
    "validation_results": {
        "indexes": [
            {"keys": [("job_id", 1)]},
            {"keys": [("session_id", 1)]},
        ]
    },
    "screening_results": {
        "indexes": [
            {"keys": [("job_id", 1)]},
            {"keys": [("candidate_id", 1)]},
            {"keys": [("total_weighted_score", -1)]},
            {"keys": [("job_id", 1), ("candidate_id", 1)], "unique": True},
        ]
    },
}


class MongoService:
    """Service for MongoDB operations.

    Provides async database operations using Motor driver.
    """

    _instance: Optional["MongoService"] = None

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize MongoDB service.

        Args:
            settings: Application settings. If None, uses default.
        """
        self.settings = settings or get_settings()
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None
        self._initialized = False

    @classmethod
    def get_instance(cls, settings: Optional[Settings] = None) -> "MongoService":
        """Get singleton instance of MongoService.

        Args:
            settings: Application settings

        Returns:
            MongoService instance
        """
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    async def connect(self) -> None:
        """Connect to MongoDB and initialize collections."""
        # Always create a fresh connection to avoid event loop issues
        try:
            self._client = AsyncIOMotorClient(
                self.settings.mongodb_uri,
                serverSelectionTimeoutMS=5000,
            )

            # Test connection
            await self._client.admin.command("ping")

            self._db = self._client[self.settings.mongodb_database]

            # Create indexes (only on first connect)
            if not self._initialized:
                await self._create_indexes()
                self._initialized = True

            logger.info(f"Connected to MongoDB: {self.settings.mongodb_database}")

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._initialized = False
            logger.info("Disconnected from MongoDB")

    async def _create_indexes(self) -> None:
        """Create indexes for all collections."""
        for collection_name, config in COLLECTION_CONFIGS.items():
            collection = self._db[collection_name]
            for index_config in config.get("indexes", []):
                try:
                    await collection.create_index(
                        index_config["keys"],
                        unique=index_config.get("unique", False),
                        background=True,
                    )
                except Exception as e:
                    logger.warning(f"Index creation warning for {collection_name}: {e}")

    @property
    def db(self) -> AsyncIOMotorDatabase:
        """Get database instance."""
        if self._db is None:
            raise RuntimeError("MongoDB not connected. Call connect() first.")
        return self._db

    # === Generic CRUD Operations ===

    async def insert_one(
        self,
        collection: str,
        document: dict[str, Any],
    ) -> str:
        """Insert a single document.

        Args:
            collection: Collection name
            document: Document to insert

        Returns:
            Inserted document ID
        """
        result = await self.db[collection].insert_one(document)
        return str(result.inserted_id)

    async def insert_many(
        self,
        collection: str,
        documents: list[dict[str, Any]],
    ) -> list[str]:
        """Insert multiple documents.

        Args:
            collection: Collection name
            documents: Documents to insert

        Returns:
            List of inserted document IDs
        """
        if not documents:
            return []
        result = await self.db[collection].insert_many(documents)
        return [str(id) for id in result.inserted_ids]

    async def find_one(
        self,
        collection: str,
        query: dict[str, Any],
        projection: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Find a single document.

        Args:
            collection: Collection name
            query: Query filter
            projection: Fields to include/exclude

        Returns:
            Document if found, None otherwise
        """
        return await self.db[collection].find_one(query, projection)

    async def find_many(
        self,
        collection: str,
        query: dict[str, Any],
        projection: Optional[dict[str, Any]] = None,
        sort: Optional[list[tuple[str, int]]] = None,
        limit: int = 0,
        skip: int = 0,
    ) -> list[dict[str, Any]]:
        """Find multiple documents.

        Args:
            collection: Collection name
            query: Query filter
            projection: Fields to include/exclude
            sort: Sort specification
            limit: Maximum documents to return
            skip: Documents to skip

        Returns:
            List of matching documents
        """
        cursor = self.db[collection].find(query, projection)

        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)

        return await cursor.to_list(length=None)

    async def update_one(
        self,
        collection: str,
        query: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
    ) -> bool:
        """Update a single document.

        Args:
            collection: Collection name
            query: Query filter
            update: Update operations
            upsert: Create if not exists

        Returns:
            True if document was modified
        """
        result = await self.db[collection].update_one(query, update, upsert=upsert)
        return result.modified_count > 0 or result.upserted_id is not None

    async def update_many(
        self,
        collection: str,
        query: dict[str, Any],
        update: dict[str, Any],
    ) -> int:
        """Update multiple documents.

        Args:
            collection: Collection name
            query: Query filter
            update: Update operations

        Returns:
            Number of modified documents
        """
        result = await self.db[collection].update_many(query, update)
        return result.modified_count

    async def delete_one(
        self,
        collection: str,
        query: dict[str, Any],
    ) -> bool:
        """Delete a single document.

        Args:
            collection: Collection name
            query: Query filter

        Returns:
            True if document was deleted
        """
        result = await self.db[collection].delete_one(query)
        return result.deleted_count > 0

    async def delete_many(
        self,
        collection: str,
        query: dict[str, Any],
    ) -> int:
        """Delete multiple documents.

        Args:
            collection: Collection name
            query: Query filter

        Returns:
            Number of deleted documents
        """
        result = await self.db[collection].delete_many(query)
        return result.deleted_count

    async def count(
        self,
        collection: str,
        query: Optional[dict[str, Any]] = None,
    ) -> int:
        """Count documents in collection.

        Args:
            collection: Collection name
            query: Optional query filter

        Returns:
            Document count
        """
        return await self.db[collection].count_documents(query or {})

    # === Bulk Operations ===

    async def bulk_upsert(
        self,
        collection: str,
        documents: list[dict[str, Any]],
        id_field: str = "_id",
    ) -> int:
        """Bulk upsert documents.

        Args:
            collection: Collection name
            documents: Documents to upsert
            id_field: Field to use as identifier

        Returns:
            Number of modified/inserted documents
        """
        if not documents:
            return 0

        operations = [
            UpdateOne(
                {id_field: doc.get(id_field)},
                {"$set": doc},
                upsert=True,
            )
            for doc in documents
        ]

        result = await self.db[collection].bulk_write(operations, ordered=False)
        return result.modified_count + result.upserted_count

    # === Model-based Operations ===

    async def save_model(
        self,
        collection: str,
        model: BaseModel,
        id_field: str = "_id",
    ) -> str:
        """Save a Pydantic model to the database.

        Args:
            collection: Collection name
            model: Pydantic model instance
            id_field: Field to use as identifier

        Returns:
            Document ID
        """
        doc = model.model_dump(by_alias=True, exclude_none=True)

        if doc.get(id_field):
            await self.db[collection].update_one(
                {id_field: doc[id_field]},
                {"$set": doc},
                upsert=True,
            )
            return doc[id_field]
        else:
            result = await self.db[collection].insert_one(doc)
            return str(result.inserted_id)

    async def load_model(
        self,
        collection: str,
        model_class: Type[T],
        query: dict[str, Any],
    ) -> Optional[T]:
        """Load a document as a Pydantic model.

        Args:
            collection: Collection name
            model_class: Pydantic model class
            query: Query filter

        Returns:
            Model instance if found, None otherwise
        """
        doc = await self.find_one(collection, query)
        if doc:
            return model_class.model_validate(doc)
        return None

    async def load_models(
        self,
        collection: str,
        model_class: Type[T],
        query: dict[str, Any],
        **kwargs,
    ) -> list[T]:
        """Load multiple documents as Pydantic models.

        Args:
            collection: Collection name
            model_class: Pydantic model class
            query: Query filter
            **kwargs: Additional find_many arguments

        Returns:
            List of model instances
        """
        docs = await self.find_many(collection, query, **kwargs)
        return [model_class.model_validate(doc) for doc in docs]

    # === Specific Collection Operations ===

    async def get_students_for_job(
        self,
        job_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get students who applied for a specific job.

        Args:
            job_id: Job listing ID
            limit: Maximum students to return

        Returns:
            List of student documents with application info
        """
        pipeline = [
            {"$match": {"job_id": job_id}},
            {
                "$lookup": {
                    "from": "students",
                    "localField": "student_id",
                    "foreignField": "_id",
                    "as": "student",
                }
            },
            {"$unwind": "$student"},
            {
                "$lookup": {
                    "from": "resumes",
                    "localField": "student_id",
                    "foreignField": "student_id",
                    "as": "resume",
                }
            },
            {"$unwind": {"path": "$resume", "preserveNullAndEmptyArrays": True}},
            {"$limit": limit},
        ]

        cursor = self.db.applications.aggregate(pipeline)
        return await cursor.to_list(length=None)

    async def save_screening_session(
        self,
        session: dict[str, Any],
    ) -> str:
        """Save a screening session.

        Args:
            session: Session data

        Returns:
            Session ID
        """
        if not session.get("_id"):
            from uuid import uuid4
            session["_id"] = str(uuid4())

        session["updated_at"] = datetime.now()

        await self.db.screening_sessions.update_one(
            {"_id": session["_id"]},
            {"$set": session},
            upsert=True,
        )
        return session["_id"]

    async def get_historical_decisions(
        self,
        job_id: str,
    ) -> list[dict[str, Any]]:
        """Get historical hiring decisions for a job.

        Args:
            job_id: Job listing ID

        Returns:
            List of historical decisions
        """
        return await self.find_many(
            "historical_decisions",
            {"job_id": job_id},
            sort=[("decision_date", -1)],
        )
