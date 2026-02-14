"""Batch processing utilities."""

import asyncio
import logging
from typing import Any, Callable, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class BatchProcessor:
    """Utilities for batch processing operations."""

    @staticmethod
    async def process_in_batches(
        items: list[T],
        process_func: Callable[[T], Coroutine[Any, Any, R]],
        batch_size: int = 10,
        max_concurrent: int = 4,
    ) -> list[R]:
        """Process items in batches with concurrency control.

        Args:
            items: Items to process
            process_func: Async function to process each item
            batch_size: Number of items per batch
            max_concurrent: Maximum concurrent operations

        Returns:
            List of results
        """
        if not items:
            return []

        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_limit(item: T) -> R:
            async with semaphore:
                try:
                    return await process_func(item)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    raise

        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            batch_results = await asyncio.gather(
                *[process_with_limit(item) for item in batch],
                return_exceptions=True,
            )

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.warning(f"Item processing failed: {result}")
                else:
                    results.append(result)

        return results

    @staticmethod
    def chunk_list(items: list[T], chunk_size: int) -> list[list[T]]:
        """Split a list into chunks.

        Args:
            items: Items to chunk
            chunk_size: Size of each chunk

        Returns:
            List of chunks
        """
        return [
            items[i:i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]

    @staticmethod
    async def parallel_map(
        items: list[T],
        func: Callable[[T], Coroutine[Any, Any, R]],
        max_concurrent: int = 10,
    ) -> list[R]:
        """Map a function over items with parallel execution.

        Args:
            items: Items to process
            func: Async function to apply
            max_concurrent: Maximum concurrent executions

        Returns:
            List of results in order
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_func(item: T) -> R:
            async with semaphore:
                return await func(item)

        return await asyncio.gather(*[limited_func(item) for item in items])
