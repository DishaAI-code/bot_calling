# memory_service.py

import os
import logging
from mem0 import MemoryClient

logger = logging.getLogger("memory-service")

MEM0_API_KEY = "m0-HM4RlfZx44GUMNuF464fKu8pnphtMLlYCL2HWnBz"  # put this in .env

client = MemoryClient(api_key=MEM0_API_KEY)


def store_user_memory(user_id: str, text: str) -> None:
    """
    Store a single user memory (facts, preferences, etc.)
    """
    try:
        logger.info(f"user id is {user_id} and we are under store_user_memory function")
        logger.info(f"text in store_user_memory function is {text}")
        client.add(
            [
                {
                    "role": "user",
                    "content": text
                }
            ],
            user_id=user_id
        )
        logger.info(f"Memory stored for user_id={user_id}")

    except Exception as e:
        logger.error(f" Failed to store memory: {e}")


def get_user_memories(user_id: str) -> list[str]:
    """
    Retrieve ALL stored memories for a user.
    """
    try:
        logger.info("inside the get_user_memories function and user id is {user_id}")
        filters = {
            "OR": [
                {"user_id": user_id}
            ]
        }

        result = client.get_all(filters=filters)
        print(f"result is {result}")
        if not result or not result.get("results"):
            logger.info(f"No memories found for user_id={user_id}")
            return []
        
        # ✅ THIS IS THE FIX
        memories = [m["memory"] for m in result["results"]]
        print(f"memory is {memories}")
        logger.info(
            f"Retrieved {len(memories)} memories for user_id={user_id}"
        )

        return memories

    except Exception as e:
        logger.error(f" Failed to fetch memories: {e}")
        return []



