# ============================================================
# src/memory_manager.py
# Persistent conversation memory — survives app restarts
# ============================================================

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from config.settings import CONVERSATIONS_DIR

logger = logging.getLogger(__name__)


class ConversationMemoryManager:
    """
    Saves and loads conversation history to JSON files on disk.

    Without this, every Streamlit page reload loses the chat.
    With this, the assistant remembers previous sessions.
    """

    def __init__(self, memory_dir: str = str(CONVERSATIONS_DIR)):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def save_conversation(
        self,
        messages: List[Dict],
        session_id: str = "default",
    ) -> str:
        """Save a conversation to a timestamped JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conv_{session_id}_{timestamp}.json"
        filepath = self.memory_dir / filename

        payload = {
            "session_id": session_id,
            "timestamp": timestamp,
            "message_count": len(messages),
            "messages": messages,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved conversation: {filename}")
        return str(filepath)

    def load_recent_conversation(self, session_id: str = "default") -> List[Dict]:
        """Load the most recent conversation for a given session ID."""
        files = list(self.memory_dir.glob(f"conv_{session_id}_*.json"))
        if not files:
            return []

        latest = max(files, key=os.path.getmtime)
        with open(latest, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded conversation: {latest.name}")
        return data.get("messages", [])

    def list_sessions(self) -> List[Dict]:
        """List all saved conversation sessions (newest first)."""
        sessions = []
        for filepath in self.memory_dir.glob("conv_*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                sessions.append(
                    {
                        "file": filepath.name,
                        "session_id": data.get("session_id"),
                        "timestamp": data.get("timestamp"),
                        "message_count": data.get("message_count", 0),
                    }
                )
            except Exception:
                pass

        return sorted(sessions, key=lambda x: x["timestamp"], reverse=True)

    def delete_session(self, session_id: str) -> int:
        """Delete all files for a session. Returns number deleted."""
        files = list(self.memory_dir.glob(f"conv_{session_id}_*.json"))
        for f in files:
            f.unlink()
        return len(files)
