from abc import ABC, abstractmethod
import json
from pathlib import Path


class BaseTokenizer(ABC):
    """Abstract Base Class for implementing custom tokenizers."""

    def __init__(self, special_tokens: list[str] | None = None):
        self.special_tokens = special_tokens or []
        
        # Vocab mappings: ID -> Bytes and Bytes(BPE) or characters and subwords -> ID
        self.vocab: dict[int, bytes] = {}
        # optional merge

    @abstractmethod
    def train(self, input_text: str, vocab_size: int) -> None:
        """Train the tokenizer vocabulary from raw text."""
        pass

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert a text string into a list of integer token IDs."""
        pass

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Convert a list of integer token IDs back into a text string."""
        pass


    @property
    def vocab_size(self) -> int:
        """Returns the total number of items in the vocabulary."""
        return len(self.vocab)

    def save(self, filepath: str | Path) -> None:
        """Serializes the tokenizer vocabulary and metadata to a JSON file."""
        filepath = Path(filepath)
        
        # Safely encode bytes to Latin-1 strings for valid JSON
        vocab_json = {
            str(idx): token_bytes.decode("latin1") 
            for idx, token_bytes in self.vocab.items()
        }

        data = {
            "type": self.__class__.__name__,
            "special_tokens": self.special_tokens,
            "vocab": vocab_json,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> "BaseTokenizer":
        """Loads a tokenizer instance from a saved JSON file."""
        filepath = Path(filepath)
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls(special_tokens=data.get("special_tokens", []))
        
        # Restore vocabulary maps from Latin-1 strings back to raw bytes
        instance.vocab = {
            int(idx): token_str.encode("latin1") 
            for idx, token_str in data["vocab"].items()
        }
        instance.vocab_inv = {
            token_bytes: idx 
            for idx, token_bytes in instance.vocab.items()
        }

        return instance