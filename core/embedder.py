import os
import ssl
import numpy as np
from typing import List, Optional
from fastembed import TextEmbedding
from config import settings

class LocalEmbedder:
    _instance: Optional["LocalEmbedder"] = None
    _model: Optional[TextEmbedding] = None
    _init_failed: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalEmbedder, cls).__new__(cls)
        return cls._instance

    def _get_model(self) -> Optional[TextEmbedding]:
        if self._model is None and not self._init_failed:
            print(f"[*] Initializing local ONNX embedding model: {settings.EMBEDDING_MODEL}")
            try:
                self._model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
                print("[+] Embedding model loaded and ready.")
            except Exception as e:
                # Handle corporate proxy/Zscaler SSL certificate interception
                print(f"[!] Standard SSL download failed ({e}). Attempting unverified SSL context fallback...")
                try:
                    orig_context = ssl._create_default_https_context
                    ssl._create_default_https_context = ssl._create_unverified_context
                    self._model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
                    ssl._create_default_https_context = orig_context
                    print("[+] Embedding model loaded successfully via fallback.")
                except Exception as e2:
                    print(f"[!] Warning: Could not initialize local ONNX embedding model: {e2}")
                    print("[!] Semantic cache disabled. Exact caching & prompt pruning remain fully operational.")
                    self._init_failed = True
                    self._model = None
        return self._model

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Embed a single text string into a normalized 1D numpy vector (float32).
        Returns None if embedding model is unavailable or fails.
        """
        try:
            model = self._get_model()
            if model is None:
                return None
            generator = model.embed([text])
            vec = next(generator)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.astype(np.float32)
        except Exception as e:
            print(f"[!] Warning: embed_text failed ({e}). Gracefully continuing without semantic cache.")
            return None

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        try:
            model = self._get_model()
            if model is None:
                return []
            vectors = []
            for vec in model.embed(texts):
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                vectors.append(vec.astype(np.float32))
            return vectors
        except Exception as e:
            print(f"[!] Warning: embed_batch failed ({e}).")
            return []

embedder = LocalEmbedder()
