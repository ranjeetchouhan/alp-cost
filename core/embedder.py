import numpy as np
from typing import List, Optional
from fastembed import TextEmbedding
from config import settings

class LocalEmbedder:
    _instance: Optional["LocalEmbedder"] = None
    _model: Optional[TextEmbedding] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalEmbedder, cls).__new__(cls)
        return cls._instance

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            print(f"[*] Initializing local ONNX embedding model: {settings.EMBEDDING_MODEL}")
            self._model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
            print("[+] Embedding model loaded and ready.")
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string into a normalized 1D numpy vector (float32).
        Cosine similarity between two normalized vectors is just np.dot(a, b).
        """
        model = self._get_model()
        generator = model.embed([text])
        vec = next(generator)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        model = self._get_model()
        vectors = []
        for vec in model.embed(texts):
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec.astype(np.float32))
        return vectors

embedder = LocalEmbedder()
