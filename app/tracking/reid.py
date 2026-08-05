"""Optional OSNet appearance embeddings and bounded identity gallery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray


Embedding = NDArray[np.float32]


class OsNetReIdentifier:
    """Run an OSNet ONNX model on BGR person crops."""

    def __init__(self, model_path: str | Path, *, providers: Sequence[str]) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ReID model does not exist: {self.model_path}")
        try:
            self._session = ort.InferenceSession(
                str(self.model_path), providers=list(providers)
            )
        except Exception as exc:
            raise RuntimeError(f"could not load OSNet ReID model {self.model_path}: {exc}") from exc
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) < 1:
            raise RuntimeError("OSNet ReID model must have one image input and an embedding output")
        shape = inputs[0].shape
        if len(shape) != 4 or shape[1] != 3:
            raise RuntimeError(f"OSNet ReID input must be NCHW RGB, got {shape}")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._height = int(shape[2]) if isinstance(shape[2], int) else 256
        self._width = int(shape[3]) if isinstance(shape[3], int) else 128

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(self._session.get_providers())

    def embed(self, frame: NDArray[np.uint8], xyxy: Sequence[float]) -> Embedding | None:
        """Return a unit-normalized embedding, or ``None`` for an empty crop."""

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = xyxy
        left = max(0, min(width, int(np.floor(x1))))
        top = max(0, min(height, int(np.floor(y1))))
        right = max(0, min(width, int(np.ceil(x2))))
        bottom = max(0, min(height, int(np.ceil(y2))))
        if right <= left or bottom <= top:
            return None
        crop = frame[top:bottom, left:right]
        resized = cv2.resize(crop, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)) / np.asarray(
            (0.229, 0.224, 0.225), dtype=np.float32
        )
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
        try:
            output = self._session.run(
                [self._output_name], {self._input_name: np.ascontiguousarray(tensor)}
            )[0]
        except Exception as exc:
            raise RuntimeError(f"OSNet ReID inference failed: {exc}") from exc
        return normalize_embedding(np.asarray(output).reshape(-1))


@dataclass(slots=True)
class _GalleryEntry:
    embedding: Embedding
    last_seen_frame: int


class ReIdGallery:
    """Keep bounded identity prototypes and perform cosine re-association."""

    def __init__(self, *, similarity_threshold: float = 0.75, max_age_frames: int = 300) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("ReID similarity_threshold must be between 0 and 1")
        if max_age_frames <= 0:
            raise ValueError("ReID max_age_frames must be positive")
        self.similarity_threshold = similarity_threshold
        self.max_age_frames = max_age_frames
        self._entries: dict[int, _GalleryEntry] = {}

    def update(self, track_id: int, embedding: Embedding, frame_index: int) -> None:
        normalized = normalize_embedding(embedding)
        previous = self._entries.get(track_id)
        if previous is not None:
            normalized = normalize_embedding(0.8 * previous.embedding + 0.2 * normalized)
        self._entries[track_id] = _GalleryEntry(normalized, frame_index)
        self._evict(frame_index)

    def match(
        self,
        embedding: Embedding,
        frame_index: int,
        *,
        excluded_track_ids: set[int],
    ) -> int | None:
        self._evict(frame_index)
        query = normalize_embedding(embedding)
        candidates = (
            (float(np.dot(query, entry.embedding)), track_id)
            for track_id, entry in self._entries.items()
            if track_id not in excluded_track_ids
        )
        score, track_id = max(candidates, default=(-1.0, -1))
        return track_id if score >= self.similarity_threshold else None

    def clear(self) -> None:
        self._entries.clear()

    def _evict(self, frame_index: int) -> None:
        for track_id, entry in tuple(self._entries.items()):
            if frame_index - entry.last_seen_frame > self.max_age_frames:
                del self._entries[track_id]


def normalize_embedding(values: NDArray[np.floating]) -> Embedding:
    embedding = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("ReID embedding must have a finite non-zero norm")
    return embedding / norm
