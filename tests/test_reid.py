"""OSNet embedding normalization and appearance-gallery tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.tracking.reid import ReIdGallery, normalize_embedding


def embedding(*values: float) -> np.ndarray:
    return normalize_embedding(np.asarray(values, dtype=np.float32))


def test_gallery_matches_similar_inactive_identity() -> None:
    gallery = ReIdGallery(similarity_threshold=0.8, max_age_frames=20)
    gallery.update(4, embedding(1.0, 0.0, 0.0), 1)
    gallery.update(8, embedding(0.0, 1.0, 0.0), 1)

    assert gallery.match(
        embedding(0.99, 0.05, 0.0), 5, excluded_track_ids=set()
    ) == 4
    assert gallery.match(
        embedding(0.99, 0.05, 0.0), 5, excluded_track_ids={4}
    ) is None


def test_gallery_rejects_dissimilar_and_expired_identities() -> None:
    gallery = ReIdGallery(similarity_threshold=0.8, max_age_frames=3)
    gallery.update(4, embedding(1.0, 0.0), 1)

    assert gallery.match(embedding(0.0, 1.0), 2, excluded_track_ids=set()) is None
    assert gallery.match(embedding(1.0, 0.0), 5, excluded_track_ids=set()) is None


def test_zero_embedding_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-zero norm"):
        normalize_embedding(np.zeros(4, dtype=np.float32))
