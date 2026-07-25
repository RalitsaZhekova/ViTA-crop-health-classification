from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    y: int
    x: int
    size: int


def _positions(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    positions = list(range(0, length - size + 1, stride))
    final_position = length - size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def split_tiles(
    image: np.ndarray,
    size: int,
    overlap: int,
    padding_mode: str = "reflect",
) -> tuple[list[np.ndarray], list[Window], tuple[int, int], tuple[int, int]]:
    if image.ndim != 3:
        raise ValueError(f"Expected C x H x W input, found {image.shape}.")
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Invalid tile size or overlap.")

    _, height, width = image.shape
    pad_height = max(0, size - height)
    pad_width = max(0, size - width)
    effective_mode = (
        "edge"
        if padding_mode == "reflect" and (height == 1 or width == 1)
        else padding_mode
    )
    padded = np.pad(
        image,
        ((0, 0), (0, pad_height), (0, pad_width)),
        mode=effective_mode,
    )

    _, padded_height, padded_width = padded.shape
    stride = size - overlap
    tiles: list[np.ndarray] = []
    windows: list[Window] = []
    for y in _positions(padded_height, size, stride):
        for x in _positions(padded_width, size, stride):
            tiles.append(padded[:, y : y + size, x : x + size])
            windows.append(Window(y=y, x=x, size=size))

    return tiles, windows, (height, width), (padded_height, padded_width)


def reconstruct(
    score_tiles: list[np.ndarray],
    windows: list[Window],
    original_shape: tuple[int, int],
    padded_shape: tuple[int, int],
) -> np.ndarray:
    """Average overlapping class-score tiles into a full-scene score tensor."""
    if not score_tiles:
        raise ValueError("No score tiles supplied.")
    if len(score_tiles) != len(windows):
        raise ValueError("Score-tile and window counts do not match.")

    class_count = score_tiles[0].shape[0]
    accumulator = np.zeros((class_count, *padded_shape), dtype=np.float32)
    observations = np.zeros(padded_shape, dtype=np.float32)

    for scores, window in zip(score_tiles, windows, strict=True):
        expected_shape = (class_count, window.size, window.size)
        if scores.shape != expected_shape:
            raise ValueError(f"Expected tile scores {expected_shape}, found {scores.shape}.")
        y_end = window.y + window.size
        x_end = window.x + window.size
        accumulator[:, window.y:y_end, window.x:x_end] += scores
        observations[window.y:y_end, window.x:x_end] += 1.0

    accumulator /= np.maximum(observations[None, :, :], 1.0)
    height, width = original_shape
    return accumulator[:, :height, :width]
