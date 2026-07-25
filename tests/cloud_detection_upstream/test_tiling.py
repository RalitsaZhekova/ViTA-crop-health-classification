import numpy as np
from cloud_detection.tiling import reconstruct, split_tiles


def test_tiling_and_reconstruction():
    image = np.ones((4, 83, 97), dtype=np.float32)
    tiles, windows, original, padded = split_tiles(image, 64, 16)
    score_tiles = []
    for tile in tiles:
        scores = np.zeros((4, tile.shape[1], tile.shape[2]), dtype=np.float32)
        scores[0] = 1.0
        score_tiles.append(scores)
    reconstructed = reconstruct(score_tiles, windows, original, padded)
    assert reconstructed.shape == (4, 83, 97)
    assert np.allclose(reconstructed[0], 1.0)
