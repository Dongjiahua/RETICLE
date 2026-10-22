"""Statistics of the tile-mixture probability map of a CAMELYON16-MS region (tile_experts.region_maps): maxima and top-k means,
the number of hot fields and the size and extent of the largest cluster at five thresholds, and each expert's maxima. They are
one of the reader candidates of the segmenter (segmenter.read, 'head_seg+tile')."""
import numpy as np

from reticle.experts.metastasis.tile_experts import read_region

TAUS = (0.5, 0.8, 0.9, 0.95, 0.99)


def map_features(grid, E, mpp):
    g = np.nan_to_num(grid, nan=0.0).reshape(-1); f = [g.max(), np.sort(g)[-3:].mean(), np.sort(g)[-10:].mean()]
    for tau in TAUS:
        _, size, _, ext = read_region(grid, tau, mpp); f += [np.log1p((g >= tau).sum()), np.log1p(size), ext/1000]
    for i in range(E.shape[-1]):
        e = np.nan_to_num(E[..., i], nan=0.0).reshape(-1); f += [e.max(), np.sort(e)[-3:].mean()]
    return np.array(f, dtype=np.float32)
