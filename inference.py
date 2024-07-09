import numpy
import onnxruntime as ort


def overlapped_fold(y, chunk_size=2048, overlap_size=512):
    assert y.ndim == 1 and chunk_size > overlap_size > 0
    hop_size = chunk_size - overlap_size
    if y.shape[0] < chunk_size:
        n_frames = 1
    else:
        n_frames = (y.shape[0] - chunk_size + hop_size - 1) // hop_size + 1
    padding = (0, (n_frames - 1) * hop_size + chunk_size - y.shape[0])
    y = numpy.pad(y, padding, mode="constant")
    heads = numpy.arange(0, n_frames) * hop_size  # [N]
    offsets = numpy.arange(0, chunk_size)  # [L]
    indices = offsets[None, :] + heads[:, None]  # [N, L]
    y_fold = y[indices]

    return y_fold


def overlapped_unfold(y, chunk_size=2048, overlap_size=512):
    assert y.ndim == 2 and y.shape[1] == chunk_size and chunk_size > overlap_size > 0
    n_frames = y.shape[0]
    if n_frames == 1:
        return y[0]
    hop_size = chunk_size - overlap_size
    heads = numpy.arange(0, n_frames) * hop_size  # [N]
    offsets = numpy.arange(0, chunk_size)  # [L]
    indices = offsets[None, :] + heads[:, None]  # [N, L]
    weights = numpy.stack([
        numpy.ones(chunk_size),
        numpy.concatenate([
            numpy.arange(1, overlap_size + 1) / overlap_size,
            numpy.ones(chunk_size - overlap_size)
        ]),
        numpy.concatenate([
            numpy.ones(chunk_size - overlap_size),
            numpy.arange(1, overlap_size + 1)[::-1] / overlap_size
        ]),
    ], axis=0).min(axis=0, keepdims=True)  # [1, L]
    y_unfold = numpy.zeros((n_frames - 1) * hop_size + chunk_size, dtype=y.dtype)
    y_unfold[indices] += y * weights
    y_unfold_weights = numpy.zeros((n_frames - 1) * hop_size + chunk_size, dtype=y.dtype)
    y_unfold_weights[indices] += weights
    y_unfold /= y_unfold_weights
    return y_unfold


def infer(
        session: ort.InferenceSession,
        waveform: numpy.ndarray,
        chunk_size: int = None,
        overlap_size: int = 8192,
        cut_size: int = 2048,
        batch_size: int = 1
):
    original_length = waveform.shape[0]
    chunked = chunk_size is not None and original_length > chunk_size

    if chunked:
        waveform_fold = overlapped_fold(
            numpy.pad(waveform, (cut_size, cut_size)),
            chunk_size=chunk_size,
            overlap_size=overlap_size
        )
    else:
        waveform_fold = waveform[None]

    harmonic_fold = numpy.zeros_like(waveform_fold)
    noise_fold = numpy.zeros_like(waveform_fold)
    for start_idx in range(0, waveform_fold.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, waveform_fold.shape[0])
        waveform_batch = waveform_fold[start_idx:end_idx, :]
        harmonic_batch, noise_batch = session.run(
            ['harmonic', 'noise'],
            input_feed={'waveform': waveform_batch}
        )
        harmonic_fold[start_idx:end_idx] = harmonic_batch
        noise_fold[start_idx:end_idx] = noise_batch

    if chunked:
        harmonic = overlapped_unfold(
            harmonic_fold[:, cut_size: -cut_size],
            chunk_size=chunk_size - cut_size * 2,
            overlap_size=overlap_size - cut_size * 2
        )[:original_length]
        noise = overlapped_unfold(
            noise_fold[:, cut_size: -cut_size],
            chunk_size=chunk_size - cut_size * 2,
            overlap_size=overlap_size - cut_size * 2
        )[:original_length]
    else:
        harmonic = harmonic_fold[0]
        noise = noise_fold[0]

    return harmonic, noise
