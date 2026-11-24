"""Classification loss and gesture event metrics."""

import numpy as np

from paceloop.learning.model import LABELS, BiteEventGate


def cross_entropy(truth, scores, class_weights=None):
    """Mean -log(p[correct class]), optionally weighted by training class counts."""
    truth, scores = np.asarray(truth), np.asarray(scores, dtype=np.float64)
    if truth.ndim != 1 or scores.shape != (len(truth), len(LABELS)) or not len(truth):
        raise ValueError("Expected nonempty labels and matching four-class probabilities")
    if not np.issubdtype(truth.dtype, np.integer) or np.any((truth < 0) | (truth >= len(LABELS))):
        raise ValueError("Invalid teacher labels")
    if (
        not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
        or not np.allclose(scores.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Invalid student probabilities")
    loss = -np.log(np.maximum(scores[np.arange(len(truth)), truth], 1e-12))
    if class_weights is None:
        return float(loss.mean())
    weights = np.asarray(class_weights, dtype=float)
    if weights.shape != (len(LABELS),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Invalid class weights")
    return float(np.average(loss, weights=weights[truth]))


def classification_metrics(truth, scores):
    predicted = scores.argmax(axis=1)
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    np.add.at(matrix, (truth, predicted), 1)
    per_class = {}
    for i, name in enumerate(LABELS):
        precision = matrix[i, i] / max(1, matrix[:, i].sum())
        recall = matrix[i, i] / max(1, matrix[i].sum())
        per_class[name] = dict(
            precision=float(precision),
            recall=float(recall),
            f1=float(2 * precision * recall / max(1e-9, precision + recall)),
            support=int(matrix[i].sum()),
        )
    return dict(
        cross_entropy=cross_entropy(truth, scores),
        accuracy=float(np.mean(predicted == truth)),
        macro_f1=float(np.mean([m["f1"] for m in per_class.values()])),
        confusion_matrix=matrix.tolist(),
        per_class=per_class,
    )


def event_metrics(episodes, scores, threshold):
    tp = fp = fn = negative_episodes = false_alert_episodes = 0
    for ep in episodes:
        gate = BiteEventGate(threshold)
        detected = [
            float(t)
            for t, p in zip(ep["time"], scores[ep["start"] : ep["stop"]])
            if gate.update(float(t), p)
        ]
        if ep["action"] == 1:
            start, stop = ep["interval"]
            matched = any(start - 0.2 <= t <= stop + 0.4 for t in detected)
            tp += int(matched)
            fn += int(not matched)
            fp += len(detected) - int(matched)
        else:
            negative_episodes += 1
            false_alert_episodes += bool(detected)
            fp += len(detected)
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return dict(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / max(1e-9, precision + recall),
        non_bite_episodes=negative_episodes,
        non_bite_episodes_with_false_event=false_alert_episodes,
    )
