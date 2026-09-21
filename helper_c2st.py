import numpy as np
from bayesflow.diagnostics.metrics import classifier_two_sample_test
from keras import regularizers
from tqdm import tqdm


def ensemble_predict(estimates, classifiers):
    scores = np.array([
        np.asarray(c(estimates, training=False)).ravel()
        for c in classifiers
    ])
    score = scores.mean(axis=0)
    statistic = np.mean((score - 0.5) ** 2)
    return score, statistic


def train_c2st(estimates, targets,
              n_random=100, batch_size=64, rng=None):
    """C2ST classifier on (posterior draw, data embedding) pairs plus label-permuted classifiers
    for the permutation p-value."""
    if estimates.shape[0] != targets.shape[0]:
        raise ValueError("Estimates and targets must have the same number of samples.")

    rng = np.random.default_rng(rng)
    # randomize order
    estimates = estimates[rng.permutation(estimates.shape[0])]
    targets = targets[rng.permutation(targets.shape[0])]

    # create train/test split
    n_test = int(estimates.shape[0] * 0.1)
    pooled_train = np.concatenate((estimates[:-n_test], targets[:-n_test]), axis=0)
    pooled_test = np.concatenate((estimates[-n_test:], targets[-n_test:]), axis=0)
    n_est = estimates[:-n_test].shape[0]
    estimates, targets = pooled_train[:n_est], pooled_train[n_est:]

    # standardize
    mean, std = np.mean(pooled_train, axis=0), np.std(pooled_train, axis=0)
    if np.any(std == 0):
        raise ValueError("Standard deviation is zero for some features, cannot standardize.")
    pooled_train = (pooled_train - mean) / std
    pooled_test = (pooled_test - mean) / std
    estimates, targets = pooled_train[:estimates.shape[0]], pooled_train[estimates.shape[0]:]

    common = dict(
        return_metric_only=False,
        batch_size=batch_size,
        standardize=False,
        cross_validation_splits=5,
        mlp_kwargs=dict(dropout=0.1, kernel_regularizer=regularizers.l2(1e-3)),
        max_epochs=100,
    )

    results = classifier_two_sample_test(
        estimates=estimates, targets=targets, **common
    )
    score, _ = ensemble_predict(pooled_test, results['classifiers'])
    results["score"] = np.maximum(score.mean(), 1 - score.mean())

    # create random classifiers for permutation test
    random_results = []
    for _ in tqdm(range(n_random)):
        shuffled = pooled_train[rng.permutation(pooled_train.shape[0])]  # permute all labels to create random classifier
        _random_results = classifier_two_sample_test(
            estimates=shuffled[:n_est], targets=shuffled[n_est:], **common
        )
        random_results.append(_random_results)
    return results, random_results, mean, std


def score_c2st(estimates_real, c2st):
    """Apply a trained C2ST to real-data (posterior draw, embedding) pairs.
    Returns per-draw scores, the test statistic and the permutation p-value."""
    results, random_results, mean, std = c2st
    x = (estimates_real - mean) / std

    # compute scores
    score, statistic = ensemble_predict(x, results['classifiers'])
    statistic_random = np.array([ensemble_predict(x, c['classifiers'])[1] for c in random_results])

    # compute test statistic and p-value
    p_val = np.mean(statistic_random >= statistic)

    # maximum, so we report value >= 0.5
    score_per_sample = np.maximum(score, 1 - score)
    mean_score = np.maximum(np.mean(score), 1 - np.mean(score))

    return mean_score, score_per_sample, statistic, p_val
