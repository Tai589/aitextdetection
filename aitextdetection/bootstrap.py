"""配对的“来源组件 bootstrap”条件区间。

- component：因近重复关系连接的一组来源；抽样时整组一起取。
- bootstrap：从原组件中有放回抽取同样多次，有的组件可能重复。
- 配对：模型 A 和 B 共用完全相同的抽取，最后计算 A - B。

点估计中每条原记录等权，组件只是重抽样单位。一个组件抽中 k 次，
其所有记录的权重都为 k；不能把每组均值视为一条等权记录。
两个模型或两个评价子集始终使用共同组件权重，保持配对关系。

区间只反映固定模型、固定阈值和固定评价人群的测试抽样变化，
不含重新训练、选参数、选门槛、阈值校准的不确定性。
它不能证明组件之间独立，不能修复选择偏差，也不是因果证据。
数值计算顺序与原实验保持一致，只增加教学注释。
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


_METRICS = ("auroc", "recall", "human_fpr")
# 分批计算只限制内存用量，不减少总抽样轮数。
_BATCH_SIZE = 50
_CONFIDENCE = 0.95


def _vector(values, name: str, *, objects: bool = False) -> np.ndarray:
    # 一维数组对应按记录顺序排列的一列数据。
    try:
        result = np.asarray(values, dtype=object if objects else None)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional array.") from error
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    return result


def _scores(values, name: str, size: int) -> np.ndarray:
    # 长度必须和标签一致，且每个分数都能用于数值运算；不能混入 NaN、
    # 无穷大或文本。发现这些问题时立即报错，避免得到没有意义的区间。
    result = _vector(values, name)
    if len(result) != size or result.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain one real score per record.")
    result = result.astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must contain finite scores in [0, 1].")
    return result


def _threshold(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar.")
    # 稍大于 1 的阈值也合法：人类验证分数为 1 时可选择全部判人类。
    return result


def _mask(values, name: str, labels: np.ndarray) -> np.ndarray:
    # mask 是一列 True / False：True 的记录参与评价，False 的记录不参与。
    # 未传入时保留全部记录。本研究主实验使用同一份完整测试记录。
    if values is None:
        result = np.ones(len(labels), dtype=bool)
    else:
        result = _vector(values, name)
        if len(result) != len(labels) or result.dtype.kind != "b":
            raise ValueError(f"{name} must contain one boolean per record.")
    if not np.any(result) or len(np.unique(labels[result])) != 2:
        raise ValueError(f"{name} must select both human and AI records.")
    return result


@dataclass(frozen=True)
class _Cohort:
    # 保存后续反复计算需要的数组；只预处理一次，抽样时复用。
    # cluster_indices 是组件编号，positive 表示真实 AI，
    # predicted_positive 表示按当前阈值预测 AI，tie_starts 标记同分组起点。
    cluster_indices: np.ndarray
    positive: np.ndarray
    predicted_positive: np.ndarray
    tie_starts: np.ndarray


def _prepare_cohort(labels: np.ndarray, scores: np.ndarray, clusters: np.ndarray,
                    mask: np.ndarray, threshold: float) -> _Cohort:
    # 先选评价人群，再按分数稳定排序；相同分数作为一组算并列。
    indices = np.flatnonzero(mask)
    indices = indices[np.argsort(scores[indices], kind="stable")]
    sorted_scores = scores[indices]
    tie_starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    return _Cohort(clusters[indices], labels[indices] == 1,
                   sorted_scores >= threshold, tie_starts)


def _cohort_metrics(cohort: _Cohort, multiplicities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用组件次数作记录权重，精确计算含并列分数的 AUROC、召回和误报。"""
    # multiplicities 每行是一轮抽样，每列是某个组件被抽中的次数。
    # 用组件编号取出每条记录所属组件的次数，作为这条记录的权重。
    # 同一组件内的记录始终一起重复或一起缺席，不独立抽取其中的文本。
    weights = multiplicities[:, cohort.cluster_indices]
    positive_weights = weights * cohort.positive
    negative_weights = weights * ~cohort.positive
    positive_by_score = np.add.reduceat(positive_weights, cohort.tie_starts, axis=1)
    negative_by_score = np.add.reduceat(negative_weights, cohort.tie_starts, axis=1)
    positive_total = positive_by_score.sum(axis=1)
    negative_total = negative_by_score.sum(axis=1)
    # 一轮中若只抽到一种标签，就无法定义 AUROC；该轮稍后共同排除。
    valid = (positive_total > 0) & (negative_total > 0)

    # AUROC：AI 分数高于人类记 1，相同记 0.5，低于记 0。
    # 已升序排序，因此累加较低人类权重，再加同分人类权重的一半。
    negative_below = np.cumsum(negative_by_score, axis=1) - negative_by_score
    concordance = np.sum(positive_by_score * (negative_below + 0.5 * negative_by_score), axis=1)
    true_positive = positive_weights[:, cohort.predicted_positive].sum(axis=1)
    false_positive = negative_weights[:, cohort.predicted_positive].sum(axis=1)
    result = np.full((len(multiplicities), len(_METRICS)), np.nan, dtype=np.float64)
    # 三列依次为 AUROC、AI 召回率、人类误报率。where 保证只除以有效分母。
    np.divide(concordance, positive_total.astype(np.float64) * negative_total,
              out=result[:, 0], where=valid)
    np.divide(true_positive, positive_total, out=result[:, 1], where=valid)
    np.divide(false_positive, negative_total, out=result[:, 2], where=valid)
    return result, valid


def paired_cluster_intervals(labels, scores_a, scores_b, clusters, threshold_a=0.5,
                             threshold_b=0.5, *, mask_a=None, mask_b=None,
                             seed=42, n_resamples=1000) -> dict:
    """Return 95% paired percentile intervals for ``A - B`` on frozen scores.

    ``labels`` uses 0 for human and 1 for AI; a score at or above its threshold
    predicts AI. All arrays must be aligned on the same unique scenario IDs in
    the same order, and ``clusters`` must contain a nonempty component string
    per record. Boolean masks can select different, including nested, cohorts;
    each point-estimate cohort must include both classes. Scores must be finite
    probabilities in [0, 1], while thresholds can be any finite real scalars.

    Invalid inputs raise ``ValueError``. ``valid_resamples`` counts draws having
    both classes in both cohorts. If none are valid, interval endpoints are
    ``None``. A seed draws sorted component IDs, making results invariant to a
    joint permutation of the input rows. Memory use is bounded by batches of
    50 resamples rather than the requested total number of resamples.
    """
    # 入口函数返回各指标的 A、B 点估计、A - B 差值和差值区间。
    # AUROC、召回率的差值为正表示 A 较高；人类误报率为正表示 A 误报更多，
    # 因此不能把所有正差值都解释成 A 更好。
    # 所有输入必须已对齐到同一批唯一记录 ID；这里不读正文、不重训练。
    labels = _vector(labels, "labels")
    if labels.dtype.kind not in "biuf" or not np.all(np.isin(labels, (0, 1))):
        raise ValueError("labels must contain only 0 (human) and 1 (AI).")
    labels = labels.astype(np.int8, copy=False)
    scores_a = _scores(scores_a, "scores_a", len(labels))
    scores_b = _scores(scores_b, "scores_b", len(labels))
    cluster_names = _vector(clusters, "clusters", objects=True)
    if (len(cluster_names) != len(labels)
            or any(not isinstance(value, str) or not value.strip() for value in cluster_names)):
        raise ValueError("clusters must contain one nonempty component string per record.")
    # 排序组件名称，使共同打乱输入行后仍得到相同的抽样。
    # return_inverse 把每条记录的组件名称换成整数编号，便于数组索引。
    unique_clusters, cluster_indices = np.unique(cluster_names, return_inverse=True)
    n_clusters = len(unique_clusters)
    threshold_a, threshold_b = _threshold(threshold_a, "threshold_a"), _threshold(threshold_b, "threshold_b")
    mask_a, mask_b = _mask(mask_a, "mask_a", labels), _mask(mask_b, "mask_b", labels)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if (isinstance(n_resamples, (bool, np.bool_))
            or not isinstance(n_resamples, Integral) or n_resamples < 1):
        raise ValueError("n_resamples must be a positive integer.")
    seed, n_resamples = int(seed), int(n_resamples)

    cohort_a = _prepare_cohort(labels, scores_a, cluster_indices, mask_a, threshold_a)
    cohort_b = _prepare_cohort(labels, scores_b, cluster_indices, mask_b, threshold_b)
    # 点估计各组件出现一次，等价于每条记录权重为 1。
    point_weights = np.ones((1, n_clusters), dtype=np.int64)
    point_a, _ = _cohort_metrics(cohort_a, point_weights)
    point_b, _ = _cohort_metrics(cohort_b, point_weights)
    point = {metric: {"a": float(point_a[0, index]), "b": float(point_b[0, index]),
                      "difference": float(point_a[0, index] - point_b[0, index])}
             for index, metric in enumerate(_METRICS)}

    # 固定种子使相同输入的抽样可复现；种子本身不会消除数据或模型偏差。
    rng = np.random.default_rng(seed)
    differences = []
    # 一批只处理 50 轮，避免把全部轮数一次展开占用过多内存。
    for start in range(0, n_resamples, _BATCH_SIZE):
        batch_size = min(_BATCH_SIZE, n_resamples - start)
        # 从全体原组件有放回抽 n_clusters 次。例如 3 个组件可能抽成
        # [2, 2, 0]：组件 2 重复两次，组件 0 一次，组件 1 不出现。
        # A、B 共用这份抽样，才能比较同一批重抽记录上的成绩差异。
        sampled = rng.integers(0, n_clusters, size=(batch_size, n_clusters))
        # 给每轮编号加上不同偏移，再一次性计数；reshape 后恢复为
        # “抽样轮数 × 组件数”的次数表，避免逐轮写 Python 循环。
        offsets = np.arange(batch_size, dtype=np.int64)[:, None] * n_clusters
        multiplicities = np.bincount((sampled + offsets).ravel(),
                                     minlength=batch_size * n_clusters).reshape(batch_size, n_clusters)
        metrics_a, valid_a = _cohort_metrics(cohort_a, multiplicities)
        metrics_b, valid_b = _cohort_metrics(cohort_b, multiplicities)
        # 任一侧缺少人类或 AI，该轮共同丢弃，不能各自留下不同轮。
        valid = valid_a & valid_b
        differences.append(metrics_a[valid] - metrics_b[valid])
    differences = np.concatenate(differences, axis=0)
    valid_resamples = len(differences)
    if valid_resamples:
        # 差值的 2.5% / 97.5% 分位数构成逐项 95% 百分位区间。
        # 区间跨过 0，表示这项重抽样比较没有给出明确的差值方向。
        # 未作多重比较校正，也不表示重新训练的稳定性：模型、C 和阈值
        # 在抽样前就已固定，训练数据变化的影响没有包含在区间内。
        endpoints = np.quantile(differences, (0.025, 0.975), axis=0)
        intervals = {metric: {"low": float(endpoints[0, index]), "high": float(endpoints[1, index]),
                              "confidence": _CONFIDENCE}
                     for index, metric in enumerate(_METRICS)}
    else:
        intervals = {metric: {"low": None, "high": None, "confidence": _CONFIDENCE}
                     for metric in _METRICS}
    return {"direction": "a_minus_b", "n_clusters": n_clusters, "n_resamples": n_resamples,
            "valid_resamples": valid_resamples, "seed": seed,
            "point": point, "intervals": intervals}
