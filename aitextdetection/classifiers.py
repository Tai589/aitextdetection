"""模型训练、验证集选参和人类误报阈值。

先理解数据的职责：
1. training 的 text 和 label 参与 fit；词表、缩放参数和分类器只从训练集学。
2. validation 的 text 只做 transform / predict；标签用于选 C 和阈值。
3. 本文件不读取任何测试 CSV，测试成绩不能反过来决定参数。

两种模型分别回答：
- length：只凭 log(1 + 词数) 能区分多少？这是诊断用的简单参照。
- combined：把词语和字符 TF-IDF 拼起来，再训练逻辑回归。

来源 ID、生成模型、领域可写入报告，但绝不当模型输入。
分数是模型输出，尚不能解释成现实中由 AI 写作的概率。
"""

from __future__ import annotations

import math
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

from aitextdetection.dataset import DataError, file_sha256, write_json, write_csv, fingerprint


VERSION = "word-baseline-v1"
# length 是诊断参照；combined 是词语与字符特征组合的文本检测器。
FAMILIES = ("length", "combined")


class LogWordCount(TransformerMixin, BaseEstimator):
    """一篇文本变成一个数：log(1 + 按空白分隔的词数)。

    sklearn 转换器须提供 fit / transform。长度公式不需要学习参数，
    后面的 StandardScaler 才学习训练集均值和标准差。
    定义可导入的类，是为了让 joblib 重新加载时能找到它。
    """

    def fit(self, texts, y=None):
        # 公式没有待学习参数，按 sklearn 约定返回自己。
        return self

    def transform(self, texts):
        # split() 按空白分词；这是本实验统一使用的长度定义，并非语言学分词。
        # log1p(x) 就是 log(1 + x)，空文本长度为 0 时也能计算。
        # sklearn 要求输入是二维表；reshape(-1, 1) 把结果整理成每篇一行、一列。
        return np.log1p(np.asarray([len(text.split()) for text in texts], dtype=float)).reshape(-1, 1)



def training_settings(config: dict) -> dict:
    # 训练前核对协议，避免特征或求解器写错后悄悄运行不同实验。
    # 这里只检查、返回配置，不读取数据，也不训练模型。
    spec = config["training"]
    if spec.get("protocol_version") != VERSION or spec.get("selection_metric") != "validation_auroc":
        raise DataError("训练协议必须为 word-baseline-v1，按验证集 AUROC 选参数。")
    grid = spec["C_values"]
    if (not isinstance(grid, list) or not 1 <= len(grid) <= 3
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in grid)
            or len(set(grid)) != len(grid)):
        raise DataError("C_values 必须是最多 3 个不重复的正数。")
    if spec["decision_threshold"] != .5:
        raise DataError("默认分类阈值必须为 0.5；人类误报阈值由验证分数另行计算。")
    word, lr = spec["word"], spec["logistic_regression"]
    if (word["ngram_range"] != [1, 2] or not isinstance(word["max_features"], int)
            or not 1 <= word["max_features"] <= 30000
            or not isinstance(word["min_df"], int) or word["min_df"] < 1
            or not isinstance(word["sublinear_tf"], bool)):
        raise DataError("词级配置无效：使用 1–2 gram、最多 30000 特征、整数 min_df。")
    if (lr["solver"] != "lbfgs" or lr["class_weight"] != "balanced"
            or not isinstance(lr["max_iter"], int) or lr["max_iter"] < 1
            or not isinstance(lr["tol"], (int, float)) or not 0 < lr["tol"] < 1):
        raise DataError("逻辑回归配置无效。")
    char = spec["char"]
    if (char["analyzer"] != "char" or char["ngram_range"] != [3, 5]
            or not isinstance(char["max_features"], int) or not 1 <= char["max_features"] <= 50000
            or not isinstance(char["min_df"], int) or char["min_df"] < 1
            or not isinstance(char["sublinear_tf"], bool)):
        raise DataError("字符配置无效：使用 3–5 字符片段、最多 50000 特征、整数 min_df。")
    if spec["combined"]["normalization"] != "l2":
        raise DataError("组合模型必须在拼接后进行 L2 归一化。")
    return spec



def build_pipeline(config: dict, c_value: float) -> Pipeline:
    spec = training_settings(config)

    # 返回尚未训练的管线。Pipeline 会依次转换文本，最后训练分类器；
    # 预测时复用已经学到的转换规则，不重新建立词表。
    def vectorizer(name):
        features = spec[name]
        # word 分支统计单词和相邻两词；char 分支统计连续 3～5 个字符。
        # TF-IDF 把这些片段变成数值。词表和 IDF 只在训练集的 fit 中确定。
        # min_df 排除只在很少文档中出现的片段，max_features 限制特征总数。
        # sublinear_tf 为真时用对数缩小重复次数的差异，避免单纯重复占过大权重。
        return TfidfVectorizer(
            analyzer="word" if name == "word" else "char",
            ngram_range=tuple(features["ngram_range"]), max_features=features["max_features"],
            min_df=features["min_df"], sublinear_tf=features["sublinear_tf"],
        )

    # 每行对应一篇文本，每列对应一个片段；未出现的片段大多为 0。
    # 稀疏矩阵只保存必要的非零项，适合普通电脑上的大量文本特征。
    # FeatureUnion 横向拼接 word、char 两路；整体 L2 归一化再把每行
    # 的向量长度调成 1。拼接后的归一化也是实验的一部分，不能省略。
    steps = [("tfidf", FeatureUnion([("word", vectorizer("word")), ("char", vectorizer("char"))])),
             ("normalize", Normalizer(norm="l2", copy=False))]
    # C 越大，对分类权重的约束越弱。它不是分类阈值。
    # balanced 按训练集中各类数量的倒数设置权重，使较少的人类类获得更高权重。
    # 这会改变拟合时的损失权重，不会复制样本，也不会把数据数量变成各一半。
    lr = spec["logistic_regression"]
    steps.append(("classifier", LogisticRegression(
            C=c_value, solver=lr["solver"], class_weight=lr["class_weight"],
            max_iter=lr["max_iter"], tol=lr["tol"], random_state=config["seed"],
        )))
    return Pipeline(steps)



def feature_blocks(pipeline: Pipeline) -> dict:
    # 只用于报告特征数，不影响拟合或预测。
    features = pipeline.named_steps["tfidf"]
    return {name: len(transformer.vocabulary_) for name, transformer in features.transformer_list}



def ai_scores(pipeline: Pipeline, texts: list[str]) -> np.ndarray:
    # 返回与输入文本顺序一致的一列分数；数值越大，模型越倾向判为 AI。
    # predict_proba 每一列对应一个类别，不能不检查类别顺序就取某一列。
    # 这是模型的类别分数，未经现实应用校准，不能作为作者身份的证明。
    classes = list(pipeline.classes_)
    if classes != [0, 1]:
        raise DataError("分类器类别顺序不符合 human=0、AI=1。")
    return pipeline.predict_proba(texts)[:, classes.index(1)]



def calculate_metrics(labels, scores, threshold: float = .5) -> dict:
    # labels 是正确标签，scores 是同一批记录的 AI 分数，二者必须逐行对应。
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or set(labels.tolist()) != {0, 1}:
        raise DataError("评价需要等长标签/分数，且必须有两类。")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise DataError("模型分数必须是 0–1 之间的有限数。")
    # 分数等于阈值也预测 AI，须与阈值校准、bootstrap 使用相同规则。
    predictions = (scores >= threshold).astype(int)
    # 混淆矩阵的行表示真实标签，列表示预测标签。
    # tn：人类判对；fp：人类被判 AI；fn：AI 被判人类；tp：AI 判对。
    # 召回率 = tp / 全部 AI；human_fpr = fp / 全部人类记录。
    # AUROC 使用连续分数衡量排序，不依赖本次选用的分类阈值。
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)), "human_fpr": float(fp / (fp + tn)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }



def train_model(train: list[dict], validation: list[dict], config: dict) -> tuple[Pipeline, np.ndarray, dict]:
    # 输入是训练、验证记录列表和配置；返回选定的管线、验证分数及训练报告。
    spec = training_settings(config)
    # 只提取 text / label，不把来源、领域、生成模型交给分类器。
    train_texts, train_labels = [r["text"] for r in train], [r["label"] for r in train]
    validation_texts, validation_labels = [r["text"] for r in validation], [r["label"] for r in validation]
    if set(train_labels) != {0, 1} or set(validation_labels) != {0, 1}:
        raise DataError("训练与验证都需要 human=0、AI=1 两类。")
    trials, best_pipeline, best_scores, best_metrics, best_c = [], None, None, None, None
    for c_value in sorted(spec["C_values"]):
        pipeline = build_pipeline(config, c_value)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always", ConvergenceWarning)
            # fit 只看训练文本；后面的验证文本仅作转换和预测。
            # 因此验证集中特有的词不会进入训练词表。
            pipeline.fit(train_texts, train_labels)
        if any(issubclass(w.category, ConvergenceWarning) for w in observed):
            # 求解器尚未达到收敛条件时，不把这次拟合登记成成功实验。
            raise DataError(f"C={c_value} 未收敛，停止保存结果；请检查 max_iter 或特征规模。")
        elapsed = time.perf_counter() - started
        scores = ai_scores(pipeline, validation_texts)
        metrics = calculate_metrics(validation_labels, scores, spec["decision_threshold"])
        trial = {"C": c_value, **metrics, "training_seconds": elapsed,
                 "feature_count": sum(feature_blocks(pipeline).values()),
                 "iterations": int(max(pipeline.named_steps["classifier"].n_iter_)), "converged": True}
        trials.append(trial)
        # 从小到大试 C，只在验证 AUROC 严格提高时替换最佳模型。
        # 即使报告里同时计算了 F1 等指标，也不能临时改用它们挑模型。
        # AUROC 完全并列时保留较小 C；测试分数不参与这个过程。
        if best_metrics is None or metrics["auroc"] > best_metrics["auroc"]:
            best_pipeline, best_scores, best_metrics, best_c = pipeline, scores, metrics, c_value
    # 多数类仅是报告参照，不参与选 C。
    majority_label = sorted(Counter(train_labels), key=lambda label: (-train_labels.count(label), label))[0]
    majority = calculate_metrics(validation_labels, np.full(len(validation), majority_label), spec["decision_threshold"])
    report = {
        "model": "combined", "best_C": best_c, "selection_metric": "validation_auroc",
        "tie_break": "smallest C", "decision_threshold": spec["decision_threshold"],
        "train_rows": len(train), "validation_rows": len(validation),
        "train_label_counts": dict(Counter(str(v) for v in train_labels)),
        "validation_label_counts": dict(Counter(str(v) for v in validation_labels)),
        "selected_metrics": best_metrics, "trials": trials,
        "majority_baseline": {"predicted_label": majority_label, **majority},
        "feature_count": sum(feature_blocks(best_pipeline).values()),
        "feature_blocks": feature_blocks(best_pipeline),
        "converged": True,
        "training_seconds_total": sum(t["training_seconds"] for t in trials),
        "input_feature": "text only", "final_test_csvs_opened": False,
        "notes": [
            "These are validation results used for parameter selection, not final test performance.",
            "Model scores have not been calibrated as real-world authorship probabilities.",
            "Generalization must be evaluated separately on each frozen test scenario.",
        ],
    }
    return best_pipeline, best_scores, report



def fit_length(training: list[dict], validation: list[dict], config: dict) -> tuple:
    """长度也用相同 balanced LR / C 网格，标准化仅拟合训练数据。"""
    lr = config["training"]["logistic_regression"]
    texts, labels = [r["text"] for r in training], [r["label"] for r in training]
    val_texts, val_labels = [r["text"] for r in validation], [r["label"] for r in validation]
    trials, best = [], None
    # 完整管线：词数 → log1p → 标准化 → 逻辑回归。
    # StandardScaler 用训练集的均值、标准差转换这一列数值；验证和测试
    # 继续使用训练时的均值、标准差，不能各自重新 fit。
    for c in sorted(config["training"]["C_values"]):
        pipeline = Pipeline([("length", LogWordCount()), ("scale", StandardScaler()),
                             ("classifier", LogisticRegression(C=c, class_weight=lr["class_weight"],
                              solver=lr["solver"], max_iter=lr["max_iter"], tol=lr["tol"], random_state=config["seed"]))])
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(texts, labels)
        if any(issubclass(item.category, ConvergenceWarning) for item in observed):
            raise DataError(f"长度模型 C={c} 未收敛，停止。")
        scores = ai_scores(pipeline, val_texts)
        metrics = calculate_metrics(val_labels, scores)
        trials.append({"C": c, **metrics})
        # 与文本模型使用相同选参规则；两者的差别主要在输入特征。
        if best is None or metrics["auroc"] > best[3]["auroc"]:
            best = (pipeline, scores, c, metrics)
    pipeline, scores, c, metrics = best
    return pipeline, scores, {"best_C": c, "selected_metrics": metrics, "trials": trials, "converged": True}



def validation_fpr_threshold(human_scores, target: float) -> dict:
    """只用验证集的人类分数，计算满足经验误报上限的阈值。

    524 个人类验证样本、目标 5%，最多允许 floor(524×.05)=26 个误报。
    因为 score >= threshold 算 AI，直接取第 27 高分数会把该条也判为 AI。
    nextafter 往更大方向移到下一个浮点数，连同并列分数一起保守处理。
    这里约束的只是验证记录，不保证新领域或现实人群误报也小于 5%。
    """
    scores = np.asarray(human_scores, dtype=float)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any() or not 0 <= target < 1):
        raise DataError("阈值计算需要有效的人类验证分数和 [0,1) 目标误判率。")
    allowed = math.floor(target * len(scores))
    # 降序后第 allowed + 1 条，就是第一个不允许误报的人类分数。
    # Python 下标从 0 开始，所以这里直接使用 allowed。
    boundary = np.sort(scores)[::-1][allowed]
    # 分数相等也判 AI，故阈值必须严格大于边界；nextafter 只向上移动
    # 一个可表示的浮点数。若多人分数并列，实际误报数可能少于 allowed。
    threshold = float(np.nextafter(boundary, np.inf))
    false_positives = int(np.count_nonzero(scores >= threshold))
    if false_positives > allowed:
        raise DataError("阈值未满足验证集经验误判率约束。")
    return {"threshold": threshold, "validation_human_rows": len(scores),
            "allowed_validation_false_positives": allowed, "validation_false_positives": false_positives,
            "validation_fpr": false_positives / len(scores), "target_validation_fpr": target,
            "method": "nextafter first disallowed validation-human score; ties handled conservatively"}



def fit_variants(regimes: dict, validation: list[dict], config: dict, settings: dict, output: Path) -> tuple[dict, dict]:
    """每个训练方案拟合两种模型，并始终使用同一份完整验证集。

    records 登记参数、阈值、摘要；pipelines 保存内存中的已拟合模型。
    调用者必须先冻结 records，之后才可开始读取测试正文。
    """
    labels = np.array([r["label"] for r in validation])
    records, pipelines = {}, {}
    validation_ids = [r["id"] for r in validation]
    for regime, training in regimes.items():
        # regime 是来源筛选方案的名字。每个方案都重新拟合两类模型，
        # 但验证集不随方案改变，否则无法分清变化来自训练还是评价人群。
        for family in FAMILIES:
            name = f"{family}__{regime}"
            if family == "length":
                model, scores, report = fit_length(training, validation, config)
            else:
                model, scores, report = train_model(training, validation, config)
            # 人类误报阈值只根据验证人类分数计算。
            calibration = validation_fpr_threshold(scores[labels == 0], settings["target_validation_fpr"])
            path = output / f"artifacts/{name}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            # 仅加载本次刚保存的模型，逐分数严格一致才接受产物。
            joblib.dump(model, path, compress=3)
            restored = joblib.load(path)
            if not np.array_equal(scores, ai_scores(restored, [r["text"] for r in validation])):
                raise DataError(f"{name} 保存后的预测不能复现。")
            record = {"family": family, "training_regime": regime, "best_C": report["best_C"],
                      "validation_metrics": report["selected_metrics"], "validation_humans": int(sum(labels == 0)),
                      "validation_ids_sha256": fingerprint(validation_ids), "model_sha256": file_sha256(path),
                      "thresholds": {"default_0_5": {"threshold": .5, "validation_fpr": report["selected_metrics"]["human_fpr"]},
                                     "validation_fpr": calibration}}
            if family == "length":
                # 长度是唯一特征，标准化不会改变大小顺序。系数为正表示
                # 文本越长 AI 分数越高；系数为负则相反。这是在该训练池中
                # 学到的关系，不表示长度决定作者类型，也不能擅自翻转分数。
                coefficient = float(model.named_steps["classifier"].coef_[0, 0])
                record.update({"length_coefficient": coefficient,
                               "coefficient_sign": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero",
                               "length_intercept": float(model.named_steps["classifier"].intercept_[0]),
                               "scaler_mean": float(model.named_steps["scale"].mean_[0]),
                               "scaler_scale": float(model.named_steps["scale"].scale_[0])})
            # 验证预测表仅存 ID、标签、分数，不额外保存正文。
            write_json(output / f"results/{name}_training.json", {**report, **record})
            write_csv(output / f"results/{name}_validation_predictions.csv", [
                {"id": row["id"], "source_id": row["source_id"], "true_label": row["label"], "model_score": float(score)}
                for row, score in zip(validation, scores, strict=True)])
            records[name], pipelines[name] = record, model
            print(f"  {name}: train={len(training)}, validation={len(validation)}, C={report['best_C']}, validation AUROC={report['selected_metrics']['auroc']:.4f}", flush=True)
    return records, pipelines

