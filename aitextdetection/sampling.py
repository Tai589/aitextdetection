"""本研究的来源准入规则：把“筛选什么”与“训练模型”分开。

这是一个纯数据模块：输入是内存中的记录列表，输出是来源集合、训练池和
描述统计；本文件不读取测试集，也不训练模型。

初学者先记住三个层次：
* 一条记录（dict）是一篇文本，含 id、source_id、model、label 等字段。
* 一个来源（source_id）在训练分区中对应 human/chatgpt/gpt4 三篇文本。
* 一个训练方案（regime）由若干完整来源组成，绝不单独丢掉其中一篇。

“200 词”沿用原实验的 text.split() 空白分词规则，不是语言学分词。
"""

from collections import Counter, defaultdict
import math
import statistics

import numpy as np

from aitextdetection.dataset import DataError, validate_word_counts


# 这三个模型标签属于已经冻结的训练分区，每个来源各有一条记录。
# Mistral 留作未见生成器测试，不用它的词数决定哪些来源进入训练。
MODELS = ("human", "chatgpt", "gpt4")


def source_groups(rows: list[dict]) -> dict[str, list[dict]]:
    """核验输入，再把记录按 source_id 归组。

    list[dict] 表示“由字典组成的列表”；dict[str, list[dict]] 表示
    “键是来源 ID、值是该来源记录列表的字典”。这只是类型提示，不会自动校验数据。
    """
    # 从正文重新核对词数，避免被改错的 word_count 改变准入结论。
    validate_word_counts(rows)
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        # id 标识一条文本；source_id 标识同一内容来源的一组文本，两者作用不同。
        if row["id"] in seen:
            raise DataError("准入审计发现重复记录 ID。")
        seen.add(row["id"])
        grouped[row["source_id"]].append(row)
    if not grouped:
        raise DataError("训练来源池为空。")
    for records in grouped.values():
        # Counter 会计数，所以它既能发现缺失模型，也能发现某模型重复两次。
        # 同时核对标签含义：human 为 0，另外两个生成器为 1。
        if (len(records) != 3 or Counter(r["model"] for r in records) != Counter(MODELS)
                or len({r["domain"] for r in records}) != 1
                or any(r["label"] != int(r["model"] != "human") for r in records)):
            raise DataError("每个训练来源必须具有同领域、标签正确的 human/chatgpt/gpt4 各一条。")
    return dict(grouped)


def admission_table(rows: list[dict], minimum: int) -> list[dict]:
    """逐来源记录 HUMAN/GROUP 是否通过，只保存计数，不写正文。"""
    table = []
    # 排序让输出顺序固定：原 CSV 的行顺序变化不会改变诊断表的来源顺序。
    for source, records in sorted(source_groups(rows).items()):
        words = {r["model"]: r["word_count"] for r in records}
        human_ok = words["human"] >= minimum
        ai_ok = all(words[m] >= minimum for m in MODELS if m != "human")
        # 这里只登记来源是否符合条件，不在这里删除任何一条文本。
        # HUMAN 只检查人类文本；GROUP 还要求两个训练 AI 文本都通过。
        # int(True)=1、int(False)=0，方便保存为 CSV 后阅读和筛选。
        table.append({"source_id": source, "domain": records[0]["domain"],
                      **{f"{m}_words": words[m] for m in MODELS},
                      "human_pass": int(human_ok), "ai_pass": int(ai_ok),
                      "group_pass": int(human_ok and ai_ok),
                      # 这个字段也可能包含 human，名称沿用已有结果表，别理解成只列 AI。
                      "failed_generators": ";".join(m for m in MODELS if words[m] < minimum)})
    return table


def sample_sources(grouped: dict, candidates: set[str], quota: dict, seed: int) -> set[str]:
    """按领域配额随机抽取来源，结果只由候选集合、配额和种子决定。"""
    # candidates 是可选来源 ID；quota 是“领域 -> 需要抽取的来源数”。
    # 局部随机数生成器避免改动其他模块的随机状态。
    rng = np.random.default_rng(seed)
    keep = set()
    for domain, count in sorted(quota.items()):
        # 先固定候选顺序，再随机抽样。只设种子而不排序，仍可能因输入顺序而变。
        pool = sorted(s for s in candidates if grouped[s][0]["domain"] == domain)
        if count > len(pool) or count < 0:
            raise DataError("候选来源不足以满足预定领域配额。")
        # replace=False 表示同一来源不能重复抽到；此处抽的是来源，不是单条文本。
        keep.update(rng.choice(pool, size=count, replace=False).tolist())
    return keep


def build_training_regimes(rows: list[dict], minimum: int, seeds: list[int]) -> dict[str, list[dict]]:
    """建立 ALL/HUMAN/GROUP，以及三类按领域等预算的来源对照。"""
    grouped = source_groups(rows)
    # 输入是原来的训练记录，筛选不会从验证集或测试集中补入来源。
    table = admission_table(rows, minimum)
    all_sources = set(grouped)
    human = {r["source_id"] for r in table if r["human_pass"]}
    group = {r["source_id"] for r in table if r["group_pass"]}
    # GROUP 必然包含于 HUMAN：多检查 AI 词数，只可能排除来源，不能新增来源。
    if not human or not group:
        raise DataError("HUMAN或GROUP没有合格训练来源；停止，不更换门槛来追求结果。")
    # 配额按“来源数”计数。每个来源三篇，所以也同时匹配文本数及 1:2 类别比例。
    human_quota = Counter(grouped[s][0]["domain"] for s in human)
    group_quota = Counter(grouped[s][0]["domain"] for s in group)
    pools = {"all": all_sources, f"human_{minimum}": human, f"group_{minimum}": group}
    for seed in seeds:
        # 每个种子得到一套预定随机对照，全部保存，而不是挑成绩最好的一套。
        # 两种 ALL 随机对照：控制各自筛选导致的训练数量与领域组成变化。
        pools[f"random_human_{seed}"] = sample_sources(grouped, all_sources, human_quota, seed)
        pools[f"random_group_{seed}"] = sample_sources(grouped, all_sources, group_quota, seed)
        # 直接对照：在 HUMAN 内抽取与 GROUP 等预算的来源。
        # 这匹配数量和领域，不保证词数、主题、文体等分布相同。
        pools[f"human_group_budget_{seed}"] = sample_sources(grouped, human, group_quota, seed)
    # 通过 HUMAN 的来源，即使某篇 AI 文本未达到词数门槛，也连同另外两篇一起保留。
    # 这里只按来源是否入选过滤，不截断正文，也不改变原训练分区的边界。
    return {name: [r for r in rows if r["source_id"] in keep] for name, keep in pools.items()}


def _stats(records: list[dict]) -> dict:
    """计算一组非空记录的长度统计；log1p(n) 就是 log(1+n)。"""
    # 这些统计描述训练池，不参与模型选参；均值和中位数都用来观察长度分布。
    values = [r["word_count"] for r in records]
    return {"rows": len(values), "mean_words": statistics.mean(values),
            "median_words": statistics.median(values), "min_words": min(values),
            "max_words": max(values), "mean_log_words": statistics.mean(math.log1p(n) for n in values)}


def describe_selection(rows: list[dict], regimes: dict, minimum: int) -> tuple[dict, dict[str, list[dict]]]:
    """返回汇总诊断及四张明细表，不训练模型，也不查看测试结果。"""
    table = admission_table(rows, minimum)
    # 人类通过/不通过 × 两个 AI 都通过/未全通过，共四种来源情况。
    cells = {f"human_{h}_ai_{a}": {"source_groups": 0, "domain_source_counts": {}}
             for h in (0, 1) for a in (0, 1)}
    for row in table:
        cell = cells[f"human_{row['human_pass']}_ai_{row['ai_pass']}"]
        cell["source_groups"] += 1
        domains = cell["domain_source_counts"]
        domains[row["domain"]] = domains.get(row["domain"], 0) + 1
    # 同来源可同时有多篇未通过，单项失败数不能相加当作不同来源总数。
    failure_counts = {m: sum(r[f"{m}_words"] < minimum for r in table) for m in MODELS}
    core = ("all", f"human_{minimum}", f"group_{minimum}")
    retained, lengths, differences, summaries = [], [], [], {}
    total_by_domain = Counter(r["domain"] for r in table)
    for name, selected in regimes.items():
        groups = source_groups(selected)
        generator_stats = {m: _stats([r for r in selected if r["model"] == m]) for m in MODELS}
        deltas = []
        for source, records in sorted(groups.items()):
            words = {r["model"]: r["word_count"] for r in records}
            source_deltas = []
            for generator in MODELS[1:]:
                # 配对的意思是比较“同一个来源”的 AI 与人类，而非两批无关文本。
                # 先求 log 差，再平均；这不同于“先平均词数再取 log”。
                delta = math.log1p(words[generator]) - math.log1p(words["human"])
                source_deltas.append(delta)
                differences.append({"training_regime": name, "source_id": source,
                                    "domain": records[0]["domain"], "generator": generator,
                                    "human_words": words["human"], "ai_words": words[generator],
                                    "ai_minus_human_words": words[generator] - words["human"],
                                    "ai_minus_human_log_words": delta})
            # 每个来源先平均两种 AI 的差值，再跨来源平均，使来源权重一致。
            deltas.append(statistics.mean(source_deltas))
        summaries[name] = {"generator_stats": generator_stats,
                           "mean_ai_minus_human_log_words": statistics.mean(deltas)}
        for domain in sorted(total_by_domain):
            # 分母始终是原始训练池中该领域的来源数，便于比较各筛选方案保留了多少。
            count = sum(rs[0]["domain"] == domain for rs in groups.values())
            for model in MODELS:
                retained.append({"training_regime": name, "domain": domain, "generator": model,
                                 "total_sources": total_by_domain[domain], "retained_sources": count,
                                 "excluded_sources": total_by_domain[domain] - count,
                                 "retention_rate": count / total_by_domain[domain]})
                records = [r for r in selected if r["domain"] == domain and r["model"] == model]
                if records:
                    lengths.append({"training_regime": name, "domain": domain, "generator": model, **_stats(records)})
    human_sources = {r["source_id"] for r in table if r["human_pass"]}
    group_sources = {r["source_id"] for r in table if r["group_pass"]}
    # 集合减法 A-B：在 A 中、却不在 B 中的来源；正是 HUMAN-only 差异来源。
    diagnostics = {"total_sources": len(table), "human_sources": len(human_sources),
                   "group_sources": len(group_sources), "human_only_sources": len(human_sources - group_sources),
                   "identical_human_group": human_sources == group_sources, "pass_cells": cells,
                   "generator_failure_counts": failure_counts,
                   "human_only_domain_counts": dict(sorted(Counter(r["domain"] for r in table
                       if r["human_pass"] and not r["group_pass"]).items())),
                   "human_only_failure_counts": {m: sum(r["human_pass"] and not r["group_pass"]
                       and r[f"{m}_words"] < minimum for r in table) for m in MODELS[1:]},
                   "core_regimes": list(core), "regime_length_statistics": summaries}
    # 第一项是给 JSON 报告使用的汇总；第二项是四张明细表，由调用方分别写为 CSV。
    return diagnostics, {"source_admission": table, "retention_by_domain_generator": retained,
                         "length_statistics": lengths, "within_source_differences": differences}
