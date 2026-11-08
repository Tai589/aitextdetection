"""把已保存的数值变成中文报告与三张图。

初学者阅读顺序：先看 write_report 的输入，再看 _metric_table 怎样把一行
字典转成 Markdown 表格，最后看三个 _plot_* 函数。这里不训练模型、不读取
原始正文、不重新计算 bootstrap，报告中的每一个数都由调用者传入。
"""

from __future__ import annotations

import math
import os
from pathlib import Path

# reporting.py 在 aitextdetection/ 内，parents[1] 是这份独立代码的项目根目录。
# 图形缓存只放本项目 .cache，避免写回旧项目或用户的全局配置目录。
_CACHE = Path(__file__).resolve().parents[1] / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))

import matplotlib

# Agg 可以在没有图形窗口的终端中保存 PNG；不会弹出阻塞程序的窗口。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SCENARIOS = ("test_id", "test_domain", "test_generator")
# 固定场景、模型族和颜色顺序，同一组数据在不同图里不会忽左忽右。
# 英文用于图上的标注，中文用于报告正文；名称字典不参与模型计算。
SCENARIO_EN = {
    "test_id": "Seen conditions / new sources",
    "test_domain": "Unseen domain (Wiki)",
    "test_generator": "Unseen generator (Mistral)",
}
SCENARIO_ZH = {
    "test_id": "同条件、新来源（ID）",
    "test_domain": "未见领域（Wiki）",
    "test_generator": "未见生成器（Mistral）",
}
FAMILIES = ("length", "combined")
FAMILY_EN = {"length": "Length only", "combined": "Combined text"}
COLORS = ("#3877ab", "#ce773a", "#6c9d63")
METRIC_ZH = {"auroc": "AUROC", "recall": "AI 召回率", "human_fpr": "人类误报率"}


# 以下小函数统一处理缺失值、数字精度和 Markdown 特殊字符，
# 避免“没有结果”被误显示成 0，也避免原始字段破坏表格结构。
def _number(value: object) -> float | None:
    # CSV/JSON 中的数字可能是字符串，也可能缺失。无法解释的值统一返回 None。
    # NaN 和无穷也不能显示成正常成绩；真实的 0 则必须保留。
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: object, *, percent: bool = False, digits: int = 4) -> str:
    result = _number(value)
    if result is None:
        return "—"
    return f"{result:.2%}" if percent else f"{result:.{digits}f}"


def _count(value: object) -> str:
    # 千位分隔符仅用于阅读，原始 CSV 数值不会因此改变。
    result = _number(value)
    return f"{int(result):,}" if result is not None else "—"


def _cell(value: object) -> str:
    # Markdown 用 | 分列，正文中的 | 要转义；单元格里的换行改为空格，避免断开表格。
    return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")


def _delta(value: object) -> str:
    # 差值保留正负号：+0.0010 与 -0.0010 表示相反方向，不能只看绝对值。
    result = _number(value)
    return f"{result:+.4f}" if result is not None else "—"


def _interval(row: dict) -> str:
    low, high = (_number(row.get(key)) for key in ("ci_low", "ci_high"))
    return f"[{low:+.4f}, {high:+.4f}]" if low is not None and high is not None else "—"


def _regime_names(summary: dict) -> tuple[int, str, str, list[int]]:
    # 名称从本次计划读取，不把 human_200、group_200 写死在每张表里。
    plan = summary.get("plan", {})
    minimum = int(plan.get("minimum_words", 200))
    return minimum, f"human_{minimum}", f"group_{minimum}", list(plan.get("matched_seeds", [42, 43, 44]))


def _ordered_regimes(summary: dict) -> list[str]:
    _, human, group, seeds = _regime_names(summary)
    # 展示顺序只影响阅读；按实验规则命名，不按测试表现高低排序。
    preferred = ["all", human, group]
    preferred += [f"{base}_{seed}" for base in ("random_human", "random_group", "human_group_budget") for seed in seeds]
    present = set(summary.get("training_regimes", {}))
    return [name for name in preferred if name in present] + sorted(present - set(preferred))


def _domains(counts: dict) -> str:
    # 这里的分母是本方案保留的来源组数，不是文本条数。
    # 一个来源有多条文本，混用两种计数会让领域占比失去含义。
    values = counts.get("domain_source_counts", {})
    total = sum(number for value in values.values() if (number := _number(value)) is not None)
    return "；".join(
        f"{_cell(domain)}: {_count(value)} ({_fmt(number / total, percent=True) if total and (number := _number(value)) is not None else '—'})"
        for domain, value in sorted(values.items())
    ) or "—"


def _finish(fig: plt.Figure, path: Path) -> None:
    # 保存后关闭 figure，防止连续生成多张图时一直占用内存。
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _empty(ax: plt.Axes, message: str) -> None:
    # 缺少保存结果时标明缺失，不画一张全为 0 的图冒充实际观测。
    ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _plot_admission(summary: dict, path: Path) -> None:
    """画训练来源的二乘二准入计数及保留来源的领域组成。"""
    diagnostics = summary.get("admission_diagnostics", {})
    cells = diagnostics.get("pass_cells", {})
    _, human, group, _ = _regime_names(summary)
    regimes = summary.get("training_regimes", {})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    if cells:
        # 横轴是两个训练 AI 是否都通过；纵轴是人类是否通过。
        # GROUP 是右下格，HUMAN 是整条下行。不能把 HUMAN-only 当成一类标签。
        matrix = [[_number(cells.get(f"human_{h}_ai_{a}", {}).get("source_groups")) for a in (0, 1)] for h in (0, 1)]
        # imshow 需要数值矩阵；缺失值仅在底色上暂用 0，格内文字仍显示“—”。
        display = [[0 if value is None else value for value in row] for row in matrix]
        ax.imshow(display, cmap="Blues", aspect="auto", vmin=0)
        ceiling = max((value for row in display for value in row), default=0)
        for h in (0, 1):
            for a in (0, 1):
                ax.text(a, h, _count(matrix[h][a]), ha="center", va="center", fontsize=15,
                        color="white" if ceiling and display[h][a] > .55 * ceiling else "#222222")
        ax.set_xticks((0, 1), ("At least one AI text fails", "Both training AI texts pass"), fontsize=8)
        ax.set_yticks((0, 1), ("Human fails", "Human passes"), fontsize=9)
        ax.set_title("Training-source admission cells\nGROUP = bottom right; HUMAN = bottom row", fontsize=10)
    else:
        _empty(ax, "No admission-cell counts")
    ax = axes[1]
    names = [name for name in ("all", human, group) if name in regimes]
    domains = sorted({domain for name in names for domain in regimes[name].get("domain_source_counts", {})})
    if not names or not domains:
        _empty(ax, "No domain source counts")
    else:
        width = .8 / len(names)
        for index, name in enumerate(names):
            # 每个领域是一组柱子；横向偏移让 ALL/HUMAN/GROUP 并排，而不是相互覆盖。
            positions = [j - .4 + width / 2 + index * width for j in range(len(domains))]
            values = [_number(regimes[name].get("domain_source_counts", {}).get(domain)) or 0 for domain in domains]
            ax.bar(positions, values, width=width, label=name, color=COLORS[index])
        ax.set_xticks(range(len(domains)), domains)
        ax.set(ylabel="Retained source groups", title="Domain composition of training sources")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Admission rules operate only within the original training partition", fontsize=12)
    # 给总标题留出顶部空间。tight_layout 只整理布局，不改变坐标中的数据。
    fig.tight_layout(rect=(0, 0, 1, .92))
    _finish(fig, path)


def _plot_lengths(summary: dict, path: Path) -> None:
    """比较来源方案的平均对数长度；不从聚合值虚构分布或个体差异。"""
    diagnostics = summary.get("admission_diagnostics", {})
    # 这些输入是均值等聚合统计，不能拿它们虚构直方图或个体观测。
    statistics = diagnostics.get("regime_length_statistics", {})
    _, human, group, _ = _regime_names(summary)
    names = [name for name in ("all", human, group) if name in statistics]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    if not names:
        for ax in axes:
            _empty(ax, "No training-length aggregates")
    else:
        ax = axes[0]
        for index, model in enumerate(("human", "chatgpt", "gpt4")):
            values = [_number(statistics[name].get("generator_stats", {}).get(model, {}).get("mean_log_words")) for name in names]
            ax.bar([j + (index - 1) * .24 for j in range(len(names))],
                   [float("nan") if value is None else value for value in values], .24, color=COLORS[index], label=model)
        ax.set_xticks(range(len(names)), names)
        ax.set(ylabel="Mean log(1 + whitespace word count)", title="Training text lengths by generator")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
        ax = axes[1]
        values = [_number(statistics[name].get("mean_ai_minus_human_log_words")) for name in names]
        # nan 让缺失的柱子空着；填 0 会把“没提供统计”误写成“AI 与人类平均长度相同”。
        ax.bar(range(len(names)), [float("nan") if value is None else value for value in values], color=COLORS[:len(names)])
        ax.axhline(0, color="#555555", linewidth=1, linestyle="--")
        # 这条零线是“来源内 AI 平均对数长度 = 人类对数长度”的参照，不是分类阈值。
        ax.set_xticks(range(len(names)), names)
        ax.set(ylabel="Mean paired log-length difference", title="AI minus human within source\nTwo seen generators averaged per source")
        ax.margins(y=.2)
        for index, value in enumerate(values):
            if value is not None:
                # 正值标签放柱顶上方，负值放下方，避免数字压在柱子上。
                ax.annotate(f"{value:+.3f}", (index, value), xytext=(0, 5 if value >= 0 else -14),
                            textcoords="offset points", ha="center", fontsize=9)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Descriptive training associations; no causal or distribution-matching claim", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .92))
    _finish(fig, path)


def _admission_rows(contrasts: list[dict], group: str) -> list[dict]:
    # AUROC 与阈值无关，保存表中有两份阈值记录；图只取固定 0.5 这份避免重复。
    return [row for row in contrasts if row.get("contrast_type") == "admission_control"
            and row.get("evaluation_cohort") == "all" and row.get("metric") == "auroc"
            and row.get("threshold_policy") == "default_0_5"
            and str(row.get("variant_a", "")).endswith(f"__{group}")
            and "__human_group_budget_" in str(row.get("variant_b", ""))]


def _plot_comparisons(summary: dict, contrasts: list[dict], path: Path) -> None:
    """在预设场景中显示 GROUP 减 HUMAN 等预算对照的 AUROC 差值。"""
    _, _, group, seeds = _regime_names(summary)
    rows = _admission_rows(contrasts, group)
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.8), squeeze=False)
    # 两行对应两个模型族，三列对应三个场景，位置按实验条件固定。
    for i, family in enumerate(FAMILIES):
        for j, scenario in enumerate(SCENARIOS):
            ax = axes[i, j]
            selected = [row for row in rows if row.get("family") == family and row.get("scenario") == scenario]
            if not selected:
                _empty(ax, "No saved admission contrast")
                continue
            for y, seed in enumerate(seeds):
                row = next((row for row in selected if str(row.get("variant_b", "")).endswith(f"_{seed}")), {})
                point, low, high = (_number(row.get(key)) for key in ("difference", "ci_low", "ci_high"))
                if point is None:
                    continue
                color = COLORS[y % len(COLORS)]
                # 线段是已算好的区间；圆点是 A-B 的点估计。这里不重新抽样。
                if low is not None and high is not None:
                    ax.plot((low, high), (y, y), color=color, linewidth=2)
                    ax.plot((low, high), (y, y), "|", color=color, markersize=7)
                ax.plot(point, y, "o", color=color, markersize=5)
            ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
            # 区间跨过零线表示本次区间包含“差值为零”；它不直接给出改进的因果解释。
            ax.set_yticks(range(len(seeds)), [f"HUMAN budget / seed {seed}" for seed in seeds], fontsize=8)
            ax.set_ylim(len(seeds) - .5, -.5)
            ax.set(title=f"{FAMILY_EN[family]}\n{SCENARIO_EN[scenario]}", xlabel="AUROC difference: GROUP - HUMAN budget")
            # 差值可能很小；稀疏刻度并关闭科学计数偏移，避免标签挤成一团。
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.grid(axis="x", alpha=.2)
    fig.suptitle("Same full test records; pointwise 95% component-cluster bootstrap intervals\nFixed trained models and thresholds; seeds are sensitivity controls", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .89))
    _finish(fig, path)


def _primary(summary: dict, contrasts: list[dict]) -> dict | None:
    _, _, group, seeds = _regime_names(summary)
    seed = seeds[0] if seeds else 42
    # 主要比较由实验计划指定；不能在出结果后挑一个差异最大的当主要发现。
    comparison_id = summary.get("plan", {}).get("primary_contrast", {}).get("comparison_id")
    return next((row for row in _admission_rows(contrasts, group)
                 if row.get("family") == "combined" and row.get("scenario") == "test_domain"
                 and (row.get("comparison_id") == comparison_id if comparison_id else
                      row.get("variant_b") == f"combined__human_group_budget_{seed}")), None)


def _contrast_table(rows: list[dict]) -> list[str]:
    # rows 是字典列表：一个字典就是一项模型对照。
    # 前两行是 Markdown 表头和分隔线，循环再把每个字典补成一行。
    lines = ["| 对照 | 模型族 | 场景 | A | B | A AUROC | B AUROC | A − B | 逐项 95% 区间 | 有效重采样 / 组件 |",
             "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| `{_cell(row.get('comparison_id'))}` | `{_cell(row.get('family'))}` | "
                     f"{SCENARIO_ZH.get(str(row.get('scenario')), _cell(row.get('scenario')))} | "
                     f"`{_cell(row.get('variant_a'))}` | `{_cell(row.get('variant_b'))}` | "
                     f"{_fmt(row.get('value_a'))} | {_fmt(row.get('value_b'))} | {_delta(row.get('difference'))} | "
                     f"{_interval(row)} | {_count(row.get('valid_resamples'))} / {_count(row.get('bootstrap_clusters'))} |")
    if not rows:
        lines.append("| 未提供此类保存对照 | — | — | — | — | — | — | — | — | — |")
    return lines


def _metric_table(metrics: list[dict], policy: str) -> list[str]:
    lines = ["| 模型 | 场景 | AUROC | AI F1 | AI 检出 / 总数 | 人类误报 / 总数 | AI 召回率 | 人类误报率 | 阈值 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics:
        # 保存表同时含两套阈值；每张表只选当前 policy，避免把两套成绩混在一起。
        if row.get("threshold_policy") != policy or row.get("evaluation_cohort") != "all":
            continue
        lines.append(f"| `{_cell(row.get('variant'))}` | {SCENARIO_ZH.get(str(row.get('scenario')), _cell(row.get('scenario')))} | "
                     f"{_fmt(row.get('auroc'))} | {_fmt(row.get('f1'))} | {_count(row.get('tp'))} / {_count(row.get('ai_rows'))} | "
                     f"{_count(row.get('fp'))} / {_count(row.get('human_rows'))} | {_fmt(row.get('recall'), percent=True)} | "
                     f"{_fmt(row.get('human_fpr'), percent=True)} | {_fmt(row.get('threshold'), digits=6)} |")
    return lines


def _coefficient_text(summary: dict) -> str:
    _, human, group, _ = _regime_names(summary)
    variants = summary.get("variants", {})
    pairs = [(regime, _number(variants.get(f"length__{regime}", {}).get("length_coefficient"))) for regime in ("all", human, group)]
    available = [(name, value) for name, value in pairs if value is not None]
    if not available:
        return "未提供长度模型系数，不能据此判断长度—标签关联的方向。"
    signs = {1 if value > 0 else -1 if value < 0 else 0 for _, value in available}
    # 系数符号取自实际拟合结果，不预设“本研究一定发生反转”。
    detail = "；".join(f"`{name}` = {value:+.6f}" for name, value in available)
    if 1 in signs and -1 in signs:
        return detail + "。这些拟合系数中出现正负符号反转；它描述所选训练样本中的关联方向，不证明筛选的因果机制。"
    return detail + "。已提供的系数没有出现正负符号反转；不要把系数幅度变化自动解释为排序能力变化。"


def write_report(output: Path, summary: dict, metrics: list[dict], contrasts: list[dict]) -> list[str]:
    """生成中文 Markdown 与三张英文标注 PNG，返回相对产物路径。

    所有数值均来自调用者保存的汇总结果，不修改输入、不重新拟合，也不按测试表现挑选方案。
    """
    # summary 是各方案的汇总字典；metrics 和 contrasts 是可逐行展开的字典列表。
    # 这些结果由 experiment.py 算好后传入。这里不读测试正文，也不根据表现删除某个方案。
    # 本研究使用已看过的测试场景，报告保留探索性标记。
    if summary.get("scope") != "exploratory":
        raise ValueError("The admission report requires scope='exploratory'.")
    output = Path(output)
    report_dir = output / "reports"
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    minimum, human, group, seeds = _regime_names(summary)
    plan = summary.get("plan", {})
    admission = summary.get("admission_diagnostics", {})
    validation = summary.get("validation", {})
    # 先生成图，再在下方 Markdown 中用相对路径引用；移动整个输出目录仍可阅读。
    _plot_admission(summary, figures / "admission_counts.png")
    _plot_lengths(summary, figures / "length_relationships.png")
    _plot_comparisons(summary, contrasts, figures / "admission_comparison.png")
    primary = _primary(summary, contrasts)
    primary_seed = seeds[0] if seeds else 42
    primary_b = primary.get("variant_b") if primary else f"combined__human_group_budget_{primary_seed}"
    # lines 是字符串列表。每个空字符串对应 Markdown 的一个空行。
    # 最后用 "\n".join(lines) 连接成整篇报告；不把数字写成固定的历史结果。
    lines = ["# 文本来源准入条件审计：HUMAN 与 GROUP", "",
             "本研究是 **exploratory（探索性）分析**，继续使用已查看过的 RAID v1 测试场景。"
             "本研究不提供独立确认，也不依据测试表现选出冠军。它不是新独立数据验证或因果实验。", "",
             f"冻结数据：`{_cell(summary.get('data_dir', '未提供'))}`。长度采用按空白分隔的词数，门槛为 **{minimum} 词**。"
             "报告只读取保存的汇总数值，不展示原始文本。", "",
             "## 1. 这次改变的是什么", "",
             "| 方案 | 训练来源的准入条件 | 来源通过后保留什么 |",
             "|---|---|---|",
             "| ALL | 原有完整训练来源 | 同来源的人类、ChatGPT 和 GPT-4 全文 |",
             f"| HUMAN / `{human}` | 只要求该来源的人类文本至少 {minimum} 词 | 同来源三条全文，包括可能未达到门槛的 AI 文本 |",
             f"| GROUP / `{group}` | 人类、ChatGPT 和 GPT-4 文本全部至少 {minimum} 词 | 同来源三条全文 |", "",
             "这些规则只作用于原训练分区，不用 Mistral 决定训练准入，不裁剪全文，不独立删除组内某条文本。"
             "GROUP 是 HUMAN 的子集。所有模型使用同一完整验证集，并在相同的完整测试记录上评估；本研究没有长文本测试子集。", "",
             "## 2. 实际筛选改变了多少来源", "",
             f"训练来源总数：**{_count(admission.get('total_sources'))}**；HUMAN：**{_count(admission.get('human_sources'))}**；"
             f"GROUP：**{_count(admission.get('group_sources'))}**；HUMAN 通过而 GROUP 未通过：**{_count(admission.get('human_only_sources'))}**。", ""]
    identical = admission.get("identical_human_group")
    # 先检查两组实际来源是否不同。有时更换门槛后两组会完全相同，不能照抄旧结论。
    if identical is True or _number(admission.get("human_only_sources")) == 0:
        lines += ["**本次 HUMAN 与 GROUP 实际保留的来源集合相同。** 当前门槛没有形成可识别的额外 AI 准入差异；"
                  "不能声称观察到了额外筛除 AI 短文本的机制。应把这是一个未形成组间差异的对照事实报告出来。", ""]
    elif identical is False or (_number(admission.get("human_only_sources")) or 0) > 0:
        lines += ["本次 HUMAN 与 GROUP 实际保留了不同来源。HUMAN-only 来源的人类文本通过门槛，"
                  "但至少一个训练生成器的 AI 文本未通过；这个集合差异可以被审计。"
                  "它同时改变来源内容和长度关系，不能仅凭模型差异认定因果机制。", ""]
    else:
        lines += ["本次输入未完整提供集合差异诊断，不能推断 HUMAN 与 GROUP 是否保留相同来源。", ""]
    lines += ["| 人类文本通过？ | 两个训练 AI 文本都通过？ | 来源组 | 各领域来源数与占比 |",
              "|---|---|---:|---|"]
    for h in (0, 1):
        for a in (0, 1):
            # 四个格子覆盖“人类是否通过 × 两条训练 AI 是否都通过”的全部组合。
            row = admission.get("pass_cells", {}).get(f"human_{h}_ai_{a}", {})
            lines.append(f"| {'是' if h else '否'} | {'是' if a else '否'} | {_count(row.get('source_groups'))} | {_domains(row)} |")
    failures = admission.get("generator_failure_counts", {})
    lines += ["", "单项未达到门槛的训练来源数：" + "；".join(f"`{model}`：{_count(failures.get(model))}" for model in ("human", "chatgpt", "gpt4"))
              + "。同一来源可能在多个单项上失败，不能把这三个数相加当成不同来源总数。", "",
              "HUMAN-only 来源中的 AI 单项失败数：" + "；".join(
                  f"`{model}`：{_count(admission.get('human_only_failure_counts', {}).get(model))}" for model in ("chatgpt", "gpt4"))
              + "；这些来源的领域组成：" + _domains({"domain_source_counts": admission.get("human_only_domain_counts", {})}) + "。", "",
              "![训练来源准入与领域计数](figures/admission_counts.png)", "",
              "## 3. 训练预算对照与固定的验证集", "",
              "| 训练方案 | 文本数 | 来源组 | 近重复关联组件 | 人类 | AI | 各领域来源组数与占比 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for name in _ordered_regimes(summary):
        row = summary["training_regimes"][name]
        lines.append(f"| `{_cell(name)}` | " + " | ".join(_count(row.get(key)) for key in ("rows", "source_groups", "component_groups", "human_rows", "ai_rows"))
                     + f" | {_domains(row)} |")
    # 等预算指来源数和领域配额相同。它没有让每条文本词数相同，也没有匹配主题或文体。
    # 因此只能报告这种控制下的关联差异，不能写成“已经排除所有其他因素”。
    lines += ["", "`random_human_<seed>` 从 ALL 按领域匹配 HUMAN 的来源预算；`random_group_<seed>` 从 ALL 匹配 GROUP 的来源预算。"
              "`human_group_budget_<seed>` 从 HUMAN 中按领域抽样，使来源数量与领域配额等于 GROUP。"
              "每个来源在本训练数据中均有一条人类和两条 AI 文本，因此相同的来源预算也匹配文本数、类别数和生成器数。", "",
              "**预算匹配不等于长度分布匹配。** 词数、主题、文体和其他组成差异仍然存在。"
              "各子集可共享来源，多个种子是预算敏感性对照，不是独立研究。近重复关联组件保持原分区归属；"
              "某个训练子集可能仅保留组件内的部分来源，这不允许组件跨到验证或测试。", "",
              f"所有模型共用完整验证集：**{_count(validation.get('rows'))} 条文本、{_count(validation.get('source_groups'))} 个来源组、"
              f"{_count(validation.get('human_rows'))} 条人类文本、{_count(validation.get('ai_rows'))} 条 AI 文本**。"
              f"候选 C：`{_cell(plan.get('C_values', [.5, 1, 2]))}`；预算种子：`{_cell(seeds)}`。"
              "特征词表和分类器仅拟合各自训练子集；C 仅由验证 AUROC 选择，分类阈值在验证集上固定。", "",
              "## 4. 来源中的长度关系与拟合方向", "",
              "| 训练方案 | 生成器 | 文本数 | 平均词数 | 中位词数 | 平均 log(1 + 词数) |",
              "|---|---|---:|---:|---:|---:|"]
    length_statistics = admission.get("regime_length_statistics", {})
    # 一层循环取训练方案，下一层取 human/chatgpt/gpt4 的已保存长度统计。
    for name in _ordered_regimes(summary):
        statistics = length_statistics.get(name, {})
        for model, row in statistics.get("generator_stats", {}).items():
            lines.append(f"| `{_cell(name)}` | `{_cell(model)}` | {_count(row.get('rows'))} | {_fmt(row.get('mean_words'), digits=2)} | "
                         f"{_fmt(row.get('median_words'), digits=2)} | {_fmt(row.get('mean_log_words'))} |")
    lines += ["", "| 训练方案 | 来源内 AI 平均 log 长度减人类 log 长度的均值 |", "|---|---:|"]
    for name in _ordered_regimes(summary):
        lines.append(f"| `{_cell(name)}` | {_delta(length_statistics.get(name, {}).get('mean_ai_minus_human_log_words'))} |")
    lines += ["", "同一来源先对 ChatGPT 和 GPT-4 的 log(1 + 词数) 取平均，再减去该来源人类文本的 log 长度，最后对来源取均值。"
              "这是训练样本中的描述性关系，不能从均值推断完整分布、具体文本质量或真实作者习惯。", "",
              "![训练来源的长度关系](figures/length_relationships.png)", "",
              _coefficient_text(summary), "",
              "长度模型只有标准化后的 log(1 + 词数) 一个特征。非零系数的符号决定排序方向；"
              "在相同测试记录上，同符号的此类模型通常给出相同 AUROC，系数反号产生相反排序。"
              "这是单调函数的数学性质，本身不是新算法发现。长度模型的变化也不证明 combined 使用了长度捷径。", "",
              "## 5. 预先指定的主要探索比较", "",
              f"主要探索比较为 `combined__{group} − {_cell(primary_b)}` 在 Wiki / `all` 上的 AUROC。"
              "计划中其余预算种子是支持性敏感性对照；其余模型族、场景和阈值是次要探索。"
              "这个指定发生在旧测试结果已经查看之后，不能改称未见数据上的预注册确认。", ""]
    lines += _contrast_table([primary] if primary else [])
    # 图路径以 report_zh.md 所在的 reports/ 为起点。
    # 整个输出目录一起移动后，figures/ 图片和 ../results/ 表格仍保持相对位置。
    lines += ["", "![GROUP 减 HUMAN 等预算对照的 AUROC 差值](figures/admission_comparison.png)", "",
              "差值统一为 A − B。AUROC 或 AI 召回率为正表示对应数值更高；人类误报率为正表示更多人类被误判。"
              "不能只选改善的场景来声称整体改进，也不根据此图挑选测试冠军。", "",
              "## 6. 全部预设 AUROC 对照", ""]
    for kind, title in (("training_selection", "筛选训练减 ALL"), ("budget_control", "筛选训练减 ALL 随机等预算"),
                        ("admission_control", "GROUP 减 HUMAN 等预算")):
        lines += [f"### {title}", ""]
        lines += _contrast_table([row for row in contrasts if row.get("contrast_type") == kind
                                  and row.get("metric") == "auroc" and row.get("threshold_policy") == "default_0_5"])
        lines += [""]
    lines += ["AUROC 不依赖阈值，上表仅显示固定 0.5 的记录以免重复。"
              "[完整差值 CSV](../results/contrasts.csv) 同时保存两个阈值下的 AUROC、AI 召回率和人类误报率及区间。", "",
              f"计划进行 {_count(plan.get('bootstrap_resamples', 1000))} 次**近重复关联组件聚类 bootstrap**。"
              "同场景的 A、B 在同一次组件抽样上配对计算，保留来源内和近重复关联文本的相关结构；"
              "若一次抽样缺少某一类别，该次对照不计算，实际有效次数见表。", "",
              "区间是逐项 95% 区间，**条件于已拟合模型、已选 C 和已固定阈值**，不包含重新训练、训练来源重抽样、"
              "C 选择或阈值校准的不确定性。没有做多重比较校正。区间不含零不等于独立确认，"
              "组件构造也不能证明所有组件在统计上完全独立。", "",
              "## 7. 同一完整测试人群的操作指标", "",
              "AUROC 衡量 AI 与人类文本的分数排序；AI F1 同时受类别比例和阈值影响。"
              "标签 1 = AI、0 = 人类，模型分数不是得到保证的作者身份概率。"
              "三个场景可能共享人类对照，不能当成三次独立研究，也不能直接相加作为唯一文本总量。", "",
              "### 固定 0.5", ""]
    lines += _metric_table(metrics, "default_0_5")
    lines += ["", "### 同一完整验证集确定的 FPR 阈值", "",
              f"验证 FPR 策略的经验目标为 {_fmt(plan.get('target_validation_fpr', .05), percent=True)}。"
              "这只约束有限验证人类样本上的经验误报率，不能保证未知领域或生成器达到相同误报率。", ""]
    lines += _metric_table(metrics, "validation_fpr")
    lines += ["", "## 8. 模型选择与审计记录", "",
              "| 模型 | 验证选择 C | 验证 AUROC | 长度系数 | 固定阈值 | 验证 FPR 阈值 | 该阈值的验证 FPR |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in summary.get("variants", {}).items():
        # 没有长度系数的文本模型显示“—”。不能填 0，因为那会误称它拟合了零长度系数。
        thresholds = row.get("thresholds", {})
        default, calibrated = thresholds.get("default_0_5", {}), thresholds.get("validation_fpr", {})
        lines.append(f"| `{_cell(name)}` | {_fmt(row.get('best_C'), digits=3)} | {_fmt(row.get('validation_metrics', {}).get('auroc'))} | "
                     f"{_delta(row.get('length_coefficient'))} | {_fmt(default.get('threshold'), digits=6)} | "
                     f"{_fmt(calibrated.get('threshold'), digits=6)} | {_fmt(calibrated.get('validation_fpr'), percent=True)} |")
    diagnostics = summary.get("diagnostics", {})
    # 审计表展示实际保存的布尔值，不因为图表生成成功就自动把所有检查填成 True。
    lines += ["", "审计状态（来自实际运行汇总）：", "",
              "| 检查项 | 保存值 |", "|---|---|"]
    for key in ("same_validation_for_all", "no_test_based_winner", "all_tests_previously_observed", "data_unchanged"):
        lines.append(f"| `{key}` | `{_cell(diagnostics.get(key))}` |")
    lines += ["", "## 9. 复现与结论边界", "",
              "本研究可以审计 HUMAN 与 GROUP 的实际集合差异，并在固定测试记录和领域来源预算下比较训练方案。"
              "它不提供因果证明，不证明长度是唯一解释，也没有覆盖新的独立数据、其他生成设置或所有检测器。"
              "后续确认应先固定研究问题、准入规则、模型、指标和阈值，再取得未参与当前探索的新来源；"
              "从 RAID 再抽来源与外部数据集验证分别回答不同层次的问题。", "",
              "从项目根目录使用一个新的输出目录，才能亲自重新训练：", "", "```bash",
              "python -m aitextdetection.experiment --config settings.json --output-dir output/run_01", "```", "",
              "同一已完成目录通常执行登记产物校验并复用；修改代码或配置后应使用新目录，保留旧产物。"
              "[完整指标 CSV](../results/metrics.csv)、[完整差值 CSV](../results/contrasts.csv)、"
              "[汇总 JSON](../results/summary.json) 和输出根目录的 `protocol.json` 保存可检查的数值与协议。", "",
              f"本报告呈现 **{len(metrics):,} 行指标、{len(contrasts):,} 行对比**。"
              "报告与三张图只消费已有汇总，不重新训练、不重新计算 bootstrap，也不添加未运行的结果。", ""]
    # UTF-8 保存中文；返回相对路径，供上层登记产物和后续文件核验。
    (report_dir / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    return ["reports/report_zh.md", "reports/figures/admission_counts.png", "reports/figures/length_relationships.png",
            "reports/figures/admission_comparison.png"]
