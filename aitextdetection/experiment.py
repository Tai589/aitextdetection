"""执行来源准入敏感性实验：训练、验证、冻结、测试、统计与报告。

所有方案共用完整验证集和测试记录，参数选择只使用验证数据。
"""

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from aitextdetection import dataset as data_io
from aitextdetection import classifiers as models
from aitextdetection.bootstrap import paired_cluster_intervals
from aitextdetection.sampling import build_training_regimes, describe_selection
from aitextdetection.dataset import DataError, file_sha256, write_json, fingerprint, verify_completed_run


# __file__ 是本文件的位置。向上两层得到项目根目录，因此换用户名或移动文件夹也能运行。
ROOT = Path(__file__).resolve().parents[1]
VERSION = "aitextdetection-study-v1"




# 先检查配置，再开始训练。写错门槛或种子时，尽早报错，不生成半套实验结果。
# C 是逻辑回归的正则化参数；最多比较三个 C、三个抽样种子，限制本研究计算量。
def load_settings(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    # scope 说明本研究继续使用看过的测试数据，属于探索分析。
    # 即使程序成功跑完，也不能把这份结果改称新数据上的独立确认。
    if spec.get("version") != VERSION or spec.get("scope") != "exploratory_previously_observed_tests":
        raise DataError("本入口只支持既有测试上的探索性来源准入协议。")
    required = ("minimum_words", "matched_seeds", "C_values", "target_validation_fpr",
                "bootstrap_resamples", "bootstrap_seed")
    if any(key not in spec for key in required):
        raise DataError("准入协议缺少必要数值设置。")
    # seeds 决定随机对照抽中哪些来源；同样的输入和种子应产生同样的对照。
    # 下方同时检查类型、范围、有限值和重复项，避免把 True、NaN 等误当成正常参数。
    n, seeds, cs = spec["minimum_words"], spec["matched_seeds"], spec["C_values"]
    if (type(n) is not int or n < 20 or not isinstance(seeds, list) or not 1 <= len(seeds) <= 3
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != len(seeds)
            or not isinstance(cs, list) or not 1 <= len(cs) <= 3
            or any(type(c) not in (int, float) or not np.isfinite(c) or c <= 0 for c in cs)
            or len(set(cs)) != len(cs)
            or type(spec["target_validation_fpr"]) not in (int, float) or not 0 < spec["target_validation_fpr"] < 1
            or type(spec["bootstrap_resamples"]) is not int or not 20 <= spec["bootstrap_resamples"] <= 2000
            or type(spec["bootstrap_seed"]) is not int or not 0 <= spec["bootstrap_seed"] < 2**32):
        raise DataError("长度、种子、C、验证FPR或bootstrap设置不合法；最多三个种子和三个C。")
    return spec



def experiment_identity(project: Path, data_dir: Path, settings: dict, manifest: dict) -> dict:
    """记录当前代码、计算参数、派生数据来源和实际依赖环境。"""
    # 身份按内容计算，整个项目一起移动不会改变来源文件的相对路径。
    # 源码注释也影响摘要；改完代码后应另开输出目录运行。
    return {
        "implementation": "aitextdetection-v1",
        "settings": settings,
        "parameters": data_io.experiment_parameters(settings),
        "dataset_manifest_sha256": file_sha256(data_dir / "manifest.json"),
        "source_dataset_manifest_sha256": manifest["origin"]["parent_run_manifest_sha256"],
        "dataset_files": manifest["files"],
        "environment": data_io.environment_record(),
        "source_files": {str(p.relative_to(project)): file_sha256(p)
                         for p in sorted((project / "aitextdetection").glob("*.py"))},
    }


# 一个 variant 是“模型族 + 训练方案”，例如 combined__human_200。
# 每个模型对每个场景只计算一次分数；两种阈值共用分数，避免重复预测。
def predict_matrix(tests: dict, pipelines: dict, variants: dict, assignments: dict) -> tuple:
    """所有模型都评估相同完整测试记录；不再筛选测试长度。"""
    metrics, predictions, scores_by_case = [], [], {}
    for name, record in variants.items():
        for scenario, rows in tests.items():
            scores = models.ai_scores(pipelines[name], [r["text"] for r in rows])
            scores_by_case[name, scenario] = scores
            labels = np.array([r["label"] for r in rows])
            # predictions 中一行对应一次模型预测。保留来源和组件编号，方便追查错误。
            # component 用于后面的成组抽样；model_score 不是得到保证的作者身份概率。
            for row, score in zip(rows, scores, strict=True):
                predictions.append({"variant": name, "family": record["family"],
                    "training_regime": record["training_regime"], "scenario": scenario,
                    "id": row["id"], "source_id": row["source_id"],
                    "component": assignments[row["source_id"]]["component"],
                    "domain": row["domain"], "generator": row["model"],
                    "true_label": row["label"], "model_score": float(score), "word_count": row["word_count"]})
            # 阈值只决定如何把分数变成 0/1；改变阈值不会重新训练，也不改变 AUROC。
            for policy, setting in record["thresholds"].items():
                result = models.calculate_metrics(labels, scores, setting["threshold"])
                matrix = result.pop("confusion_matrix")
                metrics.append({"variant": name, "family": record["family"],
                    "training_regime": record["training_regime"], "evaluation_cohort": "all",
                    "scenario": scenario, "threshold_policy": policy, "threshold": setting["threshold"],
                    "rows": len(rows), "source_groups": len({r["source_id"] for r in rows}),
                    "human_rows": int(sum(labels == 0)), "ai_rows": int(sum(labels == 1)),
                    **result, **matrix, "validation_humans": record["validation_humans"],
                    "validation_fpr": setting["validation_fpr"]})
    return metrics, predictions, scores_by_case



# 比较方向统一为 a - b。先写好要比哪两组，再看测试结果，避免只挑改善项。
# 默认三个种子时：2 项“筛选减 ALL” + 3 × 3 项预算对照，共 11 项。
def comparison_plan(settings: dict) -> list[dict]:
    """比较在读取测试前固定；seed42为主要探索，其余种子是敏感性检查。"""
    n, seeds = settings["minimum_words"], settings["matched_seeds"]
    comparisons = [
        {"comparison_id": "human_minus_all", "contrast_type": "training_selection", "a": f"human_{n}", "b": "all"},
        {"comparison_id": "group_minus_all", "contrast_type": "training_selection", "a": f"group_{n}", "b": "all"},
    ]
    for seed in seeds:
        comparisons.extend([
            {"comparison_id": f"human_minus_random_{seed}", "contrast_type": "budget_control",
             "a": f"human_{n}", "b": f"random_human_{seed}"},
            {"comparison_id": f"group_minus_random_{seed}", "contrast_type": "budget_control",
             "a": f"group_{n}", "b": f"random_group_{seed}"},
            {"comparison_id": f"group_minus_human_budget_{seed}", "contrast_type": "admission_control",
             "a": f"group_{n}", "b": f"human_group_budget_{seed}"},
        ])
    return comparisons



# A、B 共用同一次组件抽样，才是配对比较。
# 一个组件里可能包含同来源或近重复文本，不能把这些行当成互不相关的抽样单位。
def contrast_intervals(tests: dict, variants: dict, scores: dict, assignments: dict, settings: dict) -> list[dict]:
    """同一场景共同重采样近重复组件，只衡量固定模型的测试抽样差异。"""
    contrasts = []
    # 用配置中的第一个种子标记主要比较；这里没有寻找差值最大的种子。
    primary_id = f"group_minus_human_budget_{settings['matched_seeds'][0]}"
    for comparison in comparison_plan(settings):
        for family in models.FAMILIES:
            a, b = f"{family}__{comparison['a']}", f"{family}__{comparison['b']}"
            for index, (scenario, rows) in enumerate(tests.items()):
                labels = [r["label"] for r in rows]
                components = [assignments[r["source_id"]]["component"] for r in rows]
                for policy in ("default_0_5", "validation_fpr"):
                    ta, tb = variants[a]["thresholds"][policy]["threshold"], variants[b]["thresholds"][policy]["threshold"]
                    result = paired_cluster_intervals(labels, scores[a, scenario], scores[b, scenario], components,
                        ta, tb, seed=settings["bootstrap_seed"] + index, n_resamples=settings["bootstrap_resamples"])
                    # 固定同一场景的抽样种子，让各项比较面对相同的组件重采样。
                    # 区间只反映现有模型在这些测试组件上的抽样波动，不包含重新训练的波动。
                    for metric in ("auroc", "recall", "human_fpr"):
                        point, interval = result["point"][metric], result["intervals"][metric]
                        contrasts.append({"comparison_id": comparison["comparison_id"],
                            "contrast_type": comparison["contrast_type"], "family": family,
                            "scenario": scenario, "threshold_policy": policy, "evaluation_cohort": "all",
                            "variant_a": a, "variant_b": b, "metric": metric,
                            # 只给一个“比较、模型族、场景、指标、阈值”组合标记为主要项。
                            # 主要项也是既有测试上的探索结果，不会因此变成预注册确认。
                            "is_primary": int(comparison["comparison_id"] == primary_id and family == "combined"
                                and scenario == "test_domain" and metric == "auroc" and policy == "default_0_5"),
                            "value_a": point["a"], "value_b": point["b"], "difference": point["difference"],
                            "ci_low": interval["low"], "ci_high": interval["high"],
                            "bootstrap_clusters": result["n_clusters"], "valid_resamples": result["valid_resamples"],
                            "bootstrap_seed": settings["bootstrap_seed"] + index})
        print(f"  配对区间：{comparison['comparison_id']}", flush=True)
    return contrasts



# 按这条顺序读本函数：检查输入 → 保存计划 → 建立训练方案 → 训练和验证 → 冻结 → 读测试 → 汇总。
# “冻结”是把模型、参数、阈值和数据清单保存并校验；它不是删除文件或让文件永久只读。
def run_experiment(data_dir: Path, output: Path, settings_path: Path, project: Path = ROOT) -> dict:
    project, data_dir, output = project.resolve(), data_dir.resolve(), output.resolve()
    data_io.check_output(project, data_dir, output)
    settings = load_settings(settings_path.resolve())
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    data_io.check_registered_files(data_dir, manifest,
        ("dataset_config.json", "split_manifest.json", "train.csv", "validation.csv"))
    identity = experiment_identity(project, data_dir, settings, manifest)
    if output.exists() and any(output.iterdir()):
        # 非空目录不能直接覆盖。只有身份和所有登记产物都一致，才返回已经完成的运行。
        # 想亲自重训，使用 output/run_02 这样的新目录；失败目录也保留，便于排查。
        # 先确认旧目录确实完成。失败且未冻结的目录不能提前触碰测试文件。
        completed = verify_completed_run(output, identity)
        data_io.check_registered_files(data_dir, manifest)
        print("已有准入审计的输入、代码与输出摘要一致；复用完成结果。全新目录才会重新训练。", flush=True)
        return completed
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # 记录“从输出目录走到输入/源码”的相对路径。
    # 只要包内目录关系不变，整个文件夹移动后仍能找到它们。
    status = {"status": "running", "identity": identity, "scope": "exploratory",
              "data_dir": os.path.relpath(data_dir, output),
              "code_root": os.path.relpath(project, output),
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "run_manifest.json", status)
    try:
        # original 是正式实验的配置，负责解释冻结数据的字段、划分和模型设置。
        # settings 是本研究的设置，负责 200 词门槛、随机对照种子和区间抽样次数。
        # 深拷贝后只替换本研究要比较的 C，保留 original 供数据校验使用。
        original = json.loads((data_dir / "dataset_config.json").read_text(encoding="utf-8"))
        config = copy.deepcopy(original)
        config["training"]["C_values"] = settings["C_values"]
        plan = {**settings, "comparisons": comparison_plan(settings),
            "primary_contrast": {"comparison_id": f"group_minus_human_budget_{settings['matched_seeds'][0]}",
                "family": "combined", "scenario": "test_domain", "metric": "auroc", "threshold_policy": "default_0_5"},
            "admission_rule": "HUMAN qualifies using the human row only; GROUP uses human/chatgpt/gpt4 in original train only; keep complete source groups. No Mistral eligibility.",
            "validation_rule": "One full original validation set for every model; C by AUROC, exact ties choose smaller C; thresholds only from validation.",
            "evaluation_rule": "Identical ALL records for each of the three previously observed test scenarios; no test length filtering or winner selection.",
            "interval_rule": "1000 by default, paired component-bootstrap percentile95%; fixed models/thresholds; pointwise, no multiplicity adjustment; no retraining/calibration uncertainty.",
            "interpretation": "Observational source-admission sensitivity; domain/source budget controls are not length distribution matching or causal identification."}
        write_json(output / "protocol.json", plan)
        write_json(output / "settings.json", settings)
        write_json(output / "dataset_config.json", config)
        print("[1/6] 保存探索协议；只读取冻结训练/验证数据，建立HUMAN/GROUP及等预算对照…", flush=True)
        training, validation, lineage = data_io.load_development_data(data_dir, original)
        data_io.validate_word_counts(validation)
        development = {"train": training, "validation": validation}
        split = json.loads((data_dir / "split_manifest.json").read_text(encoding="utf-8"))
        assignments = split["source_assignments"]
        # 默认得到 12 个训练方案。每个方案都保留通过筛选来源的三条全文，不能独立删掉短 AI 行。
        # 后续每个方案训练 length、combined 两类模型，因此一共 12 × 2 = 24 个模型。
        regimes = build_training_regimes(training, settings["minimum_words"], settings["matched_seeds"])
        diagnostics, tables = describe_selection(training, regimes, settings["minimum_words"])
        for name, rows in tables.items():
            data_io.write_csv(output / f"results/{name}.csv", rows)
        write_json(output / "results/admission_diagnostics.json", diagnostics)
        all_sources = {r["source_id"] for r in training}
        group_sources = {r["source_id"] for r in regimes[f"group_{settings['minimum_words']}"]}
        training_counts = {name: data_io.counts(rows, assignments, all_sources, group_sources) for name, rows in regimes.items()}
        # 不只保存“用了多少条”，还保存具体来源和文本 ID。
        # 两次运行数量相同仍可能抽中了不同来源，清单可以查清这种差别。
        write_json(output / "results/training_sources.json", {n: sorted({r["source_id"] for r in rs}) for n, rs in regimes.items()})
        write_json(output / "results/training_ids.json", {n: [r["id"] for r in rs] for n, rs in regimes.items()})
        write_json(output / "results/validation_ids.json", [r["id"] for r in validation])
        print(f"  ALL={diagnostics['total_sources']}；HUMAN={diagnostics['human_sources']}；GROUP={diagnostics['group_sources']}；HUMAN独有={diagnostics['human_only_sources']} 个来源。", flush=True)
        print("[2/6] 训练长度和文本模型；全部共用完整验证集…", flush=True)
        # 所有方案使用同一份 validation，不按长度再次筛选验证集。
        # 这样选 C、定阈值时面对的是相同记录，减少额外的数据组成变化。
        variants, pipelines = models.fit_variants(regimes, validation, config, settings, output)
        frozen_path = output / "results/frozen_selection.json"
        frozen = data_io.freeze_protocol({"version": VERSION, "scope": "exploratory", "plan_sha256": fingerprint(plan),
            "split_manifest_sha256": file_sha256(data_dir / "split_manifest.json"), "lineage": lineage,
            "test_files": {n: split["files"][n] for n in data_io.SCENARIOS}, "models": variants,
            "training_ids_sha256": file_sha256(output / "results/training_ids.json"),
            "validation_ids_sha256": file_sha256(output / "results/validation_ids.json")}, frozen_path)
        print("[3/6] 模型、C和阈值已冻结；读取三个既有完整测试场景…", flush=True)
        # 上面训练完成并生成冻结清单之后，才允许加载测试正文。
        # 之后只评价既定方案，不再根据测试成绩选择 C、阈值或冠军模型。
        data_io.check_registered_files(data_dir, manifest)
        tests = data_io.load_test_sets(data_dir, original, frozen, development, frozen_path)
        write_json(output / "results/evaluation_ids.json", {n: [r["id"] for r in rs] for n, rs in tests.items()})
        metrics, predictions, scores = predict_matrix(tests, pipelines, variants, assignments)
        data_io.write_csv(output / "results/metrics.csv", metrics)
        data_io.write_csv(output / "results/predictions.csv", predictions)
        print("[4/6] 固定相同测试记录，按近重复组件计算配对差值区间…", flush=True)
        contrasts = contrast_intervals(tests, variants, scores, assignments, settings)
        data_io.write_csv(output / "results/contrasts.csv", contrasts)
        summary = {"scope": "exploratory", "data_dir": os.path.relpath(data_dir, output), "plan": plan,
            "admission_diagnostics": diagnostics, "training_regimes": training_counts,
            "validation": data_io.counts(validation, assignments), "variants": variants,
            "evaluation_cohorts": {"all": {n: data_io.counts(rs, assignments) for n, rs in tests.items()}},
            "diagnostics": {"same_validation_for_all": len({v["validation_ids_sha256"] for v in variants.values()}) == 1,
                "same_full_tests_for_all": True, "prediction_rows": len(predictions),
                "no_test_based_winner": True, "all_tests_previously_observed": True, "data_unchanged": True}}
        write_json(output / "results/summary.json", summary)
        print("[5/6] 生成来源准入、长度关系图和中文报告…", flush=True)
        # 报告函数接收已经算好的结果，只负责排表和画图，不再拟合模型。
        from aitextdetection.reporting import write_report
        write_report(output, summary, metrics, contrasts)
        print("[6/6] 检查冻结输入及代码身份，登记输出摘要…", flush=True)
        data_io.check_registered_files(data_dir, manifest)
        if identity != experiment_identity(project, data_dir, settings, manifest):
            raise DataError("运行期间代码、环境或参考清单改变；保留本目录，另用新名称重跑。")
        # 主清单不能登记自己的摘要，否则每次写入摘要都会再次改变自身内容。
        # 缓存属于画图工具的临时文件，也不作为研究产物登记。
        outputs = {str(p.relative_to(output)): file_sha256(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "run_manifest.json" and ".cache" not in p.relative_to(output).parts}
        status.update({"status": "completed", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started, "variants": len(variants),
            "metric_rows": len(metrics), "contrast_rows": len(contrasts), "prediction_rows": len(predictions),
            "data_unchanged": True, "outputs": outputs})
        write_json(output / "run_manifest.json", status)
        print(f"实验完成：{len(variants)} 个模型，{len(metrics)} 行指标，{len(contrasts)} 行配对差值；冻结输入保持不变。", flush=True)
        return status
    except Exception as exc:
        # 失败时留下错误信息和已产生的文件；不把未完成运行标为成功。
        status.update({"status": "failed", "error": str(exc)})
        write_json(output / "run_manifest.json", status)
        raise



# python -m aitextdetection.experiment 会进入 main；被其他文件 import 时不会自动训练。
# argparse 把 --config、--output-dir 等终端参数转成 Path，交给上面的总控函数。
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="HUMAN与GROUP来源准入的有界探索审计。")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/run_01")
    parser.add_argument("--config", type=Path, default=ROOT / "settings.json")
    args = parser.parse_args(argv)
    try:
        run_experiment(args.data_dir, args.output_dir, args.config)
        print(f"中文报告：{args.output_dir.resolve() / 'reports/report_zh.md'}")
        return 0
    except (DataError, OSError, ValueError, KeyError) as exc:
        print(f"实验失败：{exc}", file=sys.stderr)
        return 1



if __name__ == "__main__":
    raise SystemExit(main())
