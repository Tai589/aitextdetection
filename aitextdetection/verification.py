"""检查一次实验的代码、数据、产物，并与固定参考表进行数值比较。"""

import argparse
import json
from pathlib import Path
import sys

from aitextdetection.experiment import experiment_identity, load_settings
from aitextdetection.dataset import DataError, check_registered_files, compare_csv, verify_completed_run


# 先按记录身份对齐，CSV行序不同不应被当成成绩不同。
KEYS = {
    "metrics.csv": ("variant", "evaluation_cohort", "scenario", "threshold_policy"),
    "predictions.csv": ("variant", "scenario", "id"),
    "contrasts.csv": ("comparison_id", "family", "evaluation_cohort", "scenario", "threshold_policy", "metric"),
}


def verify_run(run: Path) -> dict:
    """从运行清单找回相对路径，重新核对本次使用的输入、源码和环境。"""
    run = run.resolve()
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise DataError("只能核验已完成运行；失败或未冻结的目录不读取测试数据。")
    data_dir = (run / manifest["data_dir"]).resolve()
    project = (run / manifest["code_root"]).resolve()
    settings = load_settings(run / "settings.json")
    data_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    check_registered_files(data_dir, data_manifest)
    identity = experiment_identity(project, data_dir, settings, data_manifest)
    return verify_completed_run(run, identity)



def check_comparison_identity(identity: dict, reference: dict) -> None:
    """固定数值参数与实际派生数据清单，防止不同输入被当作同一实验。"""
    if (identity["parameters"] != reference["parameters"]
            or identity["source_dataset_manifest_sha256"] != reference["source_dataset_manifest_sha256"]
            or identity["dataset_manifest_sha256"] != reference["dataset_manifest_sha256"]):
        raise DataError("数据清单、来源或计算参数不同，不能按同一实验比较。")


def comparable_environment(environment: dict) -> dict:
    # 历史参考只登记了部分库。比较共同的五个计算库，完整的新环境仍单独保存。
    names = ("numpy", "scipy", "scikit-learn", "joblib", "matplotlib")
    return {"python": environment["python"], "system": environment["system"],
            "machine": environment["machine"],
            "packages": {name: environment["packages"][name] for name in names}}


def compare_runs(reference: Path, actual: Path) -> dict:
    # 既比较最终指标，也比较逐条预测和差值区间，避免只看四舍五入后的表格。
    return {name: compare_csv(reference / name, actual / "results" / name, keys)
            for name, keys in KEYS.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="核验实验完整性及三张参考结果表。")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = verify_run(args.run_dir)
        print(f"运行身份与产物校验通过：{len(manifest['outputs'])} 个文件。")
        if args.reference_dir is not None:
            reference = args.reference_dir.resolve()
            recorded = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
            check_registered_files(reference, recorded)
            identity = manifest["identity"]
            # 同时绑定数据来源与这套实际文件，不能只比较自述的来源名称。
            check_comparison_identity(identity, recorded)
            if comparable_environment(identity["environment"]) != comparable_environment(recorded["environment"]):
                print("环境记录存在差异；仍按绝对容差1e-12比较，不自动放宽标准。")
            else:
                print("可对照的环境字段一致。")
            results = compare_runs(reference, args.run_dir.resolve())
            for name, result in results.items():
                print(f"{name}：{'通过' if result['passed'] else '不一致'}；"
                      f"行数={result.get('rows', '?')}；差异单元格={result.get('different_cells', '?')}。")
            # 全部表通过才报告成功，参考目录中的旧模型不参与本次计算。
            if not all(result["passed"] for result in results.values()):
                raise DataError("数值比较未全部通过；保留结果，检查参数、实现和环境。")
            print("三张表的数值核验通过。")
        return 0
    except (DataError, OSError, ValueError, KeyError) as exc:
        print(f"核验未通过：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
