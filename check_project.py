"""项目检查入口：核对交付文件，检查数据、抽样、模型和统计计算。

python check_project.py --stage files 只使用标准库。
python check_project.py --stage all 读取开发数据并检查小型合成模型。
完整研究结果需要另外运行experiment和verification。
"""

import argparse
import ast
import contextlib
import importlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import traceback


# 通过脚本自己的位置找项目，运行时不需要写死某台电脑的用户名。
HERE = Path(__file__).resolve().parent
# 一个阶段只要求已经学到的模块存在，方便分几次把代码敲完。
MODULES = {
    "data": ("dataset",),
    "sampling": ("dataset", "sampling"),
    "classifiers": ("dataset", "classifiers"),
    "bootstrap": ("bootstrap",),
    "all": ("dataset", "sampling", "classifiers", "bootstrap", "reporting", "experiment", "verification"),
}



def check_delivery(project: Path) -> None:
    """流式核对随包文件；不导入研究模块，也不加载任何模型。"""
    manifest = json.loads((project / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "aitextdetection-package-v1":
        raise ValueError("不支持的交付清单格式。")
    total = 0
    for relative, entry in manifest["files"].items():
        path = project / relative
        # 清单只登记包内文件；拒绝绝对路径、上级目录和指向包外的链接。
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"清单路径越界：{relative}")
        if not path.resolve().is_relative_to(project) or path.is_symlink():
            raise ValueError(f"清单文件越界或是符号链接：{relative}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            # 每次读1MB，模型和数据文件再大也不用一次读进内存。
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"文件已变化或传输不完整：{relative}。补注释后应使用阶段检查，不修改清单。")
        total += size
    print(f"[通过] 交付文件完整：{len(manifest['files'])} 个登记文件，{total:,} 字节。")
    print("本检查说明文件与交付包一致；接下来安装依赖、分阶段检查，再亲自训练。")


def check_sources(project: Path, stage: str) -> None:
    """先确认文件存在并能解析；检查模块能独立导入，并提前发现语法错误。"""
    missing = [f"aitextdetection/{name}.py" for name in MODULES[stage]
               if not (project / "aitextdetection" / f"{name}.py").is_file()]
    if not (project / "settings.json").is_file():
        missing.append("settings.json")
    if missing:
        raise FileNotFoundError("所需模块不完整，请先完成：" + "、".join(missing))
    for name in MODULES[stage]:
        path = project / "aitextdetection" / f"{name}.py"
        # ast.parse只解析Python语法，不执行文件中的训练和数据读取。
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            # 同时识别import xxx和from xxx import yyy两种导入写法。
            imports = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                       else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            if any(n.split(".")[0] in {"src", "research"} for n in imports):
                raise AssertionError(f"{path.name}存在包外源码依赖；请检查项目是否完整。")


def load(name: str):
    # name例如sampling，最后导入当前项目的aitextdetection.sampling。
    return importlib.import_module(f"aitextdetection.{name}")


def check_data(reference: Path) -> tuple:
    data = load("dataset")
    config = json.loads((reference / "dataset_config.json").read_text(encoding="utf-8"))
    training, validation, _ = data.load_development_data(reference, config)
    # 这些数量来自随包冻结的正式数据；不是拿测试成绩当成功标准。
    assert (len(training), len(validation)) == (2634, 1572), "请使用包内冻结数据作为本研究输入。"
    assert all(type(r["label"]) is int and type(r["word_count"]) is int for r in training)
    split = json.loads((reference / "split_manifest.json").read_text())
    counts = data.counts(training, split["source_assignments"])
    assert counts["source_groups"] == 878 and counts["rows"] == 2634
    # 故意给一个不存在的冻结文件，检查程序会先拦住测试读取。
    # 临时目录退出with后自动清理，不会在自己的实验目录留下半成品。
    with tempfile.TemporaryDirectory() as directory:
        absent = Path(directory) / "not_frozen.json"
        try:
            data.load_test_sets(reference, config, {}, {"train": training, "validation": validation}, absent)
        except ValueError:
            pass
        else:
            raise AssertionError("读取测试前的冻结检查未生效。")
    print("[通过] 数据读取：训练2634条、验证1572条；字段转换与测试冻结检查正常。")
    return training, validation, config


def small_sources() -> list[dict]:
    # 四个人工来源覆盖四种情况：都达标、人类达标但AI短、人类短、两者都短。
    # sizes里的三个数依次对应人类、ChatGPT、GPT-4的词数。
    rows = []
    for i, sizes in enumerate(((240, 250, 260), (300, 150, 250), (100, 230, 240), (100, 150, 250))):
        for model, size in zip(("human", "chatgpt", "gpt4"), sizes, strict=True):
            rows.append({"id": f"s{i}-{model}", "source_id": f"s{i}", "domain": "news" if i % 2 == 0 else "reviews",
                         "model": model, "label": int(model != "human"), "word_count": size,
                         "text": " ".join([f"word{i}"] * size)})
    return rows


def check_selection(training: list[dict]) -> None:
    selected = load("sampling")
    toy = selected.build_training_regimes(small_sources(), 200, [42])
    assert {r["source_id"] for r in toy["human_200"]} == {"s0", "s1"}
    assert {r["source_id"] for r in toy["group_200"]} == {"s0"}
    # HUMAN只判断人类文本，s1的150词ChatGPT输出也应随整个来源保留。
    assert len(toy["human_200"]) == 6
    assert any(r["model"] == "chatgpt" and r["word_count"] == 150 for r in toy["human_200"])
    regimes = selected.build_training_regimes(training, 200, [42, 43, 44])
    counts = {n: len({r["source_id"] for r in rs}) for n, rs in regimes.items()}
    assert counts["all"] == 878 and counts["human_200"] == 695 and counts["group_200"] == 657
    assert len(regimes) == 12
    for seed in (42, 43, 44):
        target = regimes["group_200"]
        control = regimes[f"human_group_budget_{seed}"]
        from collections import Counter
        # 对照与GROUP的news/reviews数量相同，且所有记录确实来自HUMAN池。
        assert Counter(r["domain"] for r in control) == Counter(r["domain"] for r in target)
        assert {r["id"] for r in control} <= {r["id"] for r in regimes["human_200"]}
    # 把输入行倒过来再抽一次，来源选择仍应一致；CSV行顺序不是随机种子。
    again = selected.build_training_regimes(list(reversed(training)), 200, [42, 43, 44])
    assert {n: {r["id"] for r in rs} for n, rs in regimes.items()} == {n: {r["id"] for r in rs} for n, rs in again.items()}
    diagnostics, _ = selected.describe_selection(training, regimes, 200)
    assert diagnostics["human_only_sources"] == 38
    assert diagnostics["human_only_failure_counts"] == {"chatgpt": 38, "gpt4": 0}
    print("[通过] 准入规则：878／695／657来源，12方案；整组保留、领域配额与稳定抽样正常。")


def check_models(config: dict) -> None:
    models = load("classifiers")
    training, validation = [], []
    # 这里只造几个很容易区分的小句子，快速检查模型管线能否走通。
    # 这些句子的成绩不能当作真实检测效果。
    for i in range(12):
        label = i % 2
        text = ("personal memory uneven account " if label == 0 else "structured systematic formal summary ") * (30 + i)
        training.append({"id": str(i), "text": text + f"training{i}", "label": label})
    for i in range(4):
        label = i % 2
        text = ("personal memory uneven account " if label == 0 else "structured systematic formal summary ") * (32 + i)
        validation.append({"id": f"v{i}", "source_id": f"v{i}", "text": text + " validationonlytoken", "label": label})
    settings = {"target_validation_fpr": .05}
    # 把模型写到临时目录；收起中间训练输出，只保留本阶段的检查结果。
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
        records, pipelines = models.fit_variants({"all": training}, validation, config, settings, Path(directory))
    assert set(records) == {"length__all", "combined__all"}
    word = dict(pipelines["combined__all"].named_steps["tfidf"].transformer_list)["word"]
    # 验证文本特意加了训练中没有的词，检查拟合词表时是否偷读了验证集。
    assert "validationonlytoken" not in word.vocabulary_, "验证文本进入了训练词表。"
    assert models.calculate_metrics([0, 1], [.1, .9])["auroc"] == 1
    assert models.calculate_metrics([0, 1], [.9, .1])["auroc"] == 0
    # 人类分数有并列值时，目标FPR为0的阈值也必须排除全部人类。
    calibration = models.validation_fpr_threshold([.2, .2, .8], 0)
    assert calibration["validation_false_positives"] == 0
    print("[通过] 合成模型：两类模型可训练/保存加载；验证词不进词表，分数方向和阈值正常。")


def check_statistics() -> None:
    stats = load("bootstrap")
    # a、b各有两条关联文本；给A和B完全相同的分数，所有配对差应为0。
    labels, scores, components = [0, 1, 0, 1], [.1, .8, .3, .7], ["a", "a", "b", "b"]
    result = stats.paired_cluster_intervals(labels, scores, scores, components, seed=42, n_resamples=50)
    assert result["valid_resamples"] == 50
    for metric in ("auroc", "recall", "human_fpr"):
        assert result["point"][metric]["difference"] == 0
        assert result["intervals"][metric]["low"] == 0.0
        assert result["intervals"][metric]["high"] == 0.0
    print("[通过] 配对统计：相同预测的差值与区间为0，同来源组件共同抽样。")


def main() -> int:
    # 不加--stage时先检查交付文件，这一步只依赖Python标准库。
    parser = argparse.ArgumentParser(description="检查自己的项目代码模块。")
    parser.add_argument("--project", type=Path, default=HERE)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--stage", choices=("files", *MODULES), default="files")
    args = parser.parse_args()
    project = args.project.resolve()
    try:
        if args.stage == "files":
            check_delivery(project)
            return 0
        check_sources(project, args.stage)
        # 优先导入--project指定的代码，便于检查自己另存的代码目录。
        sys.path.insert(0, str(project))
        config = json.loads((project / "settings.json").read_text(encoding="utf-8"))
        development = None
        if args.stage in {"data", "sampling", "all"}:
            development = check_data(args.data_dir.resolve())
        if args.stage in {"sampling", "all"}:
            check_selection(development[0])
        if args.stage in {"classifiers", "all"}:
            # 模型管线需要正式配置中的特征设置，本研究只覆盖允许比较的C值。
            model_config = json.loads((args.data_dir / "dataset_config.json").read_text())
            model_config["training"]["C_values"] = config["C_values"]
            check_models(model_config)
        if args.stage in {"bootstrap", "all"}:
            check_statistics()
        if args.stage == "all":
            # 检查入口存在，不打开真实测试文件，也不启动完整实验。
            for name in ("reporting", "experiment", "verification"):
                module = load(name)
                assert callable(getattr(module, "write_report" if name == "reporting" else "main"))
            print("[通过] 报告、总控、核验入口完整；下一步才执行完整24模型实验。")
        print("项目阶段检查通过。完整实验需另行训练并核验。")
        return 0
    except (AssertionError, FileNotFoundError, ImportError, KeyError, TypeError, ValueError,
            SyntaxError, NameError, AttributeError) as exc:
        # 除了错误原因，再给出最后报错的文件和行号，方便回到编辑器定位。
        message = str(exc) or "某项检查不满足，请查看下面的检查位置。"
        print(f"暂未通过：{type(exc).__name__}: {message}", file=sys.stderr)
        location = traceback.extract_tb(exc.__traceback__)[-1]
        print(f"位置：{location.filename}:{location.lineno}；{location.line}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
