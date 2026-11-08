"""读取冻结数据、检查来源隔离，并登记实验输入和产物摘要。

训练与验证可以先读；测试记录必须等模型和阈值冻结后再加载。
本模块只使用data目录中已准备好的文件，不下载或重新划分数据。
"""

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path


class DataError(ValueError):
    """用可读的错误说明输入或实验状态不符合约定。"""


VERSION = "source-group-v1"
SCENARIOS = ("test_id", "test_domain", "test_generator")
OUTPUT_COLUMNS = (
    "id", "source_id", "label", "model", "domain", "text", "text_sha256",
    "word_count", "adv_source_id", "decoding", "repetition_penalty", "attack", "title", "prompt",
)

# 读取文件，返回由内容计算出的 SHA-256 摘要。
# 内容只要改变一个字节，摘要通常就会不同；它用于核对文件是否仍是登记时那一份。
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # 每次只读 1 MB，避免为计算摘要把整个文件一次放进内存。
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# 查重时把连续空白统一为一个空格，但不改大小写，也不改训练时使用的正文。
def text_sha256(text: str) -> str:
    """Ignore whitespace differences for duplicate detection, not for training."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()



# 将配置或清单变成一个稳定的摘要。字典键先排序，避免只因键顺序不同而算出不同结果。
def fingerprint(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()



def write_json(path: Path, payload: object) -> None:
    """先创建父目录，再保存UTF-8 JSON；ensure_ascii=False保留中文。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    """一行字典对应一条CSV记录；newline处理正文中带引号的换行。"""
    if not rows:
        raise DataError(f"拒绝保存空结果：{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        # 以第一行字典的键作为列名；调用方需要保证每行使用相同的一组字段。
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)




# 核对配置里的划分规则，返回 split 部分；此函数不读取任何 CSV。
# “已见”是训练和验证允许使用的条件，“保留”是专门留作测试的条件。
def protocol(config: dict) -> dict:
    spec, data = config["split"], config["data"]
    if spec.get("protocol_version") != VERSION:
        raise DataError(f"划分协议必须为 {VERSION}。")
    domains, generators = spec["seen_domains"], spec["seen_generators"]
    # 两边不能交叉，也不能漏掉某个领域或生成器，否则“未见条件测试”的含义会改变。
    if (not domains or len(domains) != len(set(domains))
            or spec["held_out_domain"] in domains
            or set(domains + [spec["held_out_domain"]]) != set(data["domains"])):
        raise DataError("已见领域与保留领域必须不重叠，并覆盖取样领域。")
    if (not generators or len(generators) != len(set(generators))
            or spec["held_out_generator"] in generators
            or "human" in generators + [spec["held_out_generator"]]
            or set(generators + [spec["held_out_generator"]]) != set(data["models"])):
        raise DataError("已见生成模型与保留生成模型必须不重叠，并覆盖取样模型。")
    fractions = spec["fractions"]
    if (set(fractions) != {"train", "validation", "test"}
            or any(not isinstance(v, (float, int)) or not 0 < v < 1 for v in fractions.values())
            or not math.isclose(sum(fractions.values()), 1, abs_tol=1e-9)):
        raise DataError("train/validation/test 的比例必须均大于 0 且总和为 1。")
    near = spec["near_duplicate"]
    # 这里检查的是已登记的近重复规则，不在这里重新计算或调整近重复关系。
    if near.get("method", "inverted") not in {"inverted", "prefix"}:
        raise DataError("未知近重复检查方法。")
    if (not isinstance(near["shingle_words"], int) or not 1 <= near["shingle_words"] <= 10
            or not 0.5 <= near["jaccard_threshold"] <= 1
            or not isinstance(near["min_tokens"], int) or near["min_tokens"] < near["shingle_words"]
            or not isinstance(near["max_records"], int) or near["max_records"] < 1):
        raise DataError("近重复检查参数无效。")
    if not isinstance(config.get("seed"), int):
        raise DataError("seed 必须为整数。")
    return spec

# 读取已经取样、划分好的小 CSV，返回一行一个字典的列表；这里不是原始整份 RAID。
# CSV 读出的值原本都是字符串，只把后续需要计算的 label、word_count 转为整数。
def load_records(path: Path) -> list[dict]:
    # 正文可能很长，放宽单个字段的读取上限；这不是样本条数上限。
    csv.field_size_limit(16 * 1024 * 1024)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(OUTPUT_COLUMNS) - set(reader.fieldnames or []):
            raise DataError("数据 CSV 缺少数据准备阶段的必要字段。")
        rows = []
        for row in reader:
            if None in row or any(row.get(key) is None for key in OUTPUT_COLUMNS):
                raise DataError("数据 CSV 存在列数异常记录。")
            if row["label"] not in {"0", "1"}:
                raise DataError("label 只能为 0 或 1。")
            row["label"] = int(row["label"])
            row["word_count"] = int(row["word_count"])
            rows.append(row)
    return rows



# 返回训练记录、验证记录，以及后续冻结协议需要登记的摘要。
# 只打开 train/validation；字段齐全不代表都能当特征，来源、模型名等只用于核对。
def load_development_data(root: Path, config: dict) -> tuple[list[dict], list[dict], dict]:
    """Read only the frozen manifest and train/validation files, not tests."""
    split_spec = protocol(config)
    manifest_path = root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("integrity_status") != "passed" or manifest["protocol"] != split_spec
            or manifest["seed"] != config["seed"]):
        raise DataError("当前配置与已检查、冻结的划分不一致；请核对 data/split_manifest.json 与 data/dataset_config.json。")
    identity = {"protocol": split_spec, "seed": config["seed"], "sample_sha256": manifest["sample_sha256"]}
    # 将当前配置与冻结划分联系起来，防止拿另一套划分规则继续解释已有结果。
    if fingerprint(identity) != manifest["protocol_sha256"]:
        raise DataError("划分清单的协议摘要不一致。")
    loaded, hashes = {}, {}
    for name in ("train", "validation"):
        record = manifest["files"][name]
        path = root / f"{name}.csv"
        digest = file_sha256(path)
        # 先核对文件字节，再解析记录。即使只手动改了一个标签，也不能继续当作原数据。
        if digest != record["sha256"]:
            raise DataError(f"{name}.csv 与冻结清单不一致，停止训练。")
        rows = load_records(path)
        if len(rows) != record["rows"] or not rows or {r["label"] for r in rows} != {0, 1}:
            raise DataError(f"{name} 的样本数或两类标签覆盖异常。")
        if {r["model"] for r in rows} != {"human", *split_spec["seen_generators"]}:
            raise DataError(f"{name} 包含保留生成模型或缺少预期来源。")
        if {r["domain"] for r in rows} != set(split_spec["seen_domains"]):
            raise DataError(f"{name} 包含保留领域或缺少开发领域。")
        if len({r["id"] for r in rows}) != len(rows):
            raise DataError(f"{name} 存在重复 ID。")
        for row in rows:
            # source_assignments 说明每个来源属于哪个分区，不靠当前 CSV 的位置猜归属。
            source = manifest["source_assignments"].get(row["source_id"], {})
            if source.get("pool") != name or source.get("domain") != row["domain"]:
                raise DataError(f"{name} 的来源归属与冻结清单不一致。")
            if (row["label"] != (0 if row["model"] == "human" else 1)
                    or not row["text"].strip() or text_sha256(row["text"]) != row["text_sha256"]):
                raise DataError(f"{name} 的标签或正文摘要异常。")
        loaded[name], hashes[name] = rows, digest
    # 分别检查同一记录、同一来源、相同正文，三者任一重叠都可能泄漏信息。
    for key in ("id", "source_id", "text_sha256"):
        if {r[key] for r in loaded["train"]} & {r[key] for r in loaded["validation"]}:
            raise DataError(f"训练/验证存在 {key} 重叠。")
    components = [{manifest["source_assignments"][r["source_id"]]["component"] for r in loaded[name]}
                  for name in ("train", "validation")]
    # 不同 source_id 也可能通过近重复文本关联，所以还要检查已冻结的组件编号。
    if components[0] & components[1]:
        raise DataError("训练/验证存在近重复关联组重叠。")
    return loaded["train"], loaded["validation"], {
        "split_protocol_sha256": manifest["protocol_sha256"],
        "train_sha256": hashes["train"], "validation_sha256": hashes["validation"],
    }



# 将已经选好的模型、阈值和输入摘要保存成评估协议，返回带摘要的协议字典。
# 已存在时只接受完全相同的内容；想修改规则，需要另开输出目录保留两次实验。
def freeze_protocol(payload: dict, path: Path) -> dict:
    frozen = {**payload, "evaluation_protocol_sha256": fingerprint(payload)}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != frozen:
            raise DataError("已冻结的评估协议与当前模型、数据或阈值不同；保留本版，另存新探索实验目录。")
    else:
        write_json(path, frozen)
    return frozen

# 输入必须包含已保存的冻结协议；返回三个测试场景各自的记录列表。
# 这一步检查文件和分区边界，不能证明数据本身没有任何尚未发现的语义重复。
def load_test_sets(root: Path, config: dict, frozen: dict, development: dict, frozen_path: Path) -> dict:
    # 本函数先检查协议是否落盘，通过后才核对测试文件摘要、读取记录。
    if not frozen_path.exists() or json.loads(frozen_path.read_text(encoding="utf-8")) != frozen:
        raise DataError("必须先保存并核对评估协议，才能读取测试集。")
    payload = {k: v for k, v in frozen.items() if k != "evaluation_protocol_sha256"}
    if fingerprint(payload) != frozen["evaluation_protocol_sha256"]:
        raise DataError("评估协议摘要异常。")
    manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    if file_sha256(root / "split_manifest.json") != frozen["split_manifest_sha256"]:
        raise DataError("数据划分清单在冻结评估后发生变化。")
    spec = config["split"]
    dev_rows = development["train"] + development["validation"]
    dev_sets = {key: {r[key] for r in dev_rows} for key in ("id", "source_id", "text_sha256")}
    dev_components = {manifest["source_assignments"][r["source_id"]]["component"] for r in dev_rows}
    tests = {}
    for name in SCENARIOS:
        path = root / f"{name}.csv"
        record = frozen["test_files"][name]
        if file_sha256(path) != record["sha256"]:
            raise DataError(f"{name} 与冻结测试清单不一致。")
        rows = load_records(path)
        if len(rows) != record["rows"] or {r["label"] for r in rows} != {0, 1}:
            raise DataError(f"{name} 的样本数或标签覆盖异常。")
        expected_models = ({"human", spec["held_out_generator"]} if name == "test_generator"
                           else {"human", *spec["seen_generators"]})
        expected_domains = {spec["held_out_domain"]} if name == "test_domain" else set(spec["seen_domains"])
        # 新领域测试只换领域，新生成器测试只换生成器；两个条件不能在读入时混淆。
        if {r["model"] for r in rows} != expected_models or {r["domain"] for r in rows} != expected_domains:
            raise DataError(f"{name} 的领域或生成模型边界异常。")
        grouped = defaultdict(set)
        for row in rows:
            assignment = manifest["source_assignments"][row["source_id"]]
            if (row["label"] != (0 if row["model"] == "human" else 1)
                    or text_sha256(row["text"]) != row["text_sha256"]
                    or assignment["pool"] != ("heldout_domain" if name == "test_domain" else "test")
                    or assignment["component"] in dev_components):
                raise DataError(f"{name} 标签、正文摘要或来源归属异常。")
            grouped[row["source_id"]].add(row["model"])
        if any(models != expected_models for models in grouped.values()):
            raise DataError(f"{name} 没有完整的人类/指定生成模型配对。")
        for key in dev_sets:
            values = {r[key] for r in rows}
            if values & dev_sets[key]:
                raise DataError(f"{name} 与开发数据存在 {key} 重叠。")
        if len({r["id"] for r in rows}) != len(rows):
            raise DataError(f"{name} 中存在重复记录。")
        tests[name] = rows
    # 同条件测试和新生成器测试共用人类对照，这是原协议的设计。
    # 共享记录只允许是这批人类文本；不能因此允许它们进入训练或验证。
    id_humans = {r["id"] for r in tests["test_id"] if r["label"] == 0}
    gen_humans = {r["id"] for r in tests["test_generator"] if r["label"] == 0}
    shared_ids = {r["id"] for r in tests["test_id"]} & {r["id"] for r in tests["test_generator"]}
    if shared_ids != id_humans or shared_ids != gen_humans:
        raise DataError("两个测试场景的共享记录不是预期的人类对照。")
    domain_sources = {r["source_id"] for r in tests["test_domain"]}
    if domain_sources & {r["source_id"] for n in ("test_id", "test_generator") for r in tests[n]}:
        raise DataError("新领域与其他测试的来源发生异常重叠。")
    return tests



# 数据和参考结果都有各自的派生清单，只核验实际随包保留的文件。
# relatives指定子集时，训练前只检查开发数据，测试摘要留到冻结之后读取。
def check_registered_files(root: Path, manifest: dict, relatives=None) -> None:
    """检查清单格式、包内路径和文件内容，不冒充上游完整运行清单。"""
    root = root.resolve()
    records = manifest.get("files", {})
    if (manifest.get("format") not in {"aitextdetection-dataset-v1", "aitextdetection-reference-v1"}
            or manifest.get("status") != "verified_derivative" or not isinstance(records, dict) or not records):
        raise DataError("数据或参考结果缺少有效的派生文件清单。")
    for relative in records if relatives is None else relatives:
        if not isinstance(relative, str):
            raise DataError("文件清单中的路径必须是字符串。")
        path = root / relative
        if (relative not in records or Path(relative).is_absolute() or ".." in Path(relative).parts
                or path.is_symlink() or not path.resolve().is_relative_to(root)
                or not path.is_file() or file_sha256(path) != records[relative]):
            raise DataError(f"登记文件缺失、路径无效或摘要改变：{relative}。")


# 只比较真正控制计算的六项参数，项目名称和说明文字不作为数值实验参数。
def experiment_parameters(settings: dict) -> dict:
    return {name: settings[name] for name in (
        "minimum_words", "matched_seeds", "C_values", "target_validation_fpr",
        "bootstrap_resamples", "bootstrap_seed")}

# 词数规则固定为按空白切开文本；若缓存词数与正文不符，停止而不是悄悄改掉它。
def validate_word_counts(rows: list[dict]) -> None:
    for row in rows:
        if row["word_count"] != len(row["text"].split()):
            raise DataError(f"正文与 word_count 不一致：{row['id']}。")



# 返回本次运行的 Python、系统和依赖版本，供运行身份和复现报告登记。
# 这些信息帮助判断两次运行是否处在相同环境，不是检测文本时的输入。
def environment_record() -> dict:
    packages = (
        "cloudpickle", "contourpy", "cycler", "fonttools", "joblib", "kiwisolver", "matplotlib",
        "narwhals", "numpy", "packaging", "pillow", "pyparsing", "python-dateutil",
        "scikit-learn", "scipy", "six", "threadpoolctl",
    )
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
            "packages": {name: importlib.metadata.version(name) for name in packages}}

# 复用已有结果前，同时核对运行身份和所有登记产物；残缺目录不自动覆盖。
# 源码注释也属于文件内容，所以加完注释应使用新输出目录再运行。
def verify_completed_run(output: Path, identity: dict) -> dict:
    output = output.resolve()
    path = output / "run_manifest.json"
    if not path.exists():
        raise DataError("输出目录已非空且没有运行清单；保留它，请指定新的 --output-dir。")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("identity") != identity:
        raise DataError("已有目录未成功完成，或配置、代码、环境、原始数据已改变；保留它，请指定新的 --output-dir。")
    for relative, expected in manifest["outputs"].items():
        path = output / relative
        if Path(relative).is_absolute() or not path.resolve().is_relative_to(output):
            raise DataError("运行清单含无效输出路径。")
        if not path.is_file() or file_sha256(path) != expected:
            raise DataError(f"已有运行文件被改动或缺失：{relative}；不会覆盖，请检查或另存目录。")
    return manifest

# 按指定的若干列给每行组成一个键，返回列名和“键 -> 记录”的查找表。
# 比较时按键对齐，不要求两个 CSV 的行恰好排在同一位置。
def _csv_records(path: Path, keys: tuple[str, ...]) -> tuple[list[str], dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = {}
        for row in reader:
            key = tuple(row[name] for name in keys)
            if key in rows:
                raise DataError(f"比较文件出现重复键：{path.name} / {key}")
            rows[key] = row
    return fields, rows

# 先要求列和记录身份一致，再比较每个单元格，返回差异数量等核验结果。
# 字符串要相同，数值允许最多 1e-12 的绝对误差；这不等于比较模型文件的字节。
def compare_csv(reference: Path, actual: Path, keys: tuple[str, ...]) -> dict:
    ref_fields, ref = _csv_records(reference, keys)
    fields, current = _csv_records(actual, keys)
    if fields != ref_fields or set(current) != set(ref):
        return {"passed": False, "reason": "columns or record identities differ"}
    different, largest = 0, 0.0
    for key, row in current.items():
        for field in fields:
            if row[field] == ref[key][field]:
                continue
            try:
                left, right = float(ref[key][field]), float(row[field])
                close = math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
                largest = max(largest, abs(left - right))
            except ValueError:
                # 无法转成数字的不同字符串，直接记为不同，不把它们当成浮点误差。
                close = False
            if not close:
                different += 1
    return {"passed": different == 0, "rows": len(current), "different_cells": different,
            "max_numeric_difference": largest, "absolute_tolerance": 1e-12, "ignored_fields": []}



# 为计数审计整理来源组，返回“source_id -> 该来源记录列表”。
# 先核对词数，再检查组内领域和记录 ID；三条配对是否齐全由 sampling.source_groups 检查。
def group_for_counts(rows: list[dict]) -> dict[str, list[dict]]:
    validate_word_counts(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_id"]].append(row)
    for records in grouped.values():
        if len({r["domain"] for r in records}) != 1 or len({r["id"] for r in records}) != len(records):
            raise DataError("训练来源组跨领域或存在重复 ID。")
    return dict(grouped)

# 汇总一批记录的行数、类别数、来源数及领域组成，供报告解释训练数据的变化。
# 来源数和近重复组件数不是同一个数，一个组件可能连接多个来源。
def counts(rows: list[dict], assignments: dict, all_sources: set | None = None,
           long_sources: set | None = None) -> dict:
    grouped = group_for_counts(rows)
    sources = set(grouped)
    components = defaultdict(set)
    # 给出 all_sources 时，用它作为比较范围，检查该范围内有没有只保留一部分的组件。
    # 这只是记录筛选情况，并不把组件编号交给模型，也不会重新划分训练/测试。
    for source in sources if all_sources is None else all_sources:
        components[assignments[source]["component"]].add(source)
    return {"rows": len(rows), "source_groups": len(sources),
            "component_groups": len({assignments[s]["component"] for s in sources}),
            "human_rows": sum(r["label"] == 0 for r in rows), "ai_rows": sum(r["label"] == 1 for r in rows),
            "domain_source_counts": dict(sorted(Counter(records[0]["domain"] for records in grouped.values()).items())),
            "overlap_with_long_sources": len(sources & (long_sources or set())),
            "partially_retained_components": sum(bool(members & sources) and not members <= sources for members in components.values())}

# 检查新实验的输出位置；只允许放进本项目的 output/某个名称，不写入参考输入。
def check_output(project: Path, reference: Path, output: Path) -> None:
    """只把新结果写进output子目录，避免覆盖代码或冻结数据。"""
    project, reference, output = project.resolve(), reference.resolve(), output.resolve()
    allowed = project / "output"
    if output == allowed or not output.is_relative_to(allowed):
        raise DataError("请把新结果保存到当前项目的 output/新名称。")
    if output.is_relative_to(reference) or reference.is_relative_to(output):
        raise DataError("实验输出不能覆盖或包含正式参考目录。")
    parent = output.parent
    # 即使名称不同，也不能把新实验藏在另一个已登记实验的子目录里。
    while parent != project and parent.is_relative_to(project):
        if (parent / "run_manifest.json").exists():
            raise DataError("新运行不能嵌套在其他已登记实验目录内。")
        parent = parent.parent
