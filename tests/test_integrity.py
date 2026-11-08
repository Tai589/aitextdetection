"""检查会影响研究结论的边界，不使用真实测试成绩判断代码是否正确。"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from aitextdetection import dataset
from aitextdetection.bootstrap import paired_cluster_intervals
from aitextdetection.sampling import build_training_regimes


def source_fixture() -> list[dict]:
    """每个领域都有三种来源：全部达标、仅人类达标、人类未达标。"""
    rows = []
    for domain in ("news", "reviews"):
        for index, sizes in enumerate(((240, 260, 250), (300, 150, 250), (100, 240, 260))):
            source = f"{domain}-{index}"
            for model, size in zip(("human", "chatgpt", "gpt4"), sizes, strict=True):
                rows.append({
                    "id": f"{source}-{model}", "source_id": source, "domain": domain,
                    "model": model, "label": int(model != "human"),
                    "text": " ".join([source] * size), "word_count": size,
                })
    return rows


def registered_manifest(relative: str, body: bytes) -> dict:
    return {
        "format": "aitextdetection-dataset-v1", "status": "verified_derivative",
        "files": {relative: hashlib.sha256(body).hexdigest()},
    }


class DatasetIntegrityTests(unittest.TestCase):
    def test_changed_registered_bytes_are_rejected(self):
        # 先确认原文件能通过，再只改一个标签字节，检查清单能否发现变化。
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            body = b"id,label\nexample,0\n"
            path = root / "train.csv"
            path.write_bytes(body)
            manifest = registered_manifest("train.csv", body)
            dataset.check_registered_files(root, manifest)
            path.write_bytes(b"id,label\nexample,1\n")
            with self.assertRaises(dataset.DataError):
                dataset.check_registered_files(root, manifest)

    def test_manifest_cannot_authorize_an_outside_file(self):
        # 即使文件内容和摘要正确，清单也不能借 ../ 或绝对路径读取包外文件。
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root = parent / "data"
            root.mkdir()
            outside = parent / "outside.csv"
            body = b"outside data"
            outside.write_bytes(body)
            for relative in ("../outside.csv", str(outside)):
                with self.subTest(relative=relative):
                    manifest = registered_manifest(relative, body)
                    with self.assertRaises(dataset.DataError):
                        dataset.check_registered_files(root, manifest)

    def test_tests_are_blocked_before_any_file_hash_without_freeze(self):
        # 没有冻结文件时，连测试文件摘要也不应计算；空输入不应绕过这道检查。
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(dataset, "file_sha256", side_effect=AssertionError("提前读取了文件")) as hash_file:
                with self.assertRaises(dataset.DataError):
                    dataset.load_test_sets(root, {}, {}, {"train": [], "validation": []},
                                           root / "missing_frozen_protocol.json")
                hash_file.assert_not_called()

    def test_frozen_protocol_cannot_be_replaced_by_changed_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            payload = {"selected_model": "combined", "threshold": 0.5}
            first = dataset.freeze_protocol(payload, path)
            original_bytes = path.read_bytes()
            self.assertEqual(dataset.freeze_protocol(payload, path), first)
            with self.assertRaises(dataset.DataError):
                dataset.freeze_protocol({**payload, "threshold": 0.3}, path)
            self.assertEqual(path.read_bytes(), original_bytes)


class SamplingIntegrityTests(unittest.TestCase):
    def test_human_rule_keeps_short_ai_text_with_its_source(self):
        regimes = build_training_regimes(source_fixture(), 200, [42])
        human_rows = regimes["human_200"]
        self.assertEqual({row["source_id"] for row in human_rows},
                         {"news-0", "news-1", "reviews-0", "reviews-1"})
        self.assertEqual(Counter(row["source_id"] for row in human_rows),
                         Counter({"news-0": 3, "news-1": 3, "reviews-0": 3, "reviews-1": 3}))
        self.assertEqual(sum(row["model"] == "chatgpt" and row["word_count"] == 150
                             for row in human_rows), 2)
        self.assertEqual({row["source_id"] for row in regimes["group_200"]},
                         {"news-0", "reviews-0"})

    def test_random_controls_ignore_row_order_and_preserve_domain_budget(self):
        rows = source_fixture()
        forward = build_training_regimes(rows, 200, [42, 43, 44])
        backward = build_training_regimes(list(reversed(rows)), 200, [42, 43, 44])
        # 比较记录身份，不要求 CSV 的行顺序相同。
        self.assertEqual({name: {row["id"] for row in records} for name, records in forward.items()},
                         {name: {row["id"] for row in records} for name, records in backward.items()})
        for seed in (42, 43, 44):
            for candidate, target in ((f"random_human_{seed}", "human_200"),
                                      (f"random_group_{seed}", "group_200"),
                                      (f"human_group_budget_{seed}", "group_200")):
                with self.subTest(candidate=candidate):
                    self.assertEqual(Counter(row["domain"] for row in forward[candidate]),
                                     Counter(row["domain"] for row in forward[target]))
                    self.assertTrue(all(count == 3 for count in
                                        Counter(row["source_id"] for row in forward[candidate]).values()))
            self.assertLessEqual({row["id"] for row in forward[f"human_group_budget_{seed}"]},
                                 {row["id"] for row in forward["human_200"]})


class ExperimentBoundaryTests(unittest.TestCase):
    def test_failed_output_is_rejected_before_full_dataset_check(self):
        from aitextdetection import experiment

        # 失败目录可能尚未冻结模型，重试入口不能先去核对全部测试文件。
        # 这里只观察入口检查顺序；不读取真实数据，不启动任何模型训练。
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory).resolve()
            data_dir = project / "data"
            output = project / "output" / "failed"
            data_dir.mkdir()
            output.mkdir(parents=True)
            (data_dir / "manifest.json").write_text("{}", encoding="utf-8")
            (output / "run_manifest.json").write_text('{"status": "failed"}', encoding="utf-8")
            settings = project / "settings.json"
            shutil.copyfile(Path(__file__).resolve().parents[1] / "settings.json", settings)
            events = []

            def development_check(root, manifest, relatives=None):
                events.append(("registered_files", relatives))

            def reject_failed_run(*args):
                events.append(("failed_output", None))
                raise dataset.DataError("已有运行未完成")

            with patch.object(experiment, "experiment_identity", return_value={}), \
                    patch.object(experiment.data_io, "check_registered_files", side_effect=development_check), \
                    patch.object(experiment, "verify_completed_run", side_effect=reject_failed_run) as verify:
                with self.assertRaises(dataset.DataError):
                    experiment.run_experiment(data_dir, output, settings, project)
                verify.assert_called_once_with(output, {})

            self.assertEqual(events, [
                ("registered_files", ("dataset_config.json", "split_manifest.json", "train.csv", "validation.csv")),
                ("failed_output", None),
            ])

    def test_verification_does_not_read_dataset_for_failed_run(self):
        from aitextdetection import verification

        # 核验失败产物时不需要查看数据；状态应当先于任何数据文件检查。
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory).resolve()
            (run / "run_manifest.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
            with patch.object(verification, "check_registered_files") as registered:
                with self.assertRaises(dataset.DataError):
                    verification.verify_run(run)
                registered.assert_not_called()

    def test_comparison_rejects_changed_derived_dataset_despite_same_parent(self):
        from aitextdetection.verification import check_comparison_identity

        # 父来源相同不代表当前文件相同：派生时重新划分或删改记录也必须被发现。
        identity = {
            "parameters": {"minimum_words": 200, "matched_seeds": [42, 43, 44]},
            "source_dataset_manifest_sha256": "a" * 64,
            "dataset_manifest_sha256": "b" * 64,
        }
        check_comparison_identity(identity, dict(identity))
        changed = {**identity, "dataset_manifest_sha256": "c" * 64}
        with self.assertRaises(dataset.DataError):
            check_comparison_identity(identity, changed)


class PairedBootstrapTests(unittest.TestCase):
    def test_identical_predictions_have_zero_difference_and_interval(self):
        # 每个组件都含两类记录；模型完全相同，不该因分别抽样产生虚假的差异。
        labels = [0, 1, 0, 1, 0, 1]
        scores = [0.1, 0.8, 0.6, 0.7, 0.2, 0.4]
        components = ["a", "a", "b", "b", "c", "c"]
        result = paired_cluster_intervals(labels, scores, scores, components,
                                         seed=42, n_resamples=80)
        self.assertEqual(result["valid_resamples"], 80)
        for metric in ("auroc", "recall", "human_fpr"):
            self.assertEqual(result["point"][metric]["difference"], 0.0)
            self.assertEqual(result["intervals"][metric]["low"], 0.0)
            self.assertEqual(result["intervals"][metric]["high"], 0.0)

    def test_joint_row_permutation_does_not_change_paired_result(self):
        labels = [0, 1, 0, 1, 0, 1]
        scores_a = [0.1, 0.8, 0.6, 0.7, 0.2, 0.4]
        scores_b = [0.2, 0.4, 0.3, 0.9, 0.7, 0.8]
        components = ["a", "a", "b", "b", "c", "c"]
        original = paired_cluster_intervals(labels, scores_a, scores_b, components,
                                           seed=43, n_resamples=80)
        order = [5, 0, 3, 2, 1, 4]
        shuffled = paired_cluster_intervals(
            *[[values[index] for index in order] for values in
              (labels, scores_a, scores_b, components)], seed=43, n_resamples=80)
        self.assertEqual(shuffled, original)


if __name__ == "__main__":
    unittest.main()
