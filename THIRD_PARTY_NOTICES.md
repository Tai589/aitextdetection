# 第三方来源与许可

## 本项目代码

本项目代码和自行编写的说明文档采用根目录的 [MIT 许可证](LICENSE)，版权声明为 `Copyright (c) 2026 Tai589`。第三方材料的版权和许可单独保留。

## RAID 数据集

`data/` 中五份 CSV 的正文和上游元数据来自 RAID 公开 `train` 分区，不属于本项目原创数据。字段、筛选条件和自定义划分见 [data/README.md](data/README.md)。

- 数据卡：[liamdugan/raid](https://huggingface.co/datasets/liamdugan/raid)。
- 上游项目：[liamdugan/raid](https://github.com/liamdugan/raid)。
- 上游许可证：[LICENSE](https://github.com/liamdugan/raid/blob/main/LICENSE)。

截至 2026-11-08，上游数据卡标注的许可证为 MIT。随包保留的上游许可全文见 [LICENSES/RAID-MIT.txt](LICENSES/RAID-MIT.txt)，版权声明为 `Copyright (c) 2024- Liam Dugan`。本项目的许可证不替代上游许可，也不将原始文本及元数据的版权归给本项目作者。

使用此数据集开展研究时，请引用：

Liam Dugan, Alyssa Hwang, Filip Trhlík, Andrew Zhu, Josh Magnus Ludan, Hainiu Xu, Daphne Ippolito, and Chris Callison-Burch. 2024. [RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors](https://aclanthology.org/2024.acl-long.674/). *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 12463–12492.

`reference/` 中的表格是本项目在自定义划分上计算的参考结果，不是 RAID 官方测试榜单结果。分发本仓库的数据副本时，应一并保留本说明及上游许可证。

## Python 依赖

`requirements-lock.txt` 记录安装所需的第三方包。本仓库不包含这些包的源码；各依赖继续适用其各自的许可证。
