# aitextdetection

英语 AI 文本检测与来源筛选敏感性研究。

公开仓库：[Tai589/aitextdetection](https://github.com/Tai589/aitextdetection)。

本项目比较两类模型：只看文本长度的逻辑回归，以及结合词语、字符 TF-IDF 的逻辑回归。研究重点是：训练来源的准入规则改变后，模型在新来源、新领域和新生成器上的表现会怎样变化。

项目使用随包提供的 RAID 冻结数据，不需要下载原始大文件，也不需要 GPU 或模型 API。数据已经按来源及近重复组件划分。这里的测试场景来自 RAID 的公开 `train` 分区经过自定义划分，不是 RAID 官方测试榜单成绩。

## 文件安排

```text
aitextdetection/
├── README.md                    # 操作说明
├── CONFIGURATION.md             # 配置字段解释
├── VALIDATION.md                # 完整重训与数值核验记录
├── LICENSE                      # 本项目代码的 MIT 许可证
├── THIRD_PARTY_NOTICES.md        # RAID 来源、引用与第三方许可
├── LICENSES/
│   └── RAID-MIT.txt             # 保留的 RAID 上游许可全文
├── settings.json                # 本研究的参数
├── requirements-lock.txt        # 固定依赖版本
├── check_project.py             # 文件及分阶段检查
├── MANIFEST.json                # 交付文件摘要
├── aitextdetection/
│   ├── __init__.py              # Python 包声明
│   ├── dataset.py               # 数据读取、字段校验与实验冻结
│   ├── sampling.py              # 来源准入及等预算随机对照
│   ├── classifiers.py           # 特征、训练、参数选择与指标
│   ├── bootstrap.py             # 配对组件 bootstrap
│   ├── reporting.py             # 图表和中文报告
│   ├── experiment.py            # 实验入口
│   └── verification.py          # 运行完整性与数值对照
├── data/
│   ├── README.md                # 字段、划分与来源说明
│   ├── manifest.json           # 数据摘要和派生关系
│   ├── dataset_config.json     # 保留的数据制备及模型配置
│   ├── split_manifest.json     # 精简后的冻结划分清单
│   ├── train.csv
│   ├── validation.csv
│   ├── test_id.csv
│   ├── test_domain.csv
│   └── test_generator.csv
├── reference/
│   ├── manifest.json           # 参考表的来源和摘要
│   ├── metrics.csv             # 参考指标
│   ├── predictions.csv         # 参考逐条预测
│   └── contrasts.csv           # 参考配对差值
├── tests/
│   └── test_integrity.py        # 核心完整性测试
└── output/                     # 自己运行后生成，不随交付包提供
    └── run_01/
```

外层是项目目录，里面同名的 `aitextdetection/` 是 Python 包。运行命令时，应停留在能看到 `settings.json` 的外层目录。

本机安装、完整重训和数值比较的实测记录见 [VALIDATION.md](VALIDATION.md)。

本项目只保留实际运行、数据校验、结果解释及复现需要的文件。`reference/` 只用于数值对照，不提供预训练模型；自己的模型由实验入口重新训练。

## 在 Mac 上运行

建议使用 Python 3.15.0，与参考数值生成时的 Python 版本一致。在 GitHub 仓库页面点击 **Code → Download ZIP**，解压后将文件夹命名为 `aitextdetection`，再打开终端。下面假定文件夹在“下载”目录；如果位置不同，可输入 `cd ` 后把文件夹拖入终端，再按回车。

```bash
cd "$HOME/Downloads/aitextdetection"
python3.15 --version
python3.15 check_project.py --stage files
```

`files` 阶段只使用 Python 标准库，核对交付文件是否完整。它不训练模型，也不说明研究结论成立。

接下来创建本机环境并安装依赖。不要复制其他电脑的 `.venv`。

```bash
python3.15 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip check
```

以后每次新开终端，先进入项目目录，再执行 `source .venv/bin/activate`。终端前面出现 `(.venv)` 表示环境已启用。

依赖安装后，执行行为检查和测试：

```bash
python check_project.py --stage all
python -m unittest discover -s tests -v
```

检查通过后，开始完整实验：

```bash
python -m aitextdetection.experiment --config settings.json --output-dir output/run_01
```

训练过程中不要改代码、数据或配置。第一次运行需要拟合模型并计算抽样区间，耗时取决于电脑性能。程序会报告所处步骤；有输出且没有报错时，等待该命令结束。

最后核验自己的运行，并与随包参考表比较：

```bash
python -m aitextdetection.verification --run-dir output/run_01 --reference-dir reference
```

不同系统、硬件或依赖实现可能产生数值差异。核验使用固定容差逐单元格比较，不因环境不同自动放宽标准。参考环境记录保存在 `reference/manifest.json`；其中历史环境含有本项目并不需要的包，不必为了使环境文字完全相同而额外安装。

## 成功标准与产物

在默认配置、随包数据下，应得到：

| 项目 | 预期数量 |
|---|---:|
| 训练来源：ALL / HUMAN / GROUP | 878 / 695 / 657 |
| 训练方案 | 12 |
| 拟合模型 | 24 |
| `results/metrics.csv` | 144 行数据 |
| `results/predictions.csv` | 95,760 行数据 |
| `results/contrasts.csv` | 396 行数据 |

运行完成后，重点查看 `output/run_01/`：

- `run_manifest.json`：运行状态、代码和输入身份、产物摘要。状态应为 `completed`。
- `protocol.json` 与 `settings.json`：本次采用的规则和参数。
- `artifacts/`：本次训练得到的模型。
- `results/`：来源诊断、训练来源清单、指标、逐条预测、配对差值和汇总。
- `reports/report_zh.md`：中文研究报告及图表说明。

行数符合预期只是第一步；完整性检查和数值对照也需要通过。95,760 行预测来自多个模型对同一批测试记录重复预测，不是 95,760 条新收集文本。

不要覆盖已有结果。参数或注释改动后，改用新的输出目录：

```bash
python -m aitextdetection.experiment --config settings.json --output-dir output/run_02
```

重复使用已完成目录时，程序检查输入、代码、环境及产物是否一致；一致时可复用结果。失败目录保留用于排查，不能把它当作已完成实验。

## 怎样阅读代码

建议按下面顺序阅读，每次先弄清输入、输出，再看内部步骤：

1. `settings.json` 和 `CONFIGURATION.md`：实验准备比较什么。
2. `dataset.py`：一条记录怎样读取，来源和组件怎样用于检查。
3. `sampling.py`：哪些完整来源可以进入训练，随机对照怎样抽取。
4. `classifiers.py`：正文怎样变成特征，参数与阈值怎样选择。
5. `bootstrap.py`：怎样在相同抽样记录上比较两套预测。
6. `reporting.py`：已经算好的结果怎样汇总为图表与文字。
7. `experiment.py`：怎样将这些步骤按顺序连接起来。
8. `verification.py`：怎样发现代码、输入、模型或结果被改动。

也可以分阶段检查：

```bash
python check_project.py --stage data
python check_project.py --stage sampling
python check_project.py --stage classifiers
python check_project.py --stage bootstrap
python check_project.py --stage all
```

以上命令默认读取项目内的 `data/`。只有另存数据目录时，才需要为检查入口或实验入口添加 `--data-dir 数据目录`；`verification` 的 `--reference-dir` 指向数值参考表目录，两者用途不同。

添加自己的注释会改变文件摘要，因此交付包的 `files` 检查会报告源码与交付时不同。这是摘要检查的正常行为。保留交付清单，用行为检查测试修改后的实现，再选新的输出目录运行；不要修改清单来掩盖差异。

## 实验设计

每个训练来源保留人类、ChatGPT、GPT-4 三条完整文本：

- **ALL**：使用全部训练来源。
- **HUMAN**：只要求该来源的人类文本至少 200 词，仍保留这个来源的三条文本。
- **GROUP**：要求三条文本都至少 200 词。

Mistral 不参与训练准入判断，只出现在留出的生成器测试场景中。词数采用正文按空白分隔的统计口径，不是语言模型 tokenizer 的 token 数。

为了区分筛选规则与数据量变化，项目设置了领域配额、来源数相同的随机对照。这是**等预算对照**，没有保证各组长度分布、主题或文体相同。

所有方案使用相同完整验证集。TF-IDF 只在各方案的训练文本上拟合；`C` 按验证 AUROC 选择，相同分数时选较小的 `C`。分类阈值也只在验证集确定。保存冻结记录后，才读取三个完整测试场景；不再按测试长度筛选或根据测试成绩选冠军模型。

指标、阈值、来源组件以及实验身份都留有记录。数据来源、开发集与测试集的关系见 `data/README.md`。

## 结果可以支持什么

这是一项在**此前已观察过的测试数据**上的探索性敏感性分析，不构成未查看数据上的独立确认。

配对区间按来源关联的近重复组件进行 bootstrap，固定已训练模型、所选参数和阈值；它只描述当前测试组件的抽样波动，不包含重训和重新校准的全部不确定性。多个区间是逐项区间，不能当作已经进行了多重比较校正。

实验能检查训练来源筛选与检测表现的关联，不能单独识别因果机制，也不能据此宣称通用检测可靠或已经达到发表标准。模型分数不是经过验证的作者身份概率，仍需在实际目标场景中另外校准与验证。

## 许可与引用

本项目代码采用 [MIT 许可证](LICENSE)，允许使用、修改和分发，包括商业使用；使用时需保留版权与许可声明。

随包 RAID 数据保留上游版权与许可，来源、论文引用和许可全文见 [第三方说明](THIRD_PARTY_NOTICES.md) 与 [数据说明](data/README.md)。
