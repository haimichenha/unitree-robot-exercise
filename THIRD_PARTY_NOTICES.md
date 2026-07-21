# 第三方开源代码声明

本仓库包含一个为教学和离线复现而保留的第三方代码快照：

| 组件 | 上游与版本 | 许可证 | 本仓库位置 |
| --- | --- | --- | --- |
| RoboJuDo | [HansZ8/RoboJuDo](https://github.com/HansZ8/RoboJuDo)，`release` 分支提交 `ed7601fec766070bcd8de0b302b305a5aa73b06c` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `third_party/RoboJuDo/` |

版权归上游作者 **HansZ8** 所有。RoboJuDo 随附的 `LICENSE` 原文保留在 `third_party/RoboJuDo/LICENSE`。按照 CC BY 4.0 的署名要求，本仓库明确标注了原始项目、作者、许可证链接、固定提交版本及改动范围。

## 快照范围与改动

- `third_party/RoboJuDo/` 为上述固定提交的工作树快照，保留其源码、文档、配置、模型、资源和 `LICENSE`；
- 没有修改 RoboJuDo 的源码；仅移除了上游 Git 元数据目录 `.git/`，并在本仓库父目录新增本声明、使用说明和自有实验代码；
- 上游 `.gitmodules` 已保留，但其可选子模块未初始化或复制。需要时请在该目录根据 `submodule_cfg.yaml` 运行 `python submodule_install.py`；
- 该快照共有 238 个文件、194,267,118 字节。以“相对路径 + SHA-256”排序后再 SHA-256 的聚合树摘要为：`982fb3ed0bf4cec63bb367fdffdfe42f32b444ab54e2c1eb454d0d3ac3cd0bc8`。

本仓库其余的 `代码/vm_source/` 是本次实训的 H1/ACT/MuJoCo/验证工程代码；二者分别存放，避免把 RoboJuDo 误表述为本项目原创代码。
