# XPolicyLab 复现补丁

本目录只保存本次复现所需的小型代码补丁。大型数据、权重、视频、完整 Git bundle 和归档清单保留在本地；Git 克隆不包含它们。

| 项目 | 精确版本 |
| --- | --- |
| 上游仓库 | `https://github.com/XPolicyLab/XPolicyLab.git` |
| 补丁基线 | `fa431ecd893ee706883e64fe5fe1464ec8cd928d` |
| 本地补丁提交 / 主仓库 gitlink | `6c1bb42f0302b01d7a1822ffbb00b011fd1905b1` |

补丁为 RoboTwin 派生数据增加显式 action alignment 标记，并支持派生图像无损 LZF 压缩；未标记旧数据保留上游读取行为。原始 HDF5 不修改。差异和回归用例见 `.patch` 文件。

`xpolicylab-act-alignment.bundle` 是约 2.4 KB 的增量 bundle，只包含本地补丁提交及其新增对象，要求先具有上述上游基线。它保留原始提交哈希；单独 `git apply` 或重新 `git am` 不保证得到 gitlink 引用的同一个提交。

## 新机器恢复代码

以下命令用于全新目录。已有 XPolicyLab 改动时先核对并保留，不直接覆盖。

```bash
git clone --no-recurse-submodules --branch test/robotwin-experiments \
  https://github.com/FZAbiding/RoboTwin.git
cd RoboTwin
sha256sum -c experiments/patches/SHA256SUMS

# 上游提供基线历史；本目录的增量 bundle 提供本地提交。
git clone --no-checkout https://github.com/XPolicyLab/XPolicyLab.git XPolicyLab
git -C XPolicyLab bundle verify ../experiments/patches/xpolicylab-act-alignment.bundle
git -C XPolicyLab fetch ../experiments/patches/xpolicylab-act-alignment.bundle \
  HEAD:refs/heads/local/robotwin-act-alignment
git submodule init XPolicyLab
git submodule update --init --no-fetch XPolicyLab

git -C XPolicyLab rev-parse HEAD
git submodule status
```

子模块 HEAD 应为 `6c1bb42f0302b01d7a1822ffbb00b011fd1905b1`。将 bundle 导入具名本地分支并使用 `--no-fetch`，可避免子模块更新时再次查询官方远端获取本地提交。保留官方 `origin`，不向官方推送本地提交、不更新到上游最新模型版本。

XPolicyLab 还含上游 Git LFS 图片资源。安装 Git LFS 后，正常联网检出会从官方远端取得它；增量 bundle 不包含该二进制资源。本机独立恢复测试使用已校验的原始 LFS 对象，通过后验证子模块工作区干净。完整 bundle 和 LFS 对象的离线备份仅在本地保存。

## 数据与权重

代码恢复不等于数据、模型或运行环境已经就绪。现有产物保留在原工作区 `RoboTwin-main`，位置及历史实验限制见 [实验记录](../README.md)。迁移到其他机器时须另行复制这些产物，并显式重定位配置中的绝对路径和符号链接，不重写原始记录。

本次仅验证补丁可恢复为精确提交；没有重新训练或重跑仿真，不追加新的策略成绩。
