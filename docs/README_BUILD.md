# PyABSA Documentation Build Guide

## 概述
本文档说明如何编译PyABSA的文档。

## 编译方法

### 方法1: 使用Python脚本（推荐）
```bash
python build_docs.py
```

### 方法2: 使用批处理文件（Windows）
```bash
build_docs.bat
```

### 方法3: 手动编译
```bash
cd docs
python -m sphinx.cmd.build -b html . _build/html
```

## 查看文档
编译完成后，文档将保存在 `docs/_build/html/` 目录中。
主页面是 `index.html`。

## 依赖要求
确保已安装以下依赖：
- Python 3.10+
- Sphinx
- pandoc
- 其他在 `docs/requirements.txt` 中列出的包

## 安装依赖
```bash
cd docs
pip install -r requirements.txt
conda install -c conda-forge pandoc
```

## 故障排除

### 常见问题
1. **autodoc2兼容性问题**: 由于Windows兼容性问题，已改用传统的sphinx.ext.autodoc扩展
2. **缺少pandoc**: 使用 `conda install -c conda-forge pandoc` 安装
3. **编译警告**: 大部分警告是非致命的，不会阻止文档生成
4. **API文档**: 现在使用传统的autodoc扩展生成，功能完整

### 重新编译
如果需要重新编译，先删除 `_build` 目录：
```bash
cd docs
rm -rf _build
python -m sphinx.cmd.build -b html . _build/html
```

## 文档结构
- `0_intro/`: 介绍和安装
- `1_quick_start/`: 快速开始指南
- `2_config/`: 配置说明
- `3_inference/`: 推理示例
- `4_training/`: 训练指南
- `5_augmentation/`: 数据增强
- `6_tutorials/`: 教程和示例
- `7_datasets/`: 数据集说明
- `8_supported_tasks/`: 支持的任务
- `9_citation/`: 引用信息
