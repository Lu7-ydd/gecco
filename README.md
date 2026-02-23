# GeCCo Module Tree

基于 `code.md` 的五阶段算法实现：
- 阶段1：基因特异性二值化 + `phi/Fisher/BH` + 有效基因过滤
- 阶段2：根节点与核心初始模块构建
- 阶段3：P/N/M 软分类 + R1-R4 插入与约束回滚
- 阶段4：后处理剪枝、单子节点压缩、深度压缩、弱拮抗合并
- 阶段5：模块树输出、细胞 major/subtype 赋值与可视化

## 运行

```bash
PYTHONPATH=src python scripts/run_pipeline.py --input datasets/adata_672.h5ad --output outputs/run
```

## 数据集验证

```bash
PYTHONPATH=src python scripts/validate_dataset.py
```

## 测试

```bash
PYTHONPATH=src pytest
```
