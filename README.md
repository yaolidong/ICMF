# ICMF

ICMF（Instance-Conditioned Modality Fusion）是用于多模态实体对齐的实例条件融合模型。

## 环境

需要 Python 3.9 或更高版本。PyTorch 与 CUDA 请根据运行设备单独安装；当前测试环境为 PyTorch 2.3.1 和 CUDA 11.8。

```bash
python -m pip install -r requirements.txt
```

## 数据

数据不包含在仓库中。默认目录结构如下：

```text
data/MMKG/FB15K_DB15K
data/MMKG/FB15K_YAGO15K
data/MMKG/DBP15K
data/MMKG/pkls
```

数据集目录需要包含 `ent_ids_1`、`ent_ids_2`、`ill_ent_ids`、`triples_1`、`triples_2`、`training_attrs_1` 和 `training_attrs_2`。

DBP15K 图像特征文件为 `pkls/fr_en_GA_id_img_feature_dict.pkl`、`pkls/ja_en_GA_id_img_feature_dict.pkl` 和 `pkls/zh_en_GA_id_img_feature_dict.pkl`。FBDB15K 与 FBYG15K 图像特征文件分别为 `pkls/FBDB15K_id_img_feature_dict.pkl` 和 `pkls/FBYG15K_id_img_feature_dict.pkl`。

## 运行

非迭代训练：

```bash
bash run_ICMF.sh 0 FBDB15K norm 0.2
```

迭代训练：

```bash
bash run_ICMF_il.sh 0 FBDB15K norm 0.2
```

DBP15K 示例：

```bash
bash run_ICMF.sh 0 DBP15K fr_en 0.3
```
