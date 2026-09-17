ABLATION_RANDOM_REFERENCE=False  # 消融1：关闭Top-K，每张图随机选择参考图
ABLATION_UNIFORM_WEIGHT=False  # 消融2：跳过注意力打分，所有参考图等权平均
ABLATION_CNN_FUSION=False  # 消融3：原图3通道与参考图9通道拼接，经CNN提取信息替代交叉注意力传递

from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Optional
import torch

@dataclass
class Config:
    ablation_random_reference:bool=ABLATION_RANDOM_REFERENCE
    ablation_uniform_weight:bool=ABLATION_UNIFORM_WEIGHT
    ablation_cnn_fusion:bool=ABLATION_CNN_FUSION
    data_dir:str="data/fivek"  # FiveK 数据集目录
    output_dir:str="outputs"  # 训练输出根目录
    image_size:int=256  # 输入图像边长（像素）
    batch_size:int=4  # 每个训练批次的样本数
    num_epochs:int=1000  # 总训练轮数
    lr:float=2e-4  # AdamW 初始学习率
    weight_decay:float=1e-5  # AdamW 权重衰减

    encoder_base_dim:int=64  # 编码器基础通道数
    encoder_mid_dim:int=64  # 编码器中间通道数
    attention_dim:int=128  # 注意力特征维度
    edit_dim:int=64  # 编辑特征维度
    cross_attention_heads:int=2  # 交叉注意力头数
    attention_spatial_size:int=64  # 注意力特征图空间尺寸
    enhancer_base_dim:int=64  # 增强器基础通道数
    enhancer_mid_dim:int=96  # 增强器中间通道数

    num_reference:int=3  # 每个样本使用的参考图数量
    temperature:float=0.01  # 初始参考选择温度
    min_temperature:float=0.01  # 温度衰减下限
    group_update_interval:int=1  # 每隔多少个 epoch 对全部训练组对重新检索参考（第 1 个 epoch 也会检索）
    lambda_ssim:float=0  # SSIM 损失项权重
    residual_weight:float=0.001  # 残差正则项权重
    num_workers:int=2  # 数据加载工作进程数
    device:str="cuda" if torch.cuda.is_available() else "cpu"  # 训练设备
    seed:int=1337  # 随机种子
    train_ratio:float=0.8  # 训练集比例
    val_ratio:float=0.1  # 验证集比例
    download_url:str=""  # 留空使用公共 manifest 自动下载；不支持自定义压缩包地址
    max_visualizations:int=100  # 单次最多保存的预览图数
    visualize_val_each_epoch:bool=True  # 是否每个 epoch 保存验证集预览图
    # 原图按 discover_records 的稳定排序编号（从 1 开始、包含端点）；每张原图的全部专家组对都会被纳入。None 时沿用随机比例划分。
    train_start:Optional[int]=1
    train_end:Optional[int]=400
    val_start:Optional[int]=401
    val_end:Optional[int]=500
    def paths(self):
        root=Path(self.output_dir);return {"root":root,"checkpoints":root/"checkpoints","similarity":root/"similarity","logs":root/"logs","visuals":root/"visuals"}
    def to_dict(self):return asdict(self)
