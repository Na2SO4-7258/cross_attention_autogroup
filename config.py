from dataclasses import dataclass,asdict
from pathlib import Path
import torch

@dataclass
class Config:
    data_dir:str="data/fivek"  # FiveK 数据集目录
    output_dir:str="outputs"  # 训练输出根目录
    image_size:int=256  # 输入图像边长（像素）
    batch_size:int=4  # 每个训练批次的样本数
    num_epochs:int=1000  # 总训练轮数
    lr:float=2e-4  # AdamW 初始学习率
    weight_decay:float=1e-5  # AdamW 权重衰减
    encoder_base_dim:int=32  # 编码器基础通道数
    encoder_mid_dim:int=64  # 编码器中间通道数
    attention_dim:int=128  # 注意力特征维度
    edit_dim:int=64  # 编辑特征维度
    cross_attention_heads:int=1  # 交叉注意力头数
    attention_spatial_size:int=64  # 注意力特征图空间尺寸
    enhancer_base_dim:int=64  # 增强器基础通道数
    enhancer_mid_dim:int=96  # 增强器中间通道数
    num_reference:int=3  # 每个样本使用的参考图数量
    temperature:float=0.35  # 初始参考选择温度
    min_temperature:float=0.08  # 温度衰减下限
    group_update_interval:int=5  # 每隔多少个 epoch 更新一次分组（第 1 个 epoch 也会更新）
    lambda_ssim:float=0.2  # SSIM 损失项权重
    residual_weight:float=0.001  # 残差正则项权重
    num_workers:int=2  # 数据加载工作进程数
    device:str="cuda" if torch.cuda.is_available() else "cpu"  # 训练设备
    seed:int=1337  # 随机种子
    train_ratio:float=0.8  # 训练集比例
    val_ratio:float=0.1  # 验证集比例
    download_url:str=""  # 数据集下载地址（为空时不下载）
    max_visualizations:int=12  # 单次最多保存的预览图数
    visualize_val_each_epoch:bool=True  # 是否每个 epoch 保存验证集预览图
    reference_pool_size:int=128  # 候选参考图池大小
    def paths(self):
        root=Path(self.output_dir);return {"root":root,"checkpoints":root/"checkpoints","similarity":root/"similarity","logs":root/"logs","visuals":root/"visuals"}
    def to_dict(self):return asdict(self)
