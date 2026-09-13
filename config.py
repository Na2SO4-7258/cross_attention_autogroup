from dataclasses import dataclass,asdict
from pathlib import Path
import torch

@dataclass
class Config:
    data_dir:str="data/fivek"  # FiveK 数据集所在目录。
    output_dir:str="outputs"  # checkpoint、日志和可视化结果的输出目录。
    image_size:int=256  # 训练/评估时统一缩放的输入图像边长（像素）。
    batch_size:int=4  # 每次优化使用的样本数。
    num_epochs:int=30  # 本次启动最多训练的轮数。
    lr:float=2e-4  # AdamW 学习率。
    weight_decay:float=1e-5  # AdamW 权重衰减系数。
    image_encoder_base_dim:int=32  # 图像/编辑编码器第一层卷积通道数。
    image_encoder_mid_dim:int=64  # 图像/编辑编码器第二层卷积通道数。
    feature_dim:int=128  # 图像编码器输出、交叉注意力输入的通道维度。
    attention_dim:int=128  # 交叉注意力 Q/K/V 和上下文的总维度。
    cross_attention_heads:int=1  # 交叉注意力头数；默认单头以保持原网络不变。
    attention_spatial_size:int=64  # 交叉注意力前将图像/编辑特征压缩到的边长，即 64×64。
    edit_dim:int=64  # 编辑编码器输出和增强网络编辑特征的通道维度。
    enhancer_base_dim:int=64  # 增强 U-Net 输入层和上采样层通道数。
    enhancer_mid_dim:int=96  # 增强 U-Net 下采样层和中间层通道数。
    num_reference:int=3  # 每张待处理图像参与选择的参考图数量。
    temperature:float=0.35  # 动态参考选择 softmax 的初始温度。
    min_temperature:float=0.08  # 训练退火后的最小选择温度。
    group_update_interval:int=5  # 每训练多少个 epoch 重新计算一次参考分组。
    lambda_ssim:float=0.2  # SSIM 损失项的权重。
    residual_weight:float=0.001  # 输出相对原图残差的正则化权重。
    num_workers:int=2  # DataLoader 并行加载进程数。
    device:str="cuda" if torch.cuda.is_available() else "cpu"  # 训练使用的 PyTorch 设备。
    seed:int=1337  # 划分数据和随机参考选择的随机种子。
    train_ratio:float=0.8  # 训练集占全部样本的比例。
    val_ratio:float=0.1  # 验证集占全部样本的比例；其余为测试集。
    download_url:str=""  # 可选的自定义数据下载地址；留空使用内置下载器。
    max_visualizations:int=12  # 评估时最多保存的对比图数量。
    reference_pool_size:int=512  # 动态分组时纳入相似度计算的训练样本池大小。
    def paths(self):
        root=Path(self.output_dir)
        return {"root":root,"checkpoints":root/"checkpoints","similarity":root/"similarity","logs":root/"logs","visuals":root/"visuals"}
    def to_dict(self): return asdict(self)
