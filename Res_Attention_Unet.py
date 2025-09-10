import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    """残差块：包含两个卷积层和跳跃连接"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 跳跃连接：如果输入输出通道数不同或步长不为1，需要调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class AttentionModule(nn.Module):
    """注意力模块：通道注意力和空间注意力的结合"""
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class ResAttentionBlock(nn.Module):
    """残差注意力块：结合残差连接和注意力机制"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResAttentionBlock, self).__init__()
        self.residual_block = ResidualBlock(in_channels, out_channels, stride)
        self.attention = AttentionModule(out_channels)
    
    def forward(self, x):
        x = self.residual_block(x)
        x = self.attention(x)
        return x

class EncoderBlock(nn.Module):
    """编码器块：包含多个残差注意力块"""
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super(EncoderBlock, self).__init__()
        self.blocks = nn.ModuleList()
        
        # 第一个块可能需要改变通道数和尺寸
        self.blocks.append(ResAttentionBlock(in_channels, out_channels, stride=2))
        
        # 后续块保持通道数和尺寸不变
        for _ in range(1, num_blocks):
            self.blocks.append(ResAttentionBlock(out_channels, out_channels, stride=1))
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class DecoderBlock(nn.Module):
    """解码器块：上采样并融合特征"""
    def __init__(self, in_channels, out_channels, skip_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResAttentionBlock(out_channels, out_channels)
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        # 确保特征图尺寸匹配
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class ResAttentionUnet(nn.Module):
    """20方块网络：基于Unet的残差注意力网络"""
    def __init__(self, num_classes, in_channels=3, base_channels=64):
        super(ResAttentionUnet, self).__init__()
        
        # 添加属性以兼容训练代码
        self.n_channels = in_channels
        self.n_classes = num_classes
        
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels//2),
            nn.ReLU(inplace=True),
        )
        
        # 编码器路径 - 第一个编码器不进行下采样
        self.encoder1 = nn.Sequential(
            ResAttentionBlock(base_channels//2, base_channels, stride=1),
            ResAttentionBlock(base_channels, base_channels, stride=1),
        )
        self.encoder2 = EncoderBlock(base_channels, base_channels*2, num_blocks=2)    # 64 -> 128
        self.encoder3 = EncoderBlock(base_channels*2, base_channels*4, num_blocks=2)  # 128 -> 256
        self.encoder4 = EncoderBlock(base_channels*4, base_channels*8, num_blocks=2)  # 256 -> 512
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            ResAttentionBlock(base_channels*8, base_channels*16),
            ResAttentionBlock(base_channels*16, base_channels*8)
        )
        
        # 解码器路径
        self.decoder4 = DecoderBlock(base_channels*8, base_channels*4, base_channels*4)
        self.decoder3 = DecoderBlock(base_channels*4, base_channels*2, base_channels*2)
        self.decoder2 = DecoderBlock(base_channels*2, base_channels, base_channels)
        self.decoder1 = DecoderBlock(base_channels, base_channels//2, base_channels//2)       
        
        # 最终分类层（像素级预测）
        self.final_conv = nn.Conv2d(base_channels//2, num_classes, kernel_size=1)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 保存输入尺寸用于最终输出
        input_size = x.size()[2:]
        
        # 编码器路径
        x0 = self.initial_conv(x)      # 保持原始尺寸
        
        x1 = self.encoder1(x0)         # 保持原始尺寸
        x2 = self.encoder2(x1)         # 1/2
        x3 = self.encoder3(x2)         # 1/4
        x4 = self.encoder4(x3)         # 1/8
        
        # 瓶颈层
        x4 = self.bottleneck(x4)
        
        # 解码器路径（跳跃连接）
        x3 = self.decoder4(x4, x3)
        x2 = self.decoder3(x3, x2)
        x1 = self.decoder2(x2, x1)
        x0 = self.decoder1(x1, x0)
        
        # 最终分类（像素级预测）
        x = self.final_conv(x0)

        # 确保输出尺寸与输入匹配
        if x.size()[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x

def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__ == "__main__":
    # 创建模型实例
    model = ResAttentionUnet(num_classes=10, in_channels=3, base_channels=64)
    
    # 计算参数数量
    total_params, trainable_params = count_parameters(model)
    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 测试前向传播
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    try:
        output = model(input_tensor)
        print(f"输入形状: {input_tensor.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出类别数: {output.size(1)}")
        print("前向传播测试成功！")
        
        # 计算模型大小（MB）
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        print(f"模型大小: {model_size:.2f} MB")
        
    except Exception as e:
        print(f"前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()