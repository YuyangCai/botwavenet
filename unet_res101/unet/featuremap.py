import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from botnet_copy import BotNetEncoder
# 确保BotNetEncoder类已经定义
# class BotNetEncoder(nn.Module):
#     ...

def save_feature_maps_to_jpg(feature_maps, output_dir='feature_maps'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 假设我们只对第一个ReLU之后的特征图感兴趣
    feature_map = feature_maps[1]  # feature_maps[1]应该是ReLU之后的特征图

    feature_map = feature_map.cpu().squeeze(0)  # 移动到CPU并移除批次维度

    # 归一化特征图并保存
    for channel in range(feature_map.size(0)):
        # 提取通道
        fm = feature_map[channel]
        fm = fm - fm.min()  # 最小值归零
        fm = fm / fm.max()  # 归一化
        fm = (fm * 255).byte()  # 转换为0-255

        # 转换为PIL图像并保存
        img = Image.fromarray(fm.numpy(), mode='L')
        img.save(os.path.join(output_dir, f'feature_map_channel_{channel}.jpg'))

# 加载模型
model = BotNetEncoder()
model.eval()  # 设置为评估模式

# 加载并预处理图像
input_image = Image.open('/media/cyy_1/高分数据/Gaofendata/wavlet/img/29.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整图像大小以匹配模型的输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(input_image).unsqueeze(0)  # 添加批次维度

# 使用GPU进行计算（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = input_tensor.to(device)
model = model.to(device)

# 获取ReLU层后的特征图
with torch.no_grad():  # 在不计算梯度的情况下执行前向传播
    feature_maps = model(input_tensor)

# 保存特征图
save_feature_maps_to_jpg(feature_maps, output_dir='/media/cyy_1/高分数据/Gaofendata/ouput')
