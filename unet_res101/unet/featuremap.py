import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from botnet_copy import BotNetEncoder

# class BotNetEncoder(nn.Module):
#     ...

def save_feature_maps_to_jpg(feature_maps, output_dir='feature_maps'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    feature_map = feature_maps[1]  

    feature_map = feature_map.cpu().squeeze(0)  


    for channel in range(feature_map.size(0)):

        fm = feature_map[channel]
        fm = fm - fm.min()  
        fm = fm / fm.max()  
        fm = (fm * 255).byte()  


        img = Image.fromarray(fm.numpy(), mode='L')
        img.save(os.path.join(output_dir, f'feature_map_channel_{channel}.jpg'))


model = BotNetEncoder()
model.eval()  

# 加载并预处理图像
input_image = Image.open('/Gaofendata/wavlet/img/29.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((512, 512)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(input_image).unsqueeze(0)  #


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = input_tensor.to(device)
model = model.to(device)

with torch.no_grad():  
    feature_maps = model(input_tensor)

# 保存特征图
save_feature_maps_to_jpg(feature_maps, output_dir='/Gaofendata/ouput')
