import numpy as np
import torch
import warnings
import os
import time
from dataProcess import get_dataloader, cal_val_iou, split_train_val
from tqdm import tqdm
import segmentation_models_pytorch as smp
import glob
#from unet import unet
#from unet_res101.unet.model_copy import Unet
from unet_res101.unet.model_no_bot import Unet #消融实验 没有bot只有se
#from unet_res101.unet.model_no_se import Unet #消融实验 没有se 只有bot
#from unet_res101.unet.model import Unet #消融实验没有bot 没有SE
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, LovaszLoss

import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchsummary import summary
from torchstat import stat

#from torch.cuda.amp import autocast, GradScaler

# 忽略警告信息
warnings.filterwarnings('ignore')
# cuDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
torch.backends.cudnn.enabled = True

# Tensor和Numpy都是矩阵,区别是前者可以在GPU上运行,后者只能在CPU上
# 但是Tensor和Numpy互相转化很方便
# 将模型加载到指定设备DEVICE上 
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 


def train(EPOCHES, BATCH_SIZE, train_image_paths, train_label_paths, 
          val_image_paths, val_label_paths, channels, optimizer_name,
          model_path, loss, early_stop):
    
    train_loader = get_dataloader(train_image_paths, train_label_paths, 
                                  "train", BATCH_SIZE, shuffle=True, num_workers=8)
    valid_loader = get_dataloader(val_image_paths, val_label_paths, 
                                  "val", BATCH_SIZE, shuffle=False, num_workers=8)
    
            
    model = Unet(
        #encoder_name="resnet50",
        encoder_weights="imagenet",
        # encoder_weights=None,
        in_channels=channels,
        classes=1,
        activation='sigmoid',
        # decoder_attention_type='scse',
    )

    #计算GLOPS和参数
    #model.load_state_dict(torch.load("user_data/model_data/03aug_unet_res101_SoftCE_dice.pth"))
    #stat(model,(3,512,512))
    model = model.to(DEVICE) #将模型"model"移动到指定的设备上计算
    # model = seg_hrnet_ocr.get_seg_model()
   # model.to(DEVICE)
    # model.load_state_dict(torch.load(model_path))

    if(optimizer_name == "sgd"):
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=1e-4, weight_decay=1e-3, momentum=0.9)
    # 采用AdamM优化器
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=1e-4, weight_decay=1e-3)
                                      
    # 余弦退火调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=2, # T_0就是初始restart的epoch数目
            T_mult=2, # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 * T_mult
            eta_min=1e-5 # 最低学习率
            )


    class WeightedLoss(_Loss):
        """Wrapper class around loss function that applies weighted with fixed factor.
        This class helps to balance multiple losses if they have different scales
        """

        def __init__(self, loss, weight=1.0):
            super().__init__()
            self.loss = loss
            self.weight = weight

        def forward(self, *input):
            return self.loss(*input) * self.weight

    class JointLoss(_Loss):
        """
        Wrap two loss functions into one. This class computes a weighted sum of two losses.
        """

        def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0):
            super().__init__()
            self.first = WeightedLoss(first, first_weight)
            self.second = WeightedLoss(second, second_weight)

        def forward(self, *input):
            return self.first(*input) + self.second(*input)

    if(loss == "SoftCE_dice"):
        # 损失函数采用SoftCrossEntropyLoss+DiceLoss
        # diceloss在一定程度上可以缓解类别不平衡,但是训练容易不稳定
        DiceLoss_fn=DiceLoss(mode='binary')
        # 软交叉熵,即使用了标签平滑的交叉熵,会增加泛化性
        # SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
        Bcelosss_fn = nn.BCELoss()
        loss_fn = JointLoss(first=DiceLoss_fn, second=Bcelosss_fn,
                                      first_weight=0.5, second_weight=0.5).cuda()
        # loss_fn = DiceLoss_fn
    elif(loss == "Bce_Lovasz"):
        # 损失函数采用SoftCrossEntropyLoss+LovaszLoss
        # LovaszLoss是对基于子模块损失凸Lovasz扩展的mIoU损失的直接优化
        LovaszLoss_fn = LovaszLoss(mode='binary')
        # 软交叉熵,即使用了标签平滑的交叉熵,会增加泛化性
        Bcelosss_fn = nn.BCELoss()
        SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
        loss_fn = JointLoss(first=LovaszLoss_fn, second=Bcelosss_fn,
                              first_weight=0.5, second_weight=0.5).cuda()
        # loss_fn = LovaszLoss_fn
    elif(loss == "ce"):
        loss_fn = nn.CrossEntropyloss()
    
    header = r'Epoch/EpochNum | TrainLoss | ValidmIoU | Acc     |  F1     | Time(m)'
    raw_line = r'{:5d}/{:8d}  | {:9.3f}   | {:9.3f}   | {:9.3f} | {:9.3f} | {:9.2f}'

    
#    # 在训练最开始之前实例化一个GradScaler对象,使用autocast才需要
#    scaler = GradScaler()

    # 记录当前验证集最优mIoU,以判定是否保存当前模型
    best_miou = 0
    best_miou_epoch = 0
    train_loss_epochs, val_mIoU_epochs, lr_epochs, val_Acc, val_F1= [], [], [], [], []
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    model_name = f"model_data_{timestamp}.pth"
    model_path = os.path.join("/media/cyy_1/高分数据/BotWaveNet/model_data/inria/", model_name)
    #model_path = os.path.join("/media/cyy_1/高分数据/BotWaveNet/model_data/inria")

    
    for epoch in range(1, EPOCHES+1):
        # print("Start training the {}st epoch...".format(epoch))
        # 存储训练集每个batch的loss
        losses = []
        start_time = time.time()
        model.train()
        model.to(DEVICE)
        tbar = tqdm(train_loader)
        for batch_index, (image, target) in enumerate(tbar):
            image, target = image.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(image)
            loss = loss_fn(output, target)
            optimizer.step()
            losses.append(loss.item())


        scheduler.step()
        # 计算验证集IoU
        val_iou, val_acc, val_f1 = cal_val_iou(model, valid_loader)
        # 输出验证集每类IoU
        # print('\t'.join(np.stack(val_iou).mean(0).round(3).astype(str)))
        # 保存当前epoch的train_loss.val_mIoU.lr_epochs
        train_loss_epochs.append(np.array(losses).mean())
        val_mIoU_epochs.append(np.mean(val_iou))
        lr_epochs.append(optimizer.param_groups[0]['lr'])
        val_Acc.append(np.mean(val_acc))
        val_F1.append(np.mean(val_f1))
        # 输出进程
        print(header)
        print(raw_line.format(epoch, EPOCHES, np.array(losses).mean(), 
                              np.mean(val_iou), np.mean(val_acc),np.mean(val_f1),
                              (time.time()-start_time)/60**1), end="")    
        if best_miou < np.stack(val_iou).mean(0).mean():
            best_miou = np.stack(val_iou).mean(0).mean()
            best_miou_epoch = epoch
            torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)
            print("  valid mIoU is improved. the model is saved.")
        else:
            print("")
            if (epoch - best_miou_epoch) >= early_stop:
                break


    return train_loss_epochs, val_mIoU_epochs, lr_epochs, val_Acc


# 不加主函数这句话的话,Dataloader多线程加载数据会报错
if __name__ == '__main__':
    EPOCHES = 300
    BATCH_SIZE = 12
    image_paths = glob.glob(r'/media/cyy_1/高分数据/Gaofendata/dataset/Inria_dataset/train/*.jpg')
    label_paths = glob.glob(r'/media/cyy_1/高分数据/Gaofendata/dataset/Inria_dataset/trainmask/*.jpg')
    # 每5个数据的第val_index个数据为验证集
    val_index = 0
    train_image_paths, train_label_paths, val_image_paths, val_label_paths = split_train_val(image_paths,
                                                                                             label_paths,
                                                                                             val_index,
                                                                                             )

    loss = "Bce_Lovasz"
    print("loss_type:", loss)
    channels = 3
    optimizer_name = "adamw"
    model_path = "/media/cyy_1/高分数据/BotWaveNet/model_data/inria"
    model_path += "_" + loss
    model_path += "_train_small"
    #model_path += ".pth"
    early_stop = 100
    train_loss_epochs, val_mIoU_epochs, lr_epochs, val_Acc = train(EPOCHES,
                                                          BATCH_SIZE,
                                                          train_image_paths,
                                                          train_label_paths,
                                                          val_image_paths,
                                                          val_label_paths,
                                                          channels,
                                                          optimizer_name,
                                                          model_path,
                                                          loss,
                                                          early_stop)

    if(True):
        import matplotlib.pyplot as plt
        epochs = range(1, len(train_loss_epochs) + 1)
        plt.plot(epochs, train_loss_epochs, 'r', label = 'train loss')
        plt.plot(epochs, val_mIoU_epochs, 'b', label = 'val mIoU')
        plt.title('train loss and val mIoU')
        plt.legend()
        plt.savefig("train loss and val mIoU.png",dpi = 300)
        plt.figure()
        plt.plot(epochs, lr_epochs, 'r', label = 'learning rate')
        plt.title('learning rate')
        plt.legend()
        plt.savefig("learning rate.png", dpi = 300)
        plt.show()
