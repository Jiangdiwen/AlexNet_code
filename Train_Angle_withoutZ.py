'''
Author: Yanxin_Jiang
Date: 2023-07-26 14:22:05
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-09-18 09:01:30
Description: 

Copyright (c) 2023 by 用户/公司名, All Rights Reserved. 
'''
from os.path import join
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import alexnet
from torchvision.models.alexnet import AlexNet_Weights
from torch.utils.data import Dataset, DataLoader
import numpy as np
from multiprocessing import freeze_support
    # 定义数据预处理转换
from torchvision import transforms
from tqdm import tqdm
import logging
import time
import matplotlib.pyplot as plt
import matplotlib
import os
from datetime import datetime
# 划分数据集为训练集和验证集
from sklearn.model_selection import train_test_split
from Dataloader.dataloader_Angle_PreZ import load_npydata
from Dataloader.dataloader_npy import CustomDataset
from Model.CameraRegressor import AlexNetFeatures, CameraParamRegressor_3withZ_axis, CameraParamRegressor_1, CombinedModel, CameraParamRegressor_3


def to_float32(img):
    return img.to(torch.float32)

def normalize_img(img_data):
    num_data = img_data.shape[0]
    normalized_img = np.zeros_like(img_data, dtype=np.float32)
    for i in range(num_data):
        mean = np.mean(img_data[i])
        std = np.std(img_data[i])
        normalized_img[i] = (img_data[i] - mean) / std 
    return normalized_img


def normalize_0_1_label(labels, 
                         para_ranges = np.array([[-30, 30], [-30, 30], [-90, 90],
                                                 [-20, 20],[-20, 20], [-100, 50],
                                                 [0, 2]])):
    # 对labels的每一列参数进行归一化
    normalized_labels = np.zeros_like(labels, dtype=np.float32)
    #normalized_labels = np.zeros_like(labels)
    for i in range(len(para_ranges)):
        normalized_labels[:, i] = (labels[:, i] - para_ranges[i, 0]) / (para_ranges[i, 1] - para_ranges[i, 0])

    # 转换元素数据类型为float32
    
    return normalized_labels  

def denormalize_original_label(normalized_labels, 
                                 para_ranges = np.array([[-30, 30], [-30, 30], [-90, 90],
                                                         [-20, 20],[-20, 20], [-150, 90],
                                                         [0, 2]])):    
    if normalized_labels.is_cuda:
        normalized_labels = normalized_labels.cpu()
    normalized_labels = normalized_labels.numpy()
    # 对normalized_labels的每一列参数进行还原
    original_labels = np.zeros_like(normalized_labels)
    for i in range(len(para_ranges)):
        original_labels[:, i] = normalized_labels[:, i] * (para_ranges[i, 1] - para_ranges[i, 0]) + para_ranges[i, 0]
    
    return torch.tensor(original_labels)



def Set_Task_name(num_Task, num_epochs, num_train_data, batch_size, lr_features, lr_regressor):
    # 获取当前日期时间
    current_datetime = datetime.now()   
    today_month_day = current_datetime.strftime("%m-%d")
    Task_name = f'{today_month_day}_Task{num_Task}lr_features{lr_features}_lr_regressor{lr_regressor}_{batch_size}batch_{num_epochs}epochs_{num_train_data}trainDatas'
    
    return str(Task_name)


def add_timestamp_decorator(func):
    def wrapper(*args, **kwargs):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        print(f"{current_time}: ", end="")
        return func(*args, **kwargs)
    return wrapper

# 示例打印函数
@add_timestamp_decorator
def print_message(message):
    print(message)

logging.info = add_timestamp_decorator(logging.info)

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


    freeze_support()
    plt.switch_backend('agg') 
    
    #--------------------------------加载预训练的预测Z方向网络-------------------------------
    Z_feature_extraction = AlexNetFeatures()
    Z_regressor_model = CameraParamRegressor_1(input_size=256 * 6 * 6).float()
    base_path = "/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Projects/09-16_Task_OLz_07003_divide_modellr_features1e-05_lr_regressor1e-05_64batch_800epochs_5049trainDatas"
    Z_Pretr_feature_model_name = "best_vali_model_features.pth"
    Z_Pretr_feature_model_path = join(base_path, Z_Pretr_feature_model_name)
    Z_Pretr_regressor_model_name = "best_vali_model_regressor.pth"
    Z_Pretr_regressor_model_path = join(base_path, Z_Pretr_regressor_model_name)
    Z_feature_extraction.load_state_dict(torch.load(Z_Pretr_feature_model_path)) 
    Z_regressor_model.load_state_dict(torch.load(Z_Pretr_regressor_model_path)) 
    #--------------------------------------------------------------------------------------

    #Proj_data_path = '/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Regression_Model/data_0726t1_inkGood.npy'
    Proj_data_path = '/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Regression_Model/data/Train_datas/0916_07003_Zrangelarge_5940datas.npy'
    print(f'use proj_data : {Proj_data_path}')
    data, labels = load_npydata(Proj_data_path)  # 二维图像矩阵的列表，每个元素是一个二维矩阵
    # 在切换模型为验证模式之前定义参数名列表，用于后续的打印
    #param_names = ['Dz']
    param_names = ['Rx', 'Ry', 'Rz']
    # labels = [...]  # 包含六个参数的一维向量的列表
    #para_ranges = [-120, 90]
    para_ranges = np.array([[-10, 10], [-20, 20], [-90, 90]])
    # --------------------------------------------------------------------------------------------------------------------------------
    normal_data = normalize_img(data)
    normal_labels = normalize_0_1_label(labels, para_ranges)
    data_list_rgb = [np.stack((img, img, img), axis=-1) for img in normal_data] #已经是一个包含RGB图像的列表，且图像维度为(height, width, channels)，
    
    # 划分数据集为训练集和验证集
    train_data, val_data, train_labels, val_labels = train_test_split(data_list_rgb, normal_labels, test_size=0.15, random_state=42)
    data_transform = transforms.Compose([
        transforms.ToTensor(),          # 将图像转换为张量
        transforms.Resize((256, 256), antialias=True),   # 调整大小为AlexNet输入尺寸
        # 可以添加其他预处理转换
        transforms.Lambda(to_float32),  
        transforms.Normalize(mean=[0.], std=[1.])
    ])
    # 创建自定义数据集和数据加载器
    #custom_dataset = CustomDataset(data=data_list_rgb, labels=normal_labels, transform=data_transform)
    train_dataset = CustomDataset(data=train_data, labels=train_labels, transform=data_transform)
    val_dataset = CustomDataset(data=val_data, labels=val_labels, transform=data_transform)


    # 创建数据加载器
    batch_size = 64
    #data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 假设特征图的大小为feat_size，相机参数的数量为6； 在这里，特征图的大小（feat_size）指的是AlexNet模型最终生成的特征图的大小。在AlexNet中，全连接层之前的最后一个卷积层的输出就是特征图，也称为激活图（activation map）。这个特征图的大小决定了回归模型中全连接层的输入大小。输入数据的图像大小在训练之前已经确定，并且不会影响AlexNet模型生成的特征图的大小。AlexNet模型的输入大小是固定的，通常为224x224（HxW），因此对于不同大小的输入图像，AlexNet会自动将其缩放为224x224的大小。
    #回到特征图的大小（feat_size），它取决于特定的AlexNet变种或配置。在标准的AlexNet模型中，最终的特征图大小为7x7x256，其中7x7是特征图的空间大小，256是特征图的通道数。在实际使用时，你可以根据具体的AlexNet变种或配置来确定特征图的大小，并将这个值作为回归模型的输入大小。
    feat_size = (256, 6, 6) 

    
    # 创建回归模型实例
    num_params = len(para_ranges)
    feature_extraction = AlexNetFeatures()
    regressor_model = CameraParamRegressor_3(input_size=feat_size[0] * feat_size[1] * feat_size[2]).float()
    # ----------------------Task_parameters---------------------------------
    #num_epochs = 1500 
    num_epochs = 1500
    num_Task = '_Angle_PreZ_07003'
    num_train_data = len(train_data)        
    lr_regressor = 1e-05
    lr_features = 1e-05
    # lr_regressor = 1e-05
    # lr_features = 1e-03
    Task_name = Set_Task_name(num_Task, num_epochs, num_train_data, batch_size, lr_features, lr_regressor)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()    
    # 定义优化器
    features_optimizer = optim.Adam(feature_extraction.parameters(), lr=lr_features)
    regressor_optimizer = optim.Adam(regressor_model.parameters(), lr=lr_regressor)    
    #optimizer = optim.SGD(regressor_model.parameters(), lr=lr, momentum=0.9)
    # 设置学习率衰减因子
    lr_decay_factor = 0.1
    lr_scheduler_0 = optim.lr_scheduler.StepLR(regressor_optimizer, step_size=200, gamma=lr_decay_factor)
    lr_scheduler_1 = optim.lr_scheduler.StepLR(features_optimizer, step_size=200, gamma=lr_decay_factor)
    # 创建合并的模型;;;具体来说，CombinedModel 类是一个继承自 nn.Module 的自定义模型类。在这个类的构造函数中，我们将特征提取器和回归模型作为参数传入，并在模型中进行了组合。在模型的前向传播过程中，首先通过特征提取器（alexnet_features）提取输入数据的特征图，然后将特征图展平为一维张量，并传递给回归模型（regressor_model）进行相机参数的估计。
    

    # 准备输入数据和标签
    # 假设你已经有特征图 feature_map 和相机参数标签 camera_params
    
    Z_feature_extraction.to(device)
    Z_regressor_model.to(device)
    feature_extraction.to(device)
    regressor_model.to(device)



    #alexnet_features.to(device)
    # 训练合并的模型
      
    #best_loss = float(1000)
    best_train_loss = float('inf')  # 初始化为正无穷大，以便后续比较
    best_val_loss = float('inf')    # 初始化为正无穷大，以便后续比较
    num_epochs_without_improvement = 0
    num_train_batch = len(train_loader)
    num_val_batch = len(val_loader)
    best_train_epoch = 0
    best_val_epoch = 0
    base_path = f"/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Projects/{Task_name}"
    best_epoch_model_features_path = join(base_path, r"best_epoch_model_features.pth")
    best_epoch_model_regressor_path = join(base_path, r"best_epoch_model_regressor.pth")
    best_vali_model_features_path = join(base_path, r"best_vali_model_features.pth")
    best_vali_model_regressor_path = join(base_path, r"best_vali_model_regressor.pth" )
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    # 设置日志记录
    log_path = join(base_path, 'training_log.txt')
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    train_losses = []  # 初始化一个空列表用于存储训练损失值
    val_losses = []    # 初始化一个空列表用于存储验证损失值

    time0 = time.time()
     # 训练和验证过程
    print_message(f'{Task_name} Running!')
    for epoch in range(num_epochs):
        # 训练模型
        Z_feature_extraction.eval()
        Z_regressor_model.eval()
        feature_extraction.train()
        regressor_model.train()
        
        running_loss = 0.0
        start_time = time.time()
        
        # 假设你有一个数据加载器 train_loader，用于加载特征图和相机参数标签
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)

            features_optimizer.zero_grad()
            regressor_optimizer.zero_grad()            
        
            # 前向传播
            # Z predict net
            # Z_features = Z_feature_extraction(images)   
            # Z_predi= Z_regressor_model(Z_features)
            # angle predict net
            features = feature_extraction(images)
            #combined_features = torch.cat((features, Z_predi), dim=1)
            outputs = regressor_model(features)
            
            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            features_optimizer.step()
            regressor_optimizer.step()
            
            running_loss += loss.item()

        epoch_loss = running_loss / num_train_batch
        train_losses.append(epoch_loss)
        print_message(f"\nEpoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

        # 切换模型为验证模式，计算验证集损失
        feature_extraction.eval()
        regressor_model.eval()
        val_loss = 0.0
        # 初始化每个参数的MSE为0
        param_mses = [0.0] * num_params
        param_biases = [0.0] * num_params  # 用于保存每个参数的偏差
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # 前向传播
                # vali_Z_features = Z_feature_extraction(images)   
                # vali_Z_predi= Z_regressor_model(vali_Z_features)
                
                vali_features = feature_extraction(images)                
                #vali_combined_features = torch.cat((vali_features, vali_Z_predi), dim=1)
                vali_outputs = regressor_model(vali_features)

            
                

                
                # 如果预测结果和标签是经过归一化的值，进行还原
                denormalized_vali_outputs = denormalize_original_label(vali_outputs, para_ranges)
                denormalized_labels = denormalize_original_label(labels, para_ranges)
                
                # Cal validate loss
                vali_loss = criterion(vali_outputs, labels)
                val_loss += vali_loss.item()

                # 计算每个参数的mse
                for i, param_name in enumerate(param_names):
                    param_mse = torch.mean((denormalized_vali_outputs[:, i] - denormalized_labels[:, i])**2)
                    param_mses[i] += param_mse.item()
                    param_bias = torch.mean(torch.abs(denormalized_vali_outputs[:, i] - denormalized_labels[:, i]))
                    param_biases[i] += param_bias.item()

        val_epoch_loss = val_loss / num_val_batch
        param_mses = np.array(param_mses) / num_val_batch
        param_biases = np.array(param_biases) / num_val_batch
        val_losses.append(val_epoch_loss)

        endtime = time.time()
        epoch_time = endtime - start_time
        print_message(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_epoch_loss:.4f}, Time: {epoch_time:.2f} seconds")
        
        # 将训练信息写入日志文件
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}, Time: {epoch_time:.2f} seconds")
        
        # 打印每个参数的mse
        #num_val_samples = len(val_loader.dataset)
        for i, param_name in enumerate(param_names):
            #param_mse = param_mses[i] / num_val_samples
            ever_mse = f"Average {param_name} MSE: {param_mses[i]:.6f}, Bias: {param_biases[i]:.6f}"
            print(ever_mse)
            logging.info(ever_mse)
        
        # 保存当前模型
        # if epoch % 200 == 0:
        current_features_model_path = join(base_path, f"feature_extraction_model_latest.pth")  # 模型文件名格式可以自行定义
        current_regressor_model_path = join(base_path, f"regressor_model_latest.pth")
        torch.save(feature_extraction.state_dict(), current_features_model_path)
        torch.save(regressor_model.state_dict(), current_regressor_model_path)
        # Early stop
        # if abs(val_epoch_loss - best_val_loss) <= 0.0001:
        #     num_epochs_without_improvement += 1
        # else:
        #     num_epochs_without_improvement = 0

        # if num_epochs_without_improvement >= 10:
        #     print("Early stopping at epoch", epoch + 1)
        #     logging.info(f"Early Stop in Epoch: {epoch:.4f}")
        #     break
        
        # 在epoch结束后进行比较和更新最佳损失和轮次
        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            best_train_epoch = epoch + 1
            torch.save(feature_extraction.state_dict(), best_epoch_model_features_path)
            torch.save(regressor_model.state_dict(), best_epoch_model_regressor_path)
            #torch.save(combined_model.state_dict(), best_epoch_model_path)
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_val_epoch = epoch + 1
            torch.save(feature_extraction.state_dict(), best_vali_model_features_path)
            torch.save(regressor_model.state_dict(), best_vali_model_regressor_path)
            #torch.save(combined_model.state_dict(), best_vali_model_path) 
      

        # Plot the loss curve after each epoch
        
        plt.plot(range(1, epoch + 2), train_losses, label='Train Loss')
        plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()  # 添加图例
        plt.xlim(1, epoch)  # 设置横坐标范围从第一个epoch到最后一个epoch(no, do to now)
        plt.ylim(min(min(train_losses), min(val_losses)), max(max(train_losses), max(val_losses)))  # 设置纵坐标范围
        plt.savefig(join(base_path, 'loss_curve.png'))  # 保存当前epoch的loss曲线图
        plt.close()  # Close the current plot to start a new one for the next epoch

    time1 = time.time()
    total_time = time1 - time0

    print_message(f"Training complete! Total time: {total_time}")
    print_message(f"Best Train Loss: {best_train_loss:.4f}, Epoch: {best_train_epoch}")
    print_message(f"Best Validation Loss: {best_val_loss:.4f}, Epoch: {best_val_epoch}")
    logging.info(f"Best Train Loss: {best_train_loss:.4f}, Epoch: {best_train_epoch}")
    logging.info(f"Best Validation Loss: {best_val_loss:.4f}, Epoch: {best_val_epoch}")
    logging.info(f"Train Total Time: {total_time}")


