from Model.CameraRegressor import CameraParamRegressor, CombinedModel
from torchvision.models import alexnet
from torchvision.models.alexnet import AlexNet_Weights
from Dataloader.dataloader_split import load_npydata
from Dataloader.dataloader_npy import CustomDataset
import numpy as np
from AlexNet_withRegressionModel import normalize_0_1_label, denormalize_original_label, normalize_img
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# 划分数据集为训练集和验证集
from sklearn.model_selection import train_test_split
from AlexNet_withRegressionModel import to_float32
import torch.nn as nn
from os.path import join
import csv

def denormalize_single_label(normalized_label, 
                             para_ranges = np.array([[-30, 30], [-30, 30], [-90, 90],
                                                     [-20, 20], [-20, 20], [-100, 50],
                                                     [0, 2]])):
    normalized_label = normalized_label.cpu().numpy() if normalized_label.is_cuda else normalized_label.numpy()
    
    original_label = np.zeros_like(normalized_label)
    for i in range(len(para_ranges)):
        original_label[i] = normalized_label[i] * (para_ranges[i, 1] - para_ranges[i, 0]) + para_ranges[i, 0]
    
    return torch.tensor(original_label)

if __name__ == "__main__":
    para_ranges = np.array([[-30, 30], [-30, 30], [-90, 90],
                            [-20, 20],[-20, 20], [-150, 90]])
    alexnet_model = alexnet(weights=AlexNet_Weights.DEFAULT).to('cuda')
    alexnet_features = alexnet_model.features
    regressor_model = CameraParamRegressor(input_size=256*6*6).float()
    combined_model = CombinedModel(alexnet_features, regressor_model)
    base_path = "/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Projects/08-16_Task_changeAngle_lr1e-05_32batch_1500epochs_2295trainDatas"
    Pretr_model_name = "best_vali_model.pth"
    Pretr_model_path = join(base_path, Pretr_model_name)
    combined_model.load_state_dict(torch.load(Pretr_model_path)) 

    Proj_data_path = '/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Regression_Model/data/Train_datas/0816_0-255train_2700datas.npy'
    data, labels = load_npydata(Proj_data_path)  # 二维图像矩阵的列表，每个元素是一个二维矩阵
    normal_data = normalize_img(data)
    normal_labels = normalize_0_1_label(labels, para_ranges)
    # labels = [...]  # 包含六个参数的一维向量的列表
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
    #test_dataset = CustomDataset(data=data_list_rgb, labels=normal_labels, transform=data_transform)
    #train_dataset = CustomDataset(data=train_data, labels=train_labels, transform=data_transform)
    val_dataset = CustomDataset(data=val_data, labels=val_labels, transform=data_transform)


    # 创建数据加载器
    batch_size = 32
    criterion = nn.MSELoss()  
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    num_batch = len(val_loader)
    device = 'cuda'
    combined_model.to(device)
    combined_model.eval()
    val_loss = 0.0
    val_losses = [] 
    param_names = ['Rx', 'Ry', 'Rz', 'Dx', 'Dy', 'Dz']
    num_params = len(param_names)
    # 切换模型为验证模式，计算验证集损失
    val_loss = 0.0
    # 初始化每个参数的MSE为0
    param_mses = [0.0] * num_params
    param_biases = [0.0] * num_params  # 用于保存每个参数的偏差       
    data_list = [] # 用于保存数据的列表
    with torch.no_grad():
        
        for feature_map, labels in val_loader:
            feature_map, labels = feature_map.to(device), labels.to(device)

            # 正向传播
            vali_outputs = combined_model(feature_map)
            
            # 如果预测结果和标签是经过归一化的值，进行还原
            denormalized_vali_outputs = denormalize_original_label(vali_outputs, para_ranges)
            denormalized_labels = denormalize_original_label(labels, para_ranges)
            
            for i in range(len(denormalized_vali_outputs)):
                data_list.append({
                'labels': np.array(denormalized_labels[i]),
                'outputs': np.array(denormalized_vali_outputs[i]),
                
                })
            # Cal validate loss 
            vali_loss = criterion(vali_outputs, labels)
            val_loss += vali_loss.item()

            # 计算每个参数的mse
            for i, param_name in enumerate(param_names):
                param_mse = torch.mean((denormalized_vali_outputs[:, i] - denormalized_labels[:, i])**2)
                param_mses[i] += param_mse.item()
                param_bias = torch.mean(torch.abs(denormalized_vali_outputs[:, i] - denormalized_labels[:, i]))
                param_biases[i] += param_bias.item()
    
    val_epoch_loss = val_loss / num_batch
    param_mses = np.array(param_mses) / num_batch
    param_biases = np.array(param_biases) /num_batch
    val_losses.append(val_epoch_loss)

    print(f'val_epoch_loss: {val_epoch_loss}')
    for i, param_name in enumerate(param_names):
        #param_mse = param_mses[i] / num_val_samples
        ever_mse = f"Average {param_name} MSE: {param_mses[i]:.6f}, Bias: {param_biases[i]:.6f}"
        print(ever_mse)
                                #np.set_printoptions(precision=2, suppress=True)
        #logging.info(ever_mse)
    # print(f'param_mses : {param_mses}')
    # print(f'param_biases_rate : {param_biases_rate:.2%}')

    # for data in data_list:
    #     print(data['labels'])
    #     print(data['outputs'], '\n')
        # labels = data['labels']
        # outputs = data['outputs']
        # file.write(f'Labels: {labels}\n')
        # file.write(f'Outputs: {outputs}\n\n')
    i = 0
    with open('/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Regression_Model/0802ttt_Predict_output.csv', 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Labels', 'Outputs'])  # Write header row
        
        for data in data_list:
            if i % 20 == 0:
                labels = np.around(data['labels'], decimals=2)
                outputs = np.around(data['outputs'], decimals=2)

            row_data = []  # List to hold data for each row
            row_data.extend(labels)  # Extend the row_data with labels
            row_data.extend(outputs)  # Extend the row_data with outputs
            csv_writer.writerow(row_data)  # Write a row with all the parameters
            i += 1
            

            

        
    # 预测相机参数 (one data)


    print(f'denormalized_labels[{i}]:{denormalized_labels[i]}')
    print(f'denormalized_vali_outputs[{i}]:{denormalized_vali_outputs[i]}')
    i += 1
    i = 0
    



    with torch.no_grad():
      
        combined_model.eval()
        fe_map = feature_map[i]  # 假设你有一个用于测试的特征图 feature_map
        fe_map = fe_map.to(device)  # 将测试数据转移到设备上
        predicted_params = combined_model(fe_map.unsqueeze(0))
        vali_loss = criterion(predicted_params, labels[i].unsqueeze(0))
        print(f'vali_loss[{i}]: {vali_loss}')
        i += 1
    np.save('/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Regression_Model/data/0803_predict_test600.npy', data_list)
    cc = 1


