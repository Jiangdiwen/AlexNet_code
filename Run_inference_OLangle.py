'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-08-14 23:06:44
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-09-14 10:23:31
FilePath: /code/Run_inference_PretreDSA.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from Model.CameraRegressor import CameraParamRegressor_7, CombinedModel,CameraParamRegressor_4
from torchvision.models import alexnet
from torchvision.models.alexnet import AlexNet_Weights
from Dataloader.dataloader_npy import load_npydata, CustomDataset
import numpy as np
from Train_split import normalize_0_1_label, denormalize_original_label, normalize_img
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# 划分数据集为训练集和验证集
from sklearn.model_selection import train_test_split
from Train_split import to_float32
import torch.nn as nn
from os.path import join
import csv
import time

def denormalize_single_label(normalized_label, 
                             para_ranges = np.array([[-30, 30], [-30, 30], [-90, 90],
                                                     [-20, 20], [-20, 20], [-100, 50],
                                                     [0, 2]])):
    normalized_label = normalized_label.cpu().numpy() if normalized_label.is_cuda else normalized_label.numpy()
    
    original_label = np.zeros_like(normalized_label)
    for i in range(len(para_ranges)):
        original_label[i] = normalized_label[i] * (para_ranges[i, 1] - para_ranges[i, 0]) + para_ranges[i, 0]
    
    return torch.tensor(original_label)
def prepare_data_for_testing(data_list):
    data_transform = transforms.Compose([
    transforms.ToTensor(),          # 将图像转换为张量
    transforms.Resize((256, 256), antialias=True),   # 调整大小为AlexNet输入尺寸
    # 可以添加其他预处理转换
    # transforms.Lambda(to_float32),  
    # transforms.Normalize(mean=[0.], std=[1.])
    ])
    normal_image_list = []
    for img in data_list:
        mean = np.mean(img)
        std = np.std(img)
        standardized_image = (img - mean) / std 
        standardized_image_float32 = standardized_image.astype(np.float32)
        normal_image_list.append(standardized_image_float32)
    # 通过数据变换操作，将图像大小调整并转换为张量
    transformed_data = [data_transform(image) for image in normal_image_list]
     
    tensor_data = torch.stack(transformed_data)
    
    return tensor_data

if __name__ == "__main__":
    param_names = ['Rx', 'Ry', 'Rz']
    # labels = [...]  # 包含六个参数的一维向量的列表
    para_ranges = np.array([[-10, 0], [-20, 20], [-90, 90], [0, 2]])
    alexnet_model = alexnet(weights=AlexNet_Weights.DEFAULT).to('cuda')
    alexnet_features = alexnet_model.features
    regressor_model = CameraParamRegressor_4(input_size=256*6*6).float()
    combined_model = CombinedModel(alexnet_features, regressor_model)
    base_path = "/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Projects/08-29_Task_OLangle_side_lr1e-06_64batch_3000epochs_3570trainDatas"
    Pretr_model_name = "best_vali_model.pth"
    Pretr_model_path = join(base_path, Pretr_model_name)
    combined_model.load_state_dict(torch.load(Pretr_model_path)) 

    RealDSA_data_path = '/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Regression_Model/data/Test_datas/0821_Pretre_22DSA.npy'
    normal_image_list = np.load(RealDSA_data_path, allow_pickle=True)  # 二维图像矩阵的列表，每个元素是一个二维矩阵
    # labels = [...]  # 包含六个参数的一维向量的列表
    data_list_rgb = np.array([np.stack((img, img, img), axis=-1) for img in normal_image_list]) #已经是一个包含RGB图像的列表，且图像维度为(height, width, channels)，
    
    prepared_data = prepare_data_for_testing(data_list_rgb)
 
    device = 'cuda'
    combined_model.to(device)
    combined_model.eval()
    start_time = time.time()
    with torch.no_grad():
        
        combined_model.eval()
        feature_map = prepared_data
        feature_map = feature_map.to(device)  # 将测试数据转移到设备上
        test_outputs = combined_model(feature_map)
        denormalized_test_outputs = denormalize_original_label(test_outputs, para_ranges)
    end_time = time.time()
    print(f'predict_time: {end_time - start_time}')
        #np.set_printoptions(precision=2, suppress=True)
    output_list = []
    for i in range(len(denormalized_test_outputs)):
        output_list.append({
                'img': np.array(normal_image_list[i]),
                'predict_params': np.array(denormalized_test_outputs[i]),
                })

    #np.save('/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Regression_Model/data/Predict_datas/0823_OLangle_Pretre2_22DSA.npy', output_list)
    cc = 1
