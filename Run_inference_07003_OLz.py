from Model.CameraRegressor import CameraParamRegressor_1, CombinedModel, CameraParamRegressor_4, AlexNet_Regressor_Z
from torchvision.models import alexnet
from torchvision.models.alexnet import AlexNet_Weights
from Dataloader.dataloader_npy import load_png_toarray, CustomDataset
import numpy as np
from time import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# 划分数据集为训练集和验证集
from sklearn.model_selection import train_test_split
from Train_split import normalize_0_1_label, denormalize_original_label, normalize_img
import torch.nn as nn
from os.path import join
import csv
import cv2

def denormalize_single_label(normalized_label, 
                             para_ranges = np.array([[-30, 30], [-30, 30], [-90, 90],
                                                     [-20, 20], [-20, 20], [-100, 50],
                                                     [0, 2]])):
    normalized_label = normalized_label.cpu().numpy() if normalized_label.is_cuda else normalized_label.numpy()
    
    original_label = np.zeros_like(normalized_label)
    for i in range(len(para_ranges)):
        original_label[i] = normalized_label[i] * (para_ranges[i, 1] - para_ranges[i, 0]) + para_ranges[i, 0]
    
    return torch.tensor(original_label)

def denormalize_original_label_OLz(normalized_labels, 
                                 para_ranges = np.array([[-30, 30], [-30, 30], [-90, 90],
                                                         [-20, 20],[-20, 20], [-150, 90],
                                                         [0, 2]])):    
    if normalized_labels.is_cuda:
        normalized_labels = normalized_labels.cpu()
    normalized_labels = normalized_labels.numpy()
    # 对normalized_labels的每一列参数进行还原
    original_labels = np.zeros_like(normalized_labels)

    original_labels = normalized_labels * (para_ranges[1] - para_ranges[0]) + para_ranges[0]

    return torch.tensor(original_labels)

def prepare_data_for_testing(data_list):
    data_transform = transforms.Compose([
    transforms.ToTensor(),          # 将图像转换为张量
    transforms.Resize((256, 256), antialias=True),   # 调整大小为AlexNet输入尺寸
    # 可以添加其他预处理转换
    # transforms.Lambda(to_float32),  
    # transforms.Normalize(mean=[0.], std=[1.])
    ])

    # 通过数据变换操作，将图像大小调整并转换为张量
    transformed_data = [data_transform(image) for image in data_list]     
    tensor_data = torch.stack(transformed_data)
    
    return tensor_data

if __name__ == "__main__":
    time0 = time()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #para_ranges = np.array([[-10, 10], [-20, 20], [-90, 90], [0, 2]])
    para_ranges = [-120, 90]


    time1 = time()
    
    AlexNet_Regressor = AlexNet_Regressor_Z().float()
    #base_path = "/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Projects/08-29_Task_OLangle_side_lr1e-06_64batch_3000epochs_3570trainDatas"
    base_path = "/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Projects/09-12_Task_OLz_07003_lr1e-06_64batch_3000epochs_5355trainDatas"
    Pretr_model_name = "best_vali_model.pth"
    Pretr_model_path = join(base_path, Pretr_model_name)
    AlexNet_Regressor.load_state_dict(torch.load(Pretr_model_path)) 
    
    time2 = time()

    RealDSA_data_path = '/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Regression_Model/data/Test_datas/07003Test_JPG'
    normal_image_list, gray_list = load_png_toarray(RealDSA_data_path)  # 二维图像矩阵的列表，每个元素是一个二维矩阵
    # labels = [...]  # 包含六个参数的一维向量的列表
    #data_list_rgb = np.array([np.stack((img, img, img), axis=-1) for img in normal_image_list]) #已经是一个包含RGB图像的列表，且图像维度为(height, width, channels)，
    rgb_image_list = []
    for img in normal_image_list:
        # 将灰度图像转换为三通道的RGB图像
        rgb_image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # 将转换后的RGB图像添加到列表中
        rgb_image_list.append(rgb_image)
    #prepared_data = prepare_data_for_testing(data_list_rgb)
 
    prepared_data = prepare_data_for_testing(rgb_image_list)
    AlexNet_Regressor.to(device)
    AlexNet_Regressor.eval()
    time3 = time()
    with torch.no_grad():
        
        
        feature_map = prepared_data
        feature_map = feature_map.to(device)  # 将测试数据转移到设备上
        time3_1 = time()
        test_outputs = AlexNet_Regressor(feature_map)
        #end_time0 = time()
        #denormalized_test_outputs = denormalize_original_label(test_outputs, para_ranges)
        denormalized_test_outputs = denormalize_original_label_OLz(test_outputs, para_ranges)
        
    time4 = time()
    print(f'predict_time: {time4 - time3}')
   #print(f'start-end_time: {end_time - start_time0}')
    output_list = []
    for i in range(len(denormalized_test_outputs)):
        output_list.append({
                'img': np.array(gray_list[i]),
                'predict_params': np.array(denormalized_test_outputs[i]),
                })

    #np.save('/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Regression_Model/data/Predict_datas/0903_predict_OLangle_07003DSA58.npy', output_list)

    cc = 1