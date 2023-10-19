'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-09-12 09:55:43
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-09-18 08:58:01
FilePath: \code\Run_inference_07003_Angle_PreZ.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from Model.CameraRegressor import AlexNetFeatures, CameraParamRegressor_3withZ_axis, CameraParamRegressor_1, CombinedModel
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

if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    param_names = ['Rx', 'Ry', 'Rz']
    # labels = [...]  # 包含六个参数的一维向量的列表
    para_ranges = np.array([[-10, 10], [-20, 20], [-90, 90]])
    para_ranges_Z = [-60, 150]
    #-------------------------------------load pretrained model---------------------------
    # alexnet_model = alexnet(weights=AlexNet_Weights.DEFAULT).to(device)
    # alexnet_features = alexnet_model.features    
    # regressor_model_Z = CameraParamRegressor_1(input_size=256*6*6).float()
    # Pretrain_Z_model = CombinedModel(alexnet_features, regressor_model_Z)
    # #base_path = "/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Projects/08-29_Task_OLangle_side_lr1e-06_64batch_3000epochs_3570trainDatas"
    # base_path = "/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Projects/09-01_Task_OLz_07003_lr1e-06_64batch_3000epochs_5355trainDatas"
    # Pretr_model_name = "best_vali_model.pth"
    # Pretr_model_path = join(base_path, Pretr_model_name)
    # Pretrain_Z_model.load_state_dict(torch.load(Pretr_model_path)) 
    Z_feature_extraction = AlexNetFeatures()
    Z_regressor_model = CameraParamRegressor_1(input_size=256 * 6 * 6).float()
    base_path = "/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Projects/09-16_Task_OLz_07003_divide_modellr_features1e-05_lr_regressor1e-05_64batch_800epochs_5049trainDatas"
    Z_Pretr_feature_model_name = "best_vali_model_features.pth"
    Z_Pretr_feature_model_path = join(base_path, Z_Pretr_feature_model_name)
    Z_Pretr_regressor_model_name = "best_vali_model_regressor.pth"
    Z_Pretr_regressor_model_path = join(base_path, Z_Pretr_regressor_model_name)
    Z_feature_extraction.load_state_dict(torch.load(Z_Pretr_feature_model_path)) 
    Z_regressor_model.load_state_dict(torch.load(Z_Pretr_regressor_model_path)) 


    feature_extraction = AlexNetFeatures()
    regressor_model = CameraParamRegressor_3withZ_axis(input_size=256 * 6 * 6).float()
    base_path = "/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Projects/09-17_Task_Angle_PreZ_07003lr_features1e-05_lr_regressor1e-05_64batch_1500epochs_5049trainDatas"
    Pretr_feature_model_name = "best_vali_model_features.pth"
    Pretr_feature_model_path = join(base_path, Pretr_feature_model_name)
    Pretr_regressor_model_name = "best_vali_model_regressor.pth"
    Pretr_regressor_model_path = join(base_path, Pretr_regressor_model_name)

    feature_extraction.load_state_dict(torch.load(Pretr_feature_model_path)) 
    regressor_model.load_state_dict(torch.load(Pretr_regressor_model_path)) 
    #-------------------------------------load 07003 jpg datas------------------------------------
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


    Z_feature_extraction.to(device)    
    Z_regressor_model.to(device)
    feature_extraction.to(device)
    regressor_model.to(device)

    Z_feature_extraction.eval()
    Z_regressor_model.eval()
    feature_extraction.eval()
    regressor_model.eval()



    # GPU预热
    random_input = torch.randn(1, 3, 256, 256).to(device)     
    for _ in range(50):
        _ = Z_feature_extraction(random_input)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    iterations = 300   # 重复计算的轮次
    times = torch.zeros(iterations)     # 存储每轮iteration的时间

    time3 = time()
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            images = prepared_data
            images = images.to(device)
            time3_1 = time()
            # go
            # Z_feature = Z_feature_extraction(images)
            # Z_predi= Z_regressor_model(Z_feature)
            time3_2 = time()
    
            # go
            Z_feature = Z_feature_extraction(images)
            Z_predi= Z_regressor_model(Z_feature)
            time3_3 = time()

            features = feature_extraction(images)
            combined_features = torch.cat((features, Z_predi), dim=1)
            test_outputs = regressor_model(combined_features)
            # feature_map = feature_map.to(device)  # 将测试数据转移到设备上
            # test_outputs = combined_model(feature_map)
            #end_time0 = time()
            denormalized_test_outputs = denormalize_original_label(test_outputs, para_ranges)
            denormalized_test_outputs_Z = denormalize_original_label_OLz(Z_predi, para_ranges_Z)   

            ender.record()
            # 同步GPU时间
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # 计算时间
            times[iter] = curr_time
        
    time4 = time()
    print(f'Total_predict_time: {(time4 - time3)*1000:.6f}ms')
    #print(f'First_PretrainZ_predict_time: {time3_2- time3_1}')
    print(f'PretrainZ_predict_time: {(time3_3- time3_2)*1000:.6f}ms')
    print(f'Angle_predict_time: {(time4 - time3_3)*1000:.6f}ms')
    mean_time = times.mean().item()
    print(f"Use_pytorch_times: Inference time: {mean_time:.6f}ms, FPS: {1000/mean_time} ")
    
   #print(f'start-end_time: {end_time - start_time0}')
    output_list = []
    for i in range(len(denormalized_test_outputs)):
        output_list.append({
                'img': np.array(gray_list[i]),
                'predict_params_Angle': np.array(denormalized_test_outputs[i]),
                'predict_params_Z': np.array(denormalized_test_outputs_Z[i])
                })

    #np.save('/home/jyx/Jiangyanxin/Reg_DSA/AlexNet_Regression_Model/data/Predict_datas/0917_predict_Angle_PreZ_lr1e-05_07003_DSA58.npy', output_list)


    cc = 1

