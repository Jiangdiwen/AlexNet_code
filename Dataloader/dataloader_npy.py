'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-08-01 08:32:45
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-08-18 13:50:41
FilePath: /AlexNet_Regression_Model/dataloader_npy.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AEc
'''
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import cv2

def load_npydata(Proj_data_path):
    #Proj_data_path = r'E:\AAAAA\AAAA\A_Science\A_P08\DSA_Reg\Fake_DSA\data_0726t1_inkGood.npy'

    loaded_data = np.load(Proj_data_path, allow_pickle=True)
    # 初始化用于存储训练数据的列表
    train_data_projections = []
    train_data_vectors = []

    # 处理每例数据
    for data in loaded_data:
        rotation_angles = data['rotation_angles']
        translation_params = data['translation_params']
        projection = data['projection']

        # 将projection图像数据存储在列表中
        train_data_projections.append(projection)

        # 将rotation_angles和translation_params组合成一维向量，并存储在列表中
        vector_data = np.concatenate((rotation_angles, translation_params))
        train_data_vectors.append(vector_data)

    # 将列表转换为NumPy数组
    train_data_projections = np.array(train_data_projections)
    train_data_vectors = np.array(train_data_vectors)
    
    return train_data_projections, train_data_vectors




def load_png_toarray(image_folder):
    # 文件夹路径
  
    # 获取文件夹中所有的文件
    image_files = [file for file in os.listdir(image_folder) if file.lower().endswith(".jpg")]

    normal_image_list = []
    gray_list = []

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if gray_image is not None:
            mean = np.mean(gray_image)
            std = np.std(gray_image)
            standardized_image = (gray_image - mean) / std  # Zero mean and unit variance
            standardized_image_float32 = standardized_image.astype(np.float32)
            
            gray_list.append(gray_image)
            normal_image_list.append(standardized_image_float32)

    # 现在image_matrices列表包含了所有图像的矩阵
    # 您可以根据需要进行后续处理

    
    return normal_image_list, gray_list

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]

        # 可选：应用数据预处理转换
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
