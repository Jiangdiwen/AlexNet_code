'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-08-18 13:39:04
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-08-30 14:39:53
FilePath: \code\dataloader_divided.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np



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
        side = data['side']
        projection = data['projection']
        d = data['d']
        # 将projection图像数据存储在列表中
        train_data_projections.append(projection)

        # 将rotation_angles和translation_params组合成一维向量，并存储在列表中
        vector_data = np.concatenate((rotation_angles, translation_params, [side], [d]))
        train_data_vectors.append(vector_data)

    # 将列表转换为NumPy数组
    train_data_projections = np.array(train_data_projections)
    train_data_vectors = np.array(train_data_vectors)
    
    return train_data_projections, train_data_vectors
