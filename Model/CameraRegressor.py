'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-08-03 15:07:39
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-09-18 09:01:15
FilePath: /AlexNet_Regression_Model/CameraRegressor.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn

# torch函数定义的AlexNet特征提取模块
class AlexNetFeatures(nn.Module):
    def __init__(self):
        super(AlexNetFeatures, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    

class CameraParamRegressor_3(nn.Module):
    def __init__(self, input_size, dropout: float = 0.5):
        super(CameraParamRegressor_3, self).__init__()
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),                
            nn.Linear(input_size, 4096), #要把预训练Z网络的输出拼接到特征向量上
            nn.ReLU(),
            nn.Dropout(p=dropout),  
            nn.Linear(4096, 1024), #考虑到目前推理时间为0.8s以上了，所以在这块减少一个全连接层#  jiade meiyounamejiu shi msjibie
            nn.ReLU(),
            nn.Dropout(p=dropout),  
            nn.Linear(1024, 128), 
            nn.ReLU(), 
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.regressor(x)

        return x
    
# The output of the pre-trained network to predict the Z position is spliced with the feature vector，预测的是三个旋转角度
class CameraParamRegressor_3withZ_axis(nn.Module):
    def __init__(self, input_size, dropout: float = 0.5):
        super(CameraParamRegressor_3withZ_axis, self).__init__()
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),                
            nn.Linear(input_size + 1, 1024), #要把预训练Z网络的输出拼接到特征向量上
            nn.ReLU(),
            # nn.Dropout(p=dropout),  
            # nn.Linear(4096, 1024), #考虑到目前推理时间为0.8s以上了，所以在这块减少一个全连接层#  jiade meiyounamejiu shi msjibie
            # nn.ReLU(),
            nn.Dropout(p=dropout),  
            nn.Linear(1024, 128), 
            nn.ReLU(), 
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.regressor(x)

        return x



class AlexNet_Regressor_Z(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super(AlexNet_Regressor_Z, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),                
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),  
            nn.Linear(1024, 128), 
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x



class CameraParamRegressor_1(nn.Module):
    def __init__(self, input_size, dropout: float = 0.5):
        super(CameraParamRegressor_1, self).__init__()
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),                
            nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),  
            nn.Linear(4096, 1024),
            nn.ReLU(),
            #nn.Dropout(p=dropout),  
            nn.Linear(1024, 128), 
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.regressor(x)

        return x


class CameraParamRegressor_4(nn.Module):
    def __init__(self, input_size, dropout: float = 0.5):
        super(CameraParamRegressor_4, self).__init__()
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),                
            nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),  
            nn.Linear(4096, 1024),
            nn.ReLU(),
            #nn.Dropout(p=dropout),  
            nn.Linear(1024, 128), 
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(input_size, 512)   # 全连接层1，输入维度256*6*6，输出维度512
        # self.fc2 = nn.Linear(512, 256)           # 全连接层2，输入维度512，输出维度256
        # self.fc3 = nn.Linear(256, 128)           # 全连接层3，输入维度256，输出维度128
        # self.fc4 = nn.Linear(128, output_size)             # 全连接层4，输入维度128，输出维度6
        # self.sigmoid = nn.Sigmoid()  # 添加Sigmoid激活函数
        # self.relu = nn.ReLU()

    def forward(self, x):
        x = self.regressor(x)
        # x = self.flatten(x)     # 将特征图展平成一维向量
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        # x = self.fc4(x)         # 最后输出6个参数
        # x = self.sigmoid(x)  # 在最后输出前应用Sigmoid激活函数
        return x



# 添加回归层
class CameraParamRegressor_6(nn.Module):
    def __init__(self, input_size, dropout: float = 0.5):
        super(CameraParamRegressor_6, self).__init__()
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),                
            nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),  
            nn.Linear(4096, 1024),
            nn.ReLU(),
            #nn.Dropout(p=dropout),  
            nn.Linear(1024, 128), 
            nn.ReLU(),
            nn.Linear(128, 6),
            #nn.Sigmoid()
        )
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(input_size, 512)   # 全连接层1，输入维度256*6*6，输出维度512
        # self.fc2 = nn.Linear(512, 256)           # 全连接层2，输入维度512，输出维度256
        # self.fc3 = nn.Linear(256, 128)           # 全连接层3，输入维度256，输出维度128
        # self.fc4 = nn.Linear(128, output_size)             # 全连接层4，输入维度128，输出维度6
        # self.sigmoid = nn.Sigmoid()  # 添加Sigmoid激活函数
        # self.relu = nn.ReLU()

    def forward(self, x):
        x = self.regressor(x)
        # x = self.flatten(x)     # 将特征图展平成一维向量
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        # x = self.fc4(x)         # 最后输出6个参数
        # x = self.sigmoid(x)  # 在最后输出前应用Sigmoid激活函数
        return x

class CameraParamRegressor_7(nn.Module):
    def __init__(self, input_size, dropout: float = 0.5):
        super(CameraParamRegressor_7, self).__init__()
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),                
            nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),  
            nn.Linear(4096, 1024),
            nn.ReLU(),
            #nn.Dropout(p=dropout),  
            nn.Linear(1024, 128), 
            nn.ReLU(),
            nn.Linear(128, 7),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.regressor(x)

        return x


class CameraParamRegressor_8(nn.Module):
    def __init__(self, input_size, dropout: float = 0.5):
        super(CameraParamRegressor_8, self).__init__()
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),                
            nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),  
            nn.Linear(4096, 1024),
            nn.ReLU(),
            #nn.Dropout(p=dropout),  
            nn.Linear(1024, 128), 
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.regressor(x)

        return x

# 添加回归层
class Alex_Camera_Net(nn.Module):
    def __init__(self, output_size = 6, dropout: float = 0.5):
        super(Alex_Camera_Net, self).__init__()
        self.AlexNet_feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),                
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),  
            nn.Linear(4096, 1024),
            nn.ReLU(),
            #nn.Dropout(p=dropout),  
            nn.Linear(1024, 128), 
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.AlexNet_feature(x)
        #print("Output feature map size:", x.size())  # 添加打印语句
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        
        return x



# 将特征提取器和回归模型合并
class CombinedModel(nn.Module):
    def __init__(self, feature_extractor, regressor):
        super(CombinedModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.regressor = regressor
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.feature_extractor(x)
        #print("Output feature map size:", x.size())  # 添加打印语句
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x
    