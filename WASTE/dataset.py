import os
import json
from PIL import Image
import pandas as pd
from sklearn import model_selection, metrics
import torch
from torch.utils.data import Dataset 
from utils import *
import cv2 





### READ IN CSV FILE ### DEFINE DIRECTORIES
##########-----------##################
df = pd.read_csv("/notebooks/pixels-CLS/RICE/DATA/TRAIN.csv", encoding='unicode_escape')
 
### SPLIT FUNCTION
train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.25, random_state=42, stratify=df.Label.values)

test_df = pd.read_csv("/notebooks/pixels-CLS/RICE/DATA/TEST.csv", encoding='unicode_escape')
sample_submission = pd.read_csv("/notebooks/pixels-CLS/RICE/DATA/SampleSubmission.csv", encoding='unicode_escape')

######################
##############
####
TRAIN_PATH = "/notebooks/pixels-CLS/RICE/DATA/TRAIN-IMAGES"
TEST_PATH = "/notebooks/pixels-CLS/RICE/DATA/TEST-IMAGES"




# ====================================================
# Dataset
# ====================================================

    
    
class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['Image_id'] .values
        self.labels = df['Label'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df) 

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TRAIN_PATH}/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #if self.transform is not None:
        image = self.transform(image)#=image)["image"]
        #if self.transform:
        #    augmented = self.transform(image)['image']
        #if self.transform is not None:
            #img = self.transform(img)
        #img = np.array(image)

        label = torch.tensor(self.labels[idx]).long()
        return image, label
    
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # #https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
    
    
    
    

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['Image_id'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TEST_PATH}/{file_name}'
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        return images

        #images, labels = tuple(zip(*batch))
        #labels = torch.as_tensor(labels)
         #,labels
    
    






### DEFINE TRAIN, VALIDATION AND TEST

### DATA TRANSFORMATIONS
