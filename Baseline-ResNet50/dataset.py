import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import math

from PIL import Image
import os 
    
def get_label_unlabel_dataset(csv_path, img_path, ratio):
    data = pd.read_csv(csv_path)
    labels = np.asarray(data['label'])
    
    # Getting the percentage of positive and negative samples
    ratio_of_1 = np.count_nonzero(labels==1) / len(labels)
    ratio_of_0 = np.count_nonzero(labels==0) / len(labels)
    
    # Getting the number of labeled samples for both
    # positive and negative sampls according ot the "ratio" parameter
    num_1 = get_num_label(ratio, ratio_of_1, labels)
    num_0 = get_num_label(ratio, ratio_of_0, labels)

    # Getting the index of the labeled images in the csv file.
    labeled_img_idx_0 = np.random.choice(np.asarray(np.where(labels==0)).reshape(np.count_nonzero(labels==0),), size=num_0)
    labeled_img_idx_1 = np.random.choice(np.asarray(np.where(labels==1)).reshape(np.count_nonzero(labels==1),), size=num_1)

    # Concacenating the positive and negative labeled samples and returning their dataset.
    labeled_imgs = data.loc[np.hstack((labeled_img_idx_0, labeled_img_idx_1))]
    labeled_dataset = LabeledCancerDataset(labeled_imgs, img_path)

    return labeled_dataset


class LabeledCancerDataset(Dataset):
    def __init__(self, labeled_csv, img_path):
        self.labeled_csv = labeled_csv
        self.img_path = img_path

        self.image_arr = np.asarray(self.labeled_csv.iloc[:, 0])
        self.label_arr = np.asarray(self.labeled_csv.iloc[:, 1])
        self.data_len = len(labeled_csv)

        self.transformations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48,48)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        single_image_label = self.label_arr[index]
        img_as_img = np.asarray(Image.open(os.path.join(self.img_path, single_image_name)))
        
        trans_img = self.transformations(np.array(img_as_img))
        return (trans_img, single_image_label)

    def __len__(self):
        return self.data_len

def get_num_label(ratio, ratio_of_num, labels):
    return math.ceil(len(labels) * ratio_of_num * ratio)

def get_train_validation_test_data(train_csv_path, train_img_path, val_csv_path, val_img_path, test_csv_path, test_img_path):
    train_data = pd.read_csv(train_csv_path)
    val_data = pd.read_csv(val_csv_path)
    test_data = pd.read_csv(test_csv_path)

    train_dataset = LabeledCancerDataset(train_data, train_img_path)
    val_dataset = LabeledCancerDataset(val_data, val_img_path)
    test_dataset = LabeledCancerDataset(test_data, test_img_path)

    return train_dataset, val_dataset, test_dataset