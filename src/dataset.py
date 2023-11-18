import cv2
import pandas as pd
from typing import Optional
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from .config import (NUM_CLASSES, DATA_DIR, RESIZE_SIZE, 
                     CROP_SIZE, BATCH_SIZE, NUM_WORKERS)

class ImageDataset(Dataset):
    def __init__(self, df, input_shape, transform=None):
        self.df = df
        self.transform = transform
        self.input_shape = input_shape

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'img_path']
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        xmin, ymin, xmax, ymax = self.df.loc[idx, ['xmin', 'ymin', 'xmax', 'ymax']].values
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        img = img[ymin:ymax, xmin:xmax]
        img = cv2.resize(img, self.input_shape[:-1], interpolation=cv2.INTER_LINEAR)
        # TODO: you may not need to resize the image. Do that in the data transform

        if self.transform:
            img = self.transform(img)

        sparse_label = int(self.df.loc[idx, 'sparse_label'])
        sparse_label_tensor = torch.tensor(sparse_label)

        cat_label = F.one_hot(sparse_label_tensor, num_classes=NUM_CLASSES)
        return torch.from_numpy(img.transpose((2, 0, 1))), cat_label




class DataModule(pl.LightningDataModule):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.resize_size = RESIZE_SIZE[model_name][:-1]
        self.crop_size = CROP_SIZE[model_name][:-1]
        self.batch_size = BATCH_SIZE
        self.num_workers = NUM_WORKERS


    def prepare_data(self) -> None:
        self.df = pd.read_csv(f'{DATA_DIR}/verified_annotation_from_xml.csv')
        self.df['img_path'] =f'{DATA_DIR}/images/' + self.df['image_name']
        self.df.drop(columns=['Unnamed: 0'], inplace=True)
        self.df['label_name'] = self.df['label_name'].apply(lambda x: x.lower())
        self.df['sparse_label'] = self.df['label_name'].map({'atopic': 0, 'papular': 1,'scabies': 2})

    def setup(self, stage: Optional[str] = None) -> None:

        gs = GroupShuffleSplit(n_splits=2, train_size=.85, random_state=42)

        train_val_idx, test_idx = next(gs.split(self.df,groups=self.df.patient_id))
        train_val_df = self.df.iloc[train_val_idx]
        test_df = self.df.iloc[test_idx]

        train_idx, val_idx = next(gs.split(train_val_df, groups=train_val_df.patient_id))
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        self.train_ds = ImageDataset(
            df=train_df, 
            input_shape=self.resize_size, 
            transform=transforms.Compose(
                [
                    transforms.ToPILImage(),
                    # transforms.Resize(self.resize_size),
                    transforms.CenterCrop(self.crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        )
        self.val_ds = ImageDataset(
            df=val_df, 
            input_shape=self.resize_size, 
            transform=transforms.Compose(
                [
                    transforms.ToPILImage(),
                    # transforms.Resize(self.resize_size),
                    transforms.CenterCrop(self.crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        )

        self.test_ds = ImageDataset(
            df=test_df, 
            input_shape=self.resize_size, 
            transform=transforms.Compose(
                [
                    transforms.ToPILImage(),
                    # transforms.Resize(self.resize_size),
                    transforms.CenterCrop(self.crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )   
    
