#%%
import pandas as pd 
import os
import cv2
from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
#%%
class RaznrDataset(Dataset):
    def __init__(self, df, datadir, transforms=None, sample='train'):
        self.df = df
        self.datadir = datadir
        self.transforms = transforms
        self.sample = sample
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'StudyInstanceUID']
        img_path = f"{self.datadir}/{self.sample}/{img_id}.jpg"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        targets = self.df.iloc[idx, 1:12].values
        if self.transforms:
            img = self.transforms(image=img)['image']
        return img, targets.astype('float32')

# %%
class RaznrDataModule(pl.LightningDataModule):
    def __init__(self,datadir, batch_size=4, num_workers=0, train_transforms=None, val_transforms=None, test_transforms=None):
        super(RaznrDataModule,self).__init__()
        self.datadir = datadir
        self.bs = batch_size
        self.num_workers= num_workers
        self.train_transforms = val_transforms
        self.val_transforms = train_transforms
        self.test_transforms = test_transforms
    def prepare_data(self, *args, **kwargs):
        train_df = pd.read_csv(os.path.join(self.datadir, 'train.csv'))
        self.test_df = pd.read_csv(os.path.join(self.datadir,"sample_submission.csv"))
        train, val = train_test_split(train_df)
        self.train_df = train.reset_index(drop=True)
        self.val_df = val.reset_index(drop=True)
        self.train_ds = RaznrDataset(self.train_df, self.datadir, transforms=self.train_transforms)
        self.val_ds = RaznrDataset(self.val_df, self.datadir, transforms=self.val_transforms)
        self.test_ds = RaznrDataset(self.test_df, self.datadir, transforms=self.test_transforms, sample='test')
        return super().prepare_data(*args, **kwargs)
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        subset = Subset(self.train_ds, indices=range(2000))
        sampler = RandomSampler(subset)
        return DataLoader(subset, batch_size=self.bs, num_workers=self.num_workers, sampler=sampler)
    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.bs, num_workers=self.num_workers)
    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.bs, num_workers=self.num_workers)