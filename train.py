#%%
import os
import cv2
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
#%%
from datamodules import RaznrDataModule
from models import EfficientNetClf
#%%
DATA_DIR = '../ranzcr-clip-catheter-line-classification'
#%%
pl.seed_everything(seed=42)
os.listdir(DATA_DIR)

#%%
def get_train_transforms(w=512,h=512):
    transform = A.Compose([
        A.RandomResizedCrop(width=w, height=h),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ToFloat(max_value=255),
        ToTensorV2()])
    return transform
#%%
def get_val_transforms(w=512,h=512):
    transform = A.Compose([
        A.RandomResizedCrop(width=w, height=h),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        A.ToFloat(max_value=255),
        ToTensorV2()])
    return transform   
#%%
def get_test_transforms(w=512,h=512):
    transform = A.Compose([
        A.RandomResizedCrop(width=w, height=h),
        A.ToFloat(max_value=255),
        ToTensorV2()])
    return transform    

#%%
if __name__=="__main__":
    wandb.login(key=os.environ['WANDB_KEY'])
    logger = WandbLogger(name='efficientnet-b7', project='raznr')
    size = {'w':256, 'h':256}
    model_args = {'lr':5e-2}  
    model = EfficientNetClf(**model_args)
    logger.watch(model, log='all', log_freq=100)

    trainer_args = {'logger':logger,'gpus':1, 'precision':16, 'max_epochs':5}
    trainer = pl.Trainer(**trainer_args)
    datamodule_args = {'datadir':DATA_DIR ,'batch_size':3, 'num_workers':0, 
        'train_transforms':get_train_transforms(**size), 'val_transforms':get_val_transforms(**size), 'test_transforms': get_test_transforms(**size)}
    raznrdatamodule = RaznrDataModule(**datamodule_args)
    trainer.fit(model, raznrdatamodule)


# %%
