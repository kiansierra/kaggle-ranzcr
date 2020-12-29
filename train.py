#%%
import os
from albumentations.augmentations.transforms import Normalize
import cv2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
#%%
from datamodules import RaznrDataModule
from models import EfficientNetClf
#%%
pl.seed_everything(seed=42)
DATA_DIR = '../ranzcr-clip-catheter-line-classification'
IMAGE_SIZE = 256

#%%
def get_train_transforms(w=512,h=512):
    transform = A.Compose([
        A.RandomResizedCrop(width=w, height=h),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2),
        # A.ToFloat(max_value=255),
        A.Normalize(),
        ToTensorV2()])
    return transform
#%%
def get_val_transforms(w=512,h=512):
    transform = A.Compose([
        A.Resize(width=w, height=h),
        
        # A.HorizontalFlip(p=0.25),
        # A.VerticalFlip(p=0.25),
        # A.ToFloat(max_value=255),
        A.Normalize(),
        ToTensorV2()])
    return transform   
#%%
def get_test_transforms(w=512,h=512):
    transform = A.Compose([
        A.Resize(width=w, height=h),
        # A.ToFloat(max_value=255),
        A.Normalize(),
        ToTensorV2()])
    return transform    

#%%
if __name__=="__main__":   
    model_name = 'efficientnet-b7'
    # logger = WandbLogger(model_name, project='raznr', offline=False)
    #%%
    size = {'w':IMAGE_SIZE, 'h':IMAGE_SIZE}
    datamodule_args = {'datadir':DATA_DIR ,'batch_size':6, 'num_workers':3,  'val_pct':0.2,
        'train_transforms':get_train_transforms(**size), 'val_transforms':get_val_transforms(**size), 'test_transforms': get_test_transforms(**size)}
    #%%
    model_args = {'lr':5e-3}  
    model_args.update(size)
    model = EfficientNetClf(**model_args)
    # logger.watch(model, log='all', log_freq=100)
    #%%
    model_ckpt = ModelCheckpoint(filename="{val_auc:.2f}-{val_loss:.2f}-{epoch}", monitor='val_loss', verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopper = EarlyStopping(monitor='val_auc', verbose=True)
    trainer_args = {'gpus':1, 'precision':16, 'max_epochs':6, 'checkpoint_callback':model_ckpt, 'callbacks': [lr_monitor, early_stopper], 'profiler':'simple'}
    # trainer_args['logger'] = logger
    #%%
    trainer = pl.Trainer(**trainer_args) 
    raznrdatamodule = RaznrDataModule(**datamodule_args)  
    trainer.fit(model, raznrdatamodule)
    trainer.test(model, raznrdatamodule.test_dataloader())
    test_results = model.test_results.cpu().numpy()
    sub_df = raznrdatamodule.test_df
    sub_df.iloc[:,1:] = test_results
    sub_df.to_csv('submission.csv', index=False)
# %%
