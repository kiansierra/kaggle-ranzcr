import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional.classification import auroc

class AUC(Metric):
    def __init__(self,compute_on_step=False, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
        self.sigmoid = nn.Sigmoid()
        true_default = torch.cat([torch.ones((1,11)), torch.zeros((1,11))])
        self.add_state("true", default=true_default, dist_reduce_fx="sum")
        self.add_state("preds", default=0.5*torch.ones((2,11)), dist_reduce_fx="sum")
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        self.true = torch.cat([self.true, target], axis=0)
        self.preds = torch.cat([self.preds, self.sigmoid(preds)], axis=0)
    def compute(self):
        assert self.true.shape == self.preds.shape
        # if len(self.true) > 200:
        auroc_scores = [auroc(self.preds[:,num], self.true[:,num]) for num in range(self.true.shape[1])]
        return torch.tensor(auroc_scores).mean()
        # else:
        #     return np.mean([0.1]*self.true.shape[1])
# %%
class RaznrClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3,w=256,h=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.model  = EfficientNet.from_pretrained('efficientnet-b7', num_classes=11)
        self.loss = nn.BCEWithLogitsLoss()
        self.train_auc = AUC()
        self.val_auc = AUC()
        self.lr = lr
        self.example_input_array = torch.zeros((1,3,w,h))
    def compute_loss(self,y_hat, y):
        loss = self.loss(y_hat, y)
        return loss
    def log_step(self, loss, step='train'):
        self.log(f"{step}_loss", loss)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log_step(loss, step='train')
        self.train_auc.update(y_hat, y)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log_step(loss, step='val')
        self.val_auc.update(y_hat, y)
        return loss 
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y_hat 
    def on_epoch_end(self) -> None:
        self.log('train_auc', self.train_auc.compute(), on_step=False, prog_bar=True)
        self.log('val_auc', self.val_auc.compute(), on_step=False, prog_bar=True)
        return super().on_epoch_end()        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.3, patience=2, min_lr=1e-5)
        return {'optimizer': optimizer, 'scheduler':scheduler, 'monitor':'val_auc', 'interval':'epoch'}
#%%
class EfficientNetClf(RaznrClassifier):
    def __init__(self, *args, **kwargs):
        super(EfficientNetClf, self).__init__(*args, **kwargs)    
        self.model  = EfficientNet.from_pretrained('efficientnet-b7', num_classes=11)   
        #self.sigmoid = nn.Sigmoid() 
    def forward(self, x):
        out = self.model(x)
        #out = self.sigmoid(out)
        return out