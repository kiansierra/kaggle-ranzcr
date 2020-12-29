import numpy as np
from pytorch_lightning.loggers import wandb
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional.classification import auroc
import wandb

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
        return self.compute()
    def compute(self):
        assert self.true.shape == self.preds.shape
        # if len(self.true) > 200:
        auroc_scores = [auroc(self.preds[:,num], self.true[:,num]) for num in range(self.true.shape[1])]
        return torch.tensor(auroc_scores).mean()
        # else:
        #     return np.mean([0.1]*self.true.shape[1])
# %%
class RaznrClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3,w=256,h=256, logging=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        # self.model  = EfficientNet.from_pretrained('efficientnet-b7', num_classes=11)
        self.loss = nn.BCEWithLogitsLoss()
        self.train_auc = AUC()
        self.val_auc = AUC()
        self.sigmoid = nn.Sigmoid()
        self.lr = lr
        self.logging = logging
        self.example_input_array = torch.zeros((1,3,w,h))
    def compute_loss(self,y_hat, y):
        loss = self.loss(y_hat, y)
        return loss
    def log_step(self,x, y_hat, y, loss, batch_idx,  step='train'):
        self.log(f"{step}_loss", loss)
        self.log(f"{step}_auc", getattr(self, f"{step}_auc").update(y_hat, y), prog_bar=True)
        if batch_idx %200 ==0: self.logger.experiment.log({step:[wandb.Image(img) for img in x.permute(0,2,3,1).cpu().numpy()]})
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        if self.logging: self.log_step(x, y_hat, y, loss, batch_idx, step='train')
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        if self.logging: self.log_step(x, y_hat, y, loss, batch_idx, step='val')
        return loss 
    def on_test_epoch_start(self) -> None:
        self._test_results = []
        return super().on_test_epoch_start()     
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.sigmoid(self(x))
        self._test_results.append(y_hat)
        return y_hat 
    def on_test_epoch_end(self) -> None:
        self.test_results = torch.cat(self._test_results)
        return super().on_test_epoch_end()
    def on_epoch_end(self) -> None:
        self.train_auc.reset()
        self.val_auc.reset()
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.3, patience=2, min_lr=1e-5, verbose=True)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
        return {'optimizer': optimizer, 'lr_scheduler':scheduler, 'monitor':'val_auc', 'interval':'epoch'}
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