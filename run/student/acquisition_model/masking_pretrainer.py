import torch
import torch.optim as optim
import pytorch_lightning as pl
from acquisition_model.utils import generate_uniform_mask


class MaskingPretrainer(pl.LightningModule):


    def __init__(self,
                 model,
                 mask_layer,
                 lr,
                 loss_fn,
                 val_loss_fn,
                 factor=0.2,
                 patience=2,
                 min_lr=1e-6,
                 early_stopping_epochs=None):
        super().__init__()


        self.model = model
        self.mask_layer = mask_layer
        self.mask_size = self.mask_layer.mask_size


        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
        self.early_stopping_epochs = early_stopping_epochs


        self.loss_fn = loss_fn
        self.val_loss_fn = val_loss_fn

    def on_fit_start(self):
        self.num_bad_epochs = 0

    def training_step(self, batch, batch_idx):

        x, y = batch
        mask = generate_uniform_mask(len(x), self.mask_size).to(x.device)


        x_masked = self.mask_layer(x, mask)
        pred = self.model(x_masked)
        loss = self.loss_fn(pred, y)
        self.log('Loss Train', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mask = generate_uniform_mask(len(x), self.mask_size).to(x.device)
        x_masked = self.mask_layer(x, mask)
        pred = self.model(x_masked)
        loss = self.loss_fn(pred, y)
        metric = self.val_loss_fn(pred, y)
        self.log('Loss Val', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Perf Val', metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=self.factor, patience=self.patience,
            min_lr=self.min_lr, verbose=True)
        return {
            'optimizer': opt,
            'lr_scheduler': scheduler,
            'monitor': 'Loss Val'
        }
