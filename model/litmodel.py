import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy
from utils.training_utils import get_criterion, get_scheduler, get_optimizer

from model.ADFCNN import get_model


# from model.FBCNet import get_model
# from model.ShallowConvNet import get_model
# from model.DeepConvNet import get_model
# from model.EEGNet import get_model
# from model.IFNetV2 import get_model
# from model.TSMANet import get_model

class LitModel(LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.model = get_model(args)
        self.criterion = get_criterion()
        self.args = args

        # 存储训练历史用于绘图
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            self.sample_batch = batch['data'].type(torch.float)

        inputs = batch['data'].type(torch.float)
        labels = batch['label'].type(torch.long) if batch['label'].dim() == 1 else batch['label']

        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        acc = accuracy(preds, labels if labels.dim() == 1 else torch.argmax(labels, dim=1))

        # 记录训练指标 - 现在在进度条中显示
        self.log('train_loss', loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,  # 改为True在进度条显示
                 logger=True,
                 sync_dist=True)
        self.log('train_acc', acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,  # 改为True在进度条显示
                 logger=True,
                 sync_dist=True)
        return {'loss': loss, 'train_acc': acc}

    def on_train_epoch_end(self):
        # 记录训练历史
        # if 'train_loss_epoch' in self.trainer.callback_metrics:
        #     self.training_history['train_loss'].append(
        #         self.trainer.callback_metrics['train_loss_epoch'].item()
        #     )
        # if 'train_acc_epoch' in self.trainer.callback_metrics:
        #     self.training_history['train_acc'].append(
        #         self.trainer.callback_metrics['train_acc_epoch'].item()
        #     )
        # if 'train_loss' in self.trainer.callback_metrics:
        #     self.training_history['train_loss'].append(
        #         self.trainer.callback_metrics['train_loss'].item()
        #     )
        # if 'train_acc' in self.trainer.callback_metrics:
        #     self.training_history['train_acc'].append(
        #         self.trainer.callback_metrics['train_acc'].item()
        #     )
        #
        # # 添加计算图
        # if self.current_epoch == 0 and hasattr(self, 'sample_batch'):
        #     if hasattr(self, 'logger') and self.logger is not None:
        #         try:
        #             self.logger.experiment.add_graph(self.model, self.sample_batch)
        #         except Exception as e:
        #             print(f"无法添加计算图: {e}")

        # 记录训练历史
        callback_metrics = self.trainer.callback_metrics

        if 'train_loss' in callback_metrics:
            train_loss_value = callback_metrics['train_loss'].item()
            self.training_history['train_loss'].append(train_loss_value)

        if 'train_acc' in callback_metrics:
            train_acc_value = callback_metrics['train_acc'].item()
            self.training_history['train_acc'].append(train_acc_value)



    def evaluate(self, batch, stage=None):
        inputs = batch['data'].type(torch.float)
        labels = batch['label'].type(torch.long) if batch['label'].dim() == 1 else batch['label']

        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy(preds, labels if labels.dim() == 1 else torch.argmax(labels, dim=1))

        if stage:
            self.log(f'{stage}_loss', loss,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=False,  # 在进度条显示
                     logger=True,
                     sync_dist=True)
            self.log(f'{stage}_acc', acc,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=False,  # 在进度条显示
                     logger=True,
                     sync_dist=True)

        return loss, acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, stage='val')
        return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        # 记录验证历史
        # if 'val_loss' in self.trainer.callback_metrics:
        #     self.training_history['val_loss'].append(
        #         self.trainer.callback_metrics['val_loss'].item()
        #     )
        # if 'val_acc' in self.trainer.callback_metrics:
        #     self.training_history['val_acc'].append(
        #         self.trainer.callback_metrics['val_acc'].item()
        #     )

        # if 'val_loss' in self.trainer.callback_metrics:
        #     self.training_history['val_loss'].append(
        #         self.trainer.callback_metrics['val_loss'].item()
        #     )
        # if 'val_acc' in self.trainer.callback_metrics:
        #     self.training_history['val_acc'].append(
        #         self.trainer.callback_metrics['val_acc'].item()
        #     )

        # 记录验证历史
        callback_metrics = self.trainer.callback_metrics

        if 'val_loss' in callback_metrics:
            val_loss_value = callback_metrics['val_loss'].item()
            self.training_history['val_loss'].append(val_loss_value)

        if 'val_acc' in callback_metrics:
            val_acc_value = callback_metrics['val_acc'].item()
            self.training_history['val_acc'].append(val_acc_value)



    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage='test')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = batch['data'].type(torch.float)
        return self(inputs)

    def configure_optimizers(self):
        self.__init_optimizers()
        return {'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler_config}

    def __init_optimizers(self):
        self.optimizer = get_optimizer(self, self.args)
        self.lr_scheduler = get_scheduler(self.optimizer, self.args)
        self.lr_scheduler_config = {
            'scheduler': self.lr_scheduler,
            'interval': 'epoch'
        }

    def plot_training_curves(self, save_path=None):
        """绘制训练曲线"""
        if len(self.training_history['train_loss']) == 0:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        epochs = range(1, len(self.training_history['train_loss']) + 1)

        # 绘制损失曲线
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if len(self.training_history['val_loss']) == len(epochs):
            ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 绘制准确率曲线
        ax2.plot(epochs, self.training_history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        if len(self.training_history['val_acc']) == len(epochs):
            ax2.plot(epochs, self.training_history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig


def get_litmodel(args):
    model = LitModel(args)
    return model