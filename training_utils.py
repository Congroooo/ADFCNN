import torch
import torch.optim as optim

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.backends.cudnn import deterministic



def get_criterion():
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    # criterion = torch.nn.CrossEntropyLoss()
    return criterion


def get_optimizer(model, args):
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,betas=(0.9, 0.999), eps=1e-08)
    
    return optimizer

#余弦退火学习率调度器
# def get_scheduler(optimizer, args):
#
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.EPOCHS, eta_min=0)
#
#     return scheduler

#余弦退火+热身学习率调度器
def get_scheduler(optimizer, args):
    # 确保参数存在
    if not hasattr(args, 'warmup_epochs'):
        args.warmup_epochs = max(5, min(20, int(args.EPOCHS * 0.1)))#热身epochs数取决于数据量
        # args.warmup_epochs = 10  # 5个epoch的热身
    if not hasattr(args, 'min_lr'):
        args.min_lr = 1e-6  # 最小学习率

    # 热身阶段
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_epochs
    )

    # 余弦退火阶段
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.EPOCHS - args.warmup_epochs,
        eta_min=args.min_lr
    )

    # 组合调度器
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs]
    )

    return scheduler

def get_checkpoint_callback(fold: int, monitor: str, args):
    if monitor == 'val_acc':
        return ModelCheckpoint(monitor=monitor,
                                dirpath=f'{args.CKPT_PATH}/{args.LOG_NAME}/fold_{fold + 1}',
                                filename=f'{args.task}_S{args.target_subject:02d}_' + '{epoch:02d}-{val_acc:.3f}',
                                save_top_k=3,
                                mode='max')
    elif monitor == 'val_loss':
        return ModelCheckpoint(monitor=monitor,
                                dirpath=f'{args.CKPT_PATH}/{args.LOG_NAME}/fold_{fold + 1}',
                                filename=f'{args.task}_S{args.target_subject:02d}_' + '{epoch:02d}-{val_loss:.3f}',
                                save_top_k=3,
                                mode='min')
    else:
        return None


def get_callbacks(fold: int, monitor: str, args):
    checkpoint_callback = get_checkpoint_callback(fold=fold, monitor=monitor, args=args)
    lr_logger=LearningRateMonitor(logging_interval='epoch')
    return [checkpoint_callback, lr_logger]

