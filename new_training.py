# 训练命令
## 训练受试者 S02，使用GPU 1，只跑fold 1
# python new_training.py --subject_num 0 --gpu_num 0 --fold_num 1

'''Import libraires'''
import os, yaml
from datetime import datetime

import pandas as pd
from easydict import EasyDict
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# 添加NumPy兼容性修复
import numpy as np

try:
    # 尝试修复NumPy的clip问题
    original_clip = np.clip


    def fixed_clip(a, a_min, a_max, out=None, **kwargs):
        if out is not None and out.dtype != np.float64:
            out = None  # 避免dtype不匹配
        return original_clip(a, a_min, a_max, out=out, **kwargs)


    np.clip = fixed_clip
except Exception as e:
    print(f"NumPy clip修复失败: {e}")

# 现在导入其他库
import braindecode.preprocessing.preprocess

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from dataloader.bci_compet import get_dataset
from model.litmodel import get_litmodel
from utils.setup_utils import (
    get_device,
    get_log_name,
)
from utils.training_utils import get_callbacks

'''Argparse'''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--subject_num', type=int, default=17)
parser.add_argument('--fold_num', type=int, default=0)
parser.add_argument('--gpu_num', type=str, default='0')
parser.add_argument('--config_name', type=str, default='bcicompet2a_config')
aargs = parser.parse_args()

# GPU_check - 只保留基本信息
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
else:
    print("ℹ️  使用CPU进行训练")
print("=" * 50)

# Config setting
with open('D:/python_project/ADFCNN-main/configs/bcicompet2a_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    args = EasyDict(config)

#### Set SEED ####
seed_everything(args.SEED)

#### Set Device ####
# 根据CUDA可用性自动选择设备
use_gpu = torch.cuda.is_available() and aargs.gpu_num
if use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = aargs.gpu_num
    args['device'] = get_device(aargs.gpu_num)
    cudnn.benchmark = True
    cudnn.fastest = True
    cudnn.deterministic = True
    print(f"✅ 使用GPU {aargs.gpu_num} 进行训练")
else:
    args['device'] = torch.device('cpu')
    print("ℹ️  使用CPU进行训练")

#### Set Log ####
args['current_time'] = datetime.now().strftime('%Y%m%d')
args['LOG_NAME'] = get_log_name(args)

#### Update configs ####
args.lr = float(args.lr)
if args.downsampling != 0: args['sampling_rate'] = args.downsampling

# 创建图表保存目录
plot_dir = os.path.join(args.LOG_PATH, 'training_plots')
os.makedirs(plot_dir, exist_ok=True)

# 存储所有结果
all_results = []


def generate_plot_from_model_history(model, subject_num, fold_num, plot_dir):
    """直接从模型训练历史生成图表（包含Kappa）"""
    try:
        if not hasattr(model, 'training_history'):
            print("❌ 模型没有training_history属性")
            return None

        train_loss = model.training_history['train_loss']
        train_acc = model.training_history['train_acc']
        val_loss = model.training_history['val_loss']
        val_acc = model.training_history['val_acc']

        # 新增Kappa数据
        train_kappa = model.training_history.get('train_kappa', [])
        val_kappa = model.training_history.get('val_kappa', [])

        print(
            f"📊 数据长度 - train_loss: {len(train_loss)}, val_loss: {len(val_loss)}, train_acc: {len(train_acc)}, val_acc: {len(val_acc)}")
        if train_kappa:
            print(f"📊 Kappa长度 - train_kappa: {len(train_kappa)}, val_kappa: {len(val_kappa)}")

        if not train_loss:
            print("⚠️ 训练历史为空")
            return None

        # 创建三个单独的图表：准确率、损失、Kappa
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        fig3, ax3 = plt.subplots(figsize=(10, 6))

        # 确定共同的数据长度（取最小值）
        min_length = min(len(train_loss), len(val_loss) if val_loss else len(train_loss))

        # 如果验证数据为空，使用训练数据长度
        if not val_loss:
            min_length = len(train_loss)

        epochs = range(1, min_length + 1)

        # 图表1: 准确率曲线
        if len(train_acc) >= min_length:
            ax1.plot(epochs, train_acc[:min_length], 'b-', label='Train Accuracy', linewidth=2, marker='o',
                     markersize=3)

        if val_acc and len(val_acc) >= min_length:
            ax1.plot(epochs, val_acc[:min_length], 'r-', label='Validation Accuracy', linewidth=2, marker='s',
                     markersize=3)
            print(f"✅ 绘制验证准确率曲线，数据点: {min_length}")
        elif val_acc:
            print(f"⚠️ 验证准确率数据不足: {len(val_acc)} < {min_length}")

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Subject {subject_num:02d} Fold {fold_num} - Training vs Validation Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)  # 准确率范围0-1

        # 图表2: 损失曲线
        if len(train_loss) >= min_length:
            ax2.plot(epochs, train_loss[:min_length], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)

        if val_loss and len(val_loss) >= min_length:
            ax2.plot(epochs, val_loss[:min_length], 'r-', label='Validation Loss', linewidth=2, marker='s',
                     markersize=3)
            print(f"✅ 绘制验证损失曲线，数据点: {min_length}")
        elif val_loss:
            print(f"⚠️ 验证损失数据不足: {len(val_loss)} < {min_length}")

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'Subject {subject_num:02d} Fold {fold_num} - Training vs Validation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 图表3: Kappa曲线
        if train_kappa and len(train_kappa) >= min_length:
            ax3.plot(epochs, train_kappa[:min_length], 'b-', label='Train Kappa', linewidth=2, marker='o', markersize=3)

        if val_kappa and len(val_kappa) >= min_length:
            ax3.plot(epochs, val_kappa[:min_length], 'r-', label='Validation Kappa', linewidth=2, marker='s',
                     markersize=3)
            print(f"✅ 绘制验证Kappa曲线，数据点: {min_length}")
        elif val_kappa:
            print(f"⚠️ 验证Kappa数据不足: {len(val_kappa)} < {min_length}")

        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Kappa')
        ax3.set_title(f'Subject {subject_num:02d} Fold {fold_num} - Training vs Validation Kappa')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.1, 1.0)  # Kappa范围-1到1，这里设置-0.1到1.0

        # 保存三个图表
        acc_plot_path = os.path.join(plot_dir, f'S{subject_num:02d}_fold{fold_num}_accuracy.png')
        loss_plot_path = os.path.join(plot_dir, f'S{subject_num:02d}_fold{fold_num}_loss.png')
        kappa_plot_path = os.path.join(plot_dir, f'S{subject_num:02d}_fold{fold_num}_kappa.png')

        fig1.tight_layout()
        fig1.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)

        fig2.tight_layout()
        fig2.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)

        fig3.tight_layout()
        fig3.savefig(kappa_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig3)

        print(f"📊 从模型历史生成训练曲线:")
        print(f"   - 准确率图表: {acc_plot_path}")
        print(f"   - 损失图表: {loss_plot_path}")
        print(f"   - Kappa图表: {kappa_plot_path}")

        return [acc_plot_path, loss_plot_path, kappa_plot_path]

    except Exception as e:
        print(f"❌ 从模型历史生成图表时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_training_curves_from_csv(csv_path, subject_num, fold_num, plot_dir):
    """从TensorBoard CSV日志绘制训练曲线（包含Kappa）"""
    try:
        if not os.path.exists(csv_path):
            print(f"⚠️ CSV文件不存在: {csv_path}")
            return None

        # 读取CSV文件
        df = pd.read_csv(csv_path)

        # 创建三个图表：准确率、损失、Kappa
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        fig3, ax3 = plt.subplots(figsize=(10, 6))

        plot_paths = []

        # 图表1: 准确率曲线
        if 'train_acc_epoch' in df.columns and 'val_acc_epoch' in df.columns:
            train_acc = df['train_acc_epoch'].dropna().values
            val_acc = df['val_acc_epoch'].dropna().values
            epochs = range(1, len(train_acc) + 1)

            ax1.plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=3)
            ax1.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.set_title(f'Subject {subject_num:02d} Fold {fold_num} - Training vs Validation Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)

            # 保存准确率图表
            acc_plot_path = os.path.join(plot_dir, f'S{subject_num:02d}_fold{fold_num}_accuracy.png')
            fig1.tight_layout()
            fig1.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            plot_paths.append(acc_plot_path)
            print(f"📊 准确率曲线已保存: {acc_plot_path}")

        # 图表2: 损失曲线
        if 'train_loss_epoch' in df.columns and 'val_loss_epoch' in df.columns:
            train_loss = df['train_loss_epoch'].dropna().values
            val_loss = df['val_loss_epoch'].dropna().values
            epochs = range(1, len(train_loss) + 1)

            ax2.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
            ax2.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title(f'Subject {subject_num:02d} Fold {fold_num} - Training vs Validation Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 保存损失图表
            loss_plot_path = os.path.join(plot_dir, f'S{subject_num:02d}_fold{fold_num}_loss.png')
            fig2.tight_layout()
            fig2.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            plot_paths.append(loss_plot_path)
            print(f"📊 损失曲线已保存: {loss_plot_path}")

        # 图表3: Kappa曲线
        if 'train_kappa_epoch' in df.columns and 'val_kappa_epoch' in df.columns:
            train_kappa = df['train_kappa_epoch'].dropna().values
            val_kappa = df['val_kappa_epoch'].dropna().values
            epochs = range(1, len(train_kappa) + 1)

            ax3.plot(epochs, train_kappa, 'b-', label='Train Kappa', linewidth=2, marker='o', markersize=3)
            ax3.plot(epochs, val_kappa, 'r-', label='Validation Kappa', linewidth=2, marker='s', markersize=3)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Kappa')
            ax3.set_title(f'Subject {subject_num:02d} Fold {fold_num} - Training vs Validation Kappa')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(-0.1, 1.0)

            # 保存Kappa图表
            kappa_plot_path = os.path.join(plot_dir, f'S{subject_num:02d}_fold{fold_num}_kappa.png')
            fig3.tight_layout()
            fig3.savefig(kappa_plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig3)
            plot_paths.append(kappa_plot_path)
            print(f"📊 Kappa曲线已保存: {kappa_plot_path}")
        else:
            print("ℹ️  CSV中未找到Kappa数据")

        return plot_paths

    except Exception as e:
        print(f"❌ 绘制训练曲线时出错: {e}")
        return None


def calculate_detailed_metrics(model, dataloader, device):
    """计算详细的评估指标（准确率、Kappa、混淆矩阵）"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch['data'], batch['label']
            x = x.type(torch.float).to(device)
            y = y.type(torch.long) if y.dim() == 1 else y
            y = y.to(device)

            outputs = model(x)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            # 处理one-hot编码的标签
            if y.dim() > 1:
                y_labels = torch.argmax(y, dim=1)
            else:
                y_labels = y
            all_targets.extend(y_labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 计算各项指标
    accuracy = (all_preds == all_targets).mean()
    kappa = cohen_kappa_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)

    return {
        'accuracy': accuracy,
        'kappa': kappa,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'targets': all_targets
    }


def interpret_kappa(kappa_value):
    """解读Kappa系数"""
    if kappa_value < 0:
        return "一致性比随机猜测还差"
    elif kappa_value <= 0.20:
        return "极低的一致性"
    elif kappa_value <= 0.40:
        return "一般的一致性"
    elif kappa_value <= 0.60:
        return "中等的一致性"
    elif kappa_value <= 0.80:
        return "高度的一致性"
    else:
        return "几乎完全一致"


'''Training'''
print(f"📋 总受试者数: {args.num_subjects}, 目标受试者: {aargs.subject_num}")
print(f"📋 总Fold数: {args.k_folds}, 目标Fold: {aargs.fold_num}")

for num_subject in range(args.num_subjects):
    # 修改选择逻辑：如果subject_num是默认值17，就训练所有受试者
    if aargs.subject_num != 17 and num_subject != aargs.subject_num:
        print(f"⏭️  跳过受试者 {num_subject}")
        continue

    args['target_subject'] = num_subject
    print(f"\n🚀 开始训练受试者 S{num_subject:02d}")

    try:
        dataset = get_dataset(aargs.config_name, args)

        # 检查数据集是否为空
        if len(dataset) == 0:
            print(f"❌ 受试者 S{num_subject:02d} 数据加载失败，跳过")
            continue

        print(f"📊 数据加载完成: {len(dataset)} 个样本")

    except Exception as e:
        print(f"❌ 受试者 S{num_subject:02d} 数据加载出错: {e}")
        print("⏭️  跳过该受试者")
        continue

    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.SEED)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)))):
        # 修改选择逻辑：如果fold_num是默认值0，就训练所有fold
        if aargs.fold_num != 0 and fold != aargs.fold_num:
            print(f"⏭️  跳过Fold {fold}")
            continue

        print(f"🔄 Fold {fold + 1}/{args.k_folds}")
        print(f"  训练样本: {len(train_idx)}, 验证样本: {len(val_idx)}")

        ### Set dataloader ###
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_dataloader = DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      pin_memory=False,
                                      num_workers=args.num_workers,
                                      sampler=train_subsampler)
        val_dataloader = DataLoader(dataset,
                                    batch_size=args.batch_size,
                                    pin_memory=False,
                                    num_workers=args.num_workers,
                                    sampler=val_subsampler)

        model = get_litmodel(args)
        logger = TensorBoardLogger(args.LOG_PATH,
                                   name=f'{args.LOG_NAME}/S{args.target_subject:02d}_fold{fold + 1}', version='')
        callbacks = get_callbacks(fold=fold, monitor='val_acc', args=args)


        # 自定义进度条，显示所有指标（包含Kappa）
        class DetailedProgressBar(TQDMProgressBar):
            def init_validation_tqdm(self):
                bar = super().init_validation_tqdm()
                bar.disable = True  # 禁用验证进度条
                return bar

            def get_metrics(self, trainer, model):
                items = super().get_metrics(trainer, model)
                # 移除v_num，添加训练和验证指标
                items.pop("v_num", None)

                # 添加训练指标
                if 'train_loss' in trainer.callback_metrics:
                    items['train_loss'] = f"{trainer.callback_metrics['train_loss']:.4f}"
                if 'train_acc' in trainer.callback_metrics:
                    items['train_acc'] = f"{trainer.callback_metrics['train_acc']:.4f}"
                if 'train_kappa' in trainer.callback_metrics:
                    items['train_kappa'] = f"{trainer.callback_metrics['train_kappa']:.4f}"

                # 添加验证指标
                if 'val_loss' in trainer.callback_metrics:
                    items['val_loss'] = f"{trainer.callback_metrics['val_loss']:.4f}"
                if 'val_acc' in trainer.callback_metrics:
                    items['val_acc'] = f"{trainer.callback_metrics['val_acc']:.4f}"
                if 'val_kappa' in trainer.callback_metrics:
                    items['val_kappa'] = f"{trainer.callback_metrics['val_kappa']:.4f}"

                return items


        callbacks.append(DetailedProgressBar())

        # 根据是否使用GPU配置Trainer
        if use_gpu:
            trainer = Trainer(
                enable_progress_bar=True,
                max_epochs=args.EPOCHS,
                accelerator="gpu",
                devices=[int(aargs.gpu_num)],
                callbacks=callbacks,
                default_root_dir=args.CKPT_PATH,
                logger=logger,
                log_every_n_steps=20,
            )
        else:
            trainer = Trainer(
                enable_progress_bar=True,
                max_epochs=args.EPOCHS,
                accelerator="cpu",
                devices=1,  # CPU模式下设置为1
                callbacks=callbacks,
                default_root_dir=args.CKPT_PATH,
                logger=logger,
                log_every_n_steps=20,
            )

        # 训练并获取最佳验证准确率
        trainer.fit(model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader)

        # 获取最佳验证准确率
        best_val_acc = trainer.checkpoint_callback.best_model_score.item()

        # 进行详细评估（计算Kappa等指标）
        print("📊 进行详细评估...")
        detailed_metrics = calculate_detailed_metrics(model, val_dataloader, args.device)

        print(f"✅ 详细评估结果:")
        print(f"   准确率 (Acc): {detailed_metrics['accuracy']:.4f}")
        print(f"   Kappa系数: {detailed_metrics['kappa']:.4f}")
        print(f"   Kappa解读: {interpret_kappa(detailed_metrics['kappa'])}")
        print(f"   混淆矩阵:\n{detailed_metrics['confusion_matrix']}")

        # 生成训练曲线
        try:
            # 修正TensorBoard日志路径查找
            log_dir = logger.log_dir
            print(f"📁 日志目录: {log_dir}")

            # 查找metrics.csv文件
            csv_path = None
            for root, dirs, files in os.walk(log_dir):
                if 'metrics.csv' in files:
                    csv_path = os.path.join(root, 'metrics.csv')
                    break

            plot_paths = []
            if csv_path and os.path.exists(csv_path):
                print(f"✅ 找到CSV文件: {csv_path}")
                plot_paths = plot_training_curves_from_csv(csv_path, num_subject, fold + 1, plot_dir)
            else:
                print(f"⚠️ 未找到CSV日志文件，搜索目录: {log_dir}")
                # 尝试直接从模型历史生成图表
                plot_paths = generate_plot_from_model_history(model, num_subject, fold + 1, plot_dir)

        except Exception as e:
            print(f"⚠️ 生成训练曲线时出错: {e}")
            plot_paths = []

        all_results.append({
            'subject': num_subject,
            'fold': fold,
            'val_acc': best_val_acc,  # 来自checkpoint的最佳准确率
            'val_acc_final': detailed_metrics['accuracy'],  # 最终验证准确率
            'val_kappa': detailed_metrics['kappa'],  # Kappa系数
            'confusion_matrix': detailed_metrics['confusion_matrix'],  # 混淆矩阵
            'plot_path': plot_paths
        })

        print(f"✅ 受试者 S{num_subject:02d} Fold {fold + 1} 完成")
        print(f"   最佳验证准确率: {best_val_acc:.4f}")
        print(f"   最终验证准确率: {detailed_metrics['accuracy']:.4f}")
        print(f"   Kappa系数: {detailed_metrics['kappa']:.4f}")

        torch.cuda.empty_cache()

# 显示汇总结果
print("\n" + "=" * 80)
print("🎯 训练结果汇总 (包含Kappa系数)")
print("=" * 80)

if all_results:
    for result in all_results:
        print(f"受试者 S{result['subject']:02d} Fold {result['fold'] + 1}:")
        print(f"  ✅ 最佳验证准确率: {result['val_acc']:.4f}")
        print(f"  ✅ 最终验证准确率: {result['val_acc_final']:.4f}")
        print(f"  📊 Kappa系数: {result['val_kappa']:.4f}")
        print(f"  📝 Kappa解读: {interpret_kappa(result['val_kappa'])}")
        if result['plot_path']:
            print(f"  📈 图表路径: {result['plot_path']}")
        print()

    # 计算平均指标
    avg_acc = sum(r['val_acc'] for r in all_results) / len(all_results)
    avg_acc_final = sum(r['val_acc_final'] for r in all_results) / len(all_results)
    avg_kappa = sum(r['val_kappa'] for r in all_results) / len(all_results)

    print(f"\n📈 平均指标:")
    print(f"  ✅ 平均最佳验证准确率: {avg_acc:.4f}")
    print(f"  ✅ 平均最终验证准确率: {avg_acc_final:.4f}")
    print(f"  📊 平均Kappa系数: {avg_kappa:.4f}")
    print(f"  📝 平均Kappa解读: {interpret_kappa(avg_kappa)}")

    # 保存详细结果到CSV文件
    results_csv_path = os.path.join(plot_dir, 'detailed_results.csv')
    results_df = pd.DataFrame([
        {
            'subject': r['subject'],
            'fold': r['fold'],
            'best_val_acc': r['val_acc'],
            'final_val_acc': r['val_acc_final'],
            'kappa': r['val_kappa'],
            'kappa_interpretation': interpret_kappa(r['val_kappa'])
        }
        for r in all_results
    ])
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n💾 详细结果已保存至: {results_csv_path}")

else:
    print("❌ 没有训练结果")

print("=" * 80)