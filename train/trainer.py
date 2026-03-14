import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import os
from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD token
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # 修复：移除 verbose 参数
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

        # TensorBoard
        self.writer = SummaryWriter(config.log_dir)

        # 训练记录
        self.best_val_loss = float('inf')
        self.patience = 5
        self.patience_counter = 0
        self.current_epoch = 0

        print(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            src = batch['input'].to(self.config.device)
            trg = batch['target'].to(self.config.device)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(src, trg, self.config.teacher_forcing_ratio)

            # 计算损失
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)  # 忽略SOS
            trg = trg[:, 1:].reshape(-1)  # 忽略SOS

            loss = self.criterion(output, trg)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()

            total_loss += loss.item()

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 记录到TensorBoard
            if batch_idx % 100 == 0:
                global_step = len(self.train_loader) * self.current_epoch + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating')
            for batch in pbar:
                src = batch['input'].to(self.config.device)
                trg = batch['target'].to(self.config.device)

                # 前向传播
                output = self.model(src, trg, 0)  # validation时不用teacher forcing

                # 计算损失
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)

                loss = self.criterion(output, trg)
                total_loss += loss.item()

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        print("开始训练...")
        print(f"设备: {self.config.device}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print(f"训练集batches: {len(self.train_loader)}")
        print(f"验证集batches: {len(self.val_loader)}")
        print("=" * 50)

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print("-" * 50)

            # 训练
            train_loss = self.train_epoch()

            # 验证
            val_loss = self.validate()

            # 调整学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录
            self.writer.add_scalar('Train/Epoch_Loss', train_loss, epoch)
            self.writer.add_scalar('Val/Epoch_Loss', val_loss, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)

            print(f"\nTrain Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model(self.config.best_model_path)
                print(f"✓ 保存最佳模型，验证损失: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"验证损失未改善 {self.patience_counter}/{self.patience}")

            # 早停
            if self.patience_counter >= self.patience:
                print(f"早停：验证损失连续{self.patience}轮没有改善")
                break

            # 每轮都保存最新模型
            self.save_model(self.config.last_model_path)

        self.writer.close()
        print("\n训练完成！")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, path)

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.current_epoch = checkpoint['epoch']
        print(f"模型加载成功！")
        print(f"当前轮次: {self.current_epoch}")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")