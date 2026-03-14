import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re


class CoupletDataset(Dataset):
    def __init__(self, data_path, tokenizer, config, split='train', train_ratio=0.9):
        """
        Args:
            data_path: 数据路径
            tokenizer: 分词器
            config: 配置
            split: 'train' 或 'val'
            train_ratio: 训练集比例
        """
        self.tokenizer = tokenizer
        self.config = config  # 确保先保存config
        self.max_length = config.max_length

        # 读取数据 - CSV格式，有表头
        df = pd.read_csv(data_path)

        # 获取列名
        if len(df.columns) >= 2:
            self.input_col = df.columns[0]  # 第一列是上联
            self.target_col = df.columns[1]  # 第二列是下联
            print(f"数据列: 上联='{self.input_col}', 下联='{self.target_col}'")
        else:
            raise ValueError("数据文件必须至少有两列")

        # 数据清洗
        df = self.clean_data(df)

        # 划分训练集和验证集
        n = len(df)
        n_train = int(n * train_ratio)

        if split == 'train':
            self.data = df.iloc[:n_train].reset_index(drop=True)
        else:
            self.data = df.iloc[n_train:].reset_index(drop=True)

        print(f"{split}集大小: {len(self.data)}")

    def clean_data(self, df):
        """清洗爬取的数据"""
        original_len = len(df)

        # 1. 去除空值
        df = df.dropna(subset=[self.input_col, self.target_col])
        print(f"   去除空值后: {len(df)}")

        # 2. 确保是字符串类型
        df[self.input_col] = df[self.input_col].astype(str)
        df[self.target_col] = df[self.target_col].astype(str)

        # 3. 去除可能包含网页标签的数据
        df = df[~df[self.input_col].str.contains('<[^>]+>', regex=True, na=False)]
        df = df[~df[self.target_col].str.contains('<[^>]+>', regex=True, na=False)]
        print(f"   去除HTML标签后: {len(df)}")

        # 4. 过滤掉空字符串
        df = df[df[self.input_col].str.len() > 0]
        df = df[df[self.target_col].str.len() > 0]
        print(f"   过滤空字符串后: {len(df)}")

        # 5. 计算长度
        df['input_len'] = df[self.input_col].apply(len)
        df['target_len'] = df[self.target_col].apply(len)

        # 6. 去除过长或过短的对联 - 使用self.config
        df = df[(df['input_len'] >= 2) & (df['input_len'] <= self.config.max_length)]
        df = df[(df['target_len'] >= 2) & (df['target_len'] <= self.config.max_length)]
        print(f"   过滤长度后: {len(df)}")

        # 7. 确保上下联长度相等
        df = df[df['input_len'] == df['target_len']]
        print(f"   确保长度相等后: {len(df)}")

        # 8. 去除重复数据
        df = df.drop_duplicates(subset=[self.input_col, self.target_col])
        print(f"   去重后: {len(df)}")

        # 删除临时列
        df = df.drop(['input_len', 'target_len'], axis=1)

        print(f"数据清洗完成: {original_len} -> {len(df)} (删除了 {original_len - len(df)} 条)")
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data.iloc[idx][self.input_col]
        target_text = self.data.iloc[idx][self.target_col]

        # 编码
        input_ids = self.tokenizer.encode(input_text, self.max_length)
        target_ids = self.tokenizer.encode(target_text, self.max_length)

        return {
            'input': torch.tensor(input_ids, dtype=torch.long),
            'target': torch.tensor(target_ids, dtype=torch.long),
            'input_text': input_text,
            'target_text': target_text
        }


def create_dataloaders(config, tokenizer):
    """创建数据加载器"""
    print("=" * 50)
    print("创建数据加载器")
    print("=" * 50)

    # 创建数据集
    print("\n加载训练集...")
    train_dataset = CoupletDataset(config.data_path, tokenizer, config, split='train')

    print("\n加载验证集...")
    val_dataset = CoupletDataset(config.data_path, tokenizer, config, split='val')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if config.device == 'cuda' else False,
        drop_last=True  # 丢弃最后一个不完整的batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.device == 'cuda' else False,
        drop_last=False
    )

    print(f"\n训练集batches: {len(train_loader)}")
    print(f"验证集batches: {len(val_loader)}")

    return train_loader, val_loader


# 测试代码
if __name__ == '__main__':
    # 测试数据集
    from configs.config import Config
    from utils.tokenizer import CoupletTokenizer

    config = Config()
    tokenizer = CoupletTokenizer(config)

    # 先构建词表
    tokenizer.build_vocab(config.data_path)

    # 测试数据集
    dataset = CoupletDataset(config.data_path, tokenizer, config, split='train')
    print(f"\n数据集大小: {len(dataset)}")

    # 测试一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n样本示例:")
        print(f"  上联: {sample['input_text']}")
        print(f"  下联: {sample['target_text']}")
        print(f"  输入ids形状: {sample['input'].shape}")
        print(f"  目标ids形状: {sample['target'].shape}")