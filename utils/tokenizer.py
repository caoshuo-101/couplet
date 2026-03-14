import pandas as pd
from collections import Counter
import torch
import re


class CoupletTokenizer:
    def __init__(self, config):
        self.config = config
        self.punct_map = config.PUNCT_MAP
        self.vocab = {}
        self.idx_to_word = {}

    def tokenize(self, text):
        """将文本切分为token序列"""
        # 确保text是字符串
        text = str(text) if text is not None else ''
        tokens = []
        for char in text:
            if char in self.punct_map:
                tokens.append(self.punct_map[char])
            else:
                tokens.append(char)
        return tokens

    def detokenize(self, tokens):
        """将token序列恢复为文本"""
        # 创建反向映射
        reverse_map = {v: k for k, v in self.punct_map.items()}

        text = ''
        for token in tokens:
            if token in reverse_map:
                text += reverse_map[token]
            elif token not in [self.config.PAD_TOKEN, self.config.SOS_TOKEN,
                               self.config.EOS_TOKEN, self.config.UNK_TOKEN]:
                text += token
        return text

    def clean_text(self, text):
        """清洗单个文本"""
        # 处理非字符串类型
        if pd.isna(text) or text is None:
            return ''

        # 转换为字符串
        text = str(text).strip()

        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)

        # 去除多余空格
        text = text.strip()

        return text

    def build_vocab(self, data_path):
        """从数据构建词表"""
        print("读取数据文件...")
        # 读取数据 - 指定所有列都是字符串类型
        df = pd.read_csv(data_path, dtype=str)
        print(f"原始数据量: {len(df)}")

        # 确保列存在
        if len(df.columns) < 2:
            raise ValueError("数据文件必须至少有两列")

        input_col = df.columns[0]
        target_col = df.columns[1]
        print(f"使用列: '{input_col}' 和 '{target_col}'")

        # 清洗数据
        print("清洗数据...")
        df[input_col] = df[input_col].apply(self.clean_text)
        df[target_col] = df[target_col].apply(self.clean_text)

        # 过滤空字符串
        df = df[df[input_col].str.len() > 0]
        df = df[df[target_col].str.len() > 0]
        print(f"清洗后数据量: {len(df)}")

        # 统计词频
        print("统计词频...")
        counter = Counter()

        for text in df[input_col]:
            tokens = self.tokenize(text)
            counter.update(tokens)

        for text in df[target_col]:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # 构建词表（包含特殊token）
        special_tokens = [
            self.config.PAD_TOKEN,
            self.config.UNK_TOKEN,
            self.config.SOS_TOKEN,
            self.config.EOS_TOKEN,
            self.config.COMMA_TOKEN,
            self.config.PERIOD_TOKEN,
            self.config.QUESTION_TOKEN,
            self.config.EXCLAM_TOKEN,
            '<PAUSE>', '<SEMICOLON>', '<COLON>', '<QUOTE>', '<SQUOTE>',
            '<LPAREN>', '<RPAREN>'
        ]

        self.vocab = {token: idx for idx, token in enumerate(special_tokens)}

        # 添加高频词
        print("构建词表...")
        for word, count in counter.most_common():
            if word not in self.vocab:
                if len(self.vocab) < self.config.vocab_size:
                    self.vocab[word] = len(self.vocab)

        # 创建idx到word的映射
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}

        print(f"词表构建完成！")
        print(f"词表大小: {len(self.vocab)}")
        print(f"总字符数: {sum(counter.values()):,}")
        print(f"唯一字符数: {len(counter)}")

        return self.vocab

    def encode(self, text, max_length):
        """将文本转换为id序列"""
        # 清洗文本
        text = self.clean_text(text)
        tokens = self.tokenize(text)

        # 添加SOS和EOS
        ids = [self.vocab.get(self.config.SOS_TOKEN, 0)]

        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab.get(self.config.UNK_TOKEN, 1))

        ids.append(self.vocab.get(self.config.EOS_TOKEN, 2))

        # 填充或截断
        if len(ids) < max_length:
            ids += [self.vocab.get(self.config.PAD_TOKEN, 0)] * (max_length - len(ids))
        else:
            ids = ids[:max_length]
            ids[-1] = self.vocab.get(self.config.EOS_TOKEN, 2)

        return ids

    def decode(self, ids):
        """将id序列转换为文本"""
        tokens = []
        for idx in ids:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                if word == self.config.EOS_TOKEN:
                    break
                if word not in [self.config.PAD_TOKEN, self.config.SOS_TOKEN, self.config.UNK_TOKEN]:
                    tokens.append(word)
        return self.detokenize(tokens)


# 测试代码
if __name__ == '__main__':
    from configs.config import Config

    config = Config()
    tokenizer = CoupletTokenizer(config)

    # 测试构建词表
    vocab = tokenizer.build_vocab(config.data_path)

    # 测试编码解码
    test_text = "晚风摇树树还挺"
    encoded = tokenizer.encode(test_text, config.max_length)
    decoded = tokenizer.decode(encoded)

    print(f"\n测试文本: {test_text}")
    print(f"编码后长度: {len(encoded)}")
    print(f"解码后: {decoded}")