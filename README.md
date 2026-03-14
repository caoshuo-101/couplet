对联生成模型 (Couplet Generation)
基于Seq2Seq+Attention机制的对联生成模型，使用PyTorch实现。

项目简介
本项目使用深度学习技术，训练了一个能够根据上联自动生成下联的Seq2Seq模型。模型采用双向LSTM作为编码器，带有注意力机制的解码器，能够在保持对联格律的同时生成语义相关的下联。

模型特点
Seq2Seq架构：编码器-解码器结构

注意力机制：让模型关注上联的不同部分

双向LSTM：更好地捕捉上下文信息

支持变长序列：可处理不同长度的对联

数据集
使用 wb14123/couplet 数据集：

总数据量：774,491 条对联

训练集：690,367 条

验证集：76,708 条

主要长度分布：7字联(46.21%)、5字联(9.86%)、12字联(11.85%)

词表大小：9,127 个唯一字符

项目结构
text
couplet/
├── configs/
│   └── config.py          # 配置文件
├── dataset/
│   ├── dataset.py         # 数据集类
│   └── create_subset.py   # 数据子集创建工具
├── models/
│   └── seq2seq.py         # Seq2Seq模型定义
├── train/
│   └── trainer.py         # 训练器
├── utils/
│   ├── tokenizer.py       # 分词器
│   ├── data_analyzer.py   # 数据分析工具
│   └── evaluator.py       # 评估工具
├── test/
│   └── test_model.py      # 模型测试
├── interactive_test.py    # 交互式测试脚本
├── main.py                # 主程序入口
├── requirements.txt       # 依赖包
└── README.md              # 项目说明