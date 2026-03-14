import argparse
import torch
import os

from configs.config import Config
from utils.tokenizer import CoupletTokenizer
from utils.data_analyzer import analyze_dataset
from utils.evaluator import Evaluator
from dataset.dataset import create_dataloaders
from models.seq2seq import Encoder, Decoder, Seq2Seq
from train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='对联生成模型训练')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'analyze'],
                        help='运行模式: train (训练), evaluate (评估), analyze (数据分析)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型路径 (评估模式需要)')
    args = parser.parse_args()

    # 加载配置
    config = Config()
    print(f"使用设备: {config.device}")

    if args.mode == 'analyze':
        # 数据分析模式
        print("运行数据分析模式...")
        stats = analyze_dataset(config.data_path)
        print("\n分析完成！")
        return

    # 初始化分词器
    tokenizer = CoupletTokenizer(config)

    if args.mode == 'train':
        # 训练模式
        print("运行训练模式...")

        # 构建词表
        print("构建词表...")
        tokenizer.build_vocab(config.data_path)
        config.vocab_size = len(tokenizer.vocab)

        # 创建数据加载器
        print("创建数据加载器...")
        train_loader, val_loader = create_dataloaders(config, tokenizer)

        # 创建模型
        print("创建模型...")
        encoder = Encoder(config.vocab_size, config.embed_size,
                          config.hidden_size, config.num_layers, config.dropout)
        decoder = Decoder(config.vocab_size, config.embed_size,
                          config.hidden_size, config.num_layers, config.dropout)
        model = Seq2Seq(encoder, decoder, config).to(config.device)

        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数量: {total_params:,}")

        # 创建训练器
        trainer = Trainer(model, train_loader, val_loader, config)

        # 开始训练
        trainer.train()

    elif args.mode == 'evaluate':
        # 评估模式
        if args.model_path is None:
            print("错误：评估模式需要指定模型路径 (--model_path)")
            return

        print("运行评估模式...")

        # 构建词表
        tokenizer.build_vocab(config.data_path)
        config.vocab_size = len(tokenizer.vocab)

        # 创建模型
        encoder = Encoder(config.vocab_size, config.embed_size,
                          config.hidden_size, config.num_layers, config.dropout)
        decoder = Decoder(config.vocab_size, config.embed_size,
                          config.hidden_size, config.num_layers, config.dropout)
        model = Seq2Seq(encoder, decoder, config).to(config.device)

        # 加载模型
        trainer = Trainer(model, None, None, config)
        trainer.load_model(args.model_path)

        # 创建评估器
        evaluator = Evaluator(model, tokenizer, config)

        # 测试用例
        test_pairs = [
            ("春回大地", "福满人间"),
            ("风声雨声读书声", "家事国事天下事"),
            ("丹枫江冷人初去", "绿柳堤新燕复来"),
            ("屋后千年树", "门前万顷荷"),
            ("闲来野钓人稀处", "兴起高歌酒醉中"),
        ]

        # 评估
        results, avg_bleu = evaluator.evaluate_generation(test_pairs)

        print(f"\n平均BLEU分数: {avg_bleu:.4f}")
        evaluator.print_examples(results)


if __name__ == '__main__':
    main()