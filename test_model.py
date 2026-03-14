import torch
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from utils.tokenizer import CoupletTokenizer
from models.seq2seq import Encoder, Decoder, Seq2Seq


def load_model(model_path, config, tokenizer):
    """加载训练好的模型"""
    print(f"正在加载模型: {model_path}")

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        # 尝试查找模型文件
        possible_paths = [
            './checkpoints/best_model.pth',
            '/root/autodl-tmp/couplet/checkpoints/best_model.pth',
            'checkpoints/best_model.pth',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                print(f"找到模型文件: {path}")
                model_path = path
                break
        else:
            raise FileNotFoundError("找不到模型文件，请检查checkpoints目录")

    # 创建模型
    encoder = Encoder(config.vocab_size, config.embed_size,
                      config.hidden_size, config.num_layers, config.dropout)
    decoder = Decoder(config.vocab_size, config.embed_size,
                      config.hidden_size, config.num_layers, config.dropout)
    model = Seq2Seq(encoder, decoder, config).to(config.device)

    # 加载权重 - 使用weights_only=False因为我们的checkpoint包含Config对象
    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型加载成功！")
    if 'best_val_loss' in checkpoint:
        print(f"最佳验证损失: {checkpoint['best_val_loss']:.4f}")
    if 'epoch' in checkpoint:
        print(f"训练轮次: {checkpoint['epoch']}")

    return model


def test_couplet(model, tokenizer, input_text, config):
    """测试单个对联"""
    # 编码输入
    input_ids = tokenizer.encode(input_text, config.max_length)
    src = torch.tensor([input_ids]).to(config.device)

    # 生成
    with torch.no_grad():
        generated_ids = model.generate(src, config.max_length)
        # 取第一个batch的结果
        generated = tokenizer.decode(generated_ids[0].cpu().numpy())

    return generated


def main():
    print("=" * 60)
    print("对联生成测试")
    print("=" * 60)

    config = Config()
    tokenizer = CoupletTokenizer(config)

    # 加载词表
    print("\n1. 加载词表...")
    tokenizer.build_vocab(config.data_path)
    config.vocab_size = len(tokenizer.vocab)
    print(f"词表大小: {config.vocab_size}")

    # 加载模型
    print("\n2. 加载模型...")
    model = load_model(config.best_model_path, config, tokenizer)

    # 测试用例
    test_cases = [
        "春回大地",
        "福满人间",
        "风声雨声读书声",
        "家事国事天下事",
        "晚风摇树树还挺",
        "丹枫江冷人初去",
        "屋后千年树",
        "门前万顷荷",
        "闲来野钓人稀处",
        "兴起高歌酒醉中",
        "历史名城，九水回澜，飞扬吴楚三千韵",
    ]

    print("\n" + "=" * 60)
    print("3. 生成结果")
    print("=" * 60)

    for input_text in test_cases:
        generated = test_couplet(model, tokenizer, input_text, config)
        print(f"\n上联: {input_text}")
        print(f"下联: {generated}")
        print("-" * 40)


if __name__ == '__main__':
    main()