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
            '../checkpoints/best_model.pth'
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

    # 加载权重
    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ 模型加载成功！")
    if 'best_val_loss' in checkpoint:
        print(f"  最佳验证损失: {checkpoint['best_val_loss']:.4f}")
    if 'epoch' in checkpoint:
        print(f"  训练轮次: {checkpoint['epoch']}")

    return model


def generate_couplet(model, tokenizer, input_text, config):
    """生成下联"""
    # 编码输入
    input_ids = tokenizer.encode(input_text, config.max_length)
    src = torch.tensor([input_ids]).to(config.device)

    # 生成
    with torch.no_grad():
        generated_ids = model.generate(src, config.max_length)
        generated = tokenizer.decode(generated_ids[0].cpu().numpy())

    return generated


def print_help():
    """打印帮助信息"""
    print("\n" + "=" * 60)
    print("使用说明:")
    print("  - 直接输入上联，程序会生成对应的下联")
    print("  - 输入 'quit' 或 'exit' 退出程序")
    print("  - 输入 'help' 显示此帮助")
    print("  - 输入 'examples' 查看示例")
    print("=" * 60)


def print_examples():
    """打印示例"""
    examples = [
        "春回大地",
        "风声雨声读书声",
        "晚风摇树树还挺",
        "丹枫江冷人初去",
        "屋后千年树",
        "闲来野钓人稀处",
        "历史名城，九水回澜，飞扬吴楚三千韵",
    ]

    print("\n示例上联:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. {ex}")


def main():
    # 清屏
    os.system('clear' if os.name == 'posix' else 'cls')

    print("=" * 60)
    print("对联生成系统 - 交互式测试")
    print("=" * 60)

    # 初始化配置
    config = Config()
    tokenizer = CoupletTokenizer(config)

    # 加载词表
    print("\n正在初始化...")
    tokenizer.build_vocab(config.data_path)
    config.vocab_size = len(tokenizer.vocab)
    print(f"✓ 词表加载完成，大小: {config.vocab_size}")

    # 加载模型
    try:
        model = load_model(config.best_model_path, config, tokenizer)
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return

    print_help()
    print_examples()

    # 交互循环
    while True:
        print("\n" + "-" * 60)
        user_input = input("请输入上联 (或输入命令): ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("感谢使用，再见！")
            break

        elif user_input.lower() in ['help', 'h']:
            print_help()
            continue

        elif user_input.lower() in ['examples', 'ex']:
            print_examples()
            continue

        elif user_input == "":
            continue

        # 生成下联
        print(f"\n正在生成...")
        generated = generate_couplet(model, tokenizer, user_input, config)

        # 输出结果
        print("\n" + "=" * 60)
        print(f"上联: {user_input}")
        print(f"下联: {generated}")

        # 简单的质量检查
        if len(generated) == len(user_input):
            print(f"✓ 字数匹配")
        else:
            print(f"⚠ 字数不匹配: 上联 {len(user_input)}字, 下联 {len(generated)}字")

        print("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断，再见！")
    except Exception as e:
        print(f"\n程序出错: {e}")
        import traceback

        traceback.print_exc()