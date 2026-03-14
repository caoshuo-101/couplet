import torch
import os


class Config:
    # 数据路径
    data_path = '/root/autodl-tmp/datasets/data_30percent.csv'
    save_dir = './checkpoints'
    log_dir = './logs'

    # 创建必要的目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 模型参数 - 根据数据统计调整
    vocab_size = 9127  # 直接使用实际字符数
    embed_size = 256  # 词向量维度
    hidden_size = 512  # 隐藏层维度
    num_layers = 2  # LSTM层数
    dropout = 0.3  # Dropout比例

    # 训练参数
    batch_size = 128  # 77万数据，可以大一点
    epochs = 30  # 先跑30轮看看效果
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 优化参数
    teacher_forcing_ratio = 0.5  # 教师强制比例
    max_length = 32  # 根据数据调整为32（最大长度）
    min_freq = 1  # 因为词表已经确定，设为1

    # 保存路径
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    last_model_path = os.path.join(save_dir, 'last_model.pth')

    # 数据格式
    csv_sep = ','
    csv_has_header = True

    # 特殊token
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    SOS_TOKEN = '<SOS>'
    EOS_TOKEN = '<EOS>'
    COMMA_TOKEN = '<COMMA>'
    PERIOD_TOKEN = '<PERIOD>'
    QUESTION_TOKEN = '<QUESTION>'
    EXCLAM_TOKEN = '<EXCLAM>'

    # 标点符号映射
    PUNCT_MAP = {
        '，': COMMA_TOKEN,
        '。': PERIOD_TOKEN,
        '？': QUESTION_TOKEN,
        '！': EXCLAM_TOKEN,
        '、': '<PAUSE>',
        '；': '<SEMICOLON>',
        '：': '<COLON>',
        '“': '<QUOTE>',
        '”': '<QUOTE>',
        '‘': '<SQUOTE>',
        '’': '<SQUOTE>',
        '（': '<LPAREN>',
        '）': '<RPAREN>',
    }