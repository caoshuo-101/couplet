import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        # 将双向LSTM的输出投影到hidden_size
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))
        # embedded shape: (batch_size, seq_len, embed_size)

        encoder_states, (hidden, cell) = self.lstm(embedded)
        # encoder_states: (batch_size, seq_len, hidden_size * 2)
        # hidden: (num_layers * 2, batch_size, hidden_size)
        # cell: (num_layers * 2, batch_size, hidden_size)

        # 重塑hidden和cell
        hidden = hidden.reshape(self.num_layers, 2, x.size(0), self.hidden_size)
        hidden = hidden.mean(dim=1)  # 在双向维度上取平均

        cell = cell.reshape(self.num_layers, 2, x.size(0), self.hidden_size)
        cell = cell.mean(dim=1)

        return encoder_states, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_states):
        # hidden: (batch_size, hidden_size)
        # encoder_states: (batch_size, src_len, hidden_size * 2)

        batch_size = encoder_states.shape[0]
        src_len = encoder_states.shape[1]

        # 重复hidden以匹配encoder_states的序列长度
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # 拼接hidden和encoder_states
        energy = torch.cat((hidden, encoder_states), dim=2)

        # 计算注意力权重
        attention = torch.tanh(self.attention(energy))
        attention = self.v(attention).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size * 2, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size + hidden_size * 2 + embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell, encoder_states):
        # x: (batch_size, 1)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)
        # encoder_states: (batch_size, src_len, hidden_size * 2)

        # 嵌入
        x = x.unsqueeze(1)  # (batch_size, 1)
        embedded = self.dropout(self.embedding(x))  # (batch_size, 1, embed_size)

        # 计算注意力权重
        hidden_last = hidden[-1]  # (batch_size, hidden_size)
        attention_weights = self.attention(hidden_last, encoder_states)  # (batch_size, src_len)
        attention_weights = attention_weights.unsqueeze(1)  # (batch_size, 1, src_len)

        # 计算上下文向量
        context = torch.bmm(attention_weights, encoder_states)  # (batch_size, 1, hidden_size * 2)

        # 拼接嵌入和上下文
        lstm_input = torch.cat((embedded, context), dim=2)  # (batch_size, 1, embed_size + hidden_size * 2)

        # LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: (batch_size, 1, hidden_size)

        # 拼接output, context, embedded用于预测
        output = output.squeeze(1)  # (batch_size, hidden_size)
        context = context.squeeze(1)  # (batch_size, hidden_size * 2)
        embedded = embedded.squeeze(1)  # (batch_size, embed_size)

        prediction_input = torch.cat((output, context, embedded),
                                     dim=1)  # (batch_size, hidden_size + hidden_size*2 + embed_size)
        prediction = self.fc(prediction_input)  # (batch_size, vocab_size)

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, config):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (batch_size, src_len)
        # trg: (batch_size, trg_len)

        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size

        # 存储输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.config.device)

        # 编码
        encoder_states, hidden, cell = self.encoder(src)

        # 第一个输入是SOS token
        input = trg[:, 0]

        for t in range(1, trg_len):
            # 解码一步
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_states)
            outputs[:, t, :] = output

            # 决定是否使用teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # 获取下一个输入
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs

    def generate(self, src, max_length):
        """生成下联（用于推理）"""
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]

            # 编码
            encoder_states, hidden, cell = self.encoder(src)

            # 直接使用固定的ID值（根据tokenizer的定义）
            # 0: PAD, 1: UNK, 2: SOS, 3: EOS
            sos_id = 2
            eos_id = 3

            # 第一个输入是SOS token
            input = torch.tensor([sos_id] * batch_size).to(self.config.device)

            # 存储生成的token
            generated = []

            for _ in range(max_length):
                # 解码一步
                output, hidden, cell = self.decoder(input, hidden, cell, encoder_states)

                # 获取预测的token
                predicted = output.argmax(1)
                generated.append(predicted)

                # 下一个输入
                input = predicted

                # 如果所有序列都生成了EOS，可以提前停止
                if (predicted == eos_id).all():
                    break

            # 堆叠生成的tokens
            generated = torch.stack(generated, dim=1)

        return generated