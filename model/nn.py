import torch.nn as nn


class SimpleNN(nn.Module):
    """
    简单的神经网络模型，用于预测卫星太阳能电池板的方阵电流
    """

    def __init__(self, input_dim, hidden_dims=None, output_dim=1):
        """
        初始化神经网络模型

        参数:
        input_dim: 输入特征维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度（默认为1，即方阵电流）
        """
        super(SimpleNN, self).__init__()

        # 构建网络层
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        layers = []
        prev_dim = input_dim

        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # 添加输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        # 组合为Sequential模型
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播

        参数:
        x: 输入特征张量

        返回:
        方阵电流预测值
        """
        return self.network(x)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=1):
        super(LSTMModel, self).__init__()
        # 由于输入是2维的，我们需要将其扩展为3维
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 如果输入是2维的 (batch_size, input_dim)
        if x.dim() == 2:
            # 将其扩展为3维 (batch_size, 1, input_dim)
            x = x.unsqueeze(1)
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        output = self.fc(lstm_out[:, -1, :])
        return output



class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        residual = x
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ResNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, output_dim=1):
        super(ResNetModel, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.res_blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.fc(x)
        return x

class AttentionModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, output_dim=1):
        super(AttentionModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        # 全局平均池化
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


class AutoencoderModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, output_dim=1):
        super(AutoencoderModel, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # 解码器
        decoder_layers = []
        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # 预测头
        self.predictor = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        # 解码
        decoded = self.decoder(encoded)
        # 预测
        output = self.predictor(encoded)
        # return output, decoded  # 返回预测值和重构值
        return output

if __name__ == "__main__":
    pass