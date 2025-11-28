import math

import torch
import torch.nn as nn

class HealthGenerator(nn.Module):
    """
    根据归一化后的 day index 生成健康向量 h_d。
    输入: day_ids (0 ~ num_days-1)
    输出: h_day: [B, health_dim]
    """
    def __init__(self, health_dim: int = 8, hidden_dim: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, health_dim),
        )

    def forward(self, day_ids: torch.Tensor, num_days: int) -> torch.Tensor:
        """
        day_ids: [B] int64
        num_days: 总天数，用于归一化
        """
        # 归一化到 [0, 1]
        t = day_ids.float() / (num_days - 1)  # [B]
        t = t.unsqueeze(-1)                   # [B, 1]
        h = self.mlp(t)                       # [B, health_dim]
        return h

    def all_health_vectors(self, num_days: int, device: torch.device) -> torch.Tensor:
        """
        返回所有天的健康向量，用于正则: [num_days, health_dim]
        """
        day_ids = torch.arange(num_days, device=device, dtype=torch.long)  # [D]
        return self.forward(day_ids, num_days)  # [D, health_dim]


class SolarHealthModel(nn.Module):
    def __init__(self,
                 num_days: int,
                 input_dim: int,
                 lstm_hidden_dim: int = 64,
                 lstm_layers: int = 2,
                 health_dim: int = 8,
                 mlp_hidden_dim: int = 64):
        super().__init__()

        self.num_days = num_days
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.health_dim = health_dim

        # 1) 段内环境编码器（LSTM）
        self.env_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        # 2) 日级健康生成网络（时间 -> 向量）
        self.health_gen = HealthGenerator(health_dim=health_dim, hidden_dim=32)

        # 3) Time-distributed MLP 解码器： [env, health] -> current
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_dim + health_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)  # 输出电流
        )

        self._init_parameters()

    def _init_parameters(self):
        # 可以加一些初始化（可选）
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        # health embedding 默认随机即可

    def forward(self, X, day_ids):
        """
        X: [B, T, F]
        day_ids: [B] int64
        返回:
            I_pred: [B, T, 1]
        """
        B, T, F = X.shape

        # 1) LSTM 编码环境
        env_seq, _ = self.env_encoder(X)  # [B, T, H]

        # 2) 通过时间生成健康向量序列和衰减因子
        h_day = self.health_gen(day_ids, self.num_days)  # [B, Hh]
        h_day_seq = h_day.unsqueeze(1).repeat(1, T, 1)  # [B, T, Hh]

        # 3) 拼接 + MLP 预测电流
        z = torch.cat([env_seq, h_day_seq], dim=-1)  # [B, T, H+health_dim]
        I_pred = self.mlp(z)  # [B, T, 1]

        return I_pred

    def all_health_vectors(self, device: torch.device):
        """
        返回所有天的健康向量: [num_days, health_dim]
        """
        return self.health_gen.all_health_vectors(self.num_days, device)


class SimpleSolarModel(nn.Module):
    def __init__(self,
                 num_days: int,
                 input_dim: int,
                 lstm_hidden_dim: int = 64,
                 lstm_layers: int = 2,
                 mlp_hidden_dim: int = 64):
        super().__init__()

        self.num_days = num_days
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        # 1) 段内环境编码器（LSTM）
        self.env_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        # 2) Time-distributed MLP 解码器： [env, time_info] -> current
        # 直接使用时间信息而不是健康向量
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_dim + 1, mlp_hidden_dim),  # +1 是时间信息
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)  # 输出电流
        )

        self._init_parameters()

    def _init_parameters(self):
        # 初始化MLP权重
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, X, day_ids):
        """
        X: [B, T, F]
        day_ids: [B] int64
        返回:
            I_pred: [B, T, 1]
        """
        B, T, F = X.shape

        # 1) LSTM 编码环境
        env_seq, _ = self.env_encoder(X)  # [B, T, H]

        # 2) 将时间信息归一化并扩展到序列长度
        # 归一化到 [0, 1]
        normalized_time = day_ids.float() / (self.num_days - 1)  # [B]
        time_seq = normalized_time.unsqueeze(1).unsqueeze(2).repeat(1, T, 1)  # [B, T, 1]

        # 3) 拼接环境编码和时间信息 + MLP 预测电流
        z = torch.cat([env_seq, time_seq], dim=-1)  # [B, T, H+1]
        I_pred = self.mlp(z)  # [B, T, 1]

        return I_pred

    def all_health_vectors(self, device: torch.device):
        """
        返回所有天的时间向量作为健康向量: [num_days, 1]
        为了与SolarHealthModel兼容而添加
        """
        # 生成归一化的时间向量作为健康向量，并转为倒序
        days = torch.arange(self.num_days - 1, -1, -1, device=device, dtype=torch.float)  # 倒序生成
        normalized_time = days / (self.num_days - 1)  # 归一化到 [0, 1]
        return normalized_time.unsqueeze(1)  # [num_days, 1]


class GRUSolarModel(nn.Module):
    def __init__(self,
                 num_days: int,
                 input_dim: int,
                 gru_hidden_dim: int = 64,
                 gru_layers: int = 2,
                 mlp_hidden_dim: int = 64):
        super().__init__()

        self.num_days = num_days
        self.input_dim = input_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_layers = gru_layers

        # 1) 段内环境编码器（GRU）
        self.env_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False
        )

        # 2) Time-distributed MLP 解码器： [env, time_info] -> current
        # 直接使用时间信息而不是健康向量
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden_dim + 1, mlp_hidden_dim),  # +1 是时间信息
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)  # 输出电流
        )

        self._init_parameters()

    def _init_parameters(self):
        # 初始化MLP权重
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, X, day_ids):
        """
        X: [B, T, F]
        day_ids: [B] int64
        返回:
            I_pred: [B, T, 1]
        """
        B, T, F = X.shape

        # 1) GRU 编码环境
        env_seq, _ = self.env_encoder(X)  # [B, T, H]

        # 2) 将时间信息归一化并扩展到序列长度
        # 归一化到 [0, 1]
        normalized_time = day_ids.float() / (self.num_days - 1)  # [B]
        time_seq = normalized_time.unsqueeze(1).unsqueeze(2).repeat(1, T, 1)  # [B, T, 1]

        # 3) 拼接环境编码和时间信息 + MLP 预测电流
        z = torch.cat([env_seq, time_seq], dim=-1)  # [B, T, H+1]
        I_pred = self.mlp(z)  # [B, T, 1]

        return I_pred

    def all_health_vectors(self, device: torch.device):
        """
        返回所有天的时间向量作为健康向量: [num_days, 1]
        为了与SimpleSolarModel兼容而添加
        """
        # 生成归一化的时间向量作为健康向量，并转为倒序
        days = torch.arange(self.num_days - 1, -1, -1, device=device, dtype=torch.float)  # 倒序生成
        normalized_time = days / (self.num_days - 1)  # 归一化到 [0, 1]
        return normalized_time.unsqueeze(1)  # [num_days, 1]


class CNNLSTMSolarModel(nn.Module):
    def __init__(self,
                 num_days: int,
                 input_dim: int,
                 cnn_channels: int = 32,
                 cnn_kernel_size: int = 3,
                 lstm_hidden_dim: int = 64,
                 lstm_layers: int = 2,
                 mlp_hidden_dim: int = 64):
        super().__init__()

        self.num_days = num_days
        self.input_dim = input_dim
        self.cnn_channels = cnn_channels
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        # 1) CNN特征提取器 - 提取局部时间模式
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
            nn.ReLU()
        )

        # 2) LSTM时间依赖建模
        self.temporal_model = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        # 3) MLP解码器：[lstm_features, time_info] -> current
        self.decoder = nn.Sequential(
            nn.Linear(lstm_hidden_dim + 1, mlp_hidden_dim),  # +1 是时间信息
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)  # 输出电流
        )

        self._init_parameters()

    def _init_parameters(self):
        # 初始化CNN权重
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # 初始化MLP权重
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, X, day_ids):
        """
        X: [B, T, F]
        day_ids: [B] int64
        返回:
            I_pred: [B, T, 1]
        """
        B, T, F = X.shape

        # 1) CNN特征提取 (需要转置因为Conv1d期望的输入是[B, C, T])
        X_transposed = X.transpose(1, 2)  # [B, F, T]
        cnn_features = self.feature_extractor(X_transposed)  # [B, C, T]
        cnn_features = cnn_features.transpose(1, 2)  # [B, T, C]

        # 2) LSTM时间建模
        temporal_features, _ = self.temporal_model(cnn_features)  # [B, T, H]

        # 3) 将时间信息归一化并扩展到序列长度
        # 归一化到 [0, 1]
        normalized_time = day_ids.float() / (self.num_days - 1)  # [B]
        time_seq = normalized_time.unsqueeze(1).unsqueeze(2).repeat(1, T, 1)  # [B, T, 1]

        # 4) 拼接时间特征和时间信息 + MLP 预测电流
        z = torch.cat([temporal_features, time_seq], dim=-1)  # [B, T, H+1]
        I_pred = self.decoder(z)  # [B, T, 1]

        return I_pred

    def all_health_vectors(self, device: torch.device):
        """
        返回所有天的时间向量作为健康向量: [num_days, 1]
        为了与SimpleSolarModel兼容而添加
        """
        # 生成归一化的时间向量作为健康向量，并转为倒序
        days = torch.arange(self.num_days - 1, -1, -1, device=device, dtype=torch.float)  # 倒序生成
        normalized_time = days / (self.num_days - 1)  # 归一化到 [0, 1]
        return normalized_time.unsqueeze(1)  # [num_days, 1]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerSolarModel(nn.Module):
    def __init__(self,
                 num_days: int,
                 input_dim: int,
                 d_model: int = 64,
                 nhead: int = 8,
                 num_layers: int = 2,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1):
        super().__init__()

        self.num_days = num_days
        self.input_dim = input_dim
        self.d_model = d_model

        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)

        # 时间信息嵌入层
        self.time_embedding = nn.Linear(1, d_model)

        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 输出层：预测电流
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self._init_parameters()

    def _init_parameters(self):
        # 初始化权重
        nn.init.xavier_uniform_(self.input_embedding.weight)
        nn.init.zeros_(self.input_embedding.bias)
        nn.init.xavier_uniform_(self.time_embedding.weight)
        nn.init.zeros_(self.time_embedding.bias)
        nn.init.xavier_uniform_(self.output_layer[0].weight)
        nn.init.zeros_(self.output_layer[0].bias)
        nn.init.xavier_uniform_(self.output_layer[3].weight)
        nn.init.zeros_(self.output_layer[3].bias)

    def forward(self, X, day_ids):
        """
        X: [B, T, F] - 环境特征
        day_ids: [B] - 天数索引
        返回:
            I_pred: [B, T, 1] - 预测电流
        """
        B, T, F = X.shape

        # 1) 环境特征嵌入
        env_embedding = self.input_embedding(X)  # [B, T, d_model]

        # 2) 添加位置编码
        env_embedding = self.pos_encoder(env_embedding.transpose(0, 1)).transpose(0, 1)  # [B, T, d_model]

        # 3) 时间信息嵌入
        # 归一化到 [0, 1]
        normalized_time = day_ids.float() / (self.num_days - 1)  # [B]
        time_info = normalized_time.unsqueeze(1).unsqueeze(2).repeat(1, T, 1)  # [B, T, 1]
        time_embedding = self.time_embedding(time_info)  # [B, T, d_model]

        # 4) 融合环境特征和时间信息
        fused_features = env_embedding + time_embedding  # [B, T, d_model]

        # 5) Transformer编码
        transformer_output = self.transformer_encoder(fused_features)  # [B, T, d_model]

        # 6) 输出层预测电流
        I_pred = self.output_layer(transformer_output)  # [B, T, 1]

        return I_pred

    def all_health_vectors(self, device: torch.device):
        """
        返回所有天的时间向量作为健康向量: [num_days, 1]
        为了与SimpleSolarModel兼容而添加
        """
        # 生成归一化的时间向量作为健康向量，并转为倒序
        days = torch.arange(self.num_days - 1, -1, -1, device=device, dtype=torch.float)  # 倒序生成
        normalized_time = days / (self.num_days - 1)  # 归一化到 [0, 1]
        return normalized_time.unsqueeze(1)  # [num_days, 1]