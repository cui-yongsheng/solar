from model.segment_nn import HealthGenerator
from model.physics_model import SegmentPhysicsModel
import torch.nn as nn
from torchdiffeq import odeint
import torch

class PhysicsInformedSolarHealthModel(nn.Module):
    """
    物理信息增强的分段太阳能电池板健康模型
    """

    def __init__(self,
                 num_days: int,
                 input_dim: int,
                 lstm_hidden_dim: int = 64,
                 lstm_layers: int = 2,
                 health_dim: int = 8,
                 feat_hidden_dim: int = 64):  # 物理模型需要的输入维度
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
        
        # 3) 物理模型
        self.physics_model = SegmentPhysicsModel()

        fusion_dim = lstm_hidden_dim + health_dim + 1  # env + health + I_phys

        # 4) Time-distributed MLP 解码器： [env, health, physics] -> current
        self.fusion  = nn.Sequential(
            nn.Linear(fusion_dim, feat_hidden_dim),
            nn.ReLU(),
        )

        # 5) 预测衰减量
        self.head_decay = nn.Linear(health_dim, 1)

        # 6) 辅任务 残差
        self.head_residual = nn.Linear(feat_hidden_dim, 1)

    def forward(self, X, day_ids, raw_features=None):
        """
        X: [B, T, F]
        day_ids: [B] int64
        raw_features: [B, T, F_raw] 用于物理计算的原始特征
        返回:
            I_pred: [B, T, 1]
        """
        B, T, F = X.shape

        # 1) LSTM 编码环境
        env_seq, _ = self.env_encoder(X)  # [B, T, H]

        # 2) 健康向量
        h_day = self.health_gen(day_ids, self.num_days)  # [B, Hh]
        h_day_seq = h_day.unsqueeze(1).repeat(1, T, 1)  # [B, T, Hh]
        raw_D_day = self.head_decay(h_day)  # [B,1]
        D_day = torch.sigmoid(raw_D_day)  # (0,1) 每天一个因子
        D = D_day.unsqueeze(1).expand(-1, T, -1)  # [B,T,1]

        # 3) 物理模型标准输出
        I_phys = self.physics_model.compute_current(raw_features)  # [B, T, 1]
        I_base = D * I_phys  # [B,T,1] = 衰减后的物理输出

        # 4) 融合特征
        z = torch.cat([env_seq, h_day_seq, I_phys], dim=-1)
        feat = self.fusion(z)  # [B,T,Hf]

        # 6) 残差
        r_pred = self.head_residual(feat)  # [B,T,1]

        I_pred = I_base + r_pred  # 最终预测

        return {
            "I_pred": I_pred,
            "I_phys": I_phys,  # 标准输出
            "I_base": I_base,  # 衰减物理输出
            "D": D,
            "r_pred": r_pred,
            "h_day": h_day,
        }


    def all_health_vectors(self, device: torch.device):
        """
        返回所有天的健康向量: [num_days, health_dim]
        """
        return self.health_gen.all_health_vectors(self.num_days, device)

    def get_degradation_curve(self, device: torch.device):
        """
        获取所有天数的衰减因子(D)曲线: [num_days]
        
        参数:
        device: 计算设备
        
        返回:
        D_curve: 每日衰减因子，形状为 [num_days]
        """
        # 获取所有天数的健康向量
        all_health_vectors = self.all_health_vectors(device)  # [num_days, health_dim]
        
        # 通过head_decay层计算原始衰减值
        raw_D = self.head_decay(all_health_vectors)  # [num_days, 1]
        
        # 应用sigmoid函数确保衰减因子在(0,1)范围内
        D_curve = torch.sigmoid(raw_D).squeeze(-1)  # [num_days]
        
        return D_curve


class PhysicsInformedSolarHealthModelNoDegradation(nn.Module):
    """
    物理信息增强的分段太阳能电池板健康模型（无衰减因子版本）
    """

    def __init__(self,
                 num_days: int,
                 input_dim: int,
                 lstm_hidden_dim: int = 64,
                 lstm_layers: int = 2,
                 health_dim: int = 8,
                 feat_hidden_dim: int = 64):  # 物理模型需要的输入维度
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
        
        # 3) 物理模型
        self.physics_model = SegmentPhysicsModel()

        fusion_dim = lstm_hidden_dim + health_dim + 1  # env + health + I_phys

        # 4) Time-distributed MLP 解码器： [env, health, physics] -> current
        self.fusion  = nn.Sequential(
            nn.Linear(fusion_dim, feat_hidden_dim),
            nn.ReLU(),
        )

        # 5) 辅助任务 残差预测
        self.head_residual = nn.Sequential(
            nn.Linear(feat_hidden_dim, 1),
            nn.ReLU(),
        )
        

    def forward(self, X, day_ids, raw_features=None):
        """
        X: [B, T, F]
        day_ids: [B] int64
        raw_features: [B, T, F_raw] 用于物理计算的原始特征
        返回:
            I_pred: [B, T, 1]
        """
        B, T, F = X.shape

        # 1) LSTM 编码环境
        env_seq, _ = self.env_encoder(X)  # [B, T, H]

        # 2) 健康向量
        h_day = self.health_gen(day_ids, self.num_days)  # [B, Hh]
        h_day_seq = h_day.unsqueeze(1).repeat(1, T, 1)  # [B, T, Hh]

        # 3) 物理模型标准输出
        I_phys = self.physics_model.compute_current(raw_features)  # [B, T, 1]

        # 4) 融合特征
        z = torch.cat([env_seq, h_day_seq, I_phys], dim=-1)
        feat = self.fusion(z)  # [B,T,Hf]

        # 5) 残差
        r_pred = self.head_residual(feat)  # [B,T,1]

        # 最终预测 = 物理模型输出 + 残差
        I_pred = I_phys + r_pred

        return {
            "I_pred": I_pred,
            "I_phys": I_phys,  # 标准输出
            "r_pred": r_pred,
            "h_day": h_day,
        }


    def all_health_vectors(self, device: torch.device):
        """
        返回所有天的健康向量: [num_days, health_dim]
        """
        return self.health_gen.all_health_vectors(self.num_days, device)


class TransformerNeuralODE(nn.Module):
    """
    使用Transformer替代LSTM的NeuralODE模型
    Transformer在处理序列数据时有更好的并行性和全局注意力机制，
    可以更好地捕捉序列开始部分的特征
    """

    def __init__(self,
                 num_days: int,
                 input_dim: int,
                 d_model=64,
                 nhead=8,
                 num_layers=2,
                 health_dim=8,
                 feat_hidden_dim=64,
                 use_adjoint=False):
        super().__init__()

        self.num_days = num_days
        self.health_dim = health_dim
        self.use_adjoint = use_adjoint
        self.d_model = d_model

        # ------------------------- Transformer encoder -------------------------
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*2,
            dropout=0.1,
            batch_first=True
        )
        self.env_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ------------------------- Positional encoding -------------------------
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model) * 0.02)

        # ------------------------- Initial health h(0) -------------------------
        self.initial_health = nn.Parameter(torch.randn(health_dim) * 0.1)

        # ------------------------- ODE function f(h,t) -------------------------
        self.ode_func = nn.Sequential(
            nn.Linear(health_dim + 1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, health_dim)
        )

        # ------------------------- Physics Model -------------------------
        from model.physics_model import SegmentPhysicsModel
        self.physics_model = SegmentPhysicsModel()

        # ------------------------- Fusion network -------------------------
        fusion_dim = d_model + health_dim + 1
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, feat_hidden_dim),
            nn.ReLU(),
        )

        # ------------------------- Heads -------------------------
        self.head_decay_weight = nn.Parameter(torch.randn(health_dim) * 0.01)
        self.head_decay_bias = nn.Parameter(torch.zeros(1))

        self.head_residual = nn.Sequential(
            nn.Linear(feat_hidden_dim, 1),
        )

        # ------------------------- ODE Cache -------------------------
        self.register_buffer("health_table", None)   # [num_days+1, H]
        self._ode_cached_for_params = None           # store param hash

    # ============================================================
    #                     ODE Derivative (Batch)
    # ============================================================
    def _ode_derivative(self, t, h):
        """
        h: [H] or [B,H]   (torchdiffeq can pass tensor)
        t: scalar (0-D)
        return: same shape as h
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)  # [1,H]

        t_exp = t.expand(h.size(0), 1)
        inp = torch.cat([h, t_exp], dim=-1)
        dh = self.ode_func(inp)
        return dh

    # ============================================================
    #             ODE Full Trajectory Precomputation
    # ============================================================
    def _params_hash(self):
        """Compute a hash of relevant parameters to detect changes."""
        vec = torch.cat([
            self.initial_health.detach().flatten(),
            self.head_decay_weight.detach().flatten(),
            self.head_decay_bias.detach().flatten(),
        ])
        return hash(vec.cpu().numpy().tobytes())

    def _compute_full_ode_trajectory(self, device):
        """Compute h(t) for all t=0..num_days only once."""
        times = torch.linspace(0, 1, self.num_days + 1, device=device)

        init_h = self.initial_health.to(device)

        method = "dopri5" if not self.use_adjoint else "adjoint"

        traj = odeint(
            func=self._ode_derivative,
            y0=init_h,
            t=times,
            method="dopri5" if not self.use_adjoint else "adjoint"
        )  # [T, H]

        self.health_table = traj.detach()  # no grad needed
        self._ode_cached_for_params = self._params_hash()

    # ============================================================
    #                 Health Query by day_ids
    # ============================================================
    def _get_health(self, day_ids, device):
        """
        day_ids: [B] int
        return: [B, health_dim]
        """
        need_refresh = (self.health_table is None) or \
                       (self._ode_cached_for_params != self._params_hash())

        if need_refresh:
            self._compute_full_ode_trajectory(device)

        return self.health_table[day_ids]  # [B,H]

    # ============================================================
    #                         Forward
    # ============================================================
    def forward(self, X, day_ids, raw_features):
        """
        X: [B,T,F]
        day_ids: [B]   sample time index
        raw_features: [B,T,F_raw]
        """
        B, T, _ = X.shape
        device = X.device

        # ---- 1. Transformer encoding ----
        # 投影到模型维度
        X_proj = self.input_projection(X)
        
        # 添加位置编码
        pos_enc = self.positional_encoding[:T, :].unsqueeze(0).expand(B, T, -1)
        X_with_pos = X_proj + pos_enc
        
        # Transformer编码
        env_seq = self.env_encoder(X_with_pos)

        # ---- 2. Get health state (fast lookup) ----
        health = self._get_health(day_ids, device)   # [B,H]
        health_seq = health.unsqueeze(1).expand(-1, T, -1)

        # ---- 3. Decay factor ----
        w = self.head_decay_weight
        raw_D = (health_seq * w.view(1,1,-1)).sum(-1, keepdim=True) + self.head_decay_bias
        D = torch.sigmoid(raw_D)   # [B,T,1]

        # ---- 4. Physics model ----
        I_phys = self.physics_model.compute_current(raw_features)
        I_base = D * I_phys

        # ---- 5. Fusion ----
        z = torch.cat([env_seq, health_seq, I_phys], dim=-1)
        feat = self.fusion(z)

        # ---- 6. Residual ----
        r = self.head_residual(feat)
        I_pred = I_base + r

        return {
            "I_pred": I_pred,
            "I_phys": I_phys,
            "I_base": I_base,
            "D": D,
            "r_pred": r,
            "health_seq": health_seq,
        }

    # ============================================================
    #                 Degradation curve
    # ============================================================
    def get_degradation_curve(self, device):
        if self.health_table is None:
            self._compute_full_ode_trajectory(device)

        health_traj = self.health_table   # [T,H]
        w = self.head_decay_weight.to(device)

        raw_D = health_traj @ w + self.head_decay_bias
        return torch.sigmoid(raw_D)
    

class NeuralODEHealthModel(nn.Module):
    """
    Highly optimized version:
    - Precompute full health trajectory [num_days+1, H]
    - Batch ODE function (supports [B,H])
    - Cache full ODE trajectory and refresh when parameters change
    """

    def __init__(self,
                 num_days: int,
                 input_dim: int,
                 lstm_hidden_dim=64,
                 lstm_layers=2,
                 health_dim=8,
                 feat_hidden_dim=64,
                 use_adjoint=False):
        super().__init__()

        self.num_days = num_days
        self.health_dim = health_dim
        self.use_adjoint = use_adjoint

        # ------------------------- LSTM encoder -------------------------
        self.env_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # ------------------------- Initial health h(0) -------------------------
        self.initial_health = nn.Parameter(torch.randn(health_dim) * 0.1)

        # ------------------------- ODE function f(h,t) -------------------------
        self.ode_func = nn.Sequential(
            nn.Linear(health_dim + 1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, health_dim)
        )

        # ------------------------- Physics Model -------------------------
        from model.physics_model import SegmentPhysicsModel
        self.physics_model = SegmentPhysicsModel()

        # ------------------------- Fusion network -------------------------
        fusion_dim = lstm_hidden_dim + health_dim + 1
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, feat_hidden_dim),
            nn.ReLU(),
            nn.Linear(feat_hidden_dim, feat_hidden_dim),
            nn.ReLU(),
        )

        # ------------------------- Heads -------------------------
        self.head_decay_weight = nn.Parameter(torch.randn(health_dim) * 0.01)
        self.head_decay_bias = nn.Parameter(torch.zeros(1))

        self.head_residual = nn.Sequential(
            nn.Linear(feat_hidden_dim, feat_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_hidden_dim // 2, 1)
        )

        # ------------------------- ODE Cache -------------------------
        self.register_buffer("health_table", None)   # [num_days+1, H]
        self._ode_cached_for_params = None           # store param hash

    # ============================================================
    #                     ODE Derivative (Batch)
    # ============================================================
    def _ode_derivative(self, t, h):
        """
        h: [H] or [B,H]   (torchdiffeq can pass tensor)
        t: scalar (0-D)
        return: same shape as h
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)  # [1,H]

        t_exp = t.expand(h.size(0), 1)
        inp = torch.cat([h, t_exp], dim=-1)
        dh = self.ode_func(inp)
        return dh

    # ============================================================
    #             ODE Full Trajectory Precomputation
    # ============================================================
    def _params_hash(self):
        """Compute a hash of relevant parameters to detect changes."""
        vec = torch.cat([
            self.initial_health.detach().flatten(),
            self.head_decay_weight.detach().flatten(),
            self.head_decay_bias.detach().flatten(),
        ])
        return hash(vec.cpu().numpy().tobytes())

    def _compute_full_ode_trajectory(self, device):
        """Compute h(t) for all t=0..num_days only once."""
        times = torch.linspace(0, 1, self.num_days + 1, device=device)

        init_h = self.initial_health.to(device)

        method = "dopri5" if not self.use_adjoint else "adjoint"

        traj = odeint(
            func=self._ode_derivative,
            y0=init_h,
            t=times,
            method="dopri5" if not self.use_adjoint else "adjoint"
        )  # [T, H]

        self.health_table = traj.detach()  # no grad needed
        self._ode_cached_for_params = self._params_hash()

    # ============================================================
    #                 Health Query by day_ids
    # ============================================================
    def _get_health(self, day_ids, device):
        """
        day_ids: [B] int
        return: [B, health_dim]
        """
        need_refresh = (self.health_table is None) or \
                       (self._ode_cached_for_params != self._params_hash())

        if need_refresh:
            self._compute_full_ode_trajectory(device)

        return self.health_table[day_ids]  # [B,H]

    # ============================================================
    #                         Forward
    # ============================================================
    def forward(self, X, day_ids, raw_features):
        """
        X: [B,T,F]
        day_ids: [B]   sample time index
        raw_features: [B,T,F_raw]
        """
        B, T, _ = X.shape
        device = X.device

        # ---- 1. LSTM encoding ----
        env_seq, _ = self.env_encoder(X)

        # ---- 2. Get health state (fast lookup) ----
        health = self._get_health(day_ids, device)   # [B,H]
        health_seq = health.unsqueeze(1).expand(-1, T, -1)

        # ---- 3. Decay factor ----
        w = self.head_decay_weight
        raw_D = (health_seq * w.view(1,1,-1)).sum(-1, keepdim=True) + self.head_decay_bias
        D = torch.sigmoid(raw_D)   # [B,T,1]

        # ---- 4. Physics model ----
        I_phys = self.physics_model.compute_current(raw_features)
        I_base = D * I_phys

        # ---- 5. Fusion ----
        z = torch.cat([env_seq, health_seq, I_phys], dim=-1)
        feat = self.fusion(z)

        # ---- 6. Residual ----
        r = self.head_residual(feat)
        I_pred = I_base + r

        return {
            "I_pred": I_pred,
            "I_phys": I_phys,
            "I_base": I_base,
            "D": D,
            "r_pred": r,
            "health_seq": health_seq,
        }

    # ============================================================
    #                 Degradation curve
    # ============================================================
    def get_degradation_curve(self, device):
        if self.health_table is None:
            self._compute_full_ode_trajectory(device)

        health_traj = self.health_table   # [T,H]
        w = self.head_decay_weight.to(device)

        raw_D = health_traj @ w + self.head_decay_bias
        return torch.sigmoid(raw_D)
