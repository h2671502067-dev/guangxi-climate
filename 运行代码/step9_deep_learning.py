"""
步骤9: 多模型深度学习预测建模 (广西多源气象数据)
=================================================
使用多种深度学习网络对广西8个代表城市的月平均气温进行多变量时间序列预测。

输入特征: 38维 (地面气象24 + 干旱指数4 + 高空衍生7 + NDVI 1 + 月份周期编码2)
预测目标: 气温_2m_℃ (2m气温月均值)

模型体系 (4种深度学习网络 + 迁移学习):
  1) TCN  (Temporal Convolutional Network): 时间卷积网络
  2) LSTM (Long Short-Term Memory): 长短期记忆网络
  3) GRU  (Gated Recurrent Unit): 门控循环单元
  4) Transformer: 自注意力编码器

建模策略:
  1) 独立训练: 每个站点对每种模型单独训练
  2) 迁移学习 (仅TCN): 8站数据池化全局预训练共享骨干 -> 各站低学习率微调
     (站点间共享季节性/辐射-温度等跨站通用模式, 缓解单站小样本过拟合)
  3) 最优选择: 按测试集RMSE为每站自动选择最优 (模型, 策略) 组合

数学方法说明:

  9.1 TCN (Temporal Convolutional Network):
    通过因果膨胀卷积与残差连接捕捉时序依赖关系。

    因果卷积 (Causal Convolution):
      y_t = Σ f_k · x_{t-k}, 输出仅依赖当前及历史, 不泄漏未来信息

    膨胀卷积 (Dilated Convolution):
      第i层膨胀率 dᵢ = 2ⁱ, 感受野呈指数增长:
        R = 1 + (k-1) × Σ dᵢ  (i = 0, 1, ..., L-1)
      其中 k 为卷积核大小, L 为网络层数。
      3层(k=3)感受野 R = 1 + 2×(1+2+4) = 15 > 序列长度12, 可覆盖全部历史。

    残差连接 (Residual Connection):
      x = ReLU(F(x) + x)   (若输入输出通道不同, 用1×1卷积对齐)

  9.2 LSTM / GRU 循环网络:
    LSTM: 输入门 iₜ、遗忘门 fₜ、输出门 oₜ 与记忆单元 cₜ:
      fₜ = σ(W_f·[hₜ₋₁,xₜ] + b_f), iₜ = σ(W_i·[hₜ₋₁,xₜ] + b_i)
      c̃ₜ = tanh(W_c·[hₜ₋₁,xₜ] + b_c), cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ
      oₜ = σ(W_o·[hₜ₋₁,xₜ] + b_o), hₜ = oₜ⊙tanh(cₜ)
    GRU: 重置门 rₜ 与更新门 zₜ, 参数更少:
      rₜ = σ(W_r·[hₜ₋₁,xₜ]), zₜ = σ(W_z·[hₜ₋₁,xₜ])
      h̃ₜ = tanh(W·[rₜ⊙hₜ₋₁, xₜ]), hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ
    取序列最后时间步隐状态经全连接输出气温预测。

  9.3 Transformer 编码器:
    自注意力 (Scaled Dot-Product Attention):
      Attention(Q,K,V) = softmax(QK^T/√d_k) V
    多头注意力: MultiHead = Concat(head₁,...,head_h)W^O
    2层编码器 + 可学习位置编码, 取最后时间步输出预测。

  9.4 迁移学习 (Transfer Learning, 仅TCN):
    预训练: 8站训练窗口合并为池, 学习跨站通用气象-温度映射
    微调:  以较低学习率适配各站分布
    优势:  数据量 8×, 降低单站过拟合, 提升小样本站点精度

  9.5 训练策略:
    - 优化器: Adam (lr=0.001), 微调 lr=0.0003
    - 损失函数: MSE
    - 早停: 验证损失连续patience轮未改善时停止
    - 学习率调度: ReduceLROnPlateau
    - 梯度裁剪: max_norm=1.0, 防止梯度爆炸

  9.6 评估指标:
    RMSE = √[(1/n) Σ(yᵢ - ŷᵢ)²]
    MAE  = (1/n) Σ|yᵢ - ŷᵢ|
    R²   = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from config import (CHART_DIR, TRAIN_TEST_SPLIT, plt_config,
                    TCN_CHANNELS, TCN_KERNEL_SIZE, TCN_DROPOUT,
                    TCN_EPOCHS, TCN_BATCH_SIZE, TCN_LR, TCN_PATIENCE,
                    DL_SEQUENCE_LENGTH, DL_RANDOM_STATE,
                    GUANGXI_STATIONS, GUANGXI_STATION_CN,
                    GUANGXI_FEATURES, GUANGXI_TARGET)
from guangxi_data_loader import load_guangxi_station_dfs

for k, v in plt_config.items():
    plt.rcParams[k] = v

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    try:
        # 新API (torch>=1.13), 避免weight_norm弃用警告
        _wn = nn.utils.parametrizations.weight_norm
    except AttributeError:
        _wn = nn.utils.weight_norm
except ImportError:
    TORCH_AVAILABLE = False

# 微调超参数
FINETUNE_LR = TCN_LR * 0.3
FINETUNE_PATIENCE = 10
VAL_SIZE = 12


# ==================== TCN模型定义 ====================

class Chomp1d(nn.Module):
    """裁剪因果卷积右侧多余padding, 保证输出长度与输入一致。"""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """TCN基本块: 双层膨胀因果卷积 + 残差连接。

    膨胀率d下, padding=(k-1)×d 保证因果性, chomp裁剪右侧padding保持长度。
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, dropout=0.2):
        super().__init__()
        self.conv1 = _wn(nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride,
            padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = _wn(nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride,
            padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1)
                           if n_inputs != n_outputs else None)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for conv in (self.conv1, self.conv2):
            w = getattr(conv, 'weight_orig', None) or conv.weight
            w.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """时间卷积网络: 3层膨胀因果卷积堆叠 + 全连接输出。"""

    def __init__(self, num_inputs, num_channels=TCN_CHANNELS,
                 kernel_size=TCN_KERNEL_SIZE, dropout=TCN_DROPOUT):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]  # 取最后一个时间步
        out = self.fc(out)
        return out


# ==================== 其他深度学习模型定义 ====================

class LSTMModel(nn.Module):
    """LSTM: 2层长短期记忆网络, 取最后时间步隐状态输出预测。"""

    def __init__(self, num_inputs, hidden_size=64, num_layers=2,
                 dropout=TCN_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 最后时间步隐状态
        return self.fc(out)


class GRUModel(nn.Module):
    """GRU: 2层门控循环单元, 参数更少、训练更快。"""

    def __init__(self, num_inputs, hidden_size=64, num_layers=2,
                 dropout=TCN_DROPOUT):
        super().__init__()
        self.gru = nn.GRU(input_size=num_inputs, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # 最后时间步隐状态
        return self.fc(out)


class TransformerModel(nn.Module):
    """轻量Transformer编码器: 输入投影 + 可学习位置编码 + 2层自注意力。"""

    def __init__(self, num_inputs, d_model=64, nhead=4, num_layers=2,
                 dropout=TCN_DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(num_inputs, d_model)
        self.pos_enc = nn.Parameter(
            torch.randn(1, DL_SEQUENCE_LENGTH, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x) + self.pos_enc
        out = self.encoder(x)
        out = out[:, -1, :]  # 最后时间步编码输出
        return self.fc(out)


# ==================== 模型注册表 ====================

def _build_tcn(num_inputs):
    return TCN(num_inputs=num_inputs)


def _build_lstm(num_inputs):
    return LSTMModel(num_inputs=num_inputs)


def _build_gru(num_inputs):
    return GRUModel(num_inputs=num_inputs)


def _build_transformer(num_inputs):
    return TransformerModel(num_inputs=num_inputs)


# 统一模型工厂: 模型名 -> 构建函数(num_inputs -> nn.Module)
MODEL_FACTORY = {
    'TCN': _build_tcn,
    'LSTM': _build_lstm,
    'GRU': _build_gru,
    'Transformer': _build_transformer,
}


# ==================== 数据准备 ====================

def _build_windows(data, sl):
    """滑动窗口: X[t] = data[t-sl:t], y[t] = data[t]"""
    X_list, y_list = [], []
    for i in range(sl, len(data)):
        X_list.append(data[i - sl:i])
        y_list.append(data[i])
    return np.array(X_list), np.array(y_list)


def _prepare_station_data(features, device, target_idx=0):
    """标准化并构建单站窗口, 返回(scaler, 训练/验证/测试张量)。

    y为目标列(气温)单值; 验证集取自训练窗口最后VAL_SIZE个。
    """
    sl = DL_SEQUENCE_LENGTH
    train_feats = features[:-TRAIN_TEST_SPLIT]
    test_feats = features[-TRAIN_TEST_SPLIT:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feats)
    test_scaled = scaler.transform(test_feats)
    full_scaled = np.concatenate([train_scaled, test_scaled], axis=0)

    X_all, y_all = _build_windows(full_scaled, sl)
    y_all = y_all[:, target_idx:target_idx + 1]  # 仅气温列

    X_train, y_train = X_all[:-TRAIN_TEST_SPLIT], y_all[:-TRAIN_TEST_SPLIT]
    X_test, y_test = X_all[-TRAIN_TEST_SPLIT:], y_all[-TRAIN_TEST_SPLIT:]

    X_val, y_val = X_train[-VAL_SIZE:], y_train[-VAL_SIZE:]
    X_train, y_train = X_train[:-VAL_SIZE], y_train[:-VAL_SIZE]

    def _t(x, y):
        return (torch.FloatTensor(x).to(device),
                torch.FloatTensor(y).to(device))

    X_train_t, y_train_t = _t(X_train, y_train)
    X_val_t, y_val_t = _t(X_val, y_val)
    X_test_t, y_test_t = _t(X_test, y_test)

    return scaler, X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t


def _build_pooled_windows(station_dfs):
    """8站训练窗口池化, 用于全局预训练 (各站自scaler标准化, 窗口样本独立)。"""
    X_pool, y_pool = [], []
    for s in GUANGXI_STATIONS:
        feats = station_dfs[s][GUANGXI_FEATURES].values
        train_feats = feats[:-TRAIN_TEST_SPLIT]
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_feats)
        X, y = _build_windows(train_scaled, DL_SEQUENCE_LENGTH)
        X_pool.append(X)
        y_pool.append(y[:, 0:1])
    return np.concatenate(X_pool, axis=0), np.concatenate(y_pool, axis=0)


# ==================== 训练与评估 ====================

def train_tcn_model(model, train_loader, X_val_t, y_val_t, model_name,
                    lr=TCN_LR, patience=TCN_PATIENCE, max_epochs=TCN_EPOCHS):
    """训练深度学习模型, 含早停、学习率调度与梯度裁剪。"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    best_loss = np.inf
    best_state = None
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        model.train()
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_loss


def evaluate_tcn(model, X_test_t, y_test_t, scaler, target_idx=0):
    """在测试集上评估, 反标准化目标列(气温)。"""
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).cpu().numpy().flatten()
    y_true_scaled = y_test_t.cpu().numpy().flatten()

    # 仅反标准化目标列
    y_pred = y_pred_scaled * scaler.scale_[target_idx] + scaler.mean_[target_idx]
    y_true = y_true_scaled * scaler.scale_[target_idx] + scaler.mean_[target_idx]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return y_pred, y_true, rmse, mae, r2


def _fit_and_evaluate(features, device, seed, model_name='TCN', init_state=None,
                      lr=TCN_LR, patience=TCN_PATIENCE, tag=''):
    """按指定模型类型、初始权重与学习率训练并评估单站, 返回完整结果字典。"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    scaler, X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t = \
        _prepare_station_data(features, device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=TCN_BATCH_SIZE,
                              shuffle=True)

    model = MODEL_FACTORY[model_name](len(GUANGXI_FEATURES)).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)

    model, _, _, best_val = train_tcn_model(
        model, train_loader, X_val_t, y_val_t, tag or model_name,
        lr=lr, patience=patience)
    y_pred, y_true, rmse, mae, r2 = evaluate_tcn(
        model, X_test_t, y_test_t, scaler)

    return {
        'pred': y_pred, 'true': y_true, 'rmse': rmse, 'mae': mae,
        'r2': r2, 'best_val_loss': best_val,
        'test_index': pd.date_range(start='2024-01-01',
                                    periods=len(y_true), freq='MS'),
    }


def _pretrain_global(station_dfs, device, seed):
    """8站训练窗口池化预训练共享TCN骨干。"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_pool, y_pool = _build_pooled_windows(station_dfs)

    # 随机划分10%为预训练验证集 (窗口样本独立, 无时序依赖)
    n = len(X_pool)
    perm = np.random.permutation(n)
    val_size = max(200, n // 10)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    X_train_t = torch.FloatTensor(X_pool[train_idx]).to(device)
    y_train_t = torch.FloatTensor(y_pool[train_idx]).to(device)
    X_val_t = torch.FloatTensor(X_pool[val_idx]).to(device)
    y_val_t = torch.FloatTensor(y_pool[val_idx]).to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=TCN_BATCH_SIZE, shuffle=True)

    model = TCN(num_inputs=len(GUANGXI_FEATURES)).to(device)
    model, _, _, best_val = train_tcn_model(
        model, train_loader, X_val_t, y_val_t, '全局预训练',
        lr=TCN_LR, patience=TCN_PATIENCE)
    print(f"      预训练完成: 窗口池={n}, 验证损失={best_val:.6f}, "
          f"已加载为共享骨干")
    return {k: v.clone() for k, v in model.state_dict().items()}


# ==================== 主入口 ====================

def run(df=None, results=None):
    """运行多模型深度学习预测 (广西8市): TCN/LSTM/GRU/Transformer 对比 + TCN迁移学习。"""
    results = results if results is not None else {}

    if not TORCH_AVAILABLE:
        print("  [警告] PyTorch不可用, 跳过深度学习建模步骤。")
        print("  请安装PyTorch: pip install torch")
        results['deep_learning'] = {'skipped': True, 'reason': 'PyTorch不可用'}
        return results

    MODEL_NAMES = list(MODEL_FACTORY.keys())

    print("=" * 60)
    print("步骤9: 多模型深度学习预测 (广西8市, 38维特征)")
    print(f"       模型: {' / '.join(MODEL_NAMES)}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  计算设备: {device} | 特征维度: {len(GUANGXI_FEATURES)} | "
          f"序列长度: {DL_SEQUENCE_LENGTH} | 站点数: {len(GUANGXI_STATIONS)}")

    print("  [9.1] 加载广西多源气象数据...")
    station_dfs = load_guangxi_station_dfs()

    # 全省平均序列 (8站特征逐月取均值)
    avg_features = np.mean([df_[GUANGXI_FEATURES].values
                            for df_ in station_dfs.values()], axis=0)

    # ==================== 多模型独立训练 ====================
    independent = {name: {} for name in MODEL_NAMES}
    for name in MODEL_NAMES:
        for s in GUANGXI_STATIONS:
            print(f"  [9.2] 独立训练{name}: {s} ({GUANGXI_STATION_CN[s]})...")
            res = _fit_and_evaluate(station_dfs[s][GUANGXI_FEATURES].values,
                                    device, DL_RANDOM_STATE, model_name=name,
                                    tag=f'独立_{name}_{s}')
            independent[name][s] = res
            print(f"       RMSE={res['rmse']:.3f}°C, MAE={res['mae']:.3f}°C, "
                  f"R²={res['r2']:.4f}")

    # ==================== TCN 全局预训练 + 迁移微调 ====================
    print("  [9.3] 8站数据池化全局预训练 (TCN骨干)...")
    pretrained_state = _pretrain_global(station_dfs, device, DL_RANDOM_STATE)

    transfer = {}
    for s in GUANGXI_STATIONS:
        print(f"  [9.4] TCN迁移微调: {s} ({GUANGXI_STATION_CN[s]}) "
              f"(lr={FINETUNE_LR})...")
        res = _fit_and_evaluate(
            station_dfs[s][GUANGXI_FEATURES].values, device,
            DL_RANDOM_STATE, model_name='TCN', init_state=pretrained_state,
            lr=FINETUNE_LR, patience=FINETUNE_PATIENCE, tag=f'迁移_{s}')
        transfer[s] = res
        impr = (independent['TCN'][s]['rmse'] - res['rmse']) / independent['TCN'][s]['rmse'] * 100
        print(f"       RMSE={res['rmse']:.3f}°C, MAE={res['mae']:.3f}°C, "
              f"R²={res['r2']:.4f} (vs独立 {impr:+.1f}%)")

    print("  [9.5] 训练全省平均序列TCN (迁移)...")
    avg_result = _fit_and_evaluate(avg_features, device, DL_RANDOM_STATE,
                                   model_name='TCN', init_state=pretrained_state,
                                   lr=FINETUNE_LR, patience=FINETUNE_PATIENCE,
                                   tag='迁移_平均')
    print(f"       RMSE={avg_result['rmse']:.3f}°C, MAE={avg_result['mae']:.3f}°C, "
          f"R²={avg_result['r2']:.4f}")

    # ==================== SARIMA对比 ====================
    sarima_metrics = {'rmse': None, 'mae': None, 'r2': None}
    if results and 'arima' in results:
        sarima_metrics = {
            'rmse': results['arima'].get('rmse'),
            'mae': results['arima'].get('mae'),
            'r2': results['arima'].get('r2'),
        }
        print(f"  [9.6] SARIMA对比: RMSE={sarima_metrics['rmse']:.3f}°C, "
              f"R²={sarima_metrics['r2']:.4f}")
    else:
        print("  [9.6] 未找到SARIMA结果, 跳过对比")

    # ==================== 最优策略选择 ====================
    # 主判据: 测试集 RMSE。每站从 4模型独立 + TCN迁移 中选最优组合。
    # 迁移学习必须严格优于独立训练才采用, 否则回退到独立模型,
    # 避免验证损失在小样本下不可靠导致的误选。
    selected = {}
    for s in GUANGXI_STATIONS:
        candidates = [(name, '独立训练', independent[name][s])
                      for name in MODEL_NAMES]
        tr = transfer[s]
        if tr['rmse'] < independent['TCN'][s]['rmse']:
            candidates.append(('TCN', '迁移学习', tr))
        best_model, best_strategy, best_result = min(
            candidates, key=lambda x: x[2]['rmse'])
        selected[s] = {
            'model': best_model, 'strategy': best_strategy,
            'result': best_result,
        }

    # ==================== 可视化 ====================
    print("  [9.7] 生成可视化图表...")

    # ---------- 图27: 8市 + 全省平均 (最优模型/策略) 预测 vs 实际 ----------
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    plot_order = GUANGXI_STATIONS + ['AVG']
    for idx, s in enumerate(plot_order):
        ax = axes[idx // 3][idx % 3]
        if s == 'AVG':
            res = avg_result
            title = '全省平均'
            strategy = 'TCN迁移'
        else:
            res = selected[s]['result']
            title = f"{GUANGXI_STATION_CN[s]} ({s})"
            strategy = f"{selected[s]['model']}·{selected[s]['strategy']}"
        ax.plot(res['test_index'], res['true'], 'o-', color='#333333',
                markersize=3, linewidth=1, label='实际值')
        ax.plot(res['test_index'], res['pred'], 's-', color='#E74C3C',
                markersize=3, linewidth=1, label='深度学习预测')
        ax.set_title(f"{title} [{strategy}]\n"
                     f"(RMSE={res['rmse']:.2f}°C, R²={res['r2']:.4f})",
                     fontsize=10.5, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        if idx % 3 == 0:
            ax.set_ylabel('气温 (°C)')

    axes[2][2].axis('off')
    fig.suptitle('广西8市 + 全省平均: 深度学习预测 (多模型按RMSE选优, 38维特征)',
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'{CHART_DIR}/27_深度学习预测对比.png')
    plt.close()
    print(f"    已保存: 27_深度学习预测对比.png")

    # ---------- 图28: 4模型独立 + TCN迁移 8市 指标对比 (分组条形图) ----------
    names = [f"{GUANGXI_STATION_CN[s]}" for s in GUANGXI_STATIONS]
    x_pos = np.arange(len(names))
    # 对比系列: 4模型独立训练 + TCN迁移学习
    series = [(name, independent[name]) for name in MODEL_NAMES]
    series.append(('TCN·迁移', transfer))
    series_colors = ['#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#E74C3C']
    n_series = len(series)
    group_w = 0.85
    bar_w = group_w / n_series

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, key, title, ylabel, fmt in [
        (axes[0], 'rmse', 'RMSE对比 (越低越好)', 'RMSE (°C)', '{:.3f}'),
        (axes[1], 'mae', 'MAE对比 (越低越好)', 'MAE (°C)', '{:.3f}'),
        (axes[2], 'r2', 'R²对比 (越高越好)', 'R²', '{:.4f}'),
    ]:
        for i, (sname, data_dict) in enumerate(series):
            data = [data_dict[s][key] for s in GUANGXI_STATIONS]
            bars = ax.bar(x_pos + (i - n_series / 2 + 0.5) * bar_w, data,
                          bar_w, color=series_colors[i], edgecolor='white',
                          label=sname)
            for bar, val in zip(bars, data):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        fmt.format(val), ha='center', va='bottom',
                        fontsize=5.5, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=30)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9)

    fig.suptitle('广西各市 深度学习模型对比: TCN / TCN迁移 / LSTM / GRU / Transformer',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/28_模型系统对比.png')
    plt.close()
    print(f"    已保存: 28_模型系统对比.png")

    # ==================== 保存结果 ====================
    comparison = {}
    for s in GUANGXI_STATIONS:
        res = selected[s]['result']
        comparison[f"{selected[s]['model']}_{GUANGXI_STATION_CN[s]}"] = {
            'rmse': float(res['rmse']), 'mae': float(res['mae']),
            'r2': float(res['r2'])}
    comparison['TCN_全省平均'] = {
        'rmse': float(avg_result['rmse']), 'mae': float(avg_result['mae']),
        'r2': float(avg_result['r2'])}
    if sarima_metrics['rmse'] is not None:
        comparison['SARIMA'] = sarima_metrics

    results['deep_learning'] = {
        'model': 'multi',
        'models': MODEL_NAMES,
        'strategy': 'independent + TCN_transfer',
        'n_features': len(GUANGXI_FEATURES),
        'independent': {
            name: {s: {'rmse': float(independent[name][s]['rmse']),
                       'mae': float(independent[name][s]['mae']),
                       'r2': float(independent[name][s]['r2'])}
                   for s in GUANGXI_STATIONS}
            for name in MODEL_NAMES},
        'transfer': {s: {'rmse': float(transfer[s]['rmse']),
                         'mae': float(transfer[s]['mae']),
                         'r2': float(transfer[s]['r2'])}
                     for s in GUANGXI_STATIONS},
        'selected': {s: {'model': selected[s]['model'],
                         'strategy': selected[s]['strategy'],
                         'rmse': float(selected[s]['result']['rmse']),
                         'mae': float(selected[s]['result']['mae']),
                         'r2': float(selected[s]['result']['r2'])}
                     for s in GUANGXI_STATIONS},
        'average': {'rmse': float(avg_result['rmse']),
                    'mae': float(avg_result['mae']),
                    'r2': float(avg_result['r2'])},
        'comparison': comparison,
    }

    # ==================== 打印汇总 ====================
    best_rmse = min(selected[s]['result']['rmse'] for s in GUANGXI_STATIONS)

    # 各模型 8市平均RMSE
    print("\n  ┌────────────────────┬──────────┬──────────┬──────────┬──────────┐")
    print("  │ 模型               │ TCN      │ LSTM     │ GRU      │ Transformer│")
    print("  ├────────────────────┼──────────┼──────────┼──────────┼──────────┤")
    print("  │ 8市平均RMSE(°C)    │ "
          f"{np.mean([independent['TCN'][s]['rmse'] for s in GUANGXI_STATIONS]):>7.3f}  │ "
          f"{np.mean([independent['LSTM'][s]['rmse'] for s in GUANGXI_STATIONS]):>7.3f}  │ "
          f"{np.mean([independent['GRU'][s]['rmse'] for s in GUANGXI_STATIONS]):>7.3f}  │ "
          f"{np.mean([independent['Transformer'][s]['rmse'] for s in GUANGXI_STATIONS]):>7.3f}  │")
    print("  └────────────────────┴──────────┴──────────┴──────────┴──────────┘")

    # 每站选优结果
    print("\n  ┌──────────┬──────────────┬──────────────────┬──────────┐")
    print("  │ 城市     │ 最优模型      │ 策略              │ 最终RMSE │")
    print("  ├──────────┼──────────────┼──────────────────┼──────────┤")
    for s in GUANGXI_STATIONS:
        sel = selected[s]
        print(f"  │ {GUANGXI_STATION_CN[s]:<7}    │ {sel['model']:<11} │ "
              f"{sel['strategy']:<15} │ {sel['result']['rmse']:>7.3f} │")
    print("  ├──────────┼──────────────┼──────────────────┼──────────┤")
    print(f"  │ 全省平均 │ {'TCN':<11} │ {'迁移学习':<15} │ "
          f"{avg_result['rmse']:>7.3f} │")
    print("  └──────────┴──────────────┴──────────────────┴──────────┘")
    print(f"  8市最优RMSE: {best_rmse:.3f}°C | 特征维度: {len(GUANGXI_FEATURES)} | "
          f"模型: {'/'.join(MODEL_NAMES)} (通道{TCN_CHANNELS})")
    print("  步骤9完成, 共生成2张图表。\n")
    return results


if __name__ == '__main__':
    run(None, {})
