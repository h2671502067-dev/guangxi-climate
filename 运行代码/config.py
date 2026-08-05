"""
广西壮族自治区气象数据建模分析项目 - 全局配置
=============================================
定义项目路径、数据列名、建模参数等全局配置。
"""

import os

# ======================== 路径配置 ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
CHART_DIR = os.path.join(OUTPUT_DIR, 'charts')
PREPROCESSED_DIR = os.path.join(OUTPUT_DIR, 'preprocessed')
STATION_FEATURES_CSV = os.path.join(PREPROCESSED_DIR, 'guangxi_station_features.csv')
PROVINCE_AVG_CSV = os.path.join(PREPROCESSED_DIR, 'guangxi_province_average.csv')
REPORT_PATH = os.path.join(OUTPUT_DIR, '广西壮族自治区气象数据建模分析报告.txt')
RESULTS_PATH = os.path.join(OUTPUT_DIR, 'analysis_results.pkl')

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

# ======================== 数据列配置 ========================
# [修复] 短波辐射(MJ/m²)列全为0, 从核心列中剔除
CORE_COLUMNS = [
    '平均气温(°C)', '最高气温(°C)', '最低气温(°C)',
    '平均湿度(%)', '总降水量(mm)', '平均风速(km/h)',
    '日照时数(h)', '参考蒸散量ET₀(mm)',
    '平均海平面气压(hPa)', '土壤温度0-7cm(°C)', '土壤湿度0-7cm(m³/m³)'
]

# [修复] 剔除短波辐射, 避免无效特征参与建模
REGRESSION_FEATURES = [
    '平均湿度(%)', '总降水量(mm)', '平均风速(km/h)',
    '日照时数(h)',
    '平均海平面气压(hPa)', '土壤温度0-7cm(°C)'
]
REGRESSION_TARGET = '平均气温(°C)'

# VIF>10的变量(气压、土壤温度)剔除后的精简特征集 (用于岭回归对比)
REGRESSION_FEATURES_CLEAN = [
    '平均湿度(%)', '总降水量(mm)', '平均风速(km/h)', '日照时数(h)'
]

CLUSTER_FEATURES = [
    '平均气温(°C)', '总降水量(mm)', '平均湿度(%)',
    '平均风速(km/h)', '日照时数(h)'
]

TREND_COLUMNS = [
    '平均气温(°C)', '总降水量(mm)', '平均湿度(%)', '平均风速(km/h)',
    '日照时数(h)', '参考蒸散量ET₀(mm)', '平均海平面气压(hPa)'
]

# ======================== 建模参数 ========================
ARIMA_SEARCH_P = range(0, 4)
ARIMA_SEARCH_D = range(0, 2)
ARIMA_SEARCH_Q = range(0, 4)
SARIMA_SEARCH_P = range(0, 2)  # 季节AR阶数
SARIMA_SEARCH_D = range(0, 2)
SARIMA_SEARCH_Q = range(0, 2)  # 季节MA阶数
SARIMA_S = 12
TRAIN_TEST_SPLIT = 24

CLUSTER_K_RANGE = range(2, 9)

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42

REGRESSION_TEST_SIZE = 0.2
REGRESSION_RANDOM_STATE = 42

# ======================== 深度学习参数 ========================
DL_EPOCHS = 100
DL_BATCH_SIZE = 16
DL_LR = 0.001
DL_HIDDEN_SIZE = 64
DL_NUM_LAYERS = 2
DL_DROPOUT = 0.2
DL_SEQUENCE_LENGTH = 12  # 用过去12个月预测下1个月
DL_RANDOM_STATE = 42

# ======================== 广西数据配置 ========================
GUANGXI_DATA_DIR = os.path.join(BASE_DIR, 'data')
GUANGXI_STATIONS = ['Baise', 'Beihai', 'Fangchenggang', 'Guilin',
                    'Hechi', 'Liuzhou', 'Nanning', 'Wuzhou']
GUANGXI_STATION_CN = {'Baise': '百色', 'Beihai': '北海', 'Fangchenggang': '防城港',
                      'Guilin': '桂林', 'Hechi': '河池', 'Liuzhou': '柳州',
                      'Nanning': '南宁', 'Wuzhou': '梧州'}

# 预测目标: 2m气温
GUANGXI_TARGET = '气温_2m_℃'

# 地面气象特征 (NASA POWER), 24个
GUANGXI_FEATURES_GROUND = [
    '气温_2m_℃', '最高气温_℃', '最低气温_℃', '气温日较差_℃',
    '露点温度_℃', '地表温度_℃', '气压_kPa',
    '风速_10m_ms', '最大风速_10m_ms', '风速_50m_ms', '风向_10m_度',
    '相对湿度_%', '比湿_gkg', '降水量_mm_day',
    '短波辐射_全天空_kWhm2day', '长波辐射_全天空_kWhm2day', '光合有效辐射_kWhm2day',
    '紫外线指数', '总云量_%',
    '蒸散发_mm_day', '陆地蒸发_mm_day',
    '根区土壤湿度_01', '剖面土壤湿度_01', '表层土壤湿度_01',
]

# 干旱指数特征, 4个
GUANGXI_FEATURES_DROUGHT = ['SPI_1m', 'SPI_3m', 'SPEI_1m', 'SPEI_3m']

# 高空衍生特征 (NCEP), 7个
GUANGXI_FEATURES_UPPER = [
    'K指数_℃', 'Showalter指数_℃', '整层水汽通量_kg_ms',
    '温度_℃_850hPa', '温度_℃_500hPa',
    '比湿_gkg_850hPa', '比湿_gkg_700hPa',
]

# 植被指数特征 (MODIS), 1个
GUANGXI_NDVI_COL = 'NDVI'

# 38维特征 = 24地面 + 4干旱 + 7高空 + 1植被 + 2月份周期编码
GUANGXI_FEATURES = (GUANGXI_FEATURES_GROUND + GUANGXI_FEATURES_DROUGHT +
                    GUANGXI_FEATURES_UPPER + [GUANGXI_NDVI_COL,
                                               'month_sin', 'month_cos'])

# ======================== TCN参数 ========================
TCN_CHANNELS = [64, 64, 64]        # 每层通道数(3层, 膨胀率1/2/4)
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.2
TCN_EPOCHS = 100
TCN_BATCH_SIZE = 16
TCN_LR = 0.001
TCN_PATIENCE = 15

# ======================== 可视化配置 ========================
plt_config = {
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
}

SEASON_MAP = {
    12: '冬季', 1: '冬季', 2: '冬季',
    3: '春季', 4: '春季', 5: '春季',
    6: '夏季', 7: '夏季', 8: '夏季',
    9: '秋季', 10: '秋季', 11: '秋季'
}
SEASON_ORDER = ['春季', '夏季', '秋季', '冬季']
SEASON_COLORS = {'春季': '#2ECC71', '夏季': '#E74C3C', '秋季': '#F39C12', '冬季': '#3498DB'}
