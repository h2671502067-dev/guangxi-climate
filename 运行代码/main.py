"""
广西壮族自治区气象数据建模分析项目 - 主入口
=====================================
一键运行全部建模步骤, 生成所有图表和TXT报告。

使用方法:
    python main.py              # 运行全部步骤
    python main.py --step 4     # 仅运行步骤4
    python main.py --list       # 列出所有可用步骤
"""

import sys
import os
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_PATH, REPORT_PATH, CHART_DIR


STEPS = {
    1: ('data_loader', 'load_and_clean', '数据加载与清洗'),
    2: ('step1_descriptive', 'run', '描述性统计分析'),
    3: ('step2_correlation', 'run', '相关性分析'),
    4: ('step3_time_series', 'run', '时间序列分解与平稳性检验'),
    5: ('step4_arima', 'run', 'ARIMA/Holt-Winters预测建模'),
    6: ('step5_regression', 'run', '多元回归建模 (OLS+Ridge+RF)'),
    7: ('step6_clustering', 'run', '聚类分析'),
    8: ('step7_trend', 'run', 'Mann-Kendall趋势检验'),
    9: ('step8_extreme', 'run', '极端事件与专题分析'),
    10: ('step9_deep_learning', 'run', '深度学习预测 (TCN/LSTM/GRU/Transformer 38维)'),
}


def run_step(step_num, df, results):
    module_name, func_name, desc = STEPS[step_num]
    mod = __import__(module_name, fromlist=[func_name])
    func = getattr(mod, func_name)
    print(f"\n{'▶':>2} 步骤{step_num}: {desc}")
    if step_num == 1:
        result = func()
        if len(result) == 3:
            df, numeric_cols, invalid_cols = result
            results['numeric_cols'] = numeric_cols
            results['invalid_cols'] = invalid_cols
        # 从df动态获取数据基本信息
        if df is not None:
            results['n_samples'] = len(df)
            results['n_features'] = len([c for c in df.columns if c not in
                                         ['年份', '月份', '日期', '年份_int', '月份_int', '季节',
                                          '天气现象代码', '天气现象']])
            results['date_min'] = df['日期'].min().strftime('%Y-%m')
            results['date_max'] = df['日期'].max().strftime('%Y-%m')
        return df, results
    else:
        results = func(df, results)
        return df, results


def main(selected_steps=None):
    # 数据预处理: data/ 原始数据 -> output/preprocessed/
    try:
        from preprocess_data import run as run_preprocess
        run_preprocess()
    except Exception as e:
        print(f"  ⚠️ 数据预处理失败, 将使用回退逻辑: {e}")

    print()
    print("╔" + "═" * 58 + "╗")
    print("║   广西壮族自治区气象数据完整建模分析 (2005-2025)            ║")
    print("║   多模块数学建模 + 深度学习项目                      ║")
    print("╚" + "═" * 58 + "╝")
    print()

    total_start = time.time()
    results = {}
    df = None

    steps_to_run = selected_steps if selected_steps else list(STEPS.keys())

    for step_num in steps_to_run:
        try:
            df, results = run_step(step_num, df, results)
        except Exception as e:
            print(f"\n  ❌ 步骤{step_num}执行失败: {e}")
            traceback.print_exc()
            print(f"  跳过步骤{step_num}, 继续执行后续步骤...\n")

    if not results:
        print("  没有生成任何分析结果, 退出。")
        return

    with open(RESULTS_PATH, 'wb') as f:
        import pickle
        pickle.dump(results, f)
    print(f"  分析结果已保存: {RESULTS_PATH}")

    try:
        from generate_report import generate_report
        report_path = generate_report(results)
    except Exception as e:
        print(f"  ❌ 报告生成失败: {e}")
        traceback.print_exc()
        report_path = None

    total_time = time.time() - total_start
    chart_count = len([f for f in os.listdir(CHART_DIR) if f.endswith('.png')])

    print()
    print("╔" + "═" * 58 + "╗")
    print("║                    全部分析完成!                       ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  图表数量: {chart_count:>3} 张                                   ║")
    if report_path:
        print(f"║  报告文件: {os.path.basename(report_path):<36} ║")
    print(f"║  图表目录: output/charts/                             ║")
    print(f"║  总耗时:   {total_time:>6.1f} 秒                                 ║")
    print("╚" + "═" * 58 + "╝")
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='广西壮族自治区气象数据建模分析')
    parser.add_argument('--step', type=int, nargs='+', help='运行指定步骤 (1-11)')
    parser.add_argument('--list', action='store_true', help='列出所有可用步骤')
    args = parser.parse_args()

    if args.list:
        print("\n可用步骤:")
        for num, (_, _, desc) in STEPS.items():
            print(f"  {num:>2}. {desc}")
        print()
        sys.exit(0)

    main(selected_steps=args.step)
