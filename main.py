# coding=utf-8
"""
BTC期权波动率预测系统 - 主程序

功能：
1. 获取Deribit期权数据（DVOL、IV、Skew、期限结构）
2. 获取BTC价格数据
3. 构建特征（布林带压缩、IV-RV对比、Skew变化等）
4. 训练预测模型
5. 生成交易信号
6. 回测验证

用法：
    python main.py --mode train    # 训练模型
    python main.py --mode predict  # 生成预测
    python main.py --mode backtest # 回测
    python main.py --mode live     # 实时监控
"""

import argparse
import os
import sys
import yaml
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.deribit_fetcher import DeribitFetcher
from src.data.price_fetcher import PriceFetcher
from src.features.feature_builder import FeatureBuilder
from src.models.predictor import VolatilityPredictor
from src.signals.signal_generator import SignalGenerator, SignalType
from src.backtest.backtester import Backtester

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def fetch_data(config: dict, days: int = 180) -> tuple:
    """
    获取数据

    Returns:
        (price_df, iv_df)
    """
    logger.info(f"获取最近{days}天数据...")

    # 价格数据
    price_fetcher = PriceFetcher()
    price_df = price_fetcher.get_historical_data(days=days, interval="1h")
    logger.info(f"价格数据: {len(price_df)} 行")

    # IV数据 (由于Deribit API限制，这里用模拟数据演示)
    # 实际使用时需要长期收集IV数据
    logger.info("获取IV数据...")
    deribit = DeribitFetcher()

    # 获取当前快照
    snapshot = deribit.get_snapshot()
    logger.info(f"当前DVOL: {snapshot.get('dvol')}")
    logger.info(f"当前ATM IV 7D: {snapshot.get('atm_iv_7d')}")
    logger.info(f"当前Skew 7D: {snapshot.get('skew_7d')}")

    # 获取DVOL历史
    end_time = datetime.now()
    start_time = end_time - timedelta(days=min(days, 30))  # DVOL历史有限制

    dvol_df = deribit.get_dvol_history(start_time, end_time, resolution="60")
    logger.info(f"DVOL历史: {len(dvol_df)} 行")

    # 构建IV DataFrame
    if len(dvol_df) > 0:
        iv_df = dvol_df.rename(columns={'close': 'dvol'})
        # 补充其他IV数据（实际需要历史收集）
        iv_df['atm_iv_7d'] = iv_df['dvol'] * 1.05 + np.random.randn(len(iv_df)) * 2
        iv_df['atm_iv_30d'] = iv_df['dvol'] * 0.98 + np.random.randn(len(iv_df)) * 1
        iv_df['skew_7d'] = np.random.randn(len(iv_df)) * 3
        iv_df['skew_30d'] = np.random.randn(len(iv_df)) * 2
    else:
        iv_df = pd.DataFrame()

    return price_df, iv_df


def build_features(price_df: pd.DataFrame, iv_df: pd.DataFrame,
                   config: dict) -> pd.DataFrame:
    """构建特征"""
    logger.info("构建特征...")

    builder = FeatureBuilder(config.get('features', {}))

    # 合并价格和IV数据
    if len(iv_df) > 0:
        # 重采样到相同频率
        price_hourly = price_df.copy()
        price_hourly['time'] = pd.to_datetime(price_hourly['time']).dt.floor('H')

        iv_hourly = iv_df.copy()
        iv_hourly['time'] = pd.to_datetime(iv_hourly['time']).dt.floor('H')

        df = builder.build_all_features(price_hourly, iv_hourly)
    else:
        df = builder.build_all_features(price_df)

    logger.info(f"特征构建完成: {len(df)} 行, {len(df.columns)} 列")

    return df


def train_model(df: pd.DataFrame, config: dict) -> VolatilityPredictor:
    """训练模型"""
    logger.info("训练模型...")

    builder = FeatureBuilder()
    feature_cols = [c for c in builder.get_feature_columns() if c in df.columns]

    if 'target_dvol_change' not in df.columns:
        logger.error("缺少目标变量")
        return None

    # 过滤有效数据
    df_clean = df[feature_cols + ['target_dvol_change']].dropna()
    logger.info(f"有效训练数据: {len(df_clean)} 行")

    if len(df_clean) < 100:
        logger.warning("数据量不足，使用模拟数据演示")
        # 生成模拟数据
        n = 500
        df_clean = pd.DataFrame({
            col: np.random.randn(n) for col in feature_cols
        })
        df_clean['target_dvol_change'] = np.random.randn(n) * 0.1

    # 训练
    predictor = VolatilityPredictor(
        model_type='lgbm',
        config=config.get('model', {})
    )

    results = predictor.train(df_clean, feature_cols, 'target_dvol_change')

    logger.info(f"训练完成 - Val RMSE: {results.get('val_rmse', 'N/A'):.4f}")
    logger.info(f"方向准确率: {results.get('val_direction_accuracy', 'N/A'):.2%}")

    # 特征重要性
    if 'feature_importance' in results:
        logger.info("\n特征重要性 Top 10:")
        importance = sorted(results['feature_importance'].items(),
                           key=lambda x: x[1], reverse=True)[:10]
        for feat, imp in importance:
            logger.info(f"  {feat}: {imp:.4f}")

    # 保存模型
    model_path = "data/processed/model.pkl"
    predictor.save(model_path)

    return predictor


def generate_signals(df: pd.DataFrame, predictor: VolatilityPredictor,
                     config: dict) -> list:
    """生成交易信号"""
    logger.info("生成交易信号...")

    builder = FeatureBuilder()
    feature_cols = [c for c in builder.get_feature_columns() if c in df.columns]

    # 预测
    df_valid = df[feature_cols].dropna()
    if len(df_valid) == 0:
        logger.warning("没有有效数据进行预测")
        return []

    predictions, confidences = predictor.predict_with_confidence(df_valid)

    if len(predictions) == 0:
        return []

    # 生成信号
    generator = SignalGenerator(config.get('signals', {}))

    # 对齐数据
    df_aligned = df.iloc[-len(predictions):].copy()

    signals = generator.generate_signals_batch(df_aligned, predictions, confidences)

    # 统计
    buy_signals = [s for s in signals if s.signal_type == SignalType.BUY_VOL]
    sell_signals = [s for s in signals if s.signal_type == SignalType.SELL_VOL]
    hold_signals = [s for s in signals if s.signal_type == SignalType.HOLD]

    logger.info(f"信号统计: 买{len(buy_signals)} | 卖{len(sell_signals)} | 观望{len(hold_signals)}")

    return signals


def run_backtest(signals: list, df: pd.DataFrame, config: dict):
    """运行回测"""
    logger.info("运行回测...")

    if 'dvol' not in df.columns:
        logger.warning("缺少DVOL数据，使用模拟数据")
        df['dvol'] = 50 + np.random.randn(len(df)).cumsum() * 0.5

    # 过滤非HOLD信号
    trade_signals = [s for s in signals if s.signal_type != SignalType.HOLD]

    if len(trade_signals) == 0:
        logger.warning("没有可交易信号")
        return

    dvol_series = df.set_index('time')['dvol']

    backtester = Backtester(config.get('backtest', {}))
    result = backtester.run(trade_signals, dvol_series)

    backtester.print_result(result)

    # 保存权益曲线
    if len(result.equity_curve) > 0:
        result.equity_curve.to_csv("data/processed/equity_curve.csv", index=False)
        logger.info("权益曲线已保存: data/processed/equity_curve.csv")


def live_monitor(config: dict):
    """实时监控"""
    logger.info("启动实时监控...")

    deribit = DeribitFetcher()

    while True:
        try:
            # 获取当前快照
            snapshot = deribit.get_snapshot()

            print("\n" + "=" * 50)
            print(f"时间: {snapshot['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"DVOL: {snapshot.get('dvol', 'N/A')}")
            print(f"ATM IV 7D: {snapshot.get('atm_iv_7d', 'N/A')}")
            print(f"ATM IV 30D: {snapshot.get('atm_iv_30d', 'N/A')}")
            print(f"Skew 7D: {snapshot.get('skew_7d', 'N/A')}")
            print(f"期限斜率: {snapshot.get('term_slope', 'N/A')}")
            print("=" * 50)

            # 等待下次更新
            import time
            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("监控已停止")
            break
        except Exception as e:
            logger.error(f"监控出错: {e}")
            import time
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description='BTC期权波动率预测系统')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['fetch', 'train', 'predict', 'backtest', 'live', 'all'],
                       help='运行模式')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--days', type=int, default=90,
                       help='获取数据天数')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    if args.mode == 'live':
        live_monitor(config)
        return

    # 获取数据
    if args.mode in ['fetch', 'all']:
        price_df, iv_df = fetch_data(config, args.days)
        price_df.to_csv("data/raw/price.csv", index=False)
        if len(iv_df) > 0:
            iv_df.to_csv("data/raw/iv.csv", index=False)
    else:
        # 加载已有数据
        price_df = pd.read_csv("data/raw/price.csv")
        iv_df = pd.read_csv("data/raw/iv.csv") if os.path.exists("data/raw/iv.csv") else pd.DataFrame()

    # 构建特征
    df = build_features(price_df, iv_df, config)
    df.to_csv("data/processed/features.csv", index=False)

    # 训练模型
    if args.mode in ['train', 'all']:
        predictor = train_model(df, config)
    else:
        predictor = VolatilityPredictor(model_type='lgbm')
        predictor.load("data/processed/model.pkl")

    # 生成信号
    if args.mode in ['predict', 'all']:
        signals = generate_signals(df, predictor, config)

        # 打印最新信号
        if signals:
            print("\n最新10个信号:")
            for s in signals[-10:]:
                print(s)

    # 回测
    if args.mode in ['backtest', 'all']:
        signals = generate_signals(df, predictor, config)
        run_backtest(signals, df, config)


if __name__ == "__main__":
    main()
