# coding=utf-8
"""
特征工程模块
实现4大类特征：
1. 价格特征（布林带收窄、波动率压缩）
2. IV-RV对比特征
3. 期限结构特征
4. Skew特征
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """特征构建器"""

    def __init__(self, config: dict = None):
        self.config = config or {}

        # 默认参数
        self.bb_window = self.config.get('bb_window', 20)
        self.bb_std = self.config.get('bb_std', 2)
        self.rv_windows = self.config.get('rv_windows', [7, 14, 30, 60])

    # ==================== 1. 价格/波动率压缩特征 ====================

    def add_bollinger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        布林带特征 - 检测波动率压缩

        核心逻辑：布林带收窄 → 波动率压缩 → 即将爆发
        """
        df = df.copy()

        # 布林带
        df['bb_mid'] = df['close'].rolling(self.bb_window).mean()
        df['bb_std'] = df['close'].rolling(self.bb_window).std()
        df['bb_upper'] = df['bb_mid'] + self.bb_std * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - self.bb_std * df['bb_std']

        # 布林带宽度 (关键指标)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        # 布林带宽度的分位数 (越低越压缩)
        df['bb_width_percentile'] = df['bb_width'].rolling(90).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )

        # 布林带宽度变化率
        df['bb_width_change'] = df['bb_width'].pct_change(5)

        # 价格在布林带中的位置 (0-1)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 连续收窄天数
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)
        df['bb_squeeze_days'] = df['bb_squeeze'].groupby(
            (df['bb_squeeze'] != df['bb_squeeze'].shift()).cumsum()
        ).cumsum()

        return df

    def add_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        已实现波动率特征

        多个窗口的RV，用于和IV对比
        """
        df = df.copy()

        # 对数收益率
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

        # 多窗口已实现波动率 (年化)
        for window in self.rv_windows:
            df[f'rv_{window}d'] = df['log_return'].rolling(window).std() * np.sqrt(365) * 100

        # RV变化率
        df['rv_7d_change'] = df['rv_7d'].pct_change(7)
        df['rv_30d_change'] = df['rv_30d'].pct_change(7)

        # RV的分位数
        df['rv_7d_percentile'] = df['rv_7d'].rolling(90).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )

        # 短期RV vs 长期RV (波动率锥)
        df['rv_ratio_7_30'] = df['rv_7d'] / df['rv_30d']
        df['rv_ratio_7_60'] = df['rv_7d'] / df['rv_60d']

        return df

    def add_atr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR特征"""
        df = df.copy()

        # True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )

        # ATR
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['atr_14_pct'] = df['atr_14'] / df['close'] * 100

        # ATR变化
        df['atr_change'] = df['atr_14'].pct_change(7)

        return df

    # ==================== 2. IV-RV对比特征 ====================

    def add_iv_rv_features(self, df: pd.DataFrame,
                           iv_col: str = 'atm_iv_30d') -> pd.DataFrame:
        """
        IV vs RV 对比特征

        核心逻辑：
        - IV > RV → 期权贵，卖波动率
        - IV < RV → 期权便宜，买波动率
        """
        df = df.copy()

        if iv_col not in df.columns:
            logger.warning(f"缺少IV列: {iv_col}")
            return df

        # IV - RV 差值 (Variance Risk Premium)
        df['vrp_30d'] = df[iv_col] - df['rv_30d']
        df['vrp_7d'] = df.get('atm_iv_7d', df[iv_col]) - df['rv_7d']

        # IV/RV 比值
        df['iv_rv_ratio'] = df[iv_col] / df['rv_30d']

        # VRP的分位数
        df['vrp_percentile'] = df['vrp_30d'].rolling(60).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )

        # VRP变化
        df['vrp_change'] = df['vrp_30d'].diff(7)

        # IV相对于自身历史的位置
        df['iv_percentile'] = df[iv_col].rolling(90).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )

        return df

    # ==================== 3. 期限结构特征 ====================

    def add_term_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        期限结构特征

        核心逻辑：
        - 正常结构：远月IV > 近月IV (正斜率)
        - 倒挂结构：近月IV > 远月IV (负斜率) → 市场恐慌
        """
        df = df.copy()

        if 'atm_iv_7d' not in df.columns or 'atm_iv_30d' not in df.columns:
            logger.warning("缺少IV数据，跳过期限结构特征")
            return df

        # 期限结构斜率 (近月-远月)
        df['term_spread'] = df['atm_iv_7d'] - df['atm_iv_30d']

        # 是否倒挂
        df['term_inverted'] = (df['term_spread'] > 0).astype(int)

        # 斜率变化
        df['term_spread_change'] = df['term_spread'].diff(3)

        # 斜率分位数
        df['term_spread_percentile'] = df['term_spread'].rolling(60).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )

        # 倒挂持续天数
        df['term_inverted_days'] = df['term_inverted'].groupby(
            (df['term_inverted'] != df['term_inverted'].shift()).cumsum()
        ).cumsum() * df['term_inverted']

        return df

    # ==================== 4. Skew特征 ====================

    def add_skew_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        波动率偏斜特征

        核心逻辑：
        - Skew突然变大 → Put需求增加 → 有人买保险 → 可能有大事
        - Skew突然变小 → Call需求增加 → 市场看涨
        """
        df = df.copy()

        if 'skew_7d' not in df.columns:
            logger.warning("缺少Skew数据")
            return df

        # Skew变化 (关键指标)
        df['skew_change_1d'] = df['skew_7d'].diff(1)
        df['skew_change_3d'] = df['skew_7d'].diff(3)
        df['skew_change_7d'] = df['skew_7d'].diff(7)

        # Skew分位数
        df['skew_percentile'] = df['skew_7d'].rolling(60).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )

        # Skew突变检测 (Z-score)
        skew_mean = df['skew_7d'].rolling(20).mean()
        skew_std = df['skew_7d'].rolling(20).std()
        df['skew_zscore'] = (df['skew_7d'] - skew_mean) / skew_std

        # Skew极端值标记
        df['skew_extreme'] = (abs(df['skew_zscore']) > 2).astype(int)

        return df

    # ==================== 5. DVOL特征 ====================

    def add_dvol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DVOL指数特征

        DVOL = Deribit的波动率指数，类似VIX
        """
        df = df.copy()

        if 'dvol' not in df.columns:
            logger.warning("缺少DVOL数据")
            return df

        # DVOL均线
        df['dvol_ma7'] = df['dvol'].rolling(7).mean()
        df['dvol_ma20'] = df['dvol'].rolling(20).mean()

        # DVOL变化
        df['dvol_change_1d'] = df['dvol'].pct_change(1)
        df['dvol_change_7d'] = df['dvol'].pct_change(7)

        # DVOL分位数
        df['dvol_percentile'] = df['dvol'].rolling(90).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )

        # DVOL趋势 (高于/低于均线)
        df['dvol_above_ma'] = (df['dvol'] > df['dvol_ma20']).astype(int)

        # DVOL波动率 (波动率的波动率)
        df['dvol_volatility'] = df['dvol'].rolling(14).std()

        return df

    # ==================== 目标变量 ====================

    def add_target(self, df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
        """
        添加预测目标

        目标：未来N天DVOL的变化
        """
        df = df.copy()

        if 'dvol' not in df.columns:
            logger.warning("缺少DVOL数据，无法创建目标")
            return df

        # 未来N天的DVOL
        df['dvol_future'] = df['dvol'].shift(-horizon)

        # 目标1：DVOL变化率
        df['target_dvol_change'] = (df['dvol_future'] - df['dvol']) / df['dvol']

        # 目标2：DVOL变化方向 (分类任务)
        df['target_dvol_direction'] = (df['dvol_future'] > df['dvol']).astype(int)

        # 目标3：大幅变化 (涨>10% 或 跌>10%)
        df['target_big_move'] = (abs(df['target_dvol_change']) > 0.10).astype(int)

        return df

    # ==================== 综合构建 ====================

    def build_all_features(self, price_df: pd.DataFrame,
                           iv_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        构建所有特征

        Parameters:
            price_df: 价格数据 [time, open, high, low, close, volume]
            iv_df: IV数据 [time, dvol, atm_iv_7d, atm_iv_30d, skew_7d, skew_30d]

        Returns:
            完整特征DataFrame
        """
        df = price_df.copy()

        # 1. 价格特征
        logger.info("构建价格特征...")
        df = self.add_bollinger_features(df)
        df = self.add_realized_volatility(df)
        df = self.add_atr_features(df)

        # 2. 合并IV数据
        if iv_df is not None and len(iv_df) > 0:
            logger.info("合并IV数据...")
            df = pd.merge(df, iv_df, on='time', how='left')

            # 3. IV-RV特征
            logger.info("构建IV-RV特征...")
            df = self.add_iv_rv_features(df)

            # 4. 期限结构特征
            logger.info("构建期限结构特征...")
            df = self.add_term_structure_features(df)

            # 5. Skew特征
            logger.info("构建Skew特征...")
            df = self.add_skew_features(df)

            # 6. DVOL特征
            logger.info("构建DVOL特征...")
            df = self.add_dvol_features(df)

            # 7. 目标变量
            logger.info("构建目标变量...")
            df = self.add_target(df)

        # 移除NaN
        df = df.dropna()

        logger.info(f"特征构建完成: {len(df)} 行, {len(df.columns)} 列")

        return df

    def get_feature_columns(self) -> List[str]:
        """获取特征列名"""
        return [
            # 布林带特征
            'bb_width', 'bb_width_percentile', 'bb_width_change',
            'bb_position', 'bb_squeeze_days',

            # RV特征
            'rv_7d', 'rv_14d', 'rv_30d', 'rv_60d',
            'rv_7d_change', 'rv_7d_percentile',
            'rv_ratio_7_30', 'rv_ratio_7_60',

            # ATR特征
            'atr_14_pct', 'atr_change',

            # IV-RV特征
            'vrp_30d', 'vrp_7d', 'iv_rv_ratio',
            'vrp_percentile', 'vrp_change', 'iv_percentile',

            # 期限结构特征
            'term_spread', 'term_inverted', 'term_spread_change',
            'term_spread_percentile', 'term_inverted_days',

            # Skew特征
            'skew_7d', 'skew_change_1d', 'skew_change_3d', 'skew_change_7d',
            'skew_percentile', 'skew_zscore', 'skew_extreme',

            # DVOL特征
            'dvol', 'dvol_ma7', 'dvol_ma20',
            'dvol_change_1d', 'dvol_change_7d',
            'dvol_percentile', 'dvol_above_ma', 'dvol_volatility',
        ]


# 测试
if __name__ == "__main__":
    # 生成模拟数据测试
    import numpy as np

    np.random.seed(42)
    n = 500

    # 模拟价格数据
    price_df = pd.DataFrame({
        'time': pd.date_range('2023-01-01', periods=n, freq='D'),
        'open': 30000 + np.random.randn(n).cumsum() * 100,
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000, 10000, n)
    })
    price_df['high'] = price_df['open'] + abs(np.random.randn(n) * 500)
    price_df['low'] = price_df['open'] - abs(np.random.randn(n) * 500)
    price_df['close'] = price_df['open'] + np.random.randn(n) * 300

    # 模拟IV数据
    iv_df = pd.DataFrame({
        'time': price_df['time'],
        'dvol': 50 + np.random.randn(n).cumsum() * 2,
        'atm_iv_7d': 55 + np.random.randn(n) * 5,
        'atm_iv_30d': 50 + np.random.randn(n) * 3,
        'skew_7d': np.random.randn(n) * 3,
        'skew_30d': np.random.randn(n) * 2,
    })

    # 构建特征
    builder = FeatureBuilder()
    df = builder.build_all_features(price_df, iv_df)

    print("特征列:")
    print(df.columns.tolist())
    print(f"\n数据形状: {df.shape}")
    print(f"\n前5行:")
    print(df.head())
