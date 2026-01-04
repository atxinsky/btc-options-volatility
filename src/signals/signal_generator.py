# coding=utf-8
"""
交易信号生成模块
根据波动率预测生成期权交易信号
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型"""
    BUY_VOL = "buy_volatility"      # 买波动率（买跨式）
    SELL_VOL = "sell_volatility"    # 卖波动率（卖跨式）
    HOLD = "hold"                    # 观望


@dataclass
class TradingSignal:
    """交易信号"""
    timestamp: pd.Timestamp
    signal_type: SignalType
    confidence: float           # 置信度 0-1
    predicted_change: float     # 预测的IV变化
    current_dvol: float         # 当前DVOL
    target_dvol: float          # 预测的未来DVOL
    reason: str                 # 信号原因

    # 辅助决策信息
    bb_squeeze_days: int = 0    # 布林带压缩天数
    iv_rv_spread: float = 0     # IV-RV差值
    skew_zscore: float = 0      # Skew Z分数
    term_inverted: bool = False # 期限结构是否倒挂

    def __str__(self):
        direction = "↑" if self.signal_type == SignalType.BUY_VOL else "↓" if self.signal_type == SignalType.SELL_VOL else "→"
        return (f"[{self.timestamp.strftime('%Y-%m-%d %H:%M')}] "
                f"{direction} {self.signal_type.value} | "
                f"置信度: {self.confidence:.1%} | "
                f"预测变化: {self.predicted_change:+.1%} | "
                f"原因: {self.reason}")


class SignalGenerator:
    """交易信号生成器"""

    def __init__(self, config: dict = None):
        self.config = config or {}

        # 信号阈值
        self.buy_threshold = self.config.get('buy_threshold', 0.05)   # 预测涨>5%买
        self.sell_threshold = self.config.get('sell_threshold', -0.03)  # 预测跌>3%卖
        self.min_confidence = self.config.get('min_confidence', 0.6)

        # 辅助条件权重
        self.squeeze_weight = 0.2       # 布林带压缩加成
        self.vrp_weight = 0.15          # VRP极端加成
        self.skew_weight = 0.15         # Skew突变加成
        self.term_weight = 0.1          # 期限结构加成

    def generate_signal(self, prediction: float, confidence: float,
                        features: dict, timestamp: pd.Timestamp) -> TradingSignal:
        """
        生成单个信号

        Parameters:
            prediction: 预测的DVOL变化率
            confidence: 预测置信度
            features: 当前特征值
            timestamp: 时间戳

        Returns:
            TradingSignal
        """
        current_dvol = features.get('dvol', 50)
        target_dvol = current_dvol * (1 + prediction)

        # 基础信号判断
        if prediction > self.buy_threshold:
            base_signal = SignalType.BUY_VOL
        elif prediction < self.sell_threshold:
            base_signal = SignalType.SELL_VOL
        else:
            base_signal = SignalType.HOLD

        # 计算综合置信度
        adj_confidence = confidence

        # 辅助条件调整
        reasons = []

        # 1. 布林带压缩加成
        bb_squeeze_days = features.get('bb_squeeze_days', 0)
        if bb_squeeze_days > 10 and base_signal == SignalType.BUY_VOL:
            adj_confidence += self.squeeze_weight
            reasons.append(f"布林带压缩{bb_squeeze_days}天")

        # 2. VRP极端加成
        vrp = features.get('vrp_30d', 0)
        iv_rv_spread = vrp
        if vrp > 10 and base_signal == SignalType.SELL_VOL:
            # IV远高于RV，适合卖
            adj_confidence += self.vrp_weight
            reasons.append(f"VRP={vrp:.1f}%偏高")
        elif vrp < -5 and base_signal == SignalType.BUY_VOL:
            # IV低于RV，期权便宜
            adj_confidence += self.vrp_weight
            reasons.append(f"VRP={vrp:.1f}%偏低")

        # 3. Skew突变
        skew_zscore = features.get('skew_zscore', 0)
        if abs(skew_zscore) > 2:
            if base_signal == SignalType.BUY_VOL:
                adj_confidence += self.skew_weight
                reasons.append(f"Skew突变(Z={skew_zscore:.1f})")

        # 4. 期限结构倒挂
        term_inverted = features.get('term_inverted', 0) == 1
        if term_inverted and base_signal == SignalType.BUY_VOL:
            adj_confidence += self.term_weight
            reasons.append("期限结构倒挂")

        # 限制置信度范围
        adj_confidence = min(max(adj_confidence, 0), 1)

        # 低置信度转为观望
        if adj_confidence < self.min_confidence and base_signal != SignalType.HOLD:
            base_signal = SignalType.HOLD
            reasons.append("置信度不足")

        # 生成原因说明
        if not reasons:
            if base_signal == SignalType.BUY_VOL:
                reasons.append(f"预测DVOL上涨{prediction:.1%}")
            elif base_signal == SignalType.SELL_VOL:
                reasons.append(f"预测DVOL下跌{abs(prediction):.1%}")
            else:
                reasons.append("预测变化不大")

        return TradingSignal(
            timestamp=timestamp,
            signal_type=base_signal,
            confidence=adj_confidence,
            predicted_change=prediction,
            current_dvol=current_dvol,
            target_dvol=target_dvol,
            reason="; ".join(reasons),
            bb_squeeze_days=bb_squeeze_days,
            iv_rv_spread=iv_rv_spread,
            skew_zscore=skew_zscore,
            term_inverted=term_inverted
        )

    def generate_signals_batch(self, df: pd.DataFrame,
                                predictions: np.ndarray,
                                confidences: np.ndarray = None) -> List[TradingSignal]:
        """
        批量生成信号

        Parameters:
            df: 特征数据
            predictions: 预测值数组
            confidences: 置信度数组

        Returns:
            信号列表
        """
        if confidences is None:
            confidences = np.ones(len(predictions)) * 0.7

        signals = []

        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            # 获取对应行的特征
            row = df.iloc[i] if i < len(df) else df.iloc[-1]

            features = {
                'dvol': row.get('dvol', 50),
                'bb_squeeze_days': row.get('bb_squeeze_days', 0),
                'vrp_30d': row.get('vrp_30d', 0),
                'skew_zscore': row.get('skew_zscore', 0),
                'term_inverted': row.get('term_inverted', 0),
            }

            timestamp = row.get('time', pd.Timestamp.now())
            if not isinstance(timestamp, pd.Timestamp):
                timestamp = pd.Timestamp(timestamp)

            signal = self.generate_signal(pred, conf, features, timestamp)
            signals.append(signal)

        return signals

    def signals_to_dataframe(self, signals: List[TradingSignal]) -> pd.DataFrame:
        """转换信号为DataFrame"""
        data = []
        for s in signals:
            data.append({
                'timestamp': s.timestamp,
                'signal': s.signal_type.value,
                'confidence': s.confidence,
                'predicted_change': s.predicted_change,
                'current_dvol': s.current_dvol,
                'target_dvol': s.target_dvol,
                'reason': s.reason,
                'bb_squeeze_days': s.bb_squeeze_days,
                'iv_rv_spread': s.iv_rv_spread,
                'skew_zscore': s.skew_zscore,
                'term_inverted': s.term_inverted
            })

        return pd.DataFrame(data)

    def filter_signals(self, signals: List[TradingSignal],
                       min_confidence: float = None,
                       signal_types: List[SignalType] = None) -> List[TradingSignal]:
        """筛选信号"""
        filtered = signals

        if min_confidence:
            filtered = [s for s in filtered if s.confidence >= min_confidence]

        if signal_types:
            filtered = [s for s in filtered if s.signal_type in signal_types]

        return filtered

    def get_current_signal(self, df: pd.DataFrame,
                           prediction: float,
                           confidence: float = 0.7) -> TradingSignal:
        """获取当前最新信号"""
        row = df.iloc[-1]

        features = {
            'dvol': row.get('dvol', 50),
            'bb_squeeze_days': row.get('bb_squeeze_days', 0),
            'vrp_30d': row.get('vrp_30d', 0),
            'skew_zscore': row.get('skew_zscore', 0),
            'term_inverted': row.get('term_inverted', 0),
        }

        timestamp = row.get('time', pd.Timestamp.now())
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)

        return self.generate_signal(prediction, confidence, features, timestamp)


# 测试
if __name__ == "__main__":
    # 模拟数据
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=10, freq='D'),
        'dvol': [50, 52, 55, 53, 48, 45, 47, 50, 55, 60],
        'bb_squeeze_days': [0, 0, 5, 10, 15, 18, 20, 22, 5, 0],
        'vrp_30d': [5, 8, 10, 12, 8, -2, -5, -8, 2, 5],
        'skew_zscore': [0.5, 0.8, 1.2, 1.5, 2.5, 2.8, 1.0, 0.5, 0.2, 0.1],
        'term_inverted': [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    })

    predictions = np.array([0.02, 0.03, 0.08, 0.12, 0.15, 0.10, 0.05, -0.02, -0.05, -0.08])

    generator = SignalGenerator()
    signals = generator.generate_signals_batch(df, predictions)

    print("=== 生成的信号 ===")
    for s in signals:
        print(s)

    print("\n=== 筛选买入信号 ===")
    buy_signals = generator.filter_signals(signals, signal_types=[SignalType.BUY_VOL])
    for s in buy_signals:
        print(s)
