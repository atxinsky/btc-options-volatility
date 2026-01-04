# coding=utf-8
"""
回测框架
基于波动率交易策略的回测
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import logging

from ..signals.signal_generator import TradingSignal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """交易记录"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    signal_type: SignalType
    entry_dvol: float
    exit_dvol: float
    position_size: float    # 名义本金
    pnl: float              # 盈亏
    pnl_pct: float          # 盈亏百分比
    holding_days: int
    exit_reason: str


@dataclass
class BacktestResult:
    """回测结果"""
    # 基本信息
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    final_capital: float

    # 收益指标
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # 风险指标
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float

    # 交易指标
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_holding_days: float

    # 明细
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)


class Backtester:
    """回测器"""

    def __init__(self, config: dict = None):
        self.config = config or {}

        self.initial_capital = self.config.get('initial_capital', 100000)
        self.position_size = self.config.get('position_size', 0.1)  # 每次用10%
        self.commission = self.config.get('commission', 0.0003)
        self.slippage = self.config.get('slippage', 0.001)

        # 风控
        self.max_drawdown = self.config.get('max_drawdown', 0.15)
        self.max_position = self.config.get('max_position', 0.3)

        # 持仓状态
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []

    def reset(self):
        """重置状态"""
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []

    def _calculate_vol_pnl(self, entry_dvol: float, exit_dvol: float,
                           signal_type: SignalType, position_size: float) -> float:
        """
        计算波动率交易盈亏

        简化模型：
        - 买波动率：DVOL上涨赚钱
        - 卖波动率：DVOL下跌赚钱

        实际中需要根据Vega、期权价格等计算
        这里使用简化的线性关系
        """
        dvol_change = (exit_dvol - entry_dvol) / entry_dvol

        # 简化的Vega敏感度 (每1% DVOL变化，组合变化约0.5%)
        vega_sensitivity = 0.5

        if signal_type == SignalType.BUY_VOL:
            # 买波动率，DVOL涨则赚
            pnl_pct = dvol_change * vega_sensitivity
        else:  # SELL_VOL
            # 卖波动率，DVOL跌则赚
            pnl_pct = -dvol_change * vega_sensitivity

        # 扣除手续费和滑点
        pnl_pct -= (self.commission + self.slippage) * 2  # 开平各一次

        return position_size * pnl_pct

    def run(self, signals: List[TradingSignal],
            dvol_series: pd.Series,
            holding_period: int = 7) -> BacktestResult:
        """
        运行回测

        Parameters:
            signals: 交易信号列表
            dvol_series: DVOL时间序列 (index=timestamp)
            holding_period: 默认持仓周期（天）

        Returns:
            回测结果
        """
        self.reset()

        # 转换为按时间索引
        if not isinstance(dvol_series.index, pd.DatetimeIndex):
            dvol_series = dvol_series.set_index('time')['dvol'] if 'time' in dvol_series else dvol_series

        # 信号转DataFrame方便处理
        signals_df = pd.DataFrame([{
            'timestamp': s.timestamp,
            'signal': s.signal_type,
            'confidence': s.confidence,
            'predicted_change': s.predicted_change
        } for s in signals])

        if len(signals_df) == 0:
            logger.warning("没有信号")
            return self._create_empty_result()

        signals_df = signals_df.set_index('timestamp').sort_index()

        # 遍历每个交易日
        all_dates = dvol_series.index.sort_values()
        entry_info = None  # (entry_time, entry_dvol, signal_type, position_size)

        for date in all_dates:
            current_dvol = dvol_series.loc[date]

            # 记录权益
            unrealized_pnl = 0
            if entry_info:
                unrealized_pnl = self._calculate_vol_pnl(
                    entry_info[1], current_dvol, entry_info[2], entry_info[3]
                )

            self.equity_curve.append({
                'time': date,
                'equity': self.capital + unrealized_pnl,
                'dvol': current_dvol
            })

            # 检查是否需要平仓
            if entry_info:
                entry_time, entry_dvol, signal_type, pos_size = entry_info
                days_held = (date - entry_time).days

                should_exit = False
                exit_reason = ""

                # 1. 到期平仓
                if days_held >= holding_period:
                    should_exit = True
                    exit_reason = "holding_period"

                # 2. 止损
                temp_pnl = self._calculate_vol_pnl(entry_dvol, current_dvol, signal_type, pos_size)
                if temp_pnl / self.capital < -0.05:  # 单笔亏5%止损
                    should_exit = True
                    exit_reason = "stop_loss"

                # 3. 最大回撤止损
                peak_equity = max([e['equity'] for e in self.equity_curve])
                current_dd = (peak_equity - (self.capital + unrealized_pnl)) / peak_equity
                if current_dd > self.max_drawdown:
                    should_exit = True
                    exit_reason = "max_drawdown"

                if should_exit:
                    pnl = self._calculate_vol_pnl(entry_dvol, current_dvol, signal_type, pos_size)
                    self.capital += pnl

                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=date,
                        signal_type=signal_type,
                        entry_dvol=entry_dvol,
                        exit_dvol=current_dvol,
                        position_size=pos_size,
                        pnl=pnl,
                        pnl_pct=pnl / pos_size,
                        holding_days=days_held,
                        exit_reason=exit_reason
                    )
                    self.trades.append(trade)
                    entry_info = None

            # 检查是否有新信号要入场
            if entry_info is None and date in signals_df.index:
                row = signals_df.loc[date]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                signal_type = row['signal']
                confidence = row['confidence']

                if signal_type in [SignalType.BUY_VOL, SignalType.SELL_VOL]:
                    # 根据置信度调整仓位
                    pos_size = self.capital * self.position_size * confidence

                    # 不超过最大仓位
                    pos_size = min(pos_size, self.capital * self.max_position)

                    entry_info = (date, current_dvol, signal_type, pos_size)

        # 计算结果
        return self._calculate_result()

    def _calculate_result(self) -> BacktestResult:
        """计算回测结果"""
        if not self.equity_curve:
            return self._create_empty_result()

        equity_df = pd.DataFrame(self.equity_curve)
        equity_df = equity_df.set_index('time')

        # 基本指标
        start_date = equity_df.index[0]
        end_date = equity_df.index[-1]
        final_capital = equity_df['equity'].iloc[-1]

        # 收益
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        days = (end_date - start_date).days
        annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0

        # 波动率
        returns = equity_df['equity'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(365) if len(returns) > 0 else 0

        # 夏普比率
        rf = 0.02  # 无风险利率
        sharpe = (annual_return - rf) / volatility if volatility > 0 else 0

        # 索提诺比率
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 0 else 0.0001
        sortino = (annual_return - rf) / downside_std if downside_std > 0 else 0

        # 最大回撤
        equity_curve = equity_df['equity']
        peak = equity_curve.expanding().max()
        drawdown = (peak - equity_curve) / peak
        max_dd = drawdown.max()

        # 最大回撤持续时间
        dd_duration = 0
        if max_dd > 0:
            in_dd = drawdown > 0
            dd_groups = (in_dd != in_dd.shift()).cumsum()
            dd_lengths = in_dd.groupby(dd_groups).sum()
            dd_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0

        # 卡尔玛比率
        calmar = annual_return / max_dd if max_dd > 0 else 0

        # 交易指标
        total_trades = len(self.trades)
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(wins) / total_trades if total_trades > 0 else 0

        total_profit = sum(t.pnl for t in wins) if wins else 0
        total_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0001
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0

        avg_holding = np.mean([t.holding_days for t in self.trades]) if self.trades else 0

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=dd_duration,
            volatility=volatility,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_holding_days=avg_holding,
            trades=self.trades,
            equity_curve=equity_df.reset_index()
        )

    def _create_empty_result(self) -> BacktestResult:
        """创建空结果"""
        now = pd.Timestamp.now()
        return BacktestResult(
            start_date=now,
            end_date=now,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_return=0,
            annual_return=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            max_drawdown_duration=0,
            volatility=0,
            total_trades=0,
            win_rate=0,
            profit_factor=0,
            avg_win=0,
            avg_loss=0,
            avg_holding_days=0,
            trades=[],
            equity_curve=pd.DataFrame()
        )

    def print_result(self, result: BacktestResult):
        """打印回测结果"""
        print("\n" + "=" * 60)
        print("               回测结果")
        print("=" * 60)

        print(f"\n【时间范围】")
        print(f"  开始: {result.start_date.strftime('%Y-%m-%d')}")
        print(f"  结束: {result.end_date.strftime('%Y-%m-%d')}")

        print(f"\n【收益指标】")
        print(f"  初始资金: ${result.initial_capital:,.0f}")
        print(f"  期末资金: ${result.final_capital:,.0f}")
        print(f"  总收益率: {result.total_return:.2%}")
        print(f"  年化收益: {result.annual_return:.2%}")

        print(f"\n【风险指标】")
        print(f"  最大回撤: {result.max_drawdown:.2%}")
        print(f"  回撤时长: {result.max_drawdown_duration}天")
        print(f"  波动率:   {result.volatility:.2%}")
        print(f"  夏普比率: {result.sharpe_ratio:.2f}")
        print(f"  索提诺:   {result.sortino_ratio:.2f}")
        print(f"  卡尔玛:   {result.calmar_ratio:.2f}")

        print(f"\n【交易指标】")
        print(f"  总交易数: {result.total_trades}")
        print(f"  胜率:     {result.win_rate:.1%}")
        print(f"  盈亏比:   {result.profit_factor:.2f}")
        print(f"  平均盈利: ${result.avg_win:,.0f}")
        print(f"  平均亏损: ${result.avg_loss:,.0f}")
        print(f"  平均持仓: {result.avg_holding_days:.1f}天")

        print("\n" + "=" * 60)


# 测试
if __name__ == "__main__":
    from ..signals.signal_generator import SignalGenerator

    # 生成模拟数据
    np.random.seed(42)
    n = 200

    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    dvol = 50 + np.random.randn(n).cumsum() * 2
    dvol = np.clip(dvol, 20, 100)

    dvol_series = pd.Series(dvol, index=dates)

    # 生成模拟信号
    predictions = np.random.randn(n) * 0.1
    df = pd.DataFrame({
        'time': dates,
        'dvol': dvol,
        'bb_squeeze_days': np.random.randint(0, 20, n),
        'vrp_30d': np.random.randn(n) * 5,
        'skew_zscore': np.random.randn(n),
        'term_inverted': np.random.randint(0, 2, n),
    })

    generator = SignalGenerator()
    signals = generator.generate_signals_batch(df, predictions)

    # 只保留非HOLD信号
    signals = [s for s in signals if s.signal_type != SignalType.HOLD]

    # 回测
    backtester = Backtester()
    result = backtester.run(signals, dvol_series)
    backtester.print_result(result)
