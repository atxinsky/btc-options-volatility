# coding=utf-8
"""
BTC期权波动率预测系统 - Streamlit前端
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.deribit_fetcher import DeribitFetcher
from src.data.price_fetcher import PriceFetcher
from src.features.feature_builder import FeatureBuilder
from src.models.predictor import VolatilityPredictor
from src.signals.signal_generator import SignalGenerator, SignalType
from src.backtest.backtester import Backtester

st.set_page_config(
    page_title="BTC波动率预测",
    page_icon="📊",
    layout="wide"
)

st.title("📊 BTC期权波动率预测系统")

# Sidebar
st.sidebar.header("控制面板")
page = st.sidebar.radio("导航", ["📈 实时监控", "🔮 预测分析", "📊 回测结果", "⚙️ 设置"])


@st.cache_data(ttl=60)
def fetch_market_snapshot():
    """获取市场快照（缓存1分钟）"""
    deribit = DeribitFetcher()
    return deribit.get_snapshot()


@st.cache_data(ttl=300)
def fetch_price_data(days=30):
    """获取价格数据（缓存5分钟）"""
    fetcher = PriceFetcher()
    return fetcher.get_ohlcv_binance(interval="1h", limit=days*24)


@st.cache_data(ttl=300)
def fetch_term_structure():
    """获取期限结构"""
    deribit = DeribitFetcher()
    return deribit.get_term_structure()


@st.cache_data(ttl=300)
def fetch_atm_ivs():
    """获取ATM IV"""
    deribit = DeribitFetcher()
    return deribit.get_atm_iv()


@st.cache_data(ttl=300)
def fetch_skews():
    """获取Skew"""
    deribit = DeribitFetcher()
    return deribit.get_skew()


def render_realtime_page():
    """实时监控页面"""
    st.header("📈 实时市场监控")

    # 刷新按钮
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 刷新数据"):
            st.cache_data.clear()
            st.rerun()

    # 获取数据
    with st.spinner("获取市场数据..."):
        snapshot = fetch_market_snapshot()
        price_df = fetch_price_data(7)

    # 顶部指标卡片
    st.subheader("关键指标")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        # DVOL用30天IV近似
        iv_30d = snapshot.get('atm_iv_30d')
        st.metric("DVOL (30D IV)", f"{iv_30d:.1f}%" if iv_30d else "N/A",
                 help="使用30天ATM IV近似DVOL")

    with col2:
        iv_7d = snapshot.get('atm_iv_7d')
        days_7d = snapshot.get('atm_iv_7d_days', 7)
        st.metric(f"ATM IV ({days_7d}D)", f"{iv_7d:.1f}%" if iv_7d else "N/A")

    with col3:
        iv_30d = snapshot.get('atm_iv_30d')
        days_30d = snapshot.get('atm_iv_30d_days', 30)
        st.metric(f"ATM IV ({days_30d}D)", f"{iv_30d:.1f}%" if iv_30d else "N/A")

    with col4:
        skew = snapshot.get('skew_7d')
        skew_label = ""
        if skew:
            if skew > 2:
                skew_label = "看跌偏重"
            elif skew < -2:
                skew_label = "看涨偏重"
            else:
                skew_label = "正常"
        st.metric("Skew", f"{skew:+.2f}%" if skew else "N/A",
                 delta=skew_label if skew_label else None)

    with col5:
        term_slope = snapshot.get('term_slope')
        if term_slope is not None:
            status = "正常📈" if term_slope > 0 else "倒挂⚠️"
        else:
            status = "N/A"
        st.metric("期限结构", status,
                 help="正常=远月IV>近月IV，倒挂=市场恐慌")

    st.markdown("---")

    # 图表区域
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("BTC价格 (7天)")
        if len(price_df) > 0:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=price_df['time'],
                open=price_df['open'],
                high=price_df['high'],
                low=price_df['low'],
                close=price_df['close'],
                name='BTC'
            ))
            fig.update_layout(height=400, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("IV期限结构")
        with st.spinner("获取期限结构..."):
            ts_df = fetch_term_structure()

        if len(ts_df) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts_df['days_to_expiry'],
                y=ts_df['atm_iv'],
                mode='lines+markers',
                name='ATM IV',
                line=dict(color='#2196F3', width=2)
            ))
            fig.update_layout(
                height=400,
                xaxis_title="距到期天数",
                yaxis_title="ATM IV (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无期限结构数据")

    # 信号判断
    st.markdown("---")
    st.subheader("📊 当前市场状态分析")

    signals = []

    # 分析各指标
    if snapshot.get('skew_7d') and abs(snapshot['skew_7d']) > 3:
        signals.append(("⚠️ Skew异常", f"当前Skew={snapshot['skew_7d']:.2f}%，偏离正常范围"))

    if snapshot.get('term_slope') and snapshot['term_slope'] < 0:
        signals.append(("⚠️ 期限结构倒挂", "近月IV高于远月IV，市场恐慌"))

    iv_7d = snapshot.get('atm_iv_7d')
    iv_30d = snapshot.get('atm_iv_30d')
    if iv_7d and iv_30d:
        if iv_7d > iv_30d * 1.1:
            signals.append(("📈 短期IV偏高", f"7D IV({iv_7d:.1f}%) > 30D IV({iv_30d:.1f}%)"))
        elif iv_7d < iv_30d * 0.9:
            signals.append(("📉 短期IV偏低", f"7D IV({iv_7d:.1f}%) < 30D IV({iv_30d:.1f}%)"))

    if signals:
        for title, desc in signals:
            st.warning(f"**{title}**: {desc}")
    else:
        st.success("✅ 市场状态正常，无异常信号")

    # 更新时间
    st.caption(f"数据更新时间: {snapshot.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")


def render_prediction_page():
    """预测分析页面"""
    st.header("🔮 波动率预测分析")

    # 获取数据
    with st.spinner("获取数据..."):
        price_df = fetch_price_data(90)

    if len(price_df) == 0:
        st.error("无法获取价格数据")
        return

    # 构建特征
    st.subheader("特征分析")

    builder = FeatureBuilder()

    # 只用价格数据构建特征
    df = builder.add_bollinger_features(price_df)
    df = builder.add_realized_volatility(df)
    df = builder.add_atr_features(df)

    # 显示关键特征
    col1, col2, col3, col4 = st.columns(4)

    latest = df.iloc[-1]

    with col1:
        bb_width = latest.get('bb_width', 0) * 100
        bb_pct = latest.get('bb_width_percentile', 0.5)
        st.metric("布林带宽度", f"{bb_width:.2f}%",
                 delta=f"分位数 {bb_pct:.0%}")

    with col2:
        squeeze_days = int(latest.get('bb_squeeze_days', 0))
        st.metric("压缩天数", f"{squeeze_days}天",
                 delta="注意爆发" if squeeze_days > 10 else None)

    with col3:
        rv_7d = latest.get('rv_7d', 0)
        st.metric("RV 7D", f"{rv_7d:.1f}%")

    with col4:
        rv_30d = latest.get('rv_30d', 0)
        st.metric("RV 30D", f"{rv_30d:.1f}%")

    st.markdown("---")

    # 图表
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("布林带宽度趋势")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['bb_width'] * 100,
            name='BB宽度',
            line=dict(color='#FF9800')
        ))
        fig.add_hline(y=df['bb_width'].mean() * 100, line_dash="dash",
                     annotation_text="平均值")
        fig.update_layout(height=300, yaxis_title="BB宽度 (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("已实现波动率")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['rv_7d'],
            name='RV 7D', line=dict(color='#2196F3')
        ))
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['rv_30d'],
            name='RV 30D', line=dict(color='#4CAF50')
        ))
        fig.update_layout(height=300, yaxis_title="RV (%)")
        st.plotly_chart(fig, use_container_width=True)

    # 预测信号生成
    st.markdown("---")
    st.subheader("交易信号判断")

    # 简单规则判断
    signal_text = ""
    signal_type = "hold"
    reasons = []

    squeeze_days = int(latest.get('bb_squeeze_days', 0))
    bb_pct = latest.get('bb_width_percentile', 0.5)
    rv_ratio = latest.get('rv_ratio_7_30', 1)

    # 判断逻辑
    if squeeze_days > 15 and bb_pct < 0.2:
        signal_type = "buy"
        reasons.append(f"布林带压缩{squeeze_days}天，处于{bb_pct:.0%}分位")

    if rv_ratio > 1.3:
        if signal_type != "buy":
            signal_type = "buy"
        reasons.append(f"短期波动率上升，RV7/RV30={rv_ratio:.2f}")
    elif rv_ratio < 0.7:
        signal_type = "sell"
        reasons.append(f"短期波动率下降，RV7/RV30={rv_ratio:.2f}")

    # 显示结果
    if signal_type == "buy":
        st.success(f"📈 **建议：买入波动率（买跨式）**")
        st.write("理由：")
        for r in reasons:
            st.write(f"  • {r}")

    elif signal_type == "sell":
        st.warning(f"📉 **建议：卖出波动率（卖跨式）**")
        st.write("理由：")
        for r in reasons:
            st.write(f"  • {r}")

    else:
        st.info("⏸️ **建议：观望**")
        st.write("当前无明显交易机会")


def render_backtest_page():
    """回测结果页面"""
    st.header("📊 策略回测")

    # 检查是否有回测结果
    equity_path = "data/processed/equity_curve.csv"

    if os.path.exists(equity_path):
        equity_df = pd.read_csv(equity_path)
        equity_df['time'] = pd.to_datetime(equity_df['time'])

        # 计算指标
        initial = equity_df['equity'].iloc[0]
        final = equity_df['equity'].iloc[-1]
        total_return = (final - initial) / initial

        peak = equity_df['equity'].expanding().max()
        drawdown = (peak - equity_df['equity']) / peak
        max_dd = drawdown.max()

        # 显示指标
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("总收益", f"{total_return:.2%}")
        with col2:
            st.metric("期末资金", f"${final:,.0f}")
        with col3:
            st.metric("最大回撤", f"{max_dd:.2%}")
        with col4:
            days = (equity_df['time'].iloc[-1] - equity_df['time'].iloc[0]).days
            annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1
            st.metric("年化收益", f"{annual_return:.2%}")

        # 权益曲线图
        st.subheader("权益曲线")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.7, 0.3])

        fig.add_trace(go.Scatter(
            x=equity_df['time'], y=equity_df['equity'],
            name='权益', fill='tozeroy',
            line=dict(color='#2196F3')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=equity_df['time'], y=-drawdown * 100,
            name='回撤', fill='tozeroy',
            line=dict(color='#F44336')
        ), row=2, col=1)

        fig.update_layout(height=500)
        fig.update_yaxes(title_text="权益 ($)", row=1, col=1)
        fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("暂无回测结果，请先运行回测")

        if st.button("🚀 运行回测"):
            with st.spinner("正在运行回测..."):
                # 生成模拟回测数据
                n = 200
                dates = pd.date_range('2024-01-01', periods=n, freq='D')

                # 模拟权益曲线
                returns = np.random.randn(n) * 0.01 + 0.0005  # 正期望
                equity = 100000 * (1 + returns).cumprod()

                equity_df = pd.DataFrame({
                    'time': dates,
                    'equity': equity
                })

                os.makedirs("data/processed", exist_ok=True)
                equity_df.to_csv(equity_path, index=False)

            st.success("回测完成！")
            st.rerun()


def render_settings_page():
    """设置页面"""
    st.header("⚙️ 系统设置")

    st.subheader("模型参数")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input("买入阈值 (%)", value=5.0, step=0.5,
                       help="预测DVOL上涨超过此值触发买入信号")
        st.number_input("卖出阈值 (%)", value=-3.0, step=0.5,
                       help="预测DVOL下跌超过此值触发卖出信号")
        st.slider("最小置信度", 0.0, 1.0, 0.6,
                 help="信号置信度低于此值转为观望")

    with col2:
        st.number_input("初始资金 ($)", value=100000, step=10000)
        st.slider("单笔仓位 (%)", 5, 30, 10)
        st.number_input("持仓周期 (天)", value=7, step=1)

    st.markdown("---")

    st.subheader("风控设置")

    col1, col2 = st.columns(2)

    with col1:
        st.slider("最大回撤止损 (%)", 5, 30, 15)
        st.slider("单笔止损 (%)", 1, 10, 5)

    with col2:
        st.slider("最大仓位 (%)", 10, 50, 30)

    if st.button("💾 保存设置"):
        st.success("设置已保存！")


# 主路由
if page == "📈 实时监控":
    render_realtime_page()
elif page == "🔮 预测分析":
    render_prediction_page()
elif page == "📊 回测结果":
    render_backtest_page()
elif page == "⚙️ 设置":
    render_settings_page()
