# BTC期权波动率预测系统

预测Deribit DVOL（比特币波动率指数）变化，生成期权波动率交易信号。

## 核心策略

```
预测DVOL下跌/横盘 → 卖波动率（卖跨式）→ 吃时间价值
预测DVOL上涨     → 买波动率（买跨式）→ 赚IV上升
```

## 四大预测依据

| 依据 | 指标 | 逻辑 |
|------|------|------|
| **布林带压缩** | bb_width, bb_squeeze_days | 横盘越久，爆发概率越大 |
| **IV-RV对比** | vrp_30d, iv_rv_ratio | IV>RV期权贵适合卖，反之买 |
| **期限结构** | term_spread, term_inverted | 倒挂=恐慌，正常=稳定 |
| **Skew变化** | skew_zscore, skew_change | 突变=有人在买保险 |

## 项目结构

```
btc-options-volatility/
├── config/
│   └── config.yaml          # 配置文件
├── src/
│   ├── data/
│   │   ├── deribit_fetcher.py   # Deribit数据获取
│   │   └── price_fetcher.py     # 价格数据获取
│   ├── features/
│   │   └── feature_builder.py   # 特征工程
│   ├── models/
│   │   └── predictor.py         # 预测模型
│   ├── signals/
│   │   └── signal_generator.py  # 信号生成
│   └── backtest/
│       └── backtester.py        # 回测框架
├── data/
│   ├── raw/                 # 原始数据
│   └── processed/           # 处理后数据
├── notebooks/               # 分析notebook
├── main.py                  # 主程序
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行

```bash
# 完整流程（获取数据→训练→预测→回测）
python main.py --mode all --days 90

# 仅训练
python main.py --mode train

# 仅预测
python main.py --mode predict

# 仅回测
python main.py --mode backtest

# 实时监控
python main.py --mode live
```

## 数据来源

- **DVOL/IV/Skew**: Deribit API (免费，无需key)
- **BTC价格**: Binance API

## 模型说明

### 特征 (共40+)

**价格特征**:
- 布林带宽度、压缩天数、位置
- 已实现波动率 (7/14/30/60天)
- ATR变化

**期权特征**:
- DVOL及其变化
- ATM IV (7天/30天)
- VRP (IV-RV差)
- Skew及Z-score
- 期限结构斜率

### 预测目标

未来7天DVOL变化率

### 信号逻辑

```python
if 预测变化 > 5% and 置信度 > 60%:
    信号 = 买波动率

    # 加成条件
    if 布林带压缩 > 10天: 置信度 += 20%
    if VRP < -5%: 置信度 += 15%
    if Skew突变: 置信度 += 15%
    if 期限倒挂: 置信度 += 10%

elif 预测变化 < -3% and 置信度 > 60%:
    信号 = 卖波动率

    if VRP > 10%: 置信度 += 15%

else:
    信号 = 观望
```

## 回测说明

- 初始资金: $100,000
- 单笔仓位: 10% (根据置信度调整)
- 持仓周期: 7天
- 止损: 单笔-5%，总回撤-15%
- 手续费: 0.03%
- 滑点: 0.1%

### 盈亏计算 (简化模型)

```python
# 买波动率：DVOL涨赚钱
pnl = (exit_dvol - entry_dvol) / entry_dvol * vega_sensitivity * position_size

# 卖波动率：DVOL跌赚钱
pnl = -(exit_dvol - entry_dvol) / entry_dvol * vega_sensitivity * position_size
```

实际交易中需要根据具体期权希腊字母计算。

## 注意事项

1. **数据限制**: Deribit历史数据有限制，建议自行长期收集
2. **简化模型**: 回测使用简化的Vega敏感度，实际交易更复杂
3. **风险提示**: 期权卖方有无限风险，务必做好风控
4. **市场变化**: 模型需要定期重训练

## 下一步

- [ ] 对接实际期权定价计算
- [ ] 添加更多数据源（链上数据、情绪指标）
- [ ] 实现自动化交易
- [ ] 添加更多模型（Transformer）

## License

MIT
