# coding=utf-8
"""
BTC价格数据获取模块
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import requests
import logging

logger = logging.getLogger(__name__)


class PriceFetcher:
    """BTC价格数据获取器"""

    def __init__(self):
        self.session = requests.Session()

    def get_ohlcv_binance(self, interval: str = "1h",
                          start_time: datetime = None,
                          end_time: datetime = None,
                          limit: int = 1000) -> pd.DataFrame:
        """
        从Binance获取BTC价格数据

        Parameters:
            interval: K线周期 (1m, 5m, 15m, 1h, 4h, 1d)
            start_time: 开始时间
            end_time: 结束时间
            limit: 最大返回数量

        Returns:
            DataFrame: [time, open, high, low, close, volume]
        """
        url = "https://api.binance.com/api/v3/klines"

        params = {
            "symbol": "BTCUSDT",
            "interval": interval,
            "limit": limit
        }

        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            return df.sort_values('time').reset_index(drop=True)

        except Exception as e:
            logger.error(f"Binance数据获取失败: {e}")
            return pd.DataFrame()

    def get_ohlcv_deribit(self, resolution: str = "60",
                          start_time: datetime = None,
                          end_time: datetime = None) -> pd.DataFrame:
        """
        从Deribit获取BTC价格数据

        Parameters:
            resolution: K线周期 (1, 3, 5, 10, 15, 30, 60, 120, 180, 360, 720, 1D)
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            DataFrame: [time, open, high, low, close, volume]
        """
        url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"

        if not start_time:
            start_time = datetime.now() - timedelta(days=30)
        if not end_time:
            end_time = datetime.now()

        params = {
            "instrument_name": "BTC-PERPETUAL",
            "resolution": resolution,
            "start_timestamp": int(start_time.timestamp() * 1000),
            "end_timestamp": int(end_time.timestamp() * 1000)
        }

        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if 'result' not in data:
                return pd.DataFrame()

            result = data['result']
            df = pd.DataFrame({
                'time': pd.to_datetime(result['ticks'], unit='ms'),
                'open': result['open'],
                'high': result['high'],
                'low': result['low'],
                'close': result['close'],
                'volume': result['volume']
            })

            return df.sort_values('time').reset_index(drop=True)

        except Exception as e:
            logger.error(f"Deribit价格数据获取失败: {e}")
            return pd.DataFrame()

    def get_historical_data(self, days: int = 365,
                            interval: str = "1h",
                            source: str = "binance") -> pd.DataFrame:
        """
        获取历史数据 (分批获取以突破限制)

        Parameters:
            days: 获取多少天的数据
            interval: K线周期
            source: 数据源 (binance, deribit)

        Returns:
            完整的历史数据
        """
        all_data = []
        end_time = datetime.now()

        if source == "binance":
            # Binance每次最多1000条
            interval_hours = {
                "1h": 1, "4h": 4, "1d": 24,
                "15m": 0.25, "5m": 0.0833
            }.get(interval, 1)

            batch_hours = int(1000 * interval_hours)

            while True:
                start_time = end_time - timedelta(hours=batch_hours)

                if start_time < datetime.now() - timedelta(days=days):
                    start_time = datetime.now() - timedelta(days=days)

                df = self.get_ohlcv_binance(
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time,
                    limit=1000
                )

                if len(df) == 0:
                    break

                all_data.append(df)
                end_time = df['time'].min() - timedelta(hours=1)

                if end_time < datetime.now() - timedelta(days=days):
                    break

                logger.info(f"已获取到 {end_time.strftime('%Y-%m-%d')}")

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result.drop_duplicates(subset=['time'])
            result = result.sort_values('time').reset_index(drop=True)
            return result

        return pd.DataFrame()


# 测试
if __name__ == "__main__":
    fetcher = PriceFetcher()

    print("=== Binance 1H数据 ===")
    df = fetcher.get_ohlcv_binance(interval="1h", limit=100)
    print(df.tail())

    print("\n=== Deribit 1H数据 ===")
    df = fetcher.get_ohlcv_deribit(resolution="60")
    print(df.tail())
