# coding=utf-8
"""
Deribit数据获取模块
获取DVOL、期权IV、Skew、期限结构等数据
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeribitFetcher:
    """Deribit期权数据获取器"""

    BASE_URL = "https://www.deribit.com/api/v2"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })

    def _request(self, endpoint: str, params: dict = None) -> dict:
        """发送API请求"""
        url = f"{self.BASE_URL}{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if 'result' in data:
                return data['result']
            return data
        except Exception as e:
            logger.error(f"API请求失败: {e}")
            return None

    # ==================== DVOL指数 ====================

    def get_dvol_current(self) -> Optional[float]:
        """获取当前DVOL值"""
        result = self._request("/public/get_index_price", {
            "index_name": "btc_usd"
        })
        # DVOL需要单独接口
        dvol_result = self._request("/public/get_volatility_index_data", {
            "currency": "BTC",
            "resolution": "1"  # 1分钟
        })
        if dvol_result and 'data' in dvol_result:
            return dvol_result['data'][-1][1] if dvol_result['data'] else None
        return None

    def get_dvol_history(self, start_time: datetime, end_time: datetime,
                         resolution: str = "60") -> pd.DataFrame:
        """
        获取DVOL历史数据

        Parameters:
            start_time: 开始时间
            end_time: 结束时间
            resolution: K线周期 (1, 60, 1D)

        Returns:
            DataFrame with columns: [time, open, high, low, close]
        """
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)

        result = self._request("/public/get_volatility_index_data", {
            "currency": "BTC",
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "resolution": resolution
        })

        if result and 'data' in result:
            df = pd.DataFrame(result['data'],
                             columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop('timestamp', axis=1)
            df = df.sort_values('time').reset_index(drop=True)
            return df

        return pd.DataFrame()

    # ==================== 期权链数据 ====================

    def get_instruments(self, kind: str = "option", expired: bool = False) -> List[dict]:
        """获取所有BTC期权合约"""
        result = self._request("/public/get_instruments", {
            "currency": "BTC",
            "kind": kind,
            "expired": str(expired).lower()
        })
        return result if result else []

    def get_option_chain(self, expiry: str = None) -> pd.DataFrame:
        """
        获取期权链数据

        Returns:
            DataFrame with IV, delta, gamma for all strikes
        """
        instruments = self.get_instruments()
        if not instruments:
            return pd.DataFrame()

        # 筛选特定到期日
        if expiry:
            instruments = [i for i in instruments if expiry in i['instrument_name']]

        # 获取每个合约的ticker
        options_data = []
        for inst in instruments[:100]:  # 限制数量避免过多请求
            ticker = self._request("/public/ticker", {
                "instrument_name": inst['instrument_name']
            })
            if ticker:
                options_data.append({
                    'instrument': inst['instrument_name'],
                    'strike': inst['strike'],
                    'option_type': inst['option_type'],
                    'expiry': inst['expiration_timestamp'],
                    'mark_iv': ticker.get('mark_iv', 0),
                    'bid_iv': ticker.get('bid_iv', 0),
                    'ask_iv': ticker.get('ask_iv', 0),
                    'delta': ticker.get('greeks', {}).get('delta', 0),
                    'gamma': ticker.get('greeks', {}).get('gamma', 0),
                    'vega': ticker.get('greeks', {}).get('vega', 0),
                    'theta': ticker.get('greeks', {}).get('theta', 0),
                    'underlying_price': ticker.get('underlying_price', 0),
                    'open_interest': ticker.get('open_interest', 0),
                    'volume': ticker.get('stats', {}).get('volume', 0),
                })
            time.sleep(0.05)  # 避免请求过快

        return pd.DataFrame(options_data)

    # ==================== ATM IV ====================

    def get_atm_iv(self) -> Dict[str, float]:
        """
        获取各到期日的ATM隐含波动率

        Returns:
            {expiry_date: atm_iv}
        """
        instruments = self.get_instruments()
        if not instruments:
            return {}

        # 获取当前BTC价格
        index = self._request("/public/get_index_price", {"index_name": "btc_usd"})
        btc_price = index.get('index_price', 0) if index else 0

        # 按到期日分组
        expiries = {}
        for inst in instruments:
            exp_ts = inst['expiration_timestamp']
            exp_date = datetime.fromtimestamp(exp_ts / 1000).strftime('%Y-%m-%d')
            if exp_date not in expiries:
                expiries[exp_date] = []
            expiries[exp_date].append(inst)

        # 获取每个到期日的ATM IV
        atm_ivs = {}
        for exp_date, insts in expiries.items():
            # 找到最接近ATM的行权价
            calls = [i for i in insts if i['option_type'] == 'call']
            if not calls:
                continue

            atm_call = min(calls, key=lambda x: abs(x['strike'] - btc_price))

            ticker = self._request("/public/ticker", {
                "instrument_name": atm_call['instrument_name']
            })
            if ticker and ticker.get('mark_iv'):
                atm_ivs[exp_date] = ticker['mark_iv']
            time.sleep(0.05)

        return atm_ivs

    # ==================== Skew ====================

    def get_skew(self, expiry: str = None) -> Dict[str, float]:
        """
        获取波动率偏斜 (25D Put IV - 25D Call IV)

        正值 = Put比Call贵 = 市场担心下跌
        负值 = Call比Put贵 = 市场看涨

        Returns:
            {expiry_date: skew_value}
        """
        instruments = self.get_instruments()
        if not instruments:
            return {}

        # 按到期日分组
        expiries = {}
        for inst in instruments:
            exp_ts = inst['expiration_timestamp']
            exp_date = datetime.fromtimestamp(exp_ts / 1000).strftime('%Y-%m-%d')
            if exp_date not in expiries:
                expiries[exp_date] = {'calls': [], 'puts': []}
            if inst['option_type'] == 'call':
                expiries[exp_date]['calls'].append(inst)
            else:
                expiries[exp_date]['puts'].append(inst)

        skews = {}
        for exp_date, opts in expiries.items():
            if not opts['calls'] or not opts['puts']:
                continue

            # 获取所有期权的delta和IV
            call_25d_iv = None
            put_25d_iv = None

            for call in opts['calls'][:20]:
                ticker = self._request("/public/ticker", {
                    "instrument_name": call['instrument_name']
                })
                if ticker:
                    delta = ticker.get('greeks', {}).get('delta', 0)
                    if 0.20 <= delta <= 0.30:  # 约25D
                        call_25d_iv = ticker.get('mark_iv', 0)
                        break
                time.sleep(0.03)

            for put in opts['puts'][:20]:
                ticker = self._request("/public/ticker", {
                    "instrument_name": put['instrument_name']
                })
                if ticker:
                    delta = ticker.get('greeks', {}).get('delta', 0)
                    if -0.30 <= delta <= -0.20:  # 约-25D
                        put_25d_iv = ticker.get('mark_iv', 0)
                        break
                time.sleep(0.03)

            if call_25d_iv and put_25d_iv:
                skews[exp_date] = put_25d_iv - call_25d_iv

        return skews

    # ==================== 期限结构 ====================

    def get_term_structure(self) -> pd.DataFrame:
        """
        获取IV期限结构

        Returns:
            DataFrame: [expiry, days_to_expiry, atm_iv]
        """
        atm_ivs = self.get_atm_iv()
        if not atm_ivs:
            return pd.DataFrame()

        now = datetime.now()
        data = []
        for exp_date, iv in atm_ivs.items():
            exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
            days = (exp_dt - now).days
            if days > 0:
                data.append({
                    'expiry': exp_date,
                    'days_to_expiry': days,
                    'atm_iv': iv
                })

        df = pd.DataFrame(data)
        if len(df) > 0:
            df = df.sort_values('days_to_expiry').reset_index(drop=True)
        return df

    def get_term_structure_slope(self) -> Optional[float]:
        """
        计算期限结构斜率

        正值 = 正常结构 (远月IV > 近月IV)
        负值 = 倒挂 (近月IV > 远月IV) = 市场恐慌

        Returns:
            斜率值
        """
        ts = self.get_term_structure()
        if len(ts) < 2:
            return None

        # 用最近两个到期日计算斜率
        near = ts.iloc[0]
        far = ts.iloc[min(3, len(ts)-1)]  # 第4个或最后一个

        slope = (far['atm_iv'] - near['atm_iv']) / (far['days_to_expiry'] - near['days_to_expiry'])
        return slope

    # ==================== 综合数据 ====================

    def get_snapshot(self) -> Dict:
        """
        获取当前市场快照

        Returns:
            {
                'dvol': float,
                'atm_iv_7d': float,
                'atm_iv_30d': float,
                'skew_7d': float,
                'skew_30d': float,
                'term_slope': float,
                'timestamp': datetime
            }
        """
        logger.info("获取市场快照...")

        snapshot = {
            'timestamp': datetime.now()
        }

        # DVOL
        snapshot['dvol'] = self.get_dvol_current()

        # ATM IV - 找最接近7天和30天的
        atm_ivs = self.get_atm_iv()
        now = datetime.now()

        iv_by_days = {}
        for exp_date, iv in atm_ivs.items():
            exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
            days = (exp_dt - now).days
            if days > 0:
                iv_by_days[days] = iv

        # 找最接近7天的
        if iv_by_days:
            closest_7d = min(iv_by_days.keys(), key=lambda x: abs(x - 7))
            snapshot['atm_iv_7d'] = iv_by_days[closest_7d]
            snapshot['atm_iv_7d_days'] = closest_7d

            # 找最接近30天的
            closest_30d = min(iv_by_days.keys(), key=lambda x: abs(x - 30))
            snapshot['atm_iv_30d'] = iv_by_days[closest_30d]
            snapshot['atm_iv_30d_days'] = closest_30d

        # Skew - 同样逻辑
        skews = self.get_skew()
        skew_by_days = {}
        for exp_date, skew in skews.items():
            exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
            days = (exp_dt - now).days
            if days > 0:
                skew_by_days[days] = skew

        if skew_by_days:
            closest_7d = min(skew_by_days.keys(), key=lambda x: abs(x - 7))
            snapshot['skew_7d'] = skew_by_days[closest_7d]

            closest_30d = min(skew_by_days.keys(), key=lambda x: abs(x - 30))
            snapshot['skew_30d'] = skew_by_days[closest_30d]

        # 期限结构
        snapshot['term_slope'] = self.get_term_structure_slope()

        return snapshot


# 测试
if __name__ == "__main__":
    fetcher = DeribitFetcher()

    print("=== DVOL ===")
    dvol = fetcher.get_dvol_current()
    print(f"当前DVOL: {dvol}")

    print("\n=== ATM IV ===")
    atm_ivs = fetcher.get_atm_iv()
    for exp, iv in list(atm_ivs.items())[:5]:
        print(f"{exp}: {iv:.2f}%")

    print("\n=== Skew ===")
    skews = fetcher.get_skew()
    for exp, skew in list(skews.items())[:5]:
        print(f"{exp}: {skew:+.2f}%")

    print("\n=== 期限结构 ===")
    ts = fetcher.get_term_structure()
    print(ts.head())

    print("\n=== 市场快照 ===")
    snapshot = fetcher.get_snapshot()
    for k, v in snapshot.items():
        print(f"{k}: {v}")
