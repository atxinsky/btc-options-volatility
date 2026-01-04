# coding=utf-8
"""
波动率预测模型
支持多种模型：LightGBM、LSTM、集成模型
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
import pickle
import os

logger = logging.getLogger(__name__)


class VolatilityPredictor:
    """波动率预测器"""

    def __init__(self, model_type: str = "lgbm", config: dict = None):
        """
        Parameters:
            model_type: 模型类型 (lgbm, lstm, ensemble)
            config: 模型配置
        """
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.feature_columns = None

    def _create_lgbm_model(self):
        """创建LightGBM模型"""
        try:
            import lightgbm as lgb

            params = self.config.get('lgbm', {})
            return lgb.LGBMRegressor(
                n_estimators=params.get('n_estimators', 500),
                learning_rate=params.get('learning_rate', 0.05),
                max_depth=params.get('max_depth', 6),
                num_leaves=params.get('num_leaves', 31),
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        except ImportError:
            logger.error("需要安装lightgbm: pip install lightgbm")
            return None

    def _create_lstm_model(self, input_shape: Tuple[int, int]):
        """创建LSTM模型"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
            from tensorflow.keras.optimizers import Adam

            params = self.config.get('lstm', {})
            units = params.get('units', [64, 32])
            dropout = params.get('dropout', 0.2)

            model = Sequential()

            # 第一层LSTM
            model.add(Bidirectional(
                LSTM(units[0], return_sequences=len(units) > 1),
                input_shape=input_shape
            ))
            model.add(Dropout(dropout))

            # 额外的LSTM层
            for i, u in enumerate(units[1:]):
                return_seq = i < len(units) - 2
                model.add(Bidirectional(LSTM(u, return_sequences=return_seq)))
                model.add(Dropout(dropout))

            # 输出层
            model.add(Dense(1))

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            return model

        except ImportError:
            logger.error("需要安装tensorflow: pip install tensorflow")
            return None

    def _prepare_lstm_data(self, X: np.ndarray, y: np.ndarray = None,
                           lookback: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """准备LSTM序列数据"""
        X_seq = []
        y_seq = []

        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            if y is not None:
                y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq) if y is not None else None

    def train(self, df: pd.DataFrame, feature_columns: List[str],
              target_column: str = 'target_dvol_change',
              val_ratio: float = 0.15) -> Dict:
        """
        训练模型

        Parameters:
            df: 特征数据
            feature_columns: 特征列名
            target_column: 目标列名
            val_ratio: 验证集比例

        Returns:
            训练结果字典
        """
        self.feature_columns = feature_columns

        # 准备数据
        X = df[feature_columns].values
        y = df[target_column].values

        # 数据标准化
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 划分训练集和验证集
        val_size = int(len(X) * val_ratio)
        X_train, X_val = X_scaled[:-val_size], X_scaled[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]

        results = {'model_type': self.model_type}

        if self.model_type == 'lgbm':
            self.model = self._create_lgbm_model()
            if self.model is None:
                return {'error': 'Failed to create LightGBM model'}

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    # lgb.early_stopping(50, verbose=False)
                ]
            )

            # 特征重要性
            results['feature_importance'] = dict(zip(
                feature_columns,
                self.model.feature_importances_
            ))

        elif self.model_type == 'lstm':
            lookback = self.config.get('lstm', {}).get('lookback', 30)
            X_train_seq, y_train_seq = self._prepare_lstm_data(X_train, y_train, lookback)
            X_val_seq, y_val_seq = self._prepare_lstm_data(X_val, y_val, lookback)

            self.model = self._create_lstm_model((lookback, len(feature_columns)))
            if self.model is None:
                return {'error': 'Failed to create LSTM model'}

            from tensorflow.keras.callbacks import EarlyStopping

            history = self.model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=self.config.get('lstm', {}).get('epochs', 100),
                batch_size=self.config.get('lstm', {}).get('batch_size', 32),
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                verbose=1
            )

            results['history'] = history.history

        # 评估
        y_pred_train = self.predict(df.iloc[:-val_size][feature_columns])
        y_pred_val = self.predict(df.iloc[-val_size:][feature_columns])

        results['train_rmse'] = np.sqrt(np.mean((y_train[-len(y_pred_train):] - y_pred_train) ** 2))
        results['val_rmse'] = np.sqrt(np.mean((y_val[-len(y_pred_val):] - y_pred_val) ** 2))

        # 方向准确率
        if len(y_pred_val) > 0 and len(y_val) > 0:
            y_val_trimmed = y_val[-len(y_pred_val):]
            results['val_direction_accuracy'] = np.mean(
                (y_pred_val > 0) == (y_val_trimmed > 0)
            )

        logger.info(f"训练完成 - Train RMSE: {results['train_rmse']:.4f}, "
                   f"Val RMSE: {results['val_rmse']:.4f}")

        return results

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        预测

        Parameters:
            df: 特征数据

        Returns:
            预测值数组
        """
        if self.model is None:
            raise ValueError("模型未训练")

        X = df[self.feature_columns].values if isinstance(df, pd.DataFrame) else df
        X_scaled = self.scaler.transform(X)

        if self.model_type == 'lstm':
            lookback = self.config.get('lstm', {}).get('lookback', 30)
            X_seq, _ = self._prepare_lstm_data(X_scaled, None, lookback)
            if len(X_seq) == 0:
                return np.array([])
            return self.model.predict(X_seq, verbose=0).flatten()
        else:
            return self.model.predict(X_scaled)

    def predict_with_confidence(self, df: pd.DataFrame,
                                n_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        带置信区间的预测 (使用Bootstrap或MC Dropout)

        Returns:
            (预测均值, 预测标准差)
        """
        if self.model_type == 'lgbm':
            # LightGBM使用Bootstrap
            predictions = []
            X = df[self.feature_columns].values
            X_scaled = self.scaler.transform(X)

            for _ in range(n_iterations):
                # 添加小噪声
                X_noisy = X_scaled + np.random.normal(0, 0.01, X_scaled.shape)
                pred = self.model.predict(X_noisy)
                predictions.append(pred)

            predictions = np.array(predictions)
            return predictions.mean(axis=0), predictions.std(axis=0)

        elif self.model_type == 'lstm':
            # LSTM使用MC Dropout
            predictions = []
            lookback = self.config.get('lstm', {}).get('lookback', 30)
            X = df[self.feature_columns].values
            X_scaled = self.scaler.transform(X)
            X_seq, _ = self._prepare_lstm_data(X_scaled, None, lookback)

            if len(X_seq) == 0:
                return np.array([]), np.array([])

            # 启用训练模式以使用Dropout
            for _ in range(n_iterations):
                pred = self.model(X_seq, training=True).numpy().flatten()
                predictions.append(pred)

            predictions = np.array(predictions)
            return predictions.mean(axis=0), predictions.std(axis=0)

        return self.predict(df), np.zeros(len(df))

    def save(self, filepath: str):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        save_dict = {
            'model_type': self.model_type,
            'config': self.config,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }

        if self.model_type == 'lgbm':
            save_dict['model'] = self.model
            with open(filepath, 'wb') as f:
                pickle.dump(save_dict, f)

        elif self.model_type == 'lstm':
            with open(filepath, 'wb') as f:
                pickle.dump(save_dict, f)
            self.model.save(filepath.replace('.pkl', '_keras.h5'))

        logger.info(f"模型已保存到: {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)

        self.model_type = save_dict['model_type']
        self.config = save_dict['config']
        self.scaler = save_dict['scaler']
        self.feature_columns = save_dict['feature_columns']

        if self.model_type == 'lgbm':
            self.model = save_dict['model']

        elif self.model_type == 'lstm':
            from tensorflow.keras.models import load_model
            self.model = load_model(filepath.replace('.pkl', '_keras.h5'))

        logger.info(f"模型已加载: {filepath}")


# 测试
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.randn(n),
        'target_dvol_change': np.random.randn(n) * 0.1
    })

    feature_cols = ['feature1', 'feature2', 'feature3']

    # 测试LightGBM
    print("=== LightGBM ===")
    predictor = VolatilityPredictor(model_type='lgbm')
    results = predictor.train(df, feature_cols)
    print(f"Val RMSE: {results.get('val_rmse', 'N/A')}")
    print(f"Direction Accuracy: {results.get('val_direction_accuracy', 'N/A')}")

    # 预测
    pred = predictor.predict(df.tail(10))
    print(f"Predictions: {pred}")
