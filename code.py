DATA_SOURCE = 'synthetic'  # options: 'synthetic' or 'csv'
PATH_TO_CSV = 'sensor_data.csv'  # used if DATA_SOURCE == 'csv'
RANDOM_SEED = 42
SYNTHETIC_LENGTH = 20000  # timestamps
N_SENSORS = 3
INJECTED_ANOMALY_RATIO = 0.01

# ====== IMPORTS ======
import os
import math
import json
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from statsmodels.tsa.seasonal import seasonal_decompose

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)

# ====== UTILITIES ======

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ====== DATA GENERATION / LOADING ======

def generate_synthetic_multivariate(length=10000, n_sensors=3, anomaly_ratio=0.01, seed=42):
    np.random.seed(seed)
    timestamps = pd.date_range(start='2024-01-01', periods=length, freq='T')  # minute frequency
    data = pd.DataFrame({'timestamp': timestamps})

    base_frequencies = np.linspace(0.0005, 0.005, n_sensors)
    for i in range(n_sensors):
        # smooth sinusoidal baseline + low-frequency trend + noise
        freq = base_frequencies[i]
        trend = 0.00005 * np.arange(length)  # slow linear trend
        seasonal = 2.0 * np.sin(np.arange(length) * 2 * np.pi * freq)
        noise = 0.5 * np.random.randn(length)
        sensor = 10 + trend + seasonal + noise
        data[f'sensor_{i+1}'] = sensor

    # Inject anomalies: spikes, level shifts, and dropout
    n_anom = max(1, int(length * anomaly_ratio))
    anomalies = []
    for _ in range(n_anom):
        start = np.random.randint(0, length - 50)
        anom_type = np.random.choice(['spike', 'shift', 'dropout'])
        if anom_type == 'spike':
            idx = start + np.random.randint(0, 50)
            sensor_id = np.random.choice(n_sensors)
            magnitude = np.random.uniform(8, 20)
            data.at[idx, f'sensor_{sensor_id+1}'] += magnitude
            anomalies.append({'idx': idx, 'sensor': sensor_id+1, 'type': 'spike'})
        elif anom_type == 'shift':
            length_shift = np.random.randint(10, 200)
            sensor_id = np.random.choice(n_sensors)
            magnitude = np.random.uniform(-6, 6)
            data.loc[start:start+length_shift, f'sensor_{sensor_id+1}'] += magnitude
            anomalies.append({'start': start, 'end': start+length_shift, 'sensor': sensor_id+1, 'type': 'shift'})
        else:  # dropout
            length_drop = np.random.randint(1, 20)
            sensor_id = np.random.choice(n_sensors)
            idxs = list(range(start, start+length_drop))
            data.loc[idxs, f'sensor_{sensor_id+1}'] = np.nan
            anomalies.append({'start': start, 'end': start+length_drop, 'sensor': sensor_id+1, 'type': 'dropout'})

    # Create label column for evaluation (1 = anomaly present in any sensor for that timestamp)
    labels = np.zeros(length, dtype=int)
    for a in anomalies:
        if a['type'] == 'spike':
            labels[a['idx']] = 1
        else:
            labels[a['start']:a['end']+1] = 1
    data['is_anomaly'] = labels
    return data, anomalies


def load_csv(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df


# ====== DATA PREPARATION & EDA ======

def prepare_and_explore(df, sensors_cols, show_plots=True, nrows_preview=2000):
    logger.info('Starting EDA and data cleaning')

    # Basic info
    logger.info(f'Dataset shape: {df.shape}')
    logger.info('Columns: %s', df.columns.tolist())

    # check timestamp ordering
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Missing values
    missing_summary = df[sensors_cols].isna().sum()
    logger.info('Missing values per sensor:\n%s', missing_summary)

    # Simple imputation strategy: forward-fill then backward-fill, then median for remaining
    df[sensors_cols] = df[sensors_cols].ffill().bfill()
    df[sensors_cols] = df[sensors_cols].fillna(df[sensors_cols].median())

    # Outlier detection (preliminary): Z-score
    z_scores = (df[sensors_cols] - df[sensors_cols].mean()) / (df[sensors_cols].std() + 1e-9)
    extreme = (np.abs(z_scores) > 6).any(axis=1)
    logger.info('Number of extreme z-score rows: %d', extreme.sum())
    # We will keep them; some extreme events may be anomalies we want to detect

    # Visual EDA
    if show_plots:
        num_plot = min(len(sensors_cols), 4)
        plt.figure(figsize=(12, 3 * num_plot))
        for i, col in enumerate(sensors_cols[:num_plot]):
            plt.subplot(num_plot, 1, i+1)
            plt.plot(df['timestamp'][:nrows_preview], df[col][:nrows_preview])
            plt.title(f'Time series preview: {col}')
        plt.tight_layout()
        plt.show()

        # Correlation
        plt.figure(figsize=(6, 5))
        sns.heatmap(df[sensors_cols].corr(), annot=True, fmt='.2f')
        plt.title('Sensor correlation')
        plt.show()

    # Document data quality issues in a dict
    dq_issues = {
        'missing_values_before_imputation': missing_summary.to_dict(),
        'rows_with_extreme_zscore': int(extreme.sum())
    }

    return df, dq_issues


# ====== FEATURE ENGINEERING ======

def create_features(df, sensors_cols, windows=[5, 15, 60]):
    """Create rolling stats and lag features. windows are in number of observations (minutes for synthetic)."""
    logger.info('Creating features: rolling mean/std and lags')
    feat_df = df.copy()
    for col in sensors_cols:
        # rolling statistics
        for w in windows:
            feat_df[f'{col}_roll_mean_{w}'] = feat_df[col].rolling(window=w, min_periods=1, center=False).mean()
            feat_df[f'{col}_roll_std_{w}'] = feat_df[col].rolling(window=w, min_periods=1, center=False).std().fillna(0)
        # lags
        for lag in [1, 5, 15]:
            feat_df[f'{col}_lag_{lag}'] = feat_df[col].shift(lag).fillna(method='bfill')
        # trend indicator: difference with rolling mean
        feat_df[f'{col}_trend_1'] = feat_df[col] - feat_df[f'{col}_roll_mean_{windows[0]}']

    # Optionally: seasonal decomposition features (for longer series might be heavy)
    try:
        # apply seasonal_decompose on first sensor to get seasonal and resid as global features
        res = seasonal_decompose(df[sensors_cols[0]].values, period=1440, model='additive', extrapolate_trend='freq')
        feat_df['seasonal_approx'] = res.seasonal
        feat_df['resid_approx'] = res.resid
    except Exception as e:
        logger.warning('Seasonal decomposition skipped or failed: %s', e)
        feat_df['seasonal_approx'] = 0
        feat_df['resid_approx'] = 0

    return feat_df


# ====== SCALING ======

def scale_features(X_train, X_test=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return scaler, X_train_scaled, X_test_scaled


# ====== MODEL 1: Isolation Forest (unsupervised/statistical-like) ======

def train_isolation_forest(X, n_estimators=200, contamination='auto', random_state=42):
    logger.info('Training IsolationForest: n_estimators=%s contamination=%s', n_estimators, contamination)
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    iso.fit(X)
    # anomaly score: negative is more anomalous (scikit-learn uses negative outlier factor style)
    scores = -iso.decision_function(X)
    preds = iso.predict(X)
    # convert to 0 (normal) / 1 (anomaly)
    preds_binary = (preds == -1).astype(int)
    return iso, scores, preds_binary


# ====== MODEL 2: LSTM Autoencoder (deep learning) ======

def build_lstm_autoencoder(n_timesteps, n_features, latent_dim=8):
    # Encoder-Decoder LSTM autoencoder for multivariate time series
    inputs = layers.Input(shape=(n_timesteps, n_features))
    encoded = layers.LSTM(64, return_sequences=True)(inputs)
    encoded = layers.LSTM(latent_dim, return_sequences=False)(encoded)
    # Repeat
    decoded = layers.RepeatVector(n_timesteps)(encoded)
    decoded = layers.LSTM(latent_dim, return_sequences=True)(decoded)
    decoded = layers.LSTM(64, return_sequences=True)(decoded)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(decoded)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def create_sequences(df_values, seq_len=30):
    # df_values: numpy array shape (n_samples, n_features)
    sequences = []
    for i in range(len(df_values) - seq_len + 1):
        sequences.append(df_values[i:i+seq_len])
    return np.array(sequences)


# ====== EVALUATION & VISUALIZATION ======

def evaluate_binary(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {'precision': float(p), 'recall': float(r), 'f1': float(f)}


def plot_anomalies(df, timestamps_col, value_col, anomaly_mask, title='Anomalies'):
    plt.figure(figsize=(15, 4))
    plt.plot(df[timestamps_col], df[value_col], label=value_col)
    plt.scatter(df.loc[anomaly_mask, timestamps_col], df.loc[anomaly_mask, value_col], color='red', s=10, label='Anomaly')
    plt.title(title)
    plt.legend()
    plt.show()


# ====== MAIN PIPELINE ======

def main():
    ensure_dir('models')

    if DATA_SOURCE == 'synthetic':
        df, injected_anomalies = generate_synthetic_multivariate(length=SYNTHETIC_LENGTH, n_sensors=N_SENSORS, anomaly_ratio=INJECTED_ANOMALY_RATIO, seed=RANDOM_SEED)
        logger.info('Generated synthetic data with %d injected anomaly segments/events', len(injected_anomalies))
    else:
        df = load_csv(PATH_TO_CSV)
        injected_anomalies = None

    sensors_cols = [c for c in df.columns if c.startswith('sensor_')]

    # EDA & cleaning
    df_clean, dq_issues = prepare_and_explore(df, sensors_cols, show_plots=True)
    logger.info('Data quality issues: %s', json.dumps(dq_issues, indent=2))

    # Feature engineering
    feat_df = create_features(df_clean, sensors_cols, windows=[5, 15, 60])

    # Select features for modeling
    feature_cols = [c for c in feat_df.columns if any(s in c for s in sensors_cols) and c not in sensors_cols]
    # also include original sensors (optional)
    feature_cols = sensors_cols + feature_cols
    logger.info('Number of features created: %d', len(feature_cols))

    # Handle any remaining NaNs
    feat_df = feat_df.fillna(method='bfill').fillna(method='ffill').fillna(0)

    # Split train/test (time-based split)
    split_idx = int(len(feat_df) * 0.7)
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_test = test_df['is_anomaly'].values if 'is_anomaly' in test_df.columns else None

    # Scale
    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Model 1: Isolation Forest
    iso_model, iso_scores_train, iso_preds_train = train_isolation_forest(X_train_scaled, n_estimators=200, contamination='auto')
    # apply to test
    iso_scores_test = -iso_model.decision_function(X_test_scaled)
    iso_preds_test = (iso_model.predict(X_test_scaled) == -1).astype(int)

    # Evaluate if labels available
    if y_test is not None:
        iso_eval = evaluate_binary(y_test, iso_preds_test)
        logger.info('Isolation Forest evaluation on test set: %s', iso_eval)
    else:
        iso_eval = None

    # Visualize anomalies on first sensor
    logger.info('Plotting Isolation Forest anomalies (test segment)')
    plot_anomalies(test_df.reset_index(drop=True), 'timestamp', sensors_cols[0], iso_preds_test.astype(bool), title='IsolationForest detected anomalies (test)')

    # Model 2: LSTM Autoencoder
    # Create sequences from scaled data (we'll use only original sensor columns + key features)
    seq_len = 30
    # LSTM expects 3D data; use scaled features
    # Build sequences for training using only non-anomalous assumption: in unsupervised case, we often train on data assumed mostly normal.
    X_train_seq = create_sequences(X_train_scaled, seq_len=seq_len)
    X_test_seq = create_sequences(X_test_scaled, seq_len=seq_len)

    logger.info('LSTM Autoencoder: training sequences shapes train=%s test=%s', X_train_seq.shape, X_test_seq.shape)

    n_timesteps = X_train_seq.shape[1]
    n_features = X_train_seq.shape[2]
    autoenc = build_lstm_autoencoder(n_timesteps, n_features, latent_dim=8)
    autoenc.summary()

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = autoenc.fit(X_train_seq, X_train_seq, epochs=30, batch_size=128, validation_split=0.1, callbacks=[early_stop], verbose=1)

    # Reconstruction error on test sequences
    X_test_pred = autoenc.predict(X_test_seq)
    mse_seq = np.mean(np.mean(np.square(X_test_seq - X_test_pred), axis=2), axis=1)  # per-sequence MSE

    # Map sequence MSE back to timestamps: assign each sequence's score to its last timestamp index
    test_seq_timestamps = test_df['timestamp'].values[seq_len-1:]

    # Determine threshold: e.g. using training reconstruction error quantile
    X_train_pred = autoenc.predict(X_train_seq)
    mse_train = np.mean(np.mean(np.square(X_train_seq - X_train_pred), axis=2), axis=1)
    threshold = np.quantile(mse_train, 0.995)
    logger.info('LSTM-AE anomaly threshold (99.5th percentile of train MSE): %.6f', threshold)

    ae_preds_seq = (mse_seq > threshold).astype(int)

    # Convert sequence-level predictions to timestamp-level for comparison with y_test
    ae_preds_ts = np.zeros(len(test_df), dtype=int)
    # Each sequence corresponds to a position i..i+seq_len-1, we labeled at last timestamp; set that timestamp as anomalous
    for i, p in enumerate(ae_preds_seq):
        ts_idx = i + seq_len - 1
        if ts_idx < len(ae_preds_ts):
            ae_preds_ts[ts_idx] = max(ae_preds_ts[ts_idx], p)

    if y_test is not None:
        # align y_test length with ae_preds_ts (ae preds shorter by seq_len-1)
        y_test_aligned = y_test[seq_len-1:]
        ae_eval = evaluate_binary(y_test_aligned, ae_preds_seq)
        logger.info('LSTM Autoencoder evaluation (sequence-level): %s', ae_eval)
    else:
        ae_eval = None

    # Visualize autoencoder anomalies on test
    logger.info('Plotting LSTM-AE detected anomalies (test segment)')
    plot_anomalies(test_df.reset_index(drop=True), 'timestamp', sensors_cols[0], ae_preds_ts.astype(bool), title='LSTM-AE detected anomalies (test)')

    # Summary of results
    results = {
        'isolation_forest': {
            'eval': iso_eval,
            'preds_test_length': int(len(iso_preds_test))
        },
        'lstm_autoencoder': {
            'eval': ae_eval,
            'threshold': float(threshold),
            'preds_sequence_length': int(len(ae_preds_seq))
        }
    }

    with open('models/summary_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info('Pipeline finished. Results saved to models/summary_results.json')


if __name__ == '__main__':
    main()
