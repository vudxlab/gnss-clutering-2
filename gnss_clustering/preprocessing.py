"""
Module tiền xử lý dữ liệu GNSS:
- Hampel filter (loại bỏ ngoại lai)
- Các bộ lọc làm mịn: Moving Average, Gaussian, Butterworth, Kalman
- Giảm chiều bằng cửa sổ trung bình
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt

from . import config


# ---------------------------------------------------------------------------
# Hampel filter
# ---------------------------------------------------------------------------

def median_absolute_deviation(x):
    """Tính MAD của mảng x."""
    return np.median(np.abs(x - np.median(x)))


def hampel_filter(ts, window_size=None, n_sigmas=None):
    """
    Lọc ngoại lai bằng Median Absolute Deviation (MAD) trên từng hàng.

    Parameters
    ----------
    ts : np.ndarray or pd.DataFrame, shape (n_samples, n_features)
    window_size : int
        Nửa kích thước cửa sổ (tổng = 2*window_size).
    n_sigmas : float
        Ngưỡng số sigma.

    Returns
    -------
    filtered_data : np.ndarray
    outlier_indices : np.ndarray of bool, same shape
    """
    if window_size is None:
        window_size = config.HAMPEL_WINDOW_SIZE
    if n_sigmas is None:
        n_sigmas = config.HAMPEL_N_SIGMAS

    data = ts.values if isinstance(ts, pd.DataFrame) else ts.copy()
    if data.ndim != 2:
        raise ValueError("Input phải là mảng 2D hoặc DataFrame")

    filtered_data = data.copy()
    outlier_indices = np.zeros_like(data, dtype=bool)
    k = 1.4826  # Hệ số chuẩn hóa phân phối chuẩn

    for i in range(data.shape[0]):
        row = pd.Series(data[i])
        rolling = row.rolling(window_size * 2, center=True)
        rolling_median = rolling.median().ffill().bfill()
        rolling_sigma = k * rolling.apply(median_absolute_deviation).ffill().bfill()

        outliers = np.abs(row - rolling_median) >= (n_sigmas * rolling_sigma)
        outlier_indices[i] = outliers
        filtered_data[i, outliers] = rolling_median[outliers]

    return filtered_data, outlier_indices


# ---------------------------------------------------------------------------
# Bộ lọc làm mịn
# ---------------------------------------------------------------------------

def moving_average_2d(data, window_size=10):
    """
    Áp dụng trung bình động cho từng hàng của ma trận 2D.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
    window_size : int

    Returns
    -------
    np.ndarray cùng shape với data
    """
    filtered = np.zeros_like(data, dtype=float)
    for i in range(data.shape[0]):
        s = pd.Series(data[i])
        smoothed = s.rolling(window=window_size, center=True).mean()
        filtered[i] = smoothed.ffill().bfill().values
    return filtered


def gaussian_filter_2d(data, sigma=2):
    """
    Áp dụng bộ lọc Gaussian 1D cho từng hàng.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
    sigma : float

    Returns
    -------
    np.ndarray
    """
    filtered = np.zeros_like(data, dtype=float)
    for i in range(data.shape[0]):
        filtered[i] = gaussian_filter1d(data[i], sigma=sigma)
    return filtered


def butterworth_filter_2d(data, cutoff_frequency=None, order=None):
    """
    Áp dụng bộ lọc Butterworth low-pass cho từng hàng.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
    cutoff_frequency : float
    order : int

    Returns
    -------
    np.ndarray
    """
    if cutoff_frequency is None:
        cutoff_frequency = config.BUTTERWORTH_CUTOFF
    if order is None:
        order = config.BUTTERWORTH_ORDER

    filtered = np.zeros_like(data, dtype=float)
    b, a = butter(order, cutoff_frequency, btype='low', analog=False)
    for i in range(data.shape[0]):
        filtered[i] = filtfilt(b, a, data[i])
    return filtered


def kalman_filter_2d(data, process_variance=None, measurement_variance=None):
    """
    Áp dụng bộ lọc Kalman đơn giản cho từng hàng.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
    process_variance : float
    measurement_variance : float

    Returns
    -------
    np.ndarray
    """
    if process_variance is None:
        process_variance = config.KALMAN_PROCESS_VARIANCE
    if measurement_variance is None:
        measurement_variance = config.KALMAN_MEASUREMENT_VARIANCE

    filtered = np.zeros_like(data, dtype=float)

    for i in range(data.shape[0]):
        series = data[i]
        n = len(series)
        xhat = np.zeros(n)
        P = np.zeros(n)
        xhatminus = np.zeros(n)
        Pminus = np.zeros(n)
        K = np.zeros(n)

        xhat[0] = series[0]
        P[0] = 1.0

        for k in range(1, n):
            xhatminus[k] = xhat[k - 1]
            Pminus[k] = P[k - 1] + process_variance
            K[k] = Pminus[k] / (Pminus[k] + measurement_variance)
            xhat[k] = xhatminus[k] + K[k] * (series[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]

        filtered[i] = xhat

    return filtered


# ---------------------------------------------------------------------------
# Giảm chiều bằng cửa sổ trung bình
# ---------------------------------------------------------------------------

def reshape_by_window(data, window_size=None):
    """
    Giảm chiều mỗi hàng bằng cách tính trung bình trong cửa sổ không chồng nhau.

    Ví dụ: hàng có 3600 giá trị + window_size=10 → 360 giá trị.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
    window_size : int

    Returns
    -------
    np.ndarray, shape (n_samples, n_features // window_size)
    """
    if window_size is None:
        window_size = config.RESHAPE_WINDOW_SIZE

    reshaped = []
    for row in data:
        new_row = np.mean(row.reshape(-1, window_size), axis=1)
        reshaped.append(new_row)

    result = np.array(reshaped)
    print(f"Kích thước sau khi reshape: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Pipeline tiền xử lý đầy đủ
# ---------------------------------------------------------------------------

def preprocess_pipeline(hourly_matrix, hampel_window=None, hampel_sigma=None, reshape_window=None):
    """
    Pipeline tiền xử lý hoàn chỉnh:
    1. Hampel filter (loại bỏ ngoại lai)
    2. Reshape bằng cửa sổ trung bình (giảm chiều)
    3. Kalman filter (làm mịn)

    Parameters
    ----------
    hourly_matrix : np.ndarray, shape (n_hours, 3600)
    hampel_window : int
    hampel_sigma : float
    reshape_window : int

    Returns
    -------
    data_filtered : np.ndarray
        Dữ liệu sau toàn bộ tiền xử lý.
    hampel_data : np.ndarray
        Dữ liệu sau Hampel filter.
    reshape_data : np.ndarray
        Dữ liệu sau reshape.
    """
    print("Bước 1: Hampel filter...")
    hampel_data, _ = hampel_filter(hourly_matrix, window_size=hampel_window, n_sigmas=hampel_sigma)

    print("Bước 2: Reshape bằng cửa sổ trung bình...")
    reshape_data = reshape_by_window(hampel_data, window_size=reshape_window)

    print("Bước 3: Kalman filter...")
    data_filtered = kalman_filter_2d(reshape_data)

    print("Tiền xử lý hoàn thành.")
    return data_filtered, hampel_data, reshape_data
