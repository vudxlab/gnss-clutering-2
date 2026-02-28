"""
Module tải và tạo ma trận dữ liệu GNSS theo ngày và theo giờ
"""

import numpy as np
import pandas as pd

from . import config


def load_data(filepath=None):
    """
    Đọc dữ liệu GNSS từ file CSV.

    Parameters
    ----------
    filepath : str, optional
        Đường dẫn tới file CSV. Mặc định dùng config.DATA_PATH.

    Returns
    -------
    df : pd.DataFrame
        DataFrame với cột Timestamp (datetime) và cột Date (date).
    """
    if filepath is None:
        filepath = config.DATA_PATH

    print(f"Đang đọc file {filepath}...")
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date

    print(f"Tổng số dòng dữ liệu: {len(df):,}")
    print(f"Khoảng thời gian: từ {df['Timestamp'].min()} đến {df['Timestamp'].max()}")
    print(f"Số ngày có dữ liệu: {df['Date'].nunique()}")
    return df


def create_daily_matrix(df, save=True):
    """
    Tạo ma trận dữ liệu h_Coord theo từng giây trong ngày.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã load từ load_data().
    save : bool
        Lưu ma trận ra file .npy nếu True.

    Returns
    -------
    daily_matrix : np.ndarray, shape (n_days, 86400)
    unique_dates : list of datetime.date
    """
    print("Đang tạo ma trận dữ liệu theo giây cho mỗi ngày...")

    df = df.copy()
    df['Second_of_day'] = (
        df['Timestamp'].dt.hour * 3600
        + df['Timestamp'].dt.minute * 60
        + df['Timestamp'].dt.second
    )

    unique_dates = sorted(df['Date'].unique())
    seconds_per_day = config.SECONDS_PER_DAY
    daily_matrix = np.full((len(unique_dates), seconds_per_day), np.nan)

    for i, date in enumerate(unique_dates):
        day_data = df[df['Date'] == date].copy()
        for _, row in day_data.iterrows():
            second_idx = int(row['Second_of_day'])
            if 0 <= second_idx < seconds_per_day:
                daily_matrix[i, second_idx] = row['h_Coord']
        print(f"Ngày {date}: {day_data.shape[0]:,} điểm dữ liệu")

    total = daily_matrix.size
    nan_count = np.isnan(daily_matrix).sum()
    print(f"\nKích thước ma trận: {daily_matrix.shape}")
    print(f"Tỷ lệ dữ liệu có: {(total - nan_count) / total * 100:.2f}%")

    if save:
        np.save(config.MATRIX_FILE, daily_matrix)
        np.save(config.DATES_FILE, unique_dates)
        print(f"Đã lưu ma trận vào {config.MATRIX_FILE}")

    return daily_matrix, unique_dates


def create_hourly_matrix(daily_matrix, unique_dates, missing_threshold=None, save=True):
    """
    Phân chia ma trận ngày thành từng giờ và lọc bỏ giờ có quá nhiều dữ liệu thiếu.

    Parameters
    ----------
    daily_matrix : np.ndarray, shape (n_days, 86400)
    unique_dates : list of datetime.date
    missing_threshold : float, optional
        Ngưỡng % dữ liệu thiếu tối đa cho phép. Mặc định dùng config.MISSING_THRESHOLD.
    save : bool
        Lưu kết quả ra file nếu True.

    Returns
    -------
    hourly_matrix : np.ndarray, shape (n_valid_hours, 3600)
    valid_hours_info : pd.DataFrame
        Thông tin về các giờ hợp lệ.
    """
    if missing_threshold is None:
        missing_threshold = config.MISSING_THRESHOLD

    print("Đang tạo ma trận dữ liệu theo giờ...")

    seconds_per_hour = config.SECONDS_PER_HOUR
    hours_per_day = config.HOURS_PER_DAY

    hourly_data_list = []
    hourly_info_list = []

    for day_idx, date in enumerate(unique_dates):
        for hour in range(hours_per_day):
            start_s = hour * seconds_per_hour
            end_s = (hour + 1) * seconds_per_hour
            hour_data = daily_matrix[day_idx, start_s:end_s]

            nan_count = np.isnan(hour_data).sum()
            missing_pct = (nan_count / seconds_per_hour) * 100

            hourly_info_list.append({
                'date': date,
                'hour': hour,
                'day_idx': day_idx,
                'missing_percentage': missing_pct,
                'valid_points': seconds_per_hour - nan_count,
                'datetime': f"{date} {hour:02d}:00:00"
            })
            hourly_data_list.append(hour_data)

    hourly_info_df = pd.DataFrame(hourly_info_list)

    # Lọc theo ngưỡng
    valid_mask = hourly_info_df['missing_percentage'] <= missing_threshold
    valid_hours_info = hourly_info_df[valid_mask].copy()
    valid_indices = valid_hours_info.index.tolist()
    hourly_matrix = np.array([hourly_data_list[i] for i in valid_indices])

    print(f"Số giờ hợp lệ (≤{missing_threshold}% thiếu): {len(valid_hours_info)}/{len(hourly_info_df)}")
    print(f"Kích thước ma trận theo giờ: {hourly_matrix.shape}")

    if save:
        np.save(config.HOURLY_MATRIX_FILE, hourly_matrix)
        valid_hours_info.to_csv(config.HOURLY_INFO_FILE, index=False)
        print(f"Đã lưu vào {config.HOURLY_MATRIX_FILE} và {config.HOURLY_INFO_FILE}")

    return hourly_matrix, valid_hours_info


def load_cached_matrices():
    """
    Tải lại ma trận đã lưu từ file .npy / .csv.

    Returns
    -------
    daily_matrix : np.ndarray
    unique_dates : list
    hourly_matrix : np.ndarray
    valid_hours_info : pd.DataFrame
    """
    daily_matrix = np.load(config.MATRIX_FILE, allow_pickle=True)
    unique_dates = list(np.load(config.DATES_FILE, allow_pickle=True))
    hourly_matrix = np.load(config.HOURLY_MATRIX_FILE, allow_pickle=True)
    valid_hours_info = pd.read_csv(config.HOURLY_INFO_FILE)
    return daily_matrix, unique_dates, hourly_matrix, valid_hours_info


# ============================================================================
# Multi-axis support
# ============================================================================

def create_daily_matrices_multi(df, axes='xyh', save=True):
    """
    Tao ma tran du lieu theo tung giay trong ngay cho nhieu truc.

    Parameters
    ----------
    df : pd.DataFrame
    axes : str, e.g. 'xyh', 'xy', 'h'
    save : bool

    Returns
    -------
    daily_matrices : dict[str, np.ndarray], shape (n_days, 86400)
    unique_dates : list of datetime.date
    """
    print(f"Dang tao ma tran du lieu cho {len(axes)} truc: {list(axes)}...")

    df = df.copy()
    df['Second_of_day'] = (
        df['Timestamp'].dt.hour * 3600
        + df['Timestamp'].dt.minute * 60
        + df['Timestamp'].dt.second
    )

    unique_dates = sorted(df['Date'].unique())
    seconds_per_day = config.SECONDS_PER_DAY

    daily_matrices = {}
    for axis in axes:
        col = config.AXIS_COLUMN_MAP[axis]
        matrix = np.full((len(unique_dates), seconds_per_day), np.nan)
        for i, date in enumerate(unique_dates):
            day_data = df[df['Date'] == date]
            seconds = day_data['Second_of_day'].values.astype(int)
            values = day_data[col].values
            valid = (seconds >= 0) & (seconds < seconds_per_day)
            matrix[i, seconds[valid]] = values[valid]
        daily_matrices[axis] = matrix
        print(f"  Truc {axis} ({col}): {matrix.shape}")

    if save:
        # Luu h axis vao file cu de backward compat
        if 'h' in daily_matrices:
            np.save(config.MATRIX_FILE, daily_matrices['h'])
        np.save(config.DATES_FILE, unique_dates)
        for axis in axes:
            path = config.get_daily_matrix_path(axis)
            np.save(path, daily_matrices[axis])
            print(f"  Da luu {axis} -> {path}")

    return daily_matrices, unique_dates


def create_hourly_matrices_multi(daily_matrices, unique_dates,
                                  missing_threshold=None, save=True):
    """
    Phan chia ma tran ngay thanh tung gio cho nhieu truc.
    Valid hours duoc xac dinh dua tren truc dau tien (hoac 'h' neu co).

    Parameters
    ----------
    daily_matrices : dict[str, np.ndarray]
    unique_dates : list
    missing_threshold : float
    save : bool

    Returns
    -------
    hourly_matrices : dict[str, np.ndarray], shape (n_valid_hours, 3600)
    valid_hours_info : pd.DataFrame
    """
    if missing_threshold is None:
        missing_threshold = config.MISSING_THRESHOLD

    axes = list(daily_matrices.keys())
    # Dung truc dau tien de xac dinh valid hours
    ref_axis = 'h' if 'h' in axes else axes[0]
    ref_daily = daily_matrices[ref_axis]

    print(f"Dang tao ma tran theo gio cho {len(axes)} truc (ref: {ref_axis})...")

    seconds_per_hour = config.SECONDS_PER_HOUR
    hours_per_day = config.HOURS_PER_DAY

    # Tao hourly info tu reference axis
    hourly_info_list = []
    for day_idx, date in enumerate(unique_dates):
        for hour in range(hours_per_day):
            start_s = hour * seconds_per_hour
            end_s = (hour + 1) * seconds_per_hour
            hour_data = ref_daily[day_idx, start_s:end_s]
            nan_count = np.isnan(hour_data).sum()
            missing_pct = (nan_count / seconds_per_hour) * 100
            hourly_info_list.append({
                'date': date,
                'hour': hour,
                'day_idx': day_idx,
                'missing_percentage': missing_pct,
                'valid_points': seconds_per_hour - nan_count,
                'datetime': f"{date} {hour:02d}:00:00"
            })

    hourly_info_df = pd.DataFrame(hourly_info_list)
    valid_mask = hourly_info_df['missing_percentage'] <= missing_threshold
    valid_hours_info = hourly_info_df[valid_mask].copy()
    valid_indices = valid_hours_info.index.tolist()

    print(f"  So gio hop le: {len(valid_hours_info)}/{len(hourly_info_df)}")

    # Tao hourly matrix cho moi truc
    hourly_matrices = {}
    for axis in axes:
        daily = daily_matrices[axis]
        hourly_list = []
        for day_idx, date in enumerate(unique_dates):
            for hour in range(hours_per_day):
                start_s = hour * seconds_per_hour
                end_s = (hour + 1) * seconds_per_hour
                hourly_list.append(daily[day_idx, start_s:end_s])
        hourly_matrix = np.array([hourly_list[i] for i in valid_indices])
        hourly_matrices[axis] = hourly_matrix
        print(f"  Truc {axis}: {hourly_matrix.shape}")

    if save:
        # Backward compat: luu h vao file cu
        if 'h' in hourly_matrices:
            np.save(config.HOURLY_MATRIX_FILE, hourly_matrices['h'])
        valid_hours_info.to_csv(config.HOURLY_INFO_FILE, index=False)
        for axis in axes:
            path = config.get_hourly_matrix_path(axis)
            np.save(path, hourly_matrices[axis])
            print(f"  Da luu {axis} -> {path}")

    return hourly_matrices, valid_hours_info


def load_cached_matrices_multi(axes='h'):
    """
    Tai lai ma tran da luu cho nhieu truc.

    Parameters
    ----------
    axes : str, e.g. 'xyh', 'h'

    Returns
    -------
    daily_matrices : dict[str, np.ndarray]
    unique_dates : list
    hourly_matrices : dict[str, np.ndarray]
    valid_hours_info : pd.DataFrame
    """
    unique_dates = list(np.load(config.DATES_FILE, allow_pickle=True))
    valid_hours_info = pd.read_csv(config.HOURLY_INFO_FILE)

    daily_matrices = {}
    hourly_matrices = {}

    for axis in axes:
        daily_path = config.get_daily_matrix_path(axis)
        hourly_path = config.get_hourly_matrix_path(axis)

        import os
        if os.path.exists(daily_path):
            daily_matrices[axis] = np.load(daily_path, allow_pickle=True)
        elif axis == 'h' and os.path.exists(config.MATRIX_FILE):
            daily_matrices[axis] = np.load(config.MATRIX_FILE, allow_pickle=True)
        else:
            raise FileNotFoundError(f"Khong tim thay cache cho truc {axis}: {daily_path}")

        if os.path.exists(hourly_path):
            hourly_matrices[axis] = np.load(hourly_path, allow_pickle=True)
        elif axis == 'h' and os.path.exists(config.HOURLY_MATRIX_FILE):
            hourly_matrices[axis] = np.load(config.HOURLY_MATRIX_FILE, allow_pickle=True)
        else:
            raise FileNotFoundError(f"Khong tim thay cache cho truc {axis}: {hourly_path}")

    return daily_matrices, unique_dates, hourly_matrices, valid_hours_info


def get_daily_data(date_str, daily_matrix, unique_dates, start_time="00:00:00", end_time="23:59:59"):
    """
    Truy xuất dữ liệu h_Coord theo giây cho một ngày và khoảng thời gian cụ thể.

    Parameters
    ----------
    date_str : str
        Ngày theo định dạng 'YYYY-MM-DD'.
    daily_matrix : np.ndarray
    unique_dates : list
    start_time : str
    end_time : str

    Returns
    -------
    np.ndarray hoặc None
    """
    try:
        target_date = pd.to_datetime(date_str).date()
        if target_date not in unique_dates:
            print(f"Không có dữ liệu cho ngày {date_str}")
            return None
        date_idx = list(unique_dates).index(target_date)
        start_s = pd.to_datetime(start_time).hour * 3600 + pd.to_datetime(start_time).minute * 60 + pd.to_datetime(start_time).second
        end_s = pd.to_datetime(end_time).hour * 3600 + pd.to_datetime(end_time).minute * 60 + pd.to_datetime(end_time).second
        return daily_matrix[date_idx, start_s:end_s + 1]
    except Exception as e:
        print(f"Lỗi: {e}")
        return None
