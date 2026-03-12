"""
================================================================
LightGBM + Optuna + Feature Engineering + Trend Analysis
================================================================
ติดตั้ง dependencies:
    pip install lightgbm optuna scikit-learn pandas numpy matplotlib joblib

วิธีใช้งาน:
    python vibration_forecasting_v2.py
    แล้วใส่ path ของไฟล์ .txt ทีละไฟล์ หรือโฟลเดอร์ที่มีหลายไฟล์
================================================================
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from numpy.fft import fft

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==========================================
# CONFIG — ปรับค่าตรงนี้ได้
# ==========================================
N_LAGS        = 20
ROLLING_WINS  = [10, 20, 50]
N_SPLITS_CV   = 5
OPTUNA_TRIALS = 50
TEST_RATIO    = 0.2


# ==========================================
# 1. โหลดและ parse ไฟล์ .txt
# ==========================================
def parse_vibration_file(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('---------'):
            start_idx = i + 1
            break

    data = []
    for line in lines[start_idx:]:
        parts = line.split()
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                try:
                    data.append([float(parts[i]), float(parts[i + 1])])
                except ValueError:
                    continue

    df = pd.DataFrame(data, columns=['Time_ms', 'Amplitude'])
    return df.sort_values('Time_ms').reset_index(drop=True)


# ==========================================
# 2. Feature Engineering
# ==========================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    amp = d['Amplitude']

    for i in range(1, N_LAGS + 1):
        d[f'lag_{i}'] = amp.shift(i)

    for w in ROLLING_WINS:
        roll = amp.rolling(w)
        d[f'roll_mean_{w}']  = roll.mean()
        d[f'roll_std_{w}']   = roll.std()
        d[f'roll_max_{w}']   = roll.max()
        d[f'roll_min_{w}']   = roll.min()
        d[f'roll_range_{w}'] = d[f'roll_max_{w}'] - d[f'roll_min_{w}']

    d['diff_1'] = amp.diff(1)
    d['diff_2'] = amp.diff(2)
    d['diff_5'] = amp.diff(5)

    def dominant_freq(window):
        fft_vals = np.abs(fft(window))
        return np.argmax(fft_vals[1:len(fft_vals) // 2]) + 1

    win_size = max(ROLLING_WINS)
    d['fft_dom_freq'] = amp.rolling(win_size).apply(dominant_freq, raw=True)

    return d.dropna().reset_index(drop=True)


# ==========================================
# 3. Optuna Objective
# ==========================================
def optuna_objective(trial, X_train_full, y_train_full):
    params = {
        'objective'        : 'regression',
        'metric'           : 'rmse',
        'verbosity'        : -1,
        'boosting_type'    : 'gbdt',
        'n_estimators'     : trial.suggest_int('n_estimators', 200, 2000),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'num_leaves'       : trial.suggest_int('num_leaves', 20, 300),
        'max_depth'        : trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample'        : trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    rmse_scores = []
    for train_idx, val_idx in tscv.split(X_train_full):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train_full.iloc[train_idx], y_train_full.iloc[train_idx],
            eval_set=[(X_train_full.iloc[val_idx], y_train_full.iloc[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        preds = model.predict(X_train_full.iloc[val_idx])
        rmse_scores.append(np.sqrt(mean_squared_error(y_train_full.iloc[val_idx], preds)))
    return np.mean(rmse_scores)


# ==========================================
# 4. Plot: Forecasting Results
# ==========================================
def plot_results(filename, time_test, y_test, predictions,
                 rmse, mae, mape, model, feature_cols, output_dir):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f'Vibration Forecasting v2 — {filename}\n'
        f'RMSE: {rmse:.4f}  |  MAE: {mae:.4f}  |  MAPE: {mape:.2f}%',
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_test, y_test.values, label='Actual',    color='steelblue', alpha=0.7, lw=1)
    ax1.plot(time_test, predictions,   label='Predicted', color='tomato', linestyle='--', alpha=0.85, lw=1)
    ax1.set_title('Actual vs Predicted Amplitude')
    ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Amplitude (G)')
    ax1.legend(); ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = fig.add_subplot(gs[1, 0])
    residuals = np.array(y_test) - predictions
    ax2.scatter(predictions, residuals, alpha=0.3, s=5, color='purple')
    ax2.axhline(0, color='red', lw=1)
    ax2.set_title('Residuals vs Predicted')
    ax2.set_xlabel('Predicted'); ax2.set_ylabel('Residual')
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax3 = fig.add_subplot(gs[1, 1])
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance.nlargest(15).sort_values().plot(kind='barh', ax=ax3, color='mediumseagreen')
    ax3.set_title('Top 15 Feature Importance')
    ax3.set_xlabel('Importance Score')
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'saved_plots', filename.replace('.txt', '.png'))
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'   บันทึกกราฟ: {plot_path}')


# ==========================================
# 5. Trend Analysis
# ==========================================
def analyze_trend(df_raw: pd.DataFrame, filename: str) -> dict:
    amp = df_raw['Amplitude']
    n   = len(amp)

    seg        = n // 3
    mean_early = amp.iloc[:seg].mean()
    mean_mid   = amp.iloc[seg:2*seg].mean()
    mean_late  = amp.iloc[2*seg:].mean()
    std_early  = amp.iloc[:seg].std()
    std_late   = amp.iloc[2*seg:].std()

    x = np.arange(n)
    slope, intercept = np.polyfit(x, amp.values, 1)
    slope_pct = (slope * n) / (abs(amp.mean()) + 1e-8) * 100

    rolling_mean = amp.rolling(50, center=True).mean().fillna(amp.mean())
    rolling_std  = amp.rolling(50, center=True).std().fillna(amp.std())
    spike_mask   = np.abs(amp - rolling_mean) > 3 * rolling_std
    spike_pct    = spike_mask.sum() / n * 100

    volatility_change_pct = (std_late - std_early) / (std_early + 1e-8) * 100

    if slope_pct > 10:
        trend_label   = 'เพิ่มขึ้น (Increasing)'
        trend_color   = 'red'
        trend_summary = 'แอมพลิจูดมีแนวโน้มเพิ่มขึ้นต่อเนื่อง อาจเข้าสู่สภาวะสึกหรอหรือความไม่สมดุล'
    elif slope_pct < -10:
        trend_label   = 'ลดลง (Decreasing)'
        trend_color   = 'green'
        trend_summary = 'แอมพลิจูดมีแนวโน้มลดลง เครื่องจักรอาจเสถียรขึ้นหรืออยู่ในช่วง run-in'
    else:
        trend_label   = 'คงที่ (Stable)'
        trend_color   = 'steelblue'
        trend_summary = 'แอมพลิจูดค่อนข้างคงที่ตลอดช่วงการวัด'

    if spike_pct > 5:
        spike_summary = f'พบ spike ผิดปกติสูง ({spike_pct:.1f}% ของข้อมูล) — ควรตรวจสอบแหล่งกำเนิดการสั่นสะเทือน'
    elif spike_pct > 1:
        spike_summary = f'พบ spike บ้าง ({spike_pct:.1f}%) — ควรติดตามต่อเนื่อง'
    else:
        spike_summary = f'แทบไม่พบ spike ผิดปกติ ({spike_pct:.1f}%)'

    if volatility_change_pct > 30:
        vol_summary = f'ความผันผวนเพิ่มขึ้น {volatility_change_pct:.1f}% เมื่อเทียบช่วงต้นกับช่วงท้าย — สัญญาณเตือนว่าสภาพเครื่องอาจเปลี่ยนแปลง'
    elif volatility_change_pct < -30:
        vol_summary = f'ความผันผวนลดลง {abs(volatility_change_pct):.1f}% — เครื่องจักรเสถียรขึ้น'
    else:
        vol_summary = f'ความผันผวนเปลี่ยนแปลงเล็กน้อย ({volatility_change_pct:+.1f}%)'

    risk_score = 0
    if slope_pct > 10:             risk_score += 2
    if slope_pct > 25:             risk_score += 1
    if spike_pct > 5:              risk_score += 2
    if spike_pct > 1:              risk_score += 1
    if volatility_change_pct > 30: risk_score += 2

    if risk_score >= 4:
        risk_label = 'สูง — ควรวางแผนบำรุงรักษาโดยเร็ว'
    elif risk_score >= 2:
        risk_label = 'ปานกลาง — ควรติดตามอย่างใกล้ชิด'
    else:
        risk_label = 'ต่ำ — เครื่องจักรอยู่ในสภาพปกติ'

    return {
        'filename'           : filename,
        'mean_early'         : round(mean_early, 4),
        'mean_mid'           : round(mean_mid, 4),
        'mean_late'          : round(mean_late, 4),
        'slope_pct'          : round(slope_pct, 2),
        'spike_pct'          : round(spike_pct, 2),
        'volatility_chg_pct' : round(volatility_change_pct, 2),
        'trend_label'        : trend_label,
        'trend_color'        : trend_color,
        'trend_summary'      : trend_summary,
        'spike_summary'      : spike_summary,
        'vol_summary'        : vol_summary,
        'risk_label'         : risk_label,
        'risk_score'         : risk_score,
    }


# ==========================================
# 6. Plot: Trend Analysis
# ==========================================
def plot_trend(df_raw: pd.DataFrame, trend: dict, output_dir: str):
    amp  = df_raw['Amplitude']
    time = df_raw['Time_ms']
    n    = len(amp)

    roll_mean  = amp.rolling(50, center=True).mean()
    roll_std   = amp.rolling(50, center=True).std()
    spike_mask = np.abs(amp - roll_mean.fillna(amp.mean())) > 3 * roll_std.fillna(amp.std())

    x = np.arange(n)
    slope, intercept = np.polyfit(x, amp.values, 1)
    trend_line = slope * x + intercept

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(
        f'Trend Analysis — {trend["filename"]}\n'
        f'แนวโน้ม: {trend["trend_label"]}  |  ความเสี่ยง: {trend["risk_label"]}',
        fontsize=13, fontweight='bold'
    )

    ax = axes[0]
    ax.plot(time, amp, color='lightsteelblue', alpha=0.5, lw=0.8, label='Raw Signal')
    ax.plot(time, roll_mean, color='steelblue', lw=1.5, label='Rolling Mean (50pt)')
    ax.plot(time, trend_line, color=trend['trend_color'], lw=2,
            linestyle='--', label=f'Trend ({trend["slope_pct"]:+.1f}%)')
    ax.scatter(time[spike_mask], amp[spike_mask], color='red', s=10, zorder=5,
               label=f'Spikes ({trend["spike_pct"]:.1f}%)')
    ax.set_title('Signal Overview with Trend Line')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Amplitude (G)')
    ax.legend(fontsize=8); ax.grid(True, linestyle=':', alpha=0.6)

    ax2 = axes[1]
    seg = n // 3
    segs       = [amp.iloc[:seg], amp.iloc[seg:2*seg], amp.iloc[2*seg:]]
    time_segs  = [time.iloc[:seg], time.iloc[seg:2*seg], time.iloc[2*seg:]]
    colors_seg = ['#4878cf', '#6acc65', '#d65f5f']
    labels_seg = ['ช่วงต้น', 'ช่วงกลาง', 'ช่วงท้าย']
    for s, t, c, lb in zip(segs, time_segs, colors_seg, labels_seg):
        ax2.plot(t, s, color=c, alpha=0.6, lw=0.8)
        ax2.axhline(s.mean(), color=c, lw=2, linestyle='--',
                    label=f'{lb} mean={s.mean():.4f}')
    ax2.set_title('Mean Amplitude by Segment (Early / Mid / Late)')
    ax2.set_xlabel('Time (ms)'); ax2.set_ylabel('Amplitude (G)')
    ax2.legend(fontsize=8); ax2.grid(True, linestyle=':', alpha=0.6)

    ax3 = axes[2]
    ax3.plot(time, roll_std, color='darkorange', lw=1.2, label='Rolling Std (50pt)')
    ax3.fill_between(time, 0, roll_std, alpha=0.2, color='darkorange')
    ax3.set_title(f'Volatility (Rolling Std) — {trend["vol_summary"]}')
    ax3.set_xlabel('Time (ms)'); ax3.set_ylabel('Std (G)')
    ax3.legend(fontsize=8); ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    trend_plot_path = os.path.join(
        output_dir, 'saved_plots',
        trend['filename'].replace('.txt', '_trend.png')
    )
    plt.savefig(trend_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'   บันทึกกราฟ trend: {trend_plot_path}')


# ==========================================
# 7. ประมวลผลไฟล์เดียว
# ==========================================
def process_file(file_path: str, output_dir: str):
    filename = os.path.basename(file_path)

    os.makedirs(os.path.join(output_dir, 'saved_models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'saved_plots'),  exist_ok=True)

    print(f'\n{"="*55}')
    print(f'  กำลังประมวลผล: {filename}')
    print(f'{"="*55}')

    df_raw = parse_vibration_file(file_path)
    if len(df_raw) < 100:
        print(f'   ข้อมูลน้อยเกินไป ({len(df_raw)} แถว)')
        return None, None
    print(f'   โหลดข้อมูลสำเร็จ: {len(df_raw):,} จุด')

    df_feat = build_features(df_raw)
    feature_cols = [c for c in df_feat.columns if c not in ['Time_ms', 'Amplitude']]
    X     = df_feat[feature_cols]
    y     = df_feat['Amplitude']
    times = df_feat['Time_ms']
    print(f'   Features: {len(feature_cols)} ตัว จาก {len(df_feat):,} แถว')

    split_idx = int(len(df_feat) * (1 - TEST_RATIO))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    time_test = times.iloc[split_idx:]

    print(f'   กำลัง Optuna tuning ({OPTUNA_TRIALS} trials) ...')
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train),
        n_trials=OPTUNA_TRIALS, show_progress_bar=True
    )
    best_params = {
        **study.best_params,
        'objective': 'regression', 'metric': 'rmse',
        'verbosity': -1, 'boosting_type': 'gbdt'
    }
    print(f'   Best CV RMSE: {study.best_value:.4f}')

    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)

    predictions = final_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae  = mean_absolute_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-8))) * 100
    print(f'   Test — RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%')

    model_path = os.path.join(output_dir, 'saved_models', filename.replace('.txt', '.pkl'))
    joblib.dump({'model': final_model, 'feature_cols': feature_cols,
                 'best_params': best_params}, model_path)
    print(f'   บันทึกโมเดล: {model_path}')

    plot_results(filename, time_test, y_test, predictions,
                 rmse, mae, mape, final_model, feature_cols, output_dir)

    trend = analyze_trend(df_raw, filename)
    print(f'\n  [Trend Analysis]')
    print(f'  แนวโน้ม         : {trend["trend_label"]}')
    print(f'  {trend["trend_summary"]}')
    print(f'  {trend["spike_summary"]}')
    print(f'  {trend["vol_summary"]}')
    print(f'  ระดับความเสี่ยง : {trend["risk_label"]}')
    plot_trend(df_raw, trend, output_dir)

    result = {
        'filename'    : filename,
        'n_points'    : len(df_raw),
        'n_features'  : len(feature_cols),
        'cv_rmse'     : round(study.best_value, 4),
        'test_rmse'   : round(rmse, 4),
        'test_mae'    : round(mae, 4),
        'test_mape_%' : round(mape, 2),
    }
    return result, trend


# ==========================================
# 8. Summary Report
# ==========================================
def save_summary(results: list, trends: list, output_dir: str):
    if not results:
        print('\nไม่มีผลลัพธ์ให้สรุป')
        return

    df_summary = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'summary_report.csv')
    df_summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\nบันทึก summary_report.csv: {csv_path}')
    print(df_summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Performance — All Machines', fontsize=13, fontweight='bold')
    labels = [r['filename'].replace('.txt', '') for r in results]
    x = np.arange(len(labels))

    axes[0].bar(x, df_summary['test_rmse'], color='steelblue', alpha=0.85)
    axes[0].set_title('Test RMSE per Machine'); axes[0].set_ylabel('RMSE')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)

    axes[1].bar(x, df_summary['test_mape_%'], color='tomato', alpha=0.85)
    axes[1].set_title('Test MAPE (%) per Machine'); axes[1].set_ylabel('MAPE (%)')
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    axes[1].grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()

    if not trends:
        return

    print(f'\n{"="*60}')
    print('  TREND SUMMARY — สรุปแนวโน้มเครื่องจักรทั้งหมด')
    print(f'{"="*60}')

    df_trend = pd.DataFrame([{
        'เครื่องจักร'        : t['filename'].replace('.txt', ''),
        'แนวโน้ม'            : t['trend_label'],
        'slope (%)'          : t['slope_pct'],
        'spike (%)'          : t['spike_pct'],
        'volatility chg (%)' : t['volatility_chg_pct'],
        'ระดับความเสี่ยง'    : t['risk_label'],
    } for t in trends])
    print(df_trend.to_string(index=False))

    print('\nรายละเอียดแต่ละเครื่อง:')
    print('-' * 60)
    for t in trends:
        name = t['filename'].replace('.txt', '')
        print(f'\n  [{name}]')
        print(f'  แนวโน้ม         : {t["trend_label"]}')
        print(f'  สรุปสัญญาณ     : {t["trend_summary"]}')
        print(f'  Spike           : {t["spike_summary"]}')
        print(f'  ความผันผวน      : {t["vol_summary"]}')
        print(f'  ระดับความเสี่ยง : {t["risk_label"]}')
        print(f'  mean ต้น/กลาง/ท้าย : {t["mean_early"]:.4f} / {t["mean_mid"]:.4f} / {t["mean_late"]:.4f}')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Trend Comparison — All Machines', fontsize=13, fontweight='bold')
    names      = [t['filename'].replace('.txt', '') for t in trends]
    slopes     = [t['slope_pct'] for t in trends]
    spikes     = [t['spike_pct'] for t in trends]
    vcpcts     = [t['volatility_chg_pct'] for t in trends]
    bar_colors = [t['trend_color'] for t in trends]
    x = np.arange(len(names))

    axes[0].bar(x, slopes, color=bar_colors, alpha=0.85)
    axes[0].axhline(0, color='black', lw=0.8, linestyle='--')
    axes[0].set_title('Trend Slope (% change)'); axes[0].set_ylabel('%')
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)

    axes[1].bar(x, spikes, color='tomato', alpha=0.85)
    axes[1].axhline(1, color='orange', lw=1, linestyle='--', label='1% threshold')
    axes[1].axhline(5, color='red',    lw=1, linestyle='--', label='5% threshold')
    axes[1].set_title('Spike Rate (%)'); axes[1].set_ylabel('%')
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', linestyle=':', alpha=0.6)

    axes[2].bar(x, vcpcts, color='darkorange', alpha=0.85)
    axes[2].axhline(0,  color='black', lw=0.8, linestyle='--')
    axes[2].axhline(30, color='red',   lw=1,   linestyle='--', label='+30% threshold')
    axes[2].set_title('Volatility Change (%)'); axes[2].set_ylabel('%')
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    axes[2].legend(fontsize=8)
    axes[2].grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trend_summary.png'), dpi=150, bbox_inches='tight')
    plt.show()

    trend_csv = os.path.join(output_dir, 'trend_report.csv')
    df_trend.to_csv(trend_csv, index=False, encoding='utf-8-sig')
    print(f'\nบันทึก trend_report.csv: {trend_csv}')
    print(f'บันทึก trend_summary.png: {os.path.join(output_dir, "trend_summary.png")}')


# ==========================================
# 9. เมนูหลัก
# ==========================================
if __name__ == '__main__':
    print('=' * 55)
    print('  Vibration Forecasting v2.0')
    print('  LightGBM + Optuna + Trend Analysis')
    print('=' * 55)

    all_results = []
    all_trends  = []
    output_dir  = '.'

    while True:
        print('\nตัวเลือก:')
        print('  [1] เพิ่มไฟล์ .txt')
        print('  [2] เพิ่มโฟลเดอร์ (ประมวลผลทุก .txt ในโฟลเดอร์)')
        print('  [3] ดู Summary ทุกไฟล์ที่ประมวลผลแล้ว')
        print('  [q] ออกจากโปรแกรม')

        choice = input('\nเลือก: ').strip().lower()

        if choice == 'q':
            if all_results:
                print('\nบันทึก Summary ก่อนออกโปรแกรม...')
                save_summary(all_results, all_trends, output_dir)
            print('ออกจากโปรแกรม...')
            break

        elif choice == '3':
            save_summary(all_results, all_trends, output_dir)

        elif choice in ('1', '2'):
            raw = input('Path (หรือพิมพ์ b ย้อนกลับ): ').strip().strip("\"'")
            if raw.lower() == 'b':
                continue

            if choice == '1':
                if not raw.endswith('.txt'):
                    print('ต้องเป็นไฟล์ .txt เท่านั้น')
                    continue
                paths = [raw]
            else:
                if not os.path.isdir(raw):
                    print('ไม่พบโฟลเดอร์ดังกล่าว')
                    continue
                paths = [os.path.join(raw, f) for f in os.listdir(raw) if f.endswith('.txt')]
                print(f'พบ {len(paths)} ไฟล์ .txt ในโฟลเดอร์')

            for path in paths:
                if not os.path.exists(path):
                    print(f'ไม่พบไฟล์: {path}')
                    continue
                output_dir = os.path.dirname(os.path.abspath(path))
                result, trend = process_file(path, output_dir)
                if result:
                    all_results.append(result)
                if trend:
                    all_trends.append(trend)

        else:
            print('กรุณาเลือก 1, 2, 3 หรือ q')
