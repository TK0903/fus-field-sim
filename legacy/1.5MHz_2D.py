# -*- coding: utf-8 -*-
"""
================================================================================
 球面集束型 単素子超音波トランスデューサ 音場計算 (Rayleigh-Sommerfeld 積分)
   対象: 自作球面集束 PZT (マウス脳 FUS-BBBO 用) / 単一周波数 1.5 MHz
   GPU (CUDA / PyTorch) 対応・CPU フォールバック可
   将来 3.0 / 6.0 MHz へ拡張予定 → 周波数は冒頭定数 FREQ_HZ を変えるだけ
================================================================================

--------------------------------------------------------------------------------
 座標系 (最初に定義する)
--------------------------------------------------------------------------------
  原点 = 素子の頂点 (apex, 軸上で最も凹んだ点)、 +z を放射方向とする。
  素子は球冠 (spherical cap)。球面の中心 = 幾何学的焦点 = (0,0,R)。

        z=0 apex            +z 放射方向  →→→
          \\  __                      focus
           \\/  \\___                  (0,0,R)=(0,0,10mm)
      お椀の内側  \\____                  *
      (z<h は除外) \\    \\___          /
                    |         \\___    /
   ─────────────────┼─── rim ──────●──┼──────────────  z 軸
                    |  z=h=2.4007mm|   |
                    |  半径 a=6.5mm|   z=R=10mm (幾何焦点)
                    apex(z=0)      rim
                    <---- キャップ深さ h ---->

    ・リム(開口縁): z = h = 2.4007 mm の平面上, 半径 a = 6.5 mm の円
    ・幾何学的焦点: z = R = 10.0 mm
    ・軸上スキャン: z = 3〜18 mm  (z < h はお椀の内側なので物理的に除外)
    ・球面中心 C=(0,0,R) から見た開口半角 theta_max = asin(a/R) ≈ 40.5°
    ・F 値 = R/D ≈ 0.77 → 強集束・広角。近軸近似は使わない。

--------------------------------------------------------------------------------
 物理的仮定 (仕様 §8)
--------------------------------------------------------------------------------
  ・CW(連続波)定常解。実駆動は 6.6 ms バースト (1.5 MHz で ~1e4 サイクル)
    なので定常状態とみなせる。
  ・線形伝搬。0.57 MPa/1.5 MHz の平面波衝撃形成距離 ~17 cm >> 焦点域数 mm。
    → 非線形歪みは無視でき、|p| は peak negative pressure (PNP) と等価。
  ・自由音場(水中のみ)。頭蓋・脳組織は含まない。in-situ 値ではない。
  ・u0 は表面法線振動速度の振幅(peak)。RMS ではない。音圧も peak 表記で統一。
  ・Rayleigh 積分は本来「平面剛体バッフル」用。曲面音源への適用は標準的近似で、
    obliquity factor(斜め係数)は省略している。

--------------------------------------------------------------------------------
 時間規約 / 積分式 (仕様 §4)
--------------------------------------------------------------------------------
  時間規約: exp(+i*omega*t)。よって外向き球面波は空間的に exp(-i*k*r')。
  剛体バッフル面上の Rayleigh-Sommerfeld 積分を球冠表面 S で数値積分:

      p(r) = (i*rho*c*k*u0 / (2*pi)) * ∫∫_S  exp(-i*k*r') / r'  dS

  u0 : 表面法線振動速度(一様・peak)、 r' = |場の点 - 音源点|。
================================================================================
"""

import os
import csv
import math
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 画面表示不要・PNG 保存のみ
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 図中の日本語が豆腐(□)にならないよう、利用可能な日本語フォントを自動選択。
# Windows なら Yu Gothic / Meiryo / MS Gothic のいずれかが通常入っている。
_installed = {f.name for f in font_manager.fontManager.ttflist}
for _jp in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP",
            "IPAexGothic", "TakaoGothic"):
    if _jp in _installed:
        plt.rcParams["font.family"] = _jp
        break
plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け防止

# ==============================================================================
#  §A. パラメータ (ファイル冒頭に集約 — 3.0/6.0 MHz 拡張はここだけ変更)
# ==============================================================================

# ---- 駆動周波数 -------------------------------------------------------------
FREQ_HZ   = 1.5e6          # [Hz] 駆動周波数。将来 3.0e6 / 6.0e6 に変更するだけ。

# ---- 素子形状 (球冠) --------------------------------------------------------
APERTURE_DIAMETER_M = 13.0e-3   # 開口直径 D [m]
CURV_RADIUS_M       = 10.0e-3   # 曲率半径 R [m] (= 幾何学的焦点距離)
# 派生量 (下で計算): a = D/2, h = R - sqrt(R^2 - a^2)

# ---- 媒質 (水) --------------------------------------------------------------
WATER_TEMP_C   = 22.0      # 水温 [℃]。音速は Marczak 式で温度依存に計算。
DENSITY_RHO    = 998.0     # 密度 [kg/m^3] 固定。
USE_ATTEN      = False     # 減衰を入れるか (既定オフ)
ATTEN_DB_CM_MHZ2 = 0.0022  # 水の減衰 [dB/(cm·MHz^2)]
# 注: 焦点距離 10 mm と短く、1.5 MHz では往復でも減衰 <0.1 % で無視できる。
#     (0.0022*(1.5^2)*~2cm = ~0.01 dB ≈ 0.1%)。既定オフで妥当。
# 注: 脱気水と通常水でバルク音響特性(音速・密度・減衰)は変わらない。溶存気体は
#     キャビテーション等「測定実務」の問題であり、ここで使う媒質定数には影響しない。

# ---- 数値積分 (音源サンプリング) --------------------------------------------
SUBDIV_PER_LAMBDA = 6      # 音源要素の弧長 <= lambda/6 を目標 (λ から自動計算)
                           # 1.5 MHz で λ/6 ≈ 165 µm。高周波化でも λ から自動追従。

# ---- 場のグリッド -----------------------------------------------------------
FIELD_SUBDIV_PER_LAMBDA = 8   # 場のグリッド間隔 <= lambda/8
AXIAL_Z_MIN_M =  3.0e-3       # 軸上スキャン開始 (z<h は除外なので 3mm から)
AXIAL_Z_MAX_M = 18.0e-3       # 軸上スキャン終了
LATERAL_HALF_M = 3.0e-3       # 横方向 ±3 mm

# ---- GPU / 計算 -------------------------------------------------------------
COMPLEX_DTYPE = torch.complex128   # 既定 complex128 (1.5MHz は軽いので精度優先)
REAL_DTYPE    = torch.float64
FIELD_CHUNK   = 4096       # 場の点のチャンクサイズ (OOM 時は自動半減してリトライ)

# ---- 出力 -------------------------------------------------------------------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TAG     = f"{FREQ_HZ/1e6:.1f}MHz"

# ---- 検証 -------------------------------------------------------------------
FOCAL_GAIN_TOL = 0.01     # 焦点利得の厳密値 G=k*h に対する許容相対誤差 (1%)
TARGET_PRESSURES_MPA = [0.20, 0.40, 0.57]  # u0 逆算する目標焦点音圧 [MPa]


# ==============================================================================
#  §B. 媒質: Marczak (1997) の式で純水の音速を温度から計算
# ==============================================================================
def marczak_sound_speed(temp_c: float) -> float:
    """
    純水の音速 [m/s] を水温 T[℃] から計算 (Marczak 1997, 0-95℃, 1 atm)。
    定数を直書きせず温度依存にするのは、λ が変わると期待焦点利得 G=k*h も
    変わるため (検証で c から期待値を再計算するのに使う)。
    """
    T = temp_c
    c = (1.402385e3
         + 5.038813e0   * T
         - 5.799136e-2  * T**2
         + 3.287156e-4  * T**3
         - 1.398845e-6  * T**4
         + 2.787860e-9  * T**5)
    return c


# ==============================================================================
#  §C. 球冠音源の等面積サンプリング
# ==============================================================================
def generate_source_points(a, R, h, lam, subdiv, device):
    """
    球冠表面を等面積サンプリングする。

    等面積の原理 (Archimedes の定理):
      球面では、軸 z に垂直な平面で切った「帯(zone)」の面積は
      その z 幅にのみ比例する:  dA = 2*pi*R*dz。
      よって z を等間隔に分割すれば、各リングの面積は自動的に等しくなる。
      (緯度経度の単純分割は極付近が密になるので使わない。)

    各リングの面積 = 2*pi*R*dz を M_i 個の要素に分ける。方位分割数 M_i は
    そのリングの円筒半径 rho_i に比例させ、方位方向の弧長間隔を一定(≈λ/subdiv)
    に保つ (要素アスペクト比を 1 に近づける)。

    メッシュの向き:
      球面中心は C=(0,0,R)。高さ z のリングの円筒半径は
        rho(z) = sqrt(R^2 - (z-R)^2) = sqrt(2*R*z - z^2)。
      このとき任意の表面点から焦点(0,0,R)までの距離は厳密に R になる
      (焦点利得の解析解が数値積分に依らず厳密に出る根拠)。

    弧長間隔について:
      目標弧長 ds_target = λ/subdiv。
      ・方位方向: M_i = round(2*pi*rho_i/ds_target) で ds_target にそろう。
      ・子午線方向: 等z分割は「等面積」を保証するが弧長は非一様
        (リム側で細かく、apex 近傍で粗い)。N_z は子午線全長 R*theta_max を
        ds_target で割って平均分解能を ds_target にそろえる。
        apex 近傍が粗くなる点は許容する。理由:
          (1) 焦点利得の厳密ゲートには影響しない(全点が焦点から等距離 R)。
          (2) apex は立体角が小さく寄与が小さい。
        近軸/近距離をより厳密にしたい場合は SUBDIV_PER_LAMBDA を上げる。

    Returns
    -------
    pts : (Ns,3) float64 tensor  音源点座標 [m]
    dS  : (Ns,)  float64 tensor  各音源点の担当面積 [m^2]  (Σ dS = 2*pi*R*h 厳密)
    info: dict   N_z, N_source, 最大子午線弧長 など
    """
    theta_max = math.asin(a / R)            # 開口半角 (球面中心から)
    ds_target = lam / subdiv                # 目標弧長 [m]

    # 子午線全長 = R*theta_max。平均弧長が ds_target 以下になるよう N_z を決定。
    N_z = max(1, math.ceil(R * theta_max / ds_target))
    dz  = h / N_z                           # z 方向の等間隔幅
    # 1 リングあたりの面積 (Archimedes): 全リング共通
    ring_area = 2.0 * math.pi * R * dz

    xs, ys, zs, dSs = [], [], [], []
    max_merid = 0.0
    prev_z = 0.0
    for i in range(N_z):
        z_i   = (i + 0.5) * dz              # リング中心の z (apex z=0 の特異を回避)
        rho_i = math.sqrt(max(0.0, 2.0 * R * z_i - z_i * z_i))  # 円筒半径

        # 方位分割数: 円筒半径に比例 → 方位弧長 ≈ ds_target
        M_i = max(1, int(round(2.0 * math.pi * rho_i / ds_target)))
        dS_i = ring_area / M_i              # この要素の担当面積

        # 隣接リングとスポーク(放射状の筋)を作らないよう半ピッチ千鳥にずらす
        phi0 = math.pi / M_i if (i % 2) else 0.0
        phis = phi0 + 2.0 * math.pi * (np.arange(M_i) / M_i)

        xs.append(rho_i * np.cos(phis))
        ys.append(rho_i * np.sin(phis))
        zs.append(np.full(M_i, z_i))
        dSs.append(np.full(M_i, dS_i))

        # 子午線方向の弧長 (診断用): 隣リング中心間の球面上距離
        dtheta = abs(math.acos(min(1.0, max(-1.0, 1.0 - z_i / R)))
                     - math.acos(min(1.0, max(-1.0, 1.0 - prev_z / R))))
        max_merid = max(max_merid, R * dtheta)
        prev_z = z_i

    x = np.concatenate(xs); y = np.concatenate(ys)
    z = np.concatenate(zs); dS = np.concatenate(dSs)

    pts = torch.tensor(np.stack([x, y, z], axis=1), dtype=REAL_DTYPE, device=device)
    dS  = torch.tensor(dS, dtype=REAL_DTYPE, device=device)

    info = dict(N_z=N_z, N_source=pts.shape[0], ds_target=ds_target,
                max_meridional_arc=max_merid, area_sum=float(dS.sum().item()),
                area_exact=2.0 * math.pi * R * h)
    return pts, dS, info


# ==============================================================================
#  §D. Rayleigh 積分の GPU 計算 (全ペア・テンソル化 / 場の点でチャンク分割)
# ==============================================================================
def compute_pressure(field_pts, src_pts, src_dS, k, prefactor,
                     alpha_np_per_m, chunk, device):
    """
    p(r) = prefactor * Σ_s [ exp(-alpha*r') * exp(-i*k*r') / r' ] * dS_s
      prefactor = i*rho*c*k*u0/(2*pi)

    ・Python for ループでペアを回さない。全ペアをテンソル化 (cdist)。
    ・場の点をチャンクに切って逐次計算し、音源方向に総和 → 将来の高周波化
      (音源点数増大) でも OOM を避けられる構造。
    ・OOM 時はチャンクを自動半減してリトライ。
    ・r' の特異点処理: 場の点は z>=3mm, 音源は z<=h=2.4mm なので実際に r'->0 は
      起きないが、安全のため r' を微小値でクランプ (ゼロ割回避)。

    Returns: p (Nf,) complex tensor
    """
    Nf = field_pts.shape[0]
    out = torch.empty(Nf, dtype=COMPLEX_DTYPE, device=device)
    r_floor = 1e-9  # 特異点クランプ [m]

    i0 = 0
    cur_chunk = int(chunk)
    while i0 < Nf:
        i1 = min(i0 + cur_chunk, Nf)
        try:
            fp = field_pts[i0:i1]                          # (C,3)
            d  = torch.cdist(fp, src_pts)                  # (C,Ns) 距離 r'
            d  = torch.clamp(d, min=r_floor)
            amp = 1.0 / d                                  # 1/r'
            if alpha_np_per_m > 0.0:
                amp = amp * torch.exp(-alpha_np_per_m * d)
            # green = amp * exp(-i k r')  を complex128 で構成
            green = torch.polar(amp, -k * d)               # (C,Ns) complex
            p_chunk = (green * src_dS).sum(dim=1)          # (C,) 音源方向総和
            out[i0:i1] = prefactor * p_chunk
            i0 = i1
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and cur_chunk > 1:
                torch.cuda.empty_cache() if device.type == "cuda" else None
                cur_chunk = max(1, cur_chunk // 2)
                print(f"  [OOM] チャンクを半減して再試行: chunk={cur_chunk}")
            else:
                raise
    return out


# ==============================================================================
#  §E. 補助関数 (ビーム幅・サイドローブ抽出)
# ==============================================================================
def fwhm_at_level(x, y, level_frac):
    """
    プロファイル y(x) のピークに対し、レベル level_frac(-6dB は 0.5)を
    横切る左右の交点を線形補間で求め、全幅を返す。見つからなければ NaN。
    """
    y = np.asarray(y, float); x = np.asarray(x, float)
    ip = int(np.argmax(y))
    ypk = y[ip]
    thr = ypk * level_frac

    def cross(idx_range):
        prev = ip
        for j in idx_range:
            if y[j] < thr <= y[prev] or y[j] <= thr < y[prev]:
                # prev と j の間で補間
                x0, x1 = x[prev], x[j]
                y0, y1 = y[prev], y[j]
                if y1 == y0:
                    return x1
                return x0 + (thr - y0) * (x1 - x0) / (y1 - y0)
            prev = j
        return np.nan

    xl = cross(range(ip - 1, -1, -1))
    xr = cross(range(ip + 1, len(x)))
    if np.isnan(xl) or np.isnan(xr):
        return np.nan, xl, xr
    return abs(xr - xl), xl, xr


def find_sidelobe(x, y):
    """
    横方向プロファイル |p|(x) の主ローブ外の最大サイドローブを探す。
    ピークから外側へ歩いて最初の極小(ヌル)を越えた領域の最大値を採る。
    Returns: (level_dB_rel_peak, x_position) / 見つからなければ (nan,nan)
    """
    y = np.asarray(y, float); x = np.asarray(x, float)
    ip = int(np.argmax(y)); ypk = y[ip]

    def scan(idx_range):
        # 最初の極小を探す
        prev = y[ip]; found_null = False; best = 0.0; best_x = np.nan
        pv = ypk
        for j in idx_range:
            if not found_null:
                if y[j] > pv:      # 下降が終わって上昇 → 直前が極小
                    found_null = True
            if found_null and y[j] > best:
                best = y[j]; best_x = x[j]
            pv = y[j]
        return best, best_x

    bl, xl = scan(range(ip - 1, -1, -1))
    br, xr = scan(range(ip + 1, len(x)))
    if br >= bl:
        best, bx = br, xr
    else:
        best, bx = bl, xl
    if best <= 0 or np.isnan(bx):
        return np.nan, np.nan
    return 20.0 * math.log10(best / ypk), bx


# ==============================================================================
#  §F. メイン
# ==============================================================================
def main():
    t_start = time.time()

    # ---- デバイス選択 -------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 派生パラメータ -----------------------------------------------------
    a = APERTURE_DIAMETER_M / 2.0                      # 開口半径 [m]
    R = CURV_RADIUS_M                                  # 曲率半径 [m]
    h = R - math.sqrt(R * R - a * a)                   # キャップ深さ [m]
    c = marczak_sound_speed(WATER_TEMP_C)              # 音速 [m/s] (温度依存)
    rho = DENSITY_RHO
    lam = c / FREQ_HZ                                  # 波長 [m]
    k = 2.0 * math.pi * FREQ_HZ / c                    # 波数 [rad/m]
    theta_max = math.asin(a / R)
    f_number = R / APERTURE_DIAMETER_M
    z_focus = R                                        # 幾何学的焦点 [m]

    # 減衰係数 [Np/m] (使う場合)
    alpha_np_per_m = 0.0
    if USE_ATTEN:
        f_mhz = FREQ_HZ / 1e6
        alpha_db_cm = ATTEN_DB_CM_MHZ2 * f_mhz * f_mhz   # [dB/cm]
        alpha_np_per_m = alpha_db_cm / 8.686 * 100.0     # dB/cm → Np/m

    # u0 は場計算では 1 とし、後で目標音圧にスケールする
    u0 = 1.0
    # prefactor = i*rho*c*k*u0/(2*pi)
    prefactor = torch.tensor(1j * rho * c * k * u0 / (2.0 * math.pi),
                             dtype=COMPLEX_DTYPE, device=device)

    # ---- 音源生成 -----------------------------------------------------------
    src_pts, src_dS, sinfo = generate_source_points(a, R, h, lam,
                                                     SUBDIV_PER_LAMBDA, device)
    N_source = sinfo["N_source"]

    # ---- 場の点数の見積り (2D マップが最大) --------------------------------
    dfield = lam / FIELD_SUBDIV_PER_LAMBDA
    z_axis = np.arange(AXIAL_Z_MIN_M, AXIAL_Z_MAX_M + dfield, dfield)
    x_lat  = np.arange(-LATERAL_HALF_M, LATERAL_HALF_M + dfield, dfield)
    N_map  = len(z_axis) * len(x_lat)

    # ---- 実行前サマリ表示 (仕様 §5) ----------------------------------------
    print("=" * 70)
    print(f" 球面集束トランスデューサ 音場計算  ({TAG})")
    print("=" * 70)
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        vram_gb = prop.total_memory / (1024**3)
        print(f" デバイス      : CUDA / {prop.name}")
        print(f" VRAM 容量     : {vram_gb:.2f} GB")
    else:
        print(" デバイス      : CPU (CUDA 非対応環境)")
    # 推定 VRAM: 最大チャンク時の主要テンソル (C×Ns) を複数保持
    bytes_pair = FIELD_CHUNK * N_source * 16   # complex128 1 本 = 16 byte
    est_vram_gb = bytes_pair * 3 / (1024**3)   # cdist/amp/green で ~3 本
    print(f" 音源点数 Ns   : {N_source:,}   (N_z={sinfo['N_z']} リング)")
    print(f" 場の点数(max) : {N_map:,}   (2D マップ {len(x_lat)}×{len(z_axis)})")
    print(f" 総ペア数(max) : {N_source * N_map:,}")
    print(f" 推定VRAM(chunk={FIELD_CHUNK}) : {est_vram_gb:.3f} GB / チャンク")
    print("-" * 70)
    print(f" 水温 T        : {WATER_TEMP_C:.1f} ℃   →  c = {c:.2f} m/s (Marczak)")
    print(f" 波長 λ        : {lam*1e3:.4f} mm   波数 k = {k:.2f} rad/m")
    print(f" 開口 D/a      : {APERTURE_DIAMETER_M*1e3:.1f} / {a*1e3:.2f} mm")
    print(f" 曲率 R        : {R*1e3:.1f} mm   キャップ深さ h = {h*1e3:.4f} mm")
    print(f" 開口半角      : {math.degrees(theta_max):.1f}°   F# = {f_number:.2f}")
    print(f" 音源要素弧長  : 目標 λ/{SUBDIV_PER_LAMBDA} = {sinfo['ds_target']*1e6:.0f} µm"
          f"  (最大子午線弧長 {sinfo['max_meridional_arc']*1e6:.0f} µm)")
    print(f" 面積検証      : Σ dS = {sinfo['area_sum']*1e6:.6f} mm²  "
          f"(厳密 2πRh = {sinfo['area_exact']*1e6:.6f} mm²)")
    print(f" 減衰          : {'ON' if USE_ATTEN else 'OFF (焦点10mmでは<0.1%,無視可)'}")
    print(f" 時間規約      : exp(+iωt) → 空間項 exp(-ikr')")
    print("=" * 70)

    # =====================================================================
    #  §F-1. 検証: 一次基準 = 焦点利得の厳密解  G = k*h
    # =====================================================================
    # 球冠上の全点は焦点(0,0,R)から等距離 R → 焦点音圧は解析的に厳密:
    #   p_focus = i*rho*c*u0*k*h*exp(-i*k*R)  ,  G = k*h
    # 音速を温度依存にすると λ→k が変わるので、期待値も c から再計算する。
    G_exact = k * h
    focus_pt = torch.tensor([[0.0, 0.0, z_focus]], dtype=REAL_DTYPE, device=device)
    p_focus = compute_pressure(focus_pt, src_pts, src_dS, k, prefactor,
                               alpha_np_per_m, FIELD_CHUNK, device)
    p_surface = rho * c * u0                     # 表面音圧 (基準)
    G_num = float(torch.abs(p_focus[0]).item()) / p_surface
    rel_err = abs(G_num - G_exact) / G_exact

    print("\n[検証 §6] 一次基準: 焦点利得の厳密解 G = k*h")
    print(f"  厳密解   G_exact = k*h = {G_exact:.4f}")
    print(f"  数値積分 G_num          = {G_num:.4f}")
    print(f"  相対誤差               = {rel_err*100:.4e} %  (許容 {FOCAL_GAIN_TOL*100:.0f}%)")
    if rel_err > FOCAL_GAIN_TOL:
        raise RuntimeError(
            f"焦点利得が厳密値 G=k*h と {FOCAL_GAIN_TOL*100:.0f}% 以内で一致しません "
            f"(誤差 {rel_err*100:.3f}%)。サンプリング/実装を確認してください。")
    print("  → OK (厳密値と一致)")
    print("  注: 球冠上の全点は焦点から等距離 R のため、焦点利得は音源分割数に依らず")
    print("      厳密に k*h になる(下の収束表参照)。分割密度が効くのは軸外の場のみ。")

    # =====================================================================
    #  §F-2. グリッド収束テスト
    # =====================================================================
    print("\n[検証 §6] グリッド収束テスト")
    print(f"  {'subdiv':>6} {'N_z':>5} {'N_source':>9} {'G_num':>10} "
          f"{'相対誤差[%]':>12} {'|p|@z=7mm(軸外)':>16}")
    ref_pt = torch.tensor([[0.0, 0.0, 7.0e-3]], dtype=REAL_DTYPE, device=device)
    conv_rows = []
    for sd in [3, 4, 6, 8, 12]:
        sp, sdS, si = generate_source_points(a, R, h, lam, sd, device)
        pf = compute_pressure(focus_pt, sp, sdS, k, prefactor,
                              alpha_np_per_m, FIELD_CHUNK, device)
        gnum = float(torch.abs(pf[0]).item()) / p_surface
        pr = compute_pressure(ref_pt, sp, sdS, k, prefactor,
                              alpha_np_per_m, FIELD_CHUNK, device)
        p_off = float(torch.abs(pr[0]).item()) / p_surface
        err = abs(gnum - G_exact) / G_exact * 100
        conv_rows.append((sd, si["N_z"], si["N_source"], gnum, err, p_off))
        print(f"  {sd:>6} {si['N_z']:>5} {si['N_source']:>9,} {gnum:>10.4f} "
              f"{err:>12.3e} {p_off:>16.4f}")
    print("  焦点利得は全 subdiv で厳密(誤差~機械精度)。軸外 |p|@z=7mm は密度で収束。")

    # =====================================================================
    #  §F-3. O'Neil (1949) 軸上解との比較 (参考)
    # =====================================================================
    # 実装した Rayleigh 積分は「厳密(非近軸)」。教科書形 O'Neil 解は近軸近似を
    # 含むことがあり、F#=0.77 の強集束では数% ずれうる。一次基準はあくまで
    # G=k*h であり、O'Neil とのずれを即バグと判定しない。
    # ここでは球面波型 O'Neil の軸上 |p| を参考プロットとして重ねる。
    def oneil_axial_abs(z):
        """
        O'Neil (1949) 球面集束の軸上音圧の大きさ (球面波型・参考)。
        p_ax(z)/ (rho*c*u0) = |exp(-i k d1) - exp(-i k d2)| 型の閉形式。
        d1 = z - h (apex 平面からの距離ではなく apex からの直達),
        リム端までの距離 d2 = sqrt((z-h)^2 + a^2)。位相差で干渉。
        近軸/幾何の扱いにより教科書ごとに係数が異なるため「参考」とする。
        """
        z = np.asarray(z, float)
        d_apex = np.abs(z - 0.0)                    # apex(0,0,0) までの距離
        d_rim  = np.sqrt((z - h) ** 2 + a ** 2)     # リム端までの距離
        # 位相差 k*(d_rim - d_apex) による振幅 (球冠の直達-回折干渉, 参考式)
        return np.abs(2.0 * np.sin(k * (d_rim - d_apex) / 2.0))

    # =====================================================================
    #  §F-4. 軸上音圧分布 (z = 3〜18 mm)  … 出力①
    # =====================================================================
    ax_pts = torch.tensor(np.stack([np.zeros_like(z_axis), np.zeros_like(z_axis),
                                    z_axis], axis=1), dtype=REAL_DTYPE, device=device)
    p_axis = compute_pressure(ax_pts, src_pts, src_dS, k, prefactor,
                              alpha_np_per_m, FIELD_CHUNK, device)
    p_axis_abs = torch.abs(p_axis).detach().cpu().numpy() / p_surface  # 利得表示

    ip_ax = int(np.argmax(p_axis_abs))
    z_peak = z_axis[ip_ax]
    G_peak = p_axis_abs[ip_ax]
    z_shift = z_peak - z_focus

    # -6dB 軸方向焦点長 (depth of focus)
    dof, z6l, z6r = fwhm_at_level(z_axis, p_axis_abs, 0.5)

    # =====================================================================
    #  §F-5. 焦点面での横方向分布 (±3 mm, z=幾何焦点)  … 出力②
    # =====================================================================
    lat_pts = torch.tensor(np.stack([x_lat, np.zeros_like(x_lat),
                                     np.full_like(x_lat, z_focus)], axis=1),
                           dtype=REAL_DTYPE, device=device)
    p_lat = compute_pressure(lat_pts, src_pts, src_dS, k, prefactor,
                             alpha_np_per_m, FIELD_CHUNK, device)
    p_lat_abs = torch.abs(p_lat).detach().cpu().numpy() / p_surface

    beamw, xl6, xr6 = fwhm_at_level(x_lat, p_lat_abs, 0.5)   # -6dB 横方向ビーム幅
    sll_db, sll_x = find_sidelobe(x_lat, p_lat_abs)          # 最大サイドローブ

    # =====================================================================
    #  §F-6. 軸-横 2D 音場マップ (dB, -30dB まで)  … 出力③
    # =====================================================================
    XX, ZZ = np.meshgrid(x_lat, z_axis)     # (Nz, Nx)
    map_pts = torch.tensor(np.stack([XX.ravel(), np.zeros(XX.size),
                                     ZZ.ravel()], axis=1),
                           dtype=REAL_DTYPE, device=device)
    p_map = compute_pressure(map_pts, src_pts, src_dS, k, prefactor,
                             alpha_np_per_m, FIELD_CHUNK, device)
    p_map_abs = torch.abs(p_map).detach().cpu().numpy().reshape(XX.shape)
    p_map_abs /= p_surface
    pmax_map = p_map_abs.max()
    map_db = 20.0 * np.log10(np.clip(p_map_abs / pmax_map, 1e-6, None))

    # =====================================================================
    #  §F-7. u0 逆算表 (目標焦点音圧 → u0, 表面音圧)  … 出力⑨
    # =====================================================================
    # |p_focus| = rho*c*u0 * G   →  u0 = p_target/(rho*c*G)
    #  p_surface_needed = rho*c*u0 = p_target/G
    u0_rows = []
    for pmpa in TARGET_PRESSURES_MPA:
        p_target = pmpa * 1e6
        u0_need = p_target / (rho * c * G_exact)
        p_surf_need = rho * c * u0_need
        u0_rows.append((pmpa, u0_need, p_surf_need / 1e6))

    # =====================================================================
    #  §G. コンソール出力 (数値結果 §7 の 4〜9)
    # =====================================================================
    mouse_brain_mm = 5.0
    print("\n" + "=" * 70)
    print(" 数値結果")
    print("=" * 70)
    print(f" ④ 焦点利得 G = |p_focus|/|p_surface| = {G_num:.3f}  "
          f"(厳密 k*h={G_exact:.3f})")
    print(f" ⑤ -6dB 横方向ビーム幅          = {beamw*1e3:.3f} mm")
    print(f" ⑥ -6dB 軸方向焦点長(DOF)       = {dof*1e3:.3f} mm")
    print(f"      マウス脳厚 ~{mouse_brain_mm:.0f} mm との比: "
          f"DOF/脳厚 = {dof*1e3/mouse_brain_mm:.2f}", end="")
    if dof * 1e3 > mouse_brain_mm:
        print(" → 焦点長が脳厚より長い(軸方向に脳全体を跨ぐ)。深さ選択性は低い。")
    else:
        print(" → 焦点長は脳厚より短く、軸方向の局在は十分。")
    print(f" ⑦ 音圧最大点 z = {z_peak*1e3:.3f} mm  "
          f"(幾何焦点 {z_focus*1e3:.1f} mm からのずれ {z_shift*1e3:+.3f} mm)")
    print(f"      高利得 k*h={G_exact:.1f}>>1 のためシフトは小さい(測定値の報告)。")
    if sll_x == sll_x:  # not nan
        print(f" ⑧ 最大サイドローブ           = {sll_db:.2f} dB  "
              f"@ x = {sll_x*1e3:+.3f} mm")
    else:
        print(f" ⑧ 最大サイドローブ           = ±3mm 範囲内に明瞭なものなし")
    print("\n ⑨ 目標焦点音圧を得る表面法線振動速度 u0 と表面音圧:")
    print(f"    {'p_focus[MPa]':>12} {'u0[m/s]':>12} {'p_surface[MPa]':>16}")
    for pmpa, u0n, psurf in u0_rows:
        print(f"    {pmpa:>12.2f} {u0n:>12.5f} {psurf:>16.5f}")
    print("    ※ これは音響側の要求値。駆動電圧[Vpp]への換算には素子の電気音響")
    print("       変換効率が別途必要。本計算だけではアンプの電圧余裕は判定できない。")

    # =====================================================================
    #  §H. CSV 出力 (数値結果 4〜9)
    # =====================================================================
    csv_path = os.path.join(OUT_DIR, f"results_{TAG}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["item", "quantity", "value", "unit", "note"])
        w.writerow(["4", "focal_gain_G", f"{G_num:.6f}", "-",
                    f"exact k*h={G_exact:.6f}"])
        w.writerow(["5", "lateral_-6dB_beamwidth", f"{beamw*1e3:.6f}", "mm", ""])
        w.writerow(["6", "axial_-6dB_DOF", f"{dof*1e3:.6f}", "mm",
                    f"mouse_brain~5mm ratio={dof*1e3/mouse_brain_mm:.3f}"])
        w.writerow(["7", "peak_z", f"{z_peak*1e3:.6f}", "mm",
                    f"shift_from_geom_focus={z_shift*1e3:+.6f}mm"])
        w.writerow(["8", "max_sidelobe_level", f"{sll_db:.4f}", "dB",
                    f"x={sll_x*1e3:+.4f}mm" if sll_x == sll_x else "none"])
        w.writerow([])
        w.writerow(["9", "target_p_focus[MPa]", "u0[m/s]", "p_surface[MPa]",
                    "note: Vpp needs electro-acoustic efficiency (not computable here)"])
        for pmpa, u0n, psurf in u0_rows:
            w.writerow(["9", f"{pmpa:.2f}", f"{u0n:.6f}", f"{psurf:.6f}", ""])
        w.writerow([])
        w.writerow(["conv", "subdiv", "N_z", "N_source", "G_num", "relerr[%]",
                    "|p|@z7mm"])
        for sd, nz, ns, gnum, err, poff in conv_rows:
            w.writerow(["conv", sd, nz, ns, f"{gnum:.6f}", f"{err:.3e}",
                        f"{poff:.6f}"])
    print(f"\n CSV 保存: {csv_path}")

    # =====================================================================
    #  §I. 図の保存 (PNG)  … 図は .detach().cpu().numpy() 済みで描画
    # =====================================================================
    # 図①: 軸上音圧分布
    fig1, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(z_axis * 1e3, p_axis_abs, "b-", lw=1.5, label="Rayleigh(厳密)")
    on = oneil_axial_abs(z_axis)
    if on.max() > 0:
        ax1.plot(z_axis * 1e3, on / on.max() * G_peak, "g--", lw=1.0,
                 alpha=0.7, label="O'Neil軸上(参考,近軸含む)")
    ax1.axvline(z_focus * 1e3, color="k", ls=":", lw=0.8, label="幾何焦点 10mm")
    ax1.axvline(z_peak * 1e3, color="r", ls="-.", lw=0.8,
                label=f"最大 {z_peak*1e3:.2f}mm")
    if dof == dof:
        ax1.axhline(0.5 * G_peak, color="gray", ls="--", lw=0.6)
        ax1.annotate(f"-6dB DOF={dof*1e3:.2f}mm",
                     ((z6l + z6r) / 2 * 1e3, 0.52 * G_peak), color="gray")
    ax1.set_xlabel("z [mm] (apex=0)"); ax1.set_ylabel("利得 |p|/(ρc u0)")
    ax1.set_title(f"軸上音圧分布 {TAG}"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    fig1.tight_layout(); fig1.savefig(os.path.join(OUT_DIR, f"axial_{TAG}.png"), dpi=150)

    # 図②: 焦点面横方向分布
    fig2, ax2 = plt.subplots(figsize=(7, 4.2))
    ax2.plot(x_lat * 1e3, p_lat_abs, "b-", lw=1.5)
    ax2.axhline(0.5 * p_lat_abs.max(), color="gray", ls="--", lw=0.6,
                label="-6dB")
    if beamw == beamw:
        ax2.annotate(f"-6dB幅={beamw*1e3:.3f}mm",
                     (0, 0.52 * p_lat_abs.max()), ha="center", color="gray")
    ax2.set_xlabel("x [mm]"); ax2.set_ylabel("利得 |p|/(ρc u0)")
    ax2.set_title(f"焦点面(z={z_focus*1e3:.0f}mm)横方向分布 {TAG}")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig2.tight_layout(); fig2.savefig(os.path.join(OUT_DIR, f"lateral_{TAG}.png"), dpi=150)

    # 図③: 2D 音場マップ (dB)
    fig3, ax3 = plt.subplots(figsize=(6.5, 5.5))
    im = ax3.pcolormesh(ZZ * 1e3, XX * 1e3, map_db, cmap="inferno",
                        vmin=-30, vmax=0, shading="auto")
    ax3.axvline(z_focus * 1e3, color="cyan", ls=":", lw=0.8)
    ax3.set_xlabel("z [mm] (apex=0)"); ax3.set_ylabel("x [mm]")
    ax3.set_title(f"音場マップ [dB] {TAG} (peak=0dB)")
    ax3.set_aspect("equal")
    cb = fig3.colorbar(im, ax=ax3, label="正規化音圧 [dB]")
    fig3.tight_layout(); fig3.savefig(os.path.join(OUT_DIR, f"fieldmap_{TAG}.png"), dpi=150)

    plt.close("all")
    print(f" 図保存: axial_{TAG}.png / lateral_{TAG}.png / fieldmap_{TAG}.png")
    print(f"\n 計算時間: {time.time() - t_start:.2f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
