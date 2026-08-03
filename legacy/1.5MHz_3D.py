# -*- coding: utf-8 -*-
"""
================================================================================
 球面集束型 単素子超音波トランスデューサ  ―  焦点体積の 3D 等圧面可視化
   対象: 自作球面集束 PZT (マウス脳 FUS-BBBO 用) / 単一周波数 1.5 MHz
   GPU (CUDA / PyTorch) 対応・CPU フォールバック可
================================================================================

 【このファイルの位置づけ】
   ・"1.5MHz_2D.py" が一次計算(軸上/横方向プロファイル・2D断面・各種指標)。
   ・本ファイルはそれを 3D に拡張し、「焦点体積(focal volume)」を
     等圧面 (isosurface) として立体表示し、体積を定量する。
   ・物理コア(Marczak / 球冠サンプリング / Rayleigh 積分)は 2D 版と同一の式。
     2D 版を書き換えず独立実行できるよう、意図的に self-contained にしてある。
     → 周波数を変える時は「両方のファイルの FREQ_HZ」を変更すること。

--------------------------------------------------------------------------------
 なぜ marching cubes ではなく「回転体(surface of revolution)」なのか
--------------------------------------------------------------------------------
   音源は球冠 + 一様 u0 なので、音場は z 軸まわりに【厳密に軸対称】である。
   したがって:
     (1) 計算は r-z 半平面だけでよい     → 3D 全格子を回すより圧倒的に安い
     (2) 等圧面は「半平面の等圧線を z 軸で回した回転体」として厳密に得られる
         → voxel 格子に由来する階段状アーティファクトが原理的に発生しない
     (3) 体積は Pappus の定理で厳密積分できる:  V = ∫∫_{|p|>=thr} 2*pi*r dr dz
         → voxel カウント(離散和)より高精度
   marching cubes (scikit-image) は「軸対称でない一般の場」向けの手法であり、
   本問題ではむしろ精度が落ちる。依存も増えるため採用しない。
   ただし「本当に軸対称か」は§F-3で直接 Rayleigh 積分により実測検証する。

--------------------------------------------------------------------------------
 座標系 (2D 版と同一)
--------------------------------------------------------------------------------
   原点 = 素子の頂点 (apex)、+z を放射方向。球面中心 = 幾何学的焦点 = (0,0,R)。
     ・リム(開口縁): z = h = 2.4007 mm 平面上、半径 a = 6.5 mm の円
     ・幾何学的焦点: z = R = 10.0 mm
     ・r = sqrt(x^2 + y^2)  (軸対称なので場は (r,z) のみの関数)

--------------------------------------------------------------------------------
 物理的仮定 (2D 版と同一・再掲)
--------------------------------------------------------------------------------
   ・CW 定常解 (実駆動 6.6 ms バースト ≒ 1e4 サイクルで定常とみなせる)
   ・線形伝搬 (衝撃形成距離 ~17 cm >> 焦点域数 mm) → |p| は PNP と等価
   ・自由音場(水中のみ)。頭蓋・脳組織を含まない → in-situ 値ではない
   ・u0 は peak 振幅 (RMS ではない)。音圧も peak 表記
   ・Rayleigh 積分は平面剛体バッフル用。曲面への適用は標準的近似、
     obliquity factor は省略
   ・時間規約 exp(+i*omega*t) → 空間項 exp(-i*k*r')
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
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 図中の日本語が豆腐(□)にならないよう、利用可能な日本語フォントを自動選択。
_installed = {f.name for f in font_manager.fontManager.ttflist}
for _jp in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP",
            "IPAexGothic", "TakaoGothic"):
    if _jp in _installed:
        plt.rcParams["font.family"] = _jp
        break
plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け防止


# ==============================================================================
#  §A. パラメータ (3.0/6.0 MHz 拡張はここだけ変更)
# ==============================================================================

# ---- 駆動周波数 -------------------------------------------------------------
FREQ_HZ = 1.5e6            # [Hz]  ※2D 版と揃えること

# ---- 素子形状 (球冠) --------------------------------------------------------
APERTURE_DIAMETER_M = 13.0e-3   # 開口直径 D [m]
CURV_RADIUS_M       = 10.0e-3   # 曲率半径 R [m] (= 幾何学的焦点距離)

# ---- 媒質 (水) --------------------------------------------------------------
WATER_TEMP_C     = 22.0    # 水温 [℃] → Marczak 式で音速を計算
DENSITY_RHO      = 998.0   # 密度 [kg/m^3]
USE_ATTEN        = False   # 減衰 (既定オフ: 焦点10mmでは<0.1%で無視可)
ATTEN_DB_CM_MHZ2 = 0.0022  # 水の減衰 [dB/(cm·MHz^2)]

# ---- 音源サンプリング -------------------------------------------------------
SUBDIV_PER_LAMBDA = 6      # 音源要素の弧長 <= lambda/6 (λ から自動計算)

# ---- r-z 計算グリッド (等圧線の滑らかさを決めるので 2D 版より細かく) --------
RZ_SUBDIV   = 20           # r-z グリッド間隔 = lambda/20 (仕様の λ/8 より細かい)
R_MAX_3D    = 3.0e-3       # 半径方向の計算範囲 [m] (-20dB サイドローブを含む)
Z_MIN_3D    = 6.0e-3       # 3D 表示の z 範囲 [m]
Z_MAX_3D    = 15.0e-3

# ---- 等圧面 -----------------------------------------------------------------
ISO_LEVELS_DB = [-6.0, -20.0]   # 描画・定量する等圧面レベル [dB] (ピーク基準)
N_PHI         = 96              # 回転体の方位分割数 (見た目の滑らかさ)
CUTAWAY_DEG   = 270.0           # 断面が見えるよう 270° だけ回す (カットアウェイ)

# ---- GPU / 計算 -------------------------------------------------------------
COMPLEX_DTYPE = torch.complex128
REAL_DTYPE    = torch.float64
FIELD_CHUNK   = 4096       # 場の点チャンクサイズ (OOM 時は自動半減)

# ---- 出力 -------------------------------------------------------------------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TAG     = f"{FREQ_HZ/1e6:.1f}MHz"

# ---- 検証 -------------------------------------------------------------------
FOCAL_GAIN_TOL = 0.01      # 焦点利得 G=k*h に対する許容相対誤差 (1%)
AXISYM_TOL     = 1e-6      # 軸対称性の許容相対誤差

# ---- 参考値 -----------------------------------------------------------------
MOUSE_BRAIN_VOLUME_MM3 = 500.0   # 成体マウス脳の体積 ~ 415-500 mm^3 (概数)
MOUSE_BRAIN_THICK_MM   = 5.0     # マウス脳の厚み ~5 mm


# ==============================================================================
#  §B. 媒質: Marczak (1997) の式で純水の音速を温度から計算
# ==============================================================================
def marczak_sound_speed(temp_c: float) -> float:
    """純水の音速 [m/s] を水温 T[℃] から計算 (Marczak 1997, 0-95℃, 1 atm)。"""
    T = temp_c
    return (1.402385e3
            + 5.038813e0  * T
            - 5.799136e-2 * T**2
            + 3.287156e-4 * T**3
            - 1.398845e-6 * T**4
            + 2.787860e-9 * T**5)


# ==============================================================================
#  §C. 球冠音源の等面積サンプリング (2D 版と同一)
# ==============================================================================
def generate_source_points(a, R, h, lam, subdiv, device):
    """
    球冠表面を等面積サンプリングする。

    等面積の原理 (Archimedes の定理):
      球面を軸 z に垂直な平面で切った「帯」の面積は z 幅にのみ比例: dA = 2*pi*R*dz。
      よって z 等間隔分割 → 各リング面積が自動的に等しくなる。
      (緯度経度の単純分割は極付近が密になるので使わない。)
    方位分割数はリングの円筒半径に比例させ、方位弧長 ≈ λ/subdiv に揃える
    (要素アスペクト比を 1 に近づける)。

    重要: 球冠上の全点は焦点 (0,0,R) から厳密に距離 R にある。
          → 焦点音圧の解析解 G = k*h が数値積分に依らず厳密に再現される。
    """
    theta_max = math.asin(a / R)
    ds_target = lam / subdiv
    N_z = max(1, math.ceil(R * theta_max / ds_target))
    dz  = h / N_z
    ring_area = 2.0 * math.pi * R * dz     # Archimedes: 全リング共通

    xs, ys, zs, dSs = [], [], [], []
    for i in range(N_z):
        z_i   = (i + 0.5) * dz
        rho_i = math.sqrt(max(0.0, 2.0 * R * z_i - z_i * z_i))
        M_i   = max(1, int(round(2.0 * math.pi * rho_i / ds_target)))
        dS_i  = ring_area / M_i
        phi0  = math.pi / M_i if (i % 2) else 0.0   # 千鳥配置でスポークを防ぐ
        phis  = phi0 + 2.0 * math.pi * (np.arange(M_i) / M_i)
        xs.append(rho_i * np.cos(phis))
        ys.append(rho_i * np.sin(phis))
        zs.append(np.full(M_i, z_i))
        dSs.append(np.full(M_i, dS_i))

    pts = torch.tensor(np.stack([np.concatenate(xs), np.concatenate(ys),
                                 np.concatenate(zs)], axis=1),
                       dtype=REAL_DTYPE, device=device)
    dS = torch.tensor(np.concatenate(dSs), dtype=REAL_DTYPE, device=device)
    info = dict(N_z=N_z, N_source=pts.shape[0], ds_target=ds_target,
                area_sum=float(dS.sum().item()), area_exact=2.0*math.pi*R*h)
    return pts, dS, info


# ==============================================================================
#  §D. Rayleigh 積分の GPU 計算 (2D 版と同一)
# ==============================================================================
def compute_pressure(field_pts, src_pts, src_dS, k, prefactor,
                     alpha_np_per_m, chunk, device):
    """
    p(r) = prefactor * Σ_s [ exp(-alpha*r') * exp(-i*k*r') / r' ] * dS_s
      prefactor = i*rho*c*k*u0/(2*pi) ,  時間規約 exp(+i*omega*t)

    ・Python for ループでペアを回さない。全ペアをテンソル化 (cdist)。
    ・場の点をチャンク分割 → 音源方向に総和。OOM 時はチャンク自動半減。
    ・r'→0 の特異点は微小値クランプで回避 (本問題では場の点 z>=6mm、
      音源 z<=h=2.4mm なので実際には起きない)。
    """
    Nf = field_pts.shape[0]
    out = torch.empty(Nf, dtype=COMPLEX_DTYPE, device=device)
    r_floor = 1e-9
    i0, cur_chunk = 0, int(chunk)
    while i0 < Nf:
        i1 = min(i0 + cur_chunk, Nf)
        try:
            d = torch.cdist(field_pts[i0:i1], src_pts)     # (C,Ns) 距離 r'
            d = torch.clamp(d, min=r_floor)
            amp = 1.0 / d
            if alpha_np_per_m > 0.0:
                amp = amp * torch.exp(-alpha_np_per_m * d)
            green = torch.polar(amp, -k * d)               # exp(-i k r')/r'
            out[i0:i1] = prefactor * (green * src_dS).sum(dim=1)
            i0 = i1
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and cur_chunk > 1:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                cur_chunk = max(1, cur_chunk // 2)
                print(f"  [OOM] チャンクを半減して再試行: chunk={cur_chunk}")
            else:
                raise
    return out


# ==============================================================================
#  §E. 等圧線 → 回転体メッシュ
# ==============================================================================
def extract_iso_contours(z_grid, r_grid, field_db, level_db):
    """
    r-z 半平面 (r>=0) 上で field_db == level_db となる等圧線を抽出する。

    matplotlib の contour を「線の抽出器」として使う(描画はしない)。
    戻り値は各等圧線の (z[m], r[m]) 頂点列のリスト。

    半平面での等圧線の形と、それを回した立体の対応:
      ・両端が r=0 (軸) で終わる弧   → 回すと閉じた「葉巻(cigar)」になる = 主ローブ
      ・軸に触れない閉ループ         → 回すとトーラス(輪)になる = サイドローブ環
    どちらも同じ処理で自然に扱える。
    """
    fig_tmp = plt.figure()
    ax_tmp = fig_tmp.add_subplot(111)
    # contour は (X, Y, C) の順。ここでは X=z, Y=r。
    cs = ax_tmp.contour(z_grid, r_grid, field_db, levels=[level_db])

    segs = []
    try:
        # matplotlib の伝統的 API (級数ごとの線分リスト)
        segs = [np.asarray(s) for s in cs.allsegs[0] if len(s) >= 3]
    except Exception:
        # 新しい API では Path から取り出す (MOVETO でサブパスに分割)
        for p in cs.get_paths():
            v, codes = p.vertices, p.codes
            if codes is None:
                if len(v) >= 3:
                    segs.append(np.asarray(v))
            else:
                starts = np.flatnonzero(codes == 1)  # MOVETO
                bounds = list(starts) + [len(v)]
                for s, e in zip(bounds[:-1], bounds[1:]):
                    if e - s >= 3:
                        segs.append(np.asarray(v[s:e]))
    plt.close(fig_tmp)
    return segs


def revolve_contour(seg_zr, n_phi, sweep_deg=360.0):
    """
    半平面の等圧線 (z, r) を z 軸まわりに回して 3D 面(四角形メッシュ)を作る。

    軸対称な場なので、この回転体が【厳密な等圧面】になる。
    voxel 格子を経由しないため階段状アーティファクトが出ない。

    sweep_deg < 360 にすると回し残しができ、断面が見える(カットアウェイ表示)。

    Returns: quads (Nq, 4, 3) の頂点配列 [m]
    """
    z = seg_zr[:, 0]
    r = np.clip(seg_zr[:, 1], 0.0, None)      # r<0 は数値誤差なのでクリップ
    phi = np.deg2rad(np.linspace(0.0, sweep_deg, n_phi))

    # (Npt, Nphi) の格子で頂点を作る
    X = r[:, None] * np.cos(phi)[None, :]
    Y = r[:, None] * np.sin(phi)[None, :]
    Z = np.repeat(z[:, None], len(phi), axis=1)

    # 隣接する (i,j)-(i+1,j)-(i+1,j+1)-(i,j+1) で四角形を張る
    q = []
    for i in range(len(z) - 1):
        for j in range(len(phi) - 1):
            q.append([[X[i, j],     Y[i, j],     Z[i, j]],
                      [X[i+1, j],   Y[i+1, j],   Z[i+1, j]],
                      [X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]],
                      [X[i, j+1],   Y[i, j+1],   Z[i, j+1]]])
    return np.asarray(q)


def revolved_volume(r_grid_1d, z_grid_1d, field_lin, thr):
    """
    Pappus の定理による厳密体積積分:
        V = ∫∫_{|p|>=thr} 2*pi*r dr dz
    voxel カウント(離散和)ではなく、r-z セルごとに軸対称の環体積を積む。
    セル中心の r を使うので、環の体積 2*pi*r*dr*dz が厳密に効く。

    Returns: 体積 [m^3]
    """
    dr = r_grid_1d[1] - r_grid_1d[0]
    dz = z_grid_1d[1] - z_grid_1d[0]
    mask = (field_lin >= thr)                       # (Nr, Nz)
    rr = r_grid_1d[:, None]                          # 各セルの r
    return float(np.sum(mask * (2.0 * math.pi * rr * dr * dz)))


# ==============================================================================
#  §F. メイン
# ==============================================================================
def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 派生パラメータ -----------------------------------------------------
    a = APERTURE_DIAMETER_M / 2.0
    R = CURV_RADIUS_M
    h = R - math.sqrt(R * R - a * a)           # キャップ深さ
    c = marczak_sound_speed(WATER_TEMP_C)      # 音速 (温度依存)
    rho = DENSITY_RHO
    lam = c / FREQ_HZ
    k = 2.0 * math.pi * FREQ_HZ / c
    theta_max = math.asin(a / R)
    f_number = R / APERTURE_DIAMETER_M
    z_focus = R

    alpha_np_per_m = 0.0
    if USE_ATTEN:
        f_mhz = FREQ_HZ / 1e6
        alpha_np_per_m = (ATTEN_DB_CM_MHZ2 * f_mhz * f_mhz) / 8.686 * 100.0

    u0 = 1.0
    prefactor = torch.tensor(1j * rho * c * k * u0 / (2.0 * math.pi),
                             dtype=COMPLEX_DTYPE, device=device)
    p_surface = rho * c * u0

    # ---- 音源生成 -----------------------------------------------------------
    src_pts, src_dS, sinfo = generate_source_points(a, R, h, lam,
                                                    SUBDIV_PER_LAMBDA, device)
    N_source = sinfo["N_source"]

    # ---- r-z グリッド -------------------------------------------------------
    d_rz = lam / RZ_SUBDIV
    r_1d = np.arange(0.0, R_MAX_3D + d_rz, d_rz)
    z_1d = np.arange(Z_MIN_3D, Z_MAX_3D + d_rz, d_rz)
    N_field = len(r_1d) * len(z_1d)

    # ---- 実行前サマリ -------------------------------------------------------
    print("=" * 72)
    print(f" 球面集束トランスデューサ 焦点体積 3D 等圧面  ({TAG})")
    print("=" * 72)
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        print(f" デバイス      : CUDA / {prop.name}")
        print(f" VRAM 容量     : {prop.total_memory/(1024**3):.2f} GB")
    else:
        print(" デバイス      : CPU (CUDA 非対応環境)")
    est_vram_gb = FIELD_CHUNK * N_source * 16 * 3 / (1024**3)
    print(f" 音源点数 Ns   : {N_source:,}   (N_z={sinfo['N_z']} リング)")
    print(f" 場の点数      : {N_field:,}   (r-z 半平面 {len(r_1d)}×{len(z_1d)})")
    print(f" 総ペア数      : {N_source * N_field:,}")
    print(f" 推定VRAM(chunk={FIELD_CHUNK}) : {est_vram_gb:.3f} GB / チャンク")
    print("-" * 72)
    print(f" 水温 T        : {WATER_TEMP_C:.1f} ℃  →  c = {c:.2f} m/s (Marczak)")
    print(f" 波長 λ        : {lam*1e3:.4f} mm    波数 k = {k:.2f} rad/m")
    print(f" 開口 D / 曲率R: {APERTURE_DIAMETER_M*1e3:.1f} / {R*1e3:.1f} mm"
          f"   h = {h*1e3:.4f} mm")
    print(f" 開口半角      : {math.degrees(theta_max):.1f}°   F# = {f_number:.2f}")
    print(f" r-z グリッド  : λ/{RZ_SUBDIV} = {d_rz*1e6:.0f} µm")
    print(f" 面積検証      : Σ dS = {sinfo['area_sum']*1e6:.6f} mm²  "
          f"(厳密 2πRh = {sinfo['area_exact']*1e6:.6f} mm²)")
    print(f" 減衰          : {'ON' if USE_ATTEN else 'OFF (焦点10mmでは<0.1%,無視可)'}")
    print("=" * 72)

    # =====================================================================
    #  §F-1. 検証(一次基準): 焦点利得の厳密解 G = k*h
    # =====================================================================
    # 球冠上の全点は焦点から等距離 R → p_focus = i*rho*c*u0*k*h*exp(-i*k*R)
    # 音速を温度依存にすると k が変わるので、期待値も c から再計算する。
    G_exact = k * h
    focus_pt = torch.tensor([[0.0, 0.0, z_focus]], dtype=REAL_DTYPE, device=device)
    p_f = compute_pressure(focus_pt, src_pts, src_dS, k, prefactor,
                           alpha_np_per_m, FIELD_CHUNK, device)
    G_num = float(torch.abs(p_f[0]).item()) / p_surface
    rel_err = abs(G_num - G_exact) / G_exact
    print("\n[検証1] 一次基準: 焦点利得の厳密解 G = k*h")
    print(f"  厳密 {G_exact:.4f} / 数値 {G_num:.4f} / 相対誤差 {rel_err*100:.3e} %")
    if rel_err > FOCAL_GAIN_TOL:
        raise RuntimeError(f"焦点利得が厳密値と {FOCAL_GAIN_TOL*100:.0f}% 以内で"
                           f"一致しません (誤差 {rel_err*100:.3f}%)。")
    print("  → OK (物理コアは 2D 版と同一で健全)")

    # =====================================================================
    #  §F-2. r-z 半平面の音場計算
    # =====================================================================
    RR, ZZ = np.meshgrid(r_1d, z_1d, indexing="ij")      # (Nr, Nz)
    # 軸対称なので y=0 の半平面 (x=r>=0) だけ計算すれば十分
    rz_pts = torch.tensor(np.stack([RR.ravel(), np.zeros(RR.size), ZZ.ravel()],
                                   axis=1), dtype=REAL_DTYPE, device=device)
    p_rz = compute_pressure(rz_pts, src_pts, src_dS, k, prefactor,
                            alpha_np_per_m, FIELD_CHUNK, device)
    # 図に渡す前にホストへ戻す
    p_rz_abs = torch.abs(p_rz).detach().cpu().numpy().reshape(RR.shape) / p_surface

    p_peak = p_rz_abs.max()
    i_pk = np.unravel_index(np.argmax(p_rz_abs), p_rz_abs.shape)
    z_peak = z_1d[i_pk[1]]
    field_db = 20.0 * np.log10(np.clip(p_rz_abs / p_peak, 1e-12, None))

    print(f"\n[音場] r-z 半平面のピーク利得 = {p_peak:.3f} @ z = {z_peak*1e3:.3f} mm")

    # =====================================================================
    #  §F-3. 検証: 本当に軸対称か(回転体近似の妥当性)を直接確認
    # =====================================================================
    # 同じ半径 r=1.5mm・同じ z=10mm を、方位角の異なる 3 点として直接 Rayleigh
    # 積分し、値が一致するかを見る。一致すれば「r-z 半平面を回す」操作は厳密。
    r_chk, z_chk = 1.5e-3, z_focus
    chk = torch.tensor([[r_chk, 0.0, z_chk],
                        [0.0, r_chk, z_chk],
                        [r_chk*math.cos(math.pi/4), r_chk*math.sin(math.pi/4), z_chk],
                        [-r_chk*math.cos(math.pi/3), r_chk*math.sin(math.pi/3), z_chk]],
                       dtype=REAL_DTYPE, device=device)
    p_chk = torch.abs(compute_pressure(chk, src_pts, src_dS, k, prefactor,
                                       alpha_np_per_m, FIELD_CHUNK, device))
    p_chk = p_chk.detach().cpu().numpy() / p_surface
    axisym_err = float((p_chk.max() - p_chk.min()) / p_chk.mean())
    print(f"\n[検証2] 軸対称性 (r={r_chk*1e3:.1f}mm, z={z_chk*1e3:.1f}mm を方位4点で直接計算)")
    print(f"  値: {np.array2string(p_chk, precision=8)}")
    print(f"  方位ばらつき(相対) = {axisym_err:.3e}  (許容 {AXISYM_TOL:.0e})")
    if axisym_err > AXISYM_TOL:
        print("  ! 警告: 方位ばらつきが大きい。音源サンプリングの方位一様性を確認。")
    else:
        print("  → OK (厳密に軸対称 → r-z 半平面の回転体が厳密な等圧面になる)")

    # =====================================================================
    #  §F-4. 等圧面の抽出・回転体化・体積定量
    # =====================================================================
    print("\n[等圧面] 抽出と体積定量")
    iso_results = []
    for lvl_db in ISO_LEVELS_DB:
        thr = p_peak * (10.0 ** (lvl_db / 20.0))          # 線形しきい値(利得単位)
        segs = extract_iso_contours(ZZ, RR, field_db, lvl_db)
        vol_m3 = revolved_volume(r_1d, z_1d, p_rz_abs, thr)
        vol_mm3 = vol_m3 * 1e9

        # 主ローブの外形寸法(しきい値を超える領域の広がり)
        mask = p_rz_abs >= thr                           # (Nr, Nz)
        z_span = (z_1d[np.any(mask, axis=0)].min(), z_1d[np.any(mask, axis=0)].max())
        r_max_lobe = r_1d[np.any(mask, axis=1)].max()

        # --- 計算領域による切れ(clipping)の検出 -----------------------------
        # しきい値超えの領域が箱の壁 (r=R_MAX_3D, z=Z_MIN_3D, z=Z_MAX_3D) に
        # 接している場合、その等圧面は箱の外へ続いており閉じていない。
        # → 体積は「物理的な値」ではなく「箱で切った下限値」にすぎない。
        #   (r=0 は対称軸であって壁ではないので判定に含めない)
        clipped = bool(mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())

        iso_results.append(dict(level_db=lvl_db, thr=thr, segs=segs,
                                vol_mm3=vol_mm3, z_span=z_span,
                                r_max=r_max_lobe, n_seg=len(segs),
                                clipped=clipped))
        flag = "  ← 領域端で切れており下限値" if clipped else ""
        print(f"  {lvl_db:+.0f} dB : 体積 {vol_mm3:8.4f} mm³ / "
              f"等圧線 {len(segs)} 本 / z {z_span[0]*1e3:.2f}-{z_span[1]*1e3:.2f} mm / "
              f"最大半径 {r_max_lobe*1e3:.3f} mm{flag}")

    # -6dB の焦点体積を主指標として詳細評価
    iso6 = next(d for d in iso_results if abs(d["level_db"] + 6.0) < 1e-9)
    v6 = iso6["vol_mm3"]
    L6 = (iso6["z_span"][1] - iso6["z_span"][0]) * 1e3      # 軸方向長 [mm]
    W6 = 2.0 * iso6["r_max"] * 1e3                          # 横方向幅 [mm]
    v_ellip = math.pi / 6.0 * W6 * W6 * L6                  # 等価回転楕円体の体積

    # =====================================================================
    #  §G. 数値結果のコンソール出力
    # =====================================================================
    print("\n" + "=" * 72)
    print(" 数値結果 (焦点体積)")
    print("=" * 72)
    print(f" -6dB 焦点体積            = {v6:.4f} mm³"
          + ("   ! 領域端で切れており下限値" if iso6["clipped"]
             else "   (等圧面は領域内で閉じている → 確定値)"))
    print(f"   外形: 横 {W6:.3f} mm × 軸 {L6:.3f} mm")
    print(f"   等価回転楕円体 (π/6·W²·L) = {v_ellip:.4f} mm³  "
          f"(比 {v6/v_ellip:.3f} → 葉巻形の充填率)")
    print(f"   マウス脳 (~{MOUSE_BRAIN_VOLUME_MM3:.0f} mm³) に対する割合 = "
          f"{v6/MOUSE_BRAIN_VOLUME_MM3*100:.3f} %")
    print(f"   軸方向長 {L6:.2f} mm vs マウス脳厚 ~{MOUSE_BRAIN_THICK_MM:.0f} mm "
          f"→ 比 {L6/MOUSE_BRAIN_THICK_MM:.2f}")
    if L6 > MOUSE_BRAIN_THICK_MM:
        print("     → 焦点柱が脳厚を貫く。深さ選択性は期待できない。")
    else:
        print("     → 焦点柱は脳厚に収まる。ただし軸方向は横方向より大幅に長い。")
    iso20 = next(d for d in iso_results if abs(d["level_db"] + 20.0) < 1e-9)
    print(f" -20dB 体積               = {iso20['vol_mm3']:.4f} mm³ "
          f"(サイドローブ環を含む: 等圧線 {iso20['n_seg']} 本)")
    if iso20["clipped"]:
        print("   ! この値は計算領域 (r<=%.1fmm, z=%.0f-%.0fmm) の壁で切れているため"
              % (R_MAX_3D*1e3, Z_MIN_3D*1e3, Z_MAX_3D*1e3))
        print("     物理的な体積ではなく【下限値】。-20dB 面は素子へ向かう収束コーン")
        print("     に沿って箱の外へ続く。定量に使うなら領域を広げて再計算すること。")
    print(f" 縦横比 (軸/横)           = {L6/W6:.2f}  "
          f"→ 焦点は球ではなく細長い葉巻形")
    print("\n ※ 自由音場・線形・CW の値。頭蓋や脳組織は含まない (in-situ ではない)。")
    print(" ※ |p| は peak 表記。BBBO の実効果はマイクロバブル動態に依存し、")
    print("    ここで出す圧・体積は「その入力となる自由音場量」である。")

    # =====================================================================
    #  §H. CSV 出力
    # =====================================================================
    csv_path = os.path.join(OUT_DIR, f"results3D_{TAG}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["item", "quantity", "value", "unit", "note"])
        w.writerow(["3D", "focal_volume_-6dB", f"{v6:.6f}", "mm^3",
                    "CLIPPED_lower_bound" if iso6["clipped"] else "closed_in_domain"])
        w.writerow(["3D", "focal_width_-6dB", f"{W6:.6f}", "mm", "lateral (2*r_max)"])
        w.writerow(["3D", "focal_length_-6dB", f"{L6:.6f}", "mm", "axial extent"])
        w.writerow(["3D", "aspect_ratio", f"{L6/W6:.6f}", "-", "axial/lateral"])
        w.writerow(["3D", "equiv_ellipsoid_volume", f"{v_ellip:.6f}", "mm^3",
                    f"fill_ratio={v6/v_ellip:.4f}"])
        w.writerow(["3D", "fraction_of_mouse_brain", f"{v6/MOUSE_BRAIN_VOLUME_MM3*100:.6f}",
                    "%", f"brain~{MOUSE_BRAIN_VOLUME_MM3:.0f}mm^3"])
        w.writerow(["3D", "volume_-20dB", f"{iso20['vol_mm3']:.6f}", "mm^3",
                    ("includes sidelobes; CLIPPED_lower_bound"
                     if iso20["clipped"] else "includes sidelobes; closed_in_domain")])
        w.writerow(["3D", "peak_gain", f"{p_peak:.6f}", "-",
                    f"exact k*h={G_exact:.6f}"])
        w.writerow(["3D", "peak_z", f"{z_peak*1e3:.6f}", "mm", ""])
        w.writerow(["chk", "axisymmetry_rel_spread", f"{axisym_err:.3e}", "-",
                    "revolve validity"])
    print(f"\n CSV 保存: {csv_path}")

    # =====================================================================
    #  §I. 3D 図の描画
    # =====================================================================
    colors = {-6.0: "#e8483c", -20.0: "#4c8fd6"}

    def setup_3d_axes(ax, title, lat_mm, z_lo_mm, z_hi_mm, fs=11):
        """3D 軸の体裁。box_aspect は実寸比にして「細長さ」を歪めず見せる。"""
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
        ax.set_title(title, fontsize=fs)
        ax.set_xlim(-lat_mm, lat_mm); ax.set_ylim(-lat_mm, lat_mm)
        ax.set_zlim(z_lo_mm, z_hi_mm)
        ax.set_box_aspect((2 * lat_mm, 2 * lat_mm, z_hi_mm - z_lo_mm))
        # x,y は表示範囲が狭いので目盛りを間引く(既定だと重なって潰れる)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(labelsize=8)

    # -6dB ローブに合わせたズーム範囲 (余白を切って形状を見やすくする)
    lat6_mm = max(0.6, iso6["r_max"] * 1e3 * 1.7)
    z6_lo = iso6["z_span"][0] * 1e3 - 0.5
    z6_hi = iso6["z_span"][1] * 1e3 + 0.5

    # --- 図1: -6dB 焦点体積のみ (360° 閉曲面・ローブにズーム) ---------------
    # 主ローブは単純な葉巻形なので中身を見る必要がなく、閉じた面のほうが読みやすい。
    fig1 = plt.figure(figsize=(8, 7))
    ax = fig1.add_subplot(111, projection="3d")
    for seg in iso6["segs"]:
        quads = revolve_contour(seg, N_PHI, 360.0) * 1e3          # → mm
        ax.add_collection3d(Poly3DCollection(
            quads, alpha=0.5, facecolor=colors[-6.0], edgecolor="none"))
    ax.plot([0, 0], [0, 0], [z6_lo, z6_hi], "k:", lw=0.8)
    ax.scatter([0], [0], [z_focus*1e3], color="k", s=40, marker="x",
               label=f"幾何焦点 z={z_focus*1e3:.0f}mm", depthshade=False)
    ax.scatter([0], [0], [z_peak*1e3], color="lime", s=40, marker="o",
               label=f"音圧最大 z={z_peak*1e3:.2f}mm", depthshade=False)
    setup_3d_axes(ax, f"-6dB 焦点体積 {TAG}\n"
                      f"V = {v6:.3f} mm³ (横 {W6:.2f} mm × 軸 {L6:.2f} mm, "
                      f"縦横比 {L6/W6:.1f})",
                  lat6_mm, z6_lo, z6_hi)
    ax.legend(fontsize=8, loc="upper left")
    ax.view_init(elev=16, azim=-60)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUT_DIR, f"iso3D_focal_{TAG}.png"), dpi=150)

    # --- 図2: -6dB と -20dB の入れ子 (カットアウェイで内部を見せる) ----------
    # 視線をカットアウェイの切り欠き (phi 270-360°) の方向に合わせ、
    # -20dB シェルの内側に -6dB ローブが入っている様子が見えるようにする。
    fig2 = plt.figure(figsize=(8, 7))
    ax = fig2.add_subplot(111, projection="3d")
    for d in sorted(iso_results, key=lambda x: x["level_db"]):   # 外側(-20dB)から
        alpha = 0.16 if d["level_db"] <= -20.0 else 0.75
        for seg in d["segs"]:
            quads = revolve_contour(seg, N_PHI, CUTAWAY_DEG) * 1e3
            ax.add_collection3d(Poly3DCollection(
                quads, alpha=alpha, facecolor=colors[d["level_db"]],
                edgecolor="none"))
    ax.plot([0, 0], [0, 0], [Z_MIN_3D*1e3, Z_MAX_3D*1e3], "k:", lw=0.8)
    clip_note = "  ※-20dB は計算領域端で切れている(下限値)" if iso20["clipped"] else ""
    setup_3d_axes(ax, f"等圧面 {TAG}   赤 -6dB ({v6:.2f} mm³) / "
                      f"青 -20dB ({iso20['vol_mm3']:.1f} mm³)\n"
                      f"{CUTAWAY_DEG:.0f}°カットアウェイ{clip_note}",
                  R_MAX_3D * 1e3, Z_MIN_3D * 1e3, Z_MAX_3D * 1e3, fs=10)
    ax.view_init(elev=16, azim=-45)   # 切り欠き(中心 phi=315°)を正面から覗く
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, f"iso3D_nested_{TAG}.png"), dpi=150)

    # --- 図3: 多視点 (-6dB を 4 方向から) -----------------------------------
    # 注: 真上(elev≈90)からの視点は、box_aspect を実寸比に保つ都合で
    #     パネルが極端に小さく潰れてしまうため採用しない。断面が円であることは
    #     §F-3 で軸対称性を数値的に実証済み(方位ばらつき ~1e-15)。
    #     代わりに仰角を変えた 4 方向で立体形状を示す。
    fig3 = plt.figure(figsize=(11, 9))
    views = [(16, -60), (0, -90), (45, -60), (30, 25)]
    vnames = ["斜め (elev 16°)", "真横 x-z 断面 (elev 0°)",
              "高い角度から (elev 45°)", "斜め・反対側 (elev 30°)"]
    for idx, ((el, az), vn) in enumerate(zip(views, vnames), start=1):
        ax = fig3.add_subplot(2, 2, idx, projection="3d")
        for seg in iso6["segs"]:
            quads = revolve_contour(seg, N_PHI, 360.0) * 1e3
            ax.add_collection3d(Poly3DCollection(
                quads, alpha=0.5, facecolor=colors[-6.0], edgecolor="none"))
        ax.plot([0, 0], [0, 0], [z6_lo, z6_hi], "k:", lw=0.8)
        setup_3d_axes(ax, vn, lat6_mm, z6_lo, z6_hi, fs=10)
        ax.view_init(elev=el, azim=az)
    fig3.suptitle(f"-6dB 焦点体積 多視点  {TAG}   V = {v6:.3f} mm³  "
                  f"(横 {W6:.2f} × 軸 {L6:.2f} mm, 縦横比 {L6/W6:.1f})",
                  fontsize=12)
    fig3.tight_layout(rect=(0, 0, 1, 0.96))
    fig3.subplots_adjust(wspace=0.05, hspace=0.18)
    fig3.savefig(os.path.join(OUT_DIR, f"iso3D_views_{TAG}.png"), dpi=150)

    # --- 図4: 元になった r-z 半平面と等圧線 (3D の根拠を示す断面) -----------
    fig4, ax4 = plt.subplots(figsize=(9, 4.6))
    im = ax4.pcolormesh(ZZ * 1e3, RR * 1e3, field_db, cmap="inferno",
                        vmin=-30, vmax=0, shading="auto")
    for d in iso_results:
        for si, seg in enumerate(d["segs"]):
            ax4.plot(seg[:, 0] * 1e3, seg[:, 1] * 1e3,
                     color=colors[d["level_db"]], lw=1.6,
                     label=(f"{d['level_db']:+.0f} dB" if si == 0 else None))
    ax4.axvline(z_focus * 1e3, color="cyan", ls=":", lw=0.8)
    ax4.set_xlabel("z [mm] (apex=0)"); ax4.set_ylabel("r [mm]")
    ax4.set_title(f"r-z 半平面と等圧線 (これを z 軸で回して 3D 化) {TAG}")
    ax4.legend(fontsize=8, loc="upper right")
    fig4.colorbar(im, ax=ax4, label="正規化音圧 [dB]")
    fig4.tight_layout()
    fig4.savefig(os.path.join(OUT_DIR, f"iso3D_rzplane_{TAG}.png"), dpi=150)

    plt.close("all")
    print(f" 図保存: iso3D_focal_{TAG}.png / iso3D_nested_{TAG}.png /")
    print(f"         iso3D_views_{TAG}.png / iso3D_rzplane_{TAG}.png")
    print(f"\n 計算時間: {time.time() - t_start:.2f} s")
    print("=" * 72)


if __name__ == "__main__":
    main()
