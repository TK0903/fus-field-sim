# -*- coding: utf-8 -*-
"""
================================================================================
 設定ファイル  —  ここだけ編集すれば全計算の条件が変わる
================================================================================
 球面集束型 単素子超音波トランスデューサ (マウス脳 FUS-BBBO 用 自作 PZT)

 使い方:
   ・下の「ユーザー設定」を書き換えて  python main.py both
   ・一時的に変えるだけなら CLI で上書きできる:
        python main.py both --freq 3.0 --coating 50
     (--freq は MHz、--coating は µm)

 計算スクリプト (fus/sim2d.py, fus/sim3d.py) は両方ともこの設定を参照するので、
 2D と 3D で条件がずれることはない。
================================================================================
"""

from dataclasses import dataclass
from pathlib import Path
import math

# ==============================================================================
#  ■ ユーザー設定
# ==============================================================================

# ---- 駆動周波数 [MHz] -------------------------------------------------------
# 1.5 / 3.0 / 6.0 など。数字を変えるだけで波長・グリッド・期待利得が自動追従する。
FREQ_MHZ = 1.5

# ---- コーティング厚 [µm] ----------------------------------------------------
# 素子の凹面に塗る整合層・保護層の厚さ。既定 0 = コーティング無し。
# 厚さを入れると放射面が焦点側へ寄るので、実効的な曲率半径 R が R-t になる。
#
#  【重要な注意】これは「放射面が t だけ移動する」という幾何的な変化だけを
#  モデル化している。整合層の音響的な効果 —— 透過率・位相回転・インピーダンス
#  整合・層自体の共振 —— は一切含まない。実際の整合層の主効果は送信効率側に
#  出るため、ここで見えるのは「焦点が t だけ手前に来て利得 k*h がわずかに
#  変わる」ところまでである。効率や帯域の評価には使えない。
COATING_THICKNESS_UM = 0.0

# コーティングで形状をどう変えるかのモデル:
#   "conformal"     : 開口半角を保存 (R→R-t, a→a*(R-t)/R)。一様な等厚コートを想定。
#   "keep_aperture" : 開口半径 a を保存 (R→R-t, a はそのまま)。
#                     コートが素子リムを越えて広がらない場合を想定。
COATING_MODE = "conformal"

# ---- 素子形状 (コーティング無しの素の PZT) ----------------------------------
APERTURE_DIAMETER_MM = 13.0   # 開口直径 D [mm]  (開口半径 a = 6.5 mm)
CURV_RADIUS_MM       = 10.0   # 曲率半径 R [mm]  (= 幾何学的焦点距離)

# ---- 媒質 (水) --------------------------------------------------------------
WATER_TEMP_C     = 22.0    # 水温 [℃]。音速は Marczak の式で温度から計算する。
DENSITY_RHO      = 998.0   # 密度 [kg/m^3]
USE_ATTEN        = False   # 減衰を入れるか (既定オフ)
ATTEN_DB_CM_MHZ2 = 0.0022  # 水の減衰 [dB/(cm·MHz^2)]
# 注: 焦点距離 10 mm と短く、1.5 MHz では減衰 <0.1 % で無視できる。
# 注: 脱気水と通常水でバルク音響特性は変わらない(溶存気体は測定実務の問題)。

# ---- 数値積分 ---------------------------------------------------------------
SUBDIV_PER_LAMBDA       = 6    # 音源要素の弧長 <= λ/この値
FIELD_SUBDIV_PER_LAMBDA = 8    # 2D の場のグリッド間隔 = λ/この値
RZ_SUBDIV               = 20   # 3D の r-z グリッド間隔 = λ/この値 (等圧線用に細かめ)
FIELD_CHUNK             = 4096 # 場の点のチャンクサイズ (OOM 時は自動半減)

# ---- 2D 計算範囲 [mm] -------------------------------------------------------
AXIAL_Z_MIN_MM  = 3.0    # 軸上スキャン開始 (z < h はお椀の内側なので除外)
AXIAL_Z_MAX_MM  = 18.0   # 軸上スキャン終了
LATERAL_HALF_MM = 3.0    # 横方向 ±この値

# ---- 3D 計算範囲 [mm] -------------------------------------------------------
R_MAX_3D_MM = 3.0        # 半径方向の計算範囲
Z_MIN_3D_MM = 6.0        # z 範囲 下限
Z_MAX_3D_MM = 15.0       # z 範囲 上限
ISO_LEVELS_DB = [-6.0, -20.0]   # 等圧面のレベル [dB] (ピーク基準)
N_PHI         = 96              # 回転体の方位分割数 (見た目の滑らかさ)
CUTAWAY_DEG   = 270.0           # 断面が見えるよう回す角度 (カットアウェイ)

# ---- 検証 -------------------------------------------------------------------
FOCAL_GAIN_TOL = 0.01    # 焦点利得 G=k*h に対する許容相対誤差 (1%)
AXISYM_TOL     = 1e-6    # 軸対称性の許容相対誤差
CONV_SUBDIVS   = [3, 4, 6, 8, 12]   # グリッド収束テストで振る分割数

# ---- 目標音圧 ---------------------------------------------------------------
TARGET_PRESSURES_MPA = [0.20, 0.40, 0.57]   # u0 を逆算する目標焦点音圧 [MPa]

# ---- 参考値 -----------------------------------------------------------------
MOUSE_BRAIN_VOLUME_MM3 = 500.0   # 成体マウス脳の体積 ~415-500 mm^3 (概数)
MOUSE_BRAIN_THICK_MM   = 5.0     # マウス脳の厚み ~5 mm

# ---- 出力先 -----------------------------------------------------------------
RESULTS_DIRNAME = "results"      # 計算結果を貯めるフォルダ名


# ==============================================================================
#  ■ 以下は派生量の計算。通常は編集不要。
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent


def marczak_sound_speed(temp_c: float) -> float:
    """
    純水の音速 [m/s] を水温 T[℃] から計算 (Marczak 1997, 0-95℃, 1 atm)。
    定数を直書きせず温度依存にするのは、λ が変わると期待焦点利得 G=k*h も
    変わるため(検証で c から期待値を再計算するのに使う)。
    """
    T = temp_c
    return (1.402385e3
            + 5.038813e0  * T
            - 5.799136e-2 * T**2
            + 3.287156e-4 * T**3
            - 1.398845e-6 * T**4
            + 2.787860e-9 * T**5)


@dataclass(frozen=True)
class Config:
    """1 回の計算に使う条件一式(SI 単位)。派生量まで含めて凍結してある。"""
    # --- 入力 (人間が読む単位) ---
    freq_mhz: float
    coating_um: float
    coating_mode: str
    water_temp_c: float
    # --- 素の形状 [m] ---
    a0: float          # コート無しの開口半径
    R0: float          # コート無しの曲率半径
    # --- 実効形状 (コート反映後) [m] ---
    a: float           # 実効 開口半径
    R: float           # 実効 曲率半径 (= 幾何学的焦点距離)
    h: float           # 実効 キャップ深さ
    theta_max: float   # 開口半角 [rad]
    f_number: float    # F 値 = R / (2a)
    # --- 媒質 ---
    c: float           # 音速 [m/s]
    rho: float         # 密度 [kg/m^3]
    lam: float         # 波長 [m]
    k: float           # 波数 [rad/m]
    alpha_np_per_m: float   # 減衰 [Np/m] (0 なら無効)
    use_atten: bool
    # --- 数値 ---
    subdiv_per_lambda: int
    field_subdiv_per_lambda: int
    rz_subdiv: int
    field_chunk: int
    # --- 範囲 [m] ---
    axial_z_min: float
    axial_z_max: float
    lateral_half: float
    r_max_3d: float
    z_min_3d: float
    z_max_3d: float
    # --- 出力 ---
    run_name: str
    out_dir: Path
    # --- 設定チェックで出た警告 (main が Tee 設定後に表示する) ---
    warnings: tuple = ()

    # ---- 便利プロパティ ----------------------------------------------------
    @property
    def freq_hz(self) -> float:
        return self.freq_mhz * 1e6

    @property
    def p_surface(self) -> float:
        """u0 = 1 m/s のときの表面音圧 rho*c*u0 [Pa] (利得の基準)"""
        return self.rho * self.c

    @property
    def focal_gain_exact(self) -> float:
        """焦点利得の厳密解 G = k*h (球冠上の全点は焦点から等距離 R のため)"""
        return self.k * self.h

    @property
    def cap_area_exact(self) -> float:
        """球冠の厳密な表面積 2*pi*R*h [m^2]"""
        return 2.0 * math.pi * self.R * self.h

    def summary_lines(self):
        """設定の要約(コンソール表示・config.txt 保存の両方で使う)"""
        L = []
        L.append(f"run_name         : {self.run_name}")
        L.append(f"周波数 f         : {self.freq_mhz:.4f} MHz")
        L.append(f"コーティング厚 t : {self.coating_um:.1f} µm "
                 f"(モデル: {self.coating_mode})")
        L.append(f"水温 T           : {self.water_temp_c:.1f} ℃")
        L.append(f"音速 c (Marczak) : {self.c:.3f} m/s")
        L.append(f"波長 λ           : {self.lam*1e3:.5f} mm")
        L.append(f"波数 k           : {self.k:.3f} rad/m")
        L.append(f"素の形状 D / R   : {2*self.a0*1e3:.4f} / {self.R0*1e3:.4f} mm")
        L.append(f"実効 開口半径 a  : {self.a*1e3:.5f} mm")
        L.append(f"実効 曲率半径 R  : {self.R*1e3:.5f} mm  (= 幾何学的焦点距離)")
        L.append(f"実効 キャップ深 h: {self.h*1e3:.5f} mm")
        L.append(f"開口半角         : {math.degrees(self.theta_max):.3f}°")
        L.append(f"F 値 = R/D       : {self.f_number:.4f}")
        L.append(f"焦点利得(厳密)   : G = k*h = {self.focal_gain_exact:.5f}")
        L.append(f"球冠面積(厳密)   : 2πRh = {self.cap_area_exact*1e6:.6f} mm²")
        L.append(f"減衰             : "
                 f"{'ON (%.4g Np/m)' % self.alpha_np_per_m if self.use_atten else 'OFF (焦点10mmでは<0.1%,無視可)'}")
        L.append(f"時間規約         : exp(+iωt) → 空間項 exp(-i k r')")
        return L


def make_config(freq_mhz=None, coating_um=None, water_temp_c=None,
                coating_mode=None, tag=None) -> Config:
    """
    モジュール冒頭の設定を読み、CLI 引数で上書きし、派生量を計算して Config を返す。

    コーティングの扱い:
      素子の凹面に厚さ t を塗ると、水と接する放射面は焦点側へ t だけ寄る。
      曲率中心(= 幾何学的焦点)は動かないので、実効的な曲率半径は R-t になる。
        conformal     : 開口半角 θmax を保存 → a_eff = (R-t)*sin(θmax)
        keep_aperture : 開口半径 a を保存   → a_eff = a
      いずれも「放射面の位置」だけを動かす幾何モデルであり、
      整合層の音響的効果は含まない(モジュール冒頭の注意参照)。
    """
    f_mhz = FREQ_MHZ if freq_mhz is None else float(freq_mhz)
    t_um  = COATING_THICKNESS_UM if coating_um is None else float(coating_um)
    T_c   = WATER_TEMP_C if water_temp_c is None else float(water_temp_c)
    mode  = COATING_MODE if coating_mode is None else str(coating_mode)

    if f_mhz <= 0:
        raise ValueError(f"周波数が不正です: {f_mhz} MHz")
    if t_um < 0:
        raise ValueError(f"コーティング厚が負です: {t_um} µm")
    if mode not in ("conformal", "keep_aperture"):
        raise ValueError(f"COATING_MODE が不正です: {mode!r} "
                         "('conformal' か 'keep_aperture')")

    # --- 素の形状 [m] ---
    a0 = APERTURE_DIAMETER_MM * 1e-3 / 2.0
    R0 = CURV_RADIUS_MM * 1e-3
    if a0 >= R0:
        raise ValueError(f"開口半径 a={a0*1e3:.3f}mm が曲率半径 R={R0*1e3:.3f}mm "
                         "以上です。球冠になりません。")
    theta_max = math.asin(a0 / R0)      # 開口半角(素の形状で決まる)

    # --- コーティングを反映した実効形状 ---
    t = t_um * 1e-6
    R = R0 - t
    if R <= 0:
        raise ValueError(f"コーティング厚 {t_um} µm が曲率半径 "
                         f"{R0*1e3:.3f} mm 以上です。")
    if mode == "conformal":
        a = R * math.sin(theta_max)     # 開口半角を保存
    else:                                # keep_aperture
        a = a0
        if a >= R:
            raise ValueError(f"コーティング厚 {t_um} µm では実効曲率半径 "
                             f"{R*1e3:.3f} mm が開口半径 {a*1e3:.3f} mm 以下に "
                             "なります。'conformal' を使ってください。")
        theta_max = math.asin(a / R)    # a 固定なので半角は変わる
    h = R - math.sqrt(R * R - a * a)    # 実効キャップ深さ

    # --- 媒質 ---
    c = marczak_sound_speed(T_c)
    lam = c / (f_mhz * 1e6)
    k = 2.0 * math.pi * (f_mhz * 1e6) / c
    alpha = 0.0
    if USE_ATTEN:
        alpha = (ATTEN_DB_CM_MHZ2 * f_mhz * f_mhz) / 8.686 * 100.0  # dB/cm → Np/m

    # --- 計算範囲 [m] ---
    axial_z_min = AXIAL_Z_MIN_MM * 1e-3
    if axial_z_min <= h:
        raise ValueError(f"AXIAL_Z_MIN_MM={AXIAL_Z_MIN_MM} mm がキャップ深さ "
                         f"h={h*1e3:.4f} mm 以下です(お椀の内側)。大きくしてください。")
    z_min_3d = Z_MIN_3D_MM * 1e-3
    if z_min_3d <= h:
        raise ValueError(f"Z_MIN_3D_MM={Z_MIN_3D_MM} mm がキャップ深さ "
                         f"h={h*1e3:.4f} mm 以下です。大きくしてください。")

    # --- 計算範囲が焦点を含んでいるか ---
    # 焦点が範囲外だと、-6dB ビーム幅・焦点長・焦点体積といった主要な指標が
    # すべて無意味になる。設定ミス(特にコーティング厚の入れ間違い)を確実に
    # 捕まえるため、ここで止める。
    z_max_3d = Z_MAX_3D_MM * 1e-3
    axial_z_max = AXIAL_Z_MAX_MM * 1e-3
    if not (axial_z_min < R < axial_z_max):
        raise ValueError(
            f"幾何学的焦点 R={R*1e3:.4f} mm が 2D の軸上スキャン範囲 "
            f"{AXIAL_Z_MIN_MM}-{AXIAL_Z_MAX_MM} mm の外にあります。"
            f"\n  コーティング厚 ({t_um} µm) が大きすぎないか、"
            f"AXIAL_Z_MIN_MM / AXIAL_Z_MAX_MM を確認してください。")
    if not (z_min_3d < R < z_max_3d):
        raise ValueError(
            f"幾何学的焦点 R={R*1e3:.4f} mm が 3D の z 範囲 "
            f"{Z_MIN_3D_MM}-{Z_MAX_3D_MM} mm の外にあります。"
            f"\n  コーティング厚 ({t_um} µm) が大きすぎないか、"
            f"Z_MIN_3D_MM / Z_MAX_3D_MM を確認してください。")

    # --- 警告 (停止はしないが怪しい設定) ---
    warns = []
    if t > 0.02 * R0:
        warns.append(
            f"コーティング厚 {t_um:g} µm は素の曲率半径 {R0*1e3:.1f} mm の "
            f"{t/R0*100:.1f}% に相当します。整合層としては異例に厚い値です"
            "(通常は数十〜数百 µm)。入力単位[µm]を確認してください。")
    if t > 0:
        warns.append(
            "コーティングは『放射面が t だけ移動する』幾何モデルのみ。"
            "透過率・位相・整合・層共振といった音響的効果は含みません"
            "(効率や帯域の評価には使えません)。")

    # --- 出力先: 設定が一目でわかる名前 ---
    name = (f"f{f_mhz:.3f}MHz"
            f"_coat{t_um:g}um"
            f"_T{T_c:.1f}C")
    if mode != "conformal":
        name += f"_{mode}"
    if tag:
        name += f"_{tag}"
    out_dir = PROJECT_ROOT / RESULTS_DIRNAME / name

    return Config(
        freq_mhz=f_mhz, coating_um=t_um, coating_mode=mode, water_temp_c=T_c,
        a0=a0, R0=R0, a=a, R=R, h=h, theta_max=theta_max,
        f_number=R / (2.0 * a),
        c=c, rho=DENSITY_RHO, lam=lam, k=k,
        alpha_np_per_m=alpha, use_atten=USE_ATTEN,
        subdiv_per_lambda=SUBDIV_PER_LAMBDA,
        field_subdiv_per_lambda=FIELD_SUBDIV_PER_LAMBDA,
        rz_subdiv=RZ_SUBDIV, field_chunk=FIELD_CHUNK,
        axial_z_min=axial_z_min, axial_z_max=axial_z_max,
        lateral_half=LATERAL_HALF_MM * 1e-3,
        r_max_3d=R_MAX_3D_MM * 1e-3, z_min_3d=z_min_3d,
        z_max_3d=z_max_3d,
        run_name=name, out_dir=out_dir, warnings=tuple(warns),
    )
