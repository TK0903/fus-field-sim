# -*- coding: utf-8 -*-
"""
================================================================================
 1 次元 厚み振動 圧電振動子モデル (KLM 等価)  —  電圧 → 表面音圧
================================================================================

 目的:
   球面 PZT 素子の「印加電圧 → 焦点音圧」感度 [MPa/Vrms, MPa/Vpp] を、
   ハイドロフォンが使えない環境で理論予測する。

--------------------------------------------------------------------------------
 定式化について (重要)
--------------------------------------------------------------------------------
 KLM (Krimholtz-Leedom-Matthaei, 1970) は、厚み振動子の 1 次元圧電方程式を
 「伝送線路 + 中央タップの変成器」という等価回路に書き直したものである。
 Mason モデルも同じ方程式の別の回路表現で、3 者は数学的に厳密に等価である。

 本実装では、回路素子の規格化(変成比に掛かる sinc 因子の定義が文献ごとに
 異なる)に起因する取り違えを避けるため、【1 次元圧電構成方程式を直接解く】
 方式を採る。得られる結果は KLM と厳密に同一である。
 KLM の回路素子値 (C0, Z0, φ, ω0) は klm_elements() で参照用に出力する。

 支配方程式 (厚み方向 x, 電気変位 D は板厚方向に一様):
     T   = c33^D * S - h33 * D          (応力)
     E   = -h33 * S + D / eps33^S       (電界)
     ρ ü = ∂T/∂x                        (運動方程式) → 音速 v_D = sqrt(c33^D/ρ)

 変位を u(x) = a·cos(kx) + b·sin(kx) と置き、未知数 (a, b, D) を
 次の 3 式から解く:
     背面 x=0 :  T(0) = +Z_B · v(0)          (背面媒質へ -x 方向に放射)
     前面 x=t :  T(t) = -Z_F · v(t)          (前面媒質へ +x 方向に放射)
     電気     :  V = -h33·[u(t)-u(0)] + D·t/eps33^S

 ここで Z_B, Z_F は【比音響インピーダンス】ρc [Rayl]。

--------------------------------------------------------------------------------
 時間規約と振幅の規約 (プロジェクト全体で統一)
--------------------------------------------------------------------------------
  ・時間規約は exp(+i*omega*t)。Rayleigh 側 (fus/rayleigh.py) と同じ。
    この規約では損失は  eps -> eps*(1 - i*tanδ) ,  v_D -> v_D*(1 + i/(2*Qm))
    と入れると減衰になる。
  ・内部の複素振幅はすべて【peak 振幅】。Rayleigh 側が peak 統一 (音圧は PNP
    と等価) なので、ここで rms を混ぜると √2 の誤差が入る。
    Vrms / Vpp への換算は最後にまとめて行う:
        V_peak = √2 * V_rms ,   V_pp = 2√2 * V_rms

--------------------------------------------------------------------------------
 このモデルの限界 (絶対値は ±数十% の不確かさがある前提で使うこと)
--------------------------------------------------------------------------------
  ・1 次元平板モデルである。実素子は曲率のある厚い球殻 (t/R ≈ 0.13) なので、
    曲率由来の応力・径方向モード・スプリアス共振は再現しない。
  ・代表材料定数に基づく予測であり、個体差・電極/接着・保持条件は含まない。
  ・整合層なし、air backing を仮定。
  ・最終検証は、素子のインピーダンス実測 Z(f) との照合で行うこと。
    共振/反共振周波数と |Z| の谷の深さを合わせれば、実効的な kt・Qm・C0 が
    逆算でき、そこでモデルを較正するのが正しい進め方である。
================================================================================
"""

import math
from dataclasses import dataclass

import numpy as np

import config as cfgmod


# ==============================================================================
#  §1. 素子パラメータの導出
# ==============================================================================
def fr_fa_ratio(k):
    """
    IEEE の結合係数定義
        k^2 = (pi/2)·(fr/fa)·tan((pi/2)·(1 - fr/fa))
    を fr/fa について解く。右辺は fr/fa について単調減少なので二分法で一意。

    厚み振動子では、素直な半波長関係 fa = v_D/(2t) が成り立つのは反共振側。
    直列共振はこの比のぶんだけ低い側にずれる (kt=0.49 で約 0.892 倍)。
    """
    lo, hi = 1e-6, 1.0 - 1e-12
    for _ in range(200):
        x = 0.5 * (lo + hi)
        v = (math.pi / 2) * x * math.tan((math.pi / 2) * (1.0 - x))
        if v > k * k:
            lo = x
        else:
            hi = x
    return 0.5 * (lo + hi)



@dataclass(frozen=True)
class PZT:
    """KLM 計算に必要な素子パラメータ一式 (SI)"""
    f_design: float     # 設計周波数 [Hz]
    t: float            # 厚み [m]
    A: float            # 電極面積 (球冠の曲面積) [m^2]
    rho: float          # 密度 [kg/m^3]
    v_D: float          # 厚み方向 音速 (stiffened) [m/s]
    c33D: float         # 厚み方向 弾性定数 c33^D [Pa]
    Zp: float           # PZT の比音響インピーダンス ρ*v_D [Rayl]
    kt: float           # 厚み方向 結合係数
    eps_S: float        # クランプ誘電率 eps33^S [F/m]
    eps_S_rel: float    # eps33^S / eps0
    eps_S_mode: str     # 換算に使った式
    h33: float          # 圧電定数 h33 [V/m]
    C0: float           # クランプ容量 [F]
    Qm: float
    tan_delta: float
    Z_front: float      # 前面 (水) の比音響インピーダンス [Rayl]
    Z_back: float       # 背面 (空気) の比音響インピーダンス [Rayl]
    f_thick: float      # 厚み共振 (空気中) v_D/(2t) [Hz]

    def summary_lines(self):
        L = []
        L.append(f"設計周波数 f_d   : {self.f_design/1e6:.4f} MHz")
        L.append(f"厚み t           : {self.t*1e3:.5f} mm  (= Nt/f_d)")
        L.append(f"電極面積 A       : {self.A*1e6:.4f} mm²  (球冠の曲面積 2πRh)")
        L.append(f"Nt の定義基準    : {cfgmod.PZT_NT_BASIS}  "
                 f"({'Nt = fr·t 直列共振基準' if cfgmod.PZT_NT_BASIS == 'fr' else 'Nt = fa·t 反共振基準'})")
        L.append(f"厚み方向音速 v_D : {self.v_D:.1f} m/s  "
                 f"(= 2·Nt/(fr/fa) = 2·{cfgmod.PZT_NT:.0f}/{fr_fa_ratio(self.kt):.5f})"
                 if cfgmod.PZT_NT_BASIS == "fr" else
                 f"厚み方向音速 v_D : {self.v_D:.1f} m/s  (= 2·Nt)")
        L.append(f"c33^D            : {self.c33D/1e10:.4f} ×10^10 Pa")
        L.append(f"PZT の ρc        : {self.Zp/1e6:.4f} MRayl")
        L.append(f"水の ρc          : {self.Z_front/1e6:.4f} MRayl  "
                 f"(比 Zp/Zw = {self.Zp/self.Z_front:.2f})")
        L.append(f"背面 (空気) ρc   : {self.Z_back:.1f} Rayl")
        L.append(f"kt / k33         : {self.kt:.3f} / {cfgmod.PZT_K33:.3f}")
        L.append(f"eps33^S/eps0     : {self.eps_S_rel:.1f}  "
                 f"(換算式: {self.eps_S_mode})")
        L.append(f"h33              : {self.h33/1e9:.4f} ×10^9 V/m")
        L.append(f"クランプ容量 C0  : {self.C0*1e9:.4f} nF")
        L.append(f"空気中の反共振   : {self.f_thick/1e6:.4f} MHz  (= v_D/(2t))")
        L.append(f"空気中の直列共振 : "
                 f"{self.f_thick*fr_fa_ratio(self.kt)/1e6:.4f} MHz  "
                 f"(= 反共振 × {fr_fa_ratio(self.kt):.5f})")
        L.append(f"Qm / tanδ        : {self.Qm:.0f} / {self.tan_delta:.4f}")
        return L


def make_pzt(cfg) -> PZT:
    """
    config の定数と音場側の Config から、KLM 計算用の素子パラメータを作る。

    面積について:
      電極は球冠の【曲面】に付いているので、投影面積 π(D/2)^2 ではなく
      曲面積 2πRh を使う。ただし PZT 素子そのものの形状で計算するため、
      コーティングを反映した実効値 (cfg.R, cfg.h) ではなく素の形状
      (cfg.R0 とそこから求めた h0) を使う。
      なお送信感度 p/V は原理的に面積に依らない (駆動電界 E=V/t が面積に
      無関係なため)。面積が効くのは電気インピーダンス |Z| の絶対値だけで、
      これは実測 Z(f) と突き合わせるときに意味を持つ。
    """
    # --- 設計周波数と厚み ---
    f_d = (cfg.freq_hz if cfgmod.PZT_DESIGN_FREQ_MHZ is None
           else cfgmod.PZT_DESIGN_FREQ_MHZ * 1e6)

    # 厚み方向音速 v_D を Nt の定義基準に応じて決める。
    #   厚み振動子で「t = λ/2」という素直な関係 fa = v_D/(2t) が成り立つのは
    #   【反共振 fa】である。したがって:
    #     Nt = fa·t  なら  v_D = 2·Nt
    #     Nt = fr·t  なら  fr = (fr/fa)·v_D/(2t) なので  v_D = 2·Nt/(fr/fa)
    #   fr/fa は kt から IEEE の定義式で一意に決まる (下の fr_fa_ratio)。
    #   富士セラミックスは fr 基準 (config.PZT_NT_BASIS の根拠コメント参照)。
    if cfgmod.PZT_NT_BASIS == "fa":
        v_D = 2.0 * cfgmod.PZT_NT
    elif cfgmod.PZT_NT_BASIS == "fr":
        v_D = 2.0 * cfgmod.PZT_NT / fr_fa_ratio(cfgmod.PZT_KT)
    else:
        raise ValueError(f"PZT_NT_BASIS が不正です: {cfgmod.PZT_NT_BASIS!r} "
                         "('fr' か 'fa')")

    if cfgmod.PZT_THICKNESS_MM is not None:
        t = cfgmod.PZT_THICKNESS_MM * 1e-3         # 実測値を最優先
    else:
        # Nt = f·t という定義そのものから厚みを出す (基準によらず同じ式)。
        # 基準が "fr" なら f_d は空気中の直列共振、"fa" なら反共振になる。
        t = cfgmod.PZT_NT / f_d
        # "series_at_f" なら、水負荷の直列共振が f_d に来るよう t を縮める。
        # fr は概ね 1/t に比例するので t ← t*(fr/f_d) の反復ですぐ収束する。
        if cfgmod.PZT_THICKNESS_MODE == "series_at_f":
            for _ in range(30):
                probe = _build_pzt(cfg, f_d, t, v_D)
                fr = find_series_resonance(probe, 0.7 * f_d, 1.3 * f_d)
                if abs(fr - f_d) / f_d < 1e-7:
                    break
                t *= fr / f_d
        elif cfgmod.PZT_THICKNESS_MODE != "Nt_over_f":
            raise ValueError("PZT_THICKNESS_MODE が不正です: "
                             f"{cfgmod.PZT_THICKNESS_MODE!r}")
    return _build_pzt(cfg, f_d, t, v_D)


def find_series_resonance(pzt: PZT, f_lo, f_hi, n=4001, refine=3):
    """水負荷時の直列共振 (電気コンダクタンス Re(Y) が最大の点) を探す。"""
    lo, hi = f_lo, f_hi
    fr = 0.5 * (lo + hi)
    for _ in range(refine):
        fs = np.linspace(lo, hi, n)
        g = np.real(solve(fs, pzt)["Y"])
        i = int(np.argmax(g))
        fr = fs[i]
        d = fs[1] - fs[0]
        lo, hi = fr - 3 * d, fr + 3 * d
    return fr


def _build_pzt(cfg, f_d, t, v_D) -> PZT:
    """PZT データクラスの実体を組み立てる (make_pzt の内部処理)"""

    # --- 素の球冠の曲面積 (コーティング無しの PZT 本体) ---
    h0 = cfg.R0 - math.sqrt(cfg.R0 ** 2 - cfg.a0 ** 2)
    A = 2.0 * math.pi * cfg.R0 * h0

    # --- 誘電率 ---
    mode = cfgmod.PZT_EPS_S_MODE
    if mode == "k33":
        eps_S_rel = cfgmod.PZT_EPS33T_REL * (1.0 - cfgmod.PZT_K33 ** 2)
    elif mode == "kt":
        eps_S_rel = cfgmod.PZT_EPS33T_REL * (1.0 - cfgmod.PZT_KT ** 2)
    elif mode == "direct":
        eps_S_rel = cfgmod.PZT_EPS33S_REL
    else:
        raise ValueError(f"PZT_EPS_S_MODE が不正です: {mode!r}")
    eps_S = eps_S_rel * cfgmod.EPS0

    # --- 派生量 ---
    rho = cfgmod.PZT_DENSITY
    c33D = rho * v_D ** 2                          # 弾性定数 (stiffened)
    Zp = rho * v_D                                 # 比音響インピーダンス
    # h33 は kt の定義 kt^2 = h33^2 * eps^S / c^D から。損失なしの実定数とする。
    h33 = cfgmod.PZT_KT * math.sqrt(c33D / eps_S)
    C0 = eps_S * A / t

    return PZT(f_design=f_d, t=t, A=A, rho=rho, v_D=v_D, c33D=c33D, Zp=Zp,
               kt=cfgmod.PZT_KT, eps_S=eps_S, eps_S_rel=eps_S_rel,
               eps_S_mode=mode, h33=h33, C0=C0,
               Qm=cfgmod.PZT_QM, tan_delta=cfgmod.PZT_TAN_DELTA,
               Z_front=cfg.rho * cfg.c,            # 水 (config の値で統一)
               Z_back=cfgmod.BACKING_RHO_C,
               f_thick=v_D / (2.0 * t))


# ==============================================================================
#  §2. 1 次元圧電方程式の直接解 (KLM と厳密に等価)
# ==============================================================================
def solve(f, pzt: PZT, V_peak=1.0, lossy=True, A=None,
          Z_front=None, Z_back=None, kt=None):
    """
    周波数配列 f [Hz] に対して 1 次元圧電振動子を解く。

    駆動は電圧 V_peak (複素 peak 振幅 [V])。返り値もすべて peak 振幅。

    Parameters (テスト用に一部を上書きできる)
    ----------
    A       : 面積を上書き (面積非依存性の検証用)
    Z_front : 前面負荷を上書き (空気中共振の検証用)
    kt      : 結合係数を上書き (kt→0 の検証用)

    Returns: dict of ndarray
      Z      電気インピーダンス [Ω]
      Y      アドミタンス [S] (G=Re(Y) がコンダクタンス)
      I      電流 [A]
      v_f    前面の粒子速度 [m/s]
      p_f    前面の音圧 [Pa]  ( = Z_front * v_f )
      v_b    背面の粒子速度 [m/s]
      P_ac   前面へ放射される音響パワー [W]
      P_el   電気入力の有効電力 [W]
    """
    f = np.atleast_1d(np.asarray(f, dtype=np.float64))
    w = 2.0 * np.pi * f

    A = pzt.A if A is None else A
    ZF = pzt.Z_front if Z_front is None else Z_front
    ZB = pzt.Z_back if Z_back is None else Z_back
    ktv = pzt.kt if kt is None else kt

    # --- 損失の組み込み (exp(+iωt) 規約で減衰になる向き) ---
    if lossy:
        v_D = pzt.v_D * (1.0 + 1j / (2.0 * pzt.Qm))      # 機械損失
        eps_S = pzt.eps_S * (1.0 - 1j * pzt.tan_delta)   # 誘電損失
    else:
        v_D = complex(pzt.v_D)
        eps_S = complex(pzt.eps_S)
    c33D = pzt.rho * v_D ** 2
    Zp = pzt.rho * v_D                                   # 複素 比音響インピーダンス
    # h33 は実定数 (kt を実数に保つ)。ただし上書き kt に追従させる。
    h33 = ktv * math.sqrt(pzt.c33D / pzt.eps_S)

    k = w / v_D                    # 複素波数
    th = k * pzt.t                 # 位相厚み
    cos, sin = np.cos(th), np.sin(th)

    # --- 3x3 線形系 [a, b, D] ---
    # 行1: 背面 x=0 :  -i*w*ZB*a + w*Zp*b - h33*D = 0
    # 行2: 前面 x=t :  a(-w*Zp*sin + i*w*ZF*cos) + b(w*Zp*cos + i*w*ZF*sin) - h33*D = 0
    # 行3: 電気     :  -h33*(cos-1)*a - h33*sin*b + (t/eps_S)*D = V
    n = f.size
    M = np.empty((n, 3, 3), dtype=np.complex128)
    # numpy>=2.0 のバッチ solve は右辺を (n,3,1) で渡す必要がある
    rhs = np.zeros((n, 3, 1), dtype=np.complex128)

    M[:, 0, 0] = -1j * w * ZB
    M[:, 0, 1] = w * Zp
    M[:, 0, 2] = -h33

    M[:, 1, 0] = w * (-Zp * sin + 1j * ZF * cos)
    M[:, 1, 1] = w * (Zp * cos + 1j * ZF * sin)
    M[:, 1, 2] = -h33

    M[:, 2, 0] = -h33 * (cos - 1.0)
    M[:, 2, 1] = -h33 * sin
    M[:, 2, 2] = pzt.t / eps_S
    rhs[:, 2, 0] = V_peak

    sol = np.linalg.solve(M, rhs)[:, :, 0]
    a, b, D = sol[:, 0], sol[:, 1], sol[:, 2]

    # --- 物理量 ---
    v_f = 1j * w * (a * cos + b * sin)      # 前面の粒子速度 (peak)
    v_b = 1j * w * a                        # 背面の粒子速度 (peak)
    p_f = ZF * v_f                          # 前面の音圧 (peak)
    I = 1j * w * A * D                      # 電流 (peak)
    Z = V_peak / I
    Y = 1.0 / Z

    # 電力 (peak 振幅なので 1/2 が付く)
    P_ac = 0.5 * np.abs(v_f) ** 2 * ZF * A
    P_el = 0.5 * np.real(V_peak * np.conj(I))

    return dict(f=f, Z=Z, Y=Y, I=I, v_f=v_f, p_f=p_f, v_b=v_b,
                P_ac=P_ac, P_el=P_el)


# ==============================================================================
#  §3. KLM 等価回路の素子値 (参照用)
# ==============================================================================
def klm_elements(f, pzt: PZT):
    """
    KLM 等価回路の素子値を返す (文献との照合・実測 Z(f) のフィッティング用)。
    本体の計算には使わない (solve() が同じ物理を厳密に解いている)。

      Z0  : 音響伝送線路の特性インピーダンス ρ*v_D*A [kg/s]
      C0  : クランプ容量 eps33^S*A/t [F]
      w0  : 厚み共振角周波数 π*v_D/t (t = λ/2 になる点)
      phi : 電気-機械 変成比  kt*sqrt(π/(w*C0*Z0)) * sinc(w/(2*w0))
            ※ sinc(x) = sin(x)/x
      X'  : 直列リアクタンス (kt^2/(w*C0)) * sinc(w/w0)
    """
    f = np.atleast_1d(np.asarray(f, dtype=np.float64))
    w = 2.0 * np.pi * f
    Z0 = pzt.rho * pzt.v_D * pzt.A
    w0 = np.pi * pzt.v_D / pzt.t
    x = w / (2.0 * w0)
    sinc_half = np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / np.where(x == 0, 1, x))
    y = w / w0
    sinc_full = np.where(np.abs(y) < 1e-12, 1.0, np.sin(y) / np.where(y == 0, 1, y))
    phi = pzt.kt * np.sqrt(np.pi / (w * pzt.C0 * Z0)) * sinc_half
    Xp = (pzt.kt ** 2 / (w * pzt.C0)) * sinc_full
    return dict(Z0=Z0, C0=pzt.C0, w0=w0, phi=phi, X_series=Xp)


# ==============================================================================
#  §4. 自己検証 (解析的な極限との照合)
# ==============================================================================
def self_test(pzt: PZT):
    """
    実装の妥当性を解析的な極限で検証する。
    定式化を記憶に頼らず、物理的に確認できる性質だけで担保するのが目的。

    Returns: [(検査名, 合格か, 詳細文字列), ...]
    """
    out = []

    # (1) f→0 の容量は「面が自由」なので C0/(1-kt^2) に漸近する。
    #     (クランプ容量 C0 そのものではない点に注意。低周波では板が自由に
    #      伸縮できるため、実効誘電率が eps^S/(1-kt^2) に見える。)
    f_lo = np.array([1.0e3])
    r = solve(f_lo, pzt, lossy=False)
    C_free = pzt.C0 / (1.0 - pzt.kt ** 2)
    Z_expect = 1.0 / (1j * 2 * np.pi * f_lo * C_free)
    err = abs(r["Z"][0] - Z_expect[0]) / abs(Z_expect[0])
    out.append(("低周波容量 → C0/(1-kt²)", err < 1e-3,
                f"相対誤差 {err:.3e}  "
                f"(数値 {1/(2*np.pi*f_lo[0]*abs(r['Z'][0]))*1e9:.4f} nF, "
                f"期待 {C_free*1e9:.4f} nF)"))

    # (2) 両面フリー(空気)の【反共振】が v_D/(2t) = Nt/t に一致する。
    #     厚み振動子では Nt は反共振で定義される (IEEE 規格)。
    fs = np.linspace(0.7 * pzt.f_thick, 1.3 * pzt.f_thick, 40001)
    r = solve(fs, pzt, lossy=True, Z_front=cfgmod.BACKING_RHO_C)
    fa = fs[np.argmax(np.abs(r["Z"]))]
    rel = abs(fa - pzt.f_thick) / pzt.f_thick
    out.append(("空気中の反共振 = v_D/(2t)", rel < 5e-3,
                f"数値 {fa/1e6:.5f} MHz / 理論 {pzt.f_thick/1e6:.5f} MHz "
                f"(誤差 {rel*100:.3f} %)"))

    # (3) 空気中の直列共振 fr と反共振 fa が、IEEE の kt 定義式を満たす:
    #        kt^2 = (π/2)(fr/fa) * tan((π/2)(fa-fr)/fa)
    #     これは「実装が正しい圧電結合を再現しているか」の最も直接的な検証。
    fr = fs[np.argmax(np.real(r["Y"]))]
    x = fr / fa
    kt_ieee = math.sqrt((np.pi / 2) * x * math.tan((np.pi / 2) * (1.0 - x)))
    err = abs(kt_ieee - pzt.kt) / pzt.kt
    out.append(("IEEE の kt 定義式との整合", err < 5e-3,
                f"fr={fr/1e6:.5f} / fa={fa/1e6:.5f} MHz → kt={kt_ieee:.5f} "
                f"(設定 {pzt.kt:.5f}, 誤差 {err*100:.3f} %)"))

    # (3) 送信感度 p/V は面積に依らない (駆動電界 E=V/t が面積無関係のため)
    fs = np.array([pzt.f_design])
    p1 = solve(fs, pzt, A=pzt.A)["p_f"][0]
    p2 = solve(fs, pzt, A=pzt.A * 2.7)["p_f"][0]
    err = abs(p2 - p1) / abs(p1)
    out.append(("感度 p/V の面積非依存性", err < 1e-9,
                f"A と 2.7A で相対差 {err:.3e}"))

    # (4) 電気インピーダンスは面積に反比例する
    z1 = solve(fs, pzt, A=pzt.A)["Z"][0]
    z2 = solve(fs, pzt, A=pzt.A * 2.7)["Z"][0]
    err = abs(z2 * 2.7 - z1) / abs(z1)
    out.append(("電気インピーダンス ∝ 1/A", err < 1e-9,
                f"Z(2.7A)*2.7 と Z(A) の相対差 {err:.3e}"))

    # (5) 結合を切る (kt→0) と音響出力が消える
    p0 = solve(fs, pzt, kt=1e-9)["p_f"][0]
    out.append(("kt→0 で音響出力が消える", abs(p0) < 1e-6 * abs(p1),
                f"|p(kt≈0)|/|p(kt)| = {abs(p0)/abs(p1):.3e}"))

    # (6) エネルギー保存: 放射音響パワー <= 電気入力有効電力
    fs = np.linspace(1.2e6, 1.8e6, 601)
    r = solve(fs, pzt, lossy=True)
    ratio = r["P_ac"] / r["P_el"]
    out.append(("エネルギー保存 P_ac ≤ P_el", np.all(ratio <= 1.0 + 1e-9),
                f"最大 P_ac/P_el = {ratio.max():.4f} "
                f"(= 電気音響変換効率の上限)"))

    return out
