# -*- coding: utf-8 -*-
"""
KLM 計算 — 印加電圧 → 焦点音圧 の感度予測

出力 (すべて cfg.out_dir 以下):
  klm_response.png    |Z|・位相・コンダクタンス・表面音圧 の周波数特性
  klm_impedance.png   インピーダンス実測 Z(f) と照合するための図
  results_klm.csv     数値まとめ
  klm_sweep.csv       スイープ生データ (実測との突き合わせ用)

--------------------------------------------------------------------------------
 音場計算 (2D/3D) との連携
--------------------------------------------------------------------------------
  sim2d  : 目標焦点音圧 → 必要な表面法線振動速度 u0    (音響側の要求)
  sim_klm: その u0 を出すのに必要な電圧 Vrms / Vpp     (電気側の要求)
  この 2 つが繋がって初めて「アンプの電圧余裕が足りるか」に答えられる。

  集束利得は Rayleigh 側の厳密解 G = k*h を既定で使う (仕様の簡易式
  G = π a²/(λ d) も比較のため併記する。簡易式は近軸・平面開口近似のため
  1.5 MHz では約 12 % 低く出る)。

 【振幅の規約】音圧はすべて peak (PNP 等価)。電圧は
      V_peak = √2 · V_rms ,  V_pp = 2√2 · V_rms
 で換算する。混同すると √2 の誤差が入るので注意。
--------------------------------------------------------------------------------
"""

import csv
import math
import time

import numpy as np
import matplotlib.pyplot as plt

from . import klm

import config as cfgmod

SQRT2 = math.sqrt(2.0)


def focal_gain_exact(f, cfg):
    """厳密な集束利得 G = k_water * h (球冠上の全点が焦点から等距離 R のため)"""
    return 2.0 * np.pi * f / cfg.c * cfg.h


def focal_gain_simple(f, cfg):
    """仕様にあった簡易式 G = π a² / (λ d) (近軸・平面開口近似)"""
    lam = cfg.c / f
    return np.pi * cfg.a ** 2 / (lam * cfg.R)


def _band(fs, y, frac):
    """y(f) のピークに対し frac 倍を横切る帯域幅を返す (見つからなければ nan)"""
    pk = y.max()
    idx = np.where(y >= pk * frac)[0]
    if idx.size < 2:
        return np.nan, np.nan, np.nan
    return fs[idx[-1]] - fs[idx[0]], fs[idx[0]], fs[idx[-1]]


def run(cfg, device=None):
    """KLM 計算を実行し、結果 dict を返す。図と CSV は cfg.out_dir に保存する。"""
    t0 = time.time()

    # ---- 素子パラメータ -----------------------------------------------------
    pzt = klm.make_pzt(cfg)
    print("-" * 72)
    print(" [KLM] 素子パラメータ")
    for L in pzt.summary_lines():
        print(f"   {L}")
    print(f"   厚みの決め方     : {cfgmod.PZT_THICKNESS_MODE}"
          + ("  (PZT_THICKNESS_MM で実測値を指定中)"
             if cfgmod.PZT_THICKNESS_MM is not None else ""))
    print("-" * 72)

    # ---- 自己検証 -----------------------------------------------------------
    # 定式化を記憶に頼らず、解析的な極限だけで担保する。
    print("\n[KLM 検証] 解析的な極限との照合")
    all_ok = True
    for name, ok, det in klm.self_test(pzt):
        all_ok &= ok
        print(f"  [{'OK' if ok else 'NG'}] {name}")
        print(f"       {det}")
    if not all_ok:
        raise RuntimeError("KLM の自己検証に失敗しました。実装/設定を確認してください。")
    print("  → 全項目合格")

    # ---- 周波数スイープ -----------------------------------------------------
    fs = np.linspace(cfgmod.KLM_F_MIN_MHZ * 1e6, cfgmod.KLM_F_MAX_MHZ * 1e6,
                     cfgmod.KLM_N_FREQ)
    r = klm.solve(fs, pzt, V_peak=1.0)          # 1 V_peak 駆動
    Zabs = np.abs(r["Z"])
    Zph = np.degrees(np.angle(r["Z"]))
    Gc = np.real(r["Y"])                        # コンダクタンス
    Bc = np.imag(r["Y"])                        # サセプタンス
    p_surf = np.abs(r["p_f"])                   # 表面音圧 [Pa/V_peak]
    Gfoc = focal_gain_exact(fs, cfg)
    p_foc = Gfoc * p_surf                       # 焦点音圧 [Pa/V_peak]
    eff = r["P_ac"] / r["P_el"]                 # 電気音響変換効率

    # ---- 特徴点 -------------------------------------------------------------
    i_fr = int(np.argmax(Gc))                   # 直列共振 (コンダクタンス最大)
    i_fa = int(np.argmax(Zabs))                 # 反共振 (|Z| 最大)
    i_zmin = int(np.argmin(Zabs))
    i_pk = int(np.argmax(p_surf))               # 送信ピーク
    i_fd = int(np.argmin(np.abs(fs - pzt.f_design)))   # 設計周波数
    f_r, f_a = fs[i_fr], fs[i_fa]

    bw3, b3l, b3h = _band(fs, p_surf, 1 / SQRT2)    # -3dB (振幅 0.707)
    bw6, _, _ = _band(fs, p_surf, 0.5)              # -6dB (振幅 0.5)
    bwg3, _, _ = _band(fs, Gc, 0.5)                 # コンダクタンスの半値幅

    print("\n[KLM] 水負荷での特性")
    print(f"  直列共振 f_r (コンダクタンス最大) = {f_r/1e6:.5f} MHz  "
          f"|Z| = {Zabs[i_fr]:8.2f} Ω")
    print(f"  反共振   f_a (|Z| 最大)          = {f_a/1e6:.5f} MHz  "
          f"|Z| = {Zabs[i_fa]:8.1f} Ω")
    print(f"  |Z| 最小                         = {fs[i_zmin]/1e6:.5f} MHz  "
          f"|Z| = {Zabs[i_zmin]:8.2f} Ω")
    print(f"  送信 (表面音圧) ピーク           = {fs[i_pk]/1e6:.5f} MHz")
    print(f"  表面音圧 -3dB 帯域               = {bw3/1e3:.1f} kHz "
          f"({bw3/fs[i_pk]*100:.2f} %)   Q ≈ {fs[i_pk]/bw3:.1f}")
    print(f"  表面音圧 -6dB 帯域               = {bw6/1e3:.1f} kHz "
          f"({bw6/fs[i_pk]*100:.2f} %)")
    print(f"  電気音響変換効率 @f_r            = {eff[i_fr]*100:.2f} %")

    # ---- 集束利得 -----------------------------------------------------------
    G_ex_fr = focal_gain_exact(f_r, cfg)
    G_sp_fr = focal_gain_simple(f_r, cfg)
    print(f"\n[KLM] 集束利得 @f_r={f_r/1e6:.4f} MHz")
    print(f"  厳密解 G = k*h              = {G_ex_fr:.4f}   ← 使用")
    print(f"  簡易式 G = πa²/(λd)         = {G_sp_fr:.4f}  "
          f"(厳密比 {G_sp_fr/G_ex_fr*100:.1f} %)")

    # ---- 感度 ---------------------------------------------------------------
    # p_foc は 1 V_peak あたりの焦点 peak 音圧 [Pa]
    def sensitivity(i):
        """
        i 番目の周波数での感度をまとめる。

        規約: p_foc は V_peak=1 のときの値。
              V_peak = √2·V_rms  → 1 Vrms あたりは √2 倍
              V_pp   = 2√2·V_rms → 1 Vpp  あたりは (1/2√2)·√2 = 1/2 倍
        """
        pf_per_vpeak = p_foc[i]                       # Pa / V_peak
        S_rms_MPa = pf_per_vpeak * SQRT2 / 1e6        # MPa / Vrms
        S_pp_MPa = pf_per_vpeak / 2.0 / 1e6           # MPa / Vpp
        return dict(i=i, f=fs[i], p_surf=p_surf[i], G=Gfoc[i],
                    S_rms=S_rms_MPa, S_pp=S_pp_MPa,
                    Zabs=Zabs[i], Zph=Zph[i], eff=eff[i],
                    u_surf=abs(r["v_f"][i]) * SQRT2)  # m/s per Vrms (peak 振幅)

    s_fr = sensitivity(i_fr)
    s_fd = sensitivity(i_fd)

    print("\n" + "=" * 72)
    print(" [KLM] 感度 (焦点音圧は peak = PNP 等価)")
    print("=" * 72)
    hdr = f"  {'':22} {'f_r で駆動':>18} {'設計 %.3f MHz で駆動' % (pzt.f_design/1e6):>22}"
    print(hdr)
    print(f"  {'周波数 [MHz]':22} {s_fr['f']/1e6:>18.5f} {s_fd['f']/1e6:>22.5f}")
    print(f"  {'|Z| [Ω]':22} {s_fr['Zabs']:>18.2f} {s_fd['Zabs']:>22.2f}")
    print(f"  {'位相 [deg]':22} {s_fr['Zph']:>18.1f} {s_fd['Zph']:>22.1f}")
    print(f"  {'表面音圧 [Pa/V_peak]':22} {s_fr['p_surf']:>18.1f} {s_fd['p_surf']:>22.1f}")
    print(f"  {'集束利得 G':22} {s_fr['G']:>18.4f} {s_fd['G']:>22.4f}")
    print(f"  {'感度 [MPa/Vrms]':22} {s_fr['S_rms']:>18.5f} {s_fd['S_rms']:>22.5f}")
    print(f"  {'感度 [MPa/Vpp]':22} {s_fr['S_pp']:>18.5f} {s_fd['S_pp']:>22.5f}")
    print(f"  {'変換効率 [%]':22} {s_fr['eff']*100:>18.2f} {s_fd['eff']*100:>22.2f}")

    # ---- 駆動電圧 → 焦点音圧 ------------------------------------------------
    print("\n [KLM] 駆動電圧 → 焦点音圧 (線形外挿)")
    print(f"    {'Vpp':>8} {'Vrms':>9} {'f_r [MPa]':>12} "
          f"{'設計f [MPa]':>14}")
    volt_rows = []
    for vpp in cfgmod.KLM_DRIVE_VPP_LIST:
        vrms = vpp / (2 * SQRT2)
        p1 = s_fr["S_pp"] * vpp
        p2 = s_fd["S_pp"] * vpp
        volt_rows.append((vpp, vrms, p1, p2))
        print(f"    {vpp:>8.1f} {vrms:>9.2f} {p1:>12.4f} {p2:>14.4f}")

    # ---- 目標音圧 → 必要電圧 (sim2d との連携) -------------------------------
    print("\n [KLM] 目標焦点音圧 → 必要な駆動電圧")
    print(f"    {'目標 [MPa]':>11} {'u0 [m/s]':>10} "
          f"{'f_r [Vpp]':>12} {'設計f [Vpp]':>14}")
    need_rows = []
    for tgt in cfgmod.TARGET_PRESSURES_MPA:
        # sim2d と同じ式: u0 = p_target/(rho*c*G)  (G は厳密解)
        u0 = tgt * 1e6 / (cfg.rho * cfg.c * focal_gain_exact(f_r, cfg))
        v1 = tgt / s_fr["S_pp"]
        v2 = tgt / s_fd["S_pp"]
        need_rows.append((tgt, u0, v1, v2))
        print(f"    {tgt:>11.2f} {u0:>10.5f} {v1:>12.2f} {v2:>14.2f}")
    print("    ※ u0 は f_r における値。音響側の要求は sim2d と同じ。")

    # ---- 設計周波数が共振から外れている場合の警告 ---------------------------
    off = abs(pzt.f_design - f_r) / f_r
    if off > 0.02:
        print("\n" + "!" * 72)
        print(f" 注意: 設計周波数 {pzt.f_design/1e6:.4f} MHz は"
              f"直列共振 {f_r/1e6:.4f} MHz から {off*100:.1f} % 離れています。")
        print(f"   表面音圧は共振点の {p_surf[i_fd]/p_surf[i_fr]*100:.1f} %、"
              f"|Z| は {Zabs[i_fr]:.0f} Ω → {Zabs[i_fd]:.0f} Ω に跳ね上がります。")
        print( "   厚み振動子の周波数定数 Nt は IEEE 規格では【反共振】で定義される")
        print( "   ため、t = Nt/f では f が反共振になり、送信が最も苦しい点になります。")
        print( "   実物の厚みが分かっているなら config の PZT_THICKNESS_MM に入れるか、")
        print( "   PZT_THICKNESS_MODE='series_at_f' で共振を設計周波数に合わせてください。")
        print("!" * 72)

    print("\n ※ 代表材料定数に基づく理論予測であり、絶対値は ±数十% の不確かさがある。")
    print("    最終検証は素子のインピーダンス実測 Z(f) との照合で行うこと")
    print("    (klm_sweep.csv と klm_impedance.png を突き合わせる)。")

    # =====================================================================
    #  CSV
    # =====================================================================
    with open(cfg.out_dir / "results_klm.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["# run", cfg.run_name])
        w.writerow(["# model", "1D thickness-mode piezo (KLM-equivalent)"])
        w.writerow(["# note", "pressures are PEAK amplitude (PNP-equivalent)"])
        w.writerow([])
        w.writerow(["section", "quantity", "value", "unit", "note"])
        for key, val, unit in [
                ("pzt_thickness", pzt.t * 1e3, "mm"),
                ("pzt_area", pzt.A * 1e6, "mm^2"),
                ("pzt_v_D", pzt.v_D, "m/s"),
                ("pzt_Zp", pzt.Zp / 1e6, "MRayl"),
                ("water_Z", pzt.Z_front / 1e6, "MRayl"),
                ("eps33S_rel", pzt.eps_S_rel, "-"),
                ("C0", pzt.C0 * 1e9, "nF"),
                ("h33", pzt.h33 / 1e9, "1e9 V/m")]:
            w.writerow(["pzt", key, f"{val:.6f}", unit, pzt.eps_S_mode
                        if key == "eps33S_rel" else ""])
        w.writerow([])
        for key, val, unit, note in [
                ("f_series_resonance", f_r / 1e6, "MHz", "conductance max"),
                ("f_antiresonance", f_a / 1e6, "MHz", "|Z| max"),
                ("Z_at_fr", Zabs[i_fr], "ohm", ""),
                ("Z_at_fa", Zabs[i_fa], "ohm", ""),
                ("bandwidth_-3dB", bw3 / 1e3, "kHz",
                 f"{bw3/fs[i_pk]*100:.3f}% , Q={fs[i_pk]/bw3:.2f}"),
                ("bandwidth_-6dB", bw6 / 1e3, "kHz", ""),
                ("efficiency_at_fr", eff[i_fr] * 100, "%", "")]:
            w.writerow(["response", key, f"{val:.6f}", unit, note])
        w.writerow([])
        for tag, s in [("at_fr", s_fr), ("at_design_f", s_fd)]:
            w.writerow(["sensitivity", f"{tag}/frequency", f"{s['f']/1e6:.6f}",
                        "MHz", ""])
            w.writerow(["sensitivity", f"{tag}/p_surface_per_Vpeak",
                        f"{s['p_surf']:.6f}", "Pa", ""])
            w.writerow(["sensitivity", f"{tag}/focal_gain_G", f"{s['G']:.6f}",
                        "-", "exact k*h"])
            w.writerow(["sensitivity", f"{tag}/S_rms", f"{s['S_rms']:.8f}",
                        "MPa/Vrms", "focal peak pressure"])
            w.writerow(["sensitivity", f"{tag}/S_pp", f"{s['S_pp']:.8f}",
                        "MPa/Vpp", "focal peak pressure"])
        w.writerow([])
        w.writerow(["drive", "Vpp", "Vrms", "p_focus_at_fr[MPa]",
                    "p_focus_at_design_f[MPa]"])
        for vpp, vrms, p1, p2 in volt_rows:
            w.writerow(["drive", f"{vpp:.2f}", f"{vrms:.4f}", f"{p1:.6f}",
                        f"{p2:.6f}"])
        w.writerow([])
        w.writerow(["required", "target_p_focus[MPa]", "u0[m/s]",
                    "Vpp_at_fr", "Vpp_at_design_f"])
        for tgt, u0, v1, v2 in need_rows:
            w.writerow(["required", f"{tgt:.2f}", f"{u0:.6f}", f"{v1:.4f}",
                        f"{v2:.4f}"])

    # スイープ生データ (実測 Z(f) との突き合わせ用)
    with open(cfg.out_dir / "klm_sweep.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["f[Hz]", "Z_abs[ohm]", "Z_phase[deg]", "G[S]", "B[S]",
                    "p_surface[Pa/Vpeak]", "p_focus[Pa/Vpeak]", "efficiency"])
        for i in range(fs.size):
            w.writerow([f"{fs[i]:.1f}", f"{Zabs[i]:.6f}", f"{Zph[i]:.4f}",
                        f"{Gc[i]:.8e}", f"{Bc[i]:.8e}", f"{p_surf[i]:.4f}",
                        f"{p_foc[i]:.4f}", f"{eff[i]:.6f}"])

    # =====================================================================
    #  図
    # =====================================================================
    fmhz = fs / 1e6
    ttl = (f"{cfg.freq_mhz:.3f} MHz 設計 / C-203 / t={pzt.t*1e3:.4f} mm / "
           f"A={pzt.A*1e6:.1f} mm²")

    def marks(ax):
        ax.axvline(f_r / 1e6, color="r", ls="--", lw=0.9,
                   label=f"直列共振 {f_r/1e6:.4f} MHz")
        ax.axvline(f_a / 1e6, color="b", ls=":", lw=0.9,
                   label=f"反共振 {f_a/1e6:.4f} MHz")
        ax.axvline(pzt.f_design / 1e6, color="k", ls="-.", lw=0.9,
                   label=f"設計 {pzt.f_design/1e6:.3f} MHz")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.semilogy(fmhz, Zabs, "k-", lw=1.4)
    marks(ax); ax.set_ylabel("|Z| [Ω]"); ax.set_xlabel("周波数 [MHz]")
    ax.set_title("電気インピーダンスの大きさ"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(fmhz, Zph, "k-", lw=1.4)
    marks(ax); ax.set_ylabel("位相 [deg]"); ax.set_xlabel("周波数 [MHz]")
    ax.set_title("電気インピーダンスの位相"); ax.grid(alpha=0.3)
    ax.axhline(0, color="gray", lw=0.6)

    ax = axes[1, 0]
    ax.plot(fmhz, Gc * 1e3, "g-", lw=1.4, label="G = Re(1/Z)")
    ax.plot(fmhz, Bc * 1e3, "c--", lw=0.9, alpha=0.7, label="B = Im(1/Z)")
    marks(ax); ax.set_ylabel("アドミタンス [mS]"); ax.set_xlabel("周波数 [MHz]")
    ax.set_title("コンダクタンス / サセプタンス"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(fmhz, p_surf / 1e3, "m-", lw=1.5, label="表面音圧")
    ax.set_ylabel("表面音圧 [kPa / V_peak]", color="m")
    ax.tick_params(axis="y", labelcolor="m")
    ax2 = ax.twinx()
    ax2.plot(fmhz, p_foc / 1e6, "r-", lw=1.0, alpha=0.6, label="焦点音圧")
    ax2.set_ylabel("焦点音圧 [MPa / V_peak]", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    marks(ax); ax.set_xlabel("周波数 [MHz]")
    if not np.isnan(bw3):
        ax.axvspan(b3l / 1e6, b3h / 1e6, color="orange", alpha=0.12)
    ax.set_title(f"送信応答 (-3dB 帯域 {bw3/1e3:.0f} kHz)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"KLM 応答  {ttl}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(cfg.out_dir / "klm_response.png", dpi=150)

    # インピーダンス単独図 (実測との照合用)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    a1.semilogy(fmhz, Zabs, "k-", lw=1.5)
    marks(a1); a1.set_ylabel("|Z| [Ω]"); a1.grid(alpha=0.3, which="both")
    a1.legend(fontsize=8)
    a1.set_title("インピーダンス予測 — 実測 Z(f) と重ねて較正する\n"
                 "(共振/反共振の位置と |Z| の谷の深さを合わせる)", fontsize=10)
    a2.plot(fmhz, Zph, "k-", lw=1.5)
    marks(a2); a2.axhline(0, color="gray", lw=0.6)
    a2.set_ylabel("位相 [deg]"); a2.set_xlabel("周波数 [MHz]"); a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.out_dir / "klm_impedance.png", dpi=150)
    plt.close("all")

    print(f"\n [KLM] 保存: klm_response.png / klm_impedance.png /")
    print(f"             results_klm.csv / klm_sweep.csv")
    print(f" [KLM] 計算時間: {time.time()-t0:.2f} s")

    return dict(f_r=f_r, f_a=f_a, S_rms_fr=s_fr["S_rms"], S_pp_fr=s_fr["S_pp"],
                S_pp_design=s_fd["S_pp"], bw3_hz=bw3, pzt=pzt)
