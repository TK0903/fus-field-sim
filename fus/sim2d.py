# -*- coding: utf-8 -*-
"""
2D 計算 — 軸上プロファイル / 焦点面横方向プロファイル / 軸-横 2D 断面マップ

出力 (すべて cfg.out_dir 以下):
  axial.png / lateral.png / fieldmap.png / results_2d.csv

--------------------------------------------------------------------------------
 物理的仮定 (3D 版と共通)
--------------------------------------------------------------------------------
  ・CW(連続波)定常解。実駆動は 6.6 ms バースト(1.5 MHz で ~1e4 サイクル)
    なので定常状態とみなせる。
  ・線形伝搬。0.57 MPa/1.5 MHz の平面波衝撃形成距離 ~17 cm >> 焦点域数 mm。
    → 非線形歪みは無視でき、|p| は peak negative pressure (PNP) と等価。
  ・自由音場(水中のみ)。頭蓋・脳組織は含まない。in-situ 値ではない。
  ・u0 は表面法線振動速度の振幅(peak)。RMS ではない。音圧も peak 表記で統一。
--------------------------------------------------------------------------------
"""

import csv
import math
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from .source import generate_source_points
from .rayleigh import make_prefactor, compute_pressure, gain_of
from .analysis import fwhm_at_level, find_sidelobe

import config as cfgmod


def run(cfg, device):
    """2D 計算を実行し、結果 dict を返す。図と CSV は cfg.out_dir に保存する。"""
    t0 = time.time()
    cdt = torch.complex128
    rdt = torch.float64

    # ---- 音源生成 -----------------------------------------------------------
    src, dS, sinfo = generate_source_points(cfg, device=device, dtype=rdt)
    pref = make_prefactor(cfg, device=device, dtype=cdt)
    p_surface = cfg.p_surface

    # ---- 場のグリッド -------------------------------------------------------
    d = cfg.lam / cfg.field_subdiv_per_lambda        # λ/8 以下
    z_axis = np.arange(cfg.axial_z_min, cfg.axial_z_max + d, d)
    x_lat = np.arange(-cfg.lateral_half, cfg.lateral_half + d, d)
    N_map = len(z_axis) * len(x_lat)

    print("-" * 72)
    print(f" [2D] 音源点数 Ns : {sinfo['N_source']:,}  (N_z={sinfo['N_z']} リング)")
    print(f" [2D] 場の点数    : {N_map:,}  (2D マップ {len(x_lat)}×{len(z_axis)})")
    print(f" [2D] 総ペア数    : {sinfo['N_source'] * N_map:,}")
    est = cfg.field_chunk * sinfo["N_source"] * 16 * 3 / (1024 ** 3)
    print(f" [2D] 推定VRAM    : {est:.3f} GB / チャンク (chunk={cfg.field_chunk})")
    print(f" [2D] 音源要素弧長: 目標 λ/{cfg.subdiv_per_lambda} = "
          f"{sinfo['ds_target']*1e6:.0f} µm "
          f"(最大子午線弧長 {sinfo['max_meridional_arc']*1e6:.0f} µm)")
    print(f" [2D] 面積検証    : Σ dS = {sinfo['area_sum']*1e6:.6f} mm² "
          f"(厳密 2πRh = {sinfo['area_exact']*1e6:.6f} mm²)")
    print(f" [2D] 場グリッド  : λ/{cfg.field_subdiv_per_lambda} = {d*1e6:.0f} µm")
    print("-" * 72)

    # =====================================================================
    #  検証: 一次基準 = 焦点利得の厳密解 G = k*h
    # =====================================================================
    # 球冠上の全点は焦点(0,0,R)から等距離 R → 焦点音圧は解析的に厳密:
    #   p_focus = i*rho*c*u0*k*h*exp(-i*k*R) ,  G = k*h
    # 音速を温度依存にすると k が変わるので、期待値も cfg.c から再計算する。
    G_exact = cfg.focal_gain_exact
    focus_pt = torch.tensor([[0.0, 0.0, cfg.R]], dtype=rdt, device=device)
    p_focus = compute_pressure(focus_pt, src, dS, cfg, pref, device=device, dtype=cdt)
    G_num = float(gain_of(p_focus, cfg)[0])
    rel_err = abs(G_num - G_exact) / G_exact

    print("\n[検証] 一次基準: 焦点利得の厳密解 G = k*h")
    print(f"  厳密解 G_exact = k*h = {G_exact:.5f}")
    print(f"  数値積分 G_num        = {G_num:.5f}")
    print(f"  相対誤差             = {rel_err*100:.4e} %  "
          f"(許容 {cfgmod.FOCAL_GAIN_TOL*100:.0f}%)")
    if rel_err > cfgmod.FOCAL_GAIN_TOL:
        raise RuntimeError(
            f"焦点利得が厳密値 G=k*h と {cfgmod.FOCAL_GAIN_TOL*100:.0f}% 以内で"
            f"一致しません (誤差 {rel_err*100:.3f}%)。実装/設定を確認してください。")
    print("  → OK (厳密値と一致)")
    print("  注: 球冠上の全点は焦点から等距離 R のため、焦点利得は音源分割数に")
    print("      依らず厳密に k*h になる。分割密度が効くのは軸外の場のみ。")

    # =====================================================================
    #  グリッド収束テスト
    # =====================================================================
    print("\n[検証] グリッド収束テスト")
    print(f"  {'subdiv':>6} {'N_z':>5} {'N_source':>9} {'G_num':>10} "
          f"{'相対誤差[%]':>12} {'|p|@z=7mm(軸外)':>16}")
    ref_pt = torch.tensor([[0.0, 0.0, 7.0e-3]], dtype=rdt, device=device)
    conv_rows = []
    for sd in cfgmod.CONV_SUBDIVS:
        sp, sdS, si = generate_source_points(cfg, subdiv=sd, device=device, dtype=rdt)
        g = float(gain_of(compute_pressure(focus_pt, sp, sdS, cfg, pref,
                                           device=device, dtype=cdt), cfg)[0])
        off = float(gain_of(compute_pressure(ref_pt, sp, sdS, cfg, pref,
                                             device=device, dtype=cdt), cfg)[0])
        err = abs(g - G_exact) / G_exact * 100
        conv_rows.append((sd, si["N_z"], si["N_source"], g, err, off))
        print(f"  {sd:>6} {si['N_z']:>5} {si['N_source']:>9,} {g:>10.5f} "
              f"{err:>12.3e} {off:>16.5f}")
    print("  焦点利得は全 subdiv で厳密。軸外 |p|@z=7mm は密度で収束。")

    # =====================================================================
    #  ① 軸上音圧分布
    # =====================================================================
    ax_pts = torch.tensor(np.stack([np.zeros_like(z_axis), np.zeros_like(z_axis),
                                    z_axis], axis=1), dtype=rdt, device=device)
    g_axis = gain_of(compute_pressure(ax_pts, src, dS, cfg, pref,
                                      device=device, dtype=cdt), cfg)
    ip = int(np.argmax(g_axis))
    z_peak, G_peak = z_axis[ip], g_axis[ip]
    z_shift = z_peak - cfg.R
    dof, z6l, z6r = fwhm_at_level(z_axis, g_axis, 0.5)   # -6dB 焦点長

    # =====================================================================
    #  ② 焦点面での横方向分布
    # =====================================================================
    lat_pts = torch.tensor(np.stack([x_lat, np.zeros_like(x_lat),
                                     np.full_like(x_lat, cfg.R)], axis=1),
                           dtype=rdt, device=device)
    g_lat = gain_of(compute_pressure(lat_pts, src, dS, cfg, pref,
                                     device=device, dtype=cdt), cfg)
    beamw, xl6, xr6 = fwhm_at_level(x_lat, g_lat, 0.5)   # -6dB ビーム幅
    sll_db, sll_x = find_sidelobe(x_lat, g_lat)

    # =====================================================================
    #  ③ 軸-横 2D 音場マップ
    # =====================================================================
    XX, ZZ = np.meshgrid(x_lat, z_axis)
    map_pts = torch.tensor(np.stack([XX.ravel(), np.zeros(XX.size), ZZ.ravel()],
                                    axis=1), dtype=rdt, device=device)
    g_map = gain_of(compute_pressure(map_pts, src, dS, cfg, pref,
                                     device=device, dtype=cdt), cfg).reshape(XX.shape)
    map_db = 20.0 * np.log10(np.clip(g_map / g_map.max(), 1e-6, None))

    # =====================================================================
    #  ⑨ 目標焦点音圧 → 必要な u0 と表面音圧
    # =====================================================================
    # |p_focus| = rho*c*u0*G  →  u0 = p_target/(rho*c*G)
    u0_rows = []
    for pmpa in cfgmod.TARGET_PRESSURES_MPA:
        u0 = pmpa * 1e6 / (cfg.rho * cfg.c * G_exact)
        u0_rows.append((pmpa, u0, cfg.rho * cfg.c * u0 / 1e6))

    # =====================================================================
    #  数値結果の表示
    # =====================================================================
    brain = cfgmod.MOUSE_BRAIN_THICK_MM
    print("\n" + "=" * 72)
    print(" [2D] 数値結果")
    print("=" * 72)
    print(f" ④ 焦点利得 G                = {G_num:.4f}  (厳密 k*h={G_exact:.4f})")
    print(f" ⑤ -6dB 横方向ビーム幅       = {beamw*1e3:.4f} mm")
    print(f" ⑥ -6dB 軸方向焦点長(DOF)    = {dof*1e3:.4f} mm")
    print(f"      マウス脳厚 ~{brain:.0f} mm との比 = {dof*1e3/brain:.2f}", end="")
    print(" → 焦点長が脳厚より長く深さ選択性は低い。"
          if dof * 1e3 > brain else " → 焦点長は脳厚より短く軸方向の局在は十分。")
    print(f" ⑦ 音圧最大点 z              = {z_peak*1e3:.4f} mm "
          f"(幾何焦点 {cfg.R*1e3:.3f} mm からのずれ {z_shift*1e3:+.4f} mm)")
    print(f"      高利得 k*h={G_exact:.1f}>>1 のためシフトは小さい(測定値の報告)。")
    if sll_x == sll_x:
        print(f" ⑧ 最大サイドローブ          = {sll_db:.2f} dB @ x = {sll_x*1e3:+.4f} mm")
    else:
        print(f" ⑧ 最大サイドローブ          = 範囲内に明瞭なものなし")
    print("\n ⑨ 目標焦点音圧を得る表面法線振動速度 u0 と表面音圧:")
    print(f"    {'p_focus[MPa]':>12} {'u0[m/s]':>12} {'p_surface[MPa]':>16}")
    for pmpa, u0, psurf in u0_rows:
        print(f"    {pmpa:>12.2f} {u0:>12.5f} {psurf:>16.5f}")
    print("    ※ これは音響側の要求値。駆動電圧[Vpp]への換算には素子の電気音響")
    print("       変換効率が別途必要で、本計算だけでは電圧余裕は判定できない。")

    # =====================================================================
    #  CSV
    # =====================================================================
    csv_path = cfg.out_dir / "results_2d.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["# run", cfg.run_name])
        w.writerow(["# freq_MHz", f"{cfg.freq_mhz}", "coating_um", f"{cfg.coating_um}",
                    "mode", cfg.coating_mode, "T_C", f"{cfg.water_temp_c}"])
        w.writerow(["# R_eff_mm", f"{cfg.R*1e3:.6f}", "a_eff_mm", f"{cfg.a*1e3:.6f}",
                    "h_eff_mm", f"{cfg.h*1e3:.6f}", "c_m_s", f"{cfg.c:.4f}"])
        w.writerow([])
        w.writerow(["item", "quantity", "value", "unit", "note"])
        w.writerow(["4", "focal_gain_G", f"{G_num:.6f}", "-", f"exact k*h={G_exact:.6f}"])
        w.writerow(["5", "lateral_-6dB_beamwidth", f"{beamw*1e3:.6f}", "mm", ""])
        w.writerow(["6", "axial_-6dB_DOF", f"{dof*1e3:.6f}", "mm",
                    f"mouse_brain~{brain:.0f}mm ratio={dof*1e3/brain:.3f}"])
        w.writerow(["7", "peak_z", f"{z_peak*1e3:.6f}", "mm",
                    f"shift_from_geom_focus={z_shift*1e3:+.6f}mm"])
        w.writerow(["8", "max_sidelobe_level", f"{sll_db:.4f}", "dB",
                    f"x={sll_x*1e3:+.4f}mm" if sll_x == sll_x else "none"])
        w.writerow([])
        w.writerow(["9", "target_p_focus[MPa]", "u0[m/s]", "p_surface[MPa]",
                    "Vpp needs electro-acoustic efficiency (not computable here)"])
        for pmpa, u0, psurf in u0_rows:
            w.writerow(["9", f"{pmpa:.2f}", f"{u0:.6f}", f"{psurf:.6f}", ""])
        w.writerow([])
        w.writerow(["conv", "subdiv", "N_z", "N_source", "G_num", "relerr[%]",
                    "gain@z7mm"])
        for row in conv_rows:
            w.writerow(["conv", row[0], row[1], row[2], f"{row[3]:.6f}",
                        f"{row[4]:.3e}", f"{row[5]:.6f}"])

    # =====================================================================
    #  図
    # =====================================================================
    ttl = f"{cfg.freq_mhz:.3f} MHz / コート {cfg.coating_um:g} µm"

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(z_axis * 1e3, g_axis, "b-", lw=1.5, label="Rayleigh(厳密)")
    ax.axvline(cfg.R * 1e3, color="k", ls=":", lw=0.8,
               label=f"幾何焦点 {cfg.R*1e3:.2f}mm")
    ax.axvline(z_peak * 1e3, color="r", ls="-.", lw=0.8,
               label=f"最大 {z_peak*1e3:.2f}mm")
    if dof == dof:
        ax.axhline(0.5 * G_peak, color="gray", ls="--", lw=0.6)
        ax.annotate(f"-6dB DOF={dof*1e3:.2f}mm",
                    ((z6l + z6r) / 2 * 1e3, 0.52 * G_peak), color="gray", ha="center")
    ax.set_xlabel("z [mm] (apex=0)"); ax.set_ylabel("利得 |p|/(ρc u0)")
    ax.set_title(f"軸上音圧分布  {ttl}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(cfg.out_dir / "axial.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x_lat * 1e3, g_lat, "b-", lw=1.5)
    ax.axhline(0.5 * g_lat.max(), color="gray", ls="--", lw=0.6, label="-6dB")
    if beamw == beamw:
        ax.annotate(f"-6dB幅={beamw*1e3:.3f}mm", (0, 0.52 * g_lat.max()),
                    ha="center", color="gray")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("利得 |p|/(ρc u0)")
    ax.set_title(f"焦点面(z={cfg.R*1e3:.2f}mm)横方向分布  {ttl}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(cfg.out_dir / "lateral.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.pcolormesh(ZZ * 1e3, XX * 1e3, map_db, cmap="inferno",
                       vmin=-30, vmax=0, shading="auto")
    ax.axvline(cfg.R * 1e3, color="cyan", ls=":", lw=0.8)
    ax.set_xlabel("z [mm] (apex=0)"); ax.set_ylabel("x [mm]")
    ax.set_title(f"音場マップ [dB]  {ttl}")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="正規化音圧 [dB]")
    fig.tight_layout(); fig.savefig(cfg.out_dir / "fieldmap.png", dpi=150)
    plt.close("all")

    print(f"\n [2D] 保存: axial.png / lateral.png / fieldmap.png / results_2d.csv")
    print(f" [2D] 計算時間: {time.time()-t0:.2f} s")

    return dict(focal_gain=G_num, focal_gain_exact=G_exact,
                beamwidth_mm=beamw * 1e3, dof_mm=dof * 1e3,
                peak_z_mm=z_peak * 1e3, shift_mm=z_shift * 1e3,
                sidelobe_db=sll_db, u0_rows=u0_rows)
