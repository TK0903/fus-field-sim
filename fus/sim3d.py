# -*- coding: utf-8 -*-
"""
3D 計算 — 焦点体積の等圧面 (isosurface) 可視化と体積定量

出力 (すべて cfg.out_dir 以下):
  iso3d_focal.png / iso3d_nested.png / iso3d_views.png / iso3d_rzplane.png
  results_3d.csv

--------------------------------------------------------------------------------
 なぜ marching cubes ではなく「回転体(surface of revolution)」なのか
--------------------------------------------------------------------------------
  音源は球冠 + 一様 u0 なので、音場は z 軸まわりに【厳密に軸対称】である。したがって:
    (1) 計算は r-z 半平面だけでよい → 3D 全格子を回すより圧倒的に安い
    (2) 等圧面は「半平面の等圧線を z 軸で回した回転体」として厳密に得られる
        → voxel 格子に由来する階段状アーティファクトが原理的に発生しない
    (3) 体積は Pappus の定理で厳密積分できる: V = ∫∫_{|p|>=thr} 2*pi*r dr dz
        → voxel カウント(離散和)より高精度
  marching cubes (scikit-image) は「軸対称でない一般の場」向けの手法であり、
  本問題ではむしろ精度が落ちる。依存も増えるため採用しない。
  ただし「本当に軸対称か」は下の [検証2] で直接 Rayleigh 積分により実測する。
--------------------------------------------------------------------------------
"""

import csv
import math
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .source import generate_source_points
from .rayleigh import make_prefactor, compute_pressure, gain_of
from .analysis import (extract_iso_contours, revolve_contour, revolved_volume,
                       touches_domain_wall)

import config as cfgmod


def run(cfg, device):
    """3D 計算を実行し、結果 dict を返す。図と CSV は cfg.out_dir に保存する。"""
    t0 = time.time()
    cdt = torch.complex128
    rdt = torch.float64

    src, dS, sinfo = generate_source_points(cfg, device=device, dtype=rdt)
    pref = make_prefactor(cfg, device=device, dtype=cdt)

    # ---- r-z グリッド -------------------------------------------------------
    d = cfg.lam / cfg.rz_subdiv
    r_1d = np.arange(0.0, cfg.r_max_3d + d, d)
    z_1d = np.arange(cfg.z_min_3d, cfg.z_max_3d + d, d)
    N_field = len(r_1d) * len(z_1d)

    print("-" * 72)
    print(f" [3D] 音源点数 Ns : {sinfo['N_source']:,}  (N_z={sinfo['N_z']} リング)")
    print(f" [3D] 場の点数    : {N_field:,}  (r-z 半平面 {len(r_1d)}×{len(z_1d)})")
    print(f" [3D] 総ペア数    : {sinfo['N_source'] * N_field:,}")
    est = cfg.field_chunk * sinfo["N_source"] * 16 * 3 / (1024 ** 3)
    print(f" [3D] 推定VRAM    : {est:.3f} GB / チャンク (chunk={cfg.field_chunk})")
    print(f" [3D] r-z グリッド: λ/{cfg.rz_subdiv} = {d*1e6:.0f} µm")
    print(f" [3D] 面積検証    : Σ dS = {sinfo['area_sum']*1e6:.6f} mm² "
          f"(厳密 2πRh = {sinfo['area_exact']*1e6:.6f} mm²)")
    print("-" * 72)

    # =====================================================================
    #  [検証1] 一次基準: 焦点利得の厳密解 G = k*h
    # =====================================================================
    G_exact = cfg.focal_gain_exact
    focus_pt = torch.tensor([[0.0, 0.0, cfg.R]], dtype=rdt, device=device)
    G_num = float(gain_of(compute_pressure(focus_pt, src, dS, cfg, pref,
                                           device=device, dtype=cdt), cfg)[0])
    rel_err = abs(G_num - G_exact) / G_exact
    print("\n[検証1] 焦点利得の厳密解 G = k*h")
    print(f"  厳密 {G_exact:.5f} / 数値 {G_num:.5f} / 相対誤差 {rel_err*100:.3e} %")
    if rel_err > cfgmod.FOCAL_GAIN_TOL:
        raise RuntimeError(f"焦点利得が厳密値と "
                           f"{cfgmod.FOCAL_GAIN_TOL*100:.0f}% 以内で一致しません "
                           f"(誤差 {rel_err*100:.3f}%)。")
    print("  → OK")

    # =====================================================================
    #  r-z 半平面の音場計算 (軸対称なので y=0 の半平面だけでよい)
    # =====================================================================
    RR, ZZ = np.meshgrid(r_1d, z_1d, indexing="ij")      # (Nr, Nz)
    rz_pts = torch.tensor(np.stack([RR.ravel(), np.zeros(RR.size), ZZ.ravel()],
                                   axis=1), dtype=rdt, device=device)
    g_rz = gain_of(compute_pressure(rz_pts, src, dS, cfg, pref,
                                    device=device, dtype=cdt), cfg).reshape(RR.shape)
    p_peak = g_rz.max()
    i_pk = np.unravel_index(np.argmax(g_rz), g_rz.shape)
    z_peak = z_1d[i_pk[1]]
    field_db = 20.0 * np.log10(np.clip(g_rz / p_peak, 1e-12, None))
    print(f"\n[音場] r-z 半平面のピーク利得 = {p_peak:.4f} @ z = {z_peak*1e3:.4f} mm")

    # =====================================================================
    #  [検証2] 本当に軸対称か (回転体で等圧面を作る根拠)
    # =====================================================================
    # 同じ半径・同じ z を方位角の異なる 4 点として直接 Rayleigh 積分し、
    # 値が一致するかを見る。一致すれば「r-z 半平面を回す」操作は厳密。
    rc, zc = 1.5e-3, cfg.R
    chk = torch.tensor([[rc, 0.0, zc], [0.0, rc, zc],
                        [rc*math.cos(math.pi/4), rc*math.sin(math.pi/4), zc],
                        [-rc*math.cos(math.pi/3), rc*math.sin(math.pi/3), zc]],
                       dtype=rdt, device=device)
    g_chk = gain_of(compute_pressure(chk, src, dS, cfg, pref,
                                     device=device, dtype=cdt), cfg)
    axisym_err = float((g_chk.max() - g_chk.min()) / g_chk.mean())
    print(f"\n[検証2] 軸対称性 (r={rc*1e3:.1f}mm, z={zc*1e3:.2f}mm を方位4点で直接計算)")
    print(f"  値: {np.array2string(g_chk, precision=8)}")
    print(f"  方位ばらつき(相対) = {axisym_err:.3e}  (許容 {cfgmod.AXISYM_TOL:.0e})")
    if axisym_err > cfgmod.AXISYM_TOL:
        print("  ! 警告: 方位ばらつきが大きい。音源サンプリングの方位一様性を確認。")
    else:
        print("  → OK (厳密に軸対称 → 回転体が厳密な等圧面になる)")

    # =====================================================================
    #  等圧面の抽出・体積定量
    # =====================================================================
    print("\n[等圧面] 抽出と体積定量")
    iso = []
    for lvl in cfgmod.ISO_LEVELS_DB:
        thr = p_peak * (10.0 ** (lvl / 20.0))
        segs = extract_iso_contours(ZZ, RR, field_db, lvl)
        vol_mm3 = revolved_volume(r_1d, z_1d, g_rz, thr) * 1e9
        mask = g_rz >= thr
        zs = z_1d[np.any(mask, axis=0)]
        z_span = (zs.min(), zs.max())
        r_max = r_1d[np.any(mask, axis=1)].max()
        clipped = touches_domain_wall(mask)
        iso.append(dict(level_db=lvl, thr=thr, segs=segs, vol_mm3=vol_mm3,
                        z_span=z_span, r_max=r_max, n_seg=len(segs),
                        clipped=clipped))
        flag = "  ← 領域端で切れており下限値" if clipped else ""
        print(f"  {lvl:+.0f} dB : 体積 {vol_mm3:9.4f} mm³ / 等圧線 {len(segs)} 本 / "
              f"z {z_span[0]*1e3:.2f}-{z_span[1]*1e3:.2f} mm / "
              f"最大半径 {r_max*1e3:.4f} mm{flag}")

    iso6 = min(iso, key=lambda x: abs(x["level_db"] + 6.0))
    v6 = iso6["vol_mm3"]
    L6 = (iso6["z_span"][1] - iso6["z_span"][0]) * 1e3
    W6 = 2.0 * iso6["r_max"] * 1e3
    v_el = math.pi / 6.0 * W6 * W6 * L6
    brain_v = cfgmod.MOUSE_BRAIN_VOLUME_MM3
    brain_t = cfgmod.MOUSE_BRAIN_THICK_MM

    print("\n" + "=" * 72)
    print(" [3D] 数値結果 (焦点体積)")
    print("=" * 72)
    print(f" -6dB 焦点体積        = {v6:.4f} mm³"
          + ("  ! 領域端で切れており下限値" if iso6["clipped"]
             else "  (領域内で閉じている → 確定値)"))
    print(f"   外形: 横 {W6:.4f} mm × 軸 {L6:.4f} mm  (縦横比 {L6/W6:.2f})")
    print(f"   等価回転楕円体 (π/6·W²·L) = {v_el:.4f} mm³ (比 {v6/v_el:.3f})")
    print(f"   マウス脳 (~{brain_v:.0f} mm³) に対する割合 = {v6/brain_v*100:.4f} %")
    print(f"   軸方向長 {L6:.2f} mm vs マウス脳厚 ~{brain_t:.0f} mm "
          f"→ 比 {L6/brain_t:.2f}")
    print("     → 焦点柱が脳厚を貫く。深さ選択性は期待できない。"
          if L6 > brain_t else
          "     → 焦点柱は脳厚に収まる。ただし軸方向は横方向より大幅に長い。")
    iso20 = min(iso, key=lambda x: abs(x["level_db"] + 20.0))
    print(f" -20dB 体積           = {iso20['vol_mm3']:.4f} mm³ "
          f"(サイドローブ環含む: 等圧線 {iso20['n_seg']} 本)")
    if iso20["clipped"]:
        print(f"   ! 計算領域 (r<={cfg.r_max_3d*1e3:.1f}mm, "
              f"z={cfg.z_min_3d*1e3:.0f}-{cfg.z_max_3d*1e3:.0f}mm) の壁で切れているため")
        print("     物理的な体積ではなく【下限値】。定量に使うなら領域を広げること。")
    print("\n ※ 自由音場・線形・CW の値。頭蓋や脳組織は含まない(in-situ ではない)。")

    # =====================================================================
    #  CSV
    # =====================================================================
    with open(cfg.out_dir / "results_3d.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["# run", cfg.run_name])
        w.writerow(["# freq_MHz", f"{cfg.freq_mhz}", "coating_um", f"{cfg.coating_um}",
                    "mode", cfg.coating_mode, "T_C", f"{cfg.water_temp_c}"])
        w.writerow(["# R_eff_mm", f"{cfg.R*1e3:.6f}", "a_eff_mm", f"{cfg.a*1e3:.6f}",
                    "h_eff_mm", f"{cfg.h*1e3:.6f}", "c_m_s", f"{cfg.c:.4f}"])
        w.writerow([])
        w.writerow(["item", "quantity", "value", "unit", "note"])
        w.writerow(["3D", "focal_volume_-6dB", f"{v6:.6f}", "mm^3",
                    "CLIPPED_lower_bound" if iso6["clipped"] else "closed_in_domain"])
        w.writerow(["3D", "focal_width_-6dB", f"{W6:.6f}", "mm", "lateral (2*r_max)"])
        w.writerow(["3D", "focal_length_-6dB", f"{L6:.6f}", "mm", "axial extent"])
        w.writerow(["3D", "aspect_ratio", f"{L6/W6:.6f}", "-", "axial/lateral"])
        w.writerow(["3D", "equiv_ellipsoid_volume", f"{v_el:.6f}", "mm^3",
                    f"fill_ratio={v6/v_el:.4f}"])
        w.writerow(["3D", "fraction_of_mouse_brain", f"{v6/brain_v*100:.6f}", "%",
                    f"brain~{brain_v:.0f}mm^3"])
        w.writerow(["3D", "volume_-20dB", f"{iso20['vol_mm3']:.6f}", "mm^3",
                    ("includes sidelobes; CLIPPED_lower_bound" if iso20["clipped"]
                     else "includes sidelobes; closed_in_domain")])
        w.writerow(["3D", "peak_gain", f"{p_peak:.6f}", "-",
                    f"exact k*h={G_exact:.6f}"])
        w.writerow(["3D", "peak_z", f"{z_peak*1e3:.6f}", "mm", ""])
        w.writerow(["chk", "axisymmetry_rel_spread", f"{axisym_err:.3e}", "-",
                    "revolve validity"])

    # =====================================================================
    #  図
    # =====================================================================
    colors = {-6.0: "#e8483c", -20.0: "#4c8fd6"}
    ttl = f"{cfg.freq_mhz:.3f} MHz / コート {cfg.coating_um:g} µm"

    def axes3d(ax, title, lat_mm, z_lo, z_hi, fs=11):
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
        ax.set_title(title, fontsize=fs)
        ax.set_xlim(-lat_mm, lat_mm); ax.set_ylim(-lat_mm, lat_mm)
        ax.set_zlim(z_lo, z_hi)
        ax.set_box_aspect((2 * lat_mm, 2 * lat_mm, z_hi - z_lo))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(labelsize=8)

    lat6 = max(0.6, iso6["r_max"] * 1e3 * 1.7)
    z6lo = iso6["z_span"][0] * 1e3 - 0.5
    z6hi = iso6["z_span"][1] * 1e3 + 0.5
    NP, CUT = cfgmod.N_PHI, cfgmod.CUTAWAY_DEG

    # 図1: -6dB 焦点体積 (360° 閉曲面・ローブにズーム)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for seg in iso6["segs"]:
        ax.add_collection3d(Poly3DCollection(
            revolve_contour(seg, NP, 360.0) * 1e3, alpha=0.5,
            facecolor=colors[-6.0], edgecolor="none"))
    ax.plot([0, 0], [0, 0], [z6lo, z6hi], "k:", lw=0.8)
    ax.scatter([0], [0], [cfg.R*1e3], color="k", s=40, marker="x",
               label=f"幾何焦点 z={cfg.R*1e3:.2f}mm", depthshade=False)
    ax.scatter([0], [0], [z_peak*1e3], color="lime", s=40, marker="o",
               label=f"音圧最大 z={z_peak*1e3:.2f}mm", depthshade=False)
    axes3d(ax, f"-6dB 焦点体積  {ttl}\nV = {v6:.3f} mm³ "
               f"(横 {W6:.2f} × 軸 {L6:.2f} mm, 縦横比 {L6/W6:.1f})",
           lat6, z6lo, z6hi)
    ax.legend(fontsize=8, loc="upper left"); ax.view_init(elev=16, azim=-60)
    fig.tight_layout(); fig.savefig(cfg.out_dir / "iso3d_focal.png", dpi=150)

    # 図2: -6dB と -20dB の入れ子 (カットアウェイで内部を見せる)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for dct in sorted(iso, key=lambda x: x["level_db"]):     # 外側(-20dB)から
        alpha = 0.16 if dct["level_db"] <= -20.0 else 0.75
        col = colors.get(dct["level_db"], "#888888")
        for seg in dct["segs"]:
            ax.add_collection3d(Poly3DCollection(
                revolve_contour(seg, NP, CUT) * 1e3, alpha=alpha,
                facecolor=col, edgecolor="none"))
    ax.plot([0, 0], [0, 0], [cfg.z_min_3d*1e3, cfg.z_max_3d*1e3], "k:", lw=0.8)
    note = "  ※-20dB は領域端で切れている(下限値)" if iso20["clipped"] else ""
    axes3d(ax, f"等圧面  {ttl}\n赤 -6dB ({v6:.2f} mm³) / "
               f"青 -20dB ({iso20['vol_mm3']:.1f} mm³)\n"
               f"{CUT:.0f}°カットアウェイ{note}",
           cfg.r_max_3d*1e3, cfg.z_min_3d*1e3, cfg.z_max_3d*1e3, fs=10)
    ax.view_init(elev=16, azim=-45)      # 切り欠きを正面から覗く
    fig.tight_layout(); fig.savefig(cfg.out_dir / "iso3d_nested.png", dpi=150)

    # 図3: 多視点
    # 注: 真上(elev≈90)は box_aspect を実寸比に保つ都合でパネルが潰れるため使わない。
    #     断面が円であることは [検証2] の軸対称性で数値的に実証済み。
    fig = plt.figure(figsize=(11, 9))
    for i, ((el, az), vn) in enumerate(zip(
            [(16, -60), (0, -90), (45, -60), (30, 25)],
            ["斜め (elev 16°)", "真横 x-z 断面 (elev 0°)",
             "高い角度から (elev 45°)", "斜め・反対側 (elev 30°)"]), start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        for seg in iso6["segs"]:
            ax.add_collection3d(Poly3DCollection(
                revolve_contour(seg, NP, 360.0) * 1e3, alpha=0.5,
                facecolor=colors[-6.0], edgecolor="none"))
        ax.plot([0, 0], [0, 0], [z6lo, z6hi], "k:", lw=0.8)
        axes3d(ax, vn, lat6, z6lo, z6hi, fs=10)
        ax.view_init(elev=el, azim=az)
    fig.suptitle(f"-6dB 焦点体積 多視点  {ttl}   V = {v6:.3f} mm³", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.subplots_adjust(wspace=0.05, hspace=0.18)
    fig.savefig(cfg.out_dir / "iso3d_views.png", dpi=150)

    # 図4: 元になった r-z 半平面と等圧線 (3D 化の根拠)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    im = ax.pcolormesh(ZZ * 1e3, RR * 1e3, field_db, cmap="inferno",
                       vmin=-30, vmax=0, shading="auto")
    for dct in iso:
        col = colors.get(dct["level_db"], "#888888")
        for si, seg in enumerate(dct["segs"]):
            ax.plot(seg[:, 0]*1e3, seg[:, 1]*1e3, color=col, lw=1.6,
                    label=(f"{dct['level_db']:+.0f} dB" if si == 0 else None))
    ax.axvline(cfg.R * 1e3, color="cyan", ls=":", lw=0.8)
    ax.set_xlabel("z [mm] (apex=0)"); ax.set_ylabel("r [mm]")
    ax.set_title(f"r-z 半平面と等圧線 (これを z 軸で回して 3D 化)  {ttl}")
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax, label="正規化音圧 [dB]")
    fig.tight_layout(); fig.savefig(cfg.out_dir / "iso3d_rzplane.png", dpi=150)
    plt.close("all")

    print(f"\n [3D] 保存: iso3d_focal.png / iso3d_nested.png / iso3d_views.png /")
    print(f"            iso3d_rzplane.png / results_3d.csv")
    print(f" [3D] 計算時間: {time.time()-t0:.2f} s")

    return dict(focal_volume_mm3=v6, width_mm=W6, length_mm=L6,
                peak_gain=p_peak, peak_z_mm=z_peak * 1e3,
                axisym_err=axisym_err)
