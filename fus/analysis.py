# -*- coding: utf-8 -*-
"""
プロファイル解析・等圧面抽出・体積積分の補助関数
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
#  プロファイル解析
# ==============================================================================
def fwhm_at_level(x, y, level_frac=0.5):
    """
    プロファイル y(x) のピークに対し、レベル level_frac (-6dB は 0.5) を横切る
    左右の交点を線形補間で求め、全幅を返す。見つからなければ NaN。

    Returns: (幅, 左交点, 右交点)
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    ip = int(np.argmax(y))
    thr = y[ip] * level_frac

    def cross(idx_range):
        prev = ip
        for j in idx_range:
            if y[j] < thr <= y[prev] or y[j] <= thr < y[prev]:
                x0, x1, y0, y1 = x[prev], x[j], y[prev], y[j]
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
    ピークから外側へ歩き、最初の極小(ヌル)を越えた領域の最大値を採る。

    Returns: (ピーク基準のレベル [dB], 位置) / 見つからなければ (nan, nan)
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    ip = int(np.argmax(y))
    ypk = y[ip]

    def scan(idx_range):
        found_null = False
        best, best_x = 0.0, np.nan
        pv = ypk
        for j in idx_range:
            if not found_null and y[j] > pv:
                found_null = True          # 下降が終わって上昇 → 直前が極小
            if found_null and y[j] > best:
                best, best_x = y[j], x[j]
            pv = y[j]
        return best, best_x

    bl, xl = scan(range(ip - 1, -1, -1))
    br, xr = scan(range(ip + 1, len(x)))
    best, bx = (br, xr) if br >= bl else (bl, xl)
    if best <= 0 or np.isnan(bx):
        return np.nan, np.nan
    return 20.0 * math.log10(best / ypk), bx


# ==============================================================================
#  等圧面 (軸対称の回転体として厳密に扱う)
# ==============================================================================
def extract_iso_contours(z_grid, r_grid, field_db, level_db):
    """
    r-z 半平面 (r>=0) 上で field_db == level_db となる等圧線を抽出する。
    matplotlib の contour を「線の抽出器」として使う(描画はしない)。

    半平面での等圧線の形と、それを回した立体の対応:
      ・両端が r=0(軸)で終わる弧 → 回すと閉じた「葉巻」= 主ローブ
      ・軸に触れない閉ループ      → 回すとトーラス(輪) = サイドローブ環
    どちらも同じ処理で自然に扱える。

    Returns: 各等圧線の (z[m], r[m]) 頂点列のリスト
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contour(z_grid, r_grid, field_db, levels=[level_db])

    segs = []
    try:
        segs = [np.asarray(s) for s in cs.allsegs[0] if len(s) >= 3]
    except Exception:
        for p in cs.get_paths():           # 新しい matplotlib API
            v, codes = p.vertices, p.codes
            if codes is None:
                if len(v) >= 3:
                    segs.append(np.asarray(v))
            else:
                starts = np.flatnonzero(codes == 1)   # MOVETO
                bounds = list(starts) + [len(v)]
                for s, e in zip(bounds[:-1], bounds[1:]):
                    if e - s >= 3:
                        segs.append(np.asarray(v[s:e]))
    plt.close(fig)
    return segs


def revolve_contour(seg_zr, n_phi, sweep_deg=360.0):
    """
    半平面の等圧線 (z, r) を z 軸まわりに回して 3D 面(四角形メッシュ)を作る。

    音場が軸対称なので、この回転体が【厳密な等圧面】になる。
    voxel 格子を経由しないため階段状アーティファクトが原理的に出ない。
    sweep_deg < 360 にすると回し残しができ、断面が見える(カットアウェイ)。

    Returns: quads (Nq, 4, 3) の頂点配列 [m]
    """
    z = seg_zr[:, 0]
    r = np.clip(seg_zr[:, 1], 0.0, None)     # r<0 は数値誤差なのでクリップ
    phi = np.deg2rad(np.linspace(0.0, sweep_deg, n_phi))

    X = r[:, None] * np.cos(phi)[None, :]
    Y = r[:, None] * np.sin(phi)[None, :]
    Z = np.repeat(z[:, None], len(phi), axis=1)

    q = []
    for i in range(len(z) - 1):
        for j in range(len(phi) - 1):
            q.append([[X[i, j],     Y[i, j],     Z[i, j]],
                      [X[i+1, j],   Y[i+1, j],   Z[i+1, j]],
                      [X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]],
                      [X[i, j+1],   Y[i, j+1],   Z[i, j+1]]])
    return np.asarray(q)


def revolved_volume(r_1d, z_1d, field_lin, thr):
    """
    Pappus の定理による厳密体積積分:
        V = ∫∫_{|p|>=thr} 2*pi*r dr dz
    voxel カウント(離散和)ではなく、r-z セルごとに軸対称の環体積を積む。

    Returns: 体積 [m^3]
    """
    dr = r_1d[1] - r_1d[0]
    dz = z_1d[1] - z_1d[0]
    mask = (field_lin >= thr)                # (Nr, Nz)
    return float(np.sum(mask * (2.0 * math.pi * r_1d[:, None] * dr * dz)))


def touches_domain_wall(mask):
    """
    しきい値超え領域が計算領域の壁 (r=r_max, z=z_min, z=z_max) に接しているか。
    接していれば等圧面は箱の外へ続いており、体積は「下限値」にすぎない。
    (r=0 は対称軸であって壁ではないので判定に含めない)
    """
    return bool(mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())
