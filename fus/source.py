# -*- coding: utf-8 -*-
"""
球冠音源の等面積サンプリング

--------------------------------------------------------------------------------
 座標系 (全スクリプト共通)
--------------------------------------------------------------------------------
  原点 = 素子の頂点 (apex, 軸上で最も凹んだ点)、+z を放射方向とする。
  球面の中心 = 幾何学的焦点 = (0,0,R)。

        z=0 apex                    +z 放射方向 →→→
          \\__                       focus
             \\___                  (0,0,R)
                \\____                 *
  ───────────────────●──────────────────┼────────  z 軸
                 rim (z=h, 半径 a)      z=R  幾何焦点

    ・リム(開口縁): z = h の平面上、半径 a の円
    ・幾何学的焦点: z = R
    ・コーティングを入れると R, a, h が実効値に置き換わる (config.py 参照)
--------------------------------------------------------------------------------
"""

import math
import numpy as np
import torch


def generate_source_points(cfg, subdiv=None, device=None, dtype=None):
    """
    球冠表面を等面積サンプリングする。

    等面積の原理 (Archimedes の定理):
      球面を軸 z に垂直な平面で切った「帯(zone)」の面積は、その z 幅にのみ比例:
        dA = 2*pi*R*dz
      よって z を等間隔に分割すれば、各リングの面積は自動的に等しくなる。
      (緯度経度の単純分割は極付近が密になるので使わない。)

    各リングの面積 2*pi*R*dz を M_i 個の要素に分ける。方位分割数 M_i はその
    リングの円筒半径に比例させ、方位方向の弧長を一定(≈λ/subdiv)に保つ
    (要素アスペクト比を 1 に近づける)。

    メッシュの向き:
      球面中心は C=(0,0,R)。高さ z のリングの円筒半径は
        rho(z) = sqrt(R^2 - (z-R)^2) = sqrt(2*R*z - z^2)
      このとき任意の表面点から焦点(0,0,R)までの距離は厳密に R になる。
      これが「焦点利得 G = k*h が数値積分に依らず厳密に出る」根拠。

    弧長間隔について:
      目標弧長 ds = λ/subdiv。
        ・方位方向 : M_i = round(2*pi*rho_i/ds) で ds にそろう。
        ・子午線方向: 等 z 分割は「等面積」を保証するが弧長は非一様
          (リム側で細かく、apex 近傍で粗い)。N_z は子午線全長 R*theta_max を
          ds で割って平均分解能を ds にそろえる。apex 近傍が粗くなるのは許容:
            (1) 焦点利得の厳密性には影響しない(全点が焦点から等距離 R)
            (2) apex は立体角が小さく寄与が小さい
          より厳密にしたい場合は config の SUBDIV_PER_LAMBDA を上げる。

    Returns
    -------
    pts  : (Ns,3) tensor  音源点座標 [m]
    dS   : (Ns,)  tensor  各音源点の担当面積 [m^2]  (Σ dS = 2*pi*R*h 厳密)
    info : dict           N_z / N_source / 面積検証 など
    """
    subdiv = cfg.subdiv_per_lambda if subdiv is None else subdiv
    dtype = torch.float64 if dtype is None else dtype
    R, h = cfg.R, cfg.h

    theta_max = cfg.theta_max
    ds_target = cfg.lam / subdiv

    # 子午線全長 = R*theta_max。平均弧長が ds_target 以下になるよう N_z を決定。
    N_z = max(1, math.ceil(R * theta_max / ds_target))
    dz = h / N_z
    ring_area = 2.0 * math.pi * R * dz      # Archimedes: 全リング共通

    xs, ys, zs, dSs = [], [], [], []
    max_merid = 0.0
    prev_z = 0.0
    for i in range(N_z):
        z_i = (i + 0.5) * dz                # リング中心の z (apex の特異を回避)
        rho_i = math.sqrt(max(0.0, 2.0 * R * z_i - z_i * z_i))

        M_i = max(1, int(round(2.0 * math.pi * rho_i / ds_target)))
        dS_i = ring_area / M_i

        # 隣接リングとスポーク(放射状の筋)を作らないよう半ピッチ千鳥にずらす
        phi0 = math.pi / M_i if (i % 2) else 0.0
        phis = phi0 + 2.0 * math.pi * (np.arange(M_i) / M_i)

        xs.append(rho_i * np.cos(phis))
        ys.append(rho_i * np.sin(phis))
        zs.append(np.full(M_i, z_i))
        dSs.append(np.full(M_i, dS_i))

        # 子午線方向の弧長(診断用): 隣リング中心間の球面上距離
        def _ang(z):
            return math.acos(min(1.0, max(-1.0, 1.0 - z / R)))
        max_merid = max(max_merid, R * abs(_ang(z_i) - _ang(prev_z)))
        prev_z = z_i

    pts = torch.tensor(np.stack([np.concatenate(xs), np.concatenate(ys),
                                 np.concatenate(zs)], axis=1),
                       dtype=dtype, device=device)
    dS = torch.tensor(np.concatenate(dSs), dtype=dtype, device=device)

    info = dict(N_z=N_z, N_source=int(pts.shape[0]), ds_target=ds_target,
                max_meridional_arc=max_merid,
                area_sum=float(dS.sum().item()),
                area_exact=cfg.cap_area_exact)
    return pts, dS, info
