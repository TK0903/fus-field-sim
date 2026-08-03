# -*- coding: utf-8 -*-
"""
fus — 球面集束トランスデューサの音場計算パッケージ

  medium.py   : 媒質(水)の物性
  source.py   : 球冠音源の等面積サンプリング
  rayleigh.py : Rayleigh 積分の GPU 計算
  analysis.py : プロファイル解析・等圧面抽出・体積積分
  plotting.py : 図の共通設定
  sim2d.py    : 2D 計算 (軸上/横方向プロファイル・2D 断面マップ)
  sim3d.py    : 3D 計算 (焦点体積の等圧面)

条件はすべてルートの config.py から Config オブジェクトとして渡される。
"""
