# coding:utf-8
# 4.1.4 フィッシャーの線形判別（p.185）

import numpy as np
from pylab import *
import sys

N = 100  # データ数


def f(x, a, b):
    # 決定境界の直線の方程式
    return a * x + b


if __name__ == "__main__":
    # 訓練データを作成
    cls1 = []
    cls2 = []

    # データは正規分布に従って生成
    mean1 = [1, 3]  # クラス1の平均
    mean2 = [3, 1]  # クラス2の平均
    cov = [[2.0, 0.0], [0.0, 0.1]]  # 共分散行列（全クラス共通）

    # データ作成
    cls1.extend(np.random.multivariate_normal(mean1, cov, N/2))
    cls2.extend(np.random.multivariate_normal(mean2, cov, N/2))

    # 各クラスの平均をプロット
    m1 = np.mean(cls1, axis=0)
    m2 = np.mean(cls2, axis=0)
    plot([m1[0]], [m1[1]], 'b+')
    plot([m2[0]], [m2[1]], 'r+')
    print(m1, m2)

    # 訓練データを描画
    x1, x2 = np.array(cls1).transpose()
    plot(x1, x2, 'bo')

    x1, x2 = np.array(cls2).transpose()
    plot(x1, x2, 'ro')

    # ラグランジュの未定乗数法で解いた結果
    w = m2 - m1

    # 識別境界を描画
    # wは識別境界と直交するベクトル
    a = - (w[0] / w[1])  # 識別直線の傾き

    # 傾きがaで平均の中点mを通る直線のy切片bを求める
    m = (m1 + m2) / 2
    b = -a * m[0] + m[1]  # 識別直線のy切片

    x1 = np.linspace(-2, 6, 1000)
    x2 = [f(x, a, b) for x in x1]
    plot(x1, x2, 'g-')

    xlim(-2, 6)
    ylim(-2, 4)
    show()