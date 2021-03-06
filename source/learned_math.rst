====================================
数学
====================================

:著者: Masato

概要
====================================
数学について学んだことをまとめてます。取り扱う分野は主に線形代数と確率統計になります。

確率統計
====================================
.. math:: (a + b)^2 = a^2 + 2ab + b^2

          (a - b)^2 = a^2 - 2ab + b^2

こんな感じに書いていきます。


尤度関数
------------------------------------
尤度関数の概念は「サンプリングしてデータが観測された後、
そのデータは元々どういうパラメータをもつ確率分布から生まれたものだったか？」
という問いに答えるもの。
逆確率的なベイズの定理っぽさがあると感じる。ここで、正規分布について示す。 ::

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    import seaborn as sns
    import numpy.random as rd

    m = 10
    s = 3

    min_x = m-4*s
    max_x = m+4*s

    x = np.linspace(min_x, max_x, 201)
    y = (1/np.sqrt(2*np.pi*s**2))*np.exp(-0.5*(x-m)**2/s**2)
    
    plt.figure(figsize=(8,5))
    plt.xlim(min_x, max_x)
    plt.ylim(0,max(y)*1.1)
    plt.plot(x,y)
    plt.show()

ここで、標本が10個手に入り、 :math:`x_1 , x_2 , ...., x_{10}` が正規分布に従うことが
わかっているが、平均 :math:`mu` 標準偏差 :math:`sigma` の2つのパラメータの値
がどれくらいなのか不明であるという状況を考える。 ::

    plt.figure(figure=(8,2))
    rd.seed(7)
    data = rd.normal(10, 3, 10, )
    plt.scatter(data, np.zeros_like(data), c="r", s=50)

「10個の標本がこの値となった同時分布」を考える。
また、この10個の標本はidd(独立同一分布:同じ分布から独立にとられた標本)である
と仮定する。独立なので、それぞれの確率密度の積として表せるので、

.. math:: P(x_1,x_2,..., x_{10}) = P(x_1)P(x_2)...P(x_{10})

となる。ここで、 :math:`P(x_i)` は全て正規分布としていたので、

.. math:: P(x_1,x_2,..., x_{10}) = f(x_1)f(x_2)...f(x_{10})

としても良い。これをさらに展開していくと、

.. math:: P(x_1,x_2,..., x_{10}) = \prod_{i=1}^{10} \frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{1}{2}\frac{(x_i - \mu)^2}{\sigma^2})

これは標本10個の同時確率密度関数である。しかし、今回標本は実現値として持っているので、不確定な確率的な値ではない。
確定値である。むしろ分かっていないのは、平均 :math:`\mu` 、標準偏差 :math:`\sigma` の2つのパラメータである。
なので、 :math:`x_i` は定数と考え、:math:`\mu` 、 :math:`\sigma` であると宣言し直したものを尤度(likelihood)と
定義し、

.. math:: L(\mu,\sigma) = \prod_{i=1}^{10} \frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{1}{2}\frac{(x_i - \mu)^2}{\sigma^2})

式自体は変わらないが、意味合いが全く異なる。これをグラフにして理解する。
:math:`\mu,\sigma` が不明なので、仮に :math:`\mu = 0, \sigma = 1` だと思って、グラフを書くと、

http://qiita.com/kenmatsu4/items/b28d1b3b3d291d0cc698


線形代数
====================================
