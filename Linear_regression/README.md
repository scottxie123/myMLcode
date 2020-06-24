# 线性分类器
设数据集$x_i = (x_{i1},x_{i2},...,x_{ip}) \quad i=1,2,...,N$，$y = (y_1,y_2,...,y_N)$，转换为矩阵表示为：
$$X_{p\times N} = \left(\begin{matrix}
        x_{11} & x_{12} & \cdots &x_{1p} \\\\
        x_{21} & x_{22} & \cdots & x_{2p} \\\\
        \vdots & \vdots & \ddots & \vdots \\\\
        x_{N1} & x_{N2} & \cdots & x_{Np}  \\\\
        \end{matrix} \right)^T_{p\times N}
$$
系数真实值为$w= (w_1,w_2,...,w_p)^T$，则有$y = w^TX+\epsilon$ 其中$\epsilon$为噪声，服从标准正态分布。

最终目标为估计$w\_$，根据最小二乘法$w\_ = 0.5\argmin_w\|y-w^TX\|_2^2$，令$L(w) = 0.5\|y-w^TX\|_2^2$则有：
$$
\frac{\partial L(w)}{\partial w} = (y-w^TX)X^T=0
$$
求得
$$
w = (XX^T)^{-1}Xy^T
$$