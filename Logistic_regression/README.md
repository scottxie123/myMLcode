# Logistic Regression
数据形式：$\{(x_i,y_i)\}_{i=1}^N$, $x_i\in R^p$, $y_i\in\{0,1\}$

Sigmoid 函数：$\sigma(z) = \frac{1}{1+e^{-z}}$

Logistic回归模型定义：$p_1=p(y=1|x) = \frac{1}{1+e^{-w^Tx}}=\psi(x;w)$ , $p_0=p(y=0|x)=1-\psi(x;w)$

Loss函数：

$$
\begin{align}
    \widehat{w} &= argmax_w\log P(Y|X)\\\\
    &=argmax_w\log\prod_{i=1}^Np(y_i|x_i) \\\\
    &=argmax_w\sum_{i=1}^N\log p(y_i|x_i) \\\\
    &=argmax_w\sum_{i=1}^N(y_i\log p_1+(1-y_i)\log p_0)\\\\
    &=argmax_w\sum_{i=1}^N(y_i\log\psi(x;w)+(1-y_i)\log(1-\psi(x;w)))\\\\
    \end{align}
$$