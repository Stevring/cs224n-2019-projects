# Solution for Assignment #2

### 1 Understanding word2vec

 **(a)**  As described in the doc, $\boldsymbol{y}$ is a one-hot vector with a 1 for the true outside word $o$, that means $y_i$ is 1 if and only if $i == o$. so the proof could be below:

$\begin{aligned}
    - \sum_{w\in Vocab}y_w\log(\hat{y}_w) &= - [y_1\log(\hat{y}_1) + \cdots + y_o\log(\hat{y}_o) + \cdots + y_w\log(\hat{y}_w)] \\
    & = - y_o\log(\hat{y}_o) \\
    & = -\log(\hat{y}_o) \\
    & = -\log \mathrm{P}(O = o | C = c)
\end{aligned}$

**(b)** we know this deravatives:
$$
\because J = CE(y, \hat{y}) \\
\hat{y} = softmax(\theta)\\
\therefore \frac{\partial J}{\partial \theta} = (\hat{y} - y)^T
$$

$y$ is a column vector in the above equation. So, we can use chain rules to solve the deravitive:

$$\begin{aligned}
\frac{\partial J}{\partial v_c} &= \frac{\partial J}{\partial \theta} \frac{\partial \theta}{\partial v_c} \\
&= (\hat{y} - y) \frac{\partial U^Tv_c}{\partial v_c} \\
&= U^T(\hat{y} - y)^T
\end{aligned}$$

**(c)**
similar to the equation above.
$$\begin{aligned}
\frac{\partial J}{\partial v_c} &= \frac{\partial J}{\partial \theta} \frac{\partial \theta}{\partial U} \\
&= (\hat{y} - y) \frac{\partial U^Tv_c}{\partial U} \\
&= v_c(\hat{y} - y)^T
\end{aligned}$$

**(d)**
$$
\begin{aligned}
\frac{\partial \,\sigma(x)}{\partial x}=\sigma(x)(1-\sigma(x))
\end{aligned}
$$
**(e)**
$$
\frac{\partial J}{\partial v_c}=-(1-\sigma(u_o^Tv_c))u_o+\sum_{k=1}^K(1-\sigma(u^T_{k}v_c))u_k \\
\frac{\partial J}{\partial u_o}=-(1-\sigma(u^T_ov_c))v_c\\

\frac{\partial J}{\partial u_k}=(1-\sigma(u^T_k v_c))v_c
$$
