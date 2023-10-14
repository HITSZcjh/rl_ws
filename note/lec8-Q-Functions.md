# Q-Functions
##  基本知识
- **target networks**  
  ****
  ![](<pictures/2023-09-25 19-50-36 的屏幕截图.png>)
  有上图可知，当我们对Q Function进行训练时，目标$y_i$值也会随着改变，所以实际上并非严谨的剃度下降。  
  为解决上述问题，我们可以使用目标网络的方法:  
  ![](<pictures/2023-09-25 20-16-52 的屏幕截图.png>)
  上述过程会导致$\phi'$网络的延迟长短不一，不够优雅，可以使用下述方法：
  $$
  \phi'=\tau\phi'+(1-\tau)\phi \ ,\ \tau=0.999
  $$
  在每次更新时都更新$\phi'$网络，但加了一个很大的延迟系数。  
  ![](<pictures/2023-09-25 20-25-03 的屏幕截图.png>)
  上图三个过程实际相对独立，调整可产生众多变形。  
- **Improving Q-Learning**
  ****
  **Overestimation in Q-learning**  
  由于$E[\max(X_1,X_2)]\geq\max(E[X_1],E[X_2])$ 且 $y_j=r_j+\gamma \max_{a'_j} Q_{\phi'}(s'_j,a'_j)$，由于$Q_{\phi'}(s'_j,a'_j)$存在误差类似噪声，故会导致其目标值比真实值大，在迭代过程中越来越大。  
  为解决上述问题，我们可以采用不同网络来采集最大的动作和最大的Q值，最大程度减小噪声的相关性。  
  $$
  Q_{\phi_A}(s,a)=r+\gamma Q_{\phi_B}(s',\argmax_{a'}Q_{\phi_A}(s',a'))\\
  Q_{\phi_B}(s,a)=r+\gamma Q_{\phi_A}(s',\argmax_{a'}Q_{\phi_B}(s',a'))
  $$
  而结合上面标准Q-learning，我们以及有两个网络，仅需改变$y=r+\gamma Q_{\phi'}(s',\argmax_{a'}Q_{\phi'}(s',a'))$ 为 $y=r+\gamma Q_{\phi'}(s',\argmax_{a'}Q_{\phi}(s',a'))$
- **Multi-step returns**  
  ****
  $$
  y_j=r_j+\gamma \max_{a'_j} Q_{\phi'}(s'_j,a'_j)
  $$
  由上式可知，当Q值不够准确时，即初始化时Q值较小，其目标值主要取决于前面一项，而前面一项仅为单项回报，显然与全程回报相差较大，故会导致Q值初始化阶段训练效果较差。我们考虑类似梯度下降中采用的多步回报的方法来进行改善该问题。  
  $$
  y_{j,t}=\sum_{t'=t}^{t+N-1}{\gamma^{t-t'}}r_{j,t'}+\gamma^N \max_{a_{j,t+N}} Q_{\phi'}(s_{j,t+N},a_{j,t+N})
  $$
  但显然该方法的缺点在于需要采用最优策略$\pi$收集对应轨迹，属于on-policy的方法。而该问题我们有三种解决办法：一是忽略该问题，仍当作off-policy的方法；二是on-policy收集N步轨迹；三是重要性采样。  
- **Q-Learning with Continuous Actions**  
  ****
  对于连续动作问题，我们主要难点在于获取最大Q值。$y_j=r_j+\gamma \max_{a'_j} Q_{\phi'}(s'_j,a'_j)$  
  对于该问题有以下几个方法：  
  1. **优化**  
    梯度下降；  
    随机采样方法：如$\max_a{Q(s,a)}\approx\max\{Q(s,a_1),...,Q(s,a_N)\}$，详见PPT。
  2. **采用一些特殊形式的Q函数**  
    $$
    Q_\phi(s,a)=-\frac{1}{2}(a-\mu_\phi(s))^TP_\phi(s)(a-\mu_\phi(s))+V_\phi(s)
    $$
  3. **利用网络来估计最大值**  
    该方法类似确定策略的actor-critic框架，训练一个网络使得$\mu_\theta(s)\approx\argmax_aQ_\phi(s,a)$，即采用梯度下降的方法求得 $\theta=\argmax_\theta{Q_\phi(s,\mu_\theta(s))}$
- **Simple practical tips for Q-learning**  
  ****
  ![](<pictures/2023-09-25 21-10-48 的屏幕截图.png>)
  ![](<pictures/2023-09-25 21-10-54 的屏幕截图.png>)