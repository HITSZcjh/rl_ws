# Policy Gradients
## 基本知识
- **Policy Gradients**
  ****
  $$
  \theta^*=\argmax_\theta E_{\tau\sim p_\theta(\tau)}[\sum_t r(s_t,a_t)]
  $$
  对于有限时间情况
  $$
  \theta^*=\argmax_\theta E_{\tau\sim p_\theta(\tau)}[\sum_{t=1}^{T} r(s_t,a_t)]
  $$
  对于无限时间的情况，我们关注时间趋于无穷时的状况
  $$
  \theta^*=\argmax_\theta E_{(s,a)\sim p_\theta(s,a)}[r(s,a)]
  $$
  因此评价策略的目标如下，可以使用采样近似
  $$
  J(\theta)=E_{\tau\sim p_\theta(\tau)}[\sum_t r(s_t,a_t)]\approx \frac{1}{N}\sum_i \sum_t r(s_{i,t},a_{i,t})
  $$
  对于上式，我们简化记法，令$r(\tau)=\sum r(s_{t},a_{t})$, 则 $J(\theta)=E_{\tau\sim p_\theta(\tau)}\,r(\tau)$ 。 其梯度表示如下  
  $$
  \begin{aligned}
    \nabla _\theta J(\theta) &= \nabla _\theta [E_{\tau\sim p_\theta(\tau)}\,r(\tau)]\\
    &=\nabla _\theta \int p_\theta(\tau)\,r(\tau)\,d\tau\\
    &=\int \nabla _\theta p_\theta(\tau)\,r(\tau)\,d\tau\\
    &=\int p_\theta(\tau)\, \nabla _\theta\log{p_\theta(\tau)}\,r(\tau) d\tau\\
    &=E_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log{p_\theta(\tau)}\,r(\tau)]
  \end{aligned}
  $$
  对于$p_\theta(\tau)$，可表示如下：  
  $$
  \begin{aligned}
    p_\theta(\tau)&=p(s_1)\prod_{t=1}^T\pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)\\
    \log{p_\theta(\tau)}&=\log{p(s_1)}+\sum_{t=1}^T\log{(\pi_\theta(a_t|s_t))}+\log{p(s_{t+1}|s_t,a_t)}
  \end{aligned}
  $$
  所以对于目标梯度，化简如下：
  $$
  \begin{aligned}
    \nabla _\theta J(\theta) &= E_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log{p_\theta(\tau)}\,r(\tau)]\\
    &=E_{\tau\sim p_\theta(\tau)}[(\sum_{t=1}^T\nabla_\theta \log{(\pi_\theta(a_t|s_t))})(\sum_{t=1}^T r(s_t,a_t))]\\
    &\approx\frac{1}{N}\sum_{i=1}^N[(\sum_{t=1}^T\nabla_\theta \log{(\pi_\theta(a_{i,t}|s_{i,t}))})(\sum_{t=1}^T r(s_{i,t},a_{i,t}))]
  \end{aligned}
  $$
- **Understanding Policy Gradients**  
  ****
  **与极大似然估计比较**  
  policy gradient: $\nabla _\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^N[(\sum_{t=1}^T\nabla_\theta \log{(\pi_\theta(a_{i,t}|s_{i,t}))})(\sum_{t=1}^T r(s_{i,t},a_{i,t}))]$  
    

  maximum likelihood: $\nabla _\theta J_{ML}(\theta)\approx\frac{1}{N}\sum_{i=1}^N[(\sum_{t=1}^T\nabla_\theta \log{(\pi_\theta(a_{i,t}|s_{i,t}))})]$  
   
  可以看到，策略梯度相较极大似然，加入了reward权重。  
  ****
  **Example: Gaussian policies**  
  $$
  \begin{aligned}
    \pi_\theta(a_t|s_t)&\sim N(\mu_{network}(s_t),\Sigma)\\
    \pi_\theta(a_t|s_t)&=\frac{1}{\sqrt{(2\pi)^{k}|\Sigma|}}\mathrm{e}^{-\frac{1}{2}({a_t}-{\mu(s_t)})^\mathrm{T}{\Sigma}^{-1}({a_t}-{\mu(s_t)})}\\
    \log{\pi_\theta(a_t|s_t)}&=-\frac{1}{2}({a_t}-{\mu(s_t)})^\mathrm{T}{\Sigma}^{-1}({a_t}-{\mu(s_t)})+C\\
    \nabla _\theta \log{\pi_\theta(a_t|s_t)}&={\Sigma}^{-1}({a_t}-{\mu(s_t)}) \frac{d\mu}{d\theta}
  \end{aligned}
  $$
- **Reducing Variance：由于采样性，上述方法计算得出的梯度存在高方差，且与r的绝对数值相关，而我们希望其与r的相对值相关，实际无法使用**
  ****
  **方法一：因果性**  
  $$
  \nabla _\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^N[\sum_{t=1}^T(\nabla_\theta \log{(\pi_\theta(a_{i,t}|s_{i,t}))})(\sum_{t'=t}^T r(s_{i,t'},a_{i,t'}))]
  $$ 
  $t'>t$时，在$t'$时刻的策略无法影响$t$时刻的reward。  
  ****
  **方法二：Baselines**  
  对于方差而言，数值越小，方差相对会小一些，故我们可以采用减去baselines的方法。
  $$
  \begin{aligned}
    \nabla_\theta J(\theta)&\approx\frac{1}{N}\sum_{i=1}^N\nabla_\theta \log{p_\theta(\tau)}[r(\tau)-b]\\
    b&=\frac{1}{N}\sum_{i=1}^Nr(\tau)
  \end{aligned}
  $$
  证明该方法为无偏的：  
  $$
  \begin{aligned}
    E[\nabla_\theta \log{p_\theta(\tau)}b]&=\int{p_\theta(\tau)\nabla_\theta \log{p_\theta(\tau)}b\,d\tau}\\
    &=\int\nabla_\theta\,p_\theta(\tau)b\,d\tau\\
    &=b\,\nabla_\theta\int{p_\theta(\tau)b\,d\tau}\\
    &=b\,\nabla_\theta\,1\\
    &=0
  \end{aligned}
  $$
  同时我们可以求的一个最优的$b$使得梯度的方差最小，具体可见PPT：  
  $$
  b=\frac{E[(\nabla_\theta \log{p_\theta(\tau)})^2\,r(\tau)]}{E[(\nabla_\theta \log{p_\theta(\tau)})^2]}
  $$
- **Off-Policy Policy Gradients：有时候使用策略收集轨迹非常低效或者需要消耗大量资源，我们无法采用On-policy learning的方法。**  
  ****
  **importance sampling（重要性采样）**  
  当我们需要从某一确定分布采样求期望但不可行时，我们可以采用重要性采样，从另一可行分布进行采样。  
  $$
  \begin{aligned}
    E_{x\sim p(x)}[f(x)]&=\int p(x)f(x)dx\\
    &=\int q(x)\frac{p(x)}{q(x)}f(x)dx\\
    &=E_{x\sim q(x)}[\frac{p(x)}{q(x)}f(x)]
  \end{aligned}
  $$
  因此对于策略梯度，${\theta'}$为新参数，$\theta$为旧参数，如下：
  $$
  \begin{aligned}

    J({\theta'})&=E_{\tau\sim p_{\theta'}(\tau)}[r(\tau)]\\
    &=E_{\tau\sim p_{\theta}(\tau)}[\frac{p_{\theta'}(\tau)}{p_{\theta}(\tau)}r(\tau)]\\

    \frac{p_{\theta'}(\tau)}{p_{\theta}(\tau)}&=\frac{p(s_1)\prod_{t=1}^T\,\pi_{\theta'}(a_t|s_t)p(s_{t+1}|s_t,a_t)}{p(s_1)\prod_{t=1}^T\,\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t,a_t)}\\
    &=\frac{\prod_{t=1}^T\,\pi_{\theta'}(a_t|s_t)}{\prod_{t=1}^T\,\pi_{\theta}(a_t|s_t)}\\

    \nabla_{\theta'}J({\theta'})&=E_{\tau\sim p_{\theta}(\tau)}[\frac{\nabla_{\theta'}\,p_{\theta'}(\tau)}{p_{\theta}(\tau)}r(\tau)]\\
    &=E_{\tau\sim p_{\theta}(\tau)}[\frac{p_{\theta'}(\tau)}{p_{\theta}(\tau)}\nabla_{\theta'}\log{p_{\theta'}(\tau)}r(\tau)]\\
    &=E_{\tau\sim p_{\theta}(\tau)}[(\frac{\prod_{t=1}^T\,\pi_{\theta'}(a_t|s_t)}{\prod_{t=1}^T\,\pi_{\theta}(a_t|s_t)})(\sum_{t=1}^T\nabla_{\theta'} \log{(\pi_{\theta'}(a_{t}|s_{t}))})(\sum_{t=1}^Tr(s_t,a_t))]
  \end{aligned}
  $$

  考虑因果性，我们可以得到（详见PPT）：
  $$
  \nabla_{\theta'}J({\theta'})=E_{\tau\sim p_{\theta}(\tau)}[\sum_{t=1}^T\nabla_{\theta'} \log{(\pi_{\theta'}(a_{t}|s_{t}))}(\prod_{t'=1}^t\frac{\pi_{\theta'}(a_{t'}|s_{t'})}{\pi_{\theta}(a_{t'}|s_{t'})})(\sum_{t'=t}^Tr(s_{t'},a_{t'}))]
  $$

  但观察式子中的$\prod_{t'=1}^t\frac{\pi_{\theta'}(a_{t'}|s_{t'})}{\pi_{\theta}(a_{t'}|s_{t'})}$项，当时间轴非常长时，此项可能会趋于0或无穷，导致梯度效果很差，具体解决办法在后面部分详述。
- **Implementing Policy Gradients**  
  ****
  由前文可知，策略梯度可以视为加权极大似然，而极大似然可借助交叉熵函数求解。  
  **极大似然估计**（动作与状态均为离散one-hot编码）：
  ```python
  # Given:
  # actions -(N*T) x Da tensor of actions
  # states -(N*T) x Ds tensor of states
  # Build the graph:
  logits = policy.predictions(states) # 在给定状态 states 下的动作概率的 logits（对数概率）
  # 先通过softmax将对数概率转为实际概率，再利用交叉熵函数求的对应actions值下的负对数概率
  negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=actions, logits=logits) 
  loss = tf.reduce_mean(negative_likelihoods)
  gradients = loss.gradients(loss, variables)
  ```
  **策略梯度**：
  ```python
  # Given:
  # actions -(N*T) x Da tensor of actions
  # states -(N*T) x Ds tensor of states
  # q_values – (N*T) x 1 tensor of estimated state-action values
  # Build the graph:
  logits = policy.predictions(states) # This should return (N*T) x Da tensor of action logits
  negative_likelihoods = tf.nn.softmax_cross_entropy_with_logits(labels=actions, logits=logits)
  weighted_negative_likelihoods = tf.multiply(negative_likelihoods, q_values)
  loss = tf.reduce_mean(weighted_negative_likelihoods)
  gradients = loss.gradients(loss, variables)
  ```