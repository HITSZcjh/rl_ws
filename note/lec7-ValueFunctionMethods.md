# Value Function Methods
##  基本知识
- **Policy Iteration**  
  ****
  回顾:
  $$
    \begin{align}
        V^\pi(s)&=E_{a\sim\pi(a|s)}{[r(s,a)+\gamma E_{s'\sim p(s'|s,a)}[V^\pi(s')]]}\\
        A^\pi(s,a)&= r(s,a)+\gamma E_{s'\sim p(s'|s,a)}[V^\pi(s')]-V^\pi(s)   
    \end{align}
  $$ 
  由于我们仅使用价值函数做决策，一般来说我们采用如下策略：
  $$ 
  \begin{equation}
  \pi'(a_t|s_t)=\left\{
    \begin{aligned}
    &1 \ \ \ if\ a_t=\argmax_{a_t}{A^\pi(s_t,a_t)}=\argmax_{a_t}{Q^\pi(s_t,a_t)} \\
    &0 \ \ \ otherwise
    \end{aligned}
    \right.
  \end{equation}
  $$
  所以上述策略实际为确定性策略，故价值函数的更新公式可作简化：
  $$
  \begin{equation}
    V^\pi(s)=r(s,a)+\gamma E_{s'\sim p(s'|s,a)}[V^\pi(s')]
    \end{equation}
  $$
  所以我们的更新步骤如下：  
  1. 结合(2)(3)式获得策略；  
  2. 结合(4)式更新价值函数。  
  
  结合Q Function，我们对上述过程可以进行简化：  
  由于我们策略如(3)式所示，因此我们有 $V^\pi(s)=\max_a{Q^\pi(s,a)}$，而 $Q^\pi(s,a)=r(s,a)+\gamma E_{s'\sim p(s'|s,a)}[V^\pi(s')]$，故有如下迭代式子：
  $$
    \begin{equation}
    V^\pi(s)=\max_a{(r(s,a)+\gamma E_{s'\sim p(s'|s,a)}[V^\pi(s')])}
    \end{equation}
  $$
- **Fitted Value Iteration and Q-Iteration**  
  ****
  **Fitted Value Iteration**  
  由(4)式我们可以得到以下迭代算法：  
  ![](<pictures/2023-09-25 19-30-59 的屏幕截图.png>)
  但由于我们无法知道转移方程，显然该算法第一步无法实践，我们需要知道不同action的输出。
  ****
  **Q-Iteration**  
  为解决上述问题，我们尝试拟合Q函数：
  $$
  \begin{equation}
  \begin{aligned}
    Q^\pi(s,a)&=r(s,a)+\gamma E_{s'\sim p(s'|s,a)}[V^\pi(s')]\\
    &\approx r(s,a)+\gamma V^\pi(s')\\
    &=r(s,a)+\gamma\max_{a'}{Q(s',a')}
  \end{aligned}
  \end{equation}
  $$
  由(6)式，我们可以通过采样数据的形式避免了需要知道转移方程的问题，我们由此得到以下迭代过程:  
  ![](<pictures/2023-09-25 19-44-48 的屏幕截图.png>)
  由于我们采样仅仅是为了解决转移方程的问题，与策略无关，因此该方法属于off-policy的方法。
  ****
  **Online Q-learning Algorithms**  
  ![](<pictures/2023-09-25 19-50-36 的屏幕截图.png>)
  我们需要对收集数据的策略进行设计，确保一开始的策略具有一定的探索性。  
  $$ 
  \begin{aligned}
  \pi(a_t|s_t)&=\left\{
    \begin{aligned}
    &1-\epsilon \ \ \ if\ a_t=\argmax_{a_t}{Q^\pi(s_t,a_t)} \\
    &\frac{\epsilon}{|A|} \ \ \ otherwise
    \end{aligned}
    \right.\\
  \pi(a_t|s_t)&\sim \exp({Q(s_t,a_t)})
    \end{aligned}
  $$