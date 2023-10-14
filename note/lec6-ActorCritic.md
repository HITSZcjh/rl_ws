# Actor-Critic
## 基本知识
- **Improving the policy gradients**  
  ****
  回顾：policy gradients  
  $$
  \begin{aligned}
    \nabla _\theta J(\theta)&\approx\frac{1}{N}\sum_{i=1}^N[\sum_{t=1}^T(\nabla_\theta \log{(\pi_\theta(a_{i,t}|s_{i,t}))})\hat{Q}_{i,t}]\\
  \hat{Q}_{i,t}&=\sum_{t'=t}^T r(s_{i,t'},a_{i,t'})
  \end{aligned}
  $$
  上述$\hat{Q}_{i,t}$是single-sample，将导致梯度存在较大方差，我们希望可以使用期望对其替代。  
  $$
  \begin{aligned}
    \nabla _\theta J(\theta)&\approx\frac{1}{N}\sum_{i=1}^N[\sum_{t=1}^T(\nabla_\theta \log{(\pi_\theta(a_{i,t}|s_{i,t}))})Q^\pi(s_{i,t},a_{i,t})]\\
    Q^\pi(s_t,a_t)&=\sum_{t'=t}^T E_{\pi_\theta}[r(s_{t'},a_{t'})|s_t,a_t]
  \end{aligned}
  $$
  结合baseline：
  $$
  \begin{aligned}
    V^\pi(s_t)&=E_{a_t\sim\pi_\theta(a_t|s_t)}[Q^\pi(s_t,a_t)]\\
    A^\pi(s_t,a_t)&=Q^\pi(s_t,a_t)-V^\pi(s_t)\\
    \nabla _\theta J(\theta)&\approx\frac{1}{N}\sum_{i=1}^N[\sum_{t=1}^T(\nabla_\theta \log{(\pi_\theta(a_{i,t}|s_{i,t}))})A^\pi(s_{i,t},a_{i,t})]
  \end{aligned}
  $$

- **Policy evaluation**  
  ****
  **value function fitting**  
  $$
  \begin{aligned}
    Q^\pi(s_t,a_t)&=\sum_{t'=t}^T E_{\pi_\theta}[r(s_{t'},a_{t'})|s_t,a_t]\\
    V^\pi(s_t)&=E_{a_t\sim\pi_\theta(a_t|s_t)}[Q^\pi(s_t,a_t)]\\
    A^\pi(s_t,a_t)&=Q^\pi(s_t,a_t)-V^\pi(s_t)\\
    \nabla _\theta J(\theta)&\approx\frac{1}{N}\sum_{i=1}^N[\sum_{t=1}^T(\nabla_\theta \log{(\pi_\theta(a_{i,t}|s_{i,t}))})A^\pi(s_{i,t},a_{i,t})]
  \end{aligned}
  $$
  我们该选择拟合哪个函数？  
  $$
  \begin{aligned}
    Q^\pi(s_t,a_t)&=r(s_t,a_t)+\sum_{t'=t+1}^T E_{\pi_\theta}[r(s_{t'},a_{t'})|s_t,a_t]\\
    &\approx r(s_t,a_t)+V^\pi(s_{t+1})\\
    A^\pi(s_t,a_t)&\approx r(s_t,a_t)+V^\pi(s_{t+1})-V^\pi(s_t)
  \end{aligned}
  $$
  上述对$Q^\pi(s_t,a_t)$作了近似，忽略了$s_{t+1}$的其他可能分支，存在bias但大大减小了variance,所以我们可以仅仅拟合$V^\pi(s)$。  
  ****
  **Monte Carlo estimate**  
  Monte Carlo target: $y_{i,t}=V^\pi(s_{i,t})\approx\sum_{t'=t}^T r(s_{i,t'},a_{i,t'})$  
  not as well as:$V^\pi(s_t)\approx\frac{1}{N}\sum_{i=1}^N\sum_{t'=t}^T r(s_{i,t'},a_{i,t'})$，但此方法较难实现，因为我们常常难以返回同一状态$s_t$。  
  ****
  **Bootstrapped estimate**  
  ideal target: $y_{i,t}=V^\pi(s_{i,t})=\sum_{t'=t}^T E_{\pi_\theta}[r(s_{t'},a_{t'})|s_{i,t}]\approx r(s_{i,t},a_{i,t})+V^\pi_\phi(s_{i,t+1})$  
  Bootstrapped target: $y_{i,t}=r(s_{i,t},a_{i,t})+V^\pi_\phi(s_{i,t+1})$
    

- **Actor Critic**  
  ****
  **disconut factor**  
  $$
  \begin{aligned}
    V^\pi(s_t)&=\sum_{t'=t}^{T/\infty}{E_{\pi_\theta}[\gamma^{t'-t}r(s_{t'},a_{t'})]}\\
    y_{i,t}&\approx r(s_{i,t},a_{i,t})+\gamma V^\pi_\phi(s_{i,t+1})\,\,(\gamma\,\,usually\,\,set\,\,to\,\,0.99)
  \end{aligned}
  $$
  Bootstrapped gradients:
  $$
  \nabla _\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_\theta \log{\pi_\theta(a_{i,t}|s_{i,t})(r(s_{i,t},a_{i,t})+\gamma V^\pi_\phi(s_{i,t+1})-V^\pi_\phi(s_{i,t}))}
  $$
  Monte Carlo gradients:
  $$
  \nabla _\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_\theta \log{\pi_\theta(a_{i,t}|s_{i,t})(\sum_{t'=t}^T \gamma^{t'-t} r(s_{i,t'},a_{i,t'}))}
  $$
  ****
  **Actor-critic algorithms**  
  ![](<pictures/2023-09-21 15-28-28 的屏幕截图.png>)  
  对于online框架应该采用并行架构采取多组数据，不能单样本进行拟合。  
  ****
  **state-dependent baselines**  
  Any baselines only dependent on state is unbias.  
  Actor-critic: $\nabla _\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_\theta \log{\pi_\theta(a_{i,t}|s_{i,t})(r(s_{i,t},a_{i,t})+\gamma V^\pi_\phi(s_{i,t+1})-V^\pi_\phi(s_{i,t}))}$, lower variance(due to critic) but not unbiased(if the critic is not perfect).  
  Polict gradient: $\nabla _\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_\theta \log{\pi_\theta(a_{i,t}|s_{i,t})((\sum_{t'=t}^T \gamma^{t'-t} r(s_{i,t'},a_{i,t'}))-b)}$, no bias but higher variance(bacause single-sample estimate).  
  combined:$\nabla _\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_\theta \log{\pi_\theta(a_{i,t}|s_{i,t})((\sum_{t'=t}^T \gamma^{t'-t} r(s_{i,t'},a_{i,t'}))-V^\pi_\phi(s_{i,t}))}$, no bias and lower variance. 
  ****
  **action-dependent baselines**  
  使用$Q^\pi_\phi(s_t,a_t)$作为baselines，这样会引入bias，再使用额外项进行补偿（Q-Prop方法）。
  $$
  \nabla _\theta J(\theta)\approx\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T[\nabla_\theta \log{\pi_\theta(a_{i,t}|s_{i,t})((\sum_{t'=t}^T \gamma^{t'-t} r(s_{i,t'},a_{i,t'}))-Q^\pi_\phi(s_{i,t},a_{i,t}))}]+[\nabla_\theta E_{a\sim\pi_\theta(a_t|s_{i,t})}[Q^\pi_\phi(s_{i,t},a_t)]]
  $$
  如果策略为高斯策略，$Q^\pi_\phi(s_t,a_t)$为二次项，我们可以解析的求的补偿项的大小。
  ****
  **广义优势估计**  
  结合Monte Carlo和Bootstrapped两种方法，在variance和bias之间作出权衡。  
  n步优势函数：
  $$
  \hat{A}^\pi_n(s_t,a_t)=\sum_{t'=t}^{t+n}{\gamma^{t'-t}r(s_{t'},a_{t'})+\gamma^n V^\pi_\phi(s_{t+n})-V^\pi_\phi(s_t)}
  $$
  更进一步，我们可以对其求期望即
  $$
  \hat{A}^\pi_{GAE}(s_t,a_t)=\sum_{n=1}^\infty \omega_n\hat{A}^\pi_n(s_t,a_t)
  $$
  为了减小variance，我们可以取$\omega_n\propto\lambda^{n-1}$，这样上式可以化简为：
  $$
  \begin{aligned}
  &\hat{A}^\pi_{GAE}(s_t,a_t)=\sum_{t'=t}^\infty(\gamma\lambda)^{t'-t}\delta_{t'}\\
  &\delta_{t'}=r(s_{t'},a_{t'})+\gamma V^\pi_\phi(s_{t'+1})-V^{\pi}_\phi(s_{t'})
  \end{aligned}
  $$