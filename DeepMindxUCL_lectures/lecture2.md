# Exploration & Control

Literature: book chapter 2

RL is the science of learning to make decisions, which 

Decisions affect the reward, the agent state, and the environment state. Therefore, learning is active: decisions impact data.

Assumptions during this lecture:

- Environment only has a single state (i.e. no state at all) => actions no longer have consequences on the environment (*nonassociative* setting)
- actions still impact immediate reward

## Exploration vs. Exploitation

- Exploitation: maximise performance based on current knowledge, higher chance of success but only when knowledge is large enough, and sometimes could miss better unexperienced options
- Exploration: increase knowledge, but more risky (easily failed)

The best long-term strategy may involve short-term sacrifices

## Formalising the problem

### The multi-armed bandit

Multiple machines, each with single action and different reward distributions

![image-20211207150000423](/home/singuyen/Study/SCB/DeepRL/DeepMindxUCL_lectures/fig/image-20211207150000423.png)

Note that the poilicy can change over time as we collect more data.

The optimal value $v^*$ = value where action value $q(a)$ maximum

Regret of an action $a$: $\Delta_a = v^* - q(a)$ => the regret of the optimal action is zero.

Our goal is to minimize the total regret $L_t = \sum_{n=1}^{t} \Delta A_n$ over the full learning lifetime

Maximise cumulative reward = minimise total regret.

## Algorithms

Action value estimates (approximates of the true action value) $Q_t(a)$:

- average of sampled rewards of an action $a$ over the total count the action $a$ has been taken up to time point $t$
- take the average (like above) and then update incrementally

### Greedy

Simplest policy: select action with highest action value estimate: $A_t = argmax_a Q_t(a)$

or policy $\pi_t(a) = I(A_t = argmax_a Q_t(a))$ assuming no ties (i.e. different actions that lead to a same highest action value estimate)

Greedy is not very great, it can get stuck on a suboptimal action forever. This is because it does not explore at all, it just exploit its current knowledge

![image-20211207152920507](/home/singuyen/Study/SCB/DeepRL/DeepMindxUCL_lectures/fig/image-20211207152920507.png)

### $\epsilon$-greedy

$\epsilon$ = prob. of selecting a random action. 

Greedy action is selected with a probability of $(1-\epsilon)$, also assuming no ties. => There is some exploration here, with a fixed prob. of $\epsilon$. So new knowledge can come in.

However, because $\epsilon$ is constant, the algorithm keeps exploring all the times, even if the optimal policy has already been found => still, linear expected total regret

### Policy gradients

Learn policies directly

action preferences $H_t(a)$ (not values, just learnable policy parameters) 

policy $\pi(a)$ is then calculated by softmax of action preferences = probability of selecting an action $a$. Policies for all actions sum to 1.

Update policy parameters (action preferences) such that the **expected value of the policy** increases.

$\theta_{t+1} = \theta_t + \alpha \nabla_\theta \mathbb{E} [R_t | \pi_\theta]$

$\theta_t$ = vector of policy parameters at current time $t$. 

$\nabla_\theta \mathbb{E} [R_t | \pi_\theta]$ = gradient of the expected value of the policy with policy parameters vector $\theta$ (gradient ~ partial derivative)

(stochastic) gradient ascent could be sampled with the log-likelihood trick

$\theta_{t+1} = \theta_t + \alpha R_t \nabla_\theta log(\pi_\theta(A_t))$ is stochastic gradient ascent

$A_t$ = a random variable indicating a sampled action (the action that the agent actually takes)

The preference for a sampled / selected action goes up if the reward for that taken action is positive and the preference for other actions would go down. How much they go up / down also depends on the step size / learning rate $\alpha$. For an action that lead to higher reward, then the perference for that action once it is taken would increase to a larger extent / faster. Higher preferences ~ higher chance that action will be selected.

**The problem of gradient method is that it can get stuck at local optimum - sub-optimal policy.**

### Lai & Robbins algorithm

For any algortihm, the regret will always grows at least logarithmically over time.

=> Are there algorithms for which the upper bound is logarithmic as well?

A good algorithm ensures **smalll counts of actions with large regrets** (note that we are assuming of no future consequences of actions)

### Optimism in the face of uncertainty

An action is more probable to be selected when either:

- the estimated action value is high, or
- the uncertainty (variance of the action value distribution) is high (the more uncertainty a value is, the more important to explore that action)

### UCB (Upper confidence bounds)

For each action value, estimate the upper bound $U_t(a)$ such that it is highly probable that $q(a) \le Q_t(a) + U_t(a)$. We also need to estimate this upper bound to be small enough to converge to the true mean action value $q(a)$. The idea is that the more number of times an action were selected ($N_t(a)$) then the smaller its UCB become. But still, can we derive the optimal bound?

An action is selected if either:

- $Q_t(a)$ is large, or
- $U_t(a)$ is large, or both

The difference of UCB from greedy methods is that UCB can stop selecting an action if it appears to have a smaller value than that of other actions.

#### The optimality of UCB - Hoeffding's Inequality

The more number of samples $n$ or the more bound that we add $u$, the less likely that the sample mean + the bound $\le \mu$ - the true mean

**Calculating the UCB**: $U_t(a) = \sqrt{\frac {-\log p} {2 N_t(a)}}$ where $p$ is the **maximal** prob. that the true $q(a)$ exceeds the upper bound. The idea is to reduce this $p$ as time gone by (as we observe more action-rewards), e.g. $p= 1/t$, then $U_t(a) = \sqrt{\frac {\log t} {2 N_t(a)}} = c \sqrt{\frac {\log t} {N_t(a)}}$. This ensure that the agent always keep exploring, but not too much.

$c$ could be $1/\sqrt{2}$ and is a hyper parameter. Higher $c$ = explore more. $c=0$, we go back to the greedy methods, with no exploration.

**Auer theorem: UCB with $c=\sqrt{2}$  achieves logarithmic expected total regret**

### Bayesian approaches

#### Bayesian bandits

Bayesian approach works with model distributions over prob. values $p(q(a) | \theta_t)$. Note that $q(a)$ here is the expected value for a given action (in our problem, the agent does not know its action values, it has to learn to realize those)

The prob. here is intepreted as our belief

$\theta_t$ here are parameters of the belief distributions

Allow us to incoorperate known prior knowledge: $\theta_{t=0}$

Belief distributions are updated over time as the agent acts and receives rewards.

#### Bayesian bandits with UCB

We can estimate upper confidences from the posteriors. 

For example, $U_t(a) = c\sigma_t(a)$ where $\sigma_t(a)$ is the standard deviation of the belief distribution $p_t(q(a))$ at time $t$ (our belief of the true standard deviation of the true value). Then, the action selecting rule is picking one that maximises $Q_t(a) + c\sigma_t(a)$

### Thompson sampling

Use **probablity matching**: the probability of selecting an action $a$ (policy) = the prob. (belief) that $a$ is optimal given the "up-to-date" knowledge

But, prob. matching is optimistic in the face of uncertainty: actions can have higher prob. when either the estimated value is high, or the uncertainty is high

Thompson 1933 - sample-based probability matching:

- For an action $a$, sample $Q_t(a)$ from the belief distribution $p_t(q(a))$
- Do that for all the actions
- Select the action that maximizes the sampling: $A_t = argmax_a Q_t(a)$

=> Thompson sampling provides a way to derive the policy from belief distributions

For Bernoulli bandits (bandits with prior belief as Bernoulli distribution), Thompson sampling achieves logarithmic expected total regret, and therefore, is optimal.

**Note for UCB & Thompson sampling**: For a same problem, the two methods can derive different sequences of actions taken, but eventually still converge to similar policies.

### Planning to explore

The internal information state of the agent $S_t$ can change over time. The transition in the agent state is a Markov process, where each action (taken) $A_t$ causes a transition to  new state $S_{t+1}$ with a prob. $p(S_{t+1} | A_t, S_t)$

=> Via learning, bandits actions affect the future, though it is the future of the internal state of the agent, not that of the external environment.

To estimate the information state space, RL can be used. For example, learn a Bayesian reward distribution (how likely of receiving a certain reward is), then plan into the future about the internal states.

