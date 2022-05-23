 # Chapter 2: Multi-armed Bandits

RL uses training information to **evaluate** its action taken rather than to **instruct** 

> Purely evaluative feedback indicates how good the action taken was, but not whether it was the best or the worst action possible. Purely instructive feedback, on the other hand, indicates the correct action to take, independently of the action actually taken.

Is this analogous to the difference between using training data to validate the models (here, actions/policies) vs. to deduce the models?

This chapter focuses on balancing exploration and exploitation in simplest ways. The k-armed bandit problem allows us to see clearly the distinctive challenge of exploration-exploitation balance in reinforcement learning.

## 2.1 A k-armed Bandit Problem

k-armed ~ k actions to choose each time. Each action has a corresponding value, termed action-value, that is the expected or mean reward for taking it. The ultimate goal is to maximize the expected TOTAL reward over some time period (i.e. after a certain number of action selections).

**Why do we need to balance b/w exploration and exploitation?**

- Each action has a distribution of corresponding reward. Action value = expected (or mean) reward of an arbitrary action. The best action to take is the one corresponds to the highest value.

- However, the problem is true action values are unknown, and can only be estimated through "experience" (take action -> receive reward -> update value)

- At every time step, there is always an action with highest estimated value - greedy action. Exploitation will take the greedy action, maximizing the expected reward at that time point (short-term). Exploration will take a non-greedy action, which could have lower estimated value, but substantial large variance and this variance contain higher reward values. When such a non-greedy action is taken, its estimated value can be updated and possibly become much higher. Overall, exploration allows the agent to discover true best action without being stucked at sub-optimal greedy one and it is especially beneficial during the early time steps when there is still a large remaining time.

- > Reward is lower in the short run, during exploration, but higher in the long run because after you have discovered the better actions, you can exploit them many times. 

- Whether to explore or exploit depends on: estimated values, uncertainties, and the number of remaining steps.

## 2.2 Action-value Methods

Estimating action values can be done with sample-average method - averaging the total received reward after taking a specific action $a$ over the number of times that action has been taken up to the current time point.

*<u>Exercise 2.1</u>*: $P(A_t = greedy) = (1-\epsilon) + \epsilon*(1/numberOfActions) = (1-0.5)+0.5*1/2 = 0.75$

## 2.3 The 10-armed Testbed

Real-life problems are mostly non-stationary, where the true action values are dynamic.

Greedy is best only in deterministic and stationary scenarios, where true action-values are fixed and without uncertainty. When true action-values have uncertainties, the estimated values from "experienced" rewards can only reflect the true values when the number of trials is large enough. Exploration, or non-greedy, is needed to have sufficient experience for all the actions, in order to estimate their values precisely and not get stuck at sub-optimal action. The same holds for non-stationary scenarios where the true action-values can change over time. Exploration is essential to check if the action-values might have changed and to update the agent's knowledge about its action values.

In $\epsilon$-greedy, $\epsilon$ is the prob. that action will be chosen randomly among all actions. Large $\epsilon$ allows more exploration, would be best in the beginning. But at later on, as the optimal action is identified, it is best to be less explorative. Note that $\epsilon$ is not necessarily a constant, but can also be a time-dependent variable.

*<u>Exercise 2.2: Bandit example</u>*: 

- epsilon case = select randomly among ALL actions irrespective of estimated values
- At all time steps, epsilon case possibly could have occurred.
- At time step 4 and 5, epsilon case definitely has occurred.
  - t=4, Q(a=2) = (1-2)/2 = -0.5 < 0 but action 2 was still taken
  - t=5, Q(a=2) = (1-2+2)/3 = 1/3 > 0, but action 2 was taken, action 3 was taken instead.

*<u>Exercise 2.3</u>*: In the long run, $\epsilon$-greedy methods are expected to eventually converge to the optimal action $a*$, then the method with $\epsilon=0.01$ will perform better than the one with $\epsilon=0.1$  in terms of

- Prob. of selecting the best action = 1-0.01 + 0.01*1/10 = 99.1% (while if $\epsilon=0.1$, this probability is only 91%)
- Cumulative reward could also be higher as the chance of selecting the optimal action with highest reward is also much higher

 ## 2.4 Incremental Implementation

Update rule for action-value estimates. One approach is sample-average, which guarantees that the estimate will change as new & different reward observed but will eventually converge over time.
$$ NewEstimate \leftarrow OldEstimate + StepSize \space [Target - OldEstimate]$$

The step-size parameter is denoted as $\alpha$, or generally $\alpha_t(a)$ in case the parameter can change over time and/or take different values for different actions

## 2.5 Tracking a Nonstationary Problem

In sample-average method: $\alpha_n(a) = \frac {1} {n}$ and the conditions for convergence to the true action values are satisfied. But, convergence is usually very slow or considerable tuning is needed to obtain a satisfactory convergence rate. Therefore, it may not work effectively in nonstationary environments.

When $\alpha$ is a constant in the range of $[0,1]$, the update is called exponential recency-weighted average, where the weght to a reward $R_i$ deays exponentially as time goes backward such that recent rewards receive more weight then long-past rewards. The convergence is not met in this case, meaning that the estimates continually vary in response to the most recently received rewards. This, however, is desirable in nonstationary environments.

$$ Q_{n+1} = Q_n + \alpha[R_n - Q_n] = (1-\alpha)^n Q_1 + \sum_{i=1}^n \alpha(1-\alpha)^{n-i}R_i$$

## 2.6 Optimistic Initial Values

As can be seen, with constant $\alpha$, the action-value estimates are biased by the initial estimate $Q_1$. 

Bias on initial values can be exploited to

- utilize prior knowledge
- encourage exploration in greedy method by setting all initial values to be wildly "optimistic". This technique is called optimistic initial values. Note that exploration only happens in the beginning, because the agent is inherently greedy, so it is still not applicable for nonstationary environments.

<u>*Exercise 2.6: Mysterious Spikes*</u>: At the first 10 time steps, all the 10 possible actions are explored, as the rewards received are all less than the initial values of 5. After that, values of all actions are updated and become a bit closer to the true distributions. A high chance that the optimal action may be revealed to certain greedy agents, and so it is more selected. However, as the estimates are still quite higher from the true values, agents are easily get disappointed again, so they gave up with the optimal action and went explore.

<u>*Exercise 2.7: Unbiased Constant-Step-Size Trick*</u>: Sample averages do not produce initial bias that constant step sizes do. However, sample averages perform poorly in nonstationary problems. One way to avoid the bias of constant step sizes while maintaining their advantages on nonstationary problems is use the step size of 

$\beta_n \doteq \alpha/\overline{\omicron}_n$ where

$\overline{\omicron}_n \doteq \overline{\omicron}_{n-1} + \alpha (1- \overline{\omicron}_{n-1})$, for $n \geq 0$ and $\overline{\omicron}_0 \doteq 0$

- $\overline{\omicron}_1 = 0 + \alpha(1-0) = \alpha \Rightarrow \beta_1 = 1$
- $Q_2 = Q_1 + \beta_1 (R_1 - Q_1) = Q_1 + R_1 - Q_1 = R_1$ is not dependent on initial value $Q_1$
- $\overline{\omicron}_n$ is a geometric series converging to 1 as $n \rightarrow \infty$ so eventually over time, $\beta_n = \alpha$ and we reach the classical constant step size update.

## 2.7 Upper-Confidence-Bound Action Selection

Select the action that either:

- has high value estimate
- has NOT been selected many times, i.e. large uncertainty or variance in the value estimate

Advantage: exploration is encouraged, but as all actions have been explored many times, their value variances are assumed to be already small and only action with high value is selected.

Limitation: still not applicable to nonstationary problems, where the number of times an action has been selected does not linearly correlate to the variance of its value estimates since its true value vary over time. The method also does not work well in large state spaces.

<u>*Exercise 2.8: UCB spikes:*</u> At the beginning, the numbers of times being selected are zeros for all actions, therefore, each action is sequentially selected until 10th time step (as there are 10 possible actions). After that, all actions have already been selected once, so their variance terms are the same, but their value estimates have been updated accordingly to the rewards received and these are different among actions. So there will be one action with highest reward - highest value estimate and it is selected at time step 11th, leading to a spike in reward. Then because other actions now have less number of times being selected, so they are explored. And as exploration happens, the received reward on average would be decreased. The strength of parameter $c$ determines the weight of the value variances. Higher $c$ means the agent will be more sensitive to actions that have been less selected.

## 2.8 Gradient Bandit Algorithms

So far, we use action-value estimates to select action. Another approach is to learning and use a numerical preference for each action $a$, denoted as $H_t(a)$. 

Action is selected based on *relative* action preference of one action over another. Probablities of selecting action are determined according to a **soft-max distribution (Gibbs / Boltzmann distribution)**

$Pr \{A_t =a\} \doteq \frac {e^{H_t(a)}} {\sum_{b=1}^k e^{H_t(b)}} \doteq \pi_t(a)$

$\pi_t(a)$ is the prob. of taking action $a$ at time $t$.

Preferences are updated by stochastic gradient ascent. The prob. of taking an action in the future is increased, if the the current reward > the baseline reward, and vice versa. The non-selected actions move in the opposite direction. The baseline reward is updated by the general rule in section 2.4.

The choice of the baseline does not affect the expected update of the algorithm, but it does affect the variance of the update and thus the rate of convergence.

<u>*Exercise 2.9*</u>:  The soft-max distribution is the logistic sigmoid function in the case of 2 actions

$\pi_t(1) = \frac {e^{H_t(1)}} {e^{H_t(1)} + e^{H_t(2)}} = \frac {1} {1 + e^{H_t(2)} / e^{H_t(1)}} = \frac {1} {1 + e^{H_t(2) - H_t(1)}} = S (H_t(1) - H_t(2))$

$\pi_t(2) = \frac {e^{H_t(2)}} {e^{H_t(2)} + e^{H_t(1)}} = \frac {1} {1 + e^{H_t(1)} / e^{H_t(2)}} = \frac {1} {1 + e^{H_t(1) - H_t(2)}} = S (H_t(2) - H_t(1))$

## 2.9 Associative Search (Contextual Bandits)

Associative search tasks, or contextual bandits - search for the best actions AND associate them with the situations / contexts in which they are best.

Policy = a mapping from situations / contexts to the actions that are best in those actions

Contextual bandits = k-armed bandit + multiple contexts. Policy, therefore, needs to be learned. However, each action taken affects only the immediate reward, NOT the context.

In full RL problem, actions can affect both the immediate reward AND the next situation/context. 

<u>*Exercise 2.10*</u>: 

- If we cannot tell which case we face at any time step, it is best to just stick to one specific action the whole time. The best action to stick to would be decided by looking at the expected rewards of an action across cases: $E[a_1] = 0.5 * 0.1 + 0.5 * 0.9 = 0.5$, $E[a_2] = 0.5*0.2 + 0.5 * 0.8 = 0.5$. So either action is good, but the best reward we can get is jut 0.5
- If we can know the case we are at, then it is certainly best to choose action 2 in case A and action 1 in case B, to total expected reward we receive would be $0.2*0.5 + 0.9*0.5 = 0.55$, which is higher than that when we cannot know the case.

## Summary

Balancing exploitation and exploration by

- Greedy but with optimistic initial values
- Exploring by random action picking with $\epsilon$-greedy 
- Exploring by deterministic action picking formula with UCB
- Exploring by updating action preferences according to stochastic gradient ascent
- Bayesian methods with Gittins-index (Thompson sampling): use prior knowledge of value distribution, update this knowledge over time, actions are selected based on their posterior probability of being the best. Prob. of possible immediate reward for every action may also be computed. The rewards and probabilities of each possible chain of events can be determined, and one need only pick the best. But the tree of possibilities grows extremely rapidly; so it is not feasible to compute exactly those probabilities, but it may be feasible to approximate them. Note that here we are assuming stationary environment.

