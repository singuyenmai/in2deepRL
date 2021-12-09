 

# Introduction to Reinforcement Learning (RL)

## What is AI?

Industrial revolution was for automation of repeated physical solutions (e.g. transporting things)

Digital revolution comes from the demand of automation of repeated mental solutions (e.g. calculation, information transfer) 

In those revolutions, we need to first find the solutions & how to automate them by ourselves. AI then comes from the demand for machines that can find the solutions by themselves, such that we only need to define and provide our targeted problem and/or goal.

The goal of AI is to help us answer the question "What is the (best) solution?". But to desgin such an AI, we need to answer  "How to find the (best) solution?" or "What is the best way to find the (best) solution", which are very different questions.. 

![image-20211205121745890](/home/singuyen/Study/SCB/DeepRL/DeepMindxUCL_lectures/fig/image-20211205121745890.png)

One definition of AI is "To be able to **learn** to make **decisions** to achieve **goals**". Personal opinions:

- What does goal mean for an AI researcher? What if intelligence can emerge from a process without "goals", for instance, evolution?
- What is learning? Can learning be analogous to adaptive plasticity?

## What is RL?

RL is one approach for machine to find the solution for us

![image-20211205123407318](/home/singuyen/Study/SCB/DeepRL/DeepMindxUCL_lectures/fig/image-20211205123407318.png)

**Goal in RL:** optimise sum of rewards, through repeated interaction

RL is based on the ==**reward hypthesis** - "Any goal can be formalized as the outcome of maximizing a cumulative reward"==

Reasons to learn:

- Find solutions
- Adapt online - dealing with unforeseen circumstances. This is not (just) about generalization - but about continuing to learn efficiently online, during operation (so far, I find this very analogous to adaptive plasticity and the question on trade-offs between high & low plasticity)

RL can provide algorithms for both cases by learning to make decisions from interactions.

## Formalising the RL Problem

How to extend an RL algorithm from a discrete time fomain to continuous time domain?

**Return** = sum of all **future** rewards (it's all about the future, there are no past rewards here, because an action can only change the future, not the past) 

$$ G_t = R_{t+1} + R_{t+2}+... = R_{t+1} + G_{t+1}$$

**Value** = expected cumulative reward from a state $s$. The value depends on the actions the agent takes.

$$ v(s) = \mathbb{E}[G_t | S_t = s] = \mathbb{E} [R_{t+1} + R_{t+2}+... | S_t = s] = \mathbb{E} [R_{t+1} + v(S_{t+1}) | S_t=s] $$

**Goal** = maximizing value, by picking suitable actions

**Long-term reward**: Actions may have long term consequences & Reward may be delayed => Sometimes it is better to sacrifice an immediate reward to gain a more long-term one.

**Policy** = state-action mapping (action selection policy)

**Action values** $q(s, a) = \mathbb{E} [G_{t} | S_t=s, A_t=a]$

RL formalism includes:

- Environment
- Reward signal
- Agent, containing:
  - Agent state
  - Policy
  - Value function estimate?
  - Model?

## Inside the Agent: the Agent state

![image-20211206200439250](/home/singuyen/Study/SCB/DeepRL/DeepMindxUCL_lectures/fig/image-20211206200439250.png)

Policy --> Action --> Environment --> Observation

Agent state = anything the agent carries and could change over time, e.g. memories, learned components. Predictions could also be a part of the agent state.

Environment state (which is different), usually invisible to the agent. Even if it is visible, it may contain lots of irrelevant information. Therefore, agent state could be different from environment state.

**Full observability**: observation = environment state => agent state = environment state.

**History of the agent $H_t$** = full sequence of observations, actions, & rewards. History is used to construct the **agent state $S_{t}$**

### Markov decision processes (MDPs)

A decision process is Markov if 

$p(r, s | S_t, A_t) = p(r, s | H_t, A_t)$

Probability of reward-state at time (t+1) only depends on that at time t.

=> We just need to know the present, not the full history

Typically, the **agent state $S_t$** is some compression of $H_t$

### Partial observability

Observations are not Markovian. If they are used as agent states - parially observable Markov decision process (POMDP)

### Agent state = a function of the history

$S_t = O_t$ (full observability)

$S_{t+1} = u(S_t, A_t, R_{t+1}, O_{t+1})$ where $u$ = state update function

The state should allow good policies & value predictions.

## Inside the Agent: the Policy

Deterministic policy: $A=\pi(S)$

Stochastic policy: $\pi(A|S) = p(A|S)$

## Inside the Agent: Value Estimates

Definition:

$v_\pi(s) = \mathbb{E}[G_t | S_t=s, \pi] = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... | S_t=s, \pi]$

discount factor $\gamma \in [0,1]$ = trades off importance of immediate vs long-term rewards

The value can used to evaluate the desirability of states, then can be used to select between actions & optimize the policy.

### Bellman equation

### Approximating Value Function

Our objective is to approximate the value function, as an appropriate value function can indicate the optimal policy.

## Inside the Agent: Model

A model predicts what the environment will do next, for example: predicting the next state, or the next reward

Stochastic (generative) models: e.g. a model predicting the distribution of the next state given the current state and current action / the expected next state / a plausible next state / randomly give a state that could happen

## The maze example

Optimal policy = shortest path to the goal

The approximated agent's model can be imperfect or the approximated value function can be imperfect, but a goof policy can still be found

## Agent categories

- Value based: approximate value function --> infer policy
- Policy based: approximate policy 
- Actor Critic: approximate both value function (critic) & policy (actor)

- Model free: policy and/or value function, but no explicit dynamic model
- Model based: some agents only have a model so we have to plan in order to extract their policy

## Subproblems of RL 

- Prediction (for a given policy): e.g. learning a value function 
- Control (find the best policy): optimise the future

If we can predict well, then we can control well, because the optimal policy is the one corresponding to the maximum value $\pi^*(s) = argmax_\pi v_\pi(s)$

If we could predict everything, do we need anything else?

- Perhaps we still need to choose which is the policy that we want, perhaps there are some different policies that derive a same value?

## Learning & Planning

- Learning: the agent interacts with the environment to learn about it
- Planning: the agent plan/reason its actions according to its model about the environment. Note that if the model of the environment is learnt, then it can be inaccurate so exhaustively planning according to it could lead to policy that does not work in the true environment. Could be any computational process that helps improve predictions/control without looking at new experiences.

All agent components are functions.

Deep learning is an important tool for RL but it is difficult to combine them (this is still a rich and active research field), partly because RL problems often violate assumptions from superivsed learning (e.g. strong correlation in data points, dynamic policies &/ value functions)