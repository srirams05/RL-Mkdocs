

## Explain the concept of Reinforcement Learning in a formal way.

Below is a step-by-step formal explanation of the basic concepts in Reinforcement Learning (RL). We will frame RL in terms of a Markov Decision Process (MDP), which is the standard mathematical model used.

1. Markov Decision Process (MDP)  
   A Markov Decision Process is characterized by the tuple  
   

$$
(\mathcal{S},\,\mathcal{A},\,P,\,R,\,\gamma),
$$

  
   where:  
   • $\mathcal{S}$ is the state space. Each element $s \in \mathcal{S}$ represents a possible configuration of the environment.  
   • $\mathcal{A}$ is the action space. Each element $a \in \mathcal{A}$ is a possible action the agent can take.  
   • $P(s' \mid s,a)$ is the transition probability function, which gives the probability of transitioning from state $s$ to state $s'$ when action $a$ is taken. Formally,  
     

$$
P(s' \mid s,a) \;=\;\Pr(S_{t+1} = s' \,\big\vert\, S_t = s,\, A_t = a).
$$

  
   • $R(s,a)$ is the reward function, which specifies the reward the agent receives for taking action $a$ in state $s$. Sometimes it is also written as the expected immediate reward $R: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$.  
   • $\gamma \in [0,1]$ is the discount factor, which balances the importance of immediate rewards versus future rewards. A smaller $\gamma$ emphasizes present rewards, while a value near $1$ gives more weight to future rewards.

2. Agent-Environment Interaction  
   In Reinforcement Learning, an agent interacts with an environment (modeled as an MDP) over a sequence of discrete time steps $t = 0,1,2,\dots$. The agent observes the state $S_t$, takes an action $A_t$, receives a reward $R_{t+1}$, and the environment transitions to a new state $S_{t+1}$. The agent’s goal is to learn a strategy (policy) to choose actions that maximize cumulative reward.

3. Policy ($\pi$)  
   A policy $\pi$ is a mapping from states to probabilities of selecting each available action. Formally,  
   

$$
\pi(a \mid s) \;=\;\Pr(A_t = a \,\big\vert\,S_t = s).
$$

  
   A (deterministic) policy can also be written as a function $ \pi : \mathcal{S} \to \mathcal{A}$ that picks a specific action in each state.

4. Return (Cumulative Discounted Reward)  
   The return $G_t$ from a time step $t$ is defined as the total discounted reward from time $t$ onward. That is,  
   

$$
G_t \;=\; R_{t+1} \;+\;\gamma\,R_{t+2} \;+\;\gamma^2\,R_{t+3} \;+\;\ldots 
   \;=\; \sum_{k=0}^{\infty} \gamma^{\,k}\, R_{t+k+1}.
$$

5. Value Functions  
   Value functions quantify how “good” it is to be in a certain state or to take a certain action in a given state. Two common value functions are:  

   a) State-Value Function  
   The value of a state $s$ under a policy $\pi$, denoted $v_\pi(s)$, is the expected return when starting in state $s$ and following $\pi$ thereafter. Formally,  
   

$$
v_\pi(s) \;=\; \mathbb{E}\bigl[G_t \,\big\vert\, S_t = s,\;\pi \bigr].
$$

  

   b) Action-Value Function  
   The action-value function $q_\pi(s,a)$ is the expected return when starting from state $s$, taking action $a$, and thereafter following policy $\pi$. Formally,  
   

$$
q_\pi(s,a) \;=\; 
   \mathbb{E}\bigl[G_t \,\big\vert\, S_t = s,\;A_t = a,\;\pi \bigr].
$$

6. Optimal Value Functions and Optimal Policy  
   The goal in reinforcement learning is typically to find an optimal policy $\pi^*$ that maximizes the expected return from every state. We define the optimal state-value function $v_*(s)$ and the optimal action-value function $q_*(s,a)$ by  
   

$$
v_*(s) \;=\; \max_{\pi} \;v_\pi(s),
   \quad
   q_*(s,a) \;=\; \max_{\pi} \;q_\pi(s,a).
$$

  
   An optimal policy $\pi^*$ is any policy that achieves these optimal values.

7. Bellman Optimality Equations  
   The optimal value functions satisfy the Bellman optimality equations:  
   

$$
v_*(s) \;=\; \max_{a \in \mathcal{A}} \Bigl[\,R(s,a)\;+\;\gamma\sum_{s'}P(s' \mid s,a)\,v_*(s')\Bigr],
$$

  
   

$$
q_*(s,a) \;=\; R(s,a)\;+\;\gamma\sum_{s'} P(s' \mid s,a)\,\max_{a'} q_*(s',a').
$$

  
   These equations express a recursive relationship between the value of a state (or state-action pair) and the values of subsequent states (state-action pairs).

8. Learning in Reinforcement Learning  
   The crux of Reinforcement Learning is that the agent does not have full knowledge of the MDP (i.e., it does not initially know $P$ or $R$). The agent must estimate the optimal policy by interacting with the environment:  
   • It observes states and rewards.  
   • It updates estimates (e.g., of $v_*(s)$ or $q_*(s,a)$).  
   • It adjusts its policy to choose better actions in the future.  

   Popular algorithms include:  
   • Temporal-Difference Methods (e.g., Q-Learning, SARSA).  
   • Monte Carlo Methods.  
   • Policy Gradient Methods.  

In summary, Reinforcement Learning is about an agent learning to map states of an environment to actions in order to maximize the cumulative reward. Mathematically, RL methods are designed to solve or approximate solutions to the MDP’s optimal value functions—and thus find an optimal or near-optimal policy—when the agent must learn about the environment solely through experience.

## Reward function $R(s,a)$ dependence

***It seems to me that the reward function $R(s,a)$ should also depend upon $s’$? For a given $(s,a)$, the agent could transition to multiple $s’$ and therefore it suggests that $R$ should also depend on $s’$. Clarify this.***

**Answer**

Below is a step-by-step clarification of why, in many MDP or Reinforcement Learning formulations, one sometimes sees the reward function written as $R(s,a)$ (i.e., depending only on $(s,a)$) and other times as $R(s,a,s')$ (i.e., also including the next state):

1. Different but Equivalent MDP Definitions

In the most general Markov Decision Process (MDP) definition, the one-step reward after taking action $a$ in state $s$ and ending up in state $s'$ can indeed depend on all three: 

$$
R(s,a,s') : \mathcal{S} \times \mathcal{A} \times \mathcal{S} \;\rightarrow\; \mathbb{R}.
$$

  
This is sometimes written as $R_t = R(S_t, A_t, S_{t+1})$.

However, many treatments simplify notation by writing

$$
R(s,a) \quad\text{or}\quad R(s) \quad\text{or}\quad R(a)
$$

depending on which elements really matter for the reward in the domain under consideration.

2. Why Using R(s,a) Can Be Enough

In some problems, the one-step reward is essentially fixed once you pick $(s,a)$, regardless of the next state $s'$. More precisely, one might define:

$$
R(s,a) \;=\; \mathbb{E}\bigl[\,R_t \,\big\vert\,S_t=s,\,A_t=a\bigr],
$$

meaning that this function gives the expected immediate reward upon taking action $a$ in state $s$. If the environment is stochastic (i.e., multiple possible $s'$), then

$$
R(s,a) \;=\;\sum_{s'} P(s' \mid s,a)\,\tilde{r}(s,a,s'),
$$

where $\tilde{r}$ is some underlying reward for transitioning to $s'$. Thus, even though the actual reward might depend on $s'$, one can bundle the expectation over $s'$ into $R(s,a)$. This keeps the notation more compact while still retaining the necessary information.

3. When R(s,a,s') is Used Explicitly

If you want to specify exactly how the reward depends on the next state $s'$, or if your problem inherently requires that detail (for instance, certain transitions might yield different bonuses or penalties), you can define

$$
R(s,a,s').
$$

Then, for each transition $(s,a)\to s'$, you would get a specific reward $R(s,a,s')$ rather than an expected value. This form is more explicit but can be more cumbersome in some derivations.  

4. Equivalence in Theory

Mathematically, the two formulations
(a) $R(s,a)\,=\,\mathbb{E}[R(s,a,s')\,\vert\,s,a]$
and 
(b) $R(s,a,s')$
are equivalent if you interpret $R(s,a)$ as that conditional expectation over the randomized transition to $s'$. 

Therefore, one formulation or the other does not fundamentally change the problem; it just changes how explicitly you write the dependence on $s'$.

5. Example: Chess

Taking chess as an example:  
• One might write $R(s,a) = 0$ for all intermediate moves (since no immediate reward is given until checkmate). At the end of the game, you get +1, -1, or 0.  
• If you wanted an incremental reward for capturing certain pieces, you might define $R(s,a,s')$ that encodes a bonus whenever a capture move transitions you to a state $s'$ with fewer opponent pieces. Alternatively, you could incorporate that into an expected sense, but it becomes somewhat clearer to define it as a function of the specific transition.

6. Summary

• It is correct that there can be multiple next states $s'$ for a given $(s,a)$.  
• Hence, in the most general MDP definition, we often see $R(s,a,s')$.  
• Many RL texts and simpler MDP formulations use $R(s,a)$ to represent the expected immediate reward (averaging over the distribution of next states $s'$), making notation a bit simpler for analysis.  

In short: yes, the underlying reward can depend on $s'$. One either writes $R(s,a,s')$ explicitly or uses the shorter $R(s,a)$ to denote the expectation over $s'$. Both are valid formulations of the same concept.

## The policy $\pi(\cdot)$ 

***The policy $\pi(a \vert s) = \text{Pr}(A_t = a \vert S_t = s)$ seems to be defined for each state $a$. Once a policy is executed at $a$, then we are in a new state, say $s’$. Is the same policy $\pi$ defined for $s’$ as well? In other words, is $\pi(\cdot)$ defined for the entire state space?***

**Answer**

Below is a step-by-step explanation clarifying whether the same policy $\pi$ is defined for every state in the state space:

1. What a Policy Is  
   • In Reinforcement Learning, a policy $\pi$ is defined as a mapping from states to a probability distribution over actions. Formally:  
     

$$
\pi:\;\mathcal{S}\;\to\;\Delta(\mathcal{A}),
$$

  
where $\Delta(\mathcal{A})$ is the set of all probability distributions over the action set $\mathcal{A}$.  

   • Equivalently, we often write $\pi(a \mid s)$ to denote the probability of choosing action $a$ in state $s$.

2. Applying $\pi$ in an MDP Scenario  
   • Consider the Markov Decision Process (MDP) with state space $\mathcal{S}$. At any step $t$, the agent observes the current state $S_t\in \mathcal{S}$.  
   • The policy $\pi$ is used to select an action $A_t$. Even if $\pi$ is stochastic, $\pi(a \mid s)$ tells us how likely each action $a$ is, given $s$.

3. Transition to the Next State  
   • After executing the chosen action $A_t$ in state $S_t$, the environment transitions to a new state $S_{t+1}=s'$.  
   • Now, at state $s'$, the same policy $\pi$ applies; namely, to choose the next action $A_{t+1}$, we again use $\pi(a \mid s')$.  
   • Thus, one and the same $\pi$ determines actions for every state the agent might encounter.

4. $\pi$ Is Defined for the Entire State Space  
   • By definition, a (stationary) reinforcement-learning policy gives a decision rule for every possible state in $\mathcal{S}$.  
   • That is, for each state $s\in\mathcal{S}$, $\pi$ prescribes how the agent should act.  
   • So, yes—$\pi$ is indeed “the same policy” for all states, in the sense that there is a single function $\pi(\cdot)$ that applies throughout the state space.

5. Illustration with Deterministic Policies  
   • If $\pi$ is deterministic, then for every state $s$, $\pi(s)$ is exactly one action (rather than a distribution).  
   • In either case—deterministic or stochastic—$\pi$ unequivocally answers: “What do we do if we end up in state $s$?”

6. Policy Improvement or Changes Over Time  
   • In some learning algorithms, the policy $\pi$ may be updated as the agent gains experience (for instance, by gradient steps or by policy iteration).  
   • However, at any particular time during training (or after training), the agent’s “current” policy is still a single function defined over all states.

7. Conclusion  
   • Yes, once the agent transitions to the new state $s'$, the same policy $\pi$ is again used to pick the next action.  
   • Formally, a policy is defined over the entire state space $\mathcal{S}$, ensuring that no matter which state is encountered, the agent knows how it should proceed.

## Trajectory

***Suppose, the agent is at state $s_t$ at time $t$ under a policy $\pi$. Per this policy the action $a_t$ is applied and the agent moves to state $s_{t+1}$ getting the reward $R_{t+1}$. In this way, the agent traces out a trajectory in state space from which the Return $G_t$ for this trajectory can be calculated. Is my understanding correct?***

**Answer**

 Below is a step-by-step confirmation and elaboration on your understanding, with all mathematical symbols in LaTeX:

   1. Agent in State ${s_t}$ at Time $t$  
   At any discrete time step $t$, the agent observes the current state $s_t$.

   2. Action $a_t$ per Policy $\pi$  
   According to the agent’s policy $\pi$, the agent selects and executes an action $a_t$.  
   • If $\pi$ is stochastic, then  
     
$$
a_t \sim \pi(\cdot \,\vert\, s_t),
$$

   meaning $a_t$ is sampled from the probability distribution $\pi(a\,\vert\,s_t)$.  
   • If $\pi$ is deterministic, then     

$$
a_t \;=\;\pi(s_t).
$$

   3. Transition and Reward  
   After the agent executes action $a_t$ in state $s_t$, the environment transitions to a new state $s_{t+1}$ and provides a (possibly stochastic) reward $R_{t+1}$. Formally,  
   

$$
s_{t+1} \;\sim\; P(\,\cdot \,\vert\, s_t,\, a_t),
$$

  
   

$$
R_{t+1} \;=\; R\bigl(s_t,\,a_t,\,s_{t+1}\bigr)
   \quad
   (\text{or an expected form }R(s,a)\text{, depending on the formulation}).
$$

   4. Trajectory (Episode)  
   By repeating this process over time, one obtains a trajectory (or sequence) of states, actions, and rewards:  
   

$$
\bigl(s_0,\,a_0,\,R_1,\,s_1,\,a_1,\,R_2,\,s_2,\,a_2,\,R_3,\dots\bigr).
$$

  
   Such a sequence typically continues until a terminal condition (end of episode) is reached or indefinitely in an ongoing task.

   5. Return $G_t$  
   The return $G_t$ from time $t$ is the total discounted reward obtained by following policy $\pi$ from time $t$ onward:  
   

$$
G_t \;=\; R_{t+1} \;+\;\gamma\,R_{t+2} \;+\;\gamma^2\,R_{t+3} \;+\;\dots 
   \;=\; \sum_{k=0}^{\infty} \gamma^k R_{t+k+1},
$$

  
   where $0 \,\le\, \gamma \,\le\, 1$ is the discount factor that balances immediate and future rewards.

   6. Conclusion  
   Hence, your understanding is correct: by following the policy $\pi$, the agent traces out a trajectory $ (s_t, a_t, R_{t+1}, s_{t+1}, \dots) $ in the state space. From this trajectory, one can calculate the return $G_t$. This process—collecting states, actions, and rewards—is at the heart of Reinforcement Learning, where the agent’s goal is to maximize the expected return.  

## Transition probability and Policy Probability

   In the MDP formulation, we can regard $S_t \in \mathcal{S}$ as a random variable. Similarly, $A_t \in \mathcal{A}$ at time $t$. We have 2 probabilistic definitions:

1) State Transition:  $\text{Pr}(S_{t+1} = s ^{\prime} \vert S_t = s, A_t = a)$. This gives the probability of a particular value of the state $s^{\prime}$ at time $t+1$ given specific values of the state and action at time $t$. Therefore we have:

$$
\sum_{s^{\prime} \in \mathcal{S}} \text{Pr}(S_{t+1} = s ^{\prime} \vert S_t = s, A_t = a) = 1
$$

2) Policy function: $\pi(a \vert s) = \text{Pr}(A_t = a \vert S_t = s)$. Therefore we have:

$$
\sum_{a \in \mathcal{A}}  \text{Pr}(A_t = a \vert S_t = s) =1
$$

Is this correct?

Below is a step-by-step confirmation and explanation:

1) State Transition Probability  
• In a Markov Decision Process (MDP), the transition function  
  

$$
\Pr\bigl(S_{t+1} = s' \,\big\vert\, S_t = s,\;A_t = a\bigr)
$$

  
  is a probability distribution over the next state $s'$ given the current state $s$ and action $a$.  
• Because it is a probability distribution over $s'\in\mathcal{S}$, it must sum to 1:  
  

$$
\sum_{s' \in \mathcal{S}} \Pr\bigl(S_{t+1} = s' \,\big\vert\, S_t = s,\;A_t = a\bigr) \;=\; 1.
$$

2) Policy Function  
• A policy $\pi$ for a stationary MDP is given by  
  

$$
\pi(a \mid s) \;=\; \Pr\bigl(A_t = a \,\big\vert\, S_t = s\bigr).
$$

  
• This is a probability distribution over actions $a\in\mathcal{A}$, for each given state $s$. Hence, for each $s$,  
  

$$
\sum_{a \,\in\,\mathcal{A}} \pi(a \mid s) \;=\; 1.
$$

3) Conclusion  
Yes, both conditions you stated are correct:  
• The transition probabilities over all possible next states sum to 1.  
• The policy probabilities over all possible actions in a given state also sum to 1. 

## State value function

We have shown before that for a trajectory that starts at state $s_t$ under the policy $\pi$,  the return $G_t$ from time $t$ is the total discounted reward obtained by following policy $\pi$ from time $t$ onward:  
   

$$
G_t \;=\; R_{t+1} \;+\;\gamma\,R_{t+2} \;+\;\gamma^2\,R_{t+3} \;+\;\dots 
   \;=\; \sum_{k=0}^{\infty} \gamma^k R_{t+k+1},
$$

  
   where $0 \,\le\, \gamma \,\le\, 1$ is the discount factor. Since $S_t, A_t$ are random variables, so is $R_t$ and therefore $G_t$. Another way to say it that there are many trajectories that emanate from$s_t$ under $\pi$ and we can define an expected value of $G_t$ to be the state value function $v_{\pi}(s)$ where $S_t = s$.

$$
v_{\pi} = \mathbb{E} (G_t \vert S_t =s )
$$

Does this definition of the state value make sense and if so, the expected value implies a conditional probability distribution. How is this known?

**Answer** Below is a step-by-step explanation:

1) The Definition of the State-Value Function  
The state-value function under a policy $\pi$, denoted $v_{\pi}(s)$, is defined as  

$$
v_{\pi}(s) \;=\; \mathbb{E}\bigl[G_t \,\big\vert\, S_t = s \bigr],
$$

  
where  

$$
G_t \;=\;\sum_{k=0}^{\infty} \gamma^k\,R_{t+k+1}.
$$

  
In words, $v_{\pi}(s)$ is the expected return (sum of discounted future rewards) when starting from state $s$ at time $t$ and subsequently following policy $\pi$.

2) Why Is $G_t$ Random?  
• The trajectory that unfolds from time $t$ onward depends on:  
  – The stochastic policy $\pi(a \mid s)$, which determines how likely each action $a$ is in state $s$.  
  – The transition probabilities $P(s' \mid s,a)$, which govern how you move to the next state $s'$.  
  – The potentially stochastic reward $R_{t+1}$ itself.  
• Therefore, for a fixed $s$, there are many possible future trajectories (sequences of states, actions, and rewards). Each trajectory has a different realization of $G_t$. Hence $G_t$ is a random variable.

3) Taking the Expectation (Conditional Probability Distribution)  
Because there are multiple possible ways the future might evolve from $s$, the definition  

$$
\mathbb{E}\bigl[G_t \,\big\vert\, S_t = s\bigr]
$$

  
reflects an expectation over all those possible trajectories. Formally, this expectation is taken with respect to the conditional probability distribution of all future states, actions, and rewards given that $S_t = s$ and that the agent follows policy $\pi$. Symbolically, we can write  

$$
\mathbb{E}\bigl[G_t \,\big\vert\, S_t = s\bigr]
\;=\;
\sum_{\text{all possible future trajectories}}
\Pr\bigl(\text{trajectory}\,\big\vert\,S_t=s,\;\pi\bigr)
\;\times\;
G_t(\text{trajectory}),
$$

  
where $\Pr(\text{trajectory}\,\vert\,S_t=s,\;\pi)$ is constructed from the chain of probabilities defined by $\pi(a\mid s)$ and $P(s'\mid s,a)$.

4) Where the Conditional Distribution Comes From  
• The agent’s policy $\pi(a\mid s)$ gives a distribution over actions $a$ for every state $s$.  
• The environment’s transition function $P(s'\mid s,a)$ gives a distribution over next states $s'$ for each pair $(s,a)$.  
• Combining these, one obtains the probability tree of all future trajectories (including rewards).  
• In RL notation, we simply write $\mathbb{E}[\,G_t\mid S_t=s]$ as a shorthand. Under the hood, it involves the full conditional distribution derived from $\pi$ and $P$.

5) Conclusion  
Yes, this definition of the state-value function  

$$
v_{\pi}(s) \;=\; \mathbb{E}\bigl[G_t \,\big\vert\, S_t=s\bigr]
$$

  
makes sense: we take an expectation of the random return $G_t$ over all possible ways the future can evolve from state $s$, given that the agent acts according to $\pi$. That expectation is indeed a conditional probability distribution (conditioned on $S_t = s$), implicitly governed by $\pi$ and the transition probabilities of the MDP.

## Probability tree for $v_{\pi}(s)$

Symbolically, $v_{\pi}(s)$ is given by

$$
\mathbb{E}\bigl[G_t \,\big\vert\, S_t = s\bigr]
\;=\;
\sum_{\text{all possible future trajectories}}
\Pr\bigl(\text{trajectory}\,\big\vert\,S_t=s,\;\pi\bigr)
\;\times\;
G_t(\text{trajectory}),
$$

to illustrate how this conditional probability distribution is derived from $\pi$ and $P$, use the time instants $t, t+1, t+2$ and work out the probability tree.

**Answer** Below is a step-by-step illustration of how the probability of a particular future trajectory from time $t$ through $t+2$ (and onward) is derived from the policy $\pi$ and the transition probabilities $P$. We will explicitly write out the probability of the first couple of steps in the “probability tree” before it continues further.

--------------------------------------------------------------------

1 Starting at Time t, State s
   
-------------------------------------------------------------------

We begin with:

$$
S_t = s.
$$

Because we are conditioning on $S_t = s$, the probability of being at $s$ at time $t$ is defined to be $1$ in that conditional scenario.

--------------------------------------------------------------------

2 Choosing Action Aₜ\

-------------------------------------------------------------------

Under policy $\pi$, the agent chooses action $A_t = a \in \mathcal{A}$ with probability

$$
\pi(a \,\vert\, s).
$$

Hence,

• Probability of picking action $a$ at time $t$:

$$
\Pr\bigl(A_t = a \,\big\vert\, S_t = s,\;\pi\bigr)
\;=\;
\pi(a \,\vert\, s).
$$

---------------------------------------------------------------------

3 Transition to Time t+1
   
--------------------------------------------------------------------

After choosing $A_t = a$, the environment transitions to a new state $S_{t+1} = s' \in \mathcal{S}$ with probability

$$
P(s' \,\vert\, s,\, a).
$$

Combined with the previous step, the joint probability of picking action $a$ and ending up in state $s'$ is:

$$
\Pr\bigl(A_t = a,\,S_{t+1} = s' \,\big\vert\, S_t = s,\,\pi\bigr)
\;=\;
\pi(a \,\vert\, s)\;\times\;P(s' \,\vert\, s,\,a).
$$

----------------------------------------------------------------------

4 Choosing Action A₍t+1₎ in State s'

----------------------------------------------------------------------

Once in state $s'$ at time $t+1$, the agent chooses action $A_{t+1} = a'$ with probability

$$
\pi(a' \,\vert\, s').
$$

Hence,

• Probability of picking action $a'$ in state $s'$ at time $t+1$:

$$
\Pr\bigl(A_{t+1} = a' \,\big\vert\, S_{t+1} = s',\,\pi\bigr)
\;=\;
\pi(a' \,\vert\, s').
$$

----------------------------------------------------------------------

5 Transition to Time t+2
   
----------------------------------------------------------------------

After action $A_{t+1} = a'$, the environment transitions to $S_{t+2} = s''$ with probability

$$
P(s'' \,\vert\, s',\, a').
$$

Putting it together up to time $t+2$:

$$
\Pr\Bigl(A_t = a,\;S_{t+1} = s',\;A_{t+1} = a',\;S_{t+2} = s'' \,\Big\vert\, S_t = s,\;\pi\Bigr)
\;=\;
\pi(a \,\vert\, s)
\;\times\;
P\bigl(s' \,\vert\, s,\,a\bigr)
\;\times\;
\pi(a' \,\vert\, s')
\;\times\;
P\bigl(s'' \,\vert\, s',\,a'\bigr).
$$

------------------------------------------------------------------------

6 Extending Beyond t+2
   
------------------------------------------------------------------------

By continuing this process for $t+3, t+4, \dots$, one builds the full probability tree of subsequent states and actions:

• At each step, the agent’s policy $\pi(\cdot \,\vert\, s_j)$ supplies the probabilities of each possible action given the current state $s_j$.  
• The environment’s transition dynamics $P(s_{j+1} \,\vert\, s_j,\, a_j)$ supply the probabilities of next states.  

Therefore, any full future trajectory (from time $t$ to the end of the episode or indefinitely in an ongoing task) has a probability obtained by multiplying together all these policy and transition probabilities.

------------------------------------------------------------------------

7 Putting It All Into the Expectation for v₍π₎(s)

------------------------------------------------------------------------

When we write

$$
v_{\pi}(s)
\;=\;
\mathbb{E}\bigl[\,G_t \;\big\vert\; S_t = s\bigr]
\;=\;
\sum_{\substack{\text{all possible future} \\ \text{trajectories}}}
\Pr\bigl(\text{trajectory}\,\big\vert\,S_t = s,\;\pi\bigr)\,\times\,G_t(\text{trajectory}),
$$

we are summing (or integrating) over all possible sequences of the form

$$
\bigl(A_t,\,S_{t+1},\,A_{t+1},\,S_{t+2},\,A_{t+2},\,S_{t+3},\dots\bigr)
$$

and multiplying each trajectory’s return $G_t(\text{trajectory})$ by the probability of that trajectory, which is computed via the products of $\pi(\cdot)$ and $P(\cdot)$ factors illustrated above.

----------------------------------------------------------------------

8 Conclusion

----------------------------------------------------------------------

This small probability tree up to $t+2$ demonstrates how each branch’s probability is the product of:  
1) policy probabilities $\pi(a \,\vert\, s)$, and  
2) transition probabilities $P(s' \,\vert\, s,a)$.  

Continuing the same pattern through subsequent time steps yields the full conditional probability distribution of all possible future trajectories, thereby defining  
$\Pr(\text{trajectory}\,\vert\,S_t=s,\pi)$.  

Thus, the expectation in  
$\mathbb{E}[\,G_t \,\vert\,S_t=s]$  
is taken over this entire (potentially infinite) probability tree.

## Bellman Equation

Given a policy $\pi$, what is the Bellman equation and how is it used to calculate $v_{\pi}(s)$ for all $s \in \mathcal{S}$?
How is it derived?

**Answer** Below is a step-by-step explanation of the Bellman equation for a given policy $\pi$, how it is derived, and how it is used to calculate the value function $v_{\pi}(s)$ for all states $s\in\mathcal{S}$.

───────────────────────────────
1) Definition of the State-Value Function
   
─────────────────────────

Recall that under a policy $\pi$, the state-value function $v_{\pi}(s)$ is  

$$
v_{\pi}(s) 
\;=\;
\mathbb{E}\bigl[G_t \;\big\vert\; S_t = s\bigr]
\;=\;
\mathbb{E}\Bigl[\,
\sum_{k=0}^{\infty}
\gamma^k\,R_{t+k+1}
\;\Big\vert\;
S_t = s
\Bigr].
$$

Here:
• $G_t$ is the (random) return from time $t$ onward.  
• $R_{t+k+1}$ is the reward at time $t+k+1$.  
• $\gamma \in [0,\,1]$ is the discount factor.  
• The expectation $\mathbb{E}[\cdot]$ is taken over all possible future trajectories following policy $\pi$.  

────────────────────────────

2 One-Step Lookahead 

────────────────────────────

Consider the very next step from state $s$ under policy $\pi$. The agent will choose an action $A_t = a$ with probability $\pi(a \,\vert\, s)$. The environment will then transition to a next state $S_{t+1} = s'$ with probability $P(s' \,\vert\, s,\,a)$, and the agent will receive a reward $R_{t+1} = R(s,a,s')$ (or its expected form $R(s,a)$ depending on the problem).  

After that step, the agent continues from state $s'$, and the expected value of the return from $s'$ onward is $v_{\pi}(s')$.  

Thus the return $G_t$, when starting in state $s$ and taking action $a$ that leads to state $s'$, can be decomposed as:  

$$
G_t
\;=\;
R_{t+1} 
\;+\;
\gamma\,G_{t+1}.
$$

But the expected value of $G_{t+1}$, given $S_{t+1} = s'$ and following $\pi$ thereafter, is $v_{\pi}(s')$.

─────────────────────────────

3 The Bellman Equation for v₍π₎(s)
   
────────────────────────────

Putting it all together, we take the expectation with respect to actions (as given by $\pi$) and next states (as given by $P$). We get the Bellman expectation equation:

$$
v_{\pi}(s)
\;=\;
\sum_{a \in \mathcal{A}}
\pi(a \,\vert\, s)
\sum_{s' \in \mathcal{S}}
P(s' \,\vert\, s,\, a)
\bigl[
R(s,a,s')
\;+\;
\gamma\,v_{\pi}(s')
\bigr].
$$

Sometimes you will see it written in a shorter form (if the immediate reward does not explicitly depend on $s'$):

$$
v_{\pi}(s)
\;=\;
\sum_{a \,\in\, \mathcal{A}}
\pi(a \,\vert\, s)
\sum_{s' \,\in\, \mathcal{S}}
P(s' \,\vert\, s,a)
\bigl[
R(s,a)
\;+\;
\gamma\,v_{\pi}(s')
\bigr].
$$

This Bellman equation says:
1. From state $s$, choose an action $a$ according to $\pi$.  
2. Possibly transition to state $s'$ with probability $P(s'\mid s,a)$.  
3. Accumulate the immediate reward $R(s,a,s')$.  
4. Then continue and accumulate discounted future rewards, whose expected value is $\gamma\,v_{\pi}(s')$.

───────────────────────

4 Using the Bellman Equation to Compute v₍π₎(s)
   
───────────────────────────

• The set of equations 
  

$$
v_{\pi}(s)
  \;=\;
  \sum_{a} \pi(a \,\vert\, s)
  \sum_{s'} P(s' \,\vert\, s,a)\,\bigl[R(s,a,s') + \gamma\,v_{\pi}(s')\bigr]
  \quad
  \text{for all } s\in\mathcal{S}
$$

  can be viewed as a linear system (when $\gamma < 1$) if $\mathcal{S}$ is finite.  
• In principle, one can solve these equations exactly for $v_{\pi}$ by standard linear algebra if the state space $\mathcal{S}$ is not too large.  

More commonly, in large or unknown environments, we use iterative methods—such as iterative policy evaluation—to approximate or converge to the true $v_{\pi}(s)$:

1. Initialize $v^{(0)}(s)$ arbitrarily (e.g., zeros).  
2. For each iteration $n=0,1,2,\dots$, update
   

$$
v^{(n+1)}(s)
   \;\leftarrow\;
   \sum_{a} \pi(a \,\vert\, s)
   \sum_{s'} 
   P(s' \,\vert\, s,a)\,
   \bigl[R(s,a,s') + \gamma\,v^{(n)}(s')\bigr].
$$

3. Repeat until convergence.

────────────────

5 Why It Is Called a “Consistency” Equation

──────────

The Bellman equation essentially enforces consistency: the value of a state $s$ must match the expected immediate reward plus the discounted value of the successor state under the given policy. It is a self-consistent recursive definition of $v_{\pi}$.

─────────────────────────

6 Derivation Summary
   
───────────────────

In summary, the Bellman equation is derived by:  
1. Defining $v_{\pi}(s) = \mathbb{E}[G_t \mid S_t=s]$.  
2. Decomposing the return $G_t$ into immediate reward plus discounted future return, $R_{t+1} + \gamma\,G_{t+1}$.  
3. Taking expectations over the random action-choice and state-transition processes dictated by $\pi$ and $P$.  

Hence, we arrive at the Bellman expectation equation, which provides a system of equations for $v_{\pi}(s)$.

─────────

7 Conclusion

──────

• The Bellman equation for a policy $\pi$ is

$$
v_{\pi}(s)
\;=\;
\sum_{a \,\in\, \mathcal{A}}
\pi(a \,\vert\, s)
\sum_{s' \,\in\, \mathcal{S}}
P(s' \,\vert\, s,\,a)\,
\bigl[
R(s,a,s')
\;+\;
\gamma\,v_{\pi}(s')
\bigr].
$$

• It is used to calculate $v_{\pi}(s)$ for all $s$ either by directly solving this system (in the finite state case) or by applying iterative policy evaluation methods.  

This completes the derivation and usage of the Bellman equation in the context of evaluating a given policy $\pi$.

## Bellman Equation - Details

Back to the Bellman equation under a given policy $\pi$. The state value function is given by

\[ 
v_{\pi}(s) \;=\; \mathbb{E}\bigl[G_t \;\big\vert\; S_t = s\bigr] \;=\; \mathbb{E}\Bigl[\,\sum_{k=0}^{\infty}
\gamma^k\,R_{t+k+1}
\;\Big\vert\; S_t = s \Bigr].
\]

Using a one step look ahead, we have

\[
G_t
\;=\;
R_{t+1} 
\;+\;
\gamma\,G_{t+1}.
\]

where $R_{t+1}$ is the immediate reward and the $\gamma \, G_{t+1}$ is future discounted reward. Taking the expected value on both sides,

\[
\mathbb{E} \bigl[G_t \;\big\vert\; S_t = s \bigr] = \mathbb{E} \bigl[R_{t+1} \;\big\vert\; S_t = s \bigr] + \mathbb{E} \bigl[ \gamma G_{t+1} \;\big\vert\; S_t = s \bigr]
\]

Complete the derivation of the this equation leading to the Bellman equation for $v_{\pi}$.

**Answer**

Below is a step-by-step derivation, starting from the one-step lookahead identity and arriving at the Bellman equation for the state-value function under a given policy \(\pi\). All mathematical expressions are shown in LaTeX.

1 **Recall the One-Step Decomposition**

We have:

\[
G_t 
\;=\; 
R_{t+1} \;+\;\gamma\,G_{t+1}.
\]

Taking the expectation of both sides, conditioned on \(S_t = s\) and following policy \(\pi\), gives:

\[
\mathbb{E}[\,G_t \mid S_t = s,\,\pi]
\;=\;
\mathbb{E}[\,R_{t+1} \mid S_t = s,\,\pi]
\;+\;
\gamma\,\mathbb{E}[\,G_{t+1} \mid S_t = s,\,\pi].
\]

Because  
\(\mathbb{E}[\,G_t \mid S_t = s,\,\pi] = v_{\pi}(s)\),  
we can rewrite this as 

\[
v_{\pi}(s) 
\;=\;
\mathbb{E}[\,R_{t+1} \mid S_t = s,\,\pi]
\;+\;
\gamma\,\mathbb{E}[\,G_{t+1} \mid S_t = s,\,\pi].
\]

2 **Break Down the Expectations**

a) The term \(\mathbb{E}[\,R_{t+1} \mid S_t = s,\,\pi]\)

Under policy \(\pi\), the agent chooses action \(A_t = a\) in state \(s\) with probability \(\pi(a \mid s)\). Then, the environment transitions to a next state \(s' \in \mathcal{S}\) with probability \(P(s' \mid s,a)\), supplying a reward \(R(s,a,s')\). Therefore,

\[
\mathbb{E}[R_{t+1} \mid S_t = s,\,\pi]
\;=\;
\sum_{a \,\in\, \mathcal{A}}
\pi(a \,\vert\, s)
\sum_{s' \,\in\, \mathcal{S}}
P(s' \mid s,a)\;R(s,a,s').
\]

b) The term \(\mathbb{E}[\,G_{t+1} \mid S_t = s,\,\pi]\)

Once the agent has taken action \(a\) in state \(s\) and arrived in \(s'\), it continues following policy \(\pi\) from time \(t+1\). Hence, the expected future return from \(s'\) onward is just \(v_{\pi}(s')\). Again summing over all possible next states \(s'\) and actions \(a\):

\[
\mathbb{E}[\,G_{t+1} \mid S_t = s,\,\pi]
\;=\;
\sum_{a \,\in\, \mathcal{A}}
\pi(a \,\vert\, s)
\sum_{s' \,\in\, \mathcal{S}}
P(s' \mid s,a)\;\underbrace{
  \mathbb{E}\bigl[G_{t+1} \mid S_{t+1}=s',\,\pi\bigr]
}_{v_{\pi}(s')}.
\]

Thus:

\[
\mathbb{E}[\,G_{t+1} \mid S_t = s,\,\pi]
\;=\;
\sum_{a}
\pi(a \,\vert\, s)
\sum_{s' }
P(s' \mid s,a)\;v_{\pi}(s').
\]

3 **Combine the Two Components**

Putting them all together:

\[
v_{\pi}(s)
\;=\;
\underbrace{
  \sum_{a}
  \pi(a \mid s)
  \sum_{s'}
  P(s' \mid s,a)\,R(s,a,s')
}_{\mathbb{E}[R_{t+1}\,\vert\,S_t=s,\;\pi]}
\;+\;
\gamma
\underbrace{
  \sum_{a}
  \pi(a \mid s)
  \sum_{s'}
  P(s' \mid s,a)\,v_{\pi}(s')
}_{\mathbb{E}[G_{t+1}\,\vert\,S_t=s,\;\pi]}.
\]

Factor out the sums more compactly:

\[
v_{\pi}(s)
\;=\;
\sum_{a \,\in\, \mathcal{A}}
\pi(a \,\vert\, s)
\sum_{s' \,\in\, \mathcal{S}}
P(s' \,\vert\, s,a)\,
\Bigl[
  R(s,a,s') 
  \;+\;
  \gamma\,v_{\pi}(s')
\Bigr].
\]

This equation is referred to as the Bellman Expectation Equation (or simply the Bellman equation) for the state-value function \(v_{\pi}\).

4 **Final Bellman Equation for \(v_{\pi}\)**

Hence, the derivation shows that:

\[
\boxed{
v_{\pi}(s)
\;=\;
\sum_{a \,\in\, \mathcal{A}}
\pi(a \,\vert\, s)
\sum_{s' \,\in\, \mathcal{S}}
P(s' \,\vert\, s,a)\,
\bigl[
  R(s,a,s') \;+\; \gamma\,v_{\pi}(s')
\bigr].
}
\]

From the one-step lookahead identity 
\(
G_t = R_{t+1} + \gamma G_{t+1}
\)
and taking conditional expectations w.r.t. \((S_t = s,\pi)\), we arrive at this recursive relation for \(v_{\pi}\). This is precisely the Bellman equation under a given policy \(\pi\).

## Bellman Equation $\to$ System of Linear Equations

It seems that for the finite state space $\mathcal{S}$, the Bellman equation results in a system of linear equations in terms of the "variables" $v_{\pi}(s_t)$. Is this correct?

**Answer**

Below is a step-by-step explanation showing that for a finite state space $\mathcal{S}$, the Bellman equation corresponds to a system of linear equations in the unknowns $v_{\pi}(s)$:

1) The Bellman Equation for a Fixed Policy  
   Recall the Bellman expectation equation for a policy $\pi$ and a finite MDP:  
   

$$
v_{\pi}(s)
   \;=\;
   \sum_{a \,\in\, \mathcal{A}}
   \pi(a \,\vert\, s)
   \sum_{s' \,\in\, \mathcal{S}}
   P(s' \,\vert\, s,\,a)
   \,\Bigl[
     R(s,a,s') \;+\; \gamma\,v_{\pi}(s')
   \Bigr].
$$

  
   Here,  
   •  $\mathcal{S} = \{s_1, s_2, \dots, s_{|\mathcal{S}|}\}$ is a finite set of states.  
   •  $\mathcal{A}$ is the finite set of possible actions.  
   •  $\pi(a\mid s)$ is the probability of action $a$ in state $s$.  
   •  $P(s' \mid s,a)$ is the probability of transitioning to $s'$ from $s$ after action $a$.  
   •  $R(s,a,s')$ is the immediate reward.

2) Expressing the Equation for Each State  
   For each $s \in \mathcal{S}$, we can write a separate instance of the Bellman equation. Concretely, labeling the states $s_1, s_2, \dots, s_{|\mathcal{S}|}$, we get $|\mathcal{S}|$ equations of the form:  
   

$$
v_{\pi}(s_i)
   \;=\;
   \sum_{a \,\in\, \mathcal{A}}
   \pi(a \,\vert\, s_i)
   \sum_{s_j \,\in\, \mathcal{S}}
   P(s_j \,\vert\, s_i,\,a)\,
   \Bigl[
     R(s_i,a,s_j) \;+\; \gamma\,v_{\pi}(s_j)
   \Bigr]
   \quad\text{for}\quad i=1,2,\dots,|\mathcal{S}|.
$$

3) Rearranging into Linear System Form  
   Notice that $v_{\pi}(s_i)$ appears on the left-hand side, and $v_{\pi}(s_j)$ (with potentially different $j$) appears inside the sum on the right-hand side. What we have is a set of linear equations in the unknowns $v_{\pi}(s_1),\, v_{\pi}(s_2), \dots, v_{\pi}(s_{|\mathcal{S}|})$, because each equation can be rearranged to something like:  
   

$$
v_{\pi}(s_i)
   \;-\;
   \gamma
   \sum_{a,\,s_j}
   \Bigl[
   \text{(coefficient)} \times v_{\pi}(s_j)
   \Bigr]
   \;=\;
   \sum_{a,\,s_j}
   \Bigl[
   \text{(coefficient)} \times R(s_i,a,s_j)
   \Bigr].
$$

  
   All terms involving $v_{\pi}(s_i)$ or $v_{\pi}(s_j)$ are multiplied by constant coefficients (which come from $\pi(a\mid s_i)$, $P(s_j\mid s_i,a)$, and $\gamma$).

4) Matrix Notation  
   In matrix form, one can write:  
   

$$
\mathbf{v}_{\pi}
   \;=\;
   \mathbf{r}_{\pi} 
   \;+\;
   \gamma \,\mathbf{P}_{\pi}\,\mathbf{v}_{\pi},
$$

  
   where:  
   • $\mathbf{v}_{\pi}$ is a column vector $\bigl(v_{\pi}(s_1), v_{\pi}(s_2), \ldots, v_{\pi}(s_{|\mathcal{S}|})\bigr)^\top$.  
   • $\mathbf{r}_{\pi}$ is the vector of expected immediate rewards in each state (depending on $\pi$).  
   • $\mathbf{P}_{\pi}$ is the state transition probability matrix when actions are chosen according to $\pi$.  

   Rearranging:  
   

$$
\mathbf{v}_{\pi} 
   \;-\;
   \gamma\,\mathbf{P}_{\pi}\,\mathbf{v}_{\pi}
   \;=\;
   \mathbf{r}_{\pi},
$$

  
   or  
   

$$
(\mathbf{I} - \gamma\,\mathbf{P}_{\pi})\,\mathbf{v}_{\pi}
   \;=\;
   \mathbf{r}_{\pi}.
$$

  
   This is a linear system of dimension $|\mathcal{S}|\times|\mathcal{S}|$.

5) Conclusion  
   Therefore, for each state in a finite MDP, the Bellman equation contributes one linear equation in the unknown $v_{\pi}(s)$. Putting them all together yields a system of $|\mathcal{S}|$ linear equations. Solving this system (when $\gamma<1$ and $\mathbf{I} - \gamma\,\mathbf{P}_{\pi}$ is invertible) yields the value function $v_{\pi}(s)$ for all states simultaneously.  

Indeed, your observation is correct: in a finite state space, the Bellman equation is precisely a system of linear equations in the variables $\{\,v_{\pi}(s): s\in \mathcal{S}\}$.

## Bellman Equation matrix-vector form

For each state $s \in \mathcal{S}$, and concretely labeling the states as $s_1, s_2, \ldots, s_{|\mathcal{s}|}$, we can write a Bellman equation,

\[
v_{\pi}(s_i)
   \;-\;
   \gamma
   \sum_{a,\,s_j}
   \Bigl[
   \text{(coefficient)} \times v_{\pi}(s_j)
   \Bigr]
   \;=\;
   \sum_{a,\,s_j}
   \Bigl[
   \text{(coefficient)} \times R(s_i,a,s_j)
   \Bigr]
\]

which can be written in the form:

\[
v_{\pi}(s_i)
   \;-\;
   \gamma
   \sum_{a,\,s_j}
   \Bigl[
   \text{(coefficient)} \times v_{\pi}(s_j)
   \Bigr]
   \;=\;
   \sum_{a,\,s_j}
   \Bigl[
   \text{(coefficient)} \times R(s_i,a,s_j)
   \Bigr].
\]

which then leads to the matrix-vector form:

\[
\mathbf{v}_{\pi}
   \;=\;
   \mathbf{r}_{\pi} 
   \;+\;
   \gamma \,\mathbf{P}_{\pi}\,\mathbf{v}_{\pi}
\]

Write out the term $\text{(coefficient)}$ that appears above, fully. The entries of the matrix $\mathbf{P}_{\pi}$ must be related to $\text{(coefficient)}$. Clarify this.

***Answer***

Below is a step-by-step explanation of the coefficient that appears in the Bellman equation for a policy \(\pi\), written in full LaTeX, and how it relates to the entries of the matrix \(\mathbf{P}_{\pi}\) in the matrix-vector equation:

1 **Single-State Bellman Equation**

For a fixed state \(s_{i}\) in a finite state space \(\mathcal{S} = \{s_{1},\ldots,s_{|\mathcal{S}|}\}\), the Bellman equation under a policy \(\pi\) can be written as:

\[
v_{\pi}(s_{i})
\;=\;
\sum_{a \,\in\, \mathcal{A}}
\pi\bigl(a \,\vert\, s_{i}\bigr)
\sum_{s_{j} \,\in\, \mathcal{S}}
P\bigl(s_{j} \,\vert\, s_{i},\,a\bigr)
\Bigl[
  R\bigl(s_{i},\,a,\,s_{j}\bigr)
  \;+\;
  \gamma \, v_{\pi}(s_{j})
\Bigr].
\]

Rearranging terms to isolate \(v_{\pi}(s_{i})\) on the left and the discounted values on the right gives a form like:

\[
v_{\pi}(s_{i})
\;-\;
\gamma
\sum_{a\,\in\,\mathcal{A}}
\sum_{s_{j}\,\in\,\mathcal{S}}
\Bigl[\text{(coefficient)}\times v_{\pi}(s_{j})\Bigr]
\;=\;
\sum_{a\,\in\,\mathcal{A}}
\sum_{s_{j}\,\in\,\mathcal{S}}
\Bigl[\text{(coefficient)}\times R\bigl(s_{i},\,a,\,s_{j}\bigr)\Bigr].
\]

2 **Identifying the Coefficient**

Inside the double sum over \(a \in \mathcal{A}\) and \(s_{j} \in \mathcal{S}\), the coefficient that multiplies \(v_{\pi}(s_{j})\) (before factoring out \(\gamma\)) is

\[
\pi\bigl(a \,\vert\, s_{i}\bigr)
\;\times\;
P\bigl(s_{j} \,\vert\, s_{i},\,a\bigr).
\]

Thus, if we look at the term on the left that is multiplied by \(\gamma\), the \(\text{(coefficient)}\) is explicitly:

\[
\gamma
\;\times\;
\pi\bigl(a \,\vert\, s_{i}\bigr)
\;\times\;
P\bigl(s_{j} \,\vert\, s_{i},\,a\bigr).
\]

Similarly, the coefficient that appears with the reward \(R(s_{i},\,a,\,s_{j})\) (on the right-hand side) is

\[
\pi\bigl(a \,\vert\, s_{i}\bigr)
\;\times\;
P\bigl(s_{j} \,\vert\, s_{i},\,a\bigr).
\]

3 **Matrix-Vector Equation**

Collecting all \(|\mathcal{S}|\) such Bellman equations for \(s_{1}, \dots, s_{|\mathcal{S}|}\) yields the matrix-vector form:

\[
\mathbf{v}_{\pi}
\;=\;
\mathbf{r}_{\pi}
\;+\;
\gamma\,\mathbf{P}_{\pi}\,\mathbf{v}_{\pi}.
\]

Here:

1. \(\mathbf{v}_{\pi}\) is the \(|\mathcal{S}|\times 1\) column vector of values \(\bigl[v_{\pi}(s_{1}),\,\dots,\,v_{\pi}(s_{|\mathcal{S}|})\bigr]^{\!\top}\).  
2. \(\mathbf{r}_{\pi}\) is the \(|\mathcal{S}|\times 1\) column vector of expected immediate rewards under policy \(\pi\) for each state.  
3. \(\mathbf{P}_{\pi}\) is an \(|\mathcal{S}|\times |\mathcal{S}|\) matrix whose entries \(\bigl[\mathbf{P}_{\pi}\bigr]_{i,j}\) capture the probability of transitioning from \(s_{i}\) to \(s_{j}\) when actions are chosen according to \(\pi\).

4 **Relation of the Coefficient to \(\mathbf{P}_{\pi}\)**

By definition,

\[
\bigl[\mathbf{P}_{\pi}\bigr]_{i,j}
\;=\;
\sum_{a \,\in\,\mathcal{A}}
\pi\bigl(a \,\vert\, s_{i}\bigr)
P\bigl(s_{j} \,\vert\, s_{i},\,a\bigr).
\]

In other words, each matrix entry \([\mathbf{P}_{\pi}]_{i,j}\) is the total transition probability from state \(s_{i}\) to \(s_{j}\) under policy \(\pi\), which is the sum over all actions \(a\) of the product \(\pi(a \mid s_{i})\,P(s_{j} \mid s_{i},a)\).

Hence the coefficient \(\pi(a \mid s_{i})\,P(s_{j}\mid s_{i},a)\) that appears in the single-state Bellman equation is precisely the partial contribution to \([\mathbf{P}_{\pi}]_{i,j}\) from action \(a\). Summing that coefficient over all \(a\) in a given row \((i)\) gives the full matrix entry \([\mathbf{P}_{\pi}]_{i,j}\).

5 **Summary**

• The “\(\text{(coefficient)}\)” that multiplies \(v_{\pi}(s_{j})\) in the single-state Bellman equation is

\[
\pi\bigl(a \,\vert\, s_{i}\bigr)
\;\times\;
P\bigl(s_{j} \,\vert\, s_{i},\,a\bigr).
\]

• When writing the full system in matrix form, these coefficients for each \((i,j)\) and each action \(a\) combine into the entries of the matrix \(\mathbf{P}_{\pi}\). Specifically:

\[
[\mathbf{P}_{\pi}]_{i,j}
\;=\;
\sum_{a \,\in\, \mathcal{A}}
\pi(a \,\vert\, s_{i})\;
P\bigl(s_{j} \,\vert\, s_{i},\,a\bigr).
\]

• Consequently, the compact Bellman equation

\[
\mathbf{v}_{\pi}
\;=\;
\mathbf{r}_{\pi}
\;+\;
\gamma
\,\mathbf{P}_{\pi}\,\mathbf{v}_{\pi}
\]

follows naturally from collecting each row’s (i.e., state’s) single-state Bellman equation in vector/matrix form.

## Comapring Policies

Suppose you have two policies $\pi_1$ and $\pi_2$. And also suppose that the Bellman equation is solved so that we get $v_{\pi_1}$ and $v_{\pi_2}$.  How is this used to evaluate and compare the two policies?

**Answer**

Below is a step-by-step explanation of how having the solved value functions $v_{\pi_1}$ and $v_{\pi_2}$ (from two different policies $\pi_1$ and $\pi_2$) enables one to compare and evaluate those policies:

1 **The Meaning of $v_{\pi}(s)$**

• For a given policy $\pi$ in a Markov Decision Process (MDP), the value function  
  

$$
v_{\pi}(s)
  \;=\;
  \mathbb{E}\!\Bigl[G_t \,\Big\vert\, S_t = s\Bigr]
$$

  
  tells us the expected return (cumulative discounted reward) starting from state $s$ and following $\pi$ thereafter.  
• Thus, $v_{\pi}(s)$ captures how “good” it is to be in state $s$ when one plans to follow $\pi$.

2 **Comparing Two Policies State by State**

• Once you have computed $v_{\pi_1}(s)$ and $v_{\pi_2}(s)$ for all $s \in \mathcal{S}$, you can compare the policies pointwise by looking at each state $s$:  
  – If $v_{\pi_1}(s) > v_{\pi_2}(s)$, then $\pi_1$ yields a higher expected return than $\pi_2$ specifically from state $s$.  
  – If $v_{\pi_2}(s)$ exceeds $v_{\pi_1}(s)$, then $\pi_2$ is better from that state.  
• A strict ordering of the policies for all states might not hold (one can find states where $\pi_1$ is better and others where $\pi_2$ is better). That said, if  
  

$$
v_{\pi_1}(s)\;\ge\;v_{\pi_2}(s)
  \quad\text{for all }s,
$$

  then we can unambiguously say $\pi_1$ is at least as good as $\pi_2$ in every state.

3 **Comparing Two Policies from an Initial State or Distribution**

• In many RL problems, there is a designated initial state $s_0$ or an initial state distribution $\mu_0$. In that scenario, one often compares the policies by looking at  
  

$$
v_{\pi_1}(s_0)
  \quad\text{versus}\quad
  v_{\pi_2}(s_0),
$$

  or by evaluating the expected return under $\mu_0$:  
  

$$
\sum_{s\in \mathcal{S}} 
  \mu_0(s)\,
  v_{\pi_1}(s)
  \quad\text{versus}\quad
  \sum_{s\in \mathcal{S}} 
  \mu_0(s)\,
  v_{\pi_2}(s).
$$

• The policy that yields a higher overall expected return from the start (or on average w.r.t. $\mu_0$) is deemed better.

4 **Practical Significance**

• If you have solved the Bellman equation for both policies (i.e., found $v_{\pi_1}$ and $v_{\pi_2}$), then you have a quantitative measure of performance for each state in $\mathcal{S}$.  
• You can use this information to decide which policy performs better in specific states, or from the viewpoint of a particular start state.  
• Often, this comparison step is part of Policy Iteration or other methods: one checks if an updated policy is better (in terms of the value function) than a previous one—if it is, one replaces the old policy.

5 **Conclusion**

Yes, once you solve the Bellman equations to get $v_{\pi_1}$ and $v_{\pi_2}$, you can evaluate and compare $\pi_1$ and $\pi_2$ by:  
1. Checking their value functions state by state, if desired.  
2. Checking the expected return from a specific start state or an initial state distribution.  

The policy with the larger expected return (from the relevant initial condition) is considered better. If one policy has a higher value function in all states, it strictly dominates the other policy.

## Return value

Below is a re-presentation of the concept of the action-value function, also known as the “Q-value” function, denoted by $q_{\pi}$:

1) Motivation for $q_{\pi}(s,a)$  
   • We already know that for a given policy $\pi$, the state-value function is:  
     

$$
v_{\pi}(s) \;=\; \mathbb{E}\bigl[G_t \;\big\vert\; S_t = s\bigr].
$$

  
   • However, this quantity does not distinguish which action is taken in state $s$; it only assumes that actions will be chosen according to $\pi$.  
   • If we want to evaluate taking a specific action $a$ in state $s$, we introduce the action-value function.

2) Definition of the Action-Value  
   • The action-value function $q_{\pi}(s,a)$ is defined as:  
     

$$
q_{\pi}(s,a) \;=\; \mathbb{E}\bigl[G_t \;\big\vert\; S_t = s,\; A_t = a,\; \pi\bigr].
$$

  
   • This is the expected return obtained when:  
     1. We start in state $s$.  
     2. We immediately take action $a$.  
     3. Thereafter, we continue following policy $\pi$.  

3) Relationship to $v_{\pi}(s)$  
   • From the definition above, after we take action $a$ in state $s$, we end up in some next state $s'$ and keep following $\pi$.  
   • If $\pi$ is stochastic, then  
     

$$
v_{\pi}(s) 
     \;=\; 
     \sum_{a \in \mathcal{A}} \pi(a \mid s)\; q_{\pi}(s,a).
$$

  
   • Intuitively, $v_{\pi}(s)$ is the average of $q_{\pi}(s,a)$ over all possible actions $a$, weighted by $\pi(a \mid s)$.

4) Bellman Equation for $q_{\pi}$  
   • The Bellman expectation equation for $q_{\pi}(s,a)$ can be written as:  
     

$$
q_{\pi}(s,a)
     \;=\; 
     \mathbb{E}\Bigl[
       R_{t+1} \;+\; \gamma \, v_{\pi}(S_{t+1})
       \;\Big\vert\;
       S_t = s,\; A_t = a
     \Bigr].
$$

  
   • Equivalently, since  
     

$$
v_{\pi}(S_{t+1})
     \;=\;
     \sum_{a'} \pi(a' \mid S_{t+1})\,q_{\pi}(S_{t+1},\,a'),
$$

  
we can expand the right-hand side to express $q_{\pi}$ entirely in terms of immediate rewards and future $q_{\pi}$-values.

5) Why $q_{\pi}$ Matters  
   • Many RL algorithms, such as Q-learning and SARSA, center around learning an approximation of $q_{\pi}(s,a)$.  
   • Once $q_{\pi}$ is known (or estimated), choosing actions becomes straightforward: one can compare the values of different actions in the same state to see which leads to the greatest expected return.

6) Summary  
   • The state-value function $v_{\pi}(s)$ follows a policy everywhere.  
   • The action-value function $q_{\pi}(s,a)$ is a finer-grained measure that tells us the expected return if we pick a specific action $a$ now, then continue with $\pi$.  
   • Both functions are crucial to understanding and devising strategies in Reinforcement Learning.


## Bellman Equations for $q_{\pi}$ and $v_{\pi}$

***The step 4 above on the Bellman equation for $q_{\pi}$. Derive this with details and show the connection to the Bellman equation for $v_{\pi}$.***

***Answer***

Below is a detailed derivation of the Bellman equation for the action-value function, denoted by $q_{\pi}(s,a)$, 
showing its relationship to the Bellman equation for the state-value function $v_{\pi}(s)$.


────────────────────────────────

1) Recall the Definitions
   
───────────────────────────────

• State-Value Function (under policy $\pi$):  
  
$$
v_{\pi}(s)
  \;=\;
  \mathbb{E}\bigl[G_t 
    \;\big\vert\;
    S_t = s
  \bigr],
$$

  which is the expected return when starting in state $s$ at time $t$ and thereafter following $\pi$.

• Action-Value Function (under policy $\pi$):  
  
$$
q_{\pi}(s,a)
  \;=\;
  \mathbb{E}\bigl[G_t
    \;\big\vert\;
    S_t = s,\;
    A_t = a,\;
    \pi
  \bigr],
$$

  which is the expected return when starting in state $s$ at time $t$, taking action $a$ immediately, and then following $\pi$ thereafter.

• Return $G_t$:  
  
$$
G_t
  \;=\;
  R_{t+1} \;+\;\gamma R_{t+2} \;+\;\gamma^2 R_{t+3} \;+\;\dots
  \;=\;
  \sum_{k=0}^{\infty}
  \gamma^k \, R_{t+k+1}.
$$

  Here, $0 \le \gamma \le 1$ is the discount factor.

────────────────────────────────

2 One-Step Decomposition of the Return

────────────────────────────────

We start with the relation  

$$
G_t
\;=\;
R_{t+1} \;+\; \gamma\,G_{t+1}.
$$


Taking the conditional expectation on both sides given $\bigl(S_t = s,\; A_t = a,\; \pi\bigr)$:


$$
q_{\pi}(s,a)
\;=\;
\mathbb{E}\bigl[
  G_t
  \;\big\vert\;
  S_t = s,\;
  A_t = a,\;
  \pi
\bigr]
\;=\;
\mathbb{E}\Bigl[
  R_{t+1} \;+\; \gamma\,G_{t+1}
  \;\Big\vert\;
  S_t = s,\;
  A_t = a,\;
  \pi
\Bigr].
$$


─────────────────────────────────

3 Separate Immediate Reward and Future Return
   
────────────────────────────────

We can split the expectation:


$$
q_{\pi}(s,a)
\;=\;
\underbrace{
  \mathbb{E}\bigl[
    R_{t+1}
    \,\big\vert\,
    S_t = s,\;
    A_t = a,\;
    \pi
  \bigr]
}_{\text{expected immediate reward}}
\;+\;
\gamma
\underbrace{
  \mathbb{E}\bigl[
    G_{t+1}
    \,\big\vert\,
    S_t = s,\;
    A_t = a,\;
    \pi
  \bigr]
}_{\text{expected future discounted return}}
.
$$


1. The term $\mathbb{E}[\,R_{t+1}\,\vert\, s,a,\pi]$ is the expected reward for taking action $a$ in state $s$.  
2. The term $\mathbb{E}[\,G_{t+1}\,\vert\, s,a,\pi]$ signals that once we have taken action $a$ in state $s$, we transition to some state $s'$, and from $t+1$ onward we follow policy $\pi$. The expected future return from $s'$ under $\pi$ is $v_{\pi}(s')$.

────────────────────────────────

4 Transition to the Next State

────────────────────────────────

The environment’s transition function $P(s' \mid s,a)$ gives the probability of ending up in state $s'$ after taking action $a$ in state $s$. Hence,


$$
\mathbb{E}\bigl[
  R_{t+1}
  \;\big\vert\;
  S_t = s,\;
  A_t = a
\bigr]
\;=\;
\sum_{s' \,\in\, \mathcal{S}}
P(s' \;\vert\; s,a)\,\;R(s,a,s'),
$$


assuming $R(s,a,s')$ is the associated immediate reward for the transition $(s,a)\to s'$.  
Similarly,


$$
\mathbb{E}\bigl[
  G_{t+1}
  \;\big\vert\;
  S_t = s,\;
  A_t = a,\;
  \pi
\bigr]
\;=\;
\sum_{s' \,\in\, \mathcal{S}}
P(s' \;\vert\; s,a)
\,\underbrace{
  \mathbb{E}[\,G_{t+1}\,\vert\,S_{t+1} = s',\,\pi]
}_{v_{\pi}(s')}
,
$$

since once we arrive in $s'$, we follow $\pi$ and the expected return is $v_{\pi}(s')$.

─────────────────────────────────

5 The Bellman Equation for $q_{\pi}(s,a)$

─────────────────────────────────

Putting these pieces together:


$$
q_{\pi}(s,a)
\;=\;
\sum_{s' \,\in\, \mathcal{S}}
P(s' \,\vert\, s,a)
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma \, v_{\pi}(s')
\Bigr].
$$


This is the Bellman expectation equation for $q_{\pi}(s,a)$. It shows that the action-value under $\pi$ is the expected immediate reward plus the discounted value of the next state, averaged over all possible next states $s'$.

─────────────────────────────────

6 Connection to the Bellman Equation for $v_{\pi}(s)$

──────────────────────

The state-value function $v_{\pi}(s)$ can be expressed in terms of $q_{\pi}(s,a)$:


$$
v_{\pi}(s)
\;=\;
\sum_{a \,\in\, \mathcal{A}}
\pi(a \,\vert\, s)\;q_{\pi}(s,a).
$$


In other words, $v_{\pi}(s)$ is the expectation of $q_{\pi}(s,a)$ over the policy $\pi(a \mid s)$. Conversely, when we write $q_{\pi}(s,a)$ in its Bellman form, we use $v_{\pi}(s')$ to handle the future return once the agent transitions to the next state $s'$:


$$
q_{\pi}(s,a)
\;=\;
\sum_{s' \,\in\, \mathcal{S}}
P(s' \,\vert\, s,a)
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma \,\underbrace{
    \sum_{a' \,\in\, \mathcal{A}}
    \pi(a' \,\vert\, s')\,q_{\pi}(s',a')
  }_{v_{\pi}(s')}
\Bigr].
$$


Thus, $q_{\pi}$ and $v_{\pi}$ are tied together through these two Bellman equations.

─────────────────────────────────

7 Summary

─────────────────────────────────

• We started from the relation $G_t = R_{t+1} + \gamma G_{t+1}$, conditioning on $\{S_t = s, A_t = a\}$ and policy $\pi$.  
• We split the expectation into the immediate reward plus the discounted future return.  
• By summing over next states $s'$ and using the definition of $v_{\pi}(s')$, we arrived at


$$
q_{\pi}(s,a)
\;=\;
\sum_{s' \,\in\, \mathcal{S}}
P(s' \,\vert\, s,a)
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma\,v_{\pi}(s')
\Bigr].
$$


• Since $v_{\pi}(s') = \sum_{a'} \pi(a' \mid s')\,q_{\pi}(s',a')$, we see the close relationship between these two value functions. This completes the derivation of the Bellman equation for the action-value function $q_{\pi}(s,a)$ and clarifies its connection to the Bellman equation for $v_{\pi}(s)$.

## Optimal Policy

***Now that we have the concepts of the state value $v_{\pi}$ and the return value $q_{\pi}$ for a given policy $\pi$, and the associated Bellman equations, there is the concept of the optimal policy $\pi^*$. There is an associated concept of the Bellman Optimality Equation. Explain these concepts.***

***Answer***

Below is a step-by-step explanation of the concept of an optimal policy in Reinforcement Learning, and how it leads to the Bellman Optimality Equations.

──────────────────────────

1 Policy and Value Functions

____________________________
   

• A policy $\pi$ is any mapping from states to action probabilities (or actions, if deterministic).  
• The state-value function for a given policy $\pi$ is  

$$
v_{\pi}(s)
\;=\;
\mathbb{E}\bigl[G_t \;\big\vert\; S_t = s,\;\pi \bigr],
$$
  
where $G_t$ is the return from time $t$ onward.  
• The action-value function for $\pi$ is  

$$
q_{\pi}(s,a)
\;=\;
\mathbb{E}\bigl[G_t \;\big\vert\; S_t = s,\; A_t = a,\;\pi \bigr].
$$


These value functions describe how “good” it is to be in a particular state $s$—or to take a particular action $a$ in $s$—under the policy $\pi$.

_________________________

2 Optimal Policy $\pi^*$

─────────────────────────

• The goal in Reinforcement Learning is often to find a policy $\pi^*$ that maximizes the expected return from every state.  
• Formally, $\pi^*$ is said to be optimal if, for all states $s\in \mathcal{S}$,

$$
v_{\pi^*}(s)
\;=\;
\max_{\pi} \;
v_{\pi}(s).
$$

Equivalently, $\pi^*$ achieves or exceeds the performance of every other policy $\pi$.  
• There can be multiple optimal policies, but they share the same value functions, denoted $v_*$ and $q_*$, defined below.

──────────────────────────────

3 Optimal Value Functions

────────────────────────────

We define  

$$
v_*(s)
\;=\;
\max_{\pi} \;v_{\pi}(s),
$$

and  

$$
q_*(s,a)
\;=\;
\max_{\pi} \;q_{\pi}(s,a).
$$


• Thus, $v_*(s)$ is the maximum possible expected return achievable from state $s$, over all policies.  
• Similarly, $q_*(s,a)$ is the maximum possible expected return when starting from $s$, taking action $a$, and thereafter following the best possible policy.

───────────────────────────────

4 Bellman Optimality Equations

────────────────────────

• Just as there are Bellman equations for $v_{\pi}$ and $q_{\pi}$ under a fixed policy $\pi$, there are corresponding Bellman Optimality Equations for $v_*$ and $q_*$.  
• For the optimal state-value function $v_*(s)$, the Bellman Optimality Equation is:

$$
v_*(s)
\;=\;
\max_{a \,\in\,\mathcal{A}}
\;\biggl[
  \sum_{s' \,\in\,\mathcal{S}}
  P(s' \mid s,a)\,
  \Bigl(
    R(s,a,s') \;+\;\gamma\,v_*(s')
  \Bigr)
\biggr].
$$

• In words: from state $s$, pick the action $a$ that gives the highest expected immediate reward plus the discounted optimal value of the next state.

• For the optimal action-value function $q_*(s,a)$, the Bellman Optimality Equation is:

$$
q_*(s,a)
\;=\;
\sum_{s' \,\in\, \mathcal{S}}
P(s' \mid s,a)
\Bigl[
  R(s,a,s') \;+\;
  \gamma \,\max_{a'\,\in\,\mathcal{A}} \,q_*(s',a')
\Bigr].
$$

• Here, after taking action $a$ in state $s$ to reach $s'$, we assume the agent chooses the (optimal) action $a' = \arg\max_{\,a'} q_*(s',a')$ thereafter.

───────────────────────────────

5 Connection to the Concept of Optimality

────────────────────────────

• If the agent somehow knows $q_*$, it can act optimally by choosing, in each state $s$, the action $a$ that maximizes $q_*(s,a)$.  
• Equivalently, if the agent knows $v_*$ and the environment’s transition probabilities, then to select the optimal action from a state $s$, one can compute the action that attains the maximum in

$$
\max_{a}\;\bigl[
  \sum_{s'}
    P(s' \mid s,a)\,
    (\,R(s,a,s') + \gamma\,v_*(s')\,)
\bigr].
$$


──────────────────────────────

6 Summary

────────────────────────────

1. An optimal policy $\pi^*$ is one that achieves the highest possible value in every state (or equivalently, for every state-action pair).  
2. The optimal value functions $v_*$ and $q_*$ are defined as the supremum over all policies of $v_{\pi}$ and $q_{\pi}$, respectively.  
3. They satisfy the Bellman Optimality Equations, which replace the explicit dependence on $\pi$ with a maximization over actions.  
4. Solving these equations in principle yields $v_*$ (or $q_*$) and hence identifies which actions are optimal in each state, leading to an optimal policy.


## Lemma needed for Bellman Optimality Equation

Explain why this is true?

\[
\max_{\pi}\,
\sum_{a}
\pi(a \mid s)\,f(a)
\;=\;
\max_{a}\,f(a)
\]

**Answer**

Below is a step-by-step explanation of why the following equality holds:

\[
\max_{\pi}
\sum_{a}
\pi(a \mid s)\,f(a)
\;=\;
\max_{a}\,f(a).
\]

Here, \(\pi(a \mid s)\) is a probability distribution over actions \(a\) (i.e., \(\pi(a \mid s)\ge0\) and \(\sum_{a} \pi(a \mid s)=1\)), and \(f(a)\) is any real-valued function on the finite action set \(\mathcal{A}\).

1 **Statement of the Claim**

We want to show that for any function \(f:\mathcal{A}\to\mathbb{R}\),

\[
\max_{\pi}\,\Bigl(\sum_{a}\,\pi(a \,\vert\, s)\,f(a)\Bigr) =
\max_{a}\;f(a),
\]

where the maximum on the left is taken over all probability distributions \(\{\pi(a \mid s)\}_{a\in\mathcal{A}}\), that is, all ways of assigning nonnegative weights summing to 1, and the maximum on the right is simply the largest value of \(f(a)\) over actions \(a\).

2 **Interpretation: Probability-Weighted Averages vs. Maximum**

1. On the left, \(\sum_{a}\,\pi(a \mid s)\,f(a)\) is a weighted average of the values \(f(a)\), where the weights \(\{\pi(a \mid s)\}\) sum to 1.  
2. On the right, \(\max_{a}\,f(a)\) is the single highest value of \(f(a)\).  

Because a probability distribution \(\pi(\cdot \mid s)\) must be spread among all actions \(a\), the resulting sum is a convex combination (i.e., a weighted average) of the values \(f(a)\). A fundamental fact about taking a convex combination of numbers is that this weighted average cannot exceed the maximum of those numbers. This provides the basic intuition.

3 **Formal Argument**

Let \(M = \max_{a} f(a)\). By definition of the max, for each action \(a\),

\[
f(a) \;\le\; M.
\]

Hence, for any probability distribution \(\{\pi(a \mid s)\}\), we have

\[
\sum_{a} \pi(a \mid s)\,f(a)
\;\le\;
\sum_{a} \pi(a \mid s)\,M
\;=\;
M\,\sum_{a} \pi(a \mid s)
\;=\;
M.
\]

Therefore,

\[
\sum_{a}\,\pi(a \mid s)\,f(a)
\;\le\;
M
\quad\quad
\text{for all probability distributions }\pi.
\]

Taking the maximum over all \(\pi\) of the left-hand side, we get

\[
\max_{\pi}\,\Bigl(\sum_{a}\pi(a \mid s)\,f(a)\Bigr) \;\le\;
M.
\]

That is,

\[
\max_{\pi}\,\Bigl(\sum_{a}\,\pi(a \mid s)\,f(a)\Bigr)  \;\le\;
\max_{a}\,f(a).
\]

For the opposite direction (showing \(\ge\)), simply pick the distribution \(\pi^{*}\) that places all its mass on the action \(a^{*} = \operatorname*{argmax}_{a} f(a)\). In other words, let

\[
\pi^{*}(a \mid s) =
\begin{cases}
1, & \text{if } a=a^{*},\\
0, & \text{otherwise}.
\end{cases}
\]

Then

\[
\sum_{a}\,\pi^{*}(a \mid s)\,f(a) \;=\;
1 \cdot f(a^{*})  \;=\;
\max_{a}\,f(a).
\]

Hence,

\[
\max_{\pi}\,\Bigl(\sum_{a}\,\pi(a \mid s)\,f(a)\Bigr) \;\ge\;
\max_{a}\,f(a).
\]

Combining both inequalities yields

\[
\max_{\pi}\,\Bigl(\sum_{a}\,\pi(a \mid s)\,f(a)\Bigr) =
\max_{a}\,f(a),
\]

as desired.

4 **Conclusion**

This equality holds because:

• Weighted averages of a finite set of real values (i.e., convex combinations) can never exceed the largest of those values.  
• One can always choose a degenerate distribution that puts all probability on the single best action.

## Bellman Optimality Equation

***The selection of an optimal policy implies choosing the maximum or supremum over all policies. However, in step 4 describing the Bellman Optimality Equation, there does not seem to be an explicit selection across the set of policies. Perhaps, this is an outcome. If so, derive in detail, how the Bellman Optimality Equation follows from the notion of optimal value functions.***

***Answer***

Below is a detailed derivation showing how the notion of “maximizing over all policies” leads to the Bellman Optimality Equations, using LaTeX for all mathematical symbols and expressions.

─────────────────────────────

1 Optimal Value Functions

─────────────────────

We define the optimal state-value function and action-value function as follows:

• Optimal state-value function:  


$$
v_*(s) 
\;=\; 
\max_{\pi}\;v_{\pi}(s),
$$


where $v_{\pi}(s) = \mathbb{E}\!\bigl[G_t \,\vert\,S_t = s,\;\pi\bigr] $

• Optimal action-value function:


$$
q_*(s,a) 
\;=\;
\max_{\pi}\;q_{\pi}(s,a),
$$


where $q_{\pi}(s,a) = \mathbb{E}\!\bigl[G_t \,\vert\,S_t = s,\;A_t = a,\;\pi\bigr]$

The notation “$\max_{\pi}$” means we take the supremum (maximum) over all possible policies $\pi$. Under standard conditions (e.g., finite state space, bounded rewards, or $\gamma<1$), these maxima exist and define $v_*$ and $q_*$.

───────────────────────────────

2 Bellman Equation for a Given Policy $\pi$

─────────────────────

For reference, recall that for any fixed policy $\pi$, the Bellman expectation equations are:

• State-value:

$$
v_{\pi}(s)
\;=\;
\sum_{a \,\in\, \mathcal{A}}
\pi(a \mid s)
\sum_{s' \,\in\, \mathcal{S}}
P\bigl(s' \mid s,a\bigr)\,
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma\,v_{\pi}(s')
\Bigr].
$$


• Action-value:

$$
q_{\pi}(s,a)
\;=\;
\sum_{s' \,\in\, \mathcal{S}}
P\bigl(s' \mid s,a\bigr)
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma \,v_{\pi}(s')
\Bigr].
$$


These follow directly from the fact that $G_t = R_{t+1} + \gamma\,G_{t+1}$ and taking expectations conditioned on $(S_t=s, A_t=a)$ or $(S_t=s)$ respectively.

──────────────────────

3 Maximizing Over Policies for $v_*$

──────────────────────────────

By definition,

$$
v_*(s)
\;=\;
\max_{\pi}\,v_{\pi}(s).
$$

Substitute the Bellman equation for $v_{\pi}(s)$:


$$
v_*(s)
\;=\;
\max_{\pi}
\sum_{a}
\pi(a \mid s)
\sum_{s'}
P(s' \mid s,a)\,
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma\,v_{\pi}(s')
\Bigr].
$$


Notice that for each fixed state $s$, a policy $\pi$ chooses a distribution $\{\pi(a \mid s)\}$ over actions $a$. But taking a probability-weighted sum of values cannot exceed simply picking the single best action. Formally,


$$
\max_{\pi}\,
\sum_{a}
\pi(a \mid s)\,f(a)
\;=\;
\max_{a}\,f(a),
$$

for any function $f$. Let

$$
f(a)
\;=\;
\sum_{s'}
P(s' \mid s,a)
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma \,\underbrace{v_{\pi}(s')}_{\le v_*(s')}
\Bigr].
$$

Because $v_{\pi}(s') \le v_*(s')$ for all $\pi$ and $s'$, the best we can do is to replace $v_{\pi}(s')$ by $v_*(s')$. Thus,


$$
v_*(s)
\;=\;
\max_{a}
\sum_{s'}
P(s' \mid s,a)\,
\bigl[
  R(s,a,s')
  \;+\;
  \gamma\,v_*(s')
\bigr].
$$


This yields the Bellman Optimality Equation for $v_*(s)$:


$$
\boxed{
v_*(s)
\;=\;
\max_{a \,\in\, \mathcal{A}}\;
\sum_{s' \,\in\, \mathcal{S}}
P(s' \mid s,a)\;
\bigl[
  R(s,a,s')
  \;+\;\gamma\,v_*(s')
\bigr].
}
$$


────────────────────────────

4 Bellman Optimality Equation for $q_*$

───────────────────────

Similarly, from

$$
q_{\pi}(s,a)
\;=\;
\sum_{s'}
P(s' \mid s,a)
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma\,v_{\pi}(s')
\Bigr],
$$

we define

$$
q_*(s,a)
\;=\;
\max_{\pi}\;
q_{\pi}(s,a).
$$

But we already have

$$
v_{\pi}(s')
\;\le\;
v_*(s'),
$$

and

$$
v_*(s')
\;=\;
\max_{a'}\;
q_*(s',a').
$$

Hence, by substituting $v_*(s') = \max_{a'} q_*(s',a')$, we arrive at


$$
q_*(s,a)
\;=\;
\sum_{s'}
P(s' \mid s,a)
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma\;\max_{a'}\;q_*(s',a')
\Bigr].
$$


This is the Bellman Optimality Equation for the action-value function:


$$
\boxed{
q_*(s,a)
\;=\;
\sum_{s'\,\in\,\mathcal{S}}
P\bigl(s' \mid s,a\bigr)
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma
  \,\max_{a'\,\in\,\mathcal{A}}
  q_*(s',a')
\Bigr].
}
$$


──────────────────────────────

5 Interpretation and Summary

──────────────────────

• By defining $v_*$ and $q_*$ as the maximum over all policies of the corresponding value functions, we effectively pick the best possible action choices at each state.  
• The Markov Decision Process structure means that “maximizing over policies” can be carried out state by state, leading to a pointwise maximization over actions in the Bellman Optimality Equations.  
• Once one computes or approximates $q_*$, one obtains an optimal policy simply by greedily choosing the action that maximizes $q_*(s,a)$ in each state $s$. Equivalently, from $v_*$ and knowledge of the environment’s transitions and rewards, one can also identify the best actions.

Thus, the Bellman Optimality Equations naturally arise from the desire to choose the best action in each state—an outcome of maximizing over all possible policies in an MDP.

## Existence of optimal policies

***You claim that under standard conditions such as finite state space and bounded rewards, the optimal state value $v_*(s) \;=\; \max_{\pi}\;v_{\pi}(s)$ and the action value $q_*(s,a) \;=\;\max_{\pi}\;q_{\pi}(s,a)$ exist. Formalize the argument to support this claim.***

Below is a step-by-step argument showing why, under standard conditions such as a finite Markov Decision Process (MDP) with discounted and bounded rewards, the following optimal value functions exist and are well-defined:

• The optimal state-value function:  

\[
v_{*}(s) \;=\; \max_{\pi}\,v_{\pi}(s),
\]  

• The optimal action-value function:  

\[
q_{*}(s,a) \;=\; \max_{\pi}\,q_{\pi}(s,a).
\]

1 **Finite MDP and Bounded Reward Assumptions**

Assume:

1. A finite state space \(\mathcal{S} = \{\,s_{1},\dots, s_{|\mathcal{S}|}\}\).  
2. A finite action space \(\mathcal{A} = \{\,a_{1}, \dots, a_{|\mathcal{A}|}\}\).  
3. A discount factor \(0 \,\le\, \gamma < 1.\)  
4. Bounded rewards: there exists \(R_{\max} \ge 0\) such that  

\[
\bigl|\,R(s,a,s')\bigr| \;\le\; R_{\max}
\]

for all feasible triples \((s,a,s')\).

Under these conditions, each state-action-next-state combination appears in only finitely many possibilities, and any infinite sum of discounted, bounded rewards converges well.

2 **Value Functions for a Fixed Policy \(\pi\)**

Consider a fixed policy \(\pi\). Define the return from time \(t\) as

\[
G_{t} \;=\; \sum_{k=0}^{\infty} \gamma^{k}\,R_{t+k+1},
\]

where \(R_{t+k+1}\) is the reward at time \(t+k+1\). Because \(\gamma \in [0,1)\) and \(\bigl|R_{t+k+1}\bigr|\le R_{\max}\), the infinite series converges almost surely and satisfies

\[
\bigl|\!G_{t}\bigr|\;\le\;\sum_{k=0}^{\infty} \gamma^{k}\,R_{\max}
\;=\;\frac{R_{\max}}{1-\gamma}.
\]

Hence the state-value function

\[
v_{\pi}(s) \;=\; \mathbb{E}\bigl[G_{t}\,\big\vert\,S_{t}=s,\;\pi\bigr]
\]

is well-defined and bounded. Likewise, the action-value function

\[
q_{\pi}(s,a) \;=\;\mathbb{E}\bigl[G_{t}\,\big\vert\,S_{t}=s,\,A_{t}=a,\,\pi\bigr]
\]

is also bounded by \(\pm\,R_{\max}/(1-\gamma)\).

3 **Supremum Over All Policies**

For each state \(s\), define

\[
v_{*}(s)\;=\;\sup_{\pi}\,v_{\pi}(s).
\]

Similarly, for each state-action pair \((s,a)\), define

\[
q_{*}(s,a)\;=\;\sup_{\pi}\,q_{\pi}(s,a).
\]

Since \(v_{\pi}(s)\) and \(q_{\pi}(s,a)\) are bounded above by \(R_{\max}/(1-\gamma)\) and below by \(-\,R_{\max}/(1-\gamma)\), the suprema \(v_{*}(s)\) and \(q_{*}(s,a)\) are finite.

4 **Existence of an Optimal Policy**

Because \(\mathcal{S}\) and \(\mathcal{A}\) are finite, there are only finitely many deterministic policies: specifically \(\lvert\mathcal{A}\rvert^{\lvert\mathcal{S}\rvert}\). Consider the set

\[
\Bigl\{\;v_{\pi}\,\bigm|\,\pi \text{ is deterministic}\Bigr\}.
\]

Each \(v_{\pi}\) is a vector in \(\mathbb{R}^{\lvert\mathcal{S}\rvert}\). Hence the supremum over a finite set is actually a maximum, implying there exists at least one deterministic policy \(\pi^{*}\) such that

\[
v_{\pi^{*}}(s)\;=\;v_{*}(s)\quad \text{for all }s.
\]

Thus the supremum is in fact attained by at least one policy \(\pi^{*}\). An analogous argument holds for \(q_{*}(s,a)\).

5 **Conclusion**

Under these standard conditions:

1. For each policy \(\pi\), \(v_{\pi}\) and \(q_{\pi}\) exist and lie in \(\bigl[-\,R_{\max}/(1-\gamma),\;R_{\max}/(1-\gamma)\bigr]\).  
2. Taking the supremum over all policies yields finite quantities \(v_{*}(s)\) and \(q_{*}(s,a)\).  
3. Because only finitely many deterministic policies exist, the supremum is actually a maximum, and an optimal policy \(\pi^{*}\) attains those maximal values.  

Hence the definitions

\[
v_{*}(s)\;=\;\max_{\pi} v_{\pi}(s)
\quad\text{and}\quad
q_{*}(s,a)\;=\;\max_{\pi} q_{\pi}(s,a)
\]

are well-defined in such finite, discounted MDPs.

## Uniqueness of Optimal policies

***What about uniqueness of these optimal policies? Can there be two different optimal policies?***

Below is a step-by-step explanation of whether the optimal policy in a finite Markov Decision Process (MDP) with discounted rewards and bounded reward magnitudes must be unique. All mathematical expressions are written in standard LaTeX:

1) Existence of an Optimal Policy

In a finite discounted MDP with  \(\ 0 \le \gamma < 1 \)  
and bounded rewards, there exists at least one policy \(\ \pi^*\) that achieves the highest possible value in every state \(s\). Formally,  

\[
v_{*}(s) \;=\; v_{\pi^*}(s)
\]  

for all \(s\). Here, the state-value function for any policy \(\ \pi\) is  

\[
v_{\pi}(s) \;=\; \mathbb{E}\!\Bigl[\;\sum_{k=0}^{\infty} \gamma^k \; R_{t+k+1}\;\Bigm|\; S_t = s,\; \pi\Bigr],
\]  

and the optimal state-value function is defined by  

\[
v_{*}(s) \;=\; \max_{\pi}\; v_{\pi}(s).
\]

2) Multiple Optimal Policies

Even though an optimal policy \(\ \pi^*\) exists, it need not be unique. Multiple distinct policies can yield the same optimal value function \(v_{*}(s)\). Common reasons for this include:

• Two or more actions in a given state \(s\) might produce the exact same expected immediate reward plus the same transitions leading to the same future discounted returns.  
• In more complex scenarios, there may be different ways to behave optimally across one or more states, all yielding the same overall returns.

3) Ties in the Optimal Action-Value Function

Consider the optimal action-value function

\[
q_{*}(s,a) \;=\; \max_{\pi}\; q_{\pi}(s,a).
\]  

If, for some state \(s\) and two distinct actions \(a_1\neq a_2\), it holds that 

\[
q_{*}(s,a_1) \;=\; q_{*}(s,a_2),
\] 

then both actions \(a_1\) and \(a_2\) are equally good in state \(s\). Hence, any policy choosing either action in that state is still optimal. If multiple such ties occur across different states, this can lead to many distinct policies \(\ \pi^*\) that all achieve the same maximum returns.

4) Summary

• A finite discounted MDP with \(0 \le \gamma < 1\) and bounded rewards always has at least one optimal policy.  
• There can be more than one optimal policy; uniqueness is not guaranteed.  
• All optimal policies share the same optimal state-value function \(v_{*}(s)\) (and the same optimal action-value function \(q_{*}(s,a)\)).


## Bellman Optimality Operator I

The value function $v_{\pi}(s)$ is a function, $v_{\pi}(\cdot): \mathcal{S} \to \mathbb{R}$. In our MDP setting, this space is a space of bounded functions. Previously, we derived the Bellman Optimality Equation as:

\[
\boxed{
v_*(s)
\;=\;
\max_{a \,\in\, \mathcal{A}}\;
\sum_{s' \,\in\, \mathcal{S}}
P(s' \mid s,a)\;
\bigl[
  R(s,a,s')
  \;+\;\gamma\,v_*(s')
\bigr].
}
\]

Clarify how this results in the Bellman optimality operator $[\mathcal{T} v](\cdot)$. This operator acts on the space of bounded state value functions to what space? Explain this in a rigorous manner.

***Answer***

Below is a step-by-step explanation of how the Bellman Optimality Equation induces an operator (the Bellman Optimality Operator) on the space of bounded value functions. All mathematical symbols and expressions appear in standard LaTeX:

1) **Finite MDP and Space of Bounded Value Functions**

• Let the state space be finite: \(\mathcal{S} = \{\,s_{1},s_{2},\dots,s_{|\mathcal{S}|}\}\).  
• Consider the space of bounded real-valued functions on \(\mathcal{S}\), denoted by \(\mathcal{B}(\mathcal{S})\).  
Each element \(v \in \mathcal{B}(\mathcal{S})\) is a function 

\[
v: \;\mathcal{S}\;\to\;\mathbb{R} \quad  \text{with}  \quad
  \|\,v\,\|_{\infty}   \;=\;
  \sup_{s \in \mathcal{S}}\;\bigl|\,v(s)\bigr|
  \;<\;\infty.
\]  

• Because \(\mathcal{S}\) is finite, any real-valued function on \(\mathcal{S}\) is automatically bounded if its values are finite.

2) **Bellman Optimality Equation**

The Bellman Optimality Equation for the optimal value function \(v_{*}\) is:

\[
v_{*}(s)
\;=\;
\max_{a \,\in\, \mathcal{A}}
\sum_{s' \,\in\, \mathcal{S}}
P\bigl(s' \,\vert\, s,a\bigr)\,
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma\,v_{*}(s')
\Bigr],
\]

where  
• \(0 \le \gamma < 1\) is the discount factor,  
• \(R(s,a,s')\) is the reward for transitioning \(s \to s'\) via action \(a\), and  
• \(P(s' \mid s,a)\) is the transition probability.

3) **Defining the Bellman Optimality Operator \(\mathcal{T}\)**

We now define an operator \(\mathcal{T}\) that acts on any bounded value function \(v \in \mathcal{B}(\mathcal{S})\). For each state \(s\in \mathcal{S}\), set

\[
[\,\mathcal{T}v\,](s)
\;=\;
\max_{a \,\in\, \mathcal{A}}
\sum_{s' \,\in\, \mathcal{S}}
P\bigl(s' \,\vert\, s,a\bigr)\,
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma\,v(s')
\Bigr].
\]

Thus \(\mathcal{T}\) takes the function \(v\) and produces a new function \(\mathcal{T}v\). Concretely,

\[
\mathcal{T}:\;\mathcal{B}(\mathcal{S})\;\to\;\mathcal{B}(\mathcal{S}), 
\quad
v \;\mapsto\;\mathcal{T}v.
\]

4) **\(\mathcal{T}v\) Is Bounded**

• Suppose \(v \in \mathcal{B}(\mathcal{S})\), meaning that \(\|\,v\,\|_{\infty} = \sup_{s\in\mathcal{S}}|v(s)| < \infty\).  
• Assume rewards are bounded: \(\bigl|\,R(s,a,s')\bigr|\le R_{\max}\) for all \((s,a,s')\).  
• For each \(s\in \mathcal{S}\), the quantity \([\mathcal{T}v](s)\) is a finite maximum (since \(\mathcal{A}\) is finite) of finite sums of probabilities times \(\bigl(R(s,a,s') + \gamma\,v(s')\bigr)\).  
• Hence \([\mathcal{T}v](s)\) is finite, and collectively \(\|\mathcal{T}v\|_{\infty} < \infty\).  

Therefore, \(\mathcal{T}v\) lies in the same space \(\mathcal{B}(\mathcal{S})\). 

5) **How \(\mathcal{T}\) Relates to the Bellman Optimality Equation**

The Bellman Optimality Equation says that the optimal value function \(v_{*}\) must satisfy

\[
v_{*}(s)
\;=\;
[\mathcal{T}v_{*}](s)
\quad
\text{for each }s.
\]

In other words, \(v_{*}\) is a fixed point of the operator \(\mathcal{T}\). Symbolically,

\[
\mathcal{T}v_{*} = v_{*}.
\]

Moreover, in a discounted, finite MDP with bounded rewards, one can show (via a contraction mapping argument in the supremum norm) that \(\mathcal{T}\) has a unique fixed point in \(\mathcal{B}(\mathcal{S})\), namely \(v_{*}\).

6) **Conclusion**

• The space of bounded value functions on a finite state space is \(\mathcal{B}(\mathcal{S})\).  
• The Bellman Optimality Operator \(\mathcal{T}\) transforms any \(v\in\mathcal{B}(\mathcal{S})\) into another bounded function \(\mathcal{T}v\in\mathcal{B}(\mathcal{S})\) via

\[
[\mathcal{T}v](s) =
\max_{a}
\sum_{s'\in\mathcal{S}}
P(s'\mid s,a)\,\Bigl[
  R(s,a,s') + \gamma\,v(s')
\Bigr].
\]

• The Bellman Optimality Equation asserts that the unique optimal value function \(v_{*}\) is a fixed point of \(\mathcal{T}\).  

Hence, in rigorous terms, \(\mathcal{T}\) is well-defined as an operator (\(\mathcal{T}:\mathcal{B}(\mathcal{S})\to \mathcal{B}(\mathcal{S})\)) that captures “take the maximum over actions of expected reward plus discounted value,” exactly as described by the Bellman Optimality Equation.

## Bellman Optimality Operator II

This makes sense. I need clarification on one point in your answer. The Bellman Optimality operator is defined as

\[
[\,\mathcal{T}v\,](s)
\;=\;
\max_{a \,\in\, \mathcal{A}}
\sum_{s' \,\in\, \mathcal{S}}
P\bigl(s' \,\vert\, s,a\bigr)\,
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma\,v(s')
\Bigr].
\]

I need a more detailed explanation to see why the following is true.

\[
\mathcal{T}:\;\mathcal{B}(\mathcal{S})\;\to\;\mathcal{B}(\mathcal{S}), 
\quad
v \;\mapsto\;\mathcal{T}v.
\]

***Answer***

Below is a step-by-step clarification of why the Bellman Optimality Operator, often denoted by \(\mathcal{T}\), is defined as:

\[
[\mathcal{T}v](s)
\;=\;
\max_{a \,\in\, \mathcal{A}}
\;\sum_{s' \,\in\, \mathcal{S}}
P\bigl(s' \,\vert\, s,a\bigr)\,
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma\,v(s')
\Bigr],
\]

and how it acts from the space of bounded value functions \(\mathcal{B}(\mathcal{S})\) to itself.

1) **Context: Value Functions and Optimality**

• In a Markov Decision Process (MDP), the state-value function for a policy \(\pi\), denoted \(v_{\pi}(s)\), embodies the expected return (sum of discounted rewards) starting from state \(s\).  
• The “optimal” value function \(v_{*}\) solves the Bellman Optimality Equation:

\[
  v_{*}(s)  \;=\;   \max_{a \in \mathcal{A}}   \sum_{s' \in \mathcal{S}}
  P(s' \mid s,a)\,\Bigl[R(s,a,s') + \gamma\,v_{*}(s')\Bigr].
\]  

• This essentially says: from state \(s\), pick the action \(a\) that maximizes the immediate expected reward plus the discounted future.  

2) **One-Step Lookahead + Max**

• Observe that the right-hand side of the above equation is a function of \(v_{*}\).  
• If we do not yet know \(v_{*}\), but have some candidate bounded value function \(v\), we can \emph{approximate} what it means to “pick the best action based on \(v\),” by replacing \(v_{*}(s')\) with \(v(s')\).  
• Consequently, to get a new function \(\bigl[\mathcal{T}v\bigr]\)(s), we do exactly

\[
\max_{a \in \mathcal{A}}
\sum_{s' \in \mathcal{S}}
P(s' \mid s,a)\,
\Bigl[
  R(s,a,s') + \gamma\,v(s')
\Bigr].
\]

This procedure is “look one step ahead, then add discounted future as given by \(v\), and choose the action yielding the maximal value.”

3) **Definition of \(\mathcal{T}\): Acting on Bounded Value Functions**

Let \(\mathcal{B}(\mathcal{S})\) be the vector space of bounded real-valued functions over a finite state set \(\mathcal{S}\). Formally,

\[
v \in \mathcal{B}(\mathcal{S}) \quad\Longleftrightarrow\quad v: \mathcal{S} \to \mathbb{R},
\;\text{with}\;
\|v\|_{\infty} = \sup_{s \in \mathcal{S}}
\bigl|v(s)\bigr|
<
\infty.
\]

We define the Bellman Optimality Operator \(\mathcal{T}:\mathcal{B}(\mathcal{S})\to \mathcal{B}(\mathcal{S})\) by specifying its action on each \(v\in\mathcal{B}(\mathcal{S})\) and each state \(s\in\mathcal{S}\):

\[
\bigl[\mathcal{T}v\bigr](s)
\;=\;
\max_{a \,\in\, \mathcal{A}}
\sum_{s' \,\in\, \mathcal{S}}
P\bigl(s'\mid s,a\bigr)\;
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma\,v(s')
\Bigr].
\]

4) **\(\mathcal{T}v\) Remains Bounded**

• Because \(\|v\|_{\infty}\) is finite and the immediate rewards \(R(s,a,s')\) are assumed bounded, each summation over \(s'\) is finite.  
• There are finitely many actions \(a\), so the \(\max_{a \in \mathcal{A}}\) expression is finite as well.  
• Hence \(\|\,\mathcal{T}v\,\|_{\infty} < \infty\).  
• In other words, \(\mathcal{T}v\) is itself in \(\mathcal{B}(\mathcal{S})\). Thus \(\mathcal{T}\) indeed maps bounded functions to bounded functions:

\[
\mathcal{T}:\;\mathcal{B}(\mathcal{S})
\;\to\;
\mathcal{B}(\mathcal{S}).
\]

5) **Connection to the Bellman Optimality Equation**

• The Bellman Optimality Equation states that \(v_{*}\) is a fixed point of \(\mathcal{T}\), i.e.,

\[
v_{*}(s)
\;=\;
[\mathcal{T}v_{*}](s)
\quad\text{for all }s.
\]

• In short, if you plug \(v_{*}\) into \(\mathcal{T}\), you get the same function \(v_{*}\) back.  
• Under standard assumptions (finite \(\mathcal{S}, \mathcal{A}\), \(\gamma < 1\), bounded rewards), this fixed point is unique, and repeated application of \(\mathcal{T}\) (starting from \emph{any} bounded function \(v^{(0)}\)) converges to \(v_{*}\) in the supremum norm.

6) **Summary**

Given a bounded function \(v\), \(\bigl[\mathcal{T}v\bigr](s)\) is obtained by taking the one-step expected reward (averaging over next states \(s'\) with probabilities \(P(s'\mid s,a)\)), adding the discounted future value \(\gamma\,v(s')\), and finally choosing the action \(a\) that maximizes this sum. Mathematically,

\[
[\mathcal{T}v](s) = \max_{a} \sum_{s'}
P(s'\mid s,a)
\bigl[
  R(s,a,s')
  +
  \gamma\,v(s')
\bigr].
\]

Hence \(\mathcal{T}\) is exactly the $\textit{optimality operator}$ that, when iterated, drives a value function \(v\) closer and closer to the unique optimal value function \(v_{*}\).

## Uniqueness revisited

***Provide a more mathematically rigorous explanation for the uniqueness question. It appears that optimal policies are not unique but the state value functions are.***

Below is a mathematically more rigorous explanation regarding the uniqueness or non‐uniqueness of optimal policies versus the uniqueness of the optimal value function in a finite, discounted Markov Decision Process (MDP) with bounded rewards.

1) **Setting: Discounted Finite MDPs**

Consider an MDP with:

• A finite state space:    \(\mathcal{S} = \{\,s_1,\dots,s_{|\mathcal{S}|}\}\).  
• A finite action space:    \(\mathcal{A} = \{\,a_1,\dots,a_{|\mathcal{A}|}\}\).  
• A transition kernel:    \(P(s' \mid s,a)\).  
• A bounded reward function:    \(\lvert R(s,a,s')\rvert \le R_{\max}\).  
• A discount factor:    \(0 \le \gamma < 1\).

Define the return from time \(t\) as

\[
G_t \;=\;\sum_{k=0}^{\infty}\,\gamma^k\,R_{t+k+1}.
\]

A policy \(\pi\) determines the probabilities of selecting actions in each state. Then

\[
v_{\pi}(s)
\;=\;
\mathbb{E}\Bigl[G_t \,\big\vert\, S_t=s,\;\pi\Bigr]
\quad\text{and}\quad
q_{\pi}(s,a)
\;=\;
\mathbb{E}\Bigl[G_t \,\big\vert\, S_t=s,\;A_t=a,\;\pi\Bigr].
\]

We define the optimal state- and action-value functions:

\[
v_{*}(s) 
\;=\;
\max_{\pi} \,v_{\pi}(s),
\quad
q_{*}(s,a)
\;=\;
\max_{\pi} \,q_{\pi}(s,a).
\]

2) **Uniqueness of the Optimal Value Function**

1 Banach Fixed Point / Contraction Mapping Argument:

   The Bellman optimality operator \(\mathcal{T}\), acting on a value function \(v\), is defined by

\[
\mathcal{T}v(s)    \;=\;   \max_{a\in\mathcal{A}}   \sum_{s'\in\mathcal{S}}
P(s'\mid s,a)\,\Bigl[R(s,a,s') + \gamma \,v(s')\Bigr].
\]

   One can show \(\mathcal{T}\) is a \(\gamma\)-contraction in the supremum norm \(\|\cdot\|_{\infty}\) over the space of bounded functions from \(\mathcal{S}\) to \(\mathbb{R}\). By the Banach Fixed Point Theorem, there exists a unique fixed point \(v_{*}\) of \(\mathcal{T}\) such that

\[
   \mathcal{T}v_{*} = v_{*}.
\]

   Moreover, for any initial bounded value function \(v^{(0)}\), iterating

\[
   v^{(k+1)} = \mathcal{T}v^{(k)}
\]

   converges to \(v_{*}\) in \(\|\cdot\|_{\infty}\).  

2 Equivalence to \(v_{*}(s)\):

   Because

\[
   v_{*}(s)
   \;=\;
   \max_{\pi}\,v_{\pi}(s),
\]

   and each \(v_{\pi}\) is a fixed point of the policy-evaluation operator with respect to \(\pi\), it follows that \(v_{*}\) is exactly the unique fixed point of \(\mathcal{T}\). Conclusively, the \textit{function} \(v_{*}\) is unique among all bounded functions.

3) **Why Multiple Optimal Policies Can Exist**

  1 Ties in the Maximization:

   Even though the operator \(\mathcal{T}v\)(s) uses a “\(\max_{a}\)” over actions, it is possible for multiple actions \(a\) to \textit{achieve} that maximum in a given state \(s\). Concretely, if

\[
   \sum_{s'}P(s'\mid s,a_{1})
   \bigl[R(s,a_{1},s') + \gamma\,v_{*}(s')\bigr]
   \;=\;
   \sum_{s'}P(s'\mid s,a_{2})
   \bigl[R(s,a_{2},s') + \gamma\,v_{*}(s')\bigr],
\]

   then both \(a_{1}\) and \(a_{2}\) give the same (optimal) value in state \(s\). This means, at state \(s\), you may choose either \(a_{1}\) or \(a_{2}\) (or any distribution over them) and still achieve the same optimal outcome.

2 Constructing Distinct Optimal Policies:

   If there is at least one state \(s\) where two or more actions tie for the maximum in the Bellman optimality equation, then we can define multiple deterministic policies by picking different “best” actions in that state—and possibly more states have such ties. Each of these policies achieves

\[
   v_{\pi^{*}}(s) \;=\; v_{*}(s)
   \quad
   \forall\,s.
\]

   Consequently, the $\textit{policy}$ is not unique.

4) **Same \(v_{*}\), Different \(\pi^{*}\):**

   Because all such policies pick (in each state \(s\)) $\textit{some}$ action that maximizes the quantity \(\sum_{s'}P(s'\mid s,a)[\,R + \gamma\,v_{*}\,]\), the resulting value function cannot exceed \(v_{*}(s)\). On the other hand, it cannot be less either, since that is the maximum. Hence each of these “tie-breaking” policies yields the same \(v_{*}\). This is exactly how multiple optimal policies arise, while the optimal value function remains unique.

5) **Conclusion**

   • In a finite discounted MDP, the Bellman optimality operator is a strict contraction (given \(0 \le \gamma < 1\)), ensuring the \textbf{optimal value function} \(v_{*}\) is \textbf{unique}.

   • However, $\textit{optimal policies}$—those that achieve \(v_{*}(s)\) for all \(s\)—can be $\textbf{non‐unique}$, precisely when multiple actions tie in the state-wise maximization. All such policies still yield the same unique \(v_{*}\).

Hence, the uniqueness conferred by the contraction mapping theorem refers to the $\textbf{value function}$, not necessarily the $\textbf{policy}$, explaining why there can be multiple distinct optimal policies yet only a single optimal value function.

## $\mathcal{T}$ is a contraction mapping

***Prove in a mathematically rigorous manner that $\mathcal{T}$ is a $\gamma$-contraction in the supremum norm $\|\cdot\|_{\infty}$.***

Below is a step-by-step proof that the Bellman Optimality Operator,  

\[
[\mathcal{T}v](s)
\;=\;
\max_{a\,\in\,\mathcal{A}}
\sum_{s'\,\in\,\mathcal{S}}
P(s'\mid s,a)\,
\Bigl[
  R(s,a,s')
  \;+\;\gamma\,v(s')
\Bigr],
\]  

is a \(\gamma\)-contraction mapping in the supremum (∞-)norm. That is, we prove:

\[
\|\mathcal{T}v \;-\;\mathcal{T}w\|_{\infty}
\;\le\;
\gamma\;\|v \;-\; w\|_{\infty}
\quad\text{for all}\;\;v,w\in\mathcal{B}(\mathcal{S}).
\]

Here, \(\|\,f\,\|_{\infty} = \sup_{s\in\mathcal{S}} |f(s)|\), and \(\mathcal{B}(\mathcal{S})\) is the space of all bounded functions on the finite state space \(\mathcal{S}\).


1 **The Operator \(\mathcal{T}\) in Detail**


Given any bounded function \(v:\mathcal{S}\to\mathbb{R}\), we define

\[
[\mathcal{T}v](s)
\;=\;
\max_{a\,\in\,\mathcal{A}}
\sum_{s'\,\in\,\mathcal{S}}
P(s'\,\mid\,s,a)
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma\,v(s')
\Bigr].
\]

Such an operator \(\mathcal{T}\) is called the Bellman Optimality Operator for a discounted MDP with discount factor \(0 \le \gamma < 1\). Our goal is to show

\[
\|\mathcal{T}v - \mathcal{T}w\|_{\infty}
\;\le\;
\gamma\;\|\,v-w\,\|_{\infty}
\quad
\forall\,
v,w \in \mathcal{B}(\mathcal{S}).
\]


2 **Examine \([\mathcal{T}v](s)\) Minus \([\mathcal{T}w](s)\)**


Fix any state \(s\in\mathcal{S}\). We have:

\[
[\mathcal{T}v](s)
\;=\;
\max_{a\in\mathcal{A}}
\sum_{s'\in\mathcal{S}}
P(s'\mid s,a)\;
\Bigl[
  R(s,a,s') + \gamma\,v(s')
\Bigr],
\]

\[
[\mathcal{T}w](s)
\;=\;
\max_{a\in\mathcal{A}}
\sum_{s'\in\mathcal{S}}
P(s'\mid s,a)\;
\Bigl[
  R(s,a,s') + \gamma\,w(s')
\Bigr].
\]

Denote

\[
F_{v}(s,a)
\;=\;
\sum_{s'\in\mathcal{S}}
P(s'\mid s,a)\,[\,R(s,a,s') + \gamma\,v(s')\,],
\]

so that \([\mathcal{T}v](s) = \max_{a\in\mathcal{A}} F_{v}(s,a)\). Similarly, \([\mathcal{T}w](s) = \max_{a\in\mathcal{A}} F_{w}(s,a)\). Then

\[
[\mathcal{T}v](s) - [\mathcal{T}w](s)
\;=\;
\max_{a\in\mathcal{A}}\,F_{v}(s,a)
\;-\;
\max_{a\in\mathcal{A}}\,F_{w}(s,a).
\]

3 **Bounding the Difference Using “max - max ≤ max of differences”**

Recall a known inequality for real numbers: \(\max_{x} f(x) \;-\;\max_{x} g(x) \;\le\;\max_{x}\,[f(x) - g(x)]\).  

Hence,

\[
\max_{a}\,F_{v}(s,a)
\;-\;
\max_{a}\,F_{w}(s,a)
\;\le\;
\max_{a}\,\bigl[F_{v}(s,a) - F_{w}(s,a)\bigr].
\]

Therefore,

\[
[\mathcal{T}v](s) - [\mathcal{T}w](s)
\;\le\;
\max_{a\in\mathcal{A}}
\Bigl\{
  \sum_{s'\in\mathcal{S}}
  P(s'\mid s,a)\,
  \bigl[\,\gamma\,v(s') - \gamma\,w(s')\,\bigr]
\Bigr\},
\]

since the reward part \(R(s,a,s')\) cancels out (it appears in both \(F_{v}\) and \(F_{w}\)):

\[
F_{v}(s,a) - F_{w}(s,a) =
\sum_{s'}P(s'\mid s,a)\,
\Bigl[
  \gamma\,v(s') - \gamma\,w(s')
\Bigr].
\]

4 **Factor Out \(\gamma\) and Sum of Probabilities**

Thus

\[
[\mathcal{T}v](s) - [\mathcal{T}w](s)
\;\le\;
\max_{a}\Bigl\{
  \gamma
  \sum_{s'}
  P(s'\mid s,a)\,\bigl[v(s') - w(s')\bigr]
\Bigr\}.
\]

Observing that \(\sum_{s'} P(s'\mid s,a) = 1\), we get

\[
[\mathcal{T}v](s) - [\mathcal{T}w](s)
\;\le\;
\gamma
\max_{a}
\sum_{s'\in\mathcal{S}}
P(s'\mid s,a)\,\bigl[v(s') - w(s')\bigr].
\]

Now, for each s′, the difference \(v(s') - w(s')\) satisfies:

\[
v(s') - w(s')
\;\le\;
\|v-w\|_{\infty}.
\]

Hence,

\[
\sum_{s'}
P(s'\mid s,a)\,\bigl[v(s') - w(s')\bigr]
\;\le\;
\sum_{s'} P(s'\mid s,a)\,\|v-w\|_{\infty}
\;=\;
\|\,v-w\,\|_{\infty}
\]

(because \(\sum_{s'}P(s'\mid s,a) = 1\)). Therefore,

\[
\max_{a}
\sum_{s'}
P(s'\mid s,a)\,\bigl[v(s') - w(s')\bigr]
\;\le\;
\|v-w\|_{\infty}.
\]

Hence

\[
[\mathcal{T}v](s) - [\mathcal{T}w](s)
\;\le\;
\gamma\;\|v-w\|_{\infty}.
\]

5 **Symmetry: The Same Bound with (v,w) Swapped**

We also need a “lower bound” so we can handle absolute values. By a symmetric argument, swapping the roles of v and w,

\[
[\mathcal{T}w](s) - [\mathcal{T}v](s)
\;\le\;
\gamma\;\|v-w\|_{\infty}.
\]

6 **Conclude the Contraction Property**

Combining both, we see that

\[
|[\mathcal{T}v](s) \;-\; [\mathcal{T}w](s)|
\;\le\;
\gamma\;\|v-w\|_{\infty}
\quad
\text{for each state }s.
\]

Hence taking the supremum over $s \in \mathcal{S}$ yields:

\[
\|\mathcal{T}v \;-\; \mathcal{T}w\|_{\infty} =
\sup_{s\in\mathcal{S}}
\,\bigl|
  [\mathcal{T}v](s)
  \;-\;
  [\mathcal{T}w](s)
\bigr|
\;\le\;
\gamma\;\|v-w\|_{\infty}.
\]

This is precisely the definition of a \(\gamma\)-contraction in the ∞-norm.

7 **Summary**

Thus, the Bellman Optimality Operator \(\mathcal{T}\) satisfies

\[
\|\mathcal{T}v \;-\; \mathcal{T}w\|_{\infty}
\;\le\;
\gamma\;\|v-w\|_{\infty},
\]

showing it is a \(\gamma\)-contraction mapping in the supremum norm. This property underpins why \(\mathcal{T}\) has a unique fixed point (the optimal value function \(v_{*}\)) in the Banach space \(\bigl(\mathcal{B}(\mathcal{S}),\|\cdot\|_{\infty}\bigr)\) and why repeated iteration of \(\mathcal{T}\) converges to \(v_{*}\).


## Bellman Optimality Operator - matrix vector form

***Just as a Bellman equation for a policy $\pi$ can be written in matrix-vector form, similarly, the Bellman optimality equation can also be written in matrix-vector form. Starting with the scalar form for the Bellman optimality equation for each state $s_i \in \mathcal{S}$, write down the matrix-vector form while clarifying the entries of the associated vectors and matrix. Show the connection of the Bellman Optimality operator $\mathcal{T}$ to the matrix-vector form.  Write down all mathematical symbols and expressions in full Latex.***

Below is a step‐by‐step derivation showing how the Bellman Optimality Equation, which is initially given in a scalar “per‐state” form, can also be written in a matrix‐vector form. All mathematical symbols and expressions are in LaTeX.

1 **Scalar (Per‐State) Form of the Bellman Optimality Equation**

Consider a finite Markov Decision Process (MDP) with:

• A finite state space \(\mathcal{S} = \{s_{1},\,s_{2},\dots,s_{|\mathcal{S}|}\}\).  
• A finite action space \(\mathcal{A} = \{a_{1},\,a_{2},\dots,a_{|\mathcal{A}|}\}\).  
• Transition probabilities \(P(s_{j}\mid s_{i},a)\).  
• A discount factor \(0 \le \gamma < 1\).  
• A bounded reward function \(R(s_{i},\,a,\,s_{j})\).

The Bellman Optimality Equation states that the optimal value function \(v_{*}\) satisfies, for each state \(s_{i}\in \mathcal{S}\),

\[
v_{*}(s_{i})
\;=\;
\max_{a\,\in\,\mathcal{A}}
\sum_{s_{j}\,\in\, \mathcal{S}}
P\bigl(s_{j}\,\vert\,s_{i},\,a\bigr)
\Bigl[
  R\bigl(s_{i},\,a,\,s_{j}\bigr)
  \;+\;
  \gamma\,v_{*}(s_{j})
\Bigr].
\]

2 **Defining Matrices and Vectors for Each Action**

To move toward a matrix‐vector representation, we introduce:

1. A transition matrix \(\mathbf{P}_{a} \in \mathbb{R}^{|\mathcal{S}|\times |\mathcal{S}|}\) associated with each action \(a\). Its entries are

\[
   [\mathbf{P}_{a}]_{i,j}  \;=\;
   P\bigl(s_{j}\,\vert\,s_{i},\,a\bigr).
\]

2. A reward vector \(\mathbf{r}_{a} \in \mathbb{R}^{|\mathcal{S}|}\) associated with each action \(a\). Its \(i\)-th entry captures the expected immediate reward upon taking action \(a\) in state \(s_{i}\):

\[
   [\mathbf{r}_{a}]_{\,i}    \;=\;
   \sum_{s_{j}\,\in\,\mathcal{S}}
   P\bigl(s_{j}\,\vert\,s_{i},\,a\bigr)\;
   R\bigl(s_{i},\,a,\,s_{j}\bigr).
\]

   Equivalently, one can keep the form \(\sum_{s_{j}}P(s_{j}\mid s_{i},a)\,R(\dots)\), or define a suitable immediate‐reward function that depends on \(\bigl(s_{i},a,s_{j}\bigr)\). Either way the end result is a vector of length \(|\mathcal{S}|\).

Hence for each \(a \in \mathcal{A}\), we have:

• \(\mathbf{P}_{a}\,v\) is a vector where the \(i\)-th component is \(\sum_{s_{j}} P(s_{j}\mid s_{i},a)\,v(s_{j})\).  
• \(\mathbf{r}_{a}\) is a vector whose \(i\)-th entry accounts for \(\sum_{s_{j}} P(s_{j}\mid s_{i},a)\,R(s_{i},a,s_{j})\).

3 **The Bellman Optimality Operator \(\mathcal{T}\) in Matrix‐Vector Notation**

Define an operator \(\mathcal{T}\) that takes any vector \(v \in \mathbb{R}^{|\mathcal{S}|}\) (viewed as a function \(v:\mathcal{S}\to \mathbb{R}\)) and returns another vector \(\mathcal{T}v\in \mathbb{R}^{|\mathcal{S}|}\). The \(i\)-th component of \(\mathcal{T}v\) is:

\[
[\mathcal{T}v](s_{i})
\;=\;
\max_{a\,\in\,\mathcal{A}}
\Bigl(
  [\mathbf{r}_{a}]_{\,i}
  \;+\;
  \gamma\,[\mathbf{P}_{a}\,v]_{\,i}
\Bigr).
\]

Concretely, each action \(a\) suggests the vector

\[
\mathbf{r}_{a}
\;+\;
\gamma\,\mathbf{P}_{a}\,v
\quad\in\quad
\mathbb{R}^{|\mathcal{S}|}.
\]

We then take a coordinate‐wise maximum over \(a\). That is, for each \(i\),

\[
[\mathcal{T}v]_{i}
\;=\;
\max_{a\in\mathcal{A}}
\Bigl(
  [\mathbf{r}_{a}]_{\,i}
  \;+\;
  \gamma\,[\mathbf{P}_{a}\,v]_{\,i}
\Bigr).
\]

4 **Writing \(v_{*}\) = \(\mathcal{T}v_{*}\) in a Matrix‐Vector Way**

The Bellman Optimality Equation for the optimal value function \(v_{*}\) is equivalently the statement that

\[
v_{*}
\;=\;
\mathcal{T}\,v_{*}.
\]

Component‐wise:

\[
[v_{*}](s_{i})
\;=\;
[\mathcal{T}v_{*}](s_{i})
\;=\;
\max_{a\in\mathcal{A}}
\Bigl(
  [\mathbf{r}_{a}]_{i}
  \;+\;
  \gamma\,[\mathbf{P}_{a}\,v_{*}]_{i}
\Bigr).
\]

This matches exactly the scalar form

\[
v_{*}(s_{i})
\;=\;
\max_{a\,\in\,\mathcal{A}}
\sum_{s_{j}}
P(s_{j}\,\vert\,s_{i},a)
\,\Bigl(R(s_{i},a,s_{j}) + \gamma\,v_{*}(s_{j})\Bigr).
\]

5 **Observations**

1. Unlike the policy‐evaluation case (a single \(\mathbf{P}_{\pi}\) and \(\mathbf{r}_{\pi}\)), here we have one transition matrix \(\mathbf{P}_{a}\) and one reward vector \(\mathbf{r}_{a}\) for each action \(a\).  
2. The operator \(\mathcal{T}\) is nonlinear because of the coordinate‐wise maximum over actions. Consequently, one cannot write a single “matrix equation” like \(\mathbf{v} = \mathbf{r} + \gamma\,\mathbf{P}\,\mathbf{v}\) for the optimal scenario. Instead, the “matrix‐vector” expression is:

\[
\mathcal{T}\,v \;=\;
\max_{a \,\in\, \mathcal{A}}
\bigl(
  \mathbf{r}_{a}
  \;+\;
  \gamma\,\mathbf{P}_{a}\,v
\bigr)
\]

(where “\(\max\)” is applied coordinate‐wise).

3. The Bellman Optimality Equation then says \(v_{*}\) is the unique (under standard conditions) fixed point of \(\mathcal{T}\), i.e.,

\[
v_{*} =
\mathcal{T}\,v_{*}.
\]

6 **Summary**

• The scalar equation

\[
v_{*}(s_{i}) =
\max_{a\,\in\,\mathcal{A}}
\sum_{s_{j}\,\in\,\mathcal{S}}
P(s_{j}\,\vert\,s_{i},a)\,\Bigl[
  R(s_{i},\,a,\,s_{j})
  +
  \gamma\,v_{*}(s_{j})
\Bigr]
\]

can be interpreted in matrix‐vector notation by defining one transition matrix \(\mathbf{P}_{a}\) and one expected‐reward vector \(\mathbf{r}_{a}\) per action \(a\).  

• The Bellman Optimality Operator \(\mathcal{T}\) then “picks” for each state the best action’s vector

\[
\mathbf{r}_{a}
+
\gamma\,\mathbf{P}_{a}\,v,
\]

yielding

\[
[\mathcal{T}v](s_{i}) =
\max_{a\in \mathcal{A}}
\Bigl(
  [\mathbf{r}_{a}]_{\,i}
  \;+\;
  \gamma\,[\mathbf{P}_{a}\,v]_{\,i}
\Bigr).
\]

• The Bellman optimality equation succinctly becomes \(v_{*} = \mathcal{T}\,v_{*}\), capturing the same idea as the scalar form but in a higher‐level matrix‐vector operator sense.


## Policy Evaluation Operator $\mathcal{T}_{\pi}$

The Bellman Optimality operator is:

\[
[\mathcal{T}v](s)
\;=\;
\max_{a\,\in\,\mathcal{A}}
\sum_{s'\,\in\,\mathcal{S}}
P(s'\mid s,a)\,
\Bigl[
  R(s,a,s')
  \;+\;\gamma\,v(s')
\Bigr]
\]

For a given policy $\pi$ we have the Bellman equation:

\[
v_{\pi}(s)
\;=\;
\sum_{a \,\in\, \mathcal{A}}
\pi(a \,\vert\, s)
\sum_{s' \,\in\, \mathcal{S}}
P(s' \,\vert\, s,a)\,
\Bigl[
  R(s,a,s') 
  \;+\;
  \gamma\,v_{\pi}(s')
\Bigr]
\]

Noting the similarities, is there a concept of a “policy evaluation operator” for a given policy $\pi$?

**Answer**

Below is a step-by-step explanation showing that, in parallel to the Bellman Optimality operator, one can define a “policy evaluation operator” for a given policy \(\pi\):

1) **Bellman Optimality Operator**

The Bellman Optimality operator, often denoted by \(\mathcal{T}\), acts on any candidate value function \(v\) to produce another value function \(\mathcal{T}v\). For each state \(s\):

\[
[\mathcal{T}v](s)
\;=\;
\max_{a \,\in\,\mathcal{A}}
\sum_{s' \,\in\,\mathcal{S}}
P\bigl(s' \mid s,\,a\bigr)
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma\,v(s')
\Bigr].
\]

This operator considers all actions \(a\), chooses the best one (via \(\max_{a}\)), and hence is used for finding or expressing the value function of an optimal policy.

2) **Bellman Equation Under a Fixed Policy \(\pi\)**

If we fix a policy \(\pi\), meaning that actions \(a\) are chosen with probability \(\pi(a \mid s)\) in each state \(s\), then the Bellman equation for the value function \(v_{\pi}\) is:

\[
v_{\pi}(s)
\;=\;
\sum_{a \,\in\,\mathcal{A}}
\pi\bigl(a \mid s\bigr)
\sum_{s' \,\in\,\mathcal{S}}
P\bigl(s' \mid s,a\bigr)
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma\,v_{\pi}(s')
\Bigr].
\]

This is sometimes referred to as the Bellman Expectation Equation or the policy evaluation equation.

3) **Defining a Policy Evaluation Operator \(\mathcal{T}_{\pi}\)**

Analogous to how the Bellman Optimality operator \(\mathcal{T}\) uses a \(\max\) over actions, we can define an operator that uses the distribution \(\pi(a \mid s)\) instead of taking a maximum. Concretely, for any bounded value function \(v\), define

\[
\bigl[\mathcal{T}_{\pi}v\bigr](s)
\;=\;
\sum_{a \,\in\,\mathcal{A}}
\pi\bigl(a \mid s\bigr)
\sum_{s' \,\in\,\mathcal{S}}
P\bigl(s' \mid s,a\bigr)
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma\,v(s')
\Bigr].
\]

Hence, \(\mathcal{T}_{\pi}\) replaces the “\(\max\)” with an expectation under \(\pi(a \mid s)\). This operator \(\mathcal{T}_{\pi}\) is often called the Bellman Expectation operator (or simply the policy evaluation operator) for the policy \(\pi\).

4) **Fixed Point: \(\;v_{\pi} = \mathcal{T}_{\pi} v_{\pi}\)**

The Bellman equation for \(v_{\pi}\) can be rewritten as

\[
v_{\pi}(s)
\;=\;
\bigl[\mathcal{T}_{\pi}v_{\pi}\bigr](s),
\qquad
\forall\,s \in \mathcal{S}.
\]

That is, \(v_{\pi}\) is a fixed point of the operator \(\mathcal{T}_{\pi}\). Under standard assumptions (finite state space, \(0 \le \gamma < 1\), bounded rewards), one can show:

\[
\|\mathcal{T}_{\pi}v - \mathcal{T}_{\pi}w\|_{\infty}
\;\le\;
\gamma\;\|v-w\|_{\infty},
\]

making \(\mathcal{T}_{\pi}\) a \(\gamma\)-contraction in the supremum norm. It follows from the Banach Fixed Point Theorem that repeated application of \(\mathcal{T}_{\pi}\) on any initial \(v\) converges to \(v_{\pi}\), the unique fixed point.

5) **Summary**

Yes, there is indeed a “policy evaluation operator” for a given policy \(\pi\). One defines

\[
\bigl[\mathcal{T}_{\pi}v\bigr](s)
\;=\;
\sum_{a \,\in\,\mathcal{A}}
\pi(a \mid s)
\sum_{s' \,\in\,\mathcal{S}}
P(s' \mid s,a)
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma\,v(s')
\Bigr],
\]

and notes that \(\mathcal{T}_{\pi}\) maps bounded value functions to bounded value functions. The policy’s value function \(v_{\pi}\) is precisely the fixed point of \(\mathcal{T}_{\pi}\). This operator is a fundamental tool in Reinforcement Learning, where it is used for computing (or approximating) \(v_{\pi}\) for any given policy \(\pi\).



## Value Interation Algorithm

For the setting of a finite MDP, the Bellman optimality operator $\mathcal{T}$ has been shown to be a $\gamma$-contraction in the supremum norm. This suggests that we can use this fact to find the optimal state value $v_*(\cdot)$ by iteration. Let $k$ be the "time step" index. Let$v_0$ be the initial condition and suppose the data for the MDP - $\mathcal{S}, \mathcal{A}, P, \pi, \gamma$ are all provided. Write a step by step algorithm to find $v_*$.

**Answer**

Below is a step‐by‐step algorithm, commonly known as “Value Iteration,” that uses the fact that the Bellman Optimality operator $\mathcal{T}$ is a $\gamma$-contraction  in the supremum norm to find the optimal state‐value function $v_*$.

All mathematical symbols and expressions are presented in standard LaTeX, and we assume a finite MDP with:  
• A finite state space \(\mathcal{S} = \{s_{1},\dots,s_{|\mathcal{S}|}\}\).  
• A finite action space \(\mathcal{A} = \{a_{1},\dots,a_{|\mathcal{A}|}\}\).  
• Transition kernel \(P(s' \mid s,a)\) for each \((s,a)\).  
• Reward function \(R(s,a,s')\).  
• Discount factor \(0 \le \gamma < 1\).  

1) **Notation: The Bellman Optimality Operator**

Recall that the Bellman Optimality operator \(\mathcal{T}\) acts on a value function \(v:\mathcal{S}\to \mathbb{R}\) as follows:

\[
[\mathcal{T}v](s)
\;=\;
\max_{a \in \mathcal{A}}
\;\sum_{s' \in \mathcal{S}}
P\bigl(s' \mid s,a\bigr)\,
\Bigl[
  R(s,a,s')
  \;+\;
  \gamma\,v(s')
\Bigr].
\]

Since \(\mathcal{T}\) is a \(\gamma\)‐contraction in the \(\|\cdot\|_{\infty}\) norm (the supremum norm), repeated application of \(\mathcal{T}\) to any initial function \(v_{0}\) will converge to the unique fixed point \(v_{*}\). This fixed point is the optimal value function for the MDP.

2) **Value Iteration Algorithm**

◾ ***Input:***
(1) The set of states \(\mathcal{S}\) (finite).  
(2) The set of actions \(\mathcal{A}\) (finite).  
(3) Transition probabilities \(P(s' \mid s,a)\).  
(4) Reward function \(R(s,a,s')\).  
(5) Discount factor \(\gamma \in [0,1]\).  
(6) An initial value function \(v_{0}:\mathcal{S}\to\mathbb{R}\). Frequently \(v_{0}\) is just the zero function or arbitrary values.

◾ ***Procedure:***

(1) Initialize \(k=0\).  
(2) For each state \(s \in \mathcal{S}\), set \(v_{k}(s)\) to some initial value (e.g., 0).  
(3) Repeat (Value Iteration Step):  
   (a) For each state \(s\in \mathcal{S}\), compute

\[
   v_{k+1}(s)    \;=\;
   [\mathcal{T}v_{k}](s)    \;=\;    \max_{a\in \mathcal{A}}
   \sum_{s'\in \mathcal{S}}
   P(s'\mid s,a)    \Bigl[  R(s,a,s') + \gamma\,v_{k}(s')    \Bigr].
\]

   (b) Check for convergence in the supremum norm or any other criterion. For instance, one common stopping rule is:

\[
   \|\,v_{k+1} - v_{k}\,\|_{\infty} \;=\;    \sup_{s\in \mathcal{S}}
   \bigl|\,v_{k+1}(s) - v_{k}(s)\bigr|
   \;\le\;\varepsilon(1 - \gamma)/\gamma,
\]

   or use an iteration limit, or other threshold to decide when to stop.

(4) If not converged, increment \(k \leftarrow k+1\) and go back to step (3).  
(5) Once convergence is detected (or after sufficient iterations), the current \(v_{k}\) is (approximately) the optimal value function \(v_{*}\).  

3) **Output and Convergence**

• Output: \(v_{*}\approx v_{k}\), the converged state‐value function.  
• By the \(\gamma\)‐contraction property of \(\mathcal{T}\), we are guaranteed that \(v_{k}\) converges to the unique fixed point \(v_{*}\), satisfying

\[
v_{*}(s) =
[\mathcal{T}v_{*}](s) =
\max_{a}
\sum_{s'}
P(s'\mid s,a)\,\bigl[R(s,a,s') + \gamma\,v_{*}(s')\bigr].
\]

This \(v_{*}\) is the optimal state‐value function for the MDP.

4) **Final Algorithm Summary**

Algorithm: Value Iteration

1) Initialize \(v_{0}\) (e.g., to zeros).  
2) For \(k=0,1,2,\dots\):  
   • For each \(s\in\mathcal{S}\):

\[
   v_{k+1}(s)
   = \max_{a \in\mathcal{A}}
     \sum_{s'\in \mathcal{S}}
     P(s'\mid s,a)\,\Bigl[
       R(s,a,s')
       + \gamma\,v_{k}(s')
     \Bigr].
\]

3) If \(\|v_{k+1} - v_{k}\|_{\infty}\le\delta\) or \(k\) is sufficiently large, terminate.  
4) Return \(v_{k+1}\) as the approximate (or exact) optimal value function \(v_{*}\).

This completes the step‐by‐step algorithm for finding \(v_{*}\) by iterating the Bellman Optimality operator.

## From optimal value to optimal policy

Once the optimal $v_*$ is known, how to recover the optimal policy or policies recognizing the non-uniqueness of the optimal policy?

**Answer**

Below is a step-by-step explanation of how to recover at least one optimal policy (or multiple) after having computed the optimal value function $v_*(\cdot)$. All mathematical symbols and expressions are written in LaTeX.

1) **Setting: Known Optimal Value Function $v_*(\cdot)$**

Assume we have a finite MDP with:
• Finite state space \(\mathcal{S} = \{s_1, \dots, s_{|\mathcal{S}|}\}\).  
• Finite action space \(\mathcal{A}\).  
• Transition probabilities \(P(s' \mid s, a)\).  
• Reward function \(R(s, a, s')\).  
• Discount factor \(0 \le \gamma < 1\).  

We assume the optimal state-value function \(v_{*}(s)\) is known (e.g., from Value Iteration, Policy Iteration, or another method). Recall that \(v_{*}\) satisfies:

\[
v_{*}(s)
\;=\;
\max_{a \,\in\, \mathcal{A}}
\sum_{s' \,\in\, \mathcal{S}}
P(s' \,\vert\, s,a)
\Bigl[
  R(s,a,s') + \gamma\,v_{*}(s')
\Bigr].
\]

2) **Recovering an Optimal Policy: Basic Idea**

The Bellman Optimality equation indicates that for each state \(s\), an optimal action \(a^{*}\) achieves that maximum:

\[
a^{*}
\;\in\;
\arg\max_{a \in \mathcal{A}}
\sum_{s'}
P(s' \mid s,a)\,\Bigl[R(s,a,s') + \gamma\,v_{*}(s')\Bigr].
\]

Hence, once \(v_{*}\) is known, the “optimal” action(s) at each state \(s\) can be found by evaluating the right-hand side for every action \(a\in\mathcal{A}\) and picking an action that achieves the maximum.  

3) **Non-Uniqueness: Potential Ties**

If there is more than one action \(a\) that attains the same maximal value in state \(s\), then all such actions are equally good. That is,

\[
\sum_{s'}
P(s' \,\vert\, s,a_{1})
\Bigl[
  R(s,a_{1},s') + \gamma\,v_{*}(s')
\Bigr]
\;=\;
\sum_{s'}
P(s' \,\vert\, s,a_{2})
\Bigl[
  R(s,a_{2},s') + \gamma\,v_{*}(s')
\Bigr],
\]

and both equal \(\max_{a}\sum_{s'}[\dots]\). Consequently, there may be multiple different actions that are all optimal in state \(s\).  

4) **Constructing a Deterministic Policy**

A standard procedure:

• For each state \(s \in \mathcal{S}\), compute

\[
Q_{*}(s,a) =
\sum_{s'}
P(s' \mid s,a)\,
\Bigl[
  R(s,a,s') + \gamma\,v_{*}(s')
\Bigr]
\]

(This is the state-action “optimal” Q-value if we view \(v_{*}\) as the correct continuation. Technically $Q_*(s,a)  = R(s,a,s') + \gamma v_*(s')$, but once $v_*$ is known, we can plug it in to get $Q_*(\cdot)$.)

• Pick any action \(a^{*}(s)\) that satisfies

\[
a^{*}(s) \in \arg\max_{a\in\mathcal{A}}\;Q_{*}(s,a).
\]

• Define the deterministic policy \(\pi_{*}\) by

\[
\pi_{*}(s) = a^{*}(s).
\]

5) **Constructing a Stochastic (Mixed) Policy**

In states where multiple actions tie for the maximum, one could choose a stochastic policy that randomizes among all such tying actions. Formally:

• Let \(\mathcal{A}_{\text{opt}}(s)\) = { all a ∈ 𝓐 that achieve the maximum at s }.  
• Define

\[
\pi_{*}(a \mid s) =
\begin{cases}
\frac{1}{|\mathcal{A}_{\text{opt}}(s)|}, & a \in \mathcal{A}_{\text{opt}}(s),\\
0, & \text{otherwise}.
\end{cases}
\]

All such policies are also optimal because each action in \(\mathcal{A}_{\text{opt}}(s)\) yields the same value.  

6) **Summary: Recovering an Optimal Policy (or Policies)**

1. Once \(v_{*}\) is known, compute for each state-action pair \((s,a)\) the quantity

\[
Q_{*}(s,a) =
\sum_{s'}
P(s' \mid s,a)\,
\bigl[
  R(s,a,s') + \gamma\,v_{*}(s')
\bigr].
\]

2. For each state \(s\), find the set of actions that maximize $Q_{*}(s,a)$. This set may contain one action or multiple.  

3. To get a deterministic policy, pick any single maximizing action in each state. If multiple actions are ties, choose an arbitrary one among them.  

4. To get a stochastic policy, assign positive probabilities among all of the tying actions in \(\mathcal{A}_{\text{opt}}(s)\).  

All such resulting policies are optimal and yield the optimal value $v_*(\cdot)$ for the MDP — reflecting the potential non‐uniqueness of optimal policies in a finite MDP.