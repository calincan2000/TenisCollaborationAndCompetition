### Tennis - reinforcement learning PPO implementation

![Tenis from https://unity3d.com/machine-learning](https://github.com/calincan2000/TenisCollaborationAndCompetition/blob/master/image2-2.gif)

TenisCollaborationAndCompetition is a reinforcement learning algorithm based Multi-agent actor-critic Deep Deterministic Policy Gradient MADDPG algorithm. The environment is provided by Udacity as a part of deep reinfocement learning nanodegree. In this environment, two agents are playing [tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) with aim to reach average score of 0.5 over 100 episodes taking maximum score of the episode.



In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


### Installation

1. Download the environment from one of the links below. You need only select the environment that matches your operating system:

* Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip) </br>
* Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip) </br>
* Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip) </br>
* Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip) </br>

2. Place the file in the directory of GitHub repository files.

### Run the code in the repository 
To train the agent run all the cells in Report.ipynb notebook. For more information on how to run the environment please read the Dependencies section in this [repo](https://github.com/udacity/deep-reinforcement-learning#dependencies).


