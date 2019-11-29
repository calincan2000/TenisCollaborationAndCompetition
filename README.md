### Tennis - reinforcement learning PPO implementation
============

TenisCollaborationAndCompetition is a reinforcement learning algorithm based Multi-agent actor-critic MADDPG algorithm. The environment is provided by Udacity as a part of deep reinfocement learning nanodegree. In this environment, two agents are playing tennis with aim to reach average score of 0.5 over 100 episodes taking maximum score of the episode.

History
The simpliest approach is to make this project using DDPG but I wanted to challenge myself and I have chosen PPO. One of the reasons for choosing PPO is that Amazon AWS Deep Racer has been using PPO as a main reinforcement learning algortihm to run environment. And this choice is not a mistake at all. I enjoyed it very much as it produces very stable training result.

I looked at the repos with implementation of tennis environment and I mentioned that there is not much to choose from. I looked at the repos with multi-agent reinforcement learning and the same result. There is not much to choose from. Very complicated project.

I rewatched the course and added to my bookmarks ShangtongZhang 2017 and Reacher environment by Jeremi Kaczmarczyk 2018. I watched these repos, got an idea and reimplemented from scratch as was suggested by Udacity.

Happily it is working for two agents.

Features
You can change:

hidden neurons size
discount rate
TAU factor
gradient clip
rollout length
learning rate of an optimizer
optimizer epsilon
number of optimization epochs
clipping PPO value
entropy coefficient
mini batch number
Screenshot
Taken from real training. Tennis environment in action. Close to the target: Tennis - solving environment

Success! Tennis - success

Rewards
Tennid - rewards

Setup
Clone this repo:
git clone https://github.com/andreiliphd/tennis-ppo.git
Create and activate a new environment with Python 3.6.
conda create --name drlnd python=3.6
conda activate drlnd
Install PyTorch 0.4.0:
conda install pytorch=0.4.0 -c pytorch
Clone the Udacity repository, and navigate to the python/ folder. Then, install dependencies.
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
Create an IPython kernel for the drlnd environment.
python -m ipykernel install --user --name drlnd --display-name "drlnd"
Before running code in a notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.
Change Kernel

The rest of dependencies come with Conda environment by default.

In order to run training in a root of Github repositary run:

python solution.py
You can see that environment and training started.

Installation
Download the environment from one of the links below. You need only select the environment that matches your operating system:

Linux: click here
Mac OSX: click here
Windows (32-bit): click here
Windows (64-bit): click here
(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

Place the file in the directory of GitHub repository files.

Usage
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

