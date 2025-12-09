# Meam 5170 Final project: Robogrammar + BRAX

adaptive algorithm paper futuristic looking:
https://www.nature.com/articles/s42256-021-00320-3

Evolutionary thing DERL spider
https://arxiv.org/pdf/2102.02202 

Another evolutionary tower:
https://direct.mit.edu/artl/article/23/2/169/2866/Evolutionary-Developmental-Robotics-Improving 

Evolutionary graph approach with ppo:
https://arxiv.org/pdf/1906.05370 

Spider robot basic rl leg lengths:
https://arxiv.org/pdf/1810.03779 

RL with nested PPO leg lengths:
https://arxiv.org/pdf/1801.01432 

End to end differentiable for arm:
https://people.csail.mit.edu/jiex/papers/DiffHand/paper.pdf

Graph grammar:
https://cdfg.mit.edu/assets/files/robogrammar.pdf 

Tradeoff between morphology and control bayesian optimization:
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0186107 

Comparison of methods:
https://arxiv.org/html/2409.08621v1#S1.F1 

List of environments to add:
high v low friction

smooth v bumpy

rolly vs flat vs steps

walls vs no walls

hole vs no holes

robo school rewards

steps: 
Project idea - implement graph grammar paper on system involving links of the same length
Make short list of environments and rewards or get this list in mujoco
implement mpc rather than mppi algorithm for location of nodes 
implement robogrammar’s gnn for graph heuristic search
Make it work for many environments - find min across the environments

Problems: The original goal of this project was to use gradient-based methods in the MJX simulator to jointly optimize both the model parameters and the control policy. However, we found that gradient-based shooting methods were highly unstable, even with very small learning rates and aggressive gradient clipping. Due to time constraints, we shifted the project scope to focus on comparing different optimization methods for a quadrupedal walker. We implemented MPC, shooting, shooting with L-BFGS-B, and MPPI in JAX, all operating on an implicit model representation (i.e., without requiring an explicit model definition). MPPI corresponds to the algorithm used in the original paper. We evaluated these methods on a simple quadruped model that was initially intended only for testing. In future work, we aim to incorporate the RoboGrammar framework and extend it by using gradients to optimize morphology and model parameters. For now, our results are limited to a comparison of optimization methods using the implicit MJX model.

Results:

Implementation details - github link
In mujoco/mjx/_src/solver in line 602 changed
ctx = jax.lax._while_loop(cond, body, ctx) 
To
ctx = _while-loop_scan(cond, body, ctx, m.opt.iterations)

Discuss config lists (all parameter choices)
Three controller methods on simple spider robot
MPC
Video of walking
Computation time
MPPI
Video of walking
Computation time
Shooting
Video of walking
Computation time
Graph of costs to prove it does diverge
L-BFGS-B Shooting
Video of walking
Computation time







