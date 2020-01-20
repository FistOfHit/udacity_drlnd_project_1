# Project report

Note: Code was adapted from working in the provided workspace to working on local, but wasnt tested on local. If in doubt, just run it in the workspace and it should be fine, provided you change the banaa path back to normal and add the python installation line (`!pip -q install ./python`)

## Learning algorithm

The algorithm we use to approximate the action-value function is a Double Deep Q-Network (DDQN) with prioritised experience replay (and fixed Q-targets, implemented by udactiy team)

The DDQN itself is 4 layers deep with:
37 - > 111 -> 74 -> 4

Environment/State parameters:
- Max. steps per episode: 1000
- Max number of episodes: 2000

Q-Learning parameters:
- Initial epsilion: 1.0
- Final epsilion: 0.01
- Epsilion decay: 0.99

DQNN parameters:
- Batch size: 64
- Learning rate: 7e-4
- Tau: 1e-3
- gamma: 0.99
- buffer size: 1e5

Prioritisation parameters:
- a: 0.5
- b: 0.3

## Results

![results](plot.jpg)

```
Episode: 0, Average Score: 0.00
Episode: 10, Average Score: 0.64
Episode: 20, Average Score: 0.33
Episode: 30, Average Score: 0.55
Episode: 40, Average Score: 0.61
Episode: 50, Average Score: 0.90
Episode: 60, Average Score: 1.18
Episode: 70, Average Score: 1.42
Episode: 80, Average Score: 1.58
Episode: 90, Average Score: 1.92
Episode: 100, Average Score: 2.23
Episode: 110, Average Score: 2.68
.
.
.
And then we consistently get the error: 
cudaEventSynchronize in future::wait: device-side assert triggered.

Its safe to assume that we would reach an acceptable agent if this bug didnt plague us, and not matter what I've tried (I've been at this for 4+ hours, and there are like 3 discussions online in english), I cant fix this bug. I dont know why CUDA is doing this to me.

Because of this I cant even provide saved weights. Apologies.
```

## Future work

- Implement dueling DQN architecture and algorithms
- Optimise hyperparameters better
