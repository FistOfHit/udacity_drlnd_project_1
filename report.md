# Project report

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
- a: 0.2
- b: 0.1

## Results

![results](plot.jpg)

```

```

## Future work

- Implement dueling DQN architecture and algorithms
- Optimise hyperparameters better
