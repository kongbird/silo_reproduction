# Configuration Notes

## Parameters directly specified by the paper

- Physics timestep: 1/240 s
- Control timestep: 1/60 s
- Four physics steps per control step
- Solver position iterations: 16
- Solver velocity iterations: 1
- Cable length: 0.375 m
- Cable links: 18
- Cable radius: 0.003 m
- Cable stiffness: 0.01
- Cable damping: 0.001
- Cable plasticity beta: 0.7
- Cable observation points: 4
- Cable point noise: 1e-3
- Harness x range: [0.3, 0.75]
- Harness y range: [-0.4, 0.4]
- Harness yaw range: [-90, 90] degrees
- Cached states: 32768
- PPO gamma: 0.95
- PPO GAE lambda: 0.95
- PPO learning rate: 3e-4
- PPO network: [256, 256, 256]
- PPO environments: 1024
- PPO rollout steps: 16
- PPO episode horizon: 32

## Assumed values not specified by the paper

- Cable total mass
- Static and dynamic friction
- Restitution
- Reward distance coefficient
- Success epsilon
- U-shaped harness exact dimensions
- TCP-to-harness transform
- Robot simulation PD gains
