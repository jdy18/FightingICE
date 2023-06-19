import gym
import d4rl
env = gym.make('maze2d-umaze-v1')
dataset = d4rl.qlearning_dataset(env)
dataset["observations"]
dataset["actions"]
dataset["rewards"]
dataset["terminals"]
dataset["next_observations"]
dataset["terminals"]
