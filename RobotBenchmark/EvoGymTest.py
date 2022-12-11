import gym
import evogym.envs
from evogym import sample_robot

if __name__ == '__main__':
    body, connections = sample_robot((5,5))
    print(body, connections)
    env = gym.make('Walker-v0', body=body)
    ob = env.reset()
    print(ob)

    i = 0
    while True:
        action = env.action_space.sample()-1
        print(action)
        ob, reward, done, info = env.step(action)
        env.render()

        if done:
            env.reset()
        i += 1
        if i == 1000:
            break

    env.close()

