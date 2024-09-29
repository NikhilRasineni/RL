import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

n_arms = 10
probs = np.random.rand(n_arms)
epsilon = 0.2

record = np.zeros((n_arms, 2))


def get_reward(prob, n=10):
    reward = 0
    for _ in range(n):
        if random.random() < prob:
            reward += 1
    return reward


def get_best_arm(record):
    arm_index = np.argmax(record[:, 1])
    return arm_index


def update_record(record, choice, reward):
    updated_reward = (record[choice, 0] * record[choice, 1] + reward) / (
        record[choice, 0] + 1
    )
    record[choice, 1] = updated_reward
    record[choice, 0] += 1
    return record


fig, ax = plt.subplots(1, 1)
ax.set_xlabel("Plays")
ax.set_ylabel("Avg Reward")
fig.set_size_inches(9, 5)


rewards = [0]
for i in tqdm(range(500)):
    if random.random() > 0.2:
        choice = get_best_arm(record)
    else:
        choice = np.random.randint(10)
    reward = get_reward(probs[choice])
    record = update_record(record, choice, reward)
    mean_reward = (rewards[-1] * (i + 1) + reward) / (i + 2)
    rewards.append(mean_reward)


ax.scatter(np.arange(len(rewards)),rewards)
plt.show()


