import numpy as np
from tqdm import tqdm
import random
from typing import List
import matplotlib.pyplot as plt

n_arms=10
probs = np.random.rand(n_arms)
record = np.zeros((n_arms,2))


def softmax(probs:np.ndarray,temperature:float) -> np.ndarray:
    return (np.exp(probs/temperature)/np.sum(np.exp(probs/temperature)))

def get_reward(prob:float)->float:
    reward =0
    for i in range(10):
        if random.random() >0.2:
            reward += 1
    return reward



def update_record(record:np.ndarray,choice:int,reward:float)->np.ndarray:
    updated_reward =  (record[choice, 0] * record[choice, 1] + reward) / (
        record[choice, 0] + 1
    )
    record[choice,1] = updated_reward
    record[choice,0] =  record[choice,0] +1
    return record


fig,ax = plt.subplots(1,1)
ax.set_xlabel("Plays")
ax.set_ylabel("Avg Reward")
fig.set_size_inches(9,5)



rewards: List[float] = [0]
for i in tqdm(range(500)):
    softmax_probs = softmax(record[:,1],temperature=0.7)
    choice = np.random.choice(np.arange(10),p=softmax_probs)
    reward = get_reward(probs[choice])
    record = update_record(record,choice,reward)
    mean_reward = ((i+1)*rewards[-1] + reward)/(i+2)
    rewards.append(mean_reward)

ax.scatter(np.arange(len(rewards)),rewards)
plt.show()



## Converges faster than epsilon greedy as can be seen from the plot.