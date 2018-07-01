import numpy as np
import pandas as pd

class QLearningTable:
    # 初始化
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions              # action list
        self.lr = learning_rate             # 学习率
        self.gamma = reward_decay           # 奖励衰减
        self.epsilon = e_greedy             # 贪婪度
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)     # 初始化q_table

    # 选行为action
    def choose_action(self, observation):
        self.check_state_exist(observation)     # 检测本state是否在q_table中存在

        # 选择action
        if np.random.uniform() < self.epsilon:      # 选择Q-Value最高的action
            state_action = self.q_table.loc[observation, :]

            # 同一个 state, 可能会有多个相同的 Q action value, 所以我们乱序一下
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()

        else:       # 随机选择action
            action = np.random.choice(self.actions)

        return action

    # 学习更新参数
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)               # 检测q_table中是否存在s_
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()       # 下个state不是终止符
        else:
            q_target = r                         # 下个state是终止符

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)              # 更新对应的state-action值

    # 检测state是否存在
    # 如果没有当前state，插入一组全0数据，当作这个state的所有action的初始values
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

