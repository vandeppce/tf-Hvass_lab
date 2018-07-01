import numpy as np
import pandas as pd
import time

N_STATES = 6                # 1维世界的宽度
ACTIONS = ['left', 'right'] # 探索者的可用动作
EPSILON = 0.9               # 贪婪度 greedy
ALPHA = 0.1                 # 学习率
GAMMA = 0.9                 # 奖励递减值
MAX_EPISODES = 13           # 最大回合数
FRESH_TIME = 0.3            # 移动间隔时间

# Q表
# 将所有的Q Values(行为值) 放在q_table中，以待更新
# index是对应的state（位置），columns是action（行为）

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns = actions
    )

    return table

# q_table = build_q_table(N_STATES, ACTIONS)
# print(q_table)

# 动作
# 引入epsilon greedy，因为在初始阶段，随机探索比固定行为模式要好
# 随着探索时间的提升，可以越来越贪婪。
# 这里设置EPSILON=0.9，90%的时间在选择最优策略，10%时间来探索

# 在某个state，选择action
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]      # 选出这个state的所有action值
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0): # 非贪婪或这个state还没有被探索过
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()

    return action_name

# 环境反馈s_, r
# 反馈下个state(S_)和在上个state(S)做出action(A)所得到的reward(R)
# 只有当o移动到了t，才会获得唯一奖励，其他情况均无奖励

def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1

    return S_, R

# 环境更新
def update_env(S, episode, step_counter):
    env_list = ['-']*(N_STATES - 1) + ['T']

    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                       ', end='')

    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

# 主循环
def rl():
    q_table = build_q_table(N_STATES, ACTIONS)              # 初始化 q_table
    for episode in range(MAX_EPISODES):                     # 回合
        step_counter = 0
        S = 0                                               # 回合初始位置
        is_terminated = False                               # 是否回合结束
        update_env(S, episode, step_counter)                # 环境更新

        while not is_terminated:

            A = choose_action(S, q_table)                   # 选择行为
            S_, R = get_env_feedback(S, A)                  # 实施行为并得到环境反馈
            q_predict = q_table.loc[S, A]                   # 估算的(状态-行为)值

            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()    # 实际的(状态-行为)值（回合未结束）
            else:
                q_target = R                                # 实际的(状态-行为)值（回合结束）
                is_terminated = True

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)     # q_table更新
            S = S_                                          # 探索者移动到下一个state

            update_env(S, episode, step_counter + 1)        # 环境更新

            step_counter += 1

    return q_table

if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)