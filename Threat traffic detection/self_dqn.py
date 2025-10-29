import collections

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import time
from datetime import timedelta
import matplotlib.pyplot as plt
# from caffe2.python.helpers.train import accuracy
# from torch.ao.quantization.fx.utils import return_arg_list
from tqdm import tqdm
# from 绘制episode import input_dim


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)  # 设置随机种子为42

class Qnet(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(Qnet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.fc(x)

class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, action_dim, learning_rate, gamma,epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim,self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim,self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.criterion = nn.MSELoss() #损失函数

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        # dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values   # TD误差目标
        dqn_loss = self.criterion(q_values, q_targets)
        # dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1


class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state)

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

def evaluate(agent, X_test, y_test):
    correct = 0
    total = len(X_test)
    with torch.no_grad():
        for i in range(total):
            state = X_test[i]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action = agent.q_net(state_tensor).argmax().item()
            # q_values = agent.q_net(state).gather(1, action)
            # action = q_values.argmax().item()
            if action == y_test[i]:
                correct += 1
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == '__main__':
    data = np.load('D:\studysoftware\StudyData\pycharmproject\DQNcode/NSL-KDD-DQN\data_process/nsl_kdd_preprocessed_spearman(1).npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    print("加载数据成功")
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.01
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 32
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = X_train.shape[1]
    action_dim = 2
    agent = DQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device)
    return_list = []
    print("start Training")
    for episode in range(num_episodes):
        start_time = time.time()
        total_reward = 0
        correct_predictions = 0
        total_samples = 0
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        for i in indices:
            state = X_train[i]
            action =agent.take_action(state)
            if action == y_train[i]:
                reward = 1
            else:
                reward = 0
            total_reward += reward
            next_i = (i + 1) % len(X_train)
            next_state = X_train[next_i]
            replay_buffer.add(state, action, reward, next_state)
            if replay_buffer.size() > minimal_size: #进行Q网络的更新
                b_s, b_a, b_r, b_ns = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r
                }
                agent.update(transition_dict)
            total_samples += 1
        acc = total_reward / total_samples
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward}, Accuracy: {acc:.4f}, Epsilon: {agent.epsilon:.4f}, Time: {time.time() - start_time}")
        if episode % 2 == 0:
            test_acc = evaluate(agent, X_test, y_test)
            print(test_acc)