import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle

from sklearn.preprocessing import StandardScaler

# It is a matter or perspective why not use RNN
# Because the state changes and you do act so that it makes sense
# The trade is taken and the environment is updated accordingly

# Let's use AAPL (Apple), MSI (Motorola), SBUX (Starbucks)
def get_data(file):
  # returns a T x 3 list of stock prices
  # each row is a different stock
  # 0 = AAPL
  # 1 = MSI
  # 2 = SBUX
  df = pd.read_csv(file)
  return df.values





def get_scaler(env):
  # return scikit-learn scaler object to scale the states
  # Note: you could also populate the replay buffer here

  states = []
  for _ in range(env.n_step):
    action = np.random.choice(env.action_space)
    state, reward, done, info = env.step(action)
    states.append(state)
    if done:
      break

  scaler = StandardScaler()
  scaler.fit(states)
  return scaler




def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)



class LinearModel:
  """ A linear regression model """
  def __init__(self, input_dim, n_action):
    self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
    self.b = np.zeros(n_action)

    # momentum terms
    self.vW = 0
    self.vb = 0

    self.losses = []

  def predict(self, X):
    # make sure X is N x D
    assert(len(X.shape) == 2)
    return X.dot(self.W) + self.b

  def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
    # make sure X is N x D
    assert(len(X.shape) == 2)

    # the loss values are 2-D
    # normally we would divide by N only
    # but now we divide by N x K
    num_values = np.prod(Y.shape)

    # do one step of gradient descent
    # we multiply by 2 to get the exact gradient
    # (not adjusting the learning rate)
    # i.e. d/dx (x^2) --> 2x
    Yhat = self.predict(X)
    gW = 2 * X.T.dot(Yhat - Y) / num_values
    gb = 2 * (Yhat - Y).sum(axis=0) / num_values

    # update momentum terms
    self.vW = momentum * self.vW - learning_rate * gW
    self.vb = momentum * self.vb - learning_rate * gb

    # update params
    self.W += self.vW
    self.b += self.vb

    mse = np.mean((Yhat - Y)**2)
    self.losses.append(mse)

  def load_weights(self, filepath):
    npz = np.load(filepath)
    self.W = npz['W']
    self.b = npz['b']

  def save_weights(self, filepath):
    np.savez(filepath, W=self.W, b=self.b)




class MultiStockEnv:
  """
  A 3-stock trading environment.
  State: vector of size 7 (n_stock * 2 + 1)
    - # shares of stock 1 owned
    - # shares of stock 2 owned
    - # shares of stock 3 owned
    - price of stock 1 (using daily close price)
    - price of stock 2
    - price of stock 3
    - cash owned (can be used to purchase more stocks)
  Action: categorical variable with 27 (3^3) possibilities
    - for each stock, you can:
    - 0 = sell
    - 1 = hold
    - 2 = buy
  """
  def __init__(self, data, initial_investment=20000):
    # data
    self.stock_price_history = data
    self.n_step, self.n_stock = self.stock_price_history.shape

    # instance attributes
    self.initial_investment = initial_investment
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    self.cash_in_hand = None

    self.action_space = np.arange(3**self.n_stock)

    # action permutations
    # returns a nested list with elements like:
    # [0,0,0]
    # [0,0,1]
    # [0,0,2]
    # [0,1,0]
    # [0,1,1]
    # etc.
    # 0 = sell
    # 1 = hold
    # 2 = buy
    self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

    # calculate size of state
    self.state_dim = self.n_stock * 2 + 1

    self.reset()


  def reset(self):
    self.cur_step = 0
    self.stock_owned = np.zeros(self.n_stock)
    self.stock_price = self.stock_price_history[self.cur_step]
    self.cash_in_hand = self.initial_investment
    return self._get_obs()


  def step(self, action):
    assert action in self.action_space

    # get current value before performing the action
    prev_val = self._get_val()

    # update price, i.e. go to the next day
    self.cur_step += 1
    self.stock_price = self.stock_price_history[self.cur_step]

    # perform the trade
    self._trade(action)

    # get the new value after taking the action
    cur_val = self._get_val()

    # reward is the increase in porfolio value
    reward = cur_val - prev_val

    # done if we have run out of data
    done = self.cur_step == self.n_step - 1

    # store the current value of the portfolio here
    info = {'cur_val': cur_val}

    # conform to the Gym API
    return self._get_obs(), reward, done, info


  def _get_obs(self):
    obs = np.empty(self.state_dim)
    obs[:self.n_stock] = self.stock_owned
    obs[self.n_stock:2*self.n_stock] = self.stock_price
    obs[-1] = self.cash_in_hand
    return obs
    


  def _get_val(self):
    return self.stock_owned.dot(self.stock_price) + self.cash_in_hand


  def _trade(self, action):
    # index the action we want to perform
    # 0 = sell
    # 1 = hold
    # 2 = buy
    # e.g. [2,1,0] means:
    # buy first stock
    # hold second stock
    # sell third stock
    action_vec = self.action_list[action]

    # determine which stocks to buy or sell
    sell_index = [] # stores index of stocks we want to sell
    buy_index = [] # stores index of stocks we want to buy
    for i, a in enumerate(action_vec):
      if a == 0:
        sell_index.append(i)
      elif a == 2:
        buy_index.append(i)

    # sell any stocks we want to sell
    # then buy any stocks we want to buy
    if sell_index:
      # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
      for i in sell_index:
        self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
        self.stock_owned[i] = 0
    if buy_index:
      # NOTE: when buying, we will loop through each stock we want to buy,
      #       and buy one share at a time until we run out of cash
      can_buy = True
      while can_buy:
        for i in buy_index:
          if self.cash_in_hand > self.stock_price[i]:
            self.stock_owned[i] += 1 # buy one share
            self.cash_in_hand -= self.stock_price[i]
          else:
            can_buy = False





class DQNAgent(object):
  def __init__(self, state_size, action_size, gama=0.95, epsilon=1.0):
    self.state_size = state_size
    self.action_size = action_size
    self.gamma = gama  # discount rate
    self.epsilon = epsilon  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model = LinearModel(state_size, action_size)

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return np.random.choice(self.action_size)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0])  # returns action


  def train(self, state, action, reward, next_state, done):
    if done:
      target = reward
    else:
      target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)

    target_full = self.model.predict(state)
    target_full[0, action] = target

    # Run one training step
    self.model.sgd(state, target_full)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay


  def load(self, name):
    self.model.load_weights(name)


  def save(self, name):
    self.model.save_weights(name)


def play_one_episode(agent, env, is_train):
  # note: after transforming states are already 1xD
  state = env.reset()
  state = scaler.transform([state])
  done = False

  while not done:
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    next_state = scaler.transform([next_state])
    if is_train == 'train':
      agent.train(state, action, reward, next_state, done)
    state = next_state

  return info['cur_val']

def write_txt_header( num_episodes,batch_size,initial_investment):
  f=open("rewards_report.txt","a+")
  f.write("-----------------------------------------------\r\n")
  f.write("----------- Starting new cycle ----------------\r\n")
  f.write("-----------------------------------------------\r\n")
  txt_now = f'{datetime.now()}'
  f.write(txt_now)
  f.write("\r\n")
  f.write(f'Paramenters: num_episodes:{num_episodes}; batch_size:{batch_size}; initial_investiment:{initial_investment}')
  f.write("\r\n")
  f.close()



if __name__ == '__main__':

  exploreHyper = [
    (0.95,0.95,'train'),(0.95,0.95,'test'),
    (0.95,0.6,'train'),(0.95,0.6,'test'),
    (0.75,0.95,'train'),(0.75,0.95,'test'),
    (0.75,0.6,'train'),(0.75,0.6,'test'),
    (0.55,0.95,'train'),(0.55,0.95,'test'),
    (0.55,0.6,'train'),(0.55,0.6,'test'),
    (0.45,0.95,'train'),(0.45,0.95,'test'),
    (0.45,0.6,'train'),(0.45,0.6,'test'),
  ]

  num_episodes = 2000
  batch_size = 16
  initial_investment = 20000
  
  write_txt_header(num_episodes,batch_size,initial_investment)

  for gama,epslon, mode in exploreHyper:    
    # config
    models_folder = f'btc-linear_rl_trader_models_{gama}_{epslon}'
    rewards_folder = f'btc-linear_rl_trader_rewards_{gama}_{epslon}'
    file = 'BTCUSDT-4h-data-pure.csv' # 'BTCUSDT-4h-data-pure.csv' or 'aapl_msi_sbux.csv' 

    #parser = argparse.ArgumentParser()
    #parser.add_argument('-m', '--mode', type=str, required=True,
    #                    help='either "train" or "test"')
    #args = parser.parse_args()

    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)

    data = get_data(file)
    n_timesteps, n_stocks = data.shape

    n_train = n_timesteps // 2

    train_data = data[:n_train]
    test_data = data[n_train:]

    env = MultiStockEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size, gama, epslon)
    scaler = get_scaler(env)

    # store the final value of the portfolio (end of episode)
    portfolio_value = []

    #if args.mode == 'test':
    if mode == 'test':
        # then load the previous scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # remake the env with test data
        env = MultiStockEnv(test_data, initial_investment)

        # make sure epsilon is not 1!
        # no need to run multiple episodes if epsilon = 0, it's deterministic
        agent.epsilon = 0.01

        # load trained weights
        agent.load(f'{models_folder}/linear.npz')

    # play the game num_episodes times
    for e in range(num_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent, env, mode)
        dt = datetime.now() - t0
        #print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
        portfolio_value.append(val) # append episode end portfolio value

    # save the weights when we are done
    #if args.mode == 'train':
    if mode == 'train':
        # save the DQN
        agent.save(f'{models_folder}/linear.npz')

        # save the scaler
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # plot losses
        plt.plot(agent.model.losses)
        plt.savefig(f'{rewards_folder}/{mode}-losses.png')
        plt.clf()
        #plt.show()


    # save portfolio value for each episode
    #np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)
    np.save(f'{rewards_folder}/{mode}.npy', portfolio_value)

    # =============================================================
    # Print reward charts
    #
    # =============================================================

    url = f'{rewards_folder}/{mode}.npy'
    a = np.load(url)

    text = f"{mode}-{gama}-{epslon} - average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}"
    print(text)
    f=open("rewards_report.txt","a+")
    f.write(text)
    f.write("\r\n")
    txt_now = f'{datetime.now()}'
    f.write(txt_now)   
    print(txt_now) 
    f.write("\r\n")
    f.close()

    plt.hist(a, bins=20)
    plt.title(mode)
    # Prepare to save the figure
    #fig1 = plt.gcf()
    #plt.show()
    #fig1.savefig(f'/content/drive/My Drive/Colab Notebooks/rl/final/linear_rl_trader_rewards/{mode}-{a.mean()}.png')
    plt.savefig(f'{rewards_folder}/{mode}-{a.mean()}.png')
    plt.clf()
    
