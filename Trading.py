import numpy as np
import pandas as pd
from parameters import ROLLING_WINDOW, STRONG_BUY, STRONG_SELL, INITIAL_PORTFOLIO, WINDOW_SIZE, \
        SPLIT_RATIO, THRESHOLD, OPTIM_RATIO, WINDOW_SIZE_DATA, TRAIN_OPTIM, HOLD, SELL, BUY
from PPO import Agent
import talib as ta
import torch as T
from untrade.client import Client
import uuid
import os

class KalmanFilter:
    def __init__(self, Q=1e-7, R=1e-2):
        self.Q = Q
        self.R = R
        self.P = 1.0  
        self.X = 0.0  

    def predict(self):
        self.P = self.P + self.Q  
        return self.X

    def update(self, measurement):
        K = self.P / (self.P + self.R)  
        self.X = self.X + K * (measurement - self.X)  
        self.P = (1 - K) * self.P  


def perform_backtest_large_csv(csv_file_path:str):
    client = Client()
    file_id = str(uuid.uuid4())
    chunk_size = 90 * 1024 * 1024
    total_size = os.path.getsize(csv_file_path)
    total_chunks = (total_size + chunk_size - 1) // chunk_size
    chunk_number = 0
    if total_size <= chunk_size:
        total_chunks = 1
        result = client.backtest(
            file_path=csv_file_path,
            leverage=1,
            jupyter_id="team57_zelta_hpps",
            # result_type="Q",
        )
        for value in result:
            print(value)

        return result

    with open(csv_file_path, "rb") as f:
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            chunk_file_path = f"/tmp/{file_id}_chunk_{chunk_number}.csv"
            with open(chunk_file_path, "wb") as chunk_file:
                chunk_file.write(chunk_data)

            # Large CSV Backtest
            result = client.backtest(
                file_path=chunk_file_path,
                leverage=1,
                jupyter_id="team57_zelta_hpps",
                file_id=file_id,
                chunk_number=chunk_number,
                total_chunks=total_chunks,
                # result_type="Q",
            )
            for value in result:
                print(value)
            os.remove(chunk_file_path)
            chunk_number += 1
    return result

def normalize_data(df : pd.DataFrame):
    # normalise open and close prices
    data = df.copy()
    price = data['close']
    data['rolling_mean'] = data['close'].rolling(ROLLING_WINDOW).mean()
    data['rolling_std'] = data['close'].rolling(ROLLING_WINDOW).std()
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['normalized_close'] = (data['close'] - data['rolling_mean'])/data['rolling_std']
    
    return data

def cusum(data : pd.DataFrame, threshold):
    # cusum indicator function
    data = normalize_data(data)
    data['Sh'] = 0.0
    data['Sl'] = 0.0
    for i in range(1, len(data)) :
        data.loc[i, 'Sh'] = max(0, data.loc[i-1, 'Sh'] + data.loc[i, 'normalized_close'] - threshold)
        data.loc[i, 'Sl'] = max(0, data.loc[i-1, 'Sl'] - data.loc[i, 'normalized_close'] - threshold)
    
    return data

def create_hurst_exponent(data):
    # hurst exponent indicator function
    N = len(data)
    mean_data = np.mean(data)
    cumulative_dev = np.cumsum(data - mean_data)
    R = np.max(cumulative_dev) - np.min(cumulative_dev)  
    S = np.std(data)                                     

    if S == 0:  
        return np.nan

    H = np.log(R/S) / np.log(N)
    return H

def create_hurst(data):
    # hurst indicator function
    data2 = data.copy()
    window_size = ROLLING_WINDOW
    length = (int)(len(data2) / window_size)
    window_data = []
    for i in range(0, len(data2), window_size):
        window = data2.loc[i:i+window_size, 'close'].tolist()
        window_data.append(window)
        i+=window_size
    hurst = [
        create_hurst_exponent(window_data[i])
        for i in range(len(window_data))
    ]
    j=0
    data2['hurst_value'] = 0
    data2['weekly_hurst'] = 0
    for i in range(0, len(data2), window_size):
        if(j < len(hurst)):
            if(hurst[j] > 0.5):
                data2.loc[i:i+window_size-1, 'weekly_hurst'] = 1
                data2['hurst_value'] = data2['hurst_value'].astype(float)
                data2.loc[i:i+window_size-1, 'hurst_value'] = hurst[j]
            else:
                data2.loc[i:i+window_size-1, 'weekly_hurst'] = -1
                data2['hurst_value'] = data2['hurst_value'].astype(float)
                data2.loc[i:i+window_size-1, 'hurst_value'] = hurst[j]
            j+=1

    return data2

def calculate_portfolio_live(df: pd.DataFrame, slipage=0.002, initial_portfolio=1000.00):
    # simulate live trading
    date_time = pd.to_datetime(df['datetime'])
    signal = df[df['signals'] != 0]
    signal = signal.reset_index(drop= True)
    returns = []
    trade_time = []

    for i in range(0, len(signal) - 1, 2):
        if signal.loc[i, 'signals'] == 1 and signal.loc[i, 'trade_type'] == STRONG_BUY:  
            entry = signal.loc[i, 'close']
            exit = signal.loc[i + 1, 'close']
            returns.append((exit - entry) / entry)
            trade_time.append(date_time.iloc[i+1] - date_time.iloc[i])
        elif signal.loc[i, 'trade_type'] == STRONG_BUY:  
            entry = signal.loc[i, 'close']
            exit = signal.loc[i + 1, 'close']
            returns.append((entry - exit) / entry)
            trade_time.append(date_time.iloc[i+1] - date_time.iloc[i])
    
    if len(signal) % 2 == 1:
        last_trade = signal.iloc[-1, :]
        entry_price = last_trade['close']
        current_price = df['close'].iloc[-1] 
        if last_trade['signals'] == 1: 
            returns.append((current_price - entry_price) / entry_price)
        else:  
            returns.append((entry_price - current_price) / entry_price)

    returns = np.array(returns)       
    returns_2 = 1 + returns - slipage  
    portfolio = [initial_portfolio]    
    for ret in returns_2:
        portfolio.append(portfolio[-1] * ret)
    max_drawdown = 0
    peak_portfolio = initial_portfolio
    for value in portfolio:
        if value > peak_portfolio:
            peak_portfolio = value
        else:
            drawdown = (peak_portfolio - value) / peak_portfolio
            max_drawdown = max(drawdown, max_drawdown)

    final_portfolio = portfolio[-1]
    return final_portfolio, max_drawdown

def drawdown(portfolio):
    # define drawdown
    max_drawdown = 0
    peak_portfolio = portfolio[0]
    for value in portfolio:
        if value > peak_portfolio:
            peak_portfolio = value
        else:
            drawdown = (peak_portfolio - value) / peak_portfolio
            max_drawdown = max(drawdown, max_drawdown)
    return max_drawdown

def perform_backtest_large_csv(csv_file_path:str):
    # Backtesting with untrade engine
    client = Client()
    file_id = str(uuid.uuid4())
    chunk_size = 90 * 1024 * 1024
    total_size = os.path.getsize(csv_file_path)
    total_chunks = (total_size + chunk_size - 1) // chunk_size
    chunk_number = 0
    if total_size <= chunk_size:
        total_chunks = 1
        # Normal Backtest
        result = client.backtest(
            file_path=csv_file_path,
            leverage=1,
            jupyter_id="team57_zelta_hpps",
            # result_type="Q",
        )
        for value in result:
            print(value)

        return result

    with open(csv_file_path, "rb") as f:
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            chunk_file_path = f"/tmp/{file_id}_chunk_{chunk_number}.csv"
            with open(chunk_file_path, "wb") as chunk_file:
                chunk_file.write(chunk_data)

            # Large CSV Backtest
            result = client.backtest(
                file_path=chunk_file_path,
                leverage=1,
                jupyter_id="team57_zelta_hpps",
                file_id=file_id,
                chunk_number=chunk_number,
                total_chunks=total_chunks,
                # result_type="Q",
            )

            for value in result:
                print(value)

            os.remove(chunk_file_path)

            chunk_number += 1

    return result

# VIRUS
def reward_gen(portfolio_change, drawdown, hurst, flag, b = 0.8):
    # Basic reward function
    """ write some function here """
    return (portfolio_change - b*drawdown) + hurst * flag
# VIRUS*

class TradingEnv():
    # The entire RL + WFO trading environment
    def __init__(self, df_path, valid_df_path, train_optim, initial_portfolio = INITIAL_PORTFOLIO, window_size = WINDOW_SIZE, split_ratio = SPLIT_RATIO):
        # Constructor
        if (df_path == valid_df_path):
            data = pd.read_csv(df_path)
            split = int(len(data) * split_ratio)
            self.df = data[:split].reset_index(drop= True)
            self.valid_df = data.reset_index(drop= True)
        else:
            self.df = pd.read_csv(df_path)
            self.valid_df = pd.read_csv(valid_df_path)

        # Initialise values
        self.initial_portfolio = initial_portfolio
        self.window_size = window_size
        self.train_optim = train_optim
        self.portfolio = initial_portfolio
        self.max_drawdown = 0

        # Data Preprocessing and Implementation of technical indicators - Training data
        self.df = cusum(self.df, 0.5)
        self.df['cusum'] = self.df['Sh'] - self.df['Sl']
        self.df['RSI_Smoothed'] = ta.RSI(self.df['close'], timeperiod= 14).rolling(window= 5).mean()
        roc1 = ta.ROC(self.df['close'], timeperiod=10)
        roc2 = ta.ROC(self.df['close'], timeperiod=15)
        roc3 = ta.ROC(self.df['close'], timeperiod=20)
        roc4 = ta.ROC(self.df['close'], timeperiod=30)
        self.df['KST'] = (ta.SMA(roc1, timeperiod= 10) + 2 * ta.SMA(roc2, timeperiod= 15) + 
                          3 * ta.SMA(roc3, timeperiod= 20) + 4 * ta.SMA(roc4, timeperiod= 30))
        self.df['KST_Signal'] = ta.SMA(self.df['KST'], timeperiod= 9)
        ema13 = ta.EMA(self.df['close'], timeperiod= 13)
        self.df['Bull_Power'] = self.df['high'] - ema13
        self.df['Bear_Power'] = self.df['low'] - ema13
        self.df['Plus_DI'] = ta.PLUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)
        self.df['Minus_DI'] = ta.MINUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)
        self.df['Plus_DI_s'] = ta.PLUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=63)
        self.df['Minus_DI_s'] = ta.MINUS_DI(self.df['high'],self.df['low'], self.df['close'], timeperiod=63)
        ema12 = ta.EMA(self.df['close'], timeperiod= 12)
        ema26 = ta.EMA(self.df['close'], timeperiod= 26)
        self.df['MACD'] = ema12 - ema26
        self.df['Aroon_Osc'] = ta.AROONOSC(self.df['high'], self.df['low'], timeperiod= 25)
        self.df['Rolling_Threshold'] = self.df['close'].pct_change(5, fill_method= None).rolling(10).quantile(0.9)
        HA_close = (self.df['open'] + self.df['close'] + self.df['high'] + self.df['low']) / 4
        HA_open = self.df['open']
        HA_open = (HA_open.shift(1) + HA_close.shift(1))/ 2
        self.df['HA_Close>Open'] = (HA_close > HA_open).astype('float64')
        filtered_price=[]
        kf=KalmanFilter() 
        for i in range(len(self.df)):
            kf.update(self.df['close'].iloc[i])
            filtered_price.append(kf.predict())
        filtered_price = pd.Series(filtered_price)
        self.df['Filtered<Close'] = (self.df['close'] > filtered_price).astype('float64')
        self.df = create_hurst(self.df)
        self.df['signals'] = 0
        self.df['trade-type'] = 0

        # Data Preprocessing and Implementation of technical indicators - Testing data
        self.valid_df = cusum(self.valid_df, 0.5)
        self.valid_df['cusum'] = self.valid_df['Sh'] - self.valid_df['Sl']
        self.valid_df['RSI_Smoothed'] = ta.RSI(self.valid_df['close'], timeperiod= 14).rolling(window= 5).mean()
        vroc1 = ta.ROC(self.valid_df['close'], timeperiod=10)
        vroc2 = ta.ROC(self.valid_df['close'], timeperiod=15)
        vroc3 = ta.ROC(self.valid_df['close'], timeperiod=20)
        vroc4 = ta.ROC(self.valid_df['close'], timeperiod=30)
        self.valid_df['KST'] = (ta.SMA(vroc1, timeperiod= 10) + 2 * ta.SMA(vroc2, timeperiod= 15) + 
                          3 * ta.SMA(vroc3, timeperiod= 20) + 4 * ta.SMA(vroc4, timeperiod= 30))
        self.valid_df['KST_Signal'] = ta.SMA(self.valid_df['KST'], timeperiod= 9)
        vema13 = ta.EMA(self.valid_df['close'], timeperiod= 13)
        self.valid_df['Bull_Power'] = self.valid_df['high'] - vema13
        self.valid_df['Bear_Power'] = self.valid_df['low'] - vema13
        self.valid_df['Plus_DI'] = ta.PLUS_DI(self.valid_df['high'], self.valid_df['low'], self.valid_df['close'], timeperiod=14)
        self.valid_df['Minus_DI'] = ta.MINUS_DI(self.valid_df['high'], self.valid_df['low'], self.valid_df['close'], timeperiod=14)
        self.valid_df['Plus_DI_s'] = ta.PLUS_DI(self.valid_df['high'], self.valid_df['low'], self.valid_df['close'], timeperiod=63)
        self.valid_df['Minus_DI_s'] = ta.MINUS_DI(self.valid_df['high'],self.valid_df['low'], self.valid_df['close'], timeperiod=63)
        vema12 = ta.EMA(self.valid_df['close'], timeperiod= 12)
        vema26 = ta.EMA(self.valid_df['close'], timeperiod= 26)
        self.valid_df['MACD'] = vema12 - vema26
        self.valid_df['Aroon_Osc'] = ta.AROONOSC(self.valid_df['high'], self.valid_df['low'], timeperiod= 25)
        self.valid_df['Rolling_Threshold'] = self.valid_df['close'].pct_change(5, fill_method= None).rolling(10).quantile(0.9)
        HA_closev = (self.valid_df['open'] + self.valid_df['close'] + self.valid_df['high'] + self.valid_df['low']) / 4
        HA_openv = self.valid_df['open']
        HA_openv = (HA_openv.shift(1) + HA_closev.shift(1))/ 2
        self.valid_df['HA_Close>Open'] = (HA_closev > HA_openv).astype('float64')
        valid_filtered_price=[]
        kf=KalmanFilter() 
        for i in range(len(self.valid_df)):
            kf.update(self.valid_df['close'].iloc[i])
            valid_filtered_price.append(kf.predict())
        valid_filtered_price = pd.Series(valid_filtered_price)
        self.valid_df['Filtered<Close'] = (self.valid_df['close'] > valid_filtered_price).astype('float64')
        self.valid_df = create_hurst(self.valid_df)
        self.valid_df['signals'] = 0
        self.valid_df['trade_type'] = 0

        # Choose columns as input to model
        self.columns = ['Filtered<Close', 'HA_Close>Open', 'Rolling_Threshold', 'Aroon_Osc', 'MACD', 'Plus_DI', 'Plus_DI_s', 'Minus_DI', 'Minus_DI_s',
                        'Bull_Power', 'Bear_Power', 'KST_Signal', 'KST', 'RSI_Smoothed', 'hurst_value', 'cusum']
        for column in self.columns:
            self.df[column] = self.df[column] / self.df[column].abs().max()
            self.valid_df[column] = self.valid_df[column] / self.valid_df[column].abs().max()
        
        # Initialise PPO model
        self.positions = [0, 1, -1]
        self.agent = Agent(input_dims= len(self.columns), n_actions= 3)

    def checkdtypes(self):
        # Confirms datatypes of chosen columns
        for column in self.columns:
            print(column, ":", self.df[column].dtype)

    def optimisation_phase(self, optim_data: pd.DataFrame, wfreward=0):
        # Optimisation - Learning phase of Trading
        reward = 0
        avg_reward = 0
        flag = 0
        datetime = optim_data["datetime"]
        optim_data = optim_data.drop(columns=["datetime"]).dropna()
        datetime = datetime.loc[optim_data.index]
        optim_data["datetime"] = datetime.values
        if "level_0" in optim_data.columns:
            optim_data = optim_data.drop(columns=["level_0"])
        optim_data.reset_index(inplace=True)
        smoothed_portfolio = None
        last_learn_index = 0  
        threshold = THRESHOLD

        for i in range(1, len(optim_data) - 1):
            observation = T.tensor(optim_data[self.columns].iloc[i].values, dtype=T.float32).to(self.agent.actor.device)
            # shape of data here is 2d, batching needed here - entire structure will change, pass parts of optim data, to get actions for entire optim data. 
            # then iterate on it to get trade logic
            action, probs, value = self.agent.choose_action(observation=observation, wlkfrwd=False, flag= flag)

            # Trade Logic
            if flag == HOLD:
                flag = self.positions[action]
                optim_data.loc[i, "signals"] = flag
                optim_data.loc[i, 'trade_type'] = HOLD if flag == HOLD else STRONG_BUY if flag == BUY else STRONG_SELL
            elif flag == BUY and self.positions[action] == SELL:
                flag = HOLD
                optim_data.loc[i, "signals"] = SELL
                optim_data.loc[i, 'trade_type'] = STRONG_SELL
            elif flag == SELL and self.positions[action] == BUY:
                flag = HOLD
                optim_data.loc[i, "signals"] = BUY
                optim_data.loc[i, 'trade_type'] = STRONG_BUY

            optim_data.loc[i, 'current_portfolio'], _ = calculate_portfolio_live(optim_data[0:i+1])

            if i >= 5:
                smoothed_portfolio = optim_data.loc[i-4:i, 'current_portfolio'].mean()
                optim_data.loc[i, 'smoothed_portfolio'] = smoothed_portfolio
            else:
                smoothed_portfolio = optim_data.loc[i, 'current_portfolio']
                optim_data.loc[i, 'smoothed_portfolio'] = smoothed_portfolio
            if i >= last_learn_index + 5:  
                if i >= 5:
                    change_5 = abs((smoothed_portfolio - optim_data.loc[i-5, 'smoothed_portfolio']) / optim_data.loc[i-5, 'smoothed_portfolio'])
                if i >= 10:
                    change_10 = abs((smoothed_portfolio - optim_data.loc[i-10, 'smoothed_portfolio']) / optim_data.loc[i-10, 'smoothed_portfolio'])
                if i >= 15:
                    change_15 = abs((smoothed_portfolio - optim_data.loc[i-15, 'smoothed_portfolio']) / optim_data.loc[i-15, 'smoothed_portfolio'])
                if i >= 20:
                    change_20 = abs((smoothed_portfolio - optim_data.loc[i-20, 'smoothed_portfolio']) / optim_data.loc[i-20, 'smoothed_portfolio'])

                if (i >= 5 and change_5 > threshold) or \
                    (i >= 10 and change_10 > threshold) or \
                    (i >= 15 and change_15 > threshold) or \
                    (i >= 20 and change_20 > threshold):
                    if i >= 20:
                        drawdown_period = 20
                    elif i >= 15:
                        drawdown_period = 15
                    elif i >= 10:
                        drawdown_period = 10
                    else:
                        drawdown_period = 5

                    drawdown_value = drawdown(np.array(optim_data.loc[i-drawdown_period+1:i+1, 'current_portfolio']))
                    # here is where u change the reward fn
                    reward = reward_gen(portfolio_change=smoothed_portfolio - optim_data.loc[i-drawdown_period, 'smoothed_portfolio'], 
                                        drawdown=drawdown_value, 
                                        hurst=optim_data.loc[i, 'hurst_value'], 
                                        flag=flag)
                    # ..

            # Trigger learning
            if len(self.agent.memory.states) >= self.train_optim:
                # shape of data for learn here is 3d
                self.agent.learn()
                last_learn_index = i

            # Model memory update and learning
            done = i >= (len(optim_data) - 1)
            self.agent.remember(state=observation.cpu(), action=action, probs=probs, vals=value, reward=wfreward + reward, done=done)
            avg_reward += reward

        avg_reward /= len(optim_data)
        self.agent.remember(state=observation.cpu(), action=action, probs=probs, vals=value, reward=wfreward + reward, done=True)
        self.agent.learn()

        print(f"Average reward for optimization phase: {avg_reward}")

    def walkforward_phase(self, wfreward, wf_data: pd.DataFrame): 
        # Walk Forward - trial phase of Trading
        reward = 0
        flag = 0
        datetime = wf_data["datetime"]
        wf_data = wf_data.drop(columns=["datetime"]).dropna()
        datetime = datetime.loc[wf_data.index]
        wf_data["datetime"] = datetime.values

        if "level_0" in wf_data.columns:
            wf_data = wf_data.drop(columns=["level_0"])
        wf_data.reset_index(drop=True, inplace=True)
        smoothed_portfolio = None
        threshold = 0.03 

        for i in range(1, len(wf_data)-1):
            observation = T.tensor(wf_data[self.columns].iloc[i].values, dtype=T.float32).to(self.agent.actor.device)
            # shape of data here is 2d, batching needed here - entire structure will change, pass whole wf data, to get actions for entire optim data. 
            # then iterate on it to get trade logic
            action, probs, value = self.agent.choose_action(observation=observation, wlkfrwd= True, flag= flag)

            # Trade Logic
            if flag == HOLD:
                flag = self.positions[action]
                wf_data.loc[i, "signals"] = flag
                if (flag == HOLD):
                    wf_data.loc[i, 'trade_type'] = HOLD
                elif (flag == BUY):
                    wf_data.loc[i, 'trade_type'] = STRONG_BUY
                else:
                    wf_data.loc[i, 'trade_type'] = STRONG_SELL
            elif flag == BUY:
                if self.positions[action] == SELL:
                    flag = HOLD
                    wf_data.loc[i, "signals"] = SELL
                    wf_data.loc[i, 'trade_type'] = STRONG_SELL
            elif flag == SELL:
                if self.positions[action] == BUY:
                    flag = HOLD
                    wf_data.loc[i, "signals"] = BUY
                    wf_data.loc[i, 'trade_type'] = STRONG_BUY
                
        
        # Model memory update and training
        final_portfolio, drawdowns=calculate_portfolio_live(wf_data)
        # here is where u change the reward fn
        reward = reward_gen(portfolio_change= final_portfolio, drawdown= drawdowns, hurst= wf_data.loc[i, 'hurst_value'], flag= flag) 
        # ..
        self.agent.remember(state= observation.cpu(), action= action, probs= probs, vals= value, reward= wfreward + reward, done= True)
        # data for learn here is also of 3d shape
        self.agent.learn()
        return reward
    
    # This has to be for training of one batch, divided into multiple windows of the same window_size
    def training(self, window_size):
        # Combine Optimization and Walkforward phases
        start = 0
        wf_reward = 0
        overlap_size = window_size - int(OPTIM_RATIO * window_size)
        while start + window_size <= len(self.df):
            optim_end = start + int(OPTIM_RATIO * window_size)
            wf_end = start + window_size
            optim_data = self.df.iloc[start:optim_end]
            wf_data = self.df.iloc[optim_end:wf_end]   
            self.optimisation_phase(optim_data= optim_data, wfreward= wf_reward)
            wf_reward += self.walkforward_phase(wf_data= wf_data, wfreward= wf_reward) 
            print(f"complete training till {start}")
            print(f"Total reward from walk forward: {wf_reward}")
            start += overlap_size

    def result(self):
        # Result from training
        self.training(window_size=self.window_size)
        datetime = self.valid_df["datetime"]
        datetime1 = datetime.copy()
        self.valid_df = self.valid_df.drop(columns=["datetime"]).dropna()
        datetime = datetime.loc[self.valid_df.index]
        self.valid_df = self.valid_df.apply(pd.to_numeric, errors="raise").astype(float)
        if "level_0" in self.valid_df.columns:
            self.valid_df = self.valid_df.drop(columns=["level_0"])
        self.valid_df.reset_index(drop=True, inplace=True)
        self.valid_df["signals"] = 0
        flag = 0
        for i in range(1, len(self.valid_df)-1):
            observation = T.tensor(self.valid_df[self.columns].iloc[i].values, dtype=T.float32).to(self.agent.actor.device)
            action, _, _ = self.agent.choose_action(observation=observation, wlkfrwd= True, flag= flag)

            # Trade Logic
            if flag == HOLD:
                flag = self.positions[action]
                self.valid_df.loc[i, "signals"] = flag
                if (flag == HOLD):
                    self.valid_df.loc[i, 'trade_type'] = HOLD
                elif (flag == BUY):
                    self.valid_df.loc[i, 'trade_type'] = STRONG_BUY
                else:
                    self.valid_df.loc[i, 'trade_type'] = STRONG_SELL
            elif flag == BUY:
                if self.positions[action] == SELL:
                    flag = HOLD
                    self.valid_df.loc[i, "signals"] = SELL
                    self.valid_df.loc[i, 'trade_type'] = STRONG_SELL
            elif flag == SELL:
                if self.positions[action] == BUY:
                    flag = HOLD
                    self.valid_df.loc[i, "signals"] = BUY
                    self.valid_df.loc[i, 'trade_type'] = STRONG_SELL

        # Output
        print(self.valid_df["signals"].unique())
        self.valid_df['datetime'] = datetime1
        self.valid_df.to_csv("final_logs.csv", index=False)
        perform_backtest_large_csv(csv_file_path="final_logs.csv")  

def process_data():
    # Preprocessing data and creating Trading environment
    trading_env = TradingEnv(df_path= r'/mnt/OS/Study/Quantalytics/data_2020_2023/BTC_2019_2023_6h.csv', 
                         valid_df_path= r'/mnt/OS/Study/Quantalytics/data_2020_2023/BTC_2019_2023_6h.csv', window_size= WINDOW_SIZE_DATA, train_optim= TRAIN_OPTIM)
    return trading_env

def strat(trading_env):
    # Strategy result
    trading_env.result()

# Execution
trading_env = process_data()
strat(trading_env= trading_env)
