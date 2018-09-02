import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

import lib.dblib as db
import matplotlib.pyplot as plt

RUN_DATE_COUNT = 300 #學習過去幾天的資料
WINDOWWIDTH = 300 # 螢幕寬度

lose_cnt=[]
win_cnt=[]
point_list=[]
profit=[]
points=0
lose=0
win=0

class TradeEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : 5000
	}

	def __init__(self,mode='train'):
		self.mode=mode
		self.Trade=db.TradeImg(RUN_DATE_COUNT,mode=self.mode)
		self.Trade.prepare_data()
		self.DateList=self.Trade.DateList
		self.Date=""
		self.runGame()
		
		self.action_space = spaces.Discrete(2)
		self.observation_space = self.Trade.GetData(self.DateList[0],0) #spaces.Box(-high, high) #np.array([1,2,3,4,5])
		self.state = self.Trade.GetData(self.DateList[0],0)
		self.seed()
		self.viewer = None

	def runGame(self):
		self.game_time=db.timer()
		self.DateIndex=0
		self.TimeIndex=0
		self.Units=[]
		self.Price=[]
		self.SellStopLose=[]

	def process_action(self,action):
		terminal=True
		reward=0
		if len(self.Price)>=db.END_K_INDEX : #排除K棒不足的情況
			end_price=self.Price[self.TimeIndex+30] #60分鐘後結算價
			now_price=self.Price[self.TimeIndex]
			if action==0: 
				terminal=(self.TimeIndex>=db.END_K_INDEX)
#				if self.TimeIndex>=db.END_K_INDEX: #不下單損失
#					reward=-db.STOP_LOSE
			elif action==1:
				stoplose=0
				for i in range (self.TimeIndex,self.TimeIndex+30):
					if self.Price[i]-now_price>=db.STOP_LOSE:
						stoplose=self.Price[i]-now_price
						break
				if stoplose>0:
#					reward=-self.SellStopLose[self.TimeIndex]-db.TRADE_LOSE
					reward=-stoplose-db.TRADE_LOSE
				else:
					reward=-(end_price-now_price)-db.TRADE_LOSE

		return terminal,reward

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		self.Date=self.DateList[self.DateIndex]
		self.Price=self.Trade.GetPrice(self.Date)
		self.SellStopLose=self.Trade.GetSellStopLose(self.Date)

		self.state=self.Trade.GetData(self.DateList[self.DateIndex],self.TimeIndex)

		#print(action)
		terminal,reward=self.process_action(action)
		
		self.Units.append(self.Price[self.TimeIndex])
		self.TimeIndex+=1
		if self.TimeIndex>=WINDOWWIDTH or terminal==True : 
			self.DateIndex+=1
			self.TimeIndex=0
		if self.DateIndex>=len(self.DateList):
			self.DateIndex=0
		if terminal:	
			########################  輸出單日報表  ########################
			#print(self.Date,"reward:",reward,self.game_time.spendtime("time"))
			plt.cla()
			plt.plot(self.Price,"g")
			plt.plot(self.Units,"b")
			plt.savefig("trade.png")
			plt.cla()     
			#plt.show();
			################################################################		
			self.Units=[]
			self.Price=[]
			self.SellStopLose=[]
			self.game_time=db.timer()
			self.state=self.Trade.GetData(self.DateList[0],0)
		
		########################  輸出完整報表  ########################
		global points,profit,win,lose,lose_cnt,win_cnt,point_list
		if terminal==True:
			points+=reward
			profit.append(points)
			if reward>0: win+=1
			elif reward<0: lose+=1
            
		if self.DateIndex == 0 and lose!=0 and win!=0:
			lose_cnt.append(lose)
			win_cnt.append(win)
			point_list.append(points)
			plt.plot(lose_cnt,"g");plt.plot(win_cnt,"r");plt.savefig("profit_1.png");plt.show();
			plt.plot(point_list,"b");plt.savefig("profit_2.png");plt.show();
			plt.plot(profit,"c");plt.savefig("profit_3.png");plt.show();
			lose=0
			win=0
			points=0
			profit=[]	
		################################################################		

		return np.array(self.state), reward, terminal, {}

	def reset(self):
		#self.runGame()
		self.state = self.Trade.GetData(self.DateList[0],0)
		return self.state

	def render(self, mode='human'):
		screen_width = 600
		screen_height = 400


		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)

		if self.state is None: return None

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def close(self):
		if self.viewer: self.viewer.close()
