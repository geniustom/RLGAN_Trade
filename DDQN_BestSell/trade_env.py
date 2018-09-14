import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

import lib.dblib as db
import matplotlib.pyplot as plt

RUN_DATE_COUNT = 600 #學習過去幾天的資料
WINDOWWIDTH = 300 # 螢幕寬度

lose_cnt=[]
win_cnt=[]
point_list=[]
profit=[]
best_profit=[]
best_points=0
points=0
lose=0
win=0
RewardCache={}	

'''
	下單之後的點x往後找最小值y -> reward= (x-y)/(h-l)
'''
def GetBestShortReward(pr,order_index):
	order_index+=1 #從下一跟K棒起算
	best_reward=0
	best_high_index=0
	best_low_index=0
	for i in range(order_index,len(pr)):
		for j in range(i,len(pr)):
			if pr[i]-pr[j]>best_reward:
				best_reward=pr[i]-pr[j]
				best_high_index=i
				best_low_index=j
				
	return best_reward,best_high_index,best_low_index

def CalReward(date,pr,order_index):
	global RewardCache
	if RewardCache.__contains__(date+str(order_index))==False:
		best_reward,best_high_index,best_low_index=GetBestShortReward(pr,order_index)
		np=pr[order_index]
		reward=pr[order_index]-pr[best_low_index]
		#到最高點之前卻被停損，reward=-停損點
		#往後直到最佳空點與最佳回補點的中間沒有發生停損 reward>0  (不受尾盤拉高的限制)
		for i in range (order_index,best_high_index):
			if pr[i]-np>=db.STOP_LOSE: 
				stoplose=pr[i]-np
				reward=-stoplose
				break
		RewardCache[date+str(order_index)]=[best_reward,best_high_index,best_low_index,reward]
	else:
		best_reward,best_high_index,best_low_index,reward=RewardCache[date+str(order_index)]

	reward-=db.TRADE_LOSE
	best_reward-=db.TRADE_LOSE
	return reward,best_reward

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
		self.gen_cache()
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
		self.Best=[]

	def gen_cache(self):
		import os
		global RewardCache
		if os.path.isfile('reward_data.json'):
			fr = open("reward_data.json",'r+')
			RewardCache = eval(fr.read())
			fr.close()

		for d in self.DateList:
			pr=self.Trade.GetPrice(d)
			print("create reward cache:",d)
			for i in range(len(pr)):
				CalReward(d,pr,i)
		fw = open("reward_data.json",'w+')
		fw.write(str(RewardCache))
		fw.close()


	def process_action(self,action):
		terminal=True
		reward=0
		best_reward=0
		if len(self.Price)>=db.END_K_INDEX : #排除K棒不足的情況
			if action==0: 
				terminal=(self.TimeIndex>=db.END_K_INDEX)
				#if self.TimeIndex>=db.END_K_INDEX: #不下單損失
				#	reward=-db.END_K_INDEX  #-db.STOP_LOSE
			elif action==1:
				reward,best_reward=CalReward(self.Date,self.Price,self.TimeIndex)
				if reward>0: reward=reward#/10 #相當於加強處罰以增加學習成效

		return terminal,reward,best_reward

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		self.Date=self.DateList[self.DateIndex]
		self.Price=self.Trade.GetPrice(self.Date)

		self.state=self.Trade.GetData(self.DateList[self.DateIndex],self.TimeIndex)
		#print(action)
		terminal,reward,best_reward=self.process_action(action)
		
		self.Units.append(self.Price[self.TimeIndex])
		if terminal:    
			########################  輸出單日報表  ########################
#			if RewardCache.__contains__(self.Date+str(self.TimeIndex))==False: CalReward(self.Date,self.Price,self.TimeIndex)
#			best_reward,best_high_index,best_low_index,_=RewardCache[self.Date+str(self.TimeIndex)]
#			for i in range(len(self.Price)):
#				if i<best_high_index: self.Best.append(self.Price[best_high_index])
#				elif i>best_low_index: self.Best.append(self.Price[best_low_index])
#				else: self.Best.append(self.Price[i])
#				
#			#print(self.Date,"reward:",reward,self.game_time.spendtime("time"))
#			plt.cla()
#			plt.plot(self.Best,"b")
#			plt.plot(self.Price,"g")
#			plt.plot(self.Units,"r")
#			plt.savefig("trade.png")    
#			#plt.show();
			################################################################        
			self.Units=[]
			self.Price=[]
			self.Best=[]
			self.game_time=db.timer()
			self.state=self.Trade.GetData(self.DateList[0],0)
		
		
		self.TimeIndex+=1
		if self.TimeIndex>=WINDOWWIDTH or terminal==True : 
			self.DateIndex+=1
			self.TimeIndex=0
		if self.DateIndex>=len(self.DateList):
			self.DateIndex=0
		
		########################  輸出完整報表  ########################
		global points,profit,win,lose,lose_cnt,win_cnt,point_list,best_points,best_profit
		if terminal==True:
			best_points+=best_reward
			best_profit.append(best_points)
			points+=reward
			profit.append(points)
			if reward>0: win+=1
			elif reward<0: lose+=1

		if self.DateIndex == 0 and (lose!=0 or win!=0): #跑完完整的一輪時
			lose_cnt.append(lose)
			win_cnt.append(win)
			point_list.append(points)
			plt.cla()
			plt.plot(lose_cnt,"g");plt.plot(win_cnt,"r");plt.savefig("profit_1.png");plt.show();
			plt.plot(point_list,"b");plt.savefig("profit_2.png");plt.show();
			plt.plot(profit,"c");plt.plot(best_profit,"r");plt.savefig("profit_3.png");plt.show();
			print("best profit:",best_points,"current profit:",points)
			lose=0
			win=0
			points=0
			best_points=0
			profit=[]
			best_profit=[]
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
