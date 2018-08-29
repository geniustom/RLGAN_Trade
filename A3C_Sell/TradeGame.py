# -*- coding: utf-8 -*-
#from BrainDQN_Nature import BrainDQN
import pygame,cv2,numpy
from PIL import Image
import matplotlib.pyplot as plt
#from pygame.locals import *
import lib.dblib as db

WINDOWWIDTH = 300 # 螢幕寬度
WINDOWHEIGHT = 160 # 螢幕高度

ALIVE_REWARD = 0 #-0.05   #存活獎勵
WIN_REWARD = 1    #獎勵
LOSE_REWARD = -1  #懲罰

RUN_DATE_COUNT = 300 #學習過去幾天的資料

# 定義動作
BUY = 'buy'
SELL = 'sell'


# 神經網路的輸出
ACT_STAY =  [1, 0]
ACT_BUY  =  [0, 0]
ACT_SELL =  [0, 1]


class Game:
	def __init__(self,FPS=2000,mode='train'):
		# 定義全域變數
		global FPSCLOCK, BASICFONT,RUN_DATE_COUNT
		pygame.init() 	# 初始化pygame
		pygame.time.set_timer(pygame.USEREVENT, FPS)
		FPSCLOCK = pygame.time.Clock() 	# 獲得pygame時鐘
		FPSCLOCK.tick(FPS) # 設置幀率
		BASICFONT = pygame.font.Font('freesansbold.ttf', 18) # BASICFONT
		self.window = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT)) # 設置螢幕寬高
		self.screen = pygame.display.get_surface()
		self.mode=mode
		self.Date=""
		self.runGame()
		

	def gen_action(self,optfromNN):
		#if optfromNN[0]==1: return ACT_STAY
		#elif optfromNN[1]==1: return ACT_BUY
		#elif optfromNN[2]==1: return ACT_SELL
		#print(optfromNN)
		if optfromNN <=0.5: return ACT_STAY
		elif optfromNN <=1 : return ACT_SELL
		
	def runGame(self):
		self.Trade=db.TradeImg(RUN_DATE_COUNT,mode=self.mode)
		self.Trade.prepare_data()
		self.DateList=self.Trade.DateList
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
			if action==ACT_STAY: 
				terminal=(self.TimeIndex>=db.END_K_INDEX)
				if self.TimeIndex>=db.END_K_INDEX:
					reward=-db.STOP_LOSE
			elif action==ACT_SELL:
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
	
	def step(self, action):
		self.Date=self.DateList[self.DateIndex]
		self.Price=self.Trade.GetPrice(self.Date)
		self.SellStopLose=self.Trade.GetSellStopLose(self.Date)

		img=self.Trade.ShowImg(self.DateList[self.DateIndex],self.TimeIndex)

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
			print(self.Date,"reward:",reward,self.game_time.spendtime("time"))
			plt.plot(self.Price,"g")
			plt.plot(self.Units,"b")
			plt.savefig("trade.png")
			plt.show();
			self.Units=[]
			self.Price=[]
			self.SellStopLose=[]
			self.game_time=db.timer()
			img=self.Trade.ShowImg(self.DateList[0],0)
			
		
		#plt.imshow(img, cmap ='gray'); plt.show();
		img=cv2.cvtColor(numpy.array(img),cv2.COLOR_GRAY2BGR)
		if self.mode=='test':
			pygame.event.pump()
			gimg = pygame.image.fromstring(img.tobytes(), (WINDOWWIDTH,WINDOWHEIGHT), "RGB")
			self.screen.blit(gimg,(0,0))
			pygame.display.set_caption(self.Date) # 設置視窗的標題
			pygame.display.update()			
		return img, reward, terminal
	
	def getCurrentFrame(self):
		image_data = pygame.surfarray.array3d(pygame.display.get_surface())
		return image_data
		
		
'''		
# preprocess raw image to 80*80 gray image
def preprocess(observation,first=False):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
	#ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	#plt.imshow(observation, cmap ='gray'); plt.show();
	if first:
		return observation
	else:
		return numpy.reshape(observation,(80,80,1))
		
def playGame():
	# Step 0: Define report
	win = 0
	lose = 0
	points = 0
	cnt = 0
	lose_cnt=[]
	win_cnt=[]
	point_list=[]
	profit=[]
	# Step 1: init BrainDQN
	actions = 3
	brain = BrainDQN(actions)
	# Step 2: init Game
	bg = Game()
	# Step 3: play game
	# Step 3.1: obtain init state
	action0 = bg.gen_action(ACT_STAY)  # do nothing
	observation0, reward0, terminal = bg.step(action0)
	observation0 = preprocess(observation0,first=True)
	brain.setInitState(observation0)
	tt=db.timer()
	
	# Step 3.2: run the game
	while True:
		pygame.event.get()  #讓遊戲畫面能夠更新
		action = bg.gen_action(brain.getAction())
		#action = bg.gen_action(ACT_STAY)  # do nothing
		Observation,reward,terminal = bg.step(action)
		nextObservation = preprocess(Observation)
		brain.setPerception(nextObservation,action,reward,terminal)
		cnt+=1
		
		Observation=None
		nextObservation=None
		########################  統計輸出報表用  ########################
		if terminal==True:
			points+=reward
			profit.append(points)
			#print (tt.spendtime("Spend Time"))
			tt=db.timer()
			if reward>0:
				win+=1
			else:
				lose+=1
			#print("run:",cnt,"brain cnt:",brain.timeStep,"date:",bg.Trade.date ,"reward:",reward)
			#bg = Game()

			if bg.DateIndex == 0:
				lose_cnt.append(lose)
				win_cnt.append(win)
				point_list.append(points)
				plt.plot(lose_cnt,"g");plt.plot(win_cnt,"r");plt.show();
				plt.plot(point_list,"b");plt.show();
				plt.plot(profit,"c");plt.show();
				print("run:",cnt,"brain cnt:",brain.timeStep,"date:",bg.Trade.date ,"profit:",points)
				lose=0
				win=0
				points=0
				profit=[]
			########################  統計輸出報表用  ########################		


		
def main():
	playGame()
	
if __name__ == '__main__':
	main()
'''