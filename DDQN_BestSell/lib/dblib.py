# -*- coding: utf-8 -*-

import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

END_K_INDEX = 295
STOP_LOSE = 30 #20點停損
TRADE_LOSE = 5 #交易成本
DataCache={}

class timer:
	def __init__(self): 
		self.t=time.clock()
	def spendtime(self,msg=""):
		if msg=="":
			return str(time.clock()-self.t)
		else:
			return msg + " : " +str(time.clock()-self.t) + " secs"
			
class DBConn:
	def __init__(self,host,uid,pwd,cata):
		import win32com.client
		tt=timer()
		self.conn=win32com.client.Dispatch(r"ADODB.Connection")
		self.connstr= "Provider=SQLNCLI.1;Persist Security Info=True;Data Source="+host+";Initial Catalog="+cata+";User ID="+uid+";Password="+pwd+";"
		self.conn.Open(self.connstr)
		print (tt.spendtime("OPDB Conn Time"))
		
class Query:
	def __init__(self,conn):
		import win32com.client
		self.cm = win32com.client.Dispatch(r"ADODB.Command")
		self.cm.CommandType = 1                     #adCmdText     #http://msdn2.microsoft.com/en-us/library/ms962122.aspx
		self.cm.ActiveConnection = conn
		self.cm.ActiveConnection.CursorLocation = 3 #static 可以使用 RecordCount 屬性
	def QueryDB(self,SQL_Str):
		self.cm.CommandText = SQL_Str
		self.cm.Parameters.Refresh()
		self.cm.Prepared = True
		(rs1, result) = self.cm.Execute() 
		return rs1, rs1.recordcount   
	def ExecDB(self,SQL_Str):
		self.cm.CommandText = SQL_Str
		self.cm.Parameters.Refresh()
		self.cm.Prepared = True
		self.cm.Execute()


def seq_intg (x):
	y=np.zeros(x.shape,dtype=x.dtype)
	y[0]=x[0]
	for i in range(len(x)):  #從0開始
		if i>0: y[i]=x[i]+y[i-1]
	return y        
		
def seq_diff (x,x0=False):
	#return np.hstack((0,np.diff(x)))
	#return np.ediff1d(x, to_begin=0)
	y=np.zeros(x.shape,dtype=x.dtype)
	if x0:y[0]=x[0]
	for i in range(len(x)):  #從0開始
		if i>0: y[i]=x[i]-x[i-1]
	return y  



	
class TradeImg:
	def __init__(self,TopN=800,mode='train'):
		self.DateList=[]
		self.Stoplose=STOP_LOSE
		self.SellStopLoseCache={}
		self.fdb = DBConn(host="127.0.0.1",uid="sa",pwd="geniustom",cata="FutureHis")
		self.dbcmd = Query(self.fdb.conn)					
		#today=time.strftime('%y/%m/%d')
		r, rcnt = self.dbcmd.QueryDB("SELECT TDATE FROM (SELECT DISTINCT TDATE FROM RealTimeFuture WHERE TDATE>'13/01/01') as NEW ORDER BY TDATE DESC")
		rr=r.GetRows(rcnt) 
		for i in range(TopN):
			if mode=='train' and i%2==0:
				self.DateList.append(rr[0][TopN-i][:8])
			if mode=='test' and i%2==1:
				self.DateList.append(rr[0][TopN-i][:8])
		self.DateCount=len(self.DateList)
				
	def prepare_data(self):
		for d in self.DateList:
			self.get_data_from_cache(d)
								
	def draw_point(self,data_img,x,y,value,step=256):
		cnt=int(value/step)
		mod=value%step
		for i in range(cnt):
			data_img[x,y-1-i]=255
		data_img[x,(y-1-cnt)%160]=mod
		return data_img
	
	def get_data_from_db(self,date):
		# 價、量、買口、賣口、買筆、賣筆、買成筆、賣成筆
		sqlfield="FutureM_CurPrice,FutureM_Volume,FutureWantM_TrustBuyVol,FutureWantM_TrustSellVol,FutureWantM_TrustBuyCnt,FutureWantM_TrustSellCnt,FutureWantM_TotalBuyCnt,FutureWantM_TotalSellCnt"    #TimeIndex,Contract,Sprice,
		r, rcnt = self.dbcmd.QueryDB("select "+sqlfield+" from RealTimeFuture where TDATE='"+date+"'")
		if rcnt>0: 
			rr=r.GetRows(rcnt)
			data_pr=seq_intg(seq_diff(np.array(rr[0])))
			data_vo=seq_diff(np.array(rr[1]))/4096
			data_bco=seq_diff(np.array(rr[2]))/4096	
			data_sco=seq_diff(np.array(rr[3]))/4096
			data_bbi=seq_diff(np.array(rr[4]))/4096
			data_sbi=seq_diff(np.array(rr[5]))/4096
			data_bsbi=seq_diff(np.array(rr[6]))/4096
			data_ssbi=seq_diff(np.array(rr[7]))/4096
			DataCache[date]=[data_pr,data_vo,data_bco,data_sco,data_bbi,data_sbi,data_bsbi,data_ssbi]
			sell_lose_list=np.zeros(len(data_pr))
			for i in range(len(data_pr)):
				for j in range(i,min(END_K_INDEX,len(data_pr))):
					sell_point=data_pr[j]-data_pr[i]
					if sell_point>=self.Stoplose:
						sell_lose_list[i]=sell_point
						break
			self.SellStopLoseCache[date]=sell_lose_list
			
		
	def get_data_from_cache(self,date):
		if DataCache.__contains__(date)==False or self.SellStopLoseCache.__contains__(date)==False: 
			self.get_data_from_db(date)
			self.get_data_from_db(date)
			print("create cache:",date,"len:",len(DataCache[date][0]))
		
	def GetPrice(self,date):
		self.get_data_from_cache(date)
		return DataCache[date][0]
	
	def GetSellStopLose(self,date):
		self.get_data_from_cache(date)
		return self.SellStopLoseCache[date]
		
	def ShowImg(self,date,index):
		self.get_data_from_cache(date)
		
		data_img=np.zeros([300,160],dtype="uint8")
		w,h=data_img.shape
		base_line=(4,30,51,72,93,114,135,156)
		for i in range(w):
			for j in range(h):
				if j in base_line:
					data_img[i,j]=255
		
		for i in range(index):
			data_img=self.draw_point(data_img,i,base_line[1],DataCache[date][1][i],step=192)		#
			data_img=self.draw_point(data_img,i,base_line[2],DataCache[date][2][i],step=192)
			data_img=self.draw_point(data_img,i,base_line[3],DataCache[date][3][i],step=192)
			data_img=self.draw_point(data_img,i,base_line[4],DataCache[date][4][i],step=192)
			data_img=self.draw_point(data_img,i,base_line[5],DataCache[date][5][i],step=192)
			data_img=self.draw_point(data_img,i,base_line[6],DataCache[date][6][i],step=192)
			data_img=self.draw_point(data_img,i,base_line[7],DataCache[date][7][i],step=192)
		
		return Image.fromarray(np.transpose(data_img),'P')
    
	def GetData(self,date,index):
		#print(np.array(DataCache[date]).shape)        
		data=np.zeros((8,300),dtype=np.array(DataCache[date]).dtype)
		if index>=300:index=299
		for i in range(index):
			data[1][i]=DataCache[date][1][i]
			data[2][i]=DataCache[date][2][i]
			data[3][i]=DataCache[date][3][i]
			data[4][i]=DataCache[date][4][i]
			data[5][i]=DataCache[date][5][i]
			data[6][i]=DataCache[date][6][i]
			data[7][i]=DataCache[date][7][i]
		return np.array(data)
        
if __name__ == '__main__':
	timg=TradeImg(TopN=300,mode='test')
	timg.prepare_data()

	date="17/09/06"
	plt.imshow(timg.ShowImg(date,300)); plt.show();
	plt.plot(timg.GetPrice(date))
	plt.plot(timg.GetSellStopLose(date))
	print(timg.GetData(date,300))
	












'''
class TradeData:
	def __init__(self,conn):
		self.dbconn=conn
		self.dt=Query(conn) 
		self.sqlfield="C_CurPrice,C_TrustBuyCnt,C_TrustSellCnt,C_TrustBuyVol,C_TrustSellVol,C_TotalBuyCnt,C_TotalSellCnt,C_Volume"    #TimeIndex,Contract,Sprice,
		r, rcnt = self.dt.QueryDB("SELECT convert(varchar,TDATE,11) FROM (SELECT DISTINCT TDATE FROM RealTimeOption WHERE Future_CurPrice<>0) AS NEW ORDER BY TDATE")
		rr=r.GetRows(rcnt)   
		#print rr
		self.DateList=[]            #撈db抓到的所有tdate
		self.DateListStart=[]       #對應該tdate的起始索引位置
		self.DateListEnd=[]         #對應該tdate的結束索引位置
		self.DateTXWTag=[]          #周選擇權每天的TAG
		self.DateATMPrice=[]        #周選擇權每天的開盤價平履約價
		self.AllData=[]             #用於回測的所有資料
		for i in range(len(rr[0])):
			self.DateList.append(rr[0][i][:8])
		self.DateCount=rcnt			


cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=127.0.0.1;DATABASE=FutureHis;UID=sa;PWD=geniustom')
cursor = cnxn.cursor()

cursor.execute("select * from RealTimeFuture where TDATE='18/03/08' ")
row = cursor.fetchone()
if row:
	print(row)
'''