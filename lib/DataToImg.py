# -*- coding: utf-8 -*-

import numpy as np
import time

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

def seq_intg (x):
    y=np.zeros(x.shape,dtype=x.dtype)
    y[0]=x[0]
    for i in range(len(x)):  #從0開始
        if i>0: y[i]=x[i]+y[i-1]
    return y        
        
def seq_diff (x):
    #return np.hstack((0,np.diff(x)))
    #return np.ediff1d(x, to_begin=0)
    y=np.zeros(x.shape,dtype=x.dtype)
    for i in range(len(x)):  #從0開始
        if i>0: y[i]=x[i]-x[i-1]
    return y  



fdb = DBConn(host="127.0.0.1",uid="sa",pwd="geniustom",cata="FutureHis")
dbcmd = Query(fdb.conn)					
today=time.strftime('%y/%m/%d')
dbsql=u"select * from RealTimeFuture where TDATE='18/03/08'"

r, rcnt = dbcmd.QueryDB(dbsql)
if rcnt>0: rr=r.GetRows(rcnt)

	
								

'''
cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=127.0.0.1;DATABASE=FutureHis;UID=sa;PWD=geniustom')
cursor = cnxn.cursor()

cursor.execute("select * from RealTimeFuture where TDATE='18/03/08' ")
row = cursor.fetchone()
if row:
    print(row)
'''