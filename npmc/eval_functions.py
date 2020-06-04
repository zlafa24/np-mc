from __future__ import print_function
import pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import time
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

def getSheet(token_file=os.environ['GOOGLE_TOKEN']):

    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    try:
        with open(token_file,'rb') as token:
            creds = pickle.load(token)
    except FileNotFoundError:
        flow = InstalledAppFlow.from_client_secrets_file('/sfs/qumulo/qhome/jrs9wf/google_credentials/credentials.json',SCOPES)
        creds = flow.run_console()
        with open('/sfs/qumulo/qhome/jrs9wf/google_credentials/token.pickle','wb') as token:
            pickle.dump(creds,token)
    service = build('sheets','v4',credentials=creds)
    return service.spreadsheets()
    

def writeHeaders(sheet,sheetID):

    body = {
        'values': [['Job ID','Job Name','Directory Location','Date and Time of Submission','Expected Duration','Expected Date and Time of Completion',
            'Status','Last Checked Step','Time of Last Check']]
    }
    result = sheet.values().update(
            spreadsheetId=sheetID,range='Sheet1!A1:1',
            valueInputOption='RAW',body=body).execute()
    
    
def getHeaders(sheet,sheetID):    

    columns_dict = sheet.values().get(
    spreadsheetId=sheetID,range='Sheet1!1:1').execute()
    headers = columns_dict.get('values',[])
    return np.array(headers)


def getRowsforColumn(sheet,sheetID,column):    

    rows_dict = sheet.values().get(
    spreadsheetId=sheetID,range=f'Sheet1!{column}:{column}').execute()
    rows = rows_dict.get('values',[])
    return np.array(rows)
   
def basicPlot(steps,ys,img_name):
    
    fig1, ax1 = plt.subplots()
    for y in ys:
        ax1.plot(steps,y)
    fig1.savefig(f'{img_name}.png')
    
def pushStart(cwd,slurm_file,jobID,sheetID=os.environ['SIMLOG_SHEETID']):

    sheet = getSheet()
    job_info_array = np.genfromtxt(f'{cwd}/{slurm_file}',skip_header=2,max_rows=8,usecols=(1,2),dtype=None,comments=None,encoding=None)
    job_dict = dict(job_info_array)
    
    if 'scratch' in cwd: cwd = cwd[cwd.find('scratch')-1:]
    start = time.localtime(time.time())
    start_time = dt.datetime(start[0],start[1],start[2],hour=start[3],minute=start[4])
    duration = dt.timedelta(days=int(job_dict['-t'][0]),hours=int(job_dict['-t'][2:4]),minutes=int(job_dict['-t'][5:7]))

    headers = getHeaders(sheet,sheetID)
    if headers.size == 0: 
        writeHeaders(sheet,sheetID)
        headers = getHeaders(sheet,sheetID)
        
    ID_column = chr(ord('@')+1+int(np.where(headers[0]=='Job ID')[0][0]))
    ID_data = {
        'values': [[jobID]]
    }  
    result = sheet.values().append(
        spreadsheetId=sheetID, range=f'Sheet1!{ID_column}1', 
        valueInputOption='RAW', body=ID_data).execute()
    
    rows = getRowsforColumn(sheet,sheetID,ID_column)
    row = str(int(np.where(rows[:,0]==jobID)[0][0])+1)
    slurm_data = [
        {   
            'range': f'Sheet1!'+chr(ord('@')+1+int(np.where(headers[0]=='Job Name')[0][0]))+row,
            'values':[[job_dict['-J']]]
        },
        {
            'range': f'Sheet1!'+chr(ord('@')+1+int(np.where(headers[0]=='Directory Location')[0][0]))+row,
            'values': [[cwd]]
        },
        {   
            'range': f'Sheet1!'+chr(ord('@')+1+int(np.where(headers[0]=='Date and Time of Submission')[0][0]))+row,
            'values': [[start_time.strftime("%A, %d %B %Y %I:%M%p")]]
        },
        {   
            'range': f'Sheet1!'+chr(ord('@')+1+int(np.where(headers[0]=='Expected Duration')[0][0]))+row,
            'values': [[f"{job_dict['-t'][0]} days {int(job_dict['-t'][2:4])} hours {int(job_dict['-t'][5:7])} minutes"]]
        },
        {   
            'range': f'Sheet1!'+chr(ord('@')+1+int(np.where(headers[0]=='Expected Date and Time of Completion')[0][0]))+row,
            'values': [[(start_time+duration).strftime("%A, %d %B %Y %I:%M%p")]]
        },
        {   
            'range': f'Sheet1!'+chr(ord('@')+1+int(np.where(headers[0]=='Status')[0][0]))+row,
            'values': [['In Progress']]
        },
        {   
            'range': f'Sheet1!'+chr(ord('@')+1+int(np.where(headers[0]=='Last Checked Step')[0][0]))+row,
            'values': [[0]]
        }
    ]
    body = {
        'valueInputOption': 'RAW',
        'data': slurm_data
    }
    result = sheet.values().batchUpdate(spreadsheetId=sheetID,body=body).execute()


def checkProgress(cwd,slurm_file,jobID,sheetID=os.environ['SIMLOG_SHEETID']):

    sheet = getSheet()
    energy_file = 'Potential_Energy.txt'
    acc_file = 'Acceptance_Rate.txt'
    
    start = time.localtime(time.time())
    start_time = dt.datetime(start[0],start[1],start[2],hour=start[3],minute=start[4]) 
    energy_data = np.genfromtxt(energy_file,skip_header=1,usecols=[0,1])
    acc_data = np.genfromtxt(acc_file,skip_header=1,usecols=range(9))
    last_step = str(int(energy_data[-1,0]))
    
    headers = getHeaders(sheet,sheetID)
    ID_column = chr(ord('@')+1+int(np.where(headers[0]=='Job ID')[0][0]))
    rows = getRowsforColumn(sheet,sheetID,ID_column)
    row = str(int(np.where(rows[:,0]==jobID)[0][0])+1)
    
    slurm_data = [
        {   
            'range': f'Sheet1!'+chr(ord('@')+1+int(np.where(headers[0]=='Last Checked Step')[0][0]))+row,
            'values':[[last_step]]
        },
        {   
            'range': f'Sheet1!'+chr(ord('@')+1+int(np.where(headers[0]=='Time of Last Check')[0][0]))+row,
            'values':[[start_time.strftime("%A, %d %B %Y %I:%M%p")]]
        }
    ]
    body = {
        'valueInputOption': 'RAW',
        'data': slurm_data
    }
    result = sheet.values().batchUpdate(spreadsheetId=sheetID,body=body).execute()
    
    steps = energy_data[:,0]
    energies = energy_data[:,1]
    rates1 = acc_data[:,2]
    rates2 = acc_data[:,4]
    rates3 = acc_data[:,6]
    rates4 = acc_data[:,8]
    basicPlot(steps,[energies],'test_nrg')
    basicPlot(steps,[rates1,rates2,rates3,rates4],'test_rates')
    
    
def pushEnd(cwd,slurm_file,jobID,sheetID=os.environ['SIMLOG_SHEETID']):
    
    sheet = getSheet()
    headers = getHeaders(sheet,sheetID)
    ID_column = chr(ord('@')+1+int(np.where(headers[0]=='Job ID')[0][0]))
    rows = getRowsforColumn(sheet,sheetID,ID_column)
    row = str(int(np.where(rows[:,0]==jobID)[0][0])+1)
    
    slurm_data = [
        {   
            'range': f'Sheet1!'+chr(ord('@')+1+int(np.where(headers[0]=='Status')[0][0]))+row,
            'values':[['Completed']]
        },
    ]
    body = {
        'valueInputOption': 'RAW',
        'data': slurm_data
    }
    result = sheet.values().batchUpdate(spreadsheetId=sheetID,body=body).execute()

    
    
    
    
    