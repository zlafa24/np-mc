'''
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
   
def basicPlot(steps,ys,xlabel,ylabel,title,img_name,labels=[False]):
    
    fig1, ax1 = plt.subplots()
    for i,y in enumerate(ys):
        ax1.plot(steps,y,label=labels[i])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    if labels[0]:
        ax1.legend(loc='upper right',frameon=False)
    fig1.suptitle(title,y=1.0)
    fig1.tight_layout(rect=[0, 0, 1, 0.98])
    fig1.savefig(f'{img_name}.png',dpi=100)
    
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
    basicPlot(steps,[energies],'Simulation Time (# of MC steps)','Potential Energy (kcal/mol)','Total Potential Energy','test_nrg')
    basicPlot(steps,[rates1,rates2,rates3,rates4],'Simulation Time (# of MC steps)','Acceptance Ratio (# accepted / total)','Acceptance Rate by Move Type','test_rates',['Regrowth','Trans.','Swap','Rot.'])
            
def checkProgress_basic(cwd,energy_file='Potential_Energy.txt',acc_file='Acceptance_Rate.txt'):
    
    energy_data = np.genfromtxt(energy_file,skip_header=1,usecols=[0,1])
    acc_data = np.genfromtxt(acc_file,skip_header=1,usecols=range(9))
    
    steps = energy_data[:,0]
    energies = energy_data[:,1]
    rates1 = acc_data[:,2]
    rates2 = acc_data[:,4]
    rates3 = acc_data[:,6]
    rates4 = acc_data[:,8]
    basicPlot(steps,[energies],'Simulation Time (# of MC steps)','Potential Energy (kcal/mol)','Total Potential Energy','test_nrg')
    basicPlot(steps,[rates1,rates2,rates3,rates4],'Simulation Time (# of MC steps)','Acceptance Ratio (# accepted / total)','Acceptance Rate by Move Type','test_rates',['Regrowth','Trans.','Swap','Rot.'])            
       
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
    
def get_distance(atom1,atom2):
    
    return np.sqrt(np.sum(np.square(atom1-atom2)))

def morse(D,a,r0,r):
    
    return D*(np.exp(-2*a*(r-r0))-2*np.exp(-a*(r-r0)))
    
def check_minimize(max_dist = 3.75, min_dist = 1.4):
    
    data = np.genfromtxt(f'trajectory.xyz', skip_header=2)
    typs = data[:,0]
    xyzs = data[:,1:]
    distance1 = []
    potential1 = []
    distance2 = []
    potential2 = []
    ag_idxs = np.where(typs == 1)[0]
    s_idxs = np.where(typs==5)[0]
    D = 5.58
    a = 0.746
    r0 = 2.868

    for i, s_idx in enumerate(s_idxs):
        distance = []
        for j, ag_idx in enumerate(ag_idxs):
            d = get_distance(xyzs[s_idx,:], xyzs[ag_idx,:])
            distance.append(d)
        min_distance = np.min(distance)
        potential = morse(D,a,r0,min_distance)
        distance1.append(min_distance)
        potential1.append(potential)
        distance.clear()
    for i, s_idx in enumerate(s_idxs):
        m_potential = []
        m_distance = []
        for j, ag_idx in enumerate(ag_idxs):
            d = get_distance(xyzs[s_idx,:], xyzs[ag_idx,:])
            m = morse(D,a,r0,d)
            m_potential.append(m)
            m_distance.append(d)
        min_potential2 = np.min(m_potential)
        index_min =np.argmin(m_potential)
        min_distance2 = m_distance[index_min]
        distance2.append(min_distance2)
        potential2.append(min_potential2)
        m_potential.clear()
        m_potential.clear()
    close = 0
    far = 0
    for i in range(len(distance1)):
        if distance1[i] < min_dist:
            close +=1
    print(close)
    for i in range(len(distance2)):
        if distance2[i] > max_dist:
            far +=1
    print(far)
    if close != 0 or far != 0:
        raise Exception('atoms are not placed correctly')

def plot_minimize(len1, len2, size = 'np40'):
    data = np.genfromtxt(f'trajectory.xyz', skip_header=2)
    typs = data[:,0]
    xyzs = data[:,1:]
    distance1 = []
    potential1 = []
    distance2 = []
    potential2 = []
    ag_idxs = np.where(typs == 1)[0]
    s_idxs = np.where(typs==5)[0]
    D = 5.58
    a = 0.746
    r0 = 2.868

    for i, s_idx in enumerate(s_idxs):
        distance = []
        for j, ag_idx in enumerate(ag_idxs):
            d = get_distance(xyzs[s_idx,:], xyzs[ag_idx,:])
            distance.append(d)
        min_distance = np.min(distance)
        potential = morse(D,a,r0,min_distance)
        distance1.append(min_distance)
        potential1.append(potential)
        distance.clear()
    for i, s_idx in enumerate(s_idxs):
        m_potential = []
        m_distance = []
        for j, ag_idx in enumerate(ag_idxs):
            d = get_distance(xyzs[s_idx,:], xyzs[ag_idx,:])
            m = morse(D,a,r0,d)
            m_potential.append(m)
            m_distance.append(d)
        min_potential2 = np.min(m_potential)
        index_min =np.argmin(m_potential)
        min_distance2 = m_distance[index_min]
        distance2.append(min_distance2)
        potential2.append(min_potential2)
        m_potential.clear()
        m_potential.clear()
    r = [1.5]
    for i in range(45):
        r.append(r[i]+.1)
    e = []
    for i in range(len(r)):
        e.append(morse(D,a,r0,r[i]))

    fig1, ax1 = plt.subplots()
    ax1.scatter(distance1, potential1, color = 'powderblue')
    ax1.scatter(distance2, potential2, color = 'lightseagreen')
    ax1.plot(r, e, color = 'seagreen')
    ax1.set_xlabel('Internuclear Separation')
    ax1.set_ylabel('Energy')
    fig1.suptitle(f'Morse Potential {size} C{len1}/C{len2}')
    ax1.legend(['Theoretical Morse Potential S-Ag','Closest S-Ag Pair', 'Lowest Morse Potential'])
    fig1.savefig(f'{size}_C{len1}_C{len2}_morse_pot.png', dpi = 400) 
    '''