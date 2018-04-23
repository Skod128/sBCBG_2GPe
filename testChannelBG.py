#!/apps/free/python/2.7.10/bin/python
# -*- coding: utf-8 -*-    


#import nstrand
from LGneurons import *
from modelParams import *
import nest.raster_plot
import os
import numpy as np
#import time
import sys
import matplotlib.pyplot as plt
import math
#------------------------------------------
# Creates the populations of neurons necessary to simulate a BG circuit
#------------------------------------------
def createBG_MC():
  #==========================
  # Creation of neurons
  #-------------------------
  print '\nCreating neurons\n================'
  for N in NUCLEI:
      nbSim[N] = params['nb'+N]
      createMC(N,params['nbCh'])
      for i in range(len(Pop[N])):
          nest.SetStatus(Pop[N][i],{"I_e":params['Ie'+N]})

  nbSim['CSN'] = params['nbCSN']
  createMC('CSN',params['nbCh'], fake=True, parrot=True)
  nbSim['PTN'] = params['nbPTN']
  createMC('PTN',params['nbCh'], fake=True, parrot=True)
  nbSim['CMPf'] = params['nbCMPf']
  createMC('CMPf',params['nbCh'], fake=True, parrot=params['parrotCMPf'])

  print "Number of simulated neurons:", nbSim

#------------------------------------------
# Connects the populations of a previously created multi-channel BG circuit 
#------------------------------------------
def connectBG_MC(antagInjectionSite,antag):
  G = {'MSN': params['GMSN'],
       'FSI': params['GFSI'],
       'STN': params['GSTN'],
       'GTA': params['GGTA'],
       'GTI': params['GGTI'],
       'GPi': params['GGPi'],}

  print "Gains on LG14 syn. strength:", G

  #-------------------------
  # connection of populations
  #-------------------------
  print '\nConnecting neurons\n================'
  print "**",antag,"antagonist injection in",antagInjectionSite,"**"
  
  print '* MSN Inputs'
  connectMC('ex','CSN','MSN', params['cTypeCSNMSN'], inDegree= params['inDegCSNMSN'], gain=G['MSN'])
  connectMC('ex','PTN','MSN', params['cTypePTNMSN'], inDegree= params['inDegPTNMSN'], gain=G['MSN'])
  connectMC('ex','CMPf','MSN',params['cTypeCMPfMSN'],inDegree= params['inDegCMPfMSN'],gain=G['MSN'])
  connectMC('in','MSN','MSN', params['cTypeMSNMSN'], inDegree= params['inDegMSNMSN'], gain=G['MSN'])
  connectMC('in','FSI','MSN', params['cTypeFSIMSN'], inDegree= params['inDegFSIMSN'], gain=G['MSN'])
  # some parameterizations from LG14 have no STN->MSN or GTA->MSN synaptic contacts
  if alpha['STN->MSN'] != 0:
    print "alpha['STN->MSN']",alpha['STN->MSN']
    connectMC('ex','STN','MSN', params['cTypeSTNMSN'], inDegree= params['inDegSTNMSN'], gain=G['MSN'])
  if alpha['GTA->MSN'] != 0:
    print "alpha['GTA->MSN']",alpha['GTA->MSN']
    connectMC('in','GTA','MSN', params['cTypeGTAMSN'], inDegree= params['inDegGTAMSN'], gain=G['MSN'])

  print '* FSI Inputs'
  connectMC('ex','CSN','FSI', params['cTypeCSNFSI'], inDegree= params['inDegCSNFSI'], gain=G['FSI'])
  connectMC('ex','PTN','FSI', params['cTypePTNFSI'], inDegree= params['inDegPTNFSI'], gain=G['FSI'])
  if alpha['STN->FSI'] != 0:
    connectMC('ex','STN','FSI', params['cTypeSTNFSI'],inDegree= params['inDegSTNFSI'],gain=G['FSI'])
  connectMC('in','GTA','FSI', params['cTypeGTAFSI'], inDegree= params['inDegGTAFSI'], gain=G['FSI'])
  connectMC('ex','CMPf','FSI',params['cTypeCMPfFSI'],inDegree= params['inDegCMPfFSI'],gain=G['FSI'])
  connectMC('in','FSI','FSI', params['cTypeFSIFSI'], inDegree= params['inDegFSIFSI'], gain=G['FSI'])

  print '* STN Inputs'
  connectMC('ex','PTN','STN', params['cTypePTNSTN'], inDegree= params['inDegPTNSTN'],  gain=G['STN'])
  connectMC('ex','CMPf','STN',params['cTypeCMPfSTN'],inDegree= params['inDegCMPfSTN'], gain=G['STN'])
  connectMC('in','GTI','STN', params['cTypeGTISTN'], inDegree= params['inDegGTISTN'],  gain=G['STN'])

  
  if antagInjectionSite == 'GPe':
    if   antag == 'AMPA':
      print '* GTA Inputs'
      connectMC('NMDA','CMPf','GTA',params['cTypeCMPfGTA'],inDegree= params['inDegCMPfGTA'],gain=G['GTA'])
      connectMC('NMDA','STN','GTA', params['cTypeSTNGTA'], inDegree= params['inDegSTNGTA'], gain=G['GTA'])
      connectMC('in','MSN','GTA',   params['cTypeMSNGTA'], inDegree= params['inDegMSNGTA'], gain=G['GTA'])
      connectMC('in','GTA','GTA',   params['cTypeGTAGTA'], inDegree= params['inDegGTAGTA'], gain=G['GTA'])
      connectMC('in','GTI','GTA',   params['cTypeGTIGTA'], inDegree= params['inDegGTIGTA'], gain=G['GTA'])
      print '* GTI Inputs'
      connectMC('NMDA','CMPf','GTI',params['cTypeCMPfGTI'],inDegree= params['inDegCMPfGTI'],gain=G['GTI'])
      connectMC('NMDA','STN','GTI', params['cTypeSTNGTI'], inDegree= params['inDegSTNGTI'], gain=G['GTI'])
      connectMC('in','MSN','GTI',   params['cTypeMSNGTI'], inDegree= params['inDegMSNGTI'], gain=G['GTI'])
      connectMC('in','GTI','GTI',   params['cTypeGTIGTI'], inDegree= params['inDegGTIGTI'], gain=G['GTI'])
      connectMC('in','GTA','GTI',   params['cTypeGTAGTI'], inDegree= params['inDegGTAGTI'], gain=G['GTI'])
      
    elif antag == 'NMDA':
      print '* GTA Inputs'
      connectMC('AMPA','CMPf','GTA',params['cTypeCMPfGTA'],inDegree= params['inDegCMPfGTA'],gain=G['GTA'])
      connectMC('AMPA','STN','GTA', params['cTypeSTNGTA'], inDegree= params['inDegSTNGTA'], gain=G['GTA'])
      connectMC('in','MSN','GTA',   params['cTypeMSNGTA'], inDegree= params['inDegMSNGTA'], gain=G['GTA'])
      connectMC('in','GTA','GTA',   params['cTypeGTAGTA'], inDegree= params['inDegGTAGTA'], gain=G['GTA'])
      connectMC('in','GTI','GTA',   params['cTypeGTIGTA'], inDegree= params['inDegGTIGTA'], gain=G['GTA'])
      print '* GTI Inputs'
      connectMC('AMPA','CMPf','GTI',params['cTypeCMPfGTI'],inDegree= params['inDegCMPfGTI'],gain=G['GTI'])
      connectMC('AMPA','STN','GTI', params['cTypeSTNGTI'], inDegree= params['inDegSTNGTI'], gain=G['GTI'])
      connectMC('in','MSN','GTI',   params['cTypeMSNGTI'], inDegree= params['inDegMSNGTI'], gain=G['GTI'])
      connectMC('in','GTI','GTI',   params['cTypeGTIGTI'], inDegree= params['inDegGTIGTI'], gain=G['GTI'])
      connectMC('in','GTA','GTI',   params['cTypeGTAGTI'], inDegree= params['inDegGTAGTI'], gain=G['GTI'])
    elif antag == 'AMPA+GABAA':
      print '* GTA Inputs'
      connectMC('NMDA','CMPf','GTA',params['cTypeCMPfGTA'],inDegree= params['inDegCMPfGTA'],gain=G['GTA'])
      connectMC('NMDA','STN','GTA', params['cTypeSTNGTA'], inDegree= params['inDegSTNGTA'], gain=G['GTA'])
      print '* GTI Inputs'
      connectMC('NMDA','CMPf','GTI',params['cTypeCMPfGTI'],inDegree= params['inDegCMPfGTI'],gain=G['GTI'])
      connectMC('NMDA','STN','GTI', params['cTypeSTNGTI'], inDegree= params['inDegSTNGTI'], gain=G['GTI'])
      
    elif antag == 'GABAA':
      print '* GTA Inputs'
      connectMC('ex','CMPf','GTA',params['cTypeCMPfGTA'],inDegree= params['inDegCMPfGTA'],gain=G['GTA'])
      connectMC('ex','STN','GTA', params['cTypeSTNGTA'], inDegree= params['inDegSTNGTA'], gain=G['GTA'])
      print '* GTI Inputs'
      connectMC('ex','CMPf','GTI',params['cTypeCMPfGTI'],inDegree= params['inDegCMPfGTI'],gain=G['GTI'])
      connectMC('ex','STN','GTI', params['cTypeSTNGTI'], inDegree= params['inDegSTNGTI'], gain=G['GTI'])
      
    else:
      print antagInjectionSite,": unknown antagonist experiment:",antag    
    
  else:
    print '* GTA Inputs'
    connectMC('ex','CMPf','GTA',params['cTypeCMPfGTA'],inDegree= params['inDegCMPfGTA'],gain=G['GTA'])
    connectMC('ex','STN','GTA', params['cTypeSTNGTA'], inDegree= params['inDegSTNGTA'], gain=G['GTA'])
    connectMC('in','MSN','GTA', params['cTypeMSNGTA'], inDegree= params['inDegMSNGTA'], gain=G['GTA'])
    connectMC('in','GTA','GTA', params['cTypeGTAGTA'], inDegree= params['inDegGTAGTA'], gain=G['GTA'])
    connectMC('in','GTI','GTA', params['cTypeGTIGTA'], inDegree= params['inDegGTIGTA'], gain=G['GTA'])
    print '* GTI Inputs'
    connectMC('ex','CMPf','GTI',params['cTypeCMPfGTI'],inDegree= params['inDegCMPfGTI'],gain=G['GTI'])
    connectMC('ex','STN','GTI', params['cTypeSTNGTI'], inDegree= params['inDegSTNGTI'], gain=G['GTI'])
    connectMC('in','MSN','GTI', params['cTypeMSNGTI'], inDegree= params['inDegMSNGTI'], gain=G['GTI'])
    connectMC('in','GTI','GTI', params['cTypeGTIGTI'], inDegree= params['inDegGTIGTI'], gain=G['GTI'])
    connectMC('in','GTA','GTI',   params['cTypeGTAGTI'], inDegree= params['inDegGTAGTI'], gain=G['GTI'])

  print '* GPi Inputs'
  if antagInjectionSite =='GPi':
    if   antag == 'AMPA+NMDA+GABAA':
      pass
    elif antag == 'NMDA':
      connectMC('in','MSN','GPi',   params['cTypeMSNGPi'], inDegree= params['inDegMSNGPi'], gain=G['GPi'])
      connectMC('AMPA','STN','GPi', params['cTypeSTNGPi'], inDegree= params['inDegSTNGPi'], gain=G['GPi'])
      connectMC('in','GTI','GPi',   params['cTypeGTIGPi'], inDegree= params['inDegGTIGPi'], gain=G['GPi'])
      connectMC('AMPA','CMPf','GPi',params['cTypeCMPfGPi'],inDegree= params['inDegCMPfGPi'],gain=G['GPi'])
    elif antag == 'NMDA+AMPA':
      connectMC('in','MSN','GPi', params['cTypeMSNGPi'],inDegree= params['inDegMSNGPi'], gain=G['GPi'])
      connectMC('in','GTI','GPi', params['cTypeGTIGPi'],inDegree= params['inDegGTIGPi'], gain=G['GPi'])
    elif antag == 'AMPA':
      connectMC('in','MSN','GPi',   params['cTypeMSNGPi'], inDegree= params['inDegMSNGPi'], gain=G['GPi'])
      connectMC('NMDA','STN','GPi', params['cTypeSTNGPi'], inDegree= params['inDegSTNGPi'], gain=G['GPi'])
      connectMC('in','GTI','GPi',   params['cTypeGTIGPi'], inDegree= params['inDegGTIGPi'], gain=G['GPi'])
      connectMC('NMDA','CMPf','GPi',params['cTypeCMPfGPi'],inDegree= params['inDegCMPfGPi'],gain=G['GPi'])
    elif antag == 'GABAA':
      connectMC('ex','STN','GPi', params['cTypeSTNGPi'], inDegree= params['inDegSTNGPi'], gain=G['GPi'])
      connectMC('ex','CMPf','GPi',params['cTypeCMPfGPi'],inDegree= params['inDegCMPfGPi'],gain=G['GPi'])
    else:
      print antagInjectionSite,": unknown antagonist experiment:",antag
  else:
    connectMC('in','MSN','GPi', params['cTypeMSNGPi'], inDegree= params['inDegMSNGPi'], gain=G['GPi'])
    connectMC('ex','STN','GPi', params['cTypeSTNGPi'], inDegree= params['inDegSTNGPi'], gain=G['GPi'])
    connectMC('in','GTI','GPi', params['cTypeGTIGPi'], inDegree= params['inDegGTIGPi'], gain=G['GPi'])
    connectMC('ex','CMPf','GPi',params['cTypeCMPfGPi'],inDegree= params['inDegCMPfGPi'],gain=G['GPi'])

#------------------------------------------
# Checks that the BG model parameterization defined by the "params" dictionary can respect the electrophysiological constaints (firing rate at rest).
# If testing for a given antagonist injection experiment, specifiy the injection site in antagInjectionSite, and the type of antagonists used in antag.
# Returns [score obtained, maximal score]
# params possible keys:
# - nb{MSN,FSI,STN,GPi,GPe,CSN,PTN,CMPf} : number of simulated neurons for each population
# - Ie{GPe,GPi} : constant input current to GPe and GPi
# - G{MSN,FSI,STN,GPi,GPe} : gain to be applied on LG14 input synaptic weights for each population
#------------------------------------------

def checkAvgFR(showRasters=False,params={},antagInjectionSite='none',antag='',logFileName=''):
  nest.ResetKernel()
  dataPath='log/'
  nest.SetKernelStatus({'local_num_threads': params['nbcpu'] if ('nbcpu' in params) else 2, "data_path": dataPath})
  initNeurons()

  offsetDuration = 1000.
  # nest.SetKernelStatus({"overwrite_files":True}) # Thanks to use of timestamps, file names should now 
                                                   # be different as long as they are not created during the same second

  print '/!\ Using the following LG14 parameterization',params['LG14modelID']
  loadDictParams(params['LG14modelID'])

  # We check that all the necessary parameters have been defined. They should be in the modelParams.py file.
  # If one of them misses, we exit the program.
  necessaryParams=['nbCh','nbMSN','nbFSI','nbSTN','nbGTI','nbGTA','nbGPi','nbCSN','nbPTN','nbCMPf',
                   'IeMSN','IeFSI','IeSTN','IeGTI','IeGTA','IeGPi',
                   'GMSN','GFSI','GSTN','GGTI','GGTA','GGPi',
                   'inDegCSNMSN','inDegPTNMSN','inDegCMPfMSN','inDegMSNMSN','inDegFSIMSN','inDegSTNMSN','inDegGTAMSN',
                   'inDegCSNFSI','inDegPTNFSI','inDegSTNFSI','inDegGTAFSI','inDegCMPfFSI','inDegFSIFSI',
                   'inDegPTNSTN','inDegCMPfSTN','inDegGTISTN',
                   'inDegCMPfGTA','inDegSTNGTA','inDegMSNGTA','inDegGTAGTA','inDegGTIGTA',
                   'inDegCMPfGTI','inDegSTNGTI','inDegMSNGTI','inDegGTIGTI',
                   'inDegMSNGPi','inDegSTNGPi','inDegGTIGPi','inDegCMPfGPi',]
  
  for np in necessaryParams:
    if np not in params:
      print "Missing parameter:",np 
      exit()

  #-------------------------
  # creation and connection of the neural populations
  #-------------------------
  createBG_MC()
  connectBG_MC(antagInjectionSite,antag)

  #-------------------------
  # measures
  #-------------------------
  spkDetect={} # spike detectors used to record the experiment
  expeRate={}

  antagStr = ''
  if antagInjectionSite != 'none':
    antagStr = antagInjectionSite+'_'+antag+'_'

  for N in NUCLEI:
    # 1000ms offset period for network stabilization
    spkDetect[N] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": antagStr+N, "to_memory": False, "to_file": True, 'start':offsetDuration,'stop':offsetDuration+params['tSimu']})
    for i in range(len(Pop[N])):
      nest.Connect(Pop[N][i], spkDetect[N])

  spkDetect['CMPf'] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": antagStr+'CMPf', "to_memory": False, "to_file": True, 'start':offsetDuration,'stop':offsetDuration+params['tSimu']})
  for i in range(len(Pop['CMPf'])):
    nest.Connect(Pop['CMPf'][i], spkDetect['CMPf'])

  #-------------------------
  # Simulation
  #-------------------------
  nest.Simulate(params['tSimu']+offsetDuration)

  score = 0

  text=[]
  frstr = antagInjectionSite + ', '
  s = '----- RESULTS -----'
  print s
  text.append(s+'\n')
  if antagInjectionSite == 'none':
    for N in NUCLEI:
      strTestPassed = 'NO!'
      expeRate[N] = nest.GetStatus(spkDetect[N], 'n_events')[0] / float(nbSim[N]*params['tSimu']*params['nbCh']) * 1000
      if expeRate[N] <= FRRNormal[N][1] and expeRate[N] >= FRRNormal[N][0]:
        # if the measured rate is within acceptable values
        strTestPassed = 'OK'
        score += 1
      frstr += '%f , ' %(expeRate[N])
      s = '* '+N+' - Rate: '+str(expeRate[N])+' Hz -> '+strTestPassed+' ('+str(FRRNormal[N][0])+' , '+str(FRRNormal[N][1])+')'
      print s
      text.append(s+'\n')
  else:
    for N in NUCLEI:
      expeRate[N] = nest.GetStatus(spkDetect[N], 'n_events')[0] / float(nbSim[N]*params['tSimu']*params['nbCh']) * 1000
      if N == antagInjectionSite:
        strTestPassed = 'NO!'
        if expeRate[N] <= FRRAnt[N][antag][1] and expeRate[N] >= FRRAnt[N][antag][0]:
          # if the measured rate is within acceptable values
          strTestPassed = 'OK'
          score += 1
        s = '* '+N+' with '+antag+' antagonist(s): '+str(expeRate[N])+' Hz -> '+strTestPassed+' ('+str(FRRAnt[N][antag][0])+' , '+str(FRRAnt[N][antag][1])+')'
        print s
        text.append(s+'\n')
      else:
        s = '* '+N+' - Rate: '+str(expeRate[N])+' Hz'
        print s
        text.append(s+'\n')
      frstr += '%f , ' %(expeRate[N])

  s = '-------------------'
  print s
  text.append(s+'\n')

  frstr+='\n'
  firingRatesFile=open(dataPath+'firingRates.csv','a')
  firingRatesFile.writelines(frstr)
  firingRatesFile.close()

  #print "************************************** file writing",text
  #res = open(dataPath+'OutSummary_'+logFileName+'.txt','a')
  res = open(dataPath+'OutSummary.txt','a')
  res.writelines(text)
  res.close()

  #-------------------------
  # Displays
  #-------------------------
  if showRasters and interactive:
    displayStr = ' ('+antagStr[:-1]+')' if (antagInjectionSite != 'none') else ''
    for N in NUCLEI:
      #nest.raster_plot.from_device(spkDetect[N],hist=True,title=N+displayStr)
      nest.raster_plot.from_device(spkDetect[N],hist=False,title=N+displayStr)

    nest.raster_plot.from_device(spkDetect['CMPf'],hist=False,title='CMPf'+displayStr)

    nest.raster_plot.show()

  return score, 6 if antagInjectionSite == 'none' else 1

# -----------------------------------------------------------------------------
# This function verify if their is pauses in the GPe and if the caracteristiques
# of theses pauses are relevant with the data of the elias paper 2007 
# It is run after CheckAVGFR because it uses the gdf files of the simulation.
# -----------------------------------------------------------------------------

#---------------------------- begining getSpikes ------------------------------
# return an ordered dictionnary of the spikes occurences by neuron and in the time
def getSpikes(Directory, Nuclei):
    spikesDict = {}
    gdfList = os.listdir(Directory + '/NoeArchGdf')
    
    for f in gdfList:
        if f.find(Nuclei) != -1 and f[-4:] == ".gdf" :
            spikeData = open(Directory +'/NoeArchGdf/' + f)
            for line in spikeData: # take the spike and put it in neuronRecording
                spk = line.split('\t')
                spk.pop()
                if spk[0] in spikesDict:
                    spikesDict[spk[0]].append(float(spk[1]))
                else:
                    spikesDict[spk[0]] = [float(spk[1])]
        
    for neuron in spikesDict:
        spikesDict[neuron] = sorted(spikesDict[neuron])
    
    return spikesDict
#---------------------------- end getSpikes -----------------------------------
    
#--------------------------- begining getISIs ---------------------------------
# return ISIs ordered by neuron in a dictionnary
def getISIs(spikesDict):
    ISIsDict = {}
    for neuron in spikesDict:
        ISIsDict[neuron] = []
        for i in range(len(spikesDict[neuron]) - 1):
            ISIsDict[neuron].append(round(spikesDict[neuron][i+1] - spikesDict[neuron][i], 1))
    ISIsList = []
    for neuron in ISIsDict:
        for isi in ISIsDict[neuron]:
            ISIsList.append(isi)       
    return ISIsDict, ISIsList
#----------------------------- end getISIs ------------------------------------ 
    
#--------------------------- begining rasterPlot ------------------------------
# plot rasters figures in the directory /raster
def rasterPlot(spikesDict, Nuclei, Directory):
    rasterList = []
    
    if not os.path.exists(Directory + '/rasterPlot'):
        os.makedirs(Directory + '/rasterPlot')

    for neuron in spikesDict:
        rasterList.append(spikesDict[neuron])  
    plt.figure(figsize=(20,8))
    plt.eventplot(rasterList, linelengths = 0.8, linewidths = 0.6)
    plt.title('Spike raster plot ' + Nuclei)
    plt.grid()
    plt.savefig(Directory + '/rasterPlot/' + 'RasterPlot_' + Nuclei + '.png')
#----------------------------- end rasterPlot ---------------------------------

#--------------------------- begining BarPlot ---------------------------------
# plot the nuclei histogram of ISIs
def HistPlot(ISIsList, Nuclei, Directory):
    plt.figure()
    plt.hist(ISIsList, bins=20, normed=0.5)
    plt.title('Histogram ' + Nuclei)
    plt.grid()
    plt.savefig(Directory + '/'+ 'HistPlot_' + Nuclei + '.png')
#----------------------------- end BarPlot ------------------------------------
    
#--------------------------- begining poisson ---------------------------------
# compute the poissonian probability that n or less spike occure during T ms
def poisson(n, r, T): # Tsum of 2 isi or 3 ? n = 2
    P = 0
    for i in range(n):
        P += math.pow(r*T, i)/ math.factorial(i)

    return P*math.exp(-r*T)
#----------------------------- end poisson ------------------------------------

#----------------------- begining Pause Analysis ------------------------------
def PauseAnalysis(ISIsDict,ISIsList): # Tsum of 2 isi or 3 ? n = 2
    simuSpecs = {'meanISI': np.mean(ISIsList),}
    
    r = 1/float(simuSpecs['meanISI'])
    pausesDict = {}
    pausesList = []
    coreIList = []
    
    isiThreshold = 0

    if max(ISIsList) >= 250:
        isiThreshold = 250
    elif max(ISIsList) >= 200:
        isiThreshold = 200
    elif max(ISIsList) >= 150:
        isiThreshold = 150
    elif max(ISIsList) >= 100:
        isiThreshold = 100
    elif max(ISIsList) >= 80:
        isiThreshold = 80
    elif max(ISIsList) >= 60:
        isiThreshold = 60
    elif max(ISIsList) >= 40:
        isiThreshold = 40
    else:
        isiThreshold = 20
          
    for neuron in ISIsDict:
        skip = False
        for i in range(1,len(ISIsDict[neuron])-1):
            if ISIsDict[neuron][i] >= isiThreshold and not skip :
                coreI = ISIsDict[neuron][i]
                pause = coreI
                s = -math.log10(poisson(1, r, coreI))
                s2 = -math.log10(poisson(2, r, coreI+ISIsDict[neuron][i-1]))
                s3 = -math.log10(poisson(2, r, coreI+ISIsDict[neuron][i+1]))
                if s2 > s and s2 >= s3:
                    s = s2
                    pause += ISIsDict[neuron][i-1]
                elif s3 > s:
                    s = s3
                    pause += ISIsDict[neuron][i+1]
                    skip = True
        
                if neuron in pausesDict:
                    pausesDict[neuron].append(pause)
                    pausesList.append(pause)
                    coreIList.append(coreI)
                else:
                    pausesDict[neuron] = [pause]
                    pausesList.append(pause)
                    coreIList.append(coreI)
            else:
                skip = False
    
    simuSpecs['isiThreshold'] = isiThreshold
    simuSpecs['percentagePausers'] = len(pausesDict)/float(len(ISIsDict))*100
    simuSpecs['nbPausersNeurons'] = len(pausesDict)
    simuSpecs['meanPausesDuration'] = round(np.mean(pausesList),2)
    simuSpecs['meanCoreI'] = round(np.mean(coreIList),2)
    simuSpecs['nbPausesPerMin'] = round(len(pausesList)/float(len(pausesDict)*params['tSimu'])*60000,2)
    simuSpecs['nbPauses'] = len(pausesList)
    simuSpecs['meanISI'] = round(np.mean(ISIsList),2)

    return simuSpecs
#-------------------------- end Pause Analysis --------------------------------
    
#------------------------- begining gdf exploitation --------------------------
# call the function and plot results
def gdfExploitation(Directory):
    for N in NUCLEI:
        spikesDict = getSpikes(Directory, N)
        HistPlot(getISIs(spikesDict)[1], N, Directory)
        rasterPlot(spikesDict, N, Directory)
        
        if N == 'GTA' or N == 'GTI':
            
            simuSpecs = PauseAnalysis(getISIs(spikesDict)[0], getISIs(spikesDict)[1])
            
            text = "################# Pause Results " + N + " #################"
            text += "\n ISI threshold       = " + str(simuSpecs['isiThreshold']) + " ms    | 250 ms"
            text += "\n Mean coreI duration = " + str(simuSpecs['meanCoreI']) + " ms | [200 - 600]"
            text += "\n Mean pause duration = " + str(simuSpecs['meanPausesDuration']) + " ms | 620 ms"
            text += "\n Mean ISI            = " + str(simuSpecs['meanISI']) + " ms | 15 ms\n"
            text += "\n total Pauses Nb     = " + str(simuSpecs['nbPauses']) 
            text += "\n pause/min/neuron    = " + str(simuSpecs['nbPausesPerMin']) + "      | [13 - 24] \n"
            text += "\n Pauser neurons Nb   = " + str(simuSpecs['nbPausersNeurons'])
            text += "\n % Pauser neurons    = " + str(simuSpecs['percentagePausers'])  + "     | [60 - 100]"
            text += "\n#####################################################"
            
            res = open(Directory+'/log/OutSummary.txt','a')
            res.writelines(text)
            res.close()
            print text

#---------------------------- end gdf exploitation ----------------------------

pausesDATA = {'percentagePausers':  [40. ,    100.,   75.],        # percentage of pauser neurons in GPe [low value, High value, perfect value]
              'shortPercentageISI': [0 ,     0.70,    0.2],         # percentage of Interspike intervals inferior to 2 ms
              'meanPausesDuration': [450. ,  730.,   620.],     # change to [0.45, 0.73, 0.62] are the  extreme recorded values if it is too selective
              'nbPausesPerMin':     [8. ,     23.,    13.],            # change to [8, 23, 13] are the  extreme recorded values if it is too selective
              'meanIPI':            [2.63 ,  8.74,   6.19],     # InterPauses Inteval | [2.63, 8.74, 6.19]are the  extreme recorded values if it is too selective
              'pausersFRR':         [37.48 , 71.25, 54.37],  # change to [21.47, 76.04, 54.13] which are the  extreme recorded values if it is too selective
              'correctedPausersFRR':[44.04 , 81.00, 62.52],  # change to [22.60, 86.63, 62.52] which are the  extreme recorded values if it is too selective
              'nonPausersFRR':      [37.10 , 75.75, 56.43],} # change to [31.37, 91.70, 56.43] which are the  extreme recorded values if it is too selective

#------------------------------------------------------------------------------
def main():
    
  Directory = os.getcwd()
  
  if len(sys.argv) >= 2:
    print "Command Line Parameters"
    paramKeys = ['LG14modelID',
                 'nbMSN',
                 'nbFSI',
                 'nbSTN',
                 'nbGTA',
                 'nbGTI',
                 'nbGPi',
                 'nbCSN',
                 'nbPTN',
                 'nbCMPf',
                 'GMSN',
                 'GFSI',
                 'GSTN',
                 'GGTA',
                 'GGTI',
                 'GGPi', 
                 'IeGTA',
                 'IeGTI',
                 'IeGPi',
                 'inDegCSNMSN',
                 'inDegPTNMSN',
                 'inDegCMPfMSN',
                 'inDegFSIMSN',
                 'inDegMSNMSN', 
                 'inDegCSNFSI',
                 'inDegPTNFSI',
                 'inDegSTNFSI',
                 'inDegGTAFSI',
		 'inDegGTAMSN',
                 'inDegCMPfFSI',
                 'inDegFSIFSI',
                 'inDegPTNSTN',
                 'inDegCMPfSTN',
                 'inDegGTISTN',
                 'inDegCMPfGTA',
                 'inDegCMPfGTI',
                 'inDegSTNGTA',
                 'inDegSTNGTI',
                 'inDegMSNGTA',
                 'inDegMSNGTI',
                 'inDegGTAGTA',
                 'inDegGTIGTA',
                 'inDegGTIGTI',
                 'inDegMSNGPi',
                 'inDegSTNGPi',
                 'inDegGTIGPi',
                 'inDegCMPfGPi',]
    
    if len(sys.argv) == len(paramKeys)+1:
      print "Using command line parameters"
      print sys.argv
      i = 0
      for k in paramKeys:
        i+=1
        params[k] = float(sys.argv[i])
    else :
      print "Incorrect number of parameters:",len(sys.argv),"-",len(paramKeys),"expected"

  nest.set_verbosity("M_WARNING")
  
  score = np.zeros((2)) 
  score += checkAvgFR(params=params,antagInjectionSite='none',antag='',showRasters=True)
  
  print "******************"
  print " Score FRR:",score[0],'/',score[1]
  print "******************"
  
  os.system('mkdir NoeArchGdf')  # save the .gdf files before antagonist desaster 
  os.system('cp log/MSN* log/STN* log/GTI* log/GTA* log/GPi* log/FSI* NoeArchGdf/ ')
  os.system('rm log/MSN* log/STN* log/GTI* log/GTA* log/FSI* log/CMPf*')
  gdfExploitation(Directory)
  
  if score[0] < score[1]:
    print("Activities at rest do not match: skipping deactivation tests")
#  else:
#      for a in ['AMPA','AMPA+GABAA','NMDA','GABAA']:
#        score += checkAvgFR(params=params,antagInjectionSite='GPe',antag=a)
#    
#      for a in ['AMPA+NMDA+GABAA','AMPA','NMDA+AMPA','NMDA','GABAA']:
#        score += checkAvgFR(params=params,antagInjectionSite='GPi',antag=a)
#  os.system('rm log/G*')

  #-------------------------
  print "******************"
  print "* Score:",score[0],'/',score[1]
  print "******************"

  #-------------------------
  # log the results in a file
  #-------------------------
  #res = open('OutSummary_'+timeStr+'.txt','a')
  res = open('OutSummary.txt','a')
  for k,v in params.iteritems():
    res.writelines(k+' , '+str(v)+'\n')
  res.writelines("Score: "+str(score[0])+' , '+str(score[1]))
  res.close()

  res = open('score.txt','w')
  res.writelines(str(score[0])+'\n')
  res.close()

#---------------------------
if __name__ == '__main__':
  main()
