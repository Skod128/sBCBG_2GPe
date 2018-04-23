#!/apps/free/python/2.7.10/bin/python

import nstrand
import os
import numpy as np
from LGneurons import *
from modelParams import *
import nest.raster_plot
import nest.voltage_trace
import pylab as pl
import sys
import matplotlib.pyplot as plt
import math

#------------------------------------------
# Creates the populations of neurons necessary to simulate a BG circuit
#------------------------------------------
def createBG():
  print '\nCreating neurons\n================'
  for N in NUCLEI:
      nbSim[N] = params['nb'+N]
      create(N)
      nest.SetStatus(Pop[N],{"I_e":params['Ie'+N]})
  
  parrot = True # switch to False at your risks & perils...
  nbSim['CSN'] = params['nbCSN']
  create('CSN', fake=True, parrot=parrot)
  nbSim['PTN'] = params['nbPTN']
  create('PTN', fake=True, parrot=parrot)
  nbSim['CMPf'] = params['nbCMPf']
  create('CMPf', fake=True, parrot=params['parrotCMPf']) # was: False

  print "Number of simulated neurons:", nbSim

#------------------------------------------
# Connects the populations of a previously created multi-channel BG circuit
#------------------------------------------
def connectBG(antagInjectionSite,antag):
    
  G = {'MSN': params['GMSN'],
       'FSI': params['GFSI'],
       'STN': params['GSTN'],
       'GTA': params['GGTA'],
       'GTI': params['GGTI'],
       'GPi': params['GGPi'],}
  print "Gains on LG14 syn. strength:", G

  print '\nConnecting neurons\n================'
  print "**",antag,"antagonist injection in",antagInjectionSite,"**"
  
  print '* MSN Inputs'
  connect('ex','CSN','MSN', inDegree= params['inDegCSNMSN'], gain=G['MSN'])
  connect('ex','PTN','MSN', inDegree= params['inDegPTNMSN'], gain=G['MSN'])
  connect('ex','CMPf','MSN', inDegree=params['inDegCMPfMSN'],gain=G['MSN'])
  connect('in','MSN','MSN', inDegree= params['inDegMSNMSN'], gain=G['MSN'])
  connect('in','FSI','MSN', inDegree= params['inDegFSIMSN'], gain=G['MSN'])
  # some parameterizations from LG14 have no STN->MSN or GPe->MSN synaptic contacts
  if alpha['STN->MSN'] != 0:
    print "alpha['STN->MSN']",alpha['STN->MSN']
    connect('ex','STN','MSN', inDegree= params['inDegSTNMSN'],gain=G['MSN'])
  if alpha['GTA->MSN'] != 0:
    print "alpha['GTA->MSN']",alpha['GTA->MSN']
    connect('in','GTA','MSN', inDegree= params['inDegGTAMSN'],gain=G['MSN'])

  print '* FSI Inputs'
  connect('ex','CSN','FSI',  inDegree= params['inDegCSNFSI'], gain=G['FSI'])
  connect('ex','PTN','FSI',  inDegree= params['inDegPTNFSI'], gain=G['FSI'])
  if alpha['STN->FSI'] != 0:
    connect('ex','STN','FSI',inDegree= params['inDegSTNFSI'], gain=G['FSI'])
  connect('in','GTA','FSI',  inDegree= params['inDegGTAFSI'], gain=G['FSI'])
  connect('ex','CMPf','FSI', inDegree= params['inDegCMPfFSI'],gain=G['FSI'])
  connect('in','FSI','FSI',  inDegree= params['inDegFSIFSI'], gain=G['FSI'])

  print '* STN Inputs'
  connect('ex','PTN','STN', inDegree= params['inDegPTNSTN'], gain=G['STN'])
  connect('ex','CMPf','STN',inDegree= params['inDegCMPfSTN'],gain=G['STN'])
  connect('in','GTI','STN', inDegree= params['inDegGTISTN'], gain=G['STN'])

  print '* GTA Inputs'
  if antagInjectionSite == 'GPe':
    if   antag == 'AMPA':
      connect('NMDA','CMPf','GTA',inDegree=params['inDegCMPfGTA'],gain=G['GTA'])
      connect('NMDA','STN','GTA', inDegree=params['inDegSTNGTA'], gain=G['GTA'])
      connect('in','MSN','GTA', inDegree=  params['inDegMSNGTA'], gain=G['GTA'])
      connect('in','GTA','GTA', inDegree=  params['inDegGTAGTA'], gain=G['GTA']) 
      connect('in','GTI','GTA', inDegree=  params['inDegGTIGTA'], gain=G['GTA'])
    elif antag == 'NMDA':
      connect('AMPA','CMPf','GTA',inDegree= params['inDegCMPfGTA'],gain=G['GTA'])
      connect('AMPA','STN','GTA', inDegree= params['inDegSTNGTA'], gain=G['GTA'])
      connect('in','MSN','GTA', inDegree= params['inDegMSNGTA'],   gain=G['GTA'])
      connect('in','GTI','GTA', inDegree= params['inDegGTIGTA'],   gain=G['GTA'])
      connect('in','GTA','GTA', inDegree= params['inDegGTAGTA'],   gain=G['GTA'])
    elif antag == 'AMPA+GABAA':
      connect('NMDA','CMPf','GTA',inDegree= params['inDegCMPfGTA'],gain=G['GTA'])
      connect('NMDA','STN','GTA',inDegree= params['inDegSTNGTA'],  gain=G['GTA'])
    elif antag == 'GABAA':
      connect('ex','CMPf','GTA',inDegree= params['inDegCMPfGTA'], gain=G['GTA'])
      connect('ex','STN','GTA', inDegree= params['inDegSTNGTA'],  gain=G['GTA'])
    else:
      print antagInjectionSite,": unknown antagonist experiment:",antag
  else:
    connect('ex','CMPf','GTA',inDegree= params['inDegCMPfGTA'],gain=G['GTA'])
    connect('ex','STN','GTA', inDegree= params['inDegSTNGTA'], gain=G['GTA'])
    connect('in','MSN','GTA', inDegree= params['inDegMSNGTA'], gain=G['GTA'])
    connect('in','GTA','GTA', inDegree= params['inDegGTAGTA'], gain=G['GTA'])
    connect('in','GTI','GTA', inDegree= params['inDegGTIGTA'], gain=G['GTA'])

  print '* GTI Inputs'
  if antagInjectionSite == 'GPe':
    if   antag == 'AMPA':
      connect('NMDA','CMPf','GTI',inDegree=params['inDegCMPfGTI'],gain=G['GTI'])
      connect('NMDA','STN','GTI', inDegree=params['inDegSTNGTI'], gain=G['GTI'])
      connect('in','MSN','GTI', inDegree=  params['inDegMSNGTI'], gain=G['GTI'])
      connect('in','GTI','GTI', inDegree=  params['inDegGTIGTI'], gain=G['GTI'])
      connect('in','GTA','GTI', inDegree=  params['inDegGTAGTI'], gain=G['GTI'])
    elif antag == 'NMDA':
      connect('AMPA','CMPf','GTI',inDegree= params['inDegCMPfGTI'],gain=G['GTI'])
      connect('AMPA','STN','GTI', inDegree= params['inDegSTNGTI'], gain=G['GTI'])
      connect('in','MSN','GTI', inDegree= params['inDegMSNGTI'],   gain=G['GTI'])
      connect('in','GTI','GTI', inDegree= params['inDegGTIGTI'],   gain=G['GTI'])
      connect('in','GTA','GTI', inDegree= params['inDegGTAGTI'], gain=G['GTI'])
    elif antag == 'AMPA+GABAA':
      connect('NMDA','CMPf','GTI',inDegree= params['inDegCMPfGTI'],gain=G['GTI'])
      connect('NMDA','STN','GTI',inDegree= params['inDegSTNGTI'],  gain=G['GTI'])
    elif antag == 'GABAA':
      connect('ex','CMPf','GTI',inDegree= params['inDegCMPfGTI'], gain=G['GTI'])
      connect('ex','STN','GTI', inDegree= params['inDegSTNGTI'],  gain=G['GTI'])
    else:
      print antagInjectionSite,": unknown antagonist experiment:",antag
  else:
    connect('ex','CMPf','GTI',inDegree= params['inDegCMPfGTI'],gain=G['GTI'])
    connect('ex','STN','GTI', inDegree= params['inDegSTNGTI'], gain=G['GTI'])
    connect('in','MSN','GTI', inDegree= params['inDegMSNGTI'], gain=G['GTI'])
    connect('in','GTI','GTI', inDegree= params['inDegGTIGTI'], gain=G['GTI'])
    connect('in','GTA','GTI', inDegree= params['inDegGTAGTI'], gain=G['GTI'])

  print '* GPi Inputs'
  if antagInjectionSite =='GPi':
    if   antag == 'AMPA+NMDA+GABAA':
      pass
    elif antag == 'NMDA':
      connect('in','MSN','GPi',   inDegree= params['inDegMSNGPi'], gain=G['GPi'])
      connect('AMPA','STN','GPi', inDegree= params['inDegSTNGPi'], gain=G['GPi'])
      connect('in','GTI','GPi',   inDegree= params['inDegGTIGPi'], gain=G['GPi'])
      connect('AMPA','CMPf','GPi',inDegree= params['inDegCMPfGPi'],gain=G['GPi'])
    elif antag == 'NMDA+AMPA':
      connect('in','MSN','GPi', inDegree= params['inDegMSNGPi'], gain=G['GPi'])
      connect('in','GTI','GPi', inDegree= params['inDegGTIGPi'], gain=G['GPi'])
    elif antag == 'AMPA':
      connect('in','MSN','GPi',   inDegree= params['inDegMSNGPi'], gain=G['GPi'])
      connect('NMDA','STN','GPi', inDegree= params['inDegSTNGPi'], gain=G['GPi'])
      connect('NMDA','CMPf','GPi',inDegree= params['inDegCMPfGPi'],gain=G['GPi'])
    elif antag == 'GABAA':
      connect('ex','STN','GPi', inDegree= params['inDegSTNGPi'], gain=G['GPi'])
      connect('ex','CMPf','GPi',inDegree= params['inDegCMPfGPi'],gain=G['GPi'])
    else:
      print antagInjectionSite,": unknown antagonist experiment:",antag
  else:
    connect('in','MSN','GPi', inDegree= params['inDegMSNGPi'], gain=G['GPi'])
    connect('ex','STN','GPi', inDegree= params['inDegSTNGPi'], gain=G['GPi'])
    connect('ex','CMPf','GPi',inDegree= params['inDegCMPfGPi'],gain=G['GPi'])
    connect('in','GTI','GPi',inDegree= params['inDegGTIGPi'], gain=G['GPi'])

#------------------------------------------
# Re-weight a specific connection, characterized by a source, a target, and a receptor
# Returns the previous value of that connection (useful for 'reactivating' after a deactivation experiment)
#------------------------------------------
def alter_connection(src, tgt, tgt_receptor, altered_weight):
  recTypeEquiv = {'AMPA':1,'NMDA':2,'GABA':3, 'GABAA':3} # adds 'GABAA'
  # check that we have this connection in the current network
  conns_in = nest.GetConnections(source=Pop[src], target=Pop[tgt])
  if len(conns_in):
    receptors = nest.GetStatus(conns_in, keys='receptor')
    previous_weights = nest.GetStatus(conns_in, keys='weight')
    rec_nb = recTypeEquiv[tgt_receptor]
    if isinstance(altered_weight, int):
      altered_weights = [altered_weight] * len(receptors)
    elif len(altered_weight) == len(receptors):
      altered_weights = altered_weight # already an array
    else:
      raise LookupError('Wrong size for the `altered_weights` variable (should be scalar or a list with as many items as there are synapses in that connection - including non-targeted receptors)')
    new_weights = [{'weight': float(previous_weights[i])} if receptors[i] != rec_nb else {'weight': float(altered_weights[i])} for i in range(len(receptors))] # replace the weights for the targeted receptor
    nest.SetStatus(conns_in, new_weights)
    return previous_weights
  return None

#------------------------------------------
# gets the nuclei involved in deactivation experiments in GPe/GPi
#------------------------------------------
def get_afferents(a):
  GABA_afferents = ['MSN', 'GTA','GTI'] # afferents with gabaergic connections
  GLUT_afferents = ['STN', 'CMPf'] # afferents with glutamatergic connections
  if a == 'GABAA':
    afferents = GABA_afferents
  elif a == 'AMPA+GABAA':
    afferents = GABA_afferents + GLUT_afferents
  elif a == 'AMPA+NMDA+GABAA':
    afferents = GABA_afferents + GLUT_afferents
  else:
    afferents = GLUT_afferents
  return afferents

#------------------------------------------
# deactivate connections based on antagonist experiment
#------------------------------------------
def deactivate(site, a):
  ww = {}
  for src in get_afferents(a):
    ww[src] = None
    for rec in a.split('+'):
      w = alter_connection(src, site, rec, 0)
      if ww[src] == None:
        ww[src] = w # keep the original weights only once
  return ww

#------------------------------------------
# reactivate connections based on antagonist experiment
#------------------------------------------
def reactivate(site, a, ww):
  for src in get_afferents(a):
    for rec in a.split('+'):
      alter_connection(src, site, rec, ww[src])

#------------------------------------------
# Instantiate the BG network according to the `params` dictionnary
# For now, this instantiation respects the hardcoded antagonist injection sites
# In the future, these will be handled by changing the network weights
#------------------------------------------
def instantiate_BG(params={}, antagInjectionSite='none', antag=''):
  nest.ResetKernel()
  dataPath='log/'
  if 'nbcpu' in params:
    nest.SetKernelStatus({'local_num_threads': params['nbcpu']})

  nstrand.set_seed(params['nestSeed'], params['pythonSeed']) # sets the seed for the BG construction

  nest.SetKernelStatus({"data_path": dataPath})
  initNeurons()

  print '/!\ Using the following LG14 parameterization',params['LG14modelID']
  loadDictParams(params['LG14modelID'])
  loadThetaFromCustomparams(params)

  # We check that all the necessary parameters have been defined. They should be in the modelParams.py file.
  # If one of them misses, we exit the program.
  necessaryParams=['nbCh','nbMSN','nbFSI','nbSTN','nbGTI','nbGTA','nbGPi','nbCSN','nbPTN','nbCMPf',
                   'IeMSN','IeFSI','IeSTN','IeGTI','IeGTA','IeGPi',
                   'GMSN','GFSI','GSTN','GGTI','GGTA','GGPi',
                   'inDegCSNMSN','inDegPTNMSN','inDegCMPfMSN','inDegMSNMSN','inDegFSIMSN','inDegSTNMSN','inDegGTAMSN',
                   'inDegCSNFSI','inDegPTNFSI','inDegSTNFSI','inDegGTAFSI','inDegCMPfFSI','inDegFSIFSI',
                   'inDegPTNSTN','inDegCMPfSTN','inDegGTISTN',
                   'inDegCMPfGTA','inDegSTNGTA','inDegMSNGTA','inDegGTAGTA','inDegGTIGTA',
                   'inDegCMPfGTI','inDegSTNGTI','inDegMSNGTI','inDegGTIGTI','inDegGTAGTI',
                   'inDegMSNGPi','inDegSTNGPi','inDegGTIGPi','inDegCMPfGPi',]
  
  for np in necessaryParams:
    if np not in params:
      raise KeyError('Missing parameter: '+np)

  #------------------------
  # creation and connection of the neural populations
  #------------------------
  createBG()
  connectBG(antagInjectionSite,antag)

#------------------------------------------
# Checks whether the BG model respects the electrophysiological constaints (firing rate at rest).
# If testing for a given antagonist injection experiment, specifiy the injection site in antagInjectionSite, and the type of antagonists used in antag.
# Returns [score obtained, maximal score]
# params possible keys:
# - nb{MSN,FSI,STN,GPi,GPe,CSN,PTN,CMPf} : number of simulated neurons for each population
# - Ie{GPe,GPi} : constant input current to GPe and GPi
# - G{MSN,FSI,STN,GPi,GPe} : gain to be applied on LG14 input synaptic weights for each population
#------------------------------------------
def checkAvgFR(showRasters=False,params={},antagInjectionSite='none',antag='',logFileName=''):
  nest.ResetNetwork()
  initNeurons()  # sets the default params of iaf_alpha_psc_mutisynapse neurons to CommonParams

  showPotential = False # Switch to True to graph neurons' membrane potentials - does not handle well restarted simulations

  dataPath='log/'
  nest.SetKernelStatus({"overwrite_files":True}) # when we redo the simulation, we erase the previous traces

  nstrand.set_seed(params['nestSeed'], params['pythonSeed']) # sets the seed for the simulation

  simulationOffset = nest.GetKernelStatus('time')
  print('Simulation Offset: '+str(simulationOffset))
  offsetDuration = 1000.

  #-------------------------
  # measures
  #-------------------------
  spkDetect={} # spike detectors used to record the experiment
  multimeters={} # multimeters used to record one neuron in each population
  expeRate={}

  antagStr = ''
  if antagInjectionSite != 'none':
    antagStr = antagInjectionSite+'_'+antag+'_'
    
#  storeGDFdissociated = storeGDF  # avoid an error
#  if antagStr != '':
#      storeGDFdissociated = False # gdf files are not required for antagonist simulation
      
  for N in NUCLEI:   
    # 1000ms offset period for network stabilization
    spkDetect[N] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": antagStr+N,"to_memory": False, "to_file": storeGDF, 'start':offsetDuration+simulationOffset,'stop':offsetDuration+params['tSimu']+simulationOffset})
    nest.Connect(Pop[N], spkDetect[N])
    if showPotential:
      # multimeter records only the last 200ms in one neuron in each population
      multimeters[N] = nest.Create('multimeter', params = {"withgid": True, 'withtime': True, 'interval': 0.1, 'record_from': ['V_m'], "label": antagStr+N, "to_file": False, 'start':offsetDuration+simulationOffset+params['tSimu']-200.,'stop':offsetDuration+params['tSimu']+simulationOffset})
      nest.Connect(multimeters[N], [Pop[N][0]])

  #-------------------------
  # Simulation
  #-------------------------
  nest.Simulate(params['tSimu']+offsetDuration)

  score = 0

  text=[]
  frstr = "#" + str(params['LG14modelID'])+ " , " + antagInjectionSite + ', '
  s = '----- RESULTS -----'
  print s
  text.append(s+'\n')
  if antagInjectionSite == 'none':
    validationStr = "\n#" + str(params['LG14modelID']) + " , "
    frstr += "none , "
    for N in NUCLEI:
      strTestPassed = 'NO!'
      expeRate[N] = nest.GetStatus(spkDetect[N], 'n_events')[0] / float(nbSim[N]*params['tSimu']) * 1000
      if expeRate[N] <= FRRNormal[N][1] and expeRate[N] >= FRRNormal[N][0]:
        # if the measured rate is within acceptable values
        strTestPassed = 'OK'
        score += 1
        validationStr += N + "=OK , "
      else:
      # out of the ranges
        if expeRate[N] > FRRNormal[N][1] :
          difference = expeRate[N] - FRRNormal[N][1]
          validationStr += N + "=+%.2f , " % difference
        else:
          difference = expeRate[N] - FRRNormal[N][0]
          validationStr += N + "=%.2f , " % difference

      frstr += '%f , ' %(expeRate[N])
      s = '* '+N+' - Rate: '+str(expeRate[N])+' Hz -> '+strTestPassed+' ('+str(FRRNormal[N][0])+' , '+str(FRRNormal[N][1])+')'
      print s
      text.append(s+'\n')
  else:
    validationStr = ""
    frstr += str(antag) + " , "
    for N in NUCLEI:
      expeRate[N] = nest.GetStatus(spkDetect[N], 'n_events')[0] / float(nbSim[N]*params['tSimu']) * 1000
      if N == antagInjectionSite:
        strTestPassed = 'NO!'
        if expeRate[N] <= FRRAnt[N][antag][1] and expeRate[N] >= FRRAnt[N][antag][0]:
          # if the measured rate is within acceptable values
          strTestPassed = 'OK'
          score += 1
          validationStr += N + "_" + antag + "=OK , "
        else:
        # out of the ranges
          if expeRate[N] > FRRNormal[N][1] :
            difference = expeRate[N] - FRRNormal[N][1]
            validationStr += N + "_" + antag + "=+%.2f , " % difference
          else:
            difference = expeRate[N] - FRRNormal[N][0]
            validationStr += N + "_" + antag + "=%.2f , " % difference

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

  validationFile = open("validationArray.csv",'a')
  validationFile.write(validationStr)
  validationFile.close()
  #-------------------------
  # Displays
  #-------------------------
  if showRasters and interactive:
    displayStr = ' ('+antagStr[:-1]+')' if (antagInjectionSite != 'none') else ''
    for N in NUCLEI:
      nest.raster_plot.from_device(spkDetect[N],hist=True,title=N+displayStr)

    if showPotential:
      pl.figure()
      nsub = 231
      for N in NUCLEI:
        pl.subplot(nsub)
        nest.voltage_trace.from_device(multimeters[N],title=N+displayStr+' #0')
        nest.Disconnect(Pop[N], multimeters[N])
        pl.axhline(y=BGparams[N]['V_th'], color='r', linestyle='-')
        nsub += 1
    pl.show()

  return score, 6 if antagInjectionSite == 'none' else 1

# -----------------------------------------------------------------------------
# This function recuperates the spikes occured in each neuron of each population
# and order them
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
    
    if not os.path.exists(Directory + '/histPlot'):
        os.makedirs(Directory + '/histPlot')
        
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
                 'nbMSN','nbFSI','nbSTN','nbGTA','nbGTI','nbGPi','nbCSN','nbPTN','nbCMPf',
                 'GMSN','GFSI','GSTN','GGTA','GGTI','GGPi', 
                 'IeGTA','IeGTI','IeGPi',
                 'inDegCSNMSN','inDegPTNMSN','inDegCMPfMSN','inDegFSIMSN','inDegMSNMSN','inDegGTAMSN',
                 'inDegCSNFSI','inDegPTNFSI','inDegSTNFSI','inDegGTAFSI','inDegCMPfFSI','inDegFSIFSI',
                 'inDegPTNSTN','inDegCMPfSTN','inDegGTISTN',
                 'inDegCMPfGTA','inDegCMPfGTI','inDegSTNGTA','inDegSTNGTI','inDegMSNGTA','inDegMSNGTI','inDegGTAGTA','inDegGTIGTA','inDegGTIGTI',
                 'inDegMSNGPi','inDegSTNGPi','inDegGTIGPi','inDegCMPfGPi',]
    
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
  
  instantiate_BG(params, antagInjectionSite='none', antag='')
  score = np.zeros((2))
  score += checkAvgFR(params=params,antagInjectionSite='none',antag='',showRasters=True)  
  os.system('mkdir NoeArchGdf')  # save the .gdf files before antagonist desaster 
  os.system('cp log/MSN* log/STN* log/GTI* log/GTA* log/GPi* log/FSI* NoeArchGdf/ ')
  os.system('rm log/MSN* log/STN* log/GTI* log/GTA* log/GPi* log/FSI* ')
  
  gdfExploitation(Directory)
  
  print "******************"
  print " Score FRR:",score[0],'/',score[1]
  print "******************"
 
  # don't bother with deactivation or the pauses tests if activities at rest are not within plausible bounds
  if score[0] < score[1]:
    print("Activities at rest do not match: skipping deactivation tests")
#  else:
#    
      
#     The following implements the deactivation tests without re-wiring the BG (faster)
      
#    for a in ['AMPA','AMPA+GABAA','NMDA','GABAA']:
#      ww = deactivate('GTA', a)
#      ww = deactivate('GTI', a)
#      score += checkAvgFR(params=params,antagInjectionSite='GPe',antag=a)
#      reactivate('GTA', a, ww)
#      reactivate('GTI', a, ww)
#
#    for a in ['AMPA+NMDA+GABAA','AMPA','NMDA+AMPA','NMDA','GABAA']:
#      ww = deactivate('GPi', a)
#      score += checkAvgFR(params=params,antagInjectionSite='GPi',antag=a)
#      reactivate('GPi', a, ww)
  
#   The following implements the deactivation tests with re-creation of the entire BG every time (slower)
      
#    for a in ['AMPA','AMPA+GABAA','NMDA','GABAA']:
#      instantiate_BG(params, antagInjectionSite='GPe', antag=a)
#      score += checkAvgFR(params=params,antagInjectionSite='GPe',antag=a)
#
#    for a in ['AMPA+NMDA+GABAA','AMPA','NMDA+AMPA','NMDA','GABAA']:    
#      instantiate_BG(params, antagInjectionSite='GPi', antag=a)
#      score += checkAvgFR(params=params,antagInjectionSite='GPi',antag=a)


 
  #-------------------------
  print "******************"
  print " Total Score :",score[0],'/',score[1]
  print "******************"

  #-------------------------
  # log the results in a file
  #-------------------------
  res = open('log/OutSummary.txt','a')
  for k,v in params.iteritems():
    res.writelines(k+' , '+str(v)+'\n')
  res.writelines("Score: "+str(score[0])+' , '+str(score[1]))
  res.close()

  res = open('score.txt','w')
  res.writelines(str(score[0])+'\n')
  res.close()

  # combined params+score output, makes it quicker to read the outcome of many experiments
  params['sim_score'] = score[0]
  params['max_score'] = score[1]
  with open('params_score.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in params.items():
       writer.writerow([key, value])

#---------------------------
if __name__ == '__main__':
  main()


