
params   =       {'LG14modelID':           10, # [0, 1, 2, 10, (12)] have ALPHA_GPe_MSN value != 0 
                  'tSimu':              5000.,
                  
                  'nbMSN':            2644.*4,
                  'nbFSI':              53.*4,
                  'nbSTN':               8.*4,
                  'nbGTI':              20.*4,
                  'nbGTA':               5.*4, 
                  'nbGPi':              14.*4,
                  'nbCSN':            3000.*4,
                  'nbPTN':             100.*4,
                  'nbCMPf':           3000.*4,
                  
                  'IeMSN':               24.4, # tonic inputs
                  'IeFSI':                 7., #
                  'IeSTN':                 9., #
                  'IeGTA':               17.3, #
                  'IeGTI':               17.3, #
                  'IeGPi':               13.5, #
                  
                  'inDegCSNMSN':      0.33333, # Max = 1. for model 10
                  'inDegPTNMSN':      0.33333, # Max = 1. for model 10
                  'inDegCMPfMSN':     0.33333, # Max = 0.608 for model 10
                  'inDegFSIMSN':      0.33333, # /!\ Max = 0.0107417 for model 10
                  'inDegMSNMSN':      0.33333, # Max = 1. for model 10
                  'inDegSTNMSN':      0.33333, # Max = 1. for model 10 (0 anyway) (model 2: 0.43)
                  'inDegGTAMSN':      0.33333, # Max = 0.428 for model 10
                  'inDegCSNFSI':      0.33333, # Max = 1. for model 10
                  'inDegPTNFSI':      0.33333, # Max = 1.
                  'inDegSTNFSI':      0.33333, # Max = 0.517 for model 10
                  'inDegGTAFSI':      0.33333, # /!\ Max = 0.1756 for model 10
                  'inDegCMPfFSI':     0.33333, # Max = 1. for model 10 
                  'inDegFSIFSI':      0.33333, # Max = 1. for model 10 
                  'inDegPTNSTN':      0.33333, # Max = 0.386 for model 10 
                  'inDegCMPfSTN':     0.33333, # Max = 1. for model 10 
                  'inDegGTISTN':      0.33333, # Max = 1. for model 10 
                  'inDegCMPfGTI':     0.33333, # Max = 1. for model 10 
                  'inDegCMPfGTA':     0.33333, # Max = 1. for model 10 
                  'inDegSTNGTI':      0.33333, # /!\ Max = 0.02252 for model 10 
                  'inDegSTNGTA':      0.33333, # /!\ Max = 0.02252 for model 10 
                  'inDegMSNGTI':      0.33333, # Max = 1. for model 10 
                  'inDegMSNGTA':      0.33333, # Max = 1. for model 10 
                  'inDegGTIGTA':  (4/5.)*0.33333, # /!\ Max = 0.07936 for model 10 
                  'inDegGTIGTI':  (4/5.)*0.33333, # /!\ Max = 0.06349 for model 10 
                  'inDegGTAGTA':  (1/5.)*0.33333, # /!\ Max = 0.07936 for model 10 
                  'inDegGTAGTI':  (1/5.)*0.33333, # /!\ Max = 0.06349 for model 10 
                  'inDegMSNGPi':      0.33333, # Max = 1. for model 10 
                  'inDegSTNGPi':      0.33333, # /!\ Max = 0.047687 for model 10
                  'inDegGTIGPi':      0.33333, # Max = 1. for model 10 
                  'inDegCMPfGPi':     0.33333, # Max = 1. for model 10 

                  'parrotCMPf' :         True,}
#----- RESULTS -----
#* MSN - Rate: 0.339712556732 Hz -> OK (0.1 , 1)
#* FSI - Rate: 12.4198113208 Hz -> OK (5 , 20)
#* STN - Rate: 15.46875 Hz -> OK (15.2 , 22.8)
#* GTA - Rate: 57.81 Hz -> OK (37.48 , 71.26)
#* GTI - Rate: 62.285 Hz -> OK (37.1 , 75.76)
#* GPi - Rate: 75.4821428571 Hz -> OK (59.1 , 79.5)
#-------------------
#******************
# Score FRR: 6.0 / 6.0
#******************
#
# ------- PAUSES RESULTS --------------------
#>> /!\ NO !!! | No PAUSE DETECTED
#>> /!\ NO !!! | The longest ISI is:  99.4ms --> [250, ] Idealy 620ms

