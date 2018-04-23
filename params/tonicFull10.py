
params   =       {'LG14modelID':           10, # [0, 1, 2, 10, (12)] have ALPHA_GPe_MSN value != 0 
                  'tSimu':              5000.,
                  
                  'nbMSN':              2644.,
                  'nbFSI':                53.,
                  'nbSTN':                 8.,
                  'nbGTI':                20., # 84% GPe --> 80% GPe ( SATO 2000 )
                  'nbGTA':                 5., # 16% GPe --> 20% GPe ( SATO 2000 )
                  'nbGPi':                14.,
                  'nbCSN':              3000.,
                  'nbPTN':               100.,
                  'nbCMPf':             3000.,
                  
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




