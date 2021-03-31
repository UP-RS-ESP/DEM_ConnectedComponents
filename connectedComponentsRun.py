#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 17:13:42 2021

@author: ariane
"""
from connectedComponentsFunctions import *

path = "/home/ariane/Documents/NWArgentina/Flowlines/FlowlinesGithub/"
lsdttPath =  "/home/ariane/LSDTopoTools/LSDTopoTools2/LSDTopoTools/LSDTopoTools2/bin/"

#get channel network of testclip. Here I have to set findCompleBasins to true because it is not a closed basin
getLSDTTFlowlines(fname = "testClip", path = path, lsdttPath = lsdttPath, m_over_n= 0.4, findCompleteBasins= "true", testBoundaries= "true")  
lsdttTableTestClip = mergeLSDTToutput(fname = "testClip", path = path, resolution=3, epsg = 32719)    
#find a good pixel threshold for calculating channel slope using 20 random channels within the study area
#taking only a hand-full of channels severely speeds up processing, yet I would reccoment to carefully select a single basin or use the entire study area for more accurate results
pixThr = findPixelThreshold(lsdttTableTestClip, sampleStreams = 20, thresholdRange = np.arange(1,16))

#get channel network of debris-flow sample regions 
getLSDTTFlowlines(fname = "debrisBasins", path = path, lsdttPath = lsdttPath, m_over_n= 0.4)  
lsdttTableDebrisSamples = mergeLSDTToutput(fname = "debrisBasins", path = path, resolution=3, epsg = 32719)   
#use debris-flow sample regions to get a good slope-change threshold to constrain CCs  
dSlopeThr = findDSlopeThreshold(lsdttTableDebrisSamples, pixThr = pixThr, thresholdRange = np.arange(0.15,0.26,0.01))

#run CC Analysis with optimal parameters for both channel networks
runCCAnalysis(fname = "testClip", path = path, lsdttTable = lsdttTableTestClip, pixThr = pixThr, dSlopeThr = dSlopeThr)
runCCAnalysis(fname = "debrisBasins", path = path, lsdttTable = lsdttTableDebrisSamples, pixThr = pixThr, dSlopeThr = dSlopeThr)

#assign debris-flow similarity
dfsiValues = assignDFSI( path = path, allCCName = "testClip", debrisName = "debrisBasins", pixThr = pixThr, dSlopeThr = dSlopeThr)
#backsorting DFSI to stream network
backsorting(fname = "testClip", path = path, dfsiValues = dfsiValues, pixThr = pixThr, dSlopeThr = dSlopeThr)
