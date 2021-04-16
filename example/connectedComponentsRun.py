#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 17:13:42 2021

@author: ariane
"""
from connectedComponentsFunctions import *
import numpy as np
import pandas as pd
path = "/home/ariane/Documents/NWArgentina/Flowlines/FlowlinesGithub/"

lsdttPath =  "/home/ariane/LSDTopoTools/LSDTopoTools2/LSDTopoTools/LSDTopoTools2/bin/"

lsdttTable = mergeLSDTToutput(fname = "testClip", path = path, resolution=3, epsg = 32719)    

#optional: get a suggenstion for the regression length
#pixThr = findPixelThreshold(lsdttTable,  sampleStreams = 10 , path = path, mask = "testMask.shp")

debrisSamples = mergeLSDTToutput(fname = "debrisBasins_final", path = path, resolution=3, epsg = 32719)  

#optional: get a suggestion for the slope-change threshold
#dSlopeThr1 = findDSlopeThresholdDebrisSamples(debrisSamples, pixThr = pixThr, minCCLength = 10, bridge = 5)
#dSlopeThr2 = findDSlopeThresholdAlternative(lsdttTable, dist = [10], pixThr = pixThr, sampleStreams = 10, mask = "testMask.shp", path = path)

runCCAnalysis(fname = "testClip", path = path, lsdttTable = lsdttTable, pixThr = 7, dSlopeThr = 0.23, minCCLength = 10, bridge = 5, mask = "testMask.shp")
getCCsInAOI(fname = "testClip", path=path, mask= "testMask.shp", epsg = 32719, pixThr= 7, dSlopeThr = 0.23, bridge = 5)

runCCAnalysis(fname = "debrisBasins_final", path = path, lsdttTable = debrisSamples, pixThr = 7, dSlopeThr = 0.23, minCCLength = 10, bridge = 5)

#assign debris-flow similarity
for thr in [0.1, 0.15, 0.2, 0.23]:
    dfsiValues = assignDFSI( path = path, allCCName = "TANDEM_SPOT-7_merged_withoutMainstream_noBoundaryCheck", debrisName = "debrisBasins_final", pixThr = 7, dSlopeThr = thr, allExt = "_inAOI")
                        #clusterParameters = ["ccLength", "ccMeanSlope", "segmentLocation", "slopeChangeToPrevCC", "distDFSamples"])
#backsorting DFSI to stream network
backsorting(fname = "testClip", path = path, dfsiValues = dfsiValues, pixThr = 7, dSlopeThr = 0.23, ext = "_inAOI")
