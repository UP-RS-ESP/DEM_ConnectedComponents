from ConnectedComponents import *

#path = "~/DEM_ConnectedComponents/"
path = "./"

#load output from LSDTT
lsdttTable = mergeLSDTToutput(fname = "testClip", path = path, resolution=3, epsg = 32719)
pixThr = findPixelThreshold(lsdttTable, sampleStreams = 10 , mask = "testMask.shp", path = path)

debrisSamples = mergeLSDTToutput(fname = "debrisBasins_final", path = path, resolution=3, epsg = 32719)

dSlopeThr = findDSlopeThresholdDebrisSamples(debrisSamples, pixThr = 7, minCCLength = 10, bridge = 5, thresholdRange = np.arange(0.15,0.26,0.01))

runCCAnalysis(fname = "testClip", path = path, lsdttTable = lsdttTable, pixThr = 7, dSlopeThr = 0.23, minCCLength = 10, bridge = 5, mask = "testMask.shp")

getCCsInAOI(fname = "testClip", path=path, mask= "testMask.shp", epsg = 32719, pixThr= 7, dSlopeThr = 0.23, bridge = 5)

runCCAnalysis(fname = "debrisBasins_final", path = path, lsdttTable = debrisSamples, pixThr = 7, dSlopeThr = 0.23, minCCLength = 10, bridge = 5)

dfsiValues = assignDFSI( path = path, allCCName = "testClip", debrisName = "debrisBasins_final", pixThr = 7, dSlopeThr = 0.23, allExt = "_inAOI")

backsorting(fname = "testClip", path = path, dfsiValues = dfsiValues, pixThr = 7, dSlopeThr = 0.23, ext = "_inAOI")

plotBasin("testClip", path = path, pixThr = 7, dSlopeThr = 0.23, bridge = 5, basinIDs = [9,17], colorBy = "DFSI")

clusters = componentClustering(path = path, allCCName = "testClip", debrisName = "debrisBasins_final", pixThr = 7, dSlopeThr = 0.23,
    k = 3, clusterParameters = ["ccLength", "ccMeanSlope", "segmentLocation", "slopeChangeToPrevCC", "distDFSamples"])
