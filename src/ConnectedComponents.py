#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:59:37 2021
V 0.1

@author: Ariane Mueting
"""


import pandas as pd
from tqdm import tqdm
import numpy as np
import geopandas as gpd
import os, sys
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.stats import linregress, gaussian_kde
from kneed import KneeLocator
import concurrent.futures
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

pd.set_option('mode.chained_assignment', None)

#####################################################################################################################

def printLSDTTDriver(fname, path, minContributingPixels = 1000, maxBasinSize = int(1e07),findCompleteBasins = "true", testBoundaries = "true", m_over_n = 0.45):
    #this function converts the input geotiff to ENVI format, creates a driver file and runs lsdtt-chi mapping to extract channels

    #create driver file for lsdtt
    try:
        os.remove(path+fname+".driver")
    except OSError:
        pass
    stdoutOrigin=sys.stdout
    sys.stdout = open (path+fname+".driver", "a")

    print("""
read path: %s
read fname: %s

write path: %s
write fname: %s

# Parameter for filling the DEM
min_slope_for_fill: 0.0001

# print statements
print_channels_to_csv: true
print_segmented_M_chi_map_to_csv: true
#print_junctions_to_csv: true

# method to extract channels
print_area_threshold_channels: true

# Parameters for selecting channels and basins
threshold_contributing_pixels: %i
maximum_basin_size_pixels: %i
find_complete_basins_in_window: %s
test_drainage_boundaries: %s

# Parameters for chi analysis
A_0: 1
m_over_n: %f
n_iterations: 20
target_nodes: 80
minimum_segment_length: 10
sigma: 10.0
skip: 2
    """ % (path,fname,path,fname,minContributingPixels,maxBasinSize, findCompleteBasins, testBoundaries, m_over_n))

    sys.stdout.close()
    sys.stdout = stdoutOrigin


#####################################################################################################################

def normalize(x):
    #just a min max scaler - always useful
    return(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))


#####################################################################################################################

def normalizeCustomValues(x, minVal, maxVal):
    #a min max scaler using percentiles instead of min and max values. Values above 1 will be assigned one and
    #values below 0 will be set to 0
    scaled = (x-minVal)/(maxVal-minVal)
    scaled.loc[scaled>1]=1
    scaled.loc[scaled<0]=0
    return(scaled)
#####################################################################################################################

def toUTM(df, epsg):
    #turn lat and lon into UTM coordinates because LSD only provides geographic CRS
    gdf = df.set_geometry(gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    projected_df = gdf.to_crs("EPSG:"+str(epsg))
    df["x"] = projected_df.geometry.x
    df["y"] = projected_df.geometry.y
    return df

#####################################################################################################################

def mergeLSDTToutput(fname, path, resolution, epsg, ext = "", writeCSV = False):
    #function combines _MChiSegmented file with the reciever nodes from the _CN file to know which stream pixel comes next
    print("Merging CSV files from LSDTopoTools...")
    #load chi segmented channels from LSDTT
    lsdttTable = pd.read_csv(path+fname+"_MChiSegmented"+ext+".csv")
    #Convert long lat to UTM
    lsdttTable = toUTM(lsdttTable, epsg)
    # get coorndinates
    coords = [(x,y) for x, y in zip(lsdttTable.x, lsdttTable.y)]

    #the information from the _MChiSegmented.csv file with the _CN.csv file needs to be combined because the reciever nodes (stored in the _CN file)
    #need to be retrieved in order to properly walk downstream.
    #this is done by converting the channel data to a shapefile and rasterize it so that the reciever nodes can be extracted
    #spatial join does not work, because points are slightly offset

    #CSV to shapefile because gdal_rasterize didnt like the csv format
    if not os.path.isfile(path+fname+"_RNI.tif"):
        if not os.path.isfile(path+fname+"_CN.shp"):
            ogr2ogr = "ogr2ogr -s_srs EPSG:4326 -t_srs EPSG:32719 -oo X_POSSIBLE_nameS=lon* -oo Y_POSSIBLE_nameS=lat*  -f \"ESRI Shapefile\" "+path+fname+"_CN.shp "+path+fname+"_CN.csv"
            os.system(ogr2ogr)
        #now rasterize the information about the reciever nodes

        rasterize = "gdal_rasterize -a \"receiver_N\" -tr "+str(resolution)+" "+ str(resolution)+" -a_nodata 0 -co COMPRESS=DEFLATE -co ZLEVEL=9 "+path+fname+"_CN.shp "+path+fname+"_RNI.tif"
        os.system(rasterize)

    #open raster and extract reciever node information to points
    rni = rasterio.open(path+fname+"_RNI.tif") #load reciever node raster
    lsdttTable['RNI'] = [x[0] for x in rni.sample(coords)] #sample reciever nodes at points

    if writeCSV:
        lsdttTable.to_csv(path+fname+"_LSDTToutput_merged.csv", index = False)
        print("The file "+fname+"_LSDTToutput_merged.csv has been written.")
    return lsdttTable


#####################################################################################################################

def processBasin(basin, lsdttTable, heads, pixThr, dSlopeThr, bridge, minCCLength = 2):
    #function calculates channel slope and connected components for individual catchments

    #empty dataframe for storing output
    df = pd.DataFrame(columns = ["X","Y","StreamID","BasinID", "Elevation", "DrainageArea", "XYDistanceToNextPixel", "DownstreamDistance", "3DDistanceToNextPixel",  "Slope", "R2", "ksn", "FlowDistance_Catchment","ccID", "segmentLocation", "dSlopeToPrevSegment"])

    #convert catchment-wide flow-distance so that the maximum flow distance is the basin outlet
    #lsdttTable.flow_distance = (lsdttTable.flow_distance - max(lsdttTable.flow_distance))*-1

    #add dummy last node to the input lsdttTable for the current basin so that all channel pixels will be processed (else the last one is left out)
    dummy = pd.DataFrame(np.nan, index=[len(lsdttTable)], columns=lsdttTable.columns)
    dummy.node = -9999
    dummy.RNI = -9999
    #in lsdttTable, replace RNI of actual last pixel where nextNode == currentNode with -9999
    lsdttTable.loc[lsdttTable.node == lsdttTable.RNI, 'RNI'] = -9999
    #merge lsdttTable and dummy
    lsdttTable = pd.concat([lsdttTable,dummy])

    #find terminal node of processed basin to correctly assign the segment location
    finalNode = -9999 #lsdttTable.loc[lsdttTable.flow_distance == max(lsdttTable.flow_distance)].node.iloc[0]

    #print("Processing basin "+str(basin+1)+"/"+str(heads["basin_key"].nunique()))
    #subset channel heads by catchment
    bheads = heads.loc[(heads.basin_key == basin)].reset_index(drop = True)
    for ii, currentNode in enumerate(bheads.node):

        #print("Processing channelhead "+str(ii+1)+"/"+str(len(bheads)))
        nextNode = bheads.RNI[ii]
        #set parameters for CC extraction
        dupCounter = 0 #indicator whether or not a pixel has already been processed (after intersection)
        segmentLocationcount = 1
        mList = [] #list to store Slope values and compute the mean
        previousSlps =[] #get a list of previous Slopes to compare tha Slope change
        ccID = 0
        mean = -9999
        #generate empty elevation and dsDist array to fit regressions and calculate channel slope
        elev = np.zeros(shape=(2*pixThr+1))
        elev[:] = np.nan
        dsDist = np.zeros(shape=(2*pixThr+1))
        dsDist[:] = np.nan

        #make sure to process streams one more time after reaching the condition nextNode == currentNode
        # oncemore = iter([True, False])
        # process = True

        #empty array for storing data
        data = np.zeros([60000,13]) #if there are really really long channels (> 60000 pixels), this array needs to be expanded
        data[:] = np.nan
        j=0

        #start going down a channel
        #do so while the current node is not the next node (happens at channel outlet) and there is still space in the array
        while(currentNode != nextNode and j < len(data)):

            #identify data at current and next node
            crrnt = lsdttTable.loc[np.where(lsdttTable.node == currentNode)].reset_index(drop = True)
            nxt = lsdttTable.loc[np.where(lsdttTable.node == nextNode)].reset_index(drop = True)

            #break the loop if a next Node is not available
            #this might happen, if the lsdttTable is subsetted
            if nxt.empty:
                currentNode = nextNode
            else:
                #compute distance between two points
                #distance in XY direction only
                distXY = np.sqrt((crrnt.x.iloc[0]-nxt.x.iloc[0])**2+(crrnt.y.iloc[0]-nxt.y.iloc[0])**2)
                #3D distance
                dist3D = np.sqrt((crrnt.x.iloc[0]-nxt.x.iloc[0])**2+(crrnt.y.iloc[0]-nxt.y.iloc[0])**2+(crrnt.elevation.iloc[0]-nxt.elevation.iloc[0]))


                #dynamic elevation and dsDist array to compute regression from
                #start filling in at the beginning and remove values at the end
                elev = np.insert(elev, 0, crrnt.elevation.iloc[0]) #add current elevation
                elev = np.delete(elev, 2*pixThr+1) #remove last

                #same for downstream distance
                dsDist = np.insert(dsDist, 0, distXY)
                dsDist = np.delete(dsDist, 2*pixThr+1)

                #now store some data in the empty array
                data[j,0]= crrnt.x.iloc[0] #x coordinate
                data[j,1]= crrnt.y.iloc[0] #y coordinate
                data[j,2] = crrnt.elevation.iloc[0] #elevation
                data[j,3] = distXY #dsDist XY
                data[j,4] = crrnt.drainage_area.iloc[0] #drainage area
                data[j,5] = dist3D #dsDistXYZ
                data[j,6] = crrnt.m_chi.iloc[0] #ksn
                data[j,12] = crrnt.flow_distance.iloc[0]
                #find next node
                currentNode = nextNode
                nextNode = lsdttTable.loc[np.where(lsdttTable.node == currentNode)].RNI.iloc[0]

                #if the current combination of X and Y coordinates is already in the output dataframe, increase the duplication counter
                #because slope and cc calculation are lagging, slope calculation needs to continue at least j+pixThr+bridge rounds until it can terminate
                if((df[['X','Y']].values == [data[j,0], data[j,1]]).all(axis=1).any()):
                    dupCounter +=1

                ######################################
                #calculate slope running stream
                if(j>=pixThr): #when a significant amount of pixels (=pixelThreshold) is reached, start computing channel slope
                    #compute regression
                    #print("Calculating slope for point "+str(j-pixThr))
                    #remove nan if still present
                    mask = ~np.isnan(dsDist) & ~np.isnan(elev)
                    reg = linregress(np.cumsum(dsDist[mask]),elev[mask])
                    data[j-pixThr,7] = abs(reg[0]) #slope
                    data[j-pixThr,8] = reg[2] #R2

                    # plt.figure(figsize=(11.69,8.27))
                    # plt.scatter(np.cumsum(dsDist[mask]),elev[mask], color = "gray")
                    # plt.hlines(data[j-pixThr,2], 0 ,40)
                    # plt.show()

                ######################################
                #define connected components running stream
                if(j>=pixThr+bridge): #for connected components, the connectivity
                 #of two pixels needs to be investigated even later to make sure the channel slope is already known
                 #by the desired amount of channel pixels that are allowed to be bridged in advance
                    slopeInvestigated = data[j-pixThr-bridge,7]
                    #print("Assess connectivity of point "+str(j-pixThr-bridge))
                    if(j == pixThr+bridge): #the first pixel for which slope is calculated cannot be compared to any previous ones
                        #add slope to mean cc slope df
                        mList.append(slopeInvestigated) #append current slope
                        mean = np.mean(mList) #calculate mean
                        #add current ccID
                        data[j-pixThr-bridge, 9] = ccID
                        data[j-pixThr-bridge, 10]=segmentLocationcount


                    else:
                        if(abs(mean-slopeInvestigated)<=dSlopeThr or len(mList) < minCCLength): #if Slope of current pixel is smaller than the mean of the current connected segment
                        #and if we have already processed enough points to pass the minimum CC length

                            data[j-pixThr-bridge, 9]= ccID #assign the same ccID
                            data[j-pixThr-bridge, 10]=segmentLocationcount #assign segment location

                            mList.append(slopeInvestigated) #add Slope to Slope list and compute mean
                            mean = np.mean(mList)

                        #bridge outliers, these wont be added to the mean list
                        elif(np.any(abs(mean-data[j-pixThr-bridge+1:j-pixThr+1,7])<=dSlopeThr)): #if Slope of current pixel is smaller than the mean of the current connected segment, but up to 5 pixels downstream the slope is still within threshold bounds

                            data[j-pixThr-bridge, 9] = ccID
                            data[j-pixThr-bridge, 10]=segmentLocationcount


                        else: #terminate segment

                            #calculate slope change with regard to previous CC
                            try:
                                data[np.where(data[:,9]==ccID),11]= mean-previousSlps[-1]
                            except IndexError: #if list previousSlps is empty, it will raise an index error
                                pass

                            #if the algorithm is past a junction, the calculation is terminated to not get any duplicated CCs
                            if(dupCounter>=pixThr+bridge):
                                break

                            ccID+=1 #increase CCID
                            data[j-pixThr-bridge, 9] = ccID #assign new ccID

                            mList = [] #clear list
                            mList.append(slopeInvestigated)
                            previousSlps.append(mean) #append previous Slope values to list to be able to compare them later on
                            mean = np.mean(mList)
                            if(segmentLocationcount == 1): #increase segmentLocationcount to two if there already is a first segment
                                segmentLocationcount = 2
                            data[j-pixThr-bridge, 10]=segmentLocationcount

            ##########################################
            #if catchments are cutoff by previously masking the lsdtt flowlines, small cutoff streams will cause errors
            if(currentNode == nextNode and dupCounter == 0 and j<minCCLength):
                print("Small stream shorter than minimum required regression length removed.")
            else:
                #calculate slope at end of stream
                if(currentNode == nextNode): # currentNode = nextNode (end of stream)
                    #if we have a really short stream (<pixThr), just fit a single regression to all pixels
                    if j<pixThr:
                        #print("Stream is shorter than regression length. Let's fit a single regression for the entire channel.")
                        mask = ~np.isnan(dsDist) & ~np.isnan(elev)
                        reg = linregress(np.cumsum(dsDist[mask]),elev[mask])
                        data[0:j,7] = abs(reg[0]) #slope
                        data[0:j,8] = reg[2] #R2
                    #compute the remaining slope values (slope comutation is lagging by number of pixels chosen as pixel Threshold)
                    else:
                        for k in range(j-pixThr, j+1, 1):
                            #print("Calculating slope for point "+str(k))
                            elev = np.insert(elev, 0, np.nan) #fill with nan from start
                            elev = np.delete(elev, 2*pixThr+1) #remove last
                            dsDist = np.insert(dsDist, 0, np.nan)
                            dsDist = np.delete(dsDist, 2*pixThr+1)
                            mask = ~np.isnan(dsDist) & ~np.isnan(elev)
                            reg = linregress(np.cumsum(dsDist[mask]),elev[mask])
                            data[k,7] = abs(reg[0]) #slope
                            data[k,8] = reg[2] #R2


                        # plt.figure(figsize=(11.69,8.27))
                        # plt.scatter(np.cumsum(dsDist[mask]),elev[mask], color = "gray")
                        # plt.hlines(data[k,2], 0 ,40)
                        # plt.show()

                    #########################################
                    #finish calculating CCs
                    #if the stream is shorter than the minimal required CC length, everything is just assigned to a single component
                    #and the segment location will be 4
                    if(j < minCCLength):
                        data[0:j,9] = ccID
                        data[0:j, 10] = 4

                    # if we have short streams (< pixThr+bridge), the bridging option will be turned off and the connectivity of the stream is investigated from ch to outlet
                    elif(j < pixThr+bridge):
                        print("Short stream. Cannot bridge pixels.")
                        for k in range(j+1):
                            slopeInvestigated = data[k,7]
                            if(k == 0): #first pixel cannot be compared to other ones
                            #add slope to mean cc slope df
                                mList.append(slopeInvestigated) #append current slope
                                mean = np.mean(mList) #calculate mean
                                #add current ccID
                                data[k, 9] = ccID
                                data[k, 10]=segmentLocationcount

                            else:
                                if(abs(mean-slopeInvestigated)<=dSlopeThr or len(mList) < minCCLength): #if Slope of current pixel is smaller than the mean of the current connected segment

                                    data[k, 9]= ccID #assign the same ccID
                                    data[k, 10]=segmentLocationcount
                                    mList.append(slopeInvestigated) #add Slope to Slope list and compute mean
                                    mean = np.mean(mList)

                                    #if the stream ends naturally, the CC needs to be terminated as well (only thing missing is calculating the dPrevSLp)
                                    if(k == j):
                                        try:
                                            data[np.where(data[:,9]==ccID),11]= mean-previousSlps[-1]
                                        except IndexError: #if list previousSlps is empty, it will result in an index error
                                            pass


                                else: #terminate segment
                                    #calculate slope change with regard to previous CC
                                    try:
                                        data[np.where(data[:,9]==ccID),11]= mean-previousSlps[-1]
                                    except IndexError: #if list previousSlps is empty, it will result in an index error -> except
                                        pass
                                    ccID+=1 #increase CCID
                                    data[k, 9] = ccID #assign new ccID

                                    mList = [] #clear list
                                    mList.append(slopeInvestigated)
                                    previousSlps.append(mean) #append previous Slope values to list to be able to compare them later on
                                    mean = np.mean(mList)
                                    if(segmentLocationcount == 1): #increase segmentLocationcount to two if a first segment already exists
                                        segmentLocationcount = 2
                                    data[k, 10]=segmentLocationcount

                    # if the stream is sufficiently long, finish calculating CCs
                    else:
                        for k in range(j-pixThr-bridge, j+1, 1):
                            slopeInvestigated = data[k,7]

                            if(k == 0): #first pixel cannot be compared to other ones
                                #add slope to mean cc slope df
                                mList.append(slopeInvestigated) #append current slope
                                mean = np.mean(mList) #calculate mean
                                #add current ccID
                                data[k, 9] = ccID
                                data[k, 10]=segmentLocationcount

                            else:
                                #look five pixels ahead to see if there are any slopes that match the mean so that a CC would be bridged
                                futureLook = data[k+1:k+1+bridge,7]
                                #mask to avoid runtime error
                                futureLook = futureLook[~np.isnan(futureLook)]

                                if(abs(mean-slopeInvestigated)<=dSlopeThr or len(mList) < minCCLength): #if Slope of current pixel is smaller than the mean of the current connected segment

                                    data[k, 9]= ccID #assign the same ccID
                                    data[k, 10]=segmentLocationcount
                                    mList.append(slopeInvestigated) #add Slope to Slope list and compute mean
                                    mean = np.mean(mList)

                                    #if the stream ends naturally, the CC needs to be terminated as well (only thing missing is calculating the dPrevSLp)
                                    if(k == j):
                                        try:
                                            data[np.where(data[:,9]==ccID),11]= mean-previousSlps[-1]
                                        except IndexError: #if list previousSlps is empty, it will result in an index error
                                            pass

                                #bridge outliers, these wont be added to the mean list
                                elif(np.any(abs(mean-futureLook)<=dSlopeThr)):

                                    #if Slope of current pixel is smaller than the mean of the current connected segment, but up to 5 pixels downstream the slope is still within threshold bounds
                                    data[k, 9] = ccID
                                    data[k, 10]=segmentLocationcount

                                else: #terminate segment
                                    #calculate slope change with regard to previous CC
                                    try:
                                        data[np.where(data[:,9]==ccID),11]= mean-previousSlps[-1]
                                    except IndexError: #if list previousSlps is empty, it will result in an index error -> except
                                        pass
                                    ccID+=1 #increase CCID
                                    data[k, 9] = ccID #assign new ccID

                                    mList = [] #clear list
                                    mList.append(slopeInvestigated)
                                    previousSlps.append(mean) #append previous Slope values to list to be able to compare them later on
                                    mean = np.mean(mList)
                                    if(segmentLocationcount == 1): #increase segmentLocationcount to two if a first segment already exists
                                        segmentLocationcount = 2
                                    data[k, 10]=segmentLocationcount

            j+=1

        #for sufficiently long channels: if the final CC is too short, merge with previous one
        if(j >= minCCLength and len(mList)<minCCLength and ccID>0):
            #mList = data[np.where(data[:,9]==ccID),7][0] #get old slopes back
            segmentLocationcount = data[np.where(data[:,9]==ccID-1),10][0][0] #get the last segment location count
            data[np.where(data[:,9] == ccID), 10] = segmentLocationcount #assign it to current ccID
            data[np.where(data[:,9] == ccID), 9]= ccID-1 #assign old ccID where the current ccID was set
            if len(previousSlps)>0:
                previousSlps = np.delete(previousSlps, -1) #delete last Value from previous slopes
            #turn back ccID
            ccID -= 1
        #finish assigning the correct segment location:
        #if the terminal pixel of a stream is reached
        #and the segment location is still = 1
        #the entire stream is still one CC and needs to be assigned the value 4
        if(j>= minCCLength and currentNode == finalNode):
            if(segmentLocationcount == 1):
                data[:,10]=4
            # else if there are no middle segments and the steam does come to a natural end, set the segement Location of the last CC to 3
            elif(segmentLocationcount ==2):
                data[np.where(data[:,9]==ccID),10]=3

        data = data[~np.isnan(data[:, 9])] #remove everything that wasnt assigned a CC

        #convert array to pandas df
        #if there are isolated single pixel streams in one catchment, data will be empty, causing a ValueError
        try:
            df = pd.concat([df, pd.DataFrame({"X": data[:,0], "Y": data[:,1],"StreamID":ii, "BasinID":basin, "Elevation": data[:,2],
                            "DrainageArea":data[:,4], "XYDistanceToNextPixel": data[:,3], "DownstreamDistance":np.cumsum(data[:,3]), "3DDistanceToNextPixel":data[:,5], "ksn":data[:,6],
                            "FlowDistance_Catchment":data[:,12],"Slope":data[:,7], "R2":data[:,8],
                            "ccID":data[:,9], "segmentLocation":data[:,10], "dSlopeToPrevSegment":data[:,11]})])
        except ValueError:
            pass

        #assign truly unique CCIDs
    df["ccID"]=df["BasinID"].map(str)+"_"+df['StreamID'].map(str)+"_"+df['ccID'].map(str)

    #fill NA values for last stream pixel (computation of distance to next pixel not possible)
    df.XYDistanceToNextPixel = df.XYDistanceToNextPixel.fillna(0)
    df["3DDistanceToNextPixel"] = df["3DDistanceToNextPixel"].fillna(0)
    df.DownstreamDistance = df.DownstreamDistance.fillna(0)
    return df

#####################################################################################################################

def processBasinWithoutCCs(basin, lsdttTable, heads, pixThr):
    #function calculates channel slope for given basin
    #main purpose is to find a suitable pixel threshold
    df = pd.DataFrame(columns = ["X","Y","StreamID","BasinID", "Elevation", "DrainageArea", "XYDistanceToNextPixel", "DownstreamDistance", "3DDistanceToNextPixel","ksn" ,"Slope", "R2", "SlopeChange"])
    #print(basin)
    #subset channel heads by catchment
    bheads = heads.loc[(heads.basin_key == basin)].reset_index(drop = True)
    for ii, currentNode in enumerate(bheads.node):

        #print("Processing channelhead "+str(ii+1)+"/"+str(len(bheads)))
        nextNode = bheads.RNI[ii]

        #set duplication counter to know when to stop processing a stream after an intersection
        dupCounter = 0
        #generate empty elevation and dsDist array to fit regressions and calculate channel slope
        elev = np.zeros(shape=(2*pixThr+1))
        elev[:] = np.nan
        dsDist = np.zeros(shape=(2*pixThr+1))
        dsDist[:] = np.nan

        #empty array for storing data
        data = np.zeros([60000,9]) #if there are really really long channels (> 60000 pixels), the array might need to be extended
        data[:] = np.nan
        j=0

        #start going down a channel
        #do so while the current node is not the next node (happens at channel outlet) and there is still space in the array
        while(currentNode != nextNode and j < len(data)):

            #identify data at current and next node
            crrnt = lsdttTable.loc[np.where(lsdttTable.node == currentNode)]
            nxt = lsdttTable.loc[np.where(lsdttTable.node == nextNode)]

            #break the loop if a next Node is not available
            #this might happen, if the lsdttTable is subsetted
            if nxt.empty:
                currentNode = nextNode
            else:
                #compute distance between two points
                #distance in XY direction only
                distXY = np.sqrt((crrnt.x.iloc[0]-nxt.x.iloc[0])**2+(crrnt.y.iloc[0]-nxt.y.iloc[0])**2)
                #3D distance
                dist3D = np.sqrt((crrnt.x.iloc[0]-nxt.x.iloc[0])**2+(crrnt.y.iloc[0]-nxt.y.iloc[0])**2+(crrnt.elevation.iloc[0]-nxt.elevation.iloc[0]))

                #dynamic elevation and dsDist array to compute regression from
                #start filling in at the beginning and remove values at the end
                elev = np.insert(elev, 0, crrnt.elevation.iloc[0]) #add current elevation
                elev = np.delete(elev, 2*pixThr+1) #remove last

                #same for downstream distance
                dsDist = np.insert(dsDist, 0, distXY)
                dsDist = np.delete(dsDist, 2*pixThr+1)

                #now store some data in my array
                data[j,0]= crrnt.x.iloc[0] #x coordinate
                data[j,1]= crrnt.y.iloc[0] #y coordinate
                data[j,2] = crrnt.elevation.iloc[0] #elevation
                data[j,3] = distXY #dsDist XY
                data[j,4] = crrnt.drainage_area.iloc[0] #drainage area
                data[j,5] = dist3D #dsDistXYZ
                data[j,6] = crrnt.m_chi.iloc[0] #ksn

                #find next node
                currentNode = nextNode
                nextNode = lsdttTable.loc[np.where(lsdttTable.node == currentNode)].RNI.iloc[0]

                #if the current combination of X and Y coordinates is already in the output dataframe, increase duplication counter
                #calculations need to continue pixThr many pixels after an intersection to finish calculating the slope
                if((df[['X','Y']].values == [data[j,0], data[j,1]]).all(axis=1).any()):
                    dupCounter += 1
                    if(dupCounter >= 2*pixThr): # if a full regression length after the intersection is reached (to make sure that all unique slope values are counted),
                    #set nextNode to Current Node to stop
                        nextNode = currentNode
                ######################################
                #calculate slope for the running stream
                if(j>=pixThr): #when a significant amount of pixels (=pixelThreshold) is reached, start computing channel slope
                    #compute regression
                    #print("Calculating slope for point "+str(j-pixThr))
                    #remove nan if still present
                    mask = ~np.isnan(dsDist) & ~np.isnan(elev)
                    reg = linregress(np.cumsum(dsDist[mask]),elev[mask])
                    data[j-pixThr,7] = abs(reg[0]) #slope
                    data[j-pixThr,8] = reg[2] #R2
                    # plt.figure(figsize=(11.69,8.27))
                    # plt.scatter(np.cumsum(dsDist[mask]),elev[mask], color = "gray")
                    # plt.hlines(data[j-pixThr,2], 0 ,40)
                    # plt.show()

            #if catchments are cutoff by previously masking the lsdtt flowlines, small cutoff streams will cause errors
            if(currentNode == nextNode and dupCounter == 0 and j==1):
                print("Small stream composed of single pixel removed.")
            ##########################################
            #calculate slope at end of stream
            else:
                if(currentNode == nextNode): # currentNode = nextNode (end of stream)
                    #compute the remaining slope values (slope comutation is lagging by number of pixels chosen as pixel Threshold)
                    for k in range(j-pixThr, j, 1):
                        #print("Calculating slope for point "+str(k))
                        elev = np.insert(elev, 0, np.nan) #fill with nan from start
                        elev = np.delete(elev, 2*pixThr+1) #remove last
                        dsDist = np.insert(dsDist, 0, np.nan)
                        dsDist = np.delete(dsDist, 2*pixThr+1)
                        mask = ~np.isnan(dsDist) & ~np.isnan(elev)
                        reg = linregress(np.cumsum(dsDist[mask]),elev[mask])
                        data[k,7] = abs(reg[0]) #slope
                        data[k,8] = reg[2] #R2

            j+=1


        data = data[~np.isnan(data[:, 7])] #remove spare space in array

        #convert array to pandas df
        #if there are isolated single pixel streams in one catchment, data will be empty, causing a ValueError
        try:
            df = pd.concat([df, pd.DataFrame({"X": data[:,0], "Y": data[:,1],"StreamID":ii, "BasinID":basin, "Elevation": data[:,2],
                            "DrainageArea":data[:,4], "XYDistanceToNextPixel": data[:,3], "DownstreamDistance":np.cumsum(data[:,3]),
                            "3DDistanceToNextPixel":data[:,5], "ksn":data[:,6],"Slope":data[:,7], "R2":data[:,8], "SlopeChange":np.append(abs(np.diff(data[:,7])),np.nan)})])
        except ValueError:
            pass
    return df

#####################################################################################################################

def runCCAnalysis(fname, path, lsdttTable, pixThr = 7, dSlopeThr = 0.21, bridge = 5, minCCLength = 2, mask = "", epsg = 4326):
    #function to execute the processBasin function by passing different tasks to different cores

    #get all nodes that are NOT in reciever nodes as channel heads
    heads = lsdttTable[~lsdttTable.node.isin(lsdttTable.RNI)].reset_index(drop=True)

    #optionally constrain channelheads to be within AOI
    if mask != "":
        #convert to spatial points
        geometry = [Point(xy) for xy in zip(heads.longitude, heads.latitude)]
        crs = 'EPSG:'+str(epsg)
        gdf = gpd.GeoDataFrame(heads, crs=crs, geometry=geometry)
        #load mask
        poly = gpd.read_file(path+mask)
        if not poly.crs == crs:
            poly = poly.to_crs(crs)
        #clip with mask
        heads = gpd.clip(gdf,poly)
        heads = heads.drop(columns = ["geometry"])

    #now get all basins
    print("There are "+str(heads["basin_key"].nunique())+" catchments.\n")

    #loop over all basins in csv file and give each task to a different core
    #pass tasks
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     results = [executor.submit(processBasin, basin = basin , lsdttTable = lsdttTable.loc[lsdttTable.basin_key == basin].reset_index(drop = True), heads = heads, pixThr = pixThr, dSlopeThr = dSlopeThr, bridge = bridge, minCCLength = minCCLength) for basin in heads.basin_key.unique()]

    #pass tasks with progress bar
    l = len( heads.basin_key.unique())
    with tqdm(total=l) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(processBasin, basin = basin , lsdttTable = lsdttTable.loc[lsdttTable.basin_key == basin].reset_index(drop = True), heads = heads, pixThr = pixThr, dSlopeThr = dSlopeThr, bridge = bridge, minCCLength = minCCLength): basin for basin in heads.basin_key.unique()}
            results = {}
            for future in concurrent.futures.as_completed(futures):
                arg = futures[future]
                results[arg] = future.result()
                pbar.update(1)
    #combine results
    out = pd.DataFrame(columns = ["X","Y","StreamID","BasinID", "Elevation", "DrainageArea", "XYDistanceToNextPixel", "DownstreamDistance", "3DDistanceToNextPixel",  "Slope", "R2", "ksn", "FlowDistance_Catchment", "ccID", "segmentLocation", "dSlopeToPrevSegment"])

    for task in results:
        #print(task)
        try:
            out = pd.concat([out,results[task]])
        except ValueError:
            pass

    out = out.reset_index(drop = True)
    #now get statistics per CC
    cc = out.groupby('ccID').agg({'3DDistanceToNextPixel': 'sum', 'Slope':  ['mean', 'std'], 'DrainageArea': ['min', 'max'],
                                   'segmentLocation': 'max', "dSlopeToPrevSegment":["max"], 'ksn':["mean"]})

    #rename columns
    cc = cc.reset_index()
    cc.columns = ["_".join(x) for x in cc.columns.ravel()]
    cc.columns = ['ccID', 'ccLength', 'ccMeanSlope', 'ccStdSlope', 'minDrainageArea', 'maxDrainageArea','segmentLocation', 'slopeChangeToPrevCC', 'meanKSN']

    #remove individual small streams that are shorter than the minmal required CC length
    shortStreams = out.groupby("ccID").Slope.count().reset_index()
    shortStreams = shortStreams.loc[shortStreams.Slope < minCCLength]

    if len(shortStreams.ccID)>0:
        out = out.loc[~out.ccID.isin(shortStreams.ccID)]
        cc = cc.loc[~cc.ccID.isin(shortStreams.ccID)]

        print(str(len(shortStreams.ccID)) + " stream(s) was removed, because they were shorter than the minimal required CC length.")

    #write streams
    out.to_csv(path+fname+"_ConnectedComponents_streams_"+str(pixThr)+"_"+str(dSlopeThr)+"_"+str(bridge)+".csv", index = False)
    print("Streams with assigned CC ID were written to "+fname+"_ConnectedComponents_streams_"+str(pixThr)+"_"+str(dSlopeThr)+"_"+str(bridge)+".csv")

    #write aggregated parameters
    cc.to_csv(path+fname+"_ConnectedComponents_"+str(pixThr)+"_"+str(dSlopeThr)+"_"+str(bridge)+".csv", index = False)

    print("Aggregated parameters for all CCs were written to "+fname+"_ConnectedComponents_"+str(pixThr)+"_"+str(dSlopeThr)+"_"+str(bridge)+".csv")

#####################################################################################################################

def getCatchmentsInAOI(fname, path, mask, epsg = 4326):

    #function to subset catchments to an area of interest
    #useful if basins in area of interest are cutoff + filled with a different elevatio model
    #previous subsetting is not necessary, if channelheads are spatially contrained with a mask when running the CC Analysis

    #load lsdtt df
    df = pd.read_csv(path+fname+"_MChiSegmented.csv")
    #convert to spatial points
    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    crs = 'EPSG:'+str(epsg)
    gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    #load mask
    poly = gpd.read_file(path+mask)
    if not poly.crs == crs:
        poly = poly.to_crs(crs)
    #clip with mask
    catchOI = gpd.clip(gdf,poly)
    #find catchments inside mask layer
    df = df.loc[df.basin_key.isin(catchOI.basin_key)]
    df = df.drop(columns = ["geometry"])
    df.to_csv(path+fname+'_MChiSegmented_relevantBasins.csv', index = False)
    print("The file "+fname+'_MChiSegmented_relevantBasins.csv has been written.')

#####################################################################################################################

def getCCsInAOI(fname, path, mask, epsg = 32719, pixThr= 7, dSlopeThr = 0.21, bridge = 5):
    #function to retrieve CCs within Area of interest

    #load CC streams
    df = pd.read_csv(path+fname+"_ConnectedComponents_streams_"+str(pixThr)+"_"+str(dSlopeThr)+"_"+str(bridge)+".csv")
    #load aggreagted CCs
    df2 = pd.read_csv(path+fname+"_ConnectedComponents_"+str(pixThr)+"_"+str(dSlopeThr)+"_"+str(bridge)+".csv")
    #convert to spatial points
    geometry = [Point(xy) for xy in zip(df.X, df.Y)]
    crs = 'EPSG:'+str(epsg)
    gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    #load mask
    poly = gpd.read_file(path+mask)
    if not poly.crs == crs:
        poly = poly.to_crs(crs)

    #clip with mask
    CCsOI = gpd.clip(gdf,poly)

    #find CCs inside mask layer
    df2 = df2.loc[df2.ccID.isin(CCsOI.ccID)]
    CCsOI = CCsOI.drop(columns = ["geometry"])
    CCsOI.to_csv(path+fname+"_ConnectedComponents_streams_"+str(pixThr)+"_"+str(dSlopeThr)+"_"+str(bridge)+"_inAOI.csv", index = False)
    df2.to_csv(path+fname+"_ConnectedComponents_"+str(pixThr)+"_"+str(dSlopeThr)+"_"+str(bridge)+"_inAOI.csv", index = False)

    print("The files "+fname+"_ConnectedComponents_"+str(pixThr)+"_"+str(dSlopeThr)+"_"+str(bridge)+"_inAOI.csv\nand "+
          fname+"_ConnectedComponents_streams_"+str(pixThr)+"_"+str(dSlopeThr)+"_"+str(bridge)+"_inAOI.csv were written.")


#####################################################################################################################

def findDSlopeThresholdDebrisSamples(lsdttTable, pixThr = 7, bridge = 5, thresholdRange = np.round(np.arange(0.05,0.31,0.01),2), minCCLength = 2, writeCSV = False):
    #similar to runCCAnalysis Script. Process is repeated for different thresholds and output CCs are compared
    #lsdttTable of debris flow sample regions should be provided

    #make sure that thresholds are in ascending order
    thresholdRange = np.sort(thresholdRange)
    #get channelheads
    heads = lsdttTable[~lsdttTable.node.isin(lsdttTable.RNI)].reset_index(drop=True)
    #empty array for storing data
    data = np.zeros([len(thresholdRange),3])

    for ii, thr in enumerate(tqdm(thresholdRange, desc = "Finding CC using different slope-change thresholds")):
        #print("Testing a threshold of "+str(thr)+"...")

        #pass tasks
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(processBasin, basin = basin , lsdttTable = lsdttTable.loc[lsdttTable.basin_key == basin].reset_index(drop = True), heads = heads, pixThr = pixThr, dSlopeThr = thr, bridge = bridge, minCCLength = minCCLength) for basin in heads.basin_key.unique()]
        #combine results
        out = pd.DataFrame(columns = ["X","Y","StreamID","BasinID", "Elevation", "DrainageArea", "XYDistanceToNextPixel", "DownstreamDistance", "3DDistanceToNextPixel",  "Slope", "R2", "ksn", "FlowDistance_Catchment","ccID", "segmentLocation", "dSlopeToPrevSegment"])

        for task in concurrent.futures.as_completed(results):
            try:
                out = pd.concat([out,task.result()])
            except ValueError:
                pass
        #get toal number of streams
        nrStreamsTotal = len(out.drop_duplicates(['BasinID','StreamID']))

        #aggregate by CC ID
        cc = out.groupby('ccID').agg({'segmentLocation': 'max'})
        #get number of CCs that for 1CC per stream only
        nr1CConly = len(cc[cc.segmentLocation == 4])
        data[ii,0] = thr
        data[ii,1] = nrStreamsTotal
        data[ii,2] = nr1CConly

    #find optimal threshold = first value that lies above 0.5
    #but first check if any threshold is at all capable to cross the required mark of 50% single segments per stream
    if(np.max(data[:,2]/data[:,1])<0.5):
        print("None of the tested thresholds produces 1 CC / stream for at least 50% of the given channels. Consider a different threshold range and re-run. ")
        optThr = np.nan
    else:
        pos = np.argmax(data[:,2]/data[:,1] >=0.5)
        optThr = data[pos,0]
        print("Out of the provided slope-change thresholds "+str(thresholdRange)+", the recommended value is "+str((np.round(optThr,2))))

    #plot
    plt.figure(figsize=(11.69,8.27))
    plt.plot(data[:,0], data[:,2]/data[:,1])
    plt.scatter(data[:,0], data[:,2]/data[:,1])
    plt.hlines(0.5, xmin = thresholdRange[0], xmax = thresholdRange[-1], linestyle='-')
    plt.vlines(optThr, ymin = 0, ymax = nr1CConly/nrStreamsTotal, linestyle='-', color = "red")
    plt.xlabel("Threshold")
    plt.ylabel("Fraction of single component streams")
    plt.title("Optimal slope-change threshold to constrain CCs")
    plt.grid()
    plt.show()

    #optional: write output to csv
    if writeCSV:
        csv = pd.DataFrame(data)
        csv.columns = ["Threshold", "TotalNrStreams", "SingleCCStreams"]
        csv.to_csv("numberOfSingleCCStreams_variousSlopeChangeThresholds.csv", index = False)

    return(np.round(optThr, 2))


#####################################################################################################################

def findDSlopeThresholdAlternative(lsdttTable,dist, pixThr = 7, sampleStreams = -1, sampleBasinID = -1, mask = "", epsg = 4326, path = "./", fname = "StreamNetwork", writeCSV = False, saveFigs = False):
     #subset lsdttTable by basin (if desired) to speed up processing

    if sampleBasinID >= 0:
        #print("Subsetting the input to only sample streams from basin nr. "+str(sampleBasinID)+".")
        lsdttTable = lsdttTable.loc[np.where(lsdttTable.basin_key == sampleBasinID)].reset_index(drop=True)

    #get channelheads
    heads = lsdttTable[~lsdttTable.node.isin(lsdttTable.RNI)].reset_index(drop=True)


    #optionally constrain channelheads to be within AOI
    if mask != "":
        #convert to spatial points
        geometry = [Point(xy) for xy in zip(heads.longitude, heads.latitude)]
        crs = 'EPSG:'+str(epsg)
        gdf = gpd.GeoDataFrame(heads, crs=crs, geometry=geometry)
        #load mask
        poly = gpd.read_file(path+mask)
        if not poly.crs == crs:
            poly = poly.to_crs(crs)
        #clip with mask
        heads = gpd.clip(gdf,poly)
        heads = heads.drop(columns = ["geometry"])

    #get a random subset of x channelheads to speed up processing (if desired)
    if sampleStreams > 0:
        #print("Subsetting the input and sampling "+str(sampleStreams)+" random streams.")
        heads = heads.sample(n = sampleStreams, random_state=123).reset_index(drop = True)

    #calculate slopes
    #with concurrent.futures.ProcessPoolExecutor() as executor:
        #results = [executor.submit(processBasinWithoutCCs, basin = basin , lsdttTable = lsdttTable, heads = heads, pixThr = 7) for basin in heads.basin_key.unique()]
        #combine results

    #pass tasks with progress bar
    l = len( heads.basin_key.unique())
    with tqdm(total=l) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(processBasinWithoutCCs, basin = basin , lsdttTable = lsdttTable, heads = heads, pixThr = 7): basin for basin in heads.basin_key.unique()}
            results = {}
            for future in concurrent.futures.as_completed(futures):
                arg = futures[future]
                results[arg] = future.result()
                pbar.update(1)
    #combine results
    out = pd.DataFrame(columns = ["X","Y","StreamID","BasinID", "Elevation", "DrainageArea", "XYDistanceToNextPixel", "DownstreamDistance", "Slope", "R2", "SlopeChange"])

    for task in results:
        out = pd.concat([out,results[task]])

    #turn dist into a list if the function was not given one
    thrs = []
    if not isinstance(dist, list):
        dist = [dist]
    for d in dist:
        slpChange = out.groupby(["BasinID", "StreamID"]).Slope.diff(periods=d)
        p5 = np.nanpercentile(slpChange, 5)
        p95 = np.nanpercentile(slpChange, 95)
        # m = np.nanmean(slpChange)
        # sd = np.nanstd(slpChange)
        fig = plt.figure(figsize=(11.69,8.27))
        sns.kdeplot(slpChange, linewidth = 2)
        plt.vlines(p5,plt.ylim()[0], plt.ylim()[1], linestyles='-', color = "firebrick", label = "5th percetile: "+str(np.round(p5,2)))
        plt.vlines(p95,plt.ylim()[0], plt.ylim()[1], linestyles='-', label = "95th percetile: "+str(np.round(p95,2)))
        plt.title("Density distribution of slope change over "+str(d)+" pixels")
        plt.xlabel("Slope change")
        plt.legend()
        plt.grid()
        if saveFigs:
            fig.savefig(path+"SlopeChangeDistribution_"+str(d)+"steps.png", dpi = 200)
        plt.show()

        #append reccomended threshold to list
        thrs.append(np.round(np.mean([abs(p5),abs(p95)]),2))
        print("Investigating slope change over " + str(d) + " pixels and the reccomend value is "+ str(np.round(np.mean([abs(p5),abs(p95)]),2)))

    if writeCSV:
        out.to_csv(path+fname+"_channelSlope_"+str(pixThr)+"px.csv", index = False)


    if len(thrs)==1:
        thrs = thrs[0]
    return(thrs)
#####################################################################################################################


def findPixelThreshold(lsdttTable, thresholdRange = np.arange(1,26), sampleBasinID = -1, sampleStreams = -1, mask = "", epsg = 4326, path = "./", writeCSV = False):
    #make sure that thresholds are in ascending order
    thresholdRange = np.sort(thresholdRange)
    #empty array for storing data
    data = np.zeros([len(thresholdRange),4])

    #subset lsdttTable by basin (if desired) to speed up processing
    if sampleBasinID >= 0:
        #print("Subsetting the input to only sample streams from basin nr. "+str(sampleBasinID)+".")
        lsdttTable = lsdttTable.loc[np.where(lsdttTable.basin_key == sampleBasinID)].reset_index(drop=True)

    #get channelheads
    heads = lsdttTable[~lsdttTable.node.isin(lsdttTable.RNI)].reset_index(drop=True)


    #optionally constrain channelheads to be within AOI
    if mask != "":
        #convert to spatial points
        geometry = [Point(xy) for xy in zip(heads.longitude, heads.latitude)]
        crs = 'EPSG:'+str(epsg)
        gdf = gpd.GeoDataFrame(heads, crs=crs, geometry=geometry)
        #load mask
        poly = gpd.read_file(path+mask)
        if not poly.crs == crs:
            poly = poly.to_crs(crs)
        #clip with mask
        heads = gpd.clip(gdf,poly)
        heads = heads.drop(columns = ["geometry"])

    #get a random subset of x channelheads to speed up processing (if desired)
    if sampleStreams > 0:
        #print("Subsetting the input and sampling "+str(sampleStreams)+" random streams.")
        heads = heads.sample(n = sampleStreams, random_state=123).reset_index(drop = True)

    #loop over thresholdRange
    for ii, thr in enumerate(tqdm(thresholdRange, desc = "Calculating channel-slope using different regression lengths")):
        #print("Testing a threshold of ±"+str(thr)+ " pixel(s).")
        #pass tasks
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(processBasinWithoutCCs, basin = basin , lsdttTable = lsdttTable, heads = heads, pixThr = int(thr)) for basin in heads.basin_key.unique()]
        #combine results
        out = pd.DataFrame(columns = ["X","Y","StreamID","BasinID", "Elevation", "DrainageArea", "XYDistanceToNextPixel", "DownstreamDistance", "Slope", "R2", "SlopeChange"])

        for task in concurrent.futures.as_completed(results):
            out = pd.concat([out,task.result()])

        #calculate median, and 25th + 75th percentile
        data[ii, 0] = thr
        data[ii, 1] = np.nanmedian(out.SlopeChange)
        data[ii, 2] = np.nanpercentile(out.SlopeChange, 25)
        data[ii, 3] = np.nanpercentile(out.SlopeChange, 75)

    #locate kneepoint
    kn = KneeLocator(
    np.array(data[:,0]),
    np.array(data[:,1]),
    curve='convex',
    direction='decreasing')

    #plot
    plt.figure(figsize=(11.69,8.27))
    plt.plot(data[:,0], data[:,1])
    plt.errorbar(data[:,0], data[:,1], yerr = [ data[:,2], data[:,3]], color = "blue")
    plt.scatter(data[:,0], data[:,1])
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='-', color = "red")
    plt.xlabel("Pixels up- and downstream of the current node")
    plt.ylabel("Pixel-to-pixel slope change")
    plt.title("Optimal regression length for calculating channel slope")
    plt.grid()
    plt.show()

    #optional: write output to csv
    if writeCSV:
        csv = pd.DataFrame(data)
        csv.columns = ["Threshold", "Median", "P25", "P75"]
        csv.to_csv(path+"slopeChange_variousPixelThresholds.csv", index = False)

    print("Out of the given pixel thresholds "+str(thresholdRange)+ ", the recommended value is "+str(int(kn.knee))+".")

    return(int(kn.knee))


#####################################################################################################################

def componentClustering(path, allCCName, debrisName ="", pixThr = 7, dSlopeThr = 0.21, bridge = 5, allExt = "", debExt = "", k = 2, clusterParameters = ["ccLength", "ccMeanSlope", "segmentLocation", "slopeChangeToPrevCC", "distDFSamples"]):

    #load all CCs
    cc =  pd.read_csv(path+allCCName+"_ConnectedComponents_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+allExt+".csv")
    #cc = cc.loc[cc.ccMeanSlope<=1.5].reset_index(drop=True)
    print("Debris-flow similarity values will be assigned to all CCs in"+ allCCName+"_ConnectedComponents_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+allExt+".csv")

    validParameters = ["ccLength", "ccMeanSlope", "segmentLocation", "slopeChangeToPrevCC", "distDFSamples"]
    #check if given input paramters are valid
    if not np.all(np.in1d(clusterParameters, validParameters)):
        print("Please provide valid clustering Parameters. These are:")
        print(validParameters)
        return
    #check if debris-flow sample distances are chosen as a clustering parameter and if yes calculate weights + distances
    if "distDFSamples" in clusterParameters:
        if debrisName == "":
            print("Distance to debris-flow samples was chosen as a clustering parameter, but no sample input was provided. Exiting...")
            return
        else:

            #load debris samples
            print("The file "+debrisName+"_ConnectedComponents_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+debExt+".csv is used as debris-flow sample file.")
            deb = pd.read_csv(path+debrisName+"_ConnectedComponents_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+debExt+".csv")
            #remove single pixel CCs
            deb = deb.loc[deb['ccLength'] >0].reset_index(drop = True)

            # logtransform and normalize data
            deb["ccLengthNorm"] = normalize(np.log10(deb.ccLength))
            deb["ccMeanSlopeNorm"] = normalize(deb.ccMeanSlope)
            cc["ccLengthNorm"] = normalize(np.log10(cc.ccLength))
            cc["ccMeanSlopeNorm"] = normalize(cc.ccMeanSlope)

            #calculateDensity
            xy = np.vstack([deb.ccLengthNorm,deb.ccMeanSlopeNorm])
            deb["Weight"]=normalize(gaussian_kde(xy)(xy))

            #calculate sum of distances for all points
            def calcSummedDistances(length, slope, deb):
                sumofDist = (length-deb.ccLengthNorm)**2+(slope-deb.ccMeanSlopeNorm)**2
                weightedSOD = deb.Weight*sumofDist
                return sum(weightedSOD)

            cc["distDFSamples"] = cc.apply(lambda row: calcSummedDistances(row['ccLengthNorm'], row['ccMeanSlopeNorm'], deb), axis=1)


            #plot DF CCs and assigned weight
            # fig = plt.figure(figsize=(11.69,8.27))
            # ax = fig.add_subplot()
            # ax.scatter(cc.ccLength, cc.ccMeanSlope, s=1, color = "gray", label = "all CCs")
            # ax.scatter(deb.ccLength, deb.ccMeanSlope, s = deb.Weight*30, c = "r", label= "Debris-flow samples")
            # ax.set_xscale('log')
            # ax.set_xlabel("Mean CC Length [m]")
            # ax.set_ylabel("Mean CC Slope [m/m]")
            # ax.set_ylim(0,1.5)
            # plt.legend()
            # plt.show()

            # #density plot of distance distribution.

            # plt.figure(figsize=(11.69,8.27))
            # sns.kdeplot(cc.distDFSamples, linewidth = 2)
            # plt.title("Density distribution of summed distances to debris-flow samples")
            # plt.xlabel("Sum of distances")
            # plt.show()

            cc = cc.drop(columns = ["ccLengthNorm","ccMeanSlopeNorm"])

    #CCs are split into k groups using kmeans clustering

    print("Clustering ...")

    cdf = cc.copy()[cc.columns.intersection(clusterParameters)].fillna(0)

    if "ccLength" in clusterParameters:
        cdf.ccLength = np.log10(cdf.ccLength)

    scaled = StandardScaler().fit_transform(pd.DataFrame(cdf))
    km = KMeans(
        n_clusters = k, init='random',
        n_init=10, max_iter=100,
        tol=1e-04, random_state=123
    )
    cc["clusterKM"] = km.fit_predict(scaled)



    #plot results
    fig = plt.figure(figsize=(11.69,8.27))
    ax = fig.add_subplot()
    s = ax.scatter(cc.ccLength, cc.ccMeanSlope, s=5, c = cc.clusterKM, cmap = "Spectral")
    ax.set_xscale('log')
    ax.set_xlabel("Mean CC Length [m]")
    ax.set_ylabel("Mean CC Slope [m/m]")
    ax.set_ylim(0,1.5)
    ax.set_title("Assigned cluster for a slope-change threshold of "+ str(dSlopeThr))
    fig.colorbar(s)
    plt.show()

    #investigate correlation of clustering parameters
    scaled = pd.DataFrame(scaled, columns = cdf.columns)
    #compute pairwise correlation
    corrmat = scaled.corr()

    #plot heatmap (only if there are more than one clustering parameter)
    if len(clusterParameters)>1:
        plt.figure(figsize=(11.69,8.27))
        plt.title("Pearson correlation of clustering parameters")
        sns.heatmap(corrmat,annot=True,cmap="coolwarm")
        plt.show()

    return(pd.DataFrame({"ccID": cc.ccID, "Cluster":cc.clusterKM, "ccLength": cc.ccLength}))

#####################################################################################################################
def assignDFSI(path, allCCName, debrisName ="", pixThr = 7, dSlopeThr = 0.21, bridge = 5, l = 0.5, s = 0.5, allExt = "", debExt = "", debrisSlopeHigh = 0.6, debrisLengthHigh= np.log10(500), writeCSV = False):

    #load all CCs
    cc =  pd.read_csv(path+allCCName+"_ConnectedComponents_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+allExt+".csv")
    #cc = cc.loc[cc.ccMeanSlope<=1.5].reset_index(drop=True)
    print("Assigning debris-flow similarity values to all CCs in"+ allCCName+"_ConnectedComponents_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+allExt+".csv")

    if debrisName == "":
        print("No sample input provided. Scaling according to CC length of "+str(debrisLengthHigh)+ " and a slope of "+str(debrisSlopeHigh))
    else:
        #load debris samples
        print("The file "+debrisName+"_ConnectedComponents_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+debExt+".csv is used for debris-flow samples.")
        deb = pd.read_csv(path+debrisName+"_ConnectedComponents_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+debExt+".csv")
        #remove single pixel CCs
        deb = deb.loc[deb['ccLength'] >0].reset_index(drop = True)

        # logtransform and normalize data
        deb["ccLengthNorm"] = normalize(np.log10(deb.ccLength))
        deb["ccMeanSlopeNorm"] = normalize(deb.ccMeanSlope)

        #calculateDensity
        xy = np.vstack([deb.ccLengthNorm,deb.ccMeanSlopeNorm])
        deb["Weight"]=normalize(pd.Series(gaussian_kde(xy)(xy)))

        #get a weighted average DF slope and length

        debrisSlopeHigh = np.average(deb.ccMeanSlope, weights = deb.Weight)
        debrisLengthHigh = np.average(np.log10(deb.ccLength), weights = deb.Weight)

        debrisSlopeLow = deb.ccMeanSlope.quantile(0.05)
        debrisLengthLow = np.log10(deb.ccLength).quantile(0.05)


    #if no DF samples are provided, scale using the 5th percentile of slope and log length
    if debrisName == "":
        #cc["ccLengthNorm"] = normalizeCustomValues(np.log10(cc.ccLength), minVal = np.log10(cc.ccLength.min()), maxVal = debrisLength)
        #cc["ccMeanSlopeNorm"] = normalizeCustomValues(cc.ccMeanSlope, minVal = cc.ccMeanSlope.min(), maxVal = debrisSlope)
        cc["ccLengthNorm"] = normalizeCustomValues(np.log10(cc.ccLength), minVal = np.log10(cc.ccLength.quantile(0.05)), maxVal = debrisLengthHigh)
        cc["ccMeanSlopeNorm"] = normalizeCustomValues(cc.ccMeanSlope, minVal = cc.ccMeanSlope.quantile(0.05), maxVal = debrisSlopeHigh)

    else: # if DF sample data is present, use 5th percentile of these samples as a minmum Value
        cc["ccLengthNorm"] = normalizeCustomValues(np.log10(cc.ccLength), minVal = debrisLengthLow, maxVal = debrisLengthHigh)
        cc["ccMeanSlopeNorm"] = normalizeCustomValues(cc.ccMeanSlope, minVal = debrisSlopeLow, maxVal = debrisSlopeHigh)

    # plt.figure(figsize=(11.69,8.27))
    # sns.kdeplot(cc.ccMeanSlopeNorm, linewidth = 2)
    # plt.title("Density distribution of summed distances to debris-flow samples")
    # plt.xlabel("Sum of distances")
    # plt.show()

    cc["DFSI"] = cc.ccLengthNorm*l+cc.ccMeanSlopeNorm*s #substract 1 to scale between -1 and 1
    #plot results
    fig = plt.figure(figsize=(11.69,8.27))
    ax = fig.add_subplot()
    s = ax.scatter(cc.ccLength, cc.ccMeanSlope, s=5, c = cc.DFSI, cmap = "coolwarm")
    ax.set_xscale('log')
    ax.set_xlabel("Mean CC Length [m]")
    ax.set_ylabel("Mean CC Slope [m/m]")
    ax.set_ylim(0,1.5)
    ax.set_title("Assigned DFSI for a slope-change threshold of "+ str(dSlopeThr))
    plt.grid()
    fig.colorbar(s)
    plt.show()

    if writeCSV:
        cc.to_csv(path+allCCName+"_ConnectedComponents_withDFSI_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+".csv", index = False)

    return(pd.DataFrame({"ccID": cc.ccID, "DFSI":cc.DFSI, "ccLength": cc.ccLength}))

#####################################################################################################################


def backsorting(fname, path, dfsiValues, pixThr = 7, dSlopeThr = 0.21, bridge = 5, ext = ""):
    #function to transfer assigned DFSI values to stream network

    #load stream csv file
    flow = pd.read_csv(path+fname+"_ConnectedComponents_streams_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+ext+".csv")

    #merge dataframes based on cc ID
    merge = flow.set_index('ccID').join(dfsiValues.set_index('ccID'))

    #drop nodes with the same X and Y coordinates and keep the one that shows the highest DFSI
    df = merge.sort_values('DFSI', ascending=False).drop_duplicates(['X','Y'])
    df = df.reset_index()
    df = df.drop(columns = ["XYDistanceToNextPixel", "3DDistanceToNextPixel"])#

    df.to_csv(path+fname+"_ConnectedComponents_streams_withDFSI_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+".csv", index = False)
    print("The file "+fname+"_ConnectedComponents_streams_withDFSI_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+".csv was written.")

####################################################################################################################

def compareDebrisFlowLengthAndSlope(fname, path = "./", pixThr = 7, bridge = 5, ext = "", thresholdRange = np.round(np.arange(0.05,0.31,0.01),2), writeCSV = False):
    #function to compare weighted average and 5th percentile of CCs from debris-flow sample regions for different thresholds

    #make sure that thresholds are in ascending order
    thresholdRange = np.sort(thresholdRange)

    #empty array to store data
    data = np.zeros([len(thresholdRange),5])
    #load CC datatsets for different thresholds
    for ii, thr in enumerate(thresholdRange):
        deb = pd.read_csv(path+fname+"_ConnectedComponents_"+str(pixThr)+"_"+str(thr)+"_"+str(bridge)+ext+".csv")
        deb = deb.loc[deb['ccLength'] >0].reset_index(drop = True)

        # logtransform and normalize data
        deb["ccLengthNorm"] = normalize(np.log10(deb.ccLength))
        deb["ccMeanSlopeNorm"] = normalize(deb.ccMeanSlope)

        #calculateDensity
        xy = np.vstack([deb.ccLengthNorm,deb.ccMeanSlopeNorm])
        deb["Weight"]=normalize(pd.Series(gaussian_kde(xy)(xy)))

        #get a weighted average DF slope and length

        debrisSlopeHigh = np.average(deb.ccMeanSlope, weights = deb.Weight)
        debrisLengthHigh = np.average(np.log10(deb.ccLength), weights = deb.Weight)

        debrisSlopeLow = deb.ccMeanSlope.quantile(0.05)
        debrisLengthLow = np.log10(deb.ccLength).quantile(0.05)

        #store in array
        data[ii,0] = thr
        data[ii,1] = debrisLengthLow
        data[ii,2] = debrisLengthHigh
        data[ii,3] = debrisSlopeLow
        data[ii,4] = debrisSlopeHigh

    #plot
    plt.figure(figsize=(11.69,8.27))
    plt.plot(data[:,0],data[:,4], label = "Weighted average debris-flow slope")
    plt.plot(data[:,0],data[:,3], label = "5th percentile debris-flow slope")
    plt.legend()
    plt.xlabel("Slope-change threshold [m/m]")
    plt.ylabel("Slope [m/m]")
    plt.title("Min and max slope values derived from debris-flow sample regions\nfor different thresholds")
    plt.show()

    fig = plt.figure(figsize=(11.69,8.27))
    ax = fig.add_subplot()
    #ax.set_yscale("log")
    ax.plot(data[:,0],10**data[:,2], label = "Weighted average debris-flow component length")
    ax.plot(data[:,0],10**data[:,1], label = "5th percentile debris-flow component length")
    plt.legend()
    plt.xlabel("Slope-change threshold [m/m]")
    plt.ylabel("Connected component length [m]")
    plt.title("Min and max component length derived from debris-flow sample regions\nfor different thresholds")
    plt.show()

    if writeCSV:
        df = pd.DataFrame(data)
        df.columns = ["dSlopeThreshold", "debrisLengthLow", "debrisLengthHigh", "debrisSlopeLow", "debrisSlopeHigh"]
        df.to_csv(path+fname+"_minmax_DFslope_and_length.csv", index = False)

#########################################################################################################################
def plotBasin(fname, path = "./", pixThr = 7, dSlopeThr = 0.21, bridge = 5, ext = "", basinIDs = -1, sampleBasins = 1, colorBy = "DFSI"):
    #function to get a quick profile view of the assigned DFSI

    #read stream network with assigned DFSI
    dat = pd.read_csv(path+fname+"_ConnectedComponents_streams_withDFSI_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+".csv")
    dat.dSlopeToPrevSegment = dat.dSlopeToPrevSegment.fillna(0)
    #generate list of pre-defined basins or randomly sample
    if basinIDs != -1 :
        print("Plotting basin(s) "+str(basinIDs), "...")
        if not isinstance(basinIDs, list):
            basinIDs = [basinIDs]

    elif(sampleBasins > 0):
        print("Plotting "+str(sampleBasins)+ " random basin(s)...")
        basinIDs = np.random.choice(dat.BasinID.unique(),sampleBasins,replace=False)

    #define colorbars used for different parameters
    if (colorBy == "DFSI" or colorBy == "dSlopeToPrevSegment"):
        colors = "coolwarm"
    else:
        colors = "viridis"
    for b in basinIDs:
        basin = dat.loc[dat.BasinID == b]
        #in case the basins were cut by a polygon, the minimum flow distance might not be 0
        #so the min Flowdistance is subtracted to set the outlet to 0
        basin.FlowDistance_Catchment = basin.FlowDistance_Catchment - basin.FlowDistance_Catchment.min()
        plt.figure(figsize=(11.69,8.27))
        plt.scatter(basin.FlowDistance_Catchment, basin.Elevation, c = basin[colorBy], s = 1, cmap = colors)
        cbar = plt.colorbar()
        cbar.set_label(colorBy)
        if colorBy == "DFSI":
            plt.clim(-1,1)
        elif colorBy == "dSlopeToPrevSegment":
            minV = basin.dSlopeToPrevSegment.min()
            maxV = basin.dSlopeToPrevSegment.max()
            limV = max(abs(minV), maxV)
            plt.clim(-limV,limV)
        plt.xlabel("Downstream distance [m]")
        plt.title("Basin"+ str(b))
        plt.ylabel("Elevation [m]")
        plt.grid()
        plt.show()
