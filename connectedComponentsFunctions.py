#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:59:37 2021

@author: ariane
"""


import pandas as pd
from tqdm import tqdm
import numpy as np
import geopandas as gpd
import os, sys
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import linregress, gaussian_kde
from kneed import KneeLocator
import concurrent.futures
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

pd.set_option('mode.chained_assignment', None)

#####################################################################################################################

def getLSDTTFlowlines(fname,path, lsdttPath = "~/LSDTopoTools/LSDTopoTools2/LSDTopoTools/LSDTopoTools2/bin/", minContributingPixels = 1000, maxBasinSize = 11111111,findCompleteBasins = "false", testBoundaries = "false", m_over_n = 0.45):
    #this function converts the input geotiff to ENVI format, creates a driver file and runs lsdtt-chi mapping to extract channels
   
    #change geotiff to bil format for lsdtt
    
    print("Converting GeoTIFF ton ENVI format...")
    os.system("gdal_translate -of ENVI "+fname+".tif "+fname+".bil")
    
    #create driver file for lsdtt
    try: 
        os.remove(fname+".driver")
    except OSError:
        pass
    stdoutOrigin=sys.stdout
    sys.stdout = open (fname+".driver", "a")
    
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
    
    #run chi-mapping from lsdtt
    print("Executing lsdtt-chi-mapping...")
    os.system(lsdttPath+"lsdtt-chi-mapping "+fname+".driver")

#####################################################################################################################

def normalize(x):
    #just a min max scaler - always useful
    return(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))

#####################################################################################################################

def toUTM(df, epsg):
    #turn lat and lon into UTM coordinates because LSD only provides geographic CRS
    gdf = df.set_geometry(gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    projected_df = gdf.to_crs("EPSG:"+str(epsg))
    df["x"] = projected_df.geometry.x
    df["y"] = projected_df.geometry.y
    return df

#####################################################################################################################

def mergeLSDTToutput(fname, path, resolution, epsg):
    #function combines _MChiSegmented file with the reciever nodes from the _CN file to know which stream pixel comes next
    print("Merging CSV files from LSDTopoTools...")
    #load chi segmented channels from LSDTT
    lsdttTable = pd.read_csv(path+fname+"_MChiSegmented.csv")
    #Convert long lat to UTM
    lsdttTable = toUTM(lsdttTable, epsg)
    # get coorndinates
    coords = [(x,y) for x, y in zip(lsdttTable.x, lsdttTable.y)]
    
    #the information from the _MChiSegmented.csv file with the _CN.csv file needs to be combined because the reciever nodes (stored in the _CN file)
    #need to be retrieved in order to properly walk downstream.
    #this is done by converting the channel data to a shapefile and rasterize it so that the reciever nodes can be extracted
    #spatial join does not work, because points are slightly offset
    
    #CSV to shapefile because gdal_rasterize didnt like the csv format
    ogr2ogr = "ogr2ogr -s_srs EPSG:4326 -t_srs EPSG:32719 -oo X_POSSIBLE_nameS=lon* -oo Y_POSSIBLE_nameS=lat*  -f \"ESRI Shapefile\" "+path+fname+"_CN.shp "+path+fname+"_CN.csv"  
    os.system(ogr2ogr)
    #now rasterize the information about the reciever nodes
    rasterize = "gdal_rasterize -a \"receiver_N\" -tr "+str(resolution)+" "+ str(resolution)+" -a_nodata 0 -co COMPRESS=DEFLATE -co ZLEVEL=9 "+path+fname+"_CN.shp "+path+fname+"_RNI.tif"
    os.system(rasterize)
    
    #open raster and extract reciever node information to points
    rni = rasterio.open(path+fname+"_RNI.tif") #load reciever node raster
    lsdttTable['RNI'] = [x[0] for x in rni.sample(coords)] #sample reciever nodes at points
    
    return lsdttTable

#####################################################################################################################
 
def processBasin(basin, lsdttTable, heads, pixThr, dSlopeThr, bridge):    
    #function calculates channel slope and connected components for individual catchments
    
    #empty dataframe for storing output
    df = pd.DataFrame(columns = ["X","Y","StreamID","BasinID", "Elevation", "DrainageArea", "XYDistanceToNextPixel", "DownstreamDistance", "3DDistanceToNextPixel",  "Slope", "R2", "ksn", "ccID", "segmentLocation", "dSlopeToPrevSegment"])
       
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
            
        #empty array for storing data
        data = np.zeros([60000,12]) #if there are really really long channels (> 60000 pixels), this array needs to be expanded
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
    
                #now store some data in the empty array
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
                    
                    # plt.figure()
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
                        if(abs(mean-slopeInvestigated)<=dSlopeThr): #if Slope of current pixel is smaller than the mean of the current connected segment
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
                                data[np.where(data[:,9]==ccID),11]= previousSlps[-1]-mean
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
            #calculate slope at end of stream                                                                                                                        
            if(currentNode == nextNode and j>pixThr): # currentNode = nextNode (end of stream) + stream is longer than threshold pixels
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
                    

                    # plt.figure()
                    # plt.scatter(np.cumsum(dsDist[mask]),elev[mask], color = "gray")
                    # plt.hlines(data[k,2], 0 ,40)
                    # plt.show()
                    
                #########################################
                #finish calculating CCs
                for k in range(j-pixThr-bridge, j, 1):

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

                        if(abs(mean-slopeInvestigated)<=dSlopeThr): #if Slope of current pixel is smaller than the mean of the current connected segment

                            data[k, 9]= ccID #assign the same ccID
                            data[k, 10]=segmentLocationcount                  
                            mList.append(slopeInvestigated) #add Slope to Slope list and compute mean
                            mean = np.mean(mList)
                            
                            #if the stream ends naturally, the CC needs to be terminated as well (only thing missing is calculating the dPrevSLp)
                            if(k == j):
                                try:
                                    data[np.where(data[:,9]==ccID),11]= previousSlps[-1]-mean
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
                                data[np.where(data[:,9]==ccID),11]= previousSlps[-1]-mean
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
        
        #finish assigning the correct segment location: 
        #if the terminal pixel of a stream is reached 
        #and the segment location is still = 1
        #the entire stream is still one CC and needs to be assigned the value 4
        if(segmentLocationcount == 1 and currentNode == nextNode):
            data[:,10]=4
        # else if there are no middle segments and the steam does come to a natural end, set the segemten Locations of the last CC to 3    
        elif(segmentLocationcount == 2 and currentNode == nextNode):
            data[np.where(data[:,9]==ccID),10]=3
            
        data = data[~np.isnan(data[:, 9])] #remove everything that wasnt assigned a CC 

        #convert array to pandas df
        df = pd.concat([df, pd.DataFrame({"X": data[:,0], "Y": data[:,1],"StreamID":ii, "BasinID":basin, "Elevation": data[:,2], 
                        "DrainageArea":data[:,4], "XYDistanceToNextPixel": data[:,3], "DownstreamDistance":np.cumsum(data[:,3]), "3DDistanceToNextPixel":data[:,5], "ksn":data[:,6],"Slope":data[:,7], "R2":data[:,8],
                        "ccID":data[:,9], "segmentLocation":data[:,10], "dSlopeToPrevSegment":data[:,11]})])
        #assign truly unique CCIDs 
    df["ccID"]=df["BasinID"].map(str)+"_"+df['StreamID'].map(str)+"_"+df['ccID'].map(str) 
    return df             

#####################################################################################################################

def processBasinWithoutCCs(basin, lsdttTable, heads, pixThr):    
    #function calculates channel slope for given basin
    #main purpose is to find a suitable pixel threshold
    df = pd.DataFrame(columns = ["X","Y","StreamID","BasinID", "Elevation", "DrainageArea", "XYDistanceToNextPixel", "DownstreamDistance", "3DDistanceToNextPixel","ksn" ,"Slope", "R2", "SlopeChange"])
       
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

        
            ##########################################
            #calculate slope at end of stream                                                                                                                        
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
        df = pd.concat([df, pd.DataFrame({"X": data[:,0], "Y": data[:,1],"StreamID":ii, "BasinID":basin, "Elevation": data[:,2], 
                        "DrainageArea":data[:,4], "XYDistanceToNextPixel": data[:,3], "DownstreamDistance":np.cumsum(data[:,3]), 
                        "3DDistanceToNextPixel":data[:,5], "ksn":data[:,6],"Slope":data[:,7], "R2":data[:,8], "SlopeChange":np.append(abs(np.diff(data[:,7])),np.nan)})])

    return df     

#####################################################################################################################        

def runCCAnalysis(fname, path, lsdttTable, pixThr = 10, dSlopeThr = 0.23, bridge = 5):
    #function to execute the processBasin function by passing different tasks to different cores
    
    #get all nodes that are NOT in reciever nodes as channel heads
    heads = lsdttTable[~lsdttTable.node.isin(lsdttTable.RNI)].reset_index(drop=True)
    
    #now get all basins
    #print("I found "+str(heads["basin_key"].nunique())+" catchments.")

    #loop over all basins in csv file and give each task to a different core
    #pass tasks
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(processBasin, basin = basin , lsdttTable = lsdttTable, heads = heads, pixThr = pixThr, dSlopeThr = dSlopeThr, bridge = bridge) for basin in tqdm(heads.basin_key.unique(), desc = "Processing "+str(heads["basin_key"].nunique())+" catchments.")]
    
    #combine results
    out = pd.DataFrame(columns = ["X","Y","StreamID","BasinID", "Elevation", "DrainageArea", "XYDistanceToNextPixel", "DownstreamDistance", "3DDistanceToNextPixel",  "Slope", "R2", "ksn", "ccID", "segmentLocation", "dSlopeToPrevSegment"])
           
    for task in concurrent.futures.as_completed(results):
        out = pd.concat([out,task.result()])      
    
    #write streams
    out.to_csv(path+fname+"_ConnectedComponents_streams_"+str(pixThr)+"_"+str(dSlopeThr)+"_"+str(bridge)+".csv", index = False)
    print("I have written all streams with assigned CC ID to "+fname+"_ConnectedComponents_streams_"+str(pixThr)+"_"+str(dSlopeThr)+"_"+str(bridge)+".csv")
    
    #now get statistics per CC
    cc = out.groupby('ccID').agg({'3DDistanceToNextPixel': 'sum', 'Slope':  ['mean', 'std'], 'DrainageArea': ['min', 'max'], 
                                   'segmentLocation': 'max', "dSlopeToPrevSegment":["max"]})

    #rename columns
    cc = cc.reset_index()
    cc.columns = ["_".join(x) for x in cc.columns.ravel()]
    cc.columns = ['ccID', 'ccLength', 'ccMeanSlope', 'ccStdSlope', 'minDrainageArea', 'maxDrainageArea','segmentLocation', 'slopeChangeToPrevCC']
    
    #write aggregated parameters
    cc.to_csv(path+fname+"_ConnectedComponents_"+str(pixThr)+"_"+str(dSlopeThr)+"_"+str(bridge)+".csv", index = False)
    
    print("I have written the aggregated parameters for all CCs to "+fname+"_ConnectedComponents_"+str(pixThr)+"_"+str(dSlopeThr)+"_"+str(bridge)+".csv")

#####################################################################################################################    

def findDSlopeThreshold(lsdttTable, pixThr = 10, bridge = 5, thresholdRange = np.round(np.arange(0.05,0.31,0.01),2)):
    #similar to runCCAnalysis Script. Process is repeated for different thresholds and output CCs are compared 
    #lsdttTable of debris flow sample regions should be provided
    
    #make sure that thresholds are in ascending order
    thresholdRange = np.sort(thresholdRange)
    #get channelheads
    heads = lsdttTable[~lsdttTable.node.isin(lsdttTable.RNI)].reset_index(drop=True)
    #empty array for storing data
    data = np.zeros([len(thresholdRange),3]) 
    
    for ii, thr in enumerate(tqdm(thresholdRange, desc = "Finding connected components using different slope-change thresholds")):
        #print("Testing a threshold of "+str(thr)+"...")
        
        #pass tasks
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(processBasin, basin = basin , lsdttTable = lsdttTable, heads = heads, pixThr = pixThr, dSlopeThr = thr, bridge = bridge) for basin in heads.basin_key.unique()]
        #combine results
        out = pd.DataFrame(columns = ["X","Y","StreamID","BasinID", "Elevation", "DrainageArea", "XYDistanceToNextPixel", "DownstreamDistance", "3DDistanceToNextPixel",  "Slope", "R2", "ksn", "ccID", "segmentLocation", "dSlopeToPrevSegment"])
               
        for task in concurrent.futures.as_completed(results):
            out = pd.concat([out,task.result()]) 
        
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
        print("Out of the provided slope-change thresholds "+str(thresholdRange)+", I recommend the value "+str(optThr))
    
    #plot
    plt.figure()
    plt.plot(data[:,0], data[:,2]/data[:,1])
    plt.scatter(data[:,0], data[:,2]/data[:,1])
    plt.hlines(0.5, xmin = thresholdRange[0], xmax = thresholdRange[-1], linestyle='dashed')
    plt.vlines(optThr, ymin = 0, ymax = nr1CConly/nrStreamsTotal, linestyle='dashed', color = "red")
    plt.xlabel("Threshold")
    plt.ylabel("Fraction of single component streams")
    plt.title("Optimal slope-change threshold to constrain CCs")
    plt.show()    
    
    return(optThr)

#####################################################################################################################

def findPixelThreshold(lsdttTable, thresholdRange = np.arange(1,26), sampleBasinID = -1, sampleStreams = -1):
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
    
    #get a random subset of x channelheads to speed up processing (if desired)
    if sampleStreams > 0:
        #print("Subsetting the input and sampling "+str(sampleStreams)+" random streams.")
        heads = heads.sample(n = sampleStreams).reset_index(drop = True)
    
    #loop over thresholdRange
    for ii, thr in enumerate(tqdm(thresholdRange, desc = "Calculating channel-slope using different regression lengths")):
        #print("Testing a threshold of ±"+str(thr)+ " pixel(s).")
        #pass tasks
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(processBasinWithoutCCs, basin = basin , lsdttTable = lsdttTable, heads = heads, pixThr = int(thr)) for basin in heads.basin_key.unique()]
        #combine results
        out = pd.DataFrame(columns = ["X","Y","StreamID","BasinID", "Elevation", "DrainageArea", "XYDistanceToNextPixel", "DownstreamDistance", "3DDistanceToNextPixel", "ksn", "Slope", "R2", "SlopeChange"])
               
        for task in concurrent.futures.as_completed(results):
            out = pd.concat([out,task.result()]) 
               
        #calculate median, and 25th + 7th percentile
        data[ii, 0] = thr
        data[ii, 1] = np.nanmedian(out.SlopeChange)
        data[ii, 2] = np.nanpercentile(out.SlopeChange, 25)
        data[ii, 3] = np.nanpercentile(out.SlopeChange, 75)
        
    #locate kneepoint
    kn = KneeLocator(
    np.array(data[:,0]),
    np.array(data[:,1]),
    curve='convex',
    direction='decreasing',
    interp_method='polynomial')

    #plot
    plt.figure()
    plt.plot(data[:,0], data[:,1])
    plt.errorbar(data[:,0], data[:,1], yerr = [ data[:,2], data[:,3]], color = "blue")
    plt.scatter(data[:,0], data[:,1])
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', color = "red")
    plt.xlabel("Pixels up- and downstream of the current node")
    plt.ylabel("Pixel-to-pixel slope change")
    plt.title("Optimal regression length for calculating channel slope")
    plt.show()
    
    print("Out of the given pixel thresholds "+str(thresholdRange)+ ", I recommend taking the value "+str(int(kn.knee))+".")
    
    return(int(kn.knee))

#####################################################################################################################

def assignDFSI(path, allCCName, debrisName, pixThr = 0.23, dSlopeThr = 0.23, bridge = 5):
    #load debris samples
    print("I am going to use "+debrisName+"_ConnectedComponents_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+".csv as debris-flow samples.")
    deb = pd.read_csv(path+debrisName+"_ConnectedComponents_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+".csv")
    
    #load all CCs
    print("I am going to assign debris-flow similarity values to all CCs in"+ allCCName+"_ConnectedComponents_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+".csv")
    cc =  pd.read_csv(path+allCCName+"_ConnectedComponents_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+".csv")
    
    # logtransform and normalize data
    deb["ccLengthNorm"] = normalize(np.log(deb.ccLength))
    deb["ccMeanSlopeNorm"] = normalize(deb.ccMeanSlope)
    cc["ccLengthNorm"] = normalize(np.log(cc.ccLength))
    cc["ccMeanSlopeNorm"] = normalize(cc.ccMeanSlope)
    
    #calculateDensity
    xy = np.vstack([deb.ccLengthNorm,deb.ccMeanSlopeNorm])
    deb["Weight"]=normalize(gaussian_kde(xy)(xy))

    
    #calculate sum of distances for all points
    def calcSummedDistances(length, slope, deb):
        sumofDist = (length-deb.ccLengthNorm)**2+(slope-deb.ccMeanSlopeNorm)**2
        weightedSOD = deb.Weight*sumofDist    
        return sum(weightedSOD)
    
    cc["sumOfDistances"] = cc.apply(lambda row: calcSummedDistances(row['ccLengthNorm'], row['ccMeanSlopeNorm'], deb), axis=1)
    
    
    #plot DF CCs and assigned weight
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(cc.ccLength, cc.ccMeanSlope, s=1, color = "gray")
    ax.scatter(deb.ccLength, deb.ccMeanSlope, s = deb.Weight*5, c = "r")
    ax.set_xscale('log')
    ax.set_xlabel("Mean CC Length [m]")
    ax.set_ylabel("Mean CC Slope [m/m]")
    plt.show()
    
    #density plot of distance distribution. 
    #should ideally have two strong peaks    
    
    plt.figure()
    sns.kdeplot(cc.sumOfDistances, linewidth = 2)
    plt.title("Density distribution of summed distances to debris-flow samples")
    plt.xlabel("Sum of distances")
    plt.show()
    
    #CCs are split into two groups using kmeans clustering based on the sum of Distances to debris flow samples
    
    print("Clustering ...")
    scaled = StandardScaler().fit_transform(pd.DataFrame(cc.sumOfDistances))
    km = KMeans(
        n_clusters=2, init='random',
        n_init=10, max_iter=100, 
        tol=1e-04, random_state=0
    )
    cc["clusterKM"] = km.fit_predict(scaled)
    #find out which of these two clusters has the lower mean distance to DF samples so that the DFSI will only be assigned to that one
    clusterStats = cc.groupby("clusterKM").sumOfDistances.agg("mean")
    DFcluster = clusterStats.idxmin()
    #assign DFSI (=normalized sum of Distances for DF cluster)
    cc["DFSI"]= cc.sumOfDistances
    #set to np.nan first so  values for non-DF regions will not be considered during normalization
    cc.loc[cc.clusterKM != DFcluster, "DFSI"] = np.nan
    #normalize and invert DFSI values
    cc.DFSI = (normalize(cc.DFSI)-1)*-1
    #set nan values to -1 again
    cc.loc[cc.clusterKM != DFcluster, "DFSI"] = -1
    
    #plot results 
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(cc.ccLength, cc.ccMeanSlope, s=5, c = cc.DFSI)
    ax.set_xscale('log')
    ax.set_xlabel("Mean CC Length [m]")
    ax.set_ylabel("Mean CC Slope [m/m]")
    ax.set_title("Assigned DFSI")
    plt.show()
    
    return(pd.DataFrame({"ccID": cc.ccID, "DFSI":cc.DFSI}))

#####################################################################################################################


def backsorting(fname, path, dfsiValues, pixThr = 10, dSlopeThr = 0.23, bridge = 5):
    
    #load stream csv file
    flow = pd.read_csv(path+fname+"_ConnectedComponents_streams_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+".csv")
    
    #merge dataframes based on cc ID
    merge = flow.set_index('ccID').join(dfsiValues.set_index('ccID'))
    
    #drop nodes with the same X and Y coordinates and keep the one that shows the highest DFSI
    df = merge.sort_values('DFSI', ascending=False).drop_duplicates(['X','Y'])
    df = df.reset_index()
    
    #save to file
    df.to_csv(path+fname+"_ConnectedComponents_streams_withDFSI_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+".csv", index = False)
    print("I have written "+fname+"_ConnectedComponents_streams_withDFSI_"+str(pixThr)+"_"+str(np.round(dSlopeThr,2))+"_"+str(bridge)+".csv")

