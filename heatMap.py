# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:20:49 2018

@author: zqwang
"""
import numpy as np
from scipy import ndimage
import nibabel as nib

from traits.api import HasTraits, Range, Instance, on_trait_change, Button
from traitsui.api import View, Item, Group

from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel

import pandas as pd

import ecogcorr as ecCorr


def readScore(filename):
    table = pd.read_csv(filename)
    scoreStr = table.columns.tolist()
    score = []
    for s in scoreStr:
        try:
            f = float(s)
        except ValueError:
            loc = [i for i,v in enumerate(s) if v=='.']
            f = s[0:loc[1]]
            f = float(f)
        
        score.append(f)
    #Normalize the score
    scoreNorm = [sc/max(score)*50000 for sc in score]
    return scoreNorm,len(scoreNorm)

def readPos(filename):
    table = pd.read_csv(filename)
    xtable = table.loc[0].tolist()
    ytable = table.loc[1].tolist()
    ztable = table.loc[2].tolist()
    x = [int(i) for i in xtable]
    y = [int(i) for i in ytable]
    z = [int(i) for i in ztable]
    return x,y,z

def locateElectron(space,score,x,y,z):
    N = len(x)
    
    for i in range(0,N):
        space[x[i],y[i],z[i]] = score[i]

    return space

def createMask(brain):
    row,col,depth = brain.shape
    mask = np.zeros([row,col,depth])
    for i in range(row):
        for j in range(col):
            for k in range(depth):
                if brain[i,j,k] > 0:
                    mask[i,j,k]=1
                else:
                    mask[i,j,k]=0                    
    return mask

def calcVolume(mask):
    vol = 0
    fMask = mask.flatten()
    for voxel in fMask:
        if voxel > 0:
            vol+=1
    return vol
    
def diluteMask(mask):
    volOri = calcVolume(mask)
    dMask = ndimage.gaussian_filter(mask,1)
    dMask = createMask(dMask)
    volDilute = calcVolume(dMask)
    return dMask,volOri,volDilute

def convolutionHeatmap(bSpace,electronScore,electronX,electronY,electronZ,mask):
    eSpace = locateElectron(bSpace,electronScore,electronX,electronY,electronZ)
    cSpace = ndimage.gaussian_filter(eSpace,5)
    return cSpace*mask


def compressArray(imgArray):
    ls1d = []
    arr1d = imgArray.flatten()
    for ind in range(imgArray.size):
        if arr1d[ind] > 0:
            ls1d.append(ind)
            ls1d.append(arr1d[ind])
    return ls1d,imgArray.size,len(ls1d)

def decompressArray(ls1d,row,col,depth):
    arr1d = np.zeros(row*col*depth)
    for ind in range(len(ls1d)//2):
        arr1d[ls1d[2*ind]] = ls1d[2*ind+1]
    imgArray = arr1d.reshape([row,col,depth])
    return imgArray
    

def storeHeatmapData(electronScore,electronX,electronY,electronZ,brainDMask):
    lsCollection = []
    ori=0
    com=0
    timeLen = len(electronScore)
    print('Heatmap generation begins')
    for t in range(timeLen):
        eSpace = locateElectron(bSpace,electronScore[t],electronX,electronY,electronZ)
        cSpace = ndimage.gaussian_filter(eSpace,5)
        heatm = cSpace*brainDMask
        ls1d,oriSize,comSize = compressArray(heatm)
        ori += oriSize
        com += comSize
        lsCollection.append(ls1d)
        if t == timeLen//4:
            print('Heatmap generation 25%')
        if t == timeLen//2:
            print('Heatmap generation 50%')
        if t == (3*timeLen)//4:
            print('Heatmap generaton 75%')
    comRatio = com/ori
    print('Heatmap generation ends')
    return lsCollection,comRatio
    
    
def extractHeatmapData(lsCol,index,row,col,depth):
    ls1d = lsCol[index]
    return decompressArray(ls1d,row,col,depth)

## Load MRI brain data
print('Load MRI cerebrum data')    
#brainData = nib.load('./T1.cerebrum.mask.nii.gz')
brainData = nib.load('./standard.cerebrum.mask.nii.gz')
brainArray = brainData.get_fdata()
brainMask = createMask(brainArray)
brainDMask,volOri,volDilute = diluteMask(brainMask)
print('Load finished')   
#######################################################################

## Create heatmap space and grab electrodes information
print('Create Heatmap Space and grab electrodes')
row,col,depth = brainArray.shape
spaceSize = [row,col,depth]
bSpace = np.zeros(spaceSize)

#electronScore = [[100000,400000,300000,500000,100000],[200000,300000,400000,100000,500000],
#[300000,200000,500000,100000,400000],[400000,100000,100000,100000,100000]]
#
#electronX = [55,58,60,70,80]
#electronY = [89,78,66,55,43]
#electronZ = [65,71,77,83,89]

timestep = 1 # no temporal stamp

##filename = './Power_Jake/ECOG001.csv'
filename = './PAC_Mah/Max_PAC_ECOG1.csv'
##filename = './PAC_Mah/Mean_PAC_ECOG1.csv'
electronScore,channelNum = readScore(filename)
electronX = ecCorr.x
electronY = ecCorr.y
electronZ = ecCorr.z


#filename = './Power_Jake/EEG1.csv'
#filename = './PAC_Mah/Max_PAC_EEG1.csv'
#filename = './PAC_Mah/Mean_PAC_EEG1.csv'
#electronScore,channelNum = readScore(filename)
#electronX,electronY,electronZ = readPos('eegcorr.csv')


# obtain the initial Vmax and Vmin
heatmap = convolutionHeatmap(bSpace,electronScore,electronX,electronY,electronZ,brainDMask)
elecVmax = heatmap.max()
elecVmin = heatmap.min()

print('Electrodes set')
#######################################################################


#heatmapStore,comRatio = storeHeatmapData(electronScore,electronX,electronY,electronZ,brainDMask)


class MyModel(HasTraits):
    time = Range(0,timestep-1,0)
    button = Button('Electrodes')
    
    scene = Instance(MlabSceneModel,())
    plot = Instance(PipelineBase)
    
    onoff = 1
    
    @on_trait_change('button')
    def update_electrodes(self):
        if self.onoff==1:
            self.elec.stop()
        else:
            self.elec = self.scene.mlab.points3d(electronX, electronY, electronZ, scale_factor=5, resolution=20, scale_mode='none', color = (1,0,1), opacity = 1.0, figure=self.scene.mayavi_scene)
            self.scene.mlab.view(azimuth=180,elevation=80,distance=350)
        
        self.onoff += 1
        self.onoff = self.onoff%2
        
    @on_trait_change('time,scene.activated')
    def update_plot(self):
#        heatmap = extractHeatmapData(heatmapStore,self.time,*spaceSize)
        heatmap = convolutionHeatmap(bSpace,electronScore,electronX,electronY,electronZ,brainDMask)
        if self.plot is None:
            source = self.scene.mlab.pipeline.scalar_field(heatmap)
            self.plot = self.scene.mlab.pipeline.volume(source,vmax=elecVmin + .8*(elecVmax-elecVmin), vmin=elecVmin,figure=self.scene.mayavi_scene)
            source2 = self.scene.mlab.pipeline.scalar_field(brainMask)
            self.scene.mlab.pipeline.iso_surface(source2,vmin=0,vmax=1,opacity=1.0,colormap='gray',figure=self.scene.mayavi_scene)


 
            self.elec = self.scene.mlab.points3d(electronX, electronY, electronZ, scale_factor=5, resolution=20, scale_mode='none', color = (1,0,1), opacity = 1.0, figure=self.scene.mayavi_scene)
                
            self.scene.mlab.view(azimuth=180,elevation=80,distance=350)
        else:
            self.plot.mlab_source.scalars = heatmap

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=700, width=800, show_label=False),
                Group(
                        '_', 'time', 'button'
                     ),
                resizable=True,
                )

my_model = MyModel()
my_model.configure_traits()