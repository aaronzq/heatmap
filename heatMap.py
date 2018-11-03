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
    return score,len(score)

def locateElectron(space,score,x,y,z):
    N = len(score)
    
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


print('Load MRI cerebrum data')    
brainData = nib.load('./T1.cerebrum.mask.nii.gz')
brainArray = brainData.get_fdata()
brainMask = createMask(brainArray)
brainDMask,volOri,volDilute = diluteMask(brainMask)
print('Load finished')   


row,col,depth = brainArray.shape
spaceSize = [row,col,depth]
bSpace = np.zeros(spaceSize)

electronScore = [[100000,400000,300000,500000,100000],[200000,300000,400000,100000,500000],
[300000,200000,500000,100000,400000],[400000,100000,100000,100000,100000]]

electronX = [55,58,60,70,80]
electronY = [89,78,66,55,43]
electronZ = [65,71,77,83,89]


#######################################################################
eSpace = locateElectron(bSpace,electronScore[0],electronX,electronY,electronZ)
cSpace = ndimage.gaussian_filter(eSpace,5)
elecVmax = cSpace.max()
elecVmin = cSpace.min()

heatmapStore,comRatio = storeHeatmapData(electronScore,electronX,electronY,electronZ,brainDMask)

#dataStore = []
#for h in range(20):
#    eSpace = locateElectron(bSpace,electronScore[h],electronX,electronY,electronZ)
#    cSpace = ndimage.gaussian_filter(eSpace,5)
#    dataStore.append(cSpace*brainDMask)

class MyModel(HasTraits):
    time = Range(0,0,0)
#    button = Button('Electrodes')
    scene = Instance(MlabSceneModel,())

    plot = Instance(PipelineBase)
    
#    @on_trait_change('button1')
#    def update_electrodes(self):
#    
    @on_trait_change('time,scene.activated')
    def update_plot(self):
        heatmap = extractHeatmapData(heatmapStore,self.time,*spaceSize)
        if self.plot is None:
            source = self.scene.mlab.pipeline.scalar_field(heatmap)
#            source = self.scene.mlab.pipeline.scalar_field(cSpace*brainMask)
            self.plot = self.scene.mlab.pipeline.volume(source,vmax=elecVmin + .8*(elecVmax-elecVmin), vmin=elecVmin,figure=self.scene.mayavi_scene)
            source2 = self.scene.mlab.pipeline.scalar_field(brainMask)
            self.scene.mlab.pipeline.iso_surface(source2,vmin=0,vmax=1,opacity=1.0,colormap='gray',figure=self.scene.mayavi_scene)
            self.scene.mlab.view(azimuth=180,elevation=80,distance=350)
        else:
            self.plot.mlab_source.scalars = heatmap

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=700, width=800, show_label=False),
                Group(
                        '_', 'time',
                     ),
                resizable=True,
                )

my_model = MyModel()
my_model.configure_traits()