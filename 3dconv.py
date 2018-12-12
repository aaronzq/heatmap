# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:20:49 2018

@author: zqwang
"""
import numpy as np
from scipy import ndimage
from mayavi import mlab
import nibabel as nib

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
    row,col,depth = mask.shape
    volOri = calcVolume(mask)
    dMask = ndimage.gaussian_filter(mask,1)
    dMask = createMask(dMask)
    volDilute = calcVolume(dMask)
    return dMask,volOri,volDilute


    
brainData = nib.load('./T1.cerebrum.mask.nii.gz')
brainArray = brainData.get_fdata()
brainMask = createMask(brainArray)
brainDMask,volOri,volDilute = diluteMask(brainMask)


row,col,depth = brainArray.shape
spaceSize = [row,col,depth]
bSpace = np.zeros(spaceSize)

electronScore = [[100000,400000,300000,500000,100000],[200000,300000,400000,100000,500000],
[300000,200000,500000,100000,400000],[400000,100000,100000,100000,100000]]
electronX = [55,58,60,70,80]
electronY = [89,78,66,55,43]
electronZ = [65,71,77,83,89]

eSpace = locateElectron(bSpace,electronScore[0],electronX,electronY,electronZ)
cSpace = ndimage.gaussian_filter(eSpace,5)
elecVmax = cSpace.max()
elecVmin = cSpace.min()

#x, y, z = np.mgrid[-5:5:50j, -5:5:50j, -5:0:25j]

#X = x.reshape([x.size])
#Y = y.reshape([y.size])
#Z = z.reshape([z.size])
#r = cSpace*mask
#R = r.reshape([r.size])


mlab.figure(figure='Brain',size=(800,700))

#l = mlab.points3d(X, Y, Z, R, line_width=50.0,colormap='Spectral') 
#mlab.contour3d(r)
#source = mlab.pipeline.scalar_field(x,y,z,r)
#vol = mlab.pipeline.scalar_scatter(r)

source2 = mlab.pipeline.scalar_field(brainMask)
vol2 = mlab.pipeline.iso_surface(source2,vmin=0,vmax=1,opacity=1.0,colormap='gray')

source = mlab.pipeline.scalar_field(cSpace*brainDMask)
#vol = mlab.pipeline.volume(source,vmax = 0.8*elecVmax, vmin = 0.2*elecVmin)
vol = mlab.pipeline.volume(source,vmax=(elecVmin + .8*(elecVmax-elecVmin)), vmin=elecVmin)

#vol = mlab.pipeline.volume(source,vmax=1000, vmin=100)

# Store the information
view = mlab.view(azimuth=180,elevation=80,distance=350)
#roll = mlab.roll()


## Reposition the camera
#mlab.view(*view)
#mlab.roll(roll)

@mlab.animate(delay=100)
def anim():
    for i in range(100):
        eSpace = locateElectron(bSpace,electronScore[i%4],electronX,electronY,electronZ)
        cSpace = ndimage.gaussian_filter(eSpace,5)
        vol.mlab_source.scalars = cSpace*brainDMask
#        vol.mlab_source.scalars = cSpace*brainMask*(1+i/100)
        yield

anim()
mlab.show()





