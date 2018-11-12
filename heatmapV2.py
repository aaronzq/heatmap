# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage
import nibabel as nib

from traits.api import HasTraits, Range, Instance, on_trait_change, Button
from traitsui.api import View, Item, Group, VGroup, HGroup

from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel

import pandas as pd

import ecogcorr as ecCorr


NO_READIN = 0
YES_READIN = 1
    
SOURCE_IS_EEG = 0
SOURCE_IS_ECOG = 1
ELECTRODE_OFF = 0
ELECTRODE_ON = 1

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

def readScore(filename):
    # For CSV format
    # This readScore function now only read one row of values
    # thus ONLY for one time point, revise it after we have multi-timestep data

    table = pd.read_csv(filename)
    scoreStr = table.columns.tolist()
    score = []
    for s in scoreStr:
        try:
            f = float(s)
        except ValueError:
            # this is to handle some abnormal values in csv file like '123.41.21'
            point = [i for i,v in enumerate(s) if v=='.']
            f = s[0:point[1]]
            f = float(f)
        
        score.append(f)
    
    scoreNorm = [sc/max(score)*50000 for sc in score] #Normalize the score

    return scoreNorm,len(scoreNorm)

def readPos(filename):
    # For CSV format
    # This readPos function read 3 rows of positions: X, Y and Z

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

def convolutionHeatmap(bSpace,electronScore,electronX,electronY,electronZ,mask):
    eSpace = locateElectron(bSpace,electronScore,electronX,electronY,electronZ)
    cSpace = ndimage.gaussian_filter(eSpace,5)
    return cSpace*mask

class HeatMap:
    def __init__(self, mriBrainMask):
        brain = nib.load(mriBrainMask)
        self.brainMask = createMask(brain.get_fdata())
        self.brainDMask, volOri, volDilute = diluteMask(self.brainMask)
        self.row, self.col, self.depth = self.brainMask.shape
        print('MRI brain mask loaded successfully')

        self.mapSpace = np.zeros([self.row, self.col, self.depth])
        self.EEG_POS_FLAG = NO_READIN
        self.EEG_SCORE_FLAG = NO_READIN
        self.ECOG_POS_FLAG = NO_READIN
        self.ECOG_SCORE_FLAG = NO_READIN


    def readEEG_Pos(self,filename):
        self.EEG_PosX, self.EEG_PosY, self.EEG_PosZ = readPos(filename)
        self.EEG_POS_FLAG = YES_READIN
        print('Load EEG electrodes positions successfully')

    def readECOG_Pos(self,filename):
        self.ECOG_PosX, self.ECOG_PosY, self.ECOG_PosZ = readPos(filename)
        self.ECOG_POS_FLAG = YES_READIN
        print('Load ECoG electrodes positions successfully')

    def readEEG_Score(self,filename):
        self.EEG_Score, self.EEG_ChannelNum = readScore(filename)
        self.EEG_SCORE_FLAG = YES_READIN
        print('Load EEG scores successfully')

    def readECOG_Score(self,filename):
        self.ECOG_Score, self.ECOG_ChannelNum = readScore(filename)
        self.ECOG_SCORE_FLAG = YES_READIN
        print('Load ECoG scores successfully')

    def generateHeatmap(self):
        if not (self.EEG_POS_FLAG and self.EEG_SCORE_FLAG and self.ECOG_POS_FLAG and self.ECOG_SCORE_FLAG):
            print('Requiring data missing. Flag:  ', self.EEG_POS_FLAG, self.EEG_SCORE_FLAG, self.ECOG_POS_FLAG, self.ECOG_SCORE_FLAG)
        else:
            self.eegHeatmap = convolutionHeatmap(self.mapSpace,self.EEG_Score,self.EEG_PosX,self.EEG_PosY,self.EEG_PosZ,self.brainDMask)
            self.eegHeatmapVmax = self.eegHeatmap.max()
            self.eegHeatmapVmin = self.eegHeatmap.min()
           
            self.ecogHeatmap = convolutionHeatmap(self.mapSpace,self.ECOG_Score,self.ECOG_PosX,self.ECOG_PosY,self.ECOG_PosZ,self.brainDMask)
            self.ecogHeatmapVmax = self.ecogHeatmap.max()
            self.ecogHeatmapVmin = self.ecogHeatmap.min()

            self.HEATMAP_FLAG = YES_READIN

            print('Heatmap generation finished')

    def show(self):
        if not self.HEATMAP_FLAG:
            print('Requiring heatmap not generated')
        else:
            my_model = MyModel(self.brainMask,self.eegHeatmap,self.ecogHeatmap,self.EEG_PosX,self.EEG_PosY,self.EEG_PosZ,self.ECOG_PosX,self.ECOG_PosY,self.ECOG_PosZ,self.eegHeatmapVmin,self.eegHeatmapVmax,self.ecogHeatmapVmin,self.ecogHeatmapVmax)
            my_model.configure_traits()
        

class MyModel(HasTraits):

    def __init__(self,brain,eegheatmap,ecogheatmap,eeg_posx,eeg_posy,eeg_posz,ecog_posx,ecog_posy,ecog_posz,eegmin,eegmax,ecogmin,ecogmax):
        HasTraits.__init__(self)
        self.brain = brain
        self.eegheatmap = eegheatmap
        self.ecogheatmap = ecogheatmap

        self.eeg_posx = eeg_posx
        self.eeg_posy = eeg_posy
        self.eeg_posz = eeg_posz
        self.ecog_posx = ecog_posx
        self.ecog_posy = ecog_posy
        self.ecog_posz = ecog_posz

        self.eegmin = eegmin
        self.eegmax = eegmax
        self.ecogmin = ecogmin
        self.ecogmax = ecogmax

        self.source = SOURCE_IS_EEG
        self.electrode = ELECTRODE_OFF


        
    scene = Instance(MlabSceneModel,())
    plot = Instance(PipelineBase)
    
    signalType = Button('EEG/ECoG')
    electrodesType = Button('On/Off Electrodes')
    time = Range(0,0,0)

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                    height=700, width=800, show_label=False),
                VGroup(
                        '_', Item(name='time'),'_', 
                        HGroup(
                            Item(name = 'signalType', label= 'Choose data sourse',show_label=True),
                            Item(name = 'electrodesType', label = 'Electrodes',show_label=True)
                            ),
                        show_border=True
                        
                        ),
                resizable=True
                )

    @on_trait_change('time,scene.activated')
    def update_time(self):
        if self.source == SOURCE_IS_EEG:
            heatmap = self.eegheatmap
            elecVmin = self.eegmin
            elecVmax = self.eegmax
        if self.source == SOURCE_IS_ECOG:
            heatmap = self.ecogheatmap
            elecVmin = self.ecogmin
            elecVmax = self.ecogmax

        if self.plot is None:
            background = self.scene.mlab.pipeline.scalar_field(self.brain)
            self.scene.mlab.pipeline.iso_surface(background,vmin=0,vmax=1,opacity=1.0,colormap='gray',figure=self.scene.mayavi_scene)

            heat = self.scene.mlab.pipeline.scalar_field(heatmap)
            self.plot = self.scene.mlab.pipeline.volume(heat,vmax=elecVmin + .8*(elecVmax-elecVmin), vmin=elecVmin,figure=self.scene.mayavi_scene)
                
            self.scene.mlab.view(azimuth=180,elevation=80,distance=350)
        else:
            self.plot.mlab_source.scalars = heatmap

    @on_trait_change('electrodesType')
    def update_points(self):
        if self.source == SOURCE_IS_EEG:
            electronX = self.eeg_posx
            electronY = self.eeg_posy
            electronZ = self.eeg_posz
        if self.source == SOURCE_IS_ECOG:
            electronX = self.ecog_posx
            electronY = self.ecog_posy
            electronZ = self.ecog_posz                  

        if self.electrode == ELECTRODE_OFF:
            self.electrode = ELECTRODE_ON
            self.elec = self.scene.mlab.points3d(electronX, electronY, electronZ, scale_factor=5, resolution=20, scale_mode='none', color = (1,0,1), opacity = 1.0, figure=self.scene.mayavi_scene)
            self.scene.mlab.view(azimuth=180,elevation=80,distance=350)
        else:
            self.electrode = ELECTRODE_OFF
            self.elec.stop()           


    @on_trait_change('signalType')
    def update_source(self):
        if self.source == SOURCE_IS_EEG:
            self.source = SOURCE_IS_ECOG
            heatmap = self.ecogheatmap
            self.plot.mlab_source.scalars = heatmap
        else:
            self.source = SOURCE_IS_EEG
            heatmap = self.eegheatmap
            self.plot.mlab_source.scalars = heatmap



    


    
if __name__ == '__main__':
    brain = './standard.cerebrum.mask.nii.gz'
    Test = HeatMap(brain)

    Test.readEEG_Pos('eegcorr.csv')
    Test.readEEG_Score('./Power_Jake/EEG1.csv')

    Test.readECOG_Pos('eegcorr.csv')  #no ecog csv for now
    Test.readECOG_Score('./Power_Jake/ECOG001.csv')

    Test.generateHeatmap()

    Test.show()
