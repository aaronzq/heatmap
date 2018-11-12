## Heatmap Module
Zhaoqiang Wang -- Heatmap part -- M223A class

#### Function
1. Generate a visualized heatmap on a brain cortex
2. Change hearmap between EEG and ECoG source
3. Switch on electrodes showing or not
4. Change time if we have temporal data

#### Language   
Python 3.6    
#### Required Packages    
- numpy
- scipy
- nibabel
- traits
- mayavi
- pandas    
##### __Note:__ All of the packages can be easily installed by `pip`.
#### Usage
1. Download the heatmapV2.py to your working directory
2. Import __HeatMap__ module from heatmapV2.py in your script
3. Create an instance of __HeatMap__ by passing a _Brain_ file to it
4. Call __HeatMap's__ methods: _readEEG_Pos()_, _readECOG_Pos()_, _readEEG_Score()_, _readECOG_Score()_ to pass in _coordinates_ and _score_
for both EEG and ECOG data source
5. Call _generateHeatmap()_ to generate heatmap
6. Call _show()_ to pop up a UI for visualization
##### example
```
    brain = './standard.cerebrum.mask.nii.gz'
    Test = HeatMap(brain)

    Test.readEEG_Pos('eegcorr.csv')
    Test.readEEG_Score('./Power_Jake/EEG1.csv')

    Test.readECOG_Pos('eegcorr.csv')  #no ECoG coordinates currently, use EEG's for subsitute temporarily
    Test.readECOG_Score('./Power_Jake/ECOG001.csv')

    Test.generateHeatmap()

    Test.show()
```
##### __Note:__ 
- _Brain_ file has to be __nii.gz__ format. 
- _Electrodes coordinates_ and _Scores_ have to be __csv__ format.

#### Contact
For any problems, you can email me or discuss in our final class! :)
