# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 18:21:45 2018

@author: zqwang
"""

import numpy as np
from mayavi import mlab
#
#
#x, y, z = np.mgrid[-5:5:64j, -5:5:64j, -5:5:64j]
#
#scalars = x * x * 0.5 + y * y * 0.5 + z * 0.5
#
#obj = mlab.contour3d(x,y,z,scalars, contours=4, transparent=False)
#
#mlab.show()




"""Generates a pretty set of lines."""
n_mer, n_long = 6, 11
dphi = np.pi / 1000.0
phi = np.arange(0.0, 2 * np.pi + 0.5 * dphi, dphi)
mu = phi * n_mer
x = np.cos(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
y = np.sin(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
z = np.sin(n_long * mu / n_mer) * 0.5


x = [100,70,140]
y = [100,150,90]
z = [50,50,50]
hehe = np.arange(0,3)


l = mlab.plot3d(x, y, z, np.sin(mu), tube_radius=0.025, colormap='Spectral')
l = mlab.plot3d(x, y, z, hehe, tube_radius=0.025, colormap='Spectral') 
l = mlab.contour3d(x,y,z,hehe,colormap='Spectral') 
 
mlab.show()