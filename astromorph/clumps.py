import numpy as np
from . import utils
import matplotlib.pyplot as mpl

import scipy.ndimage as snd
from skimage.morphology import watershed


class ClumpFinder:

    THRESHOLD = 3.0
    LEVELS = np.arange(0.1,1.0,0.05)

    def __init__(self,image,pixelscale,sigma=0.25):
        self.data = snd.gaussian_filter(image,sigma=sigma)
        self.pixelscale = pixelscale

        self.set_threshold()


    def set_threshold(self,threshold=None):
        if threshold is None:
            self.threshold = ClumpFinder.THRESHOLD
        else:
            self.threshold = threshold

    def segmap_levels(self, segmap, levels = None):

        if levels is None:
            levels = ClumpFinder.LEVELS
        assert (levels == np.sort(levels)).all()

        totalFlux= np.sum(self.data[segmap ==1])
        maxFlux= np.amax(self.data[segmap ==1])

        fmaps = []
        ft = 0
        fluxSampling = np.linspace(0,maxFlux,1000)[::-1]

        i=0
        FullMap = np.zeros_like(segmap,dtype=np.float32)
        fraction = levels[i]
        for f in fluxSampling:

            area_flux = np.sum(self.data[(segmap==1)*(self.data>=f)])
            if area_flux>=fraction*totalFlux:
                level_map = np.zeros(self.data.shape)
                level_map[(segmap==1)*(self.data>=f)]=1.0
                i+=1
                if i == len(levels):
                    break
                fraction = levels[i]
                fmaps.append(level_map)
                FullMap += level_map
        fmaps.append(segmap)
        return fmaps,FullMap

    def identify_clumps(self,segmap):
        labels,nclumps = snd.label(segmap)
        positions_max = []
        for i in range(nclumps):

            single_clump_map=np.zeros_like(segmap)
            single_clump_map[labels==(i+1)]=1
            posmax = np.where(self.data*single_clump_map==np.amax(self.data*single_clump_map))



            if np.size(posmax)>2:
                x  = int(np.mean(posmax[0]))
                y  = int(np.mean(posmax[1]))
                positions_max.append((x,y))
            else:
                x = posmax[0][0]
                y = posmax[1][0]
                positions_max.append((x,y))
        return positions_max

    def clump_segmentation(self, threshold = None):
        N,M = self.data.shape
        smap = utils.gen_segmap_tresh(self.data,N/2,M/2,self.pixelscale,\
                                      thresh=self.threshold,\
                                      all_detection=True)

        nGals = np.amax(smap)
        FullMap = np.zeros_like(smap,dtype=np.float32)
        FullPositions = []
        for i in range(nGals):
            singleGalaxyMap = np.zeros_like(smap)
            singleGalaxyMap[smap==i+1] = 1

            AllMaps,SummedMap = self.segmap_levels(singleGalaxyMap)
            for cmap in AllMaps:
                pos = self.identify_clumps(cmap)
                FullPositions += pos

            FullMap += SummedMap

        peaksImage = np.zeros_like(smap)
        ClumpPeaks = set(FullPositions)
        for cp in ClumpPeaks:
            x,y = cp
            peaksImage[x,y] = 1

        markers = snd.label(peaksImage)[0]
        labels = watershed(-FullMap, markers, mask=smap)
        return FullMap, peaksImage, labels
