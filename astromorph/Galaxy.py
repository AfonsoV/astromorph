import matplotlib.pyplot as mpl
import numpy as np
import astropy.io.fits as pyfits
import scipy.ndimage as snd
from astropy.convolution import convolve

from . import utils
from . import plot_utils as putils
from . import simulation
import time

class Galaxy(object):

    def __init__(self,imgname=None,imgdata=None,coords=None):

        assert (imgname is not None) or (imgdata is not None),\
                "One of imgname/imgdata must be provided."

        self.original_name = imgname
        self.coords = coords
        if imgdata is not None:
            self.cutout = imgdata
        else:
            self.cutout = None

        self.psf = None

    def draw_cutout(self,size,pixelscale,eixo=None,**kwargs):
        if self.cutout is None:
            data = utils.get_cutout(self.original_name,self.coords,size,pixelscale)
        else:
            data = self.cutout
        if data is not None:

            if eixo is None:
                fig,ax = mpl.subplots()
            else:
                ax = eixo
            putils.show_image(ax,data,scale="linear",vmin=-0.001,vmax=0.01,\
            extent=(-size/2,size/2,-size/2,size/2),**kwargs)
            putils.draw_cross(ax,0,0,gap=1.0,size=1.5,color="white",lw=3)
        else:
            print("Galaxy cutout outside image region")

    def get_image_box_coordinates(self):
        assert (self.bounding_box  is not None),"A bounding box must be assigned to the galaxy"
        header = pyfits.getheader(self.original_name)
        x0,x1,y0,y1=self.bounding_box
        def fx(x):
            return (x-header["CRPIX1"])*header["CD1_1"] + header["CRVAL1"]
        def fy(y):
            return (y-header["CRPIX2"])*header["CD2_2"] + header["CRVAL2"]
        return (fx(x0),fx(x1),fy(y0),fy(y1))

    def set_bounding_box(self,size,pixelscale):
        self.bounding_box = utils.get_bounding_box(self.original_name,self.coords,size,pixelscale)
        self.cutout = utils.get_cutout(self.original_name,self.coords,size,pixelscale)
        return None

    def set_psf(self,filename,resize=None):
        self.psf = pyfits.getdata(filename)
        if self.psf.shape[0]%2==0:
            self.psf = np.pad(self.psf,((0,1),(0,1)),mode="edge")
            self.psf = snd.shift(self.psf,+0.5)

        if resize is not None:
            self.psf = self.psf[50:151,50:151]
        return None

    def fit(self,initPars,mag_zeropoint,exposure_time,lensingPars=None,oversampling=3):
        xc,yc,mag,radius,sersic_index,axis_ratio,position_angle = initPars

        if lensingPars is None:
            lensMu = 1
            majAxis_Factor = 1
            shear_factor = 1
        else:
            lensKappa,lensGamma,lensMu = lensingPars
            majAxis_Factor = 1/np.abs(1-lensKappa-lensGamma)
            shear_factor = (1-lensKappa-lensGamma)/(1-lensKappa+lensGamma)

        if self.cutout is None:
            raise ValueError("No cutout for this galaxy is defined")
        else:
            N,M=self.cutout.shape

            t1 = time.time()
            model = simulation.generate_sersic_model((N,M),\
                    (xc,yc,mag-2.5*np.log10(lensMu),\
                    radius*majAxis_Factor,sersic_index,\
                    axis_ratio*shear_factor,position_angle),\
                    mag_zeropoint,exposure_time)
            t2 = time.time()
            # # model = snd.rotate(model,position_angle,reshape=False)
            # t3 = time.time()
            # # model = utils.rebin2d(model,(N,M))
            t4 = time.time()
            if self.psf is not None:
                model = convolve(model, self.psf)
            t5 = time.time()

            print("First Model : %.4f ms"%(1000*(t2-t1)))
            # print("Rotated Model : %.4f ms"%(1000*(t3-t2)))
            # print("Rebin Model : %.4f ms"%(1000*(t4-t3)))
            print("Convolved Model : %.4f ms"%(1000*(t5-t4)))
            print("Total Time : %.4f ms"%(1000*(t5-t1)))
            return model
