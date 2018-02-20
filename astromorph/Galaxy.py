import matplotlib.pyplot as mpl
import numpy as np
import astropy.io.fits as pyfits
import scipy.ndimage as snd
from astropy.convolution import convolve

from . import utils
from . import plot_utils as putils
from . import simulation
from . import galfit
import time

class Galaxy(object):

    def __init__(self,imgname=None,imgdata=None,coords=None):

        assert (imgname is not None) or (imgdata is not None),\
                "One of imgname/imgdata must be provided."

        self.original_name = imgname
        self.coords = coords
        if imgdata is not None:
            self.cutout = imgdata
            self.cutout[np.isnan(self.cutout)]=0
        else:
            self.cutout = None

        self.psf = None
        self.psfname = "none"
        self.mask = None

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

    def set_segmentation_mask(self,mask):
        self.mask = mask
        return None

    def create_segmentation_mask(self,pixscale,radius,**kwargs):
        N,M=self.cutout.shape
        xc=N/2
        yc=M/2
        segmap = utils.gen_segmap_tresh(self.cutout,xc,yc,pixscale,radius=radius,**kwargs)
        objmap = utils.select_object_map(xc,yc,segmap,pixscale,radius)
        segmap[segmap>0]=1
        self.set_segmentation_mask(segmap - objmap)
        return self.mask

    def set_psf(self,filename,resize=None):
        self.psfname = filename
        self.psf = pyfits.getdata(filename)
        if self.psf.shape[0]%2==0:
            self.psf = np.pad(self.psf,((0,1),(0,1)),mode="edge")
            self.psf = snd.shift(self.psf,+0.5)

        if resize is not None:
            self.psf = self.psf[75:126,75:126]
        return None

    def randomize_new_pars(self,pars):
        xc,yc,mag,radius,sersic_index,axis_ratio,position_angle = pars

        new_xc = np.random.normal(xc,0.5)
        new_yc = np.random.normal(yc,0.5)
        new_mag = np.random.normal(mag,0.25)
        new_radius = np.random.normal(radius,1.5)
        new_sersic_index = np.random.normal(sersic_index,0.25)
        new_axis_ratio = np.random.normal(axis_ratio,0.05)
        new_position_angle = np.random.normal(position_angle,5)

        if new_radius<0.1:
            new_radius = 0.1
        if new_axis_ratio<0.05:
            new_axis_ratio = 0.05
        if new_axis_ratio>1.00:
            new_axis_ratio = 1.00
        if new_position_angle<-180:
            new_position_angle += 360
        if new_position_angle>180:
            new_position_angle -= 360
        if new_sersic_index < 0.1:
            new_sersic_index = 0.1
        if new_sersic_index > 5:
            new_sersic_index = 5

        return new_xc,new_yc,new_mag,new_radius,new_sersic_index,new_axis_ratio,new_position_angle

    def montecarlo_fit(self,initPars,mag_zeropoint,exposure_time,lensingPars,nRun = 10):

        N,M=self.cutout.shape
        pars = initPars

        if lensingPars is None:
            lensMu = 1
            majAxis_Factor = 1
            shear_factor = 1
        else:
            lensKappa,lensGamma,lensMu = lensingPars
            majAxis_Factor = 1/np.abs(1-lensKappa-lensGamma)
            shear_factor = (1-lensKappa-lensGamma)/(1-lensKappa+lensGamma)

        nPars = 7
        Nchain = np.zeros([nPars,nRun])
        Nchain[:,0] = initPars
        oldChi = 1e10
        newChi = 1
        i=1
        while i < nRun:

            xc,yc,mag,radius,sersic_index,axis_ratio,position_angle = self.randomize_new_pars(Nchain[:,i-1])
            # model = simulation.generate_sersic_model((N,M),\
            #             (xc,yc,mag-2.5*np.log10(lensMu),\
            #             radius*majAxis_Factor,1,\
            #             axis_ratio*shear_factor,position_angle),\
            #             mag_zeropoint,exposure_time)
            #
            # if self.psf is not None:
            #     model = convolve(model,self.psf)

            model = galfit.galaxy_maker(mag_zeropoint,N,M,"sersic",xc,yc,\
            (mag,radius,sersic_index,axis_ratio),position_angle,sky=0,psfname=self.psfname)

            diff_image = (model-self.cutout)*np.abs(1-self.mask)

            newChi = np.sum((diff_image*diff_image).ravel())

            print(i,newChi,newChi/oldChi)
            if newChi/oldChi > np.random.uniform(1,1.05):
                print("Rejected Fit")
                continue
            else:
                Nchain[:,i] = xc,yc,mag,radius,sersic_index,axis_ratio,position_angle
                oldChi = newChi
                i+=1


        return Nchain

    def fit(self,initPars,mag_zeropoint,exposure_time,lensingPars=None,oversampling=3,nRun=10):


        if self.cutout is None:
            raise ValueError("No cutout for this galaxy is defined.")

        if self.mask is None:
            raise ValueError("No mask defined for this galaxy.")

        modelChain = self.montecarlo_fit(initPars,mag_zeropoint,exposure_time,lensingPars,nRun=nRun)

        return modelChain
