import matplotlib.pyplot as mpl
import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import scipy.ndimage as snd
from astropy.convolution import convolve_fft
from scipy.signal import fftconvolve
from . import utils
from . import plot_utils as putils
from . import simulation
from . import galfit
from .models import LensedSersicModel,LensPars,PsfModel,LensedSersicPSFModel
from astropy.modeling import models, fitting
import warnings
fitterLSQ = fitting.LevMarLSQFitter()
from scipy.optimize import leastsq,OptimizeWarning

import time
import emcee
import warnings

N_MAX = 10.0


class SersicParameters:

    def __init__(self,sky):
        self.sigma_xc = 1.5
        self.sigma_yc = 1.5
        self.sigma_mag = 10.5
        self.sigma_radius = 5.0
        self.sigma_sersic_index = 1.0
        self.sigma_axis_ratio = 0.075
        self.sigma_position_angle = 5.0
        self.sigma_sky = np.abs(0.25*sky) + 1e-3

        self.Sigmas = [self.sigma_xc,self.sigma_yc,self.sigma_mag,\
                      self.sigma_radius,self.sigma_sersic_index,\
                      self.sigma_axis_ratio,self.sigma_position_angle]



    def get_sigma(self,npars):
        p = []
        for i in range(npars):
            p +=self.Sigmas
        return p + [self.sigma_sky]

class SersicPSFParameters:

    def __init__(self,sky):
        self.sigma_xc = 2.5
        self.sigma_yc = 2.5
        self.sigma_mag = 15.0
        self.sigma_radius = 5.0
        self.sigma_sersic_index = 1.0
        self.sigma_axis_ratio = 0.075
        self.sigma_position_angle = 5.0
        self.sigma_xPSF = 2.5
        self.sigma_yPSF= 2.5
        self.sigma_IPSF= 15.0
        self.sigma_sky = np.abs(0.25*sky) + 1e-3

        self.Sigmas = [self.sigma_xc,self.sigma_yc,self.sigma_mag,\
                      self.sigma_radius,self.sigma_sersic_index,\
                      self.sigma_axis_ratio,self.sigma_position_angle,\
                      self.sigma_xPSF,self.sigma_yPSF,self.sigma_IPSF]


    def get_sigma(self,npars):
        p = []
        for i in range(npars):
            p +=self.Sigmas
        return p + [self.sigma_sky]

################################################################################
################################################################################
# MCMC
################################################################################
################################################################################


def lnlikelihood_LensedSersicPSF(modelPars,image,sigma,mag_zeropoint,exposure_time,psf,lensingPars):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """


    xc=modelPars[0]
    yc=modelPars[1]
    Ieff=modelPars[2]
    radius=modelPars[2]
    n=modelPars[4]
    axis_ratio=modelPars[5]
    position_angle=modelPars[6]
    xPSF=modelPars[6]
    yPSF=modelPars[7]
    iPSF=modelPars[8]
    skyValue=modelPars[9]

    sersicModel = LensedSersicPSFModel(lensPars=lensingPars,\
                                     magZP=mag_zeropoint,\
                                     psf=psf,\
                                     x_0=xc,\
                                     y_0=yc,\
                                     I_eff=Ieff,\
                                     r_eff=radius,\
                                     n=n,\
                                     axratio=axis_ratio,\
                                     theta=position_angle,\
                                     xPSF=xPSF,\
                                     yPSF=yPSF,\
                                     I_psf=iPSF,\
                                     OverSampling=10)
    skyModel = models.Const2D(amplitude = skyValue)

    model = sersicModel+skyModel

    N,M = image.shape
    x,y = np.mgrid[:N,:M]
    modelEval = model(x,y) ## + dx_sky*(x-N/2) +dy_sky*(y-M/2)

    if psf is not None:
        modelFinal = fftconvolve(modelEval,psf,mode="same")
        # model = convolve_fft(model,self.psf)
    else:
        warnings.warn("No PSF provided. Convlution not performed.")


    return -0.5*np.ma.sum( (image-modelFinal)*(image-modelFinal)/(sigma*sigma) + np.ma.log(2*np.pi*sigma*sigma) )

def prior_LensedSersicPSF(pars,shape):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    N,M = shape
    if  (N/4<pars[0]<3*N/4) and\
        (M/4<pars[1]<3*M/4) and\
        (0<pars[2]<=1e4) and\
        (1e-3<=pars[3]<=20) and\
        (0.1<=pars[4]<=N_MAX) and\
        (0.1<=pars[5]<=1) and\
        (-90<=pars[6]<=90) and\
        (N/4<pars[7]<3*N/4) and\
        (M/4<pars[8]<3*M/4) and\
        (0<pars[9]<1e4):
        return 0
    else:
        return -np.inf

def lnprobability_LensedSersicPSF(pars,image,sigma,mag_zeropoint,exposure_time,psf=None,lensingPars=None):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    pr = prior_LensedSersicPSF(pars,image.shape)
    if not np.isfinite(pr):
        return -np.inf
    LL = pr + lnlikelihood_LensedSersicPSF(pars,image,sigma,mag_zeropoint,exposure_time,psf,lensingPars)
    if np.isfinite(LL):
        return LL
    else:
        return -np.inf


################################################################################
# Multi-Model
################################################################################


def lnlikelihood_MP(modelPars,image,sigma,mag_zeropoint,exposure_time,nModels,psf,lensingPars):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """

    model = np.zeros_like(image)
    for i in range(nModels):

        xc=modelPars[0+7*i]
        yc=modelPars[1+7*i]
        mag=modelPars[2+7*i]
        radius=modelPars[2+7*i]
        n=modelPars[4+7*i]
        axis_ratio=modelPars[5+7*i]
        position_angle=modelPars[6+7*i]

        if lensingPars is None:
            SingleModel = simulation.generate_sersic_model(image.shape,\
                        (xc,yc,mag,radius,n,axis_ratio,position_angle),\
                        mag_zeropoint,exposure_time)
            warnings.warn("No lensing parameters provided. Distortion not performed.")
        else:
            SingleModel = simulation.generate_lensed_sersic_model(image.shape,\
                        (xc,yc,mag,radius,n,axis_ratio,position_angle),\
                        lensingPars,mag_zeropoint,exposure_time)
        model += SingleModel

    if psf is not None:
        model = fftconvolve(model,psf,mode="same")
        # model = convolve_fft(model,self.psf)
    else:
        warnings.warn("No PSF provided. Convlution not performed.")

    N,M = model.shape
    x,y = np.meshgrid(range(N),range(M))
    model += modelPars[-1] ## + dx_sky*(x-N/2) +dy_sky*(y-M/2)

    return -0.5*np.ma.sum( (image-model)*(image-model)/(sigma*sigma) + np.ma.log(2*np.pi*sigma*sigma) )

def prior_MP(pars,shape,nModels):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    N,M = shape

    totalProb = 0
    for i in range(nModels):
        if  (N/4<pars[0+7*i]<3*N/4) and\
            (M/4<pars[1+7*i]<3*M/4) and\
            (0<pars[2+7*i]<=35) and\
            (1e-3<=pars[3+7*i]<=20) and\
            (0.1<=pars[4+7*i]<=N_MAX) and\
            (0.1<=pars[5+7*i]<=1) and\
            (-90<=pars[6+7*i]<=90):
            totalProb += 1

    if totalProb == nModels:
        return 0
    else:
        return -np.inf

def lnprobability_MP(pars,image,sigma,mag_zeropoint,exposure_time,nModels=1,psf=None,lensingPars=None):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    pr = prior_MP(pars,image.shape,nModels)
    if not np.isfinite(pr):
        return -np.inf
    LL = pr + lnlikelihood_MP(pars,image,sigma,mag_zeropoint,exposure_time,nModels,psf,lensingPars)
    if np.isfinite(LL):
        return LL
    else:
        return -np.inf


################################################################################
# Single-Model (fixed sersic index)
################################################################################
def lnlikelihood(pars,image,sigma,mag_zeropoint,exposure_time,psf,lensingPars):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    # xc,yc,mag,radius,axis_ratio,position_angle,sky,dx_sky,dy_sky = pars
    xc,yc,mag,radius,axis_ratio,position_angle,sky = pars
    if lensingPars is None:
        model = simulation.generate_sersic_model(image.shape,\
                    (xc,yc,mag,radius,1.0,axis_ratio,position_angle),\
                    mag_zeropoint,exposure_time)
        warnings.warn("No lensing parameters provided. Distortion not performed.")
    else:
        model = simulation.generate_lensed_sersic_model(image.shape,\
                    (xc,yc,mag,radius,1.0,axis_ratio,position_angle),\
                    lensingPars,mag_zeropoint,exposure_time)

    if psf is not None:
        model = fftconvolve(model,psf,mode="same")
        # model = convolve_fft(model,self.psf)
    else:
        warnings.warn("No PSF provided. Convolution not performed.")

    N,M = model.shape
    x,y = np.meshgrid(range(N),range(M))
    model += sky ## + dx_sky*(x-N/2) +dy_sky*(y-M/2)

    return -0.5*np.ma.sum( (image-model)*(image-model)/(sigma*sigma) + np.ma.log(2*np.pi*sigma*sigma) )

def prior(pars,shape):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    N,M = shape
    if  (N/4<pars[0]<3*N/4) and\
        (M/4<pars[1]<3*M/4) and\
        (0<pars[2]<=35) and\
        (1e-3<=pars[3]<=20) and\
        (0.1<=pars[4]<=1) and\
        (-90<=pars[5]<=90):
        return 0.0
    return -np.inf

def lnprobability(pars,image,sigma,mag_zeropoint,exposure_time,psf=None,lensingPars=None):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    pr = prior(pars,image.shape)
    if not np.isfinite(pr):
        return -np.inf
    LL = pr + lnlikelihood(pars,image,sigma,mag_zeropoint,exposure_time,psf,lensingPars)
    if np.isfinite(LL):
        return LL
    else:
        return -np.inf



class Galaxy(object):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """


    def __init__(self,imgname=None,imgdata=None,coords=None):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
        assert (imgname is not None) or (imgdata is not None),\
                "One of imgname/imgdata must be provided."

        self.original_name = imgname
        self.coords = coords
        if imgdata is not None:
            self.cutout = imgdata
            self.cutout[np.isnan(self.cutout)]=0
        else:
            self.cutout = None
            self.imgdata = pyfits.getdata(imgname)
            self.imgheader = pyfits.getheader(imgname)

        self.psf = None
        self.psfname = "none"
        self.mask = None
        self.sigmaImage = None
        self.objectMask = None

    def __repr__(self):
        stringShow =  f"Galaxy Data: {self.original_name}\n"
        stringShow += f"Galaxy coordinates: {self.coords}\n"
        if self.cutout is not None:
            stringShow += f"Cutout size: {self.cutout.shape}\n"
        else:
            stringShow += f"Cutout size: {self.cutout}\n"
        return stringShow

    def __str__(self):
        if self.cutout is None:
            data = f"with no associated data"
        else:
            data = f"with associated {self.cutout.shape} shape data"
        return f"Galaxy class object {data}"

    def set_coords(self,coords):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
        self.coords=coords
        return None

    def draw_cutout(self,size,pixelscale,eixo=None,**kwargs):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
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

    def get_bounding_box_coordinates(self):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
        assert (self.bounding_box  is not None),"A bounding box must be assigned to the galaxy"
        header = self.imgheader
        x0,x1,y0,y1=self.bounding_box
        wcs=pywcs.WCS(header)
        pixCoords=wcs.wcs_pix2world([[x0,y0],[x1,y1]],1)
        pixLow = pixCoords[0]
        pixHig = pixCoords[1]
        return (pixLow[0],pixHig[0],pixLow[1],pixHig[1])

    def get_image_coordinates(self):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
        header = self.imgheader
        x0,x1,y0,y1=0,self.imgdata.shape[1],0,self.imgdata.shape[0]
        wcs=pywcs.WCS(header)
        pixCoords=wcs.wcs_pix2world([[x0,y0],[x1,y1]],1)
        pixLow = pixCoords[0]
        pixHig = pixCoords[1]
        return (pixLow[0],pixHig[0],pixLow[1],pixHig[1])

    def set_bounding_box(self,size,pixelscale):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
        self.bounding_box = utils.get_bounding_box(self.imgheader,self.coords,size,pixelscale)
        x0,x1,y0,y1=self.bounding_box
        self.cutout = self.imgdata[y0:y1,x0:x1]
        return None

    def compute_local_sky(self,k=3):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
        if self.cutout is None:
            raise ValueError("the data cutout has to be defined")
        self.sky_value, self.sky_rms = utils.sky_value(self.cutout,k=k)
        return self.sky_value, self.sky_rms


    def set_segmentation_mask(self,mask):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
        self.mask = mask
        return None

    def create_segmentation_mask(self,pixscale,radius,**kwargs):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
        N,M=self.cutout.shape
        xc=N/2
        yc=M/2
        # segmap = utils.gen_segmap_tresh(self.cutout,xc,yc,pixscale,radius=radius,**kwargs)
        segmap = utils.gen_segmap_watershed(self.cutout,**kwargs)
        objmap = utils.select_object_map(xc,yc,segmap,pixscale,radius)
        segmap[segmap>0]=1
        self.set_segmentation_mask(segmap - objmap)
        return self.mask

    def set_object_mask(self,mask,radius=15):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
        if mask[mask>0].size==0:
            print("Warning, object mask is empty. Setting default mask.")
            self.objectMask = self.default_mask(radius)
        else:
            self.objectMask = mask
        return None

    def create_object_mask(self,pixscale,radius,**kwargs):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
        N,M=self.cutout.shape
        xc=N/2
        yc=M/2
        segmap = utils.gen_segmap_watershed(self.cutout,**kwargs)
        objmap = utils.select_object_map(xc,yc,segmap,pixscale,radius)
        return objmap

    def set_psf(self,filename=None,psfdata=None,resize=None):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
        if filename is not None:
            self.psfname = filename
            self.psf = pyfits.getdata(filename)
        elif psfdata is not None:
            self.psf = psfdata
        else:
            raise ValueError("Either filename or psfdata must be given.")

        if self.psf.shape[0]%2==0:
            self.psf = np.pad(self.psf,((0,1),(0,1)),mode="edge")
            self.psf = snd.shift(self.psf,+0.5)

        # if resize is not None:
        #     self.psf = self.psf[50:-50,50:-50]
        return None

    def set_sigma(self,sigma):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
        self.sigmaImage = sigma
        return None

    def default_mask(self,radius):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
        if self.cutout is None:
            raise ValueError("cutuout attribute must be defined.")

        N,M = self.cutout.shape
        xc,yc = N/2,M/2
        R = utils.compute_ellipse_distmat(self.cutout,xc,yc)
        mask = np.zeros_like(self.cutout)
        mask[R<radius]=1
        return mask

    def estimate_parameters(self,mag_zeropoint,exposure_time,pixscale,radius,rPsf=2.5,**kwargs):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
        if self.mask is not None:
            detection_image = self.cutout*(1-self.mask)
        else:
            detection_image = self.cutout

        if self.objectMask is None:
            self.objectMask = self.create_object_mask(pixscale,radius,**kwargs)
            if self.objectMask[self.objectMask>0].size ==0:
                self.objectMask = self.default_mask(radius/pixscale)

        xc,yc=utils.barycenter(self.cutout,self.objectMask)

        if np.nansum(self.cutout*self.objectMask)<0:
            mag = 28
        else:
            mag = -2.5*np.log10(np.nansum(self.cutout*self.objectMask)/exposure_time)+mag_zeropoint
        # r100 = max(0.5,np.sqrt(self.cutout[object_mask==1].size/np.pi - 2.5*2.5))
        r50 = utils.get_half_light_radius(self.cutout,self.objectMask)
        if r50<rPsf:
            r50 = 0.5
        else:
            r50 = np.sqrt(r50*r50-rPsf*rPsf)
        axisRatio = utils.get_axis_ratio(self.cutout,self.objectMask)
        positionAngle = utils.get_position_angle(self.cutout,self.objectMask) - 90

        while positionAngle>90:
            positionAngle -= 180

        while positionAngle<-90:
            positionAngle += 180

        return (xc,yc,mag,r50,axisRatio,positionAngle)

    def randomize_new_pars(self,pars):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
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



    def emcee_fit(self,initPars,mag_zeropoint,exposure_time,lensingPars=None,nchain=20,nsamples=10000,nexclude=None,plot=False,threads=1,ntemps=None):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
        if nexclude is None:
            nexclude = nsamples//2
        if nexclude > nsamples:
            raise ValueError("nexclude cannot be greater than nsamples.")

        masked_image = np.ma.masked_array(self.cutout,mask=self.mask)
        masked_sigma = np.ma.masked_array(self.sigmaImage,mask=self.mask)

        ndim, nwalkers = len(initPars), nchain
        sigmaPars = [1.5,1.5,0.5,5.0,0.075,5.0,np.abs(0.25*initPars[6])]
        if len(sigmaPars) != len(initPars):
            for i in range(len(sigmaPars),len(initPars)):
                sigmaPars.append(1e-3)

        pos = np.array([np.random.normal(initPars,sigmaPars) for i in range(nwalkers)])

        ## Standardize random starting points to be within accepted ranges
        pos[:,0][pos[:,0]<0]=1
        pos[:,0][pos[:,0]>self.cutout.shape[0]]=self.cutout.shape[0]-1
        pos[:,1][pos[:,1]<0]=1
        pos[:,1][pos[:,1]>self.cutout.shape[1]]=self.cutout.shape[1]-1
        pos[:,3][pos[:,3]<0.5]=0.5
        pos[:,3][pos[:,3]>50]=50
        pos[:,4][pos[:,4]<0.1]=0.1
        pos[:,4][pos[:,4]>1]=1.0
        pos[:,5][pos[:,5]>90]-=180
        pos[:,5][pos[:,5]<-90]+=180

        tstart = time.time()

        if ntemps is None:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobability,\
                        args=(masked_image, masked_sigma,mag_zeropoint,\
                              exposure_time, self.psf,lensingPars),\
                        threads=threads,a=1.5)
            sampler.run_mcmc(pos, nsamples)
        else:
            sampler = emcee.PTSampler(ntemps,nwalkers, ndim, lnprobability,prior,\
                    loglargs=(masked_image, masked_sigma,mag_zeropoint,\
                              exposure_time, self.psf,lensingPars),\
                    logpargs=(masked_image.shape,),threads=threads)
            sampler.run_mcmc([pos]*ntemps, nsamples)


        print('\telapsed %.8f seconds'%(time.time()-tstart))
        print("\tMean acceptance fraction: %.3f"%(np.mean(sampler.acceptance_fraction)))
        if plot is True:
            plot_results(sampler,[r"$x_c$",r"$y_c$",r'$mag$',\
                                  r'$r_e\ [\mathrm{arcsec}]$',r"$(b/a)$",\
                                  r"$\theta_\mathrm{PA}$",r"sky"],\
                                  ntemps=ntemps)

        if  ntemps is not None:
            samples = sampler.chain[:,:, nexclude:, :].reshape((-1, ndim))
        else:
            samples = sampler.chain[:, nexclude:, :].reshape((-1, ndim))

        return samples.T


    def emcee_fit_MP(self,initPars,mag_zeropoint,exposure_time,lensingPars=None,nchain=100,nsamples=5000,nexclude=None,plot=False,threads=1,ntemps=None):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """

        if nexclude is None:
            nexclude = nsamples//2
        if nexclude > nsamples:
            raise ValueError("nexclude cannot be greater than nsamples.")

        nModels = (len(initPars)-1)//7
        modelPars = SersicParameters(initPars[-1])

        masked_image = np.ma.masked_array(self.cutout,mask=self.mask)
        masked_sigma = np.ma.masked_array(self.sigmaImage,mask=self.mask)

        ndim, nwalkers = len(initPars), nchain

        sigmaPars = modelPars.get_sigma(nModels)
        if len(sigmaPars) != len(initPars):
            for i in range(len(sigmaPars),len(initPars)):
                sigmaPars.append(1e-3)

        pos = np.array([np.random.normal(initPars,sigmaPars) for i in range(nwalkers)])

        ## Standardize random starting points to be within accepted ranges
        for i in range(nModels):
            pos[:,0+7*i][pos[:,0+7*i]<0]=1
            pos[:,0+7*i][pos[:,0+7*i]>self.cutout.shape[0]]=self.cutout.shape[0]-1
            pos[:,1+7*i][pos[:,1+7*i]<0]=1
            pos[:,1+7*i][pos[:,1+7*i]>self.cutout.shape[1]]=self.cutout.shape[1]-1
            pos[:,3+7*i][pos[:,3+7*i]<0.5]=0.5
            pos[:,3+7*i][pos[:,3+7*i]>50]=50
            pos[:,4+7*i][pos[:,4+7*i]<0.1]=0.1
            pos[:,4+7*i][pos[:,4+7*i]>N_MAX]=N_MAX-0.1
            pos[:,5+7*i][pos[:,5+7*i]<0.1]=0.1
            pos[:,5+7*i][pos[:,5+7*i]>1]=1.0
            pos[:,6+7*i][pos[:,6+7*i]>90]-=180
            pos[:,6+7*i][pos[:,6+7*i]<-90]+=180

        tstart = time.time()

        if ntemps is None:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobability_MP,\
                        args=(masked_image, masked_sigma,mag_zeropoint,\
                              exposure_time,nModels,self.psf,lensingPars),\
                        threads=threads,a=1.5)
            sampler.run_mcmc(pos, nsamples)
        else:
            sampler = emcee.PTSampler(ntemps,nwalkers, ndim, lnprobability_MP,prior_MP,\
                    loglargs=(masked_image, masked_sigma,mag_zeropoint,\
                              exposure_time,nModels,self.psf,lensingPars),\
                    logpargs=(masked_image.shape,nModels),threads=threads)
            sampler.run_mcmc([pos]*ntemps, nsamples)


        print('\telapsed %.8f seconds'%(time.time()-tstart))
        print("\tMean acceptance fraction: %.3f"%(np.mean(sampler.acceptance_fraction)))
        if plot is True:
            labels = [r"$x_c$",r"$y_c$",r'$mag$',\
                      r'$r_e$',r"$n$",r"$q$",\
                      r"$\theta$"]*nModels + ["sky"]
            print(labels)
            plot_results(sampler,labels,ntemps=ntemps)

        if  ntemps is not None:
            samples = sampler.chain[:,:, nexclude:, :].reshape((-1, ndim))
        else:
            samples = sampler.chain[:, nexclude:, :].reshape((-1, ndim))

        return samples.T


    def emcee_fit_LensedSersicPSF(self,initPars,mag_zeropoint,exposure_time,lensingPars=None,nchain=100,nsamples=5000,nexclude=None,plot=False,threads=1,ntemps=None):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """

        if nexclude is None:
            nexclude = nsamples//2
        if nexclude > nsamples:
            raise ValueError("nexclude cannot be greater than nsamples.")

        modelPars = SersicPSFParameters(initPars[-1])

        masked_image = np.ma.masked_array(self.cutout,mask=self.mask)
        masked_sigma = np.ma.masked_array(self.sigmaImage,mask=self.mask)

        ndim, nwalkers = len(initPars), nchain

        sigmaPars = modelPars.get_sigma(1)

        pos = np.array([np.random.normal(initPars,sigmaPars) for i in range(nwalkers)])

        pos[:,0][pos[:,0]<0]=1
        pos[:,0][pos[:,0]>self.cutout.shape[0]]=self.cutout.shape[0]-1
        pos[:,1][pos[:,1]<0]=1
        pos[:,1][pos[:,1]>self.cutout.shape[1]]=self.cutout.shape[1]-1
        pos[:,2][pos[:,2]<0]=0.5
        pos[:,3][pos[:,3]<0.5]=0.5
        pos[:,3][pos[:,3]>50]=50
        pos[:,4][pos[:,4]<0.1]=0.1
        pos[:,4][pos[:,4]>N_MAX]=N_MAX-0.1
        pos[:,5][pos[:,5]<0.1]=0.1
        pos[:,5][pos[:,5]>1]=1.0
        pos[:,6][pos[:,6]>90]-=180
        pos[:,6][pos[:,6]<-90]+=180
        pos[:,7][pos[:,7]<0]=1
        pos[:,7][pos[:,7]>self.cutout.shape[0]]=self.cutout.shape[0]-1
        pos[:,8][pos[:,8]<0]=1
        pos[:,8][pos[:,8]>self.cutout.shape[1]]=self.cutout.shape[1]-1
        pos[:,9][pos[:,9]<0]=0.5

        tstart = time.time()

        if ntemps is None:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobability_LensedSersicPSF,\
                        args=(masked_image, masked_sigma,mag_zeropoint,\
                              exposure_time,self.psf,lensingPars),\
                        threads=threads,a=1.5)
            sampler.run_mcmc(pos, nsamples)
        else:
            sampler = emcee.PTSampler(ntemps,nwalkers, ndim, lnprobability_LensedSersicPSF,prior_LensedSersicPSF,\
                    loglargs=(masked_image, masked_sigma,mag_zeropoint,\
                              exposure_time,self.psf,lensingPars),\
                    logpargs=(masked_image.shape),threads=threads)
            sampler.run_mcmc([pos]*ntemps, nsamples)


        print('\telapsed %.8f seconds'%(time.time()-tstart))
        print("\tMean acceptance fraction: %.3f"%(np.mean(sampler.acceptance_fraction)))
        if plot is True:
            labels = [r"$x_c$",r"$y_c$",r'$mag$',\
                      r'$r_e$',r"$n$",r"$q$",\
                      r"$\theta$"] + [r"$x_{PSF}$",r"$y_{PSF}$",r'$I_{PSF}$'] +\
                    ["sky"]
            print(labels)
            plot_results(sampler,labels,ntemps=ntemps)

        if  ntemps is not None:
            samples = sampler.chain[:,:, nexclude:, :].reshape((-1, ndim))
        else:
            samples = sampler.chain[:, nexclude:, :].reshape((-1, ndim))

        return samples.T



    def montecarlo_fit(self,initPars,mag_zeropoint,exposure_time,lensingPars,nRun = 10,verbose=False):
        r"""

        Parameters
        ----------

        Returns
        -------

        References
        ----------

        Examples
        --------

        """
        N,M=self.cutout.shape
        pars = initPars
        if self.sigmaImage is None:
            sigma = np.ones_like(self.cutout)
        else:
            sigma = self.sigmaImage

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
            model = simulation.generate_sersic_model((N,M),\
                        (xc,yc,mag-2.5*np.log10(lensMu),\
                        radius*majAxis_Factor,1.0,\
                        axis_ratio*shear_factor,position_angle),\
                        mag_zeropoint,exposure_time)

            if self.psf is not None:
                model = fftconvolve(model,self.psf)

            # model = galfit.galaxy_maker(mag_zeropoint,N,M,"sersic",xc,yc,\
            # (mag,radius,sersic_index,axis_ratio),position_angle,sky=0,psfname=self.psfname)

            diff_image = (model-self.cutout)*np.abs(1-self.mask)

            newChi = np.sum((diff_image*diff_image/(sigma*sigma)).ravel())

            if verbose:
                print(i,newChi,newChi/oldChi)
            if newChi/oldChi > np.random.uniform(1,1.5):
                if verbose:
                    print("Rejected Fit")
                continue
            else:
                Nchain[:,i] = xc,yc,mag,radius,1.0,axis_ratio,position_angle
                oldChi = newChi
                i+=1


        return Nchain

    def fit(self,initPars,mag_zeropoint,exposure_time,lensingPars=None,oversampling=3,nRun=10,verbose=False):


        if self.cutout is None:
            raise ValueError("No cutout for this galaxy is defined.")

        if self.mask is None:
            raise ValueError("No mask defined for this galaxy.")

        modelChain = self.montecarlo_fit(initPars,mag_zeropoint,exposure_time,\
                                         lensingPars,nRun=nRun,verbose=verbose)

        return modelChain

    def chi2fit_lensed(self,initPars,mag_zeropoint,exposure_time,lensingPars,oversampling=5,nRun=2):

        lensPars = LensPars(*lensingPars)
        nModels = (len(initPars)-1)//7

        modelPars = SersicParameters(initPars[-1])
        sigmaPars = modelPars.get_sigma(nModels)
        if len(sigmaPars) != len(initPars):
            for i in range(len(sigmaPars),len(initPars)):
                sigmaPars.append(1e-3)

        xc,yc,mag,radius,sersic_index,axis_ratio,position_angle = initPars[:7]
        skyValue = initPars[-1]

        sersicModel = LensedSersicModel(lensPars=lensPars,\
                                        magZP=mag_zeropoint,\
                                        psf=self.psf,\
                                        OverSampling=oversampling)
        skyModel = models.Const2D(amplitude = skyValue)
        model = sersicModel+skyModel#+psfModel

        if nModels>1:
            for i in range(nModels-1):
                sersicModel = LensedSersicModel(lensPars=lensPars,\
                                                magZP=mag_zeropoint,\
                                                psf=self.psf)

                model += sersicModel

        masked_image = np.ma.masked_array(self.cutout,mask=self.mask)
        masked_sigma = np.ma.masked_array(self.sigmaImage,mask=self.mask)

        N,M = masked_image.shape
        x,y = np.mgrid[:N,:M]

        # print(f"{utils.CRED}Init Pars:{utils.CEND}",",".join([f"{utils.CRED}{m:12.2f}{utils.CEND}" for m in initPars]))

        finalResults = np.zeros([model.parameters.size+1,nRun])
        i=0
        nbadFits = 0
        while i<nRun and nbadFits<100*nRun:
            initPos = np.random.normal(initPars,sigmaPars)
            model.x_0_0 = initPos[0]
            model.y_0_0 = initPos[1]
            model.I_eff_0 = np.random.lognormal(np.log10(initPars[2]),1.5)
            model.r_eff_0 = np.random.lognormal(np.log10(initPars[3]),1)
            model.n_0 = 1#initPos[4]
            # model.n_0.fixed = True
            model.axratio_0 = initPos[5]
            model.theta_0 = initPos[6]
            model.amplitude_1 = initPars[-1]
            if nModels ==2:
                model.x_0_2 = initPos[7]
                model.y_0_2 = initPos[8]
                model.I_eff_2 = initPos[9]
                model.r_eff_2 = initPos[10]
                model.n_2 = initPos[11]
                model.axratio_2 = initPos[12]
                model.theta_2 = initPos[13]

            # print(f"{utils.CBLUE}guess:{utils.CEND}",",".join([f"{utils.CBLUE}{m:12.2f}{utils.CEND}" for m in model.parameters]))

            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    modelFit = fitterLSQ(model, x, y, masked_image, weights=masked_sigma,maxiter = 5000)
                    nEvaluations = fitterLSQ.fit_info["nfev"]
                    if nEvaluations < 50:
                        nbadFits += 1
                        continue
                    chi = (1/masked_image.count())*np.ma.sum( (masked_image-modelFit(x,y))*(masked_image-modelFit(x,y))/(masked_sigma*masked_sigma))
                    # print(f"{utils.CGREEN}final:{utils.CEND}",",".join([f"{utils.CGREEN}{m:12.2f}{utils.CEND}" for m in modelFit.parameters]),f"chi={chi:.3f}")
                    finalResults[:-1,i] = modelFit.parameters
                    finalResults[-1,i] = chi
                    i+=1
                except (RuntimeError,OptimizeWarning) as e:
                    print(fitterLSQ.fit_info["nfev"])
                    print(fitterLSQ.fit_info["message"])
                    nbadFits += 1
                    continue

        # fig,ax = mpl.subplots(1,3,figsize=(20,10))
        # fig.subplots_adjust(wspace=0)
        # ax[0].imshow(masked_image)
        # ax[1].imshow(modelFit(x,y))
        # ax[2].imshow(masked_image-modelFit(x,y),cmap="RdYlGn")
        return finalResults


    def chi2fit_sersicPSFlensed(self,initPars,mag_zeropoint,exposure_time,lensingPars,oversampling=5,nRun=2,debug=False):

        lensPars = LensPars(*lensingPars)

        modelPars = SersicParameters(initPars[-1])
        sigmaPars = modelPars.get_sigma(1)
        if len(sigmaPars) != len(initPars):
            for i in range(len(sigmaPars),len(initPars)):
                sigmaPars.append(1e-3)

        xc,yc,mag,radius,sersic_index,axis_ratio,position_angle = initPars[:7]
        skyValue = initPars[-1]

        print("LensPars input:", lensPars)
        sersicModel = LensedSersicPSFModel(lensPars=lensPars,\
                                        magZP=mag_zeropoint,\
                                        psf=self.psf,\
                                        OverSampling=oversampling)
        skyModel = models.Const2D(amplitude = skyValue)
        model = sersicModel+skyModel

        masked_image = np.ma.masked_array(self.cutout,mask=self.mask)
        masked_sigma = np.ma.masked_array(self.sigmaImage,mask=self.mask)

        N,M = masked_image.shape
        x,y = np.mgrid[:N,:M]


        finalResults = np.zeros([model.parameters.size+1,nRun])
        i=0
        nbadFits = 0
        # print(f"{utils.CRED}Init Pars:{utils.CEND}",",".join([f"{utils.CRED}{m:12.2f}{utils.CEND}" for m in initPars]))

        while i<nRun and nbadFits<100*nRun:
            initPos = np.random.normal(initPars,sigmaPars)
            model.x_0_0 = initPos[0]
            model.y_0_0 = initPos[1]
            model.I_eff_0 = np.random.lognormal(np.log10(initPars[2]),1.5)
            model.r_eff_0 = np.random.lognormal(initPars[3],1)
            model.n_0 = 1.0#np.random.lognormal(initPars[4],0.5)
            # model.n_0.fixed = True
            model.axratio_0 = initPos[5]
            model.theta_0 = initPos[6]

            initPos = np.random.normal(initPars[7:9],sigmaPars[:2])
            model.xPSF_0 = initPos[0]
            model.yPSF_0 = initPos[1]
            model.I_psf_0 = np.abs(np.random.normal(initPars[9],5))

            model.amplitude_1 = initPars[-1]#np.random.normal(initPars[10],0.1)
            # model.amplitude_1.fixed=True


            # print(f"{utils.CBLUE}guess:{utils.CEND}",",".join([f"{utils.CBLUE}{m:12.2f}{utils.CEND}" for m in model.parameters]))


            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    modelFit = fitterLSQ(model, x, y, masked_image, weights=masked_sigma, maxiter = 5000)
                    # modelFit = fitterLSQ(model, x, y, masked_image, maxiter = 5000)
                    nEvaluations = fitterLSQ.fit_info["nfev"]
                    if nEvaluations < 50 or modelFit.I_eff_0 == 0 or modelFit.I_psf_0 == 0:
                        nbadFits += 1
                        continue

                    chi = (1/masked_image.count())*np.ma.sum( (masked_image-modelFit(x,y))*(masked_image-modelFit(x,y))/(masked_sigma*masked_sigma))
                    # print(f"{utils.CGREEN}final:{utils.CEND}",",".join([f"{utils.CGREEN}{m:12.2f}{utils.CEND}" for m in modelFit.parameters]),f"chi={chi:.3f}")
                    finalResults[:-1,i] = modelFit.parameters
                    finalResults[-1,i] = chi
                    i+=1
                except (RuntimeError,OptimizeWarning) as e:
                    print(fitterLSQ.fit_info["nfev"])
                    print(fitterLSQ.fit_info["message"])
                    nbadFits += 1
                    continue

        print(f"Acceptance rate: {i/(i+nbadFits):.3f}")

        # fig,ax = mpl.subplots(1,3,figsize=(20,10))
        # fig.subplots_adjust(wspace=0)
        # ax[0].imshow(masked_image)
        # ax[1].imshow(modelFit(x,y))
        # ax[2].imshow(masked_image-modelFit(x,y),cmap="RdYlGn")
        return finalResults

    def gridSearch_sersicPSFlensed(self,initPars,mag_zeropoint,exposure_time,lensingPars,oversampling=5,nRun=2):

        # lensPars = LensPars(*lensingPars)
        #
        # modelPars = SersicParameters(initPars[-1])
        # sigmaPars = modelPars.get_sigma(1)
        # if len(sigmaPars) != len(initPars):
        #     for i in range(len(sigmaPars),len(initPars)):
        #         sigmaPars.append(1e-3)
        #
        # xc,yc,mag,radius,sersic_index,axis_ratio,position_angle = initPars[:7]
        # skyValue = initPars[-1]
        #
        # print("LesnPars input:", lensPars)
        # sersicModel = LensedSersicPSFModel(lensPars=lensPars,\
        #                                  magZP=mag_zeropoint,\
        #                                  psf=self.psf,\
        #                                  OverSampling=oversampling)
        # skyModel = models.Const2D(amplitude = skyValue)
        # model = sersicModel+skyModel
        #
        # masked_image = np.ma.masked_array(self.cutout,mask=self.mask)
        # masked_sigma = np.ma.masked_array(self.sigmaImage,mask=self.mask)
        #
        # N,M = masked_image.shape
        # x,y = np.mgrid[:N,:M]
        #
        #
        # finalResults = np.zeros([model.parameters.size,nRun])
        # i=0
        # nbadFits = 0
        # print(f"{utils.CRED}Init Pars:{utils.CEND}",",".join([f"{utils.CRED}{m:12.2f}{utils.CEND}" for m in initPars]))
        #
        # while i<nRun and nbadFits<100*nRun:
        #     initPos = np.random.normal(initPars,sigmaPars)
        #     model.x_0_0 = initPos[0]
        #     model.y_0_0 = initPos[1]
        #     model.I_eff_0 = np.random.lognormal(np.log10(initPars[2]),1.5)
        #     model.r_eff_0 = np.random.lognormal(initPars[3],1)
        #     model.n_0 = np.random.lognormal(initPars[4],0.5)
        #     model.axratio_0 = np.random.uniform(0.1,1)
        #     model.theta_0 = initPos[6]
        #     model.amplitude_1 = initPos[-1]
        #
        #     initPos = np.random.normal(initPars[7:9],sigmaPars[:2])
        #     model.xPSF_0 = initPos[0]
        #     model.yPSF_0 = initPos[1]
        #     model.I_psf_0 = np.abs(np.random.normal(initPars[9],5))
        #
        #
        #     print(f"{utils.CBLUE}guess:{utils.CEND}",",".join([f"{utils.CBLUE}{m:12.2f}{utils.CEND}" for m in model.parameters]))
        #
        #
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("error", OptimizeWarning)
        #         try:
        #             modelFit = fitterLSQ(model, x, y, masked_image, weights=masked_sigma, maxiter = 5000)
        #             nEvaluations = fitterLSQ.fit_info["nfev"]
        #             if nEvaluations < 50:
        #                 nbadFits += 1
        #                 continue
        #
        #             # fig,ax=mpl.subplots()
        #             # ax.imshow(modelFit(x,y))
        #             # ax.set_title(r"$I_\mathrm{eff}=%.2f,\ I_\mathrm{psf}=%.2f$"%(modelFit.I_eff_0.value,modelFit.I_psf_0.value))
        #             print(f"{utils.CGREEN}final:{utils.CEND}",",".join([f"{utils.CGREEN}{m:12.2f}{utils.CEND}" for m in modelFit.parameters]))
        #             finalResults[:,i] = modelFit.parameters
        #             i+=1
        #         except (RuntimeError,OptimizeWarning) as e:
        #             print(fitterLSQ.fit_info["nfev"])
        #             print(fitterLSQ.fit_info["message"])
        #             nbadFits += 1
        #             continue
        #
        # print(f"Acceptance rate: {i/(i+nbadFits):.3f}")

        return finalResults



def plot_results(sampler,pars,ntemps=None):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    ColorsTemps = ["Silver","DodgerBlue","Crimson","ForestGreen",\
                   "DarkOrange","Indigo","Goldenrod","Magenta","SteelBlue",\
                   "LimeGreen","Coral","Brown","Violet"]
    import matplotlib.gridspec as gridspec
    assert len(pars)==sampler.dim
    if ntemps is None:
        nwalkers = sampler.chain.shape[0]
        ntemps = 1
        chain = sampler.chain[np.newaxis,...]
        flatChain = sampler.flatchain[np.newaxis,...]
    else:
        nwalkers = sampler.chain.shape[1]
        chain = sampler.chain
        flatChain = sampler.flatchain
    NP=len(pars)

    fig=mpl.figure(figsize=(25,NP*4))

    gs = gridspec.GridSpec(NP, 2,width_ratios=[5,1])
    ax = [mpl.subplot(gs[i]) for i in range(0,2*NP,2)] +[mpl.subplot(gs[i]) for i in range(1,2*NP,2)]
    mpl.subplots_adjust(hspace=0.0,wspace=0.0)

    for n in range(NP):
        for j in range(ntemps):
            for i in range(nwalkers):
                ax[n].plot(chain[j,i,:,n],alpha=0.35,color=ColorsTemps[j],lw=0.5)
                ax[n].set_ylabel(pars[n],fontsize=16)
            # ax[NP+n].hist(flatChain[j,:,n],bins=50,orientation='horizontal',color='silver',histtype='stepfilled')
        # print(flatChain.shape,flatChain[:,:,n].shape,flatChain[:,:,n].T.shape)
        ax[NP+n].hist(flatChain[:,:,n].T,bins=50,orientation='horizontal',color=ColorsTemps[:ntemps], histtype='bar', stacked=True)
        if n<(NP-1):
            ax[n].tick_params(labelbottom=False)
            ax[NP+n].tick_params(labelbottom=False)
        ax[NP+n].tick_params(labelleft=False,labelsize=14)
        ax[n].tick_params(labelsize=12)

    ax[NP-1].set_xlabel(r'$N_\mathrm{step}$')
    ax[-1].set_xlabel(r'$N_\mathrm{sol}$')

    return fig,ax
