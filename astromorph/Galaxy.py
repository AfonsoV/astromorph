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
import time
import emcee



### MCMC
def lnlikelihood(pars,image,sigma,mag_zeropoint,exposure_time,psf,lensingPars):
    xc,yc,mag,radius,axis_ratio,position_angle,sky = pars
    if lensingPars is None:
        model = simulation.generate_sersic_model(image.shape,\
                    (xc,yc,mag,radius,1.0,axis_ratio,position_angle),\
                    mag_zeropoint,exposure_time)
    else:
        model = simulation.generate_lensed_sersic_model(image.shape,\
                    (xc,yc,mag,radius,1.0,axis_ratio,position_angle),\
                    lensingPars,mag_zeropoint,exposure_time)

    if psf is not None:
        model = fftconvolve(model,psf,mode="same")
        # model = convolve_fft(model,self.psf)
    model+=sky

    return -0.5*np.ma.sum( (image-model)*(image-model)/(sigma*sigma) + np.ma.log(2*np.pi*sigma*sigma) )

def prior(pars,shape):
    x,y,m,r,q,t,s=pars
##    ,n,q,t=pars
    N,M = shape
    if  (N/4<x<3*N/4) and\
        (M/4<y<3*M/4) and\
        (0<m<35) and\
        (0.1<r<50) and\
        (0.1<q<1) and\
        (-90<t<90):
        return 0.0
    return -np.inf

def lnprobability(pars,image,sigma,mag_zeropoint,exposure_time,psf=None,lensingPars=None):
    pr = prior(pars,image.shape)
    if not np.isfinite(pr):
        return -np.inf
    LL = pr + lnlikelihood(pars,image,sigma,mag_zeropoint,exposure_time,psf,lensingPars)
    if np.isfinite(LL):
        return LL
    else:
        return -np.inf



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
            self.imgdata = pyfits.getdata(imgname)
            self.imgheader = pyfits.getheader(imgname)

        self.psf = None
        self.psfname = "none"
        self.mask = None
        self.sigmaImage = None

    def set_coords(self,coords):
        self.coords=coords
        return None

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

    def get_bounding_box_coordinates(self):
        assert (self.bounding_box  is not None),"A bounding box must be assigned to the galaxy"
        header = self.imgheader
        x0,x1,y0,y1=self.bounding_box
        wcs=pywcs.WCS(header)
        pixCoords=wcs.wcs_pix2world([[x0,y0],[x1,y1]],1)
        pixLow = pixCoords[0]
        pixHig = pixCoords[1]
        return (pixLow[0],pixHig[0],pixLow[1],pixHig[1])

    def get_image_coordinates(self):
        header = self.imgheader
        x0,x1,y0,y1=0,self.imgdata.shape[1],0,self.imgdata.shape[0]
        wcs=pywcs.WCS(header)
        pixCoords=wcs.wcs_pix2world([[x0,y0],[x1,y1]],1)
        pixLow = pixCoords[0]
        pixHig = pixCoords[1]
        return (pixLow[0],pixHig[0],pixLow[1],pixHig[1])

    def set_bounding_box(self,size,pixelscale):
        self.bounding_box = utils.get_bounding_box(self.imgheader,self.coords,size,pixelscale)
        x0,x1,y0,y1=self.bounding_box
        self.cutout = self.imgdata[y0:y1,x0:x1]
        return None

    def set_segmentation_mask(self,mask):
        self.mask = mask
        return None

    def create_segmentation_mask(self,pixscale,radius,**kwargs):
        N,M=self.cutout.shape
        xc=N/2
        yc=M/2
        # segmap = utils.gen_segmap_tresh(self.cutout,xc,yc,pixscale,radius=radius,**kwargs)
        segmap = utils.gen_segmap_watershed(self.cutout,**kwargs)
        objmap = utils.select_object_map(xc,yc,segmap,pixscale,radius)
        segmap[segmap>0]=1
        self.set_segmentation_mask(segmap - objmap)
        return self.mask

    def create_object_mask(self,pixscale,radius,**kwargs):
        N,M=self.cutout.shape
        xc=N/2
        yc=M/2
        segmap = utils.gen_segmap_watershed(self.cutout,**kwargs)
        objmap = utils.select_object_map(xc,yc,segmap,pixscale,radius)
        return objmap

    def set_psf(self,filename=None,psfdata=None,resize=None):
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

        if resize is not None:
            self.psf = self.psf[50:-50,50:-50]
        return None

    def set_sigma(self,sigma):
        self.sigmaImage = sigma
        return None

    def default_mask(self,radius):
        if self.cutout is None:
            raise ValueError("cutuout attribute must be defined.")

        N,M = self.cutout.shape
        xc,yc = N/2,M/2
        R = utils.compute_ellipse_distmat(self.cutout,xc,yc)
        mask = np.zeros_like(self.cutout)
        mask[R<radius]=1
        return mask

    def estimate_parameters(self,mag_zeropoint,exposure_time,pixscale,radius,rPsf=2.5,**kwargs):
        if self.mask is not None:
            detection_image = self.cutout*(1-self.mask)
        else:
            detection_image = self.cutout
        objectMask = self.create_object_mask(pixscale,radius,**kwargs)
        if objectMask[objectMask>0].size ==0:
            objectMask = self.default_mask(radius/pixscale)

        xc,yc=utils.barycenter(self.cutout,objectMask)
        mag = -2.5*np.log10(np.nansum(self.cutout*objectMask)/exposure_time)+mag_zeropoint
        # r100 = max(0.5,np.sqrt(self.cutout[object_mask==1].size/np.pi - 2.5*2.5))
        r50 = utils.get_half_light_radius(self.cutout,objectMask)
        r50 = max(0.5,np.sqrt(r50*r50-rPsf*rPsf))
        axisRatio = utils.get_axis_ratio(self.cutout,objectMask)
        positionAngle = utils.get_position_angle(self.cutout,objectMask) - 90

        while positionAngle>90:
            positionAngle -= 180

        while positionAngle<-90:
            positionAngle += 180

        return (xc,yc,mag,r50,axisRatio,positionAngle)

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



    def emcee_fit(self,initPars,mag_zeropoint,exposure_time,lensingPars=None,nchain=20,nsamples=10000,plot=False,threads=1,ntemps=None):
        nexclude = nsamples//2

        masked_image = np.ma.masked_array(self.cutout,mask=self.mask)
        masked_sigma = np.ma.masked_array(self.sigmaImage,mask=self.mask)

        ndim, nwalkers = len(initPars), nchain
        sigmaPars = [2.5,2.5,0.5,5.0,0.075,5.0,np.abs(0.25*initPars[-1])]
        pos = np.array([np.random.normal(initPars,sigmaPars) for i in range(nwalkers)])
        pos[:,0][pos[:,0]<0]=1
        pos[:,0][pos[:,0]>self.cutout.shape[0]]=self.cutout.shape[0]-1
        pos[:,1][pos[:,1]<0]=1
        pos[:,1][pos[:,1]>self.cutout.shape[1]]=self.cutout.shape[1]-1
        pos[:,3][pos[:,3]<0]=1
        pos[:,4][pos[:,4]<=0]=0.1
        pos[:,4][pos[:,4]>1]=1.0

        tstart = time.time()
        if ntemps is None:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobability,\
                        args=(masked_image, masked_sigma,mag_zeropoint,\
                              exposure_time, self.psf,lensingPars),\
                        threads=threads,a=2.4)
            sampler.run_mcmc(pos, nsamples)
        else:
            sampler = emcee.PTSampler(ntemps,nwalkers, ndim, lnprobability,prior,\
                    loglargs=(masked_image, masked_sigma,mag_zeropoint,\
                              exposure_time),\
                    logpargs=(masked_image.shape,),threads=threads)
            sampler.run_mcmc([pos]*ntemps, nsamples)


        print('elapsed %.8f seconds'%(time.time()-tstart))
        print("Mean acceptance fraction: %.3f"%(np.mean(sampler.acceptance_fraction)))
        if plot is True:
            plot_results(sampler,[r"$x_c$",r"$y_c$",r'$mag$',\
                                  r'$r_e\ [\mathrm{arcsec}]$',r"$(b/a)$",\
                                  r"$\theta_\mathrm{PA}$",r"sky"],ntemps=ntemps)

        if  ntemps is not None:
            samples = sampler.chain[:,:, nexclude:, :].reshape((-1, ndim))
        else:
            samples = sampler.chain[:, nexclude:, :].reshape((-1, ndim))

        return samples.T


    def montecarlo_fit(self,initPars,mag_zeropoint,exposure_time,lensingPars,nRun = 10,verbose=False):

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


##def plot_results(sampler,pars):
##    assert len(pars)==sampler.dim
##    ntemps = sampler.chain.shape[0]
##    nwalkers = sampler.chain.shape[1]
##    NP=len(pars)
##
##    fig=mpl.figure(figsize=(25,NP*4))
##
##    gs = gridspec.GridSpec(NP, 2,width_ratios=[5,1])
##
##    main_axes = [mpl.subplot(gs[k]) for k in range(0,2*NP,2)]
##    hist_axes = [mpl.subplot(gs[k]) for k in range(1,2*NP,2)]
##
##    ax=main_axes+hist_axes
##    mpl.subplots_adjust(hspace=0.0,wspace=0.0)
##
##    for n in range(NP):
##        for j in range(ntemps):
##            for i in range(nwalkers):
##                ax[n].plot(sampler.chain[j,i,:,n],color=cm.rainbow_r(float(j)/ntemps),alpha=0.35)
##                ax[n].set_ylabel(pars[n])
##
##            ax[NP+n].hist(sampler.flatchain[j,:,n],bins=50,orientation='horizontal',color=cm.rainbow_r(float(j)/ntemps),histtype='stepfilled',alpha=0.35)
##
##        if n<(NP-1):
##            ax[n].tick_params(labelbottom='off')
##            ax[NP+n].tick_params(labelbottom='off')
##        ax[NP+n].tick_params(labelleft='off')
##
##    ax[NP-1].set_xlabel(r'$N_\mathrm{step}$')
##    ax[-1].set_xlabel(r'$N_\mathrm{sol}$')
##
##    return fig,ax



def plot_results(sampler,pars,ntemps=None):
    ColorsTemps = ["black","DodgerBlue","Crimson","ForestGreen",\
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
                ax[n].plot(chain[j,i,:,n],alpha=0.35,color=ColorsTemps[j])
                ax[n].set_ylabel(pars[n])
            ax[NP+n].hist(flatChain[j,:,n],bins=50,orientation='horizontal',color='silver',histtype='stepfilled')
        if n<(NP-1):
            ax[n].tick_params(labelbottom='off')
            ax[NP+n].tick_params(labelbottom='off')
        ax[NP+n].tick_params(labelleft='off')

    ax[NP-1].set_xlabel(r'$N_\mathrm{step}$')
    ax[-1].set_xlabel(r'$N_\mathrm{sol}$')

    return fig,ax
