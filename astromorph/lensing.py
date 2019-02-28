import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import matplotlib.pyplot as mpl
import scipy.interpolate as sip

from . import utils
from . import plot_utils as putils
from .cosmology import angular_distance


class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class ModelError(Error):
    """Exception raised for errors in the model definition.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class LensingModel(object):
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

    def __init__(self,redshift_lens,pixelScale):
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
        self.gamma = None
        self.kappa = None
        self.xdeflect = None
        self.ydeflect = None

        self.kappa_at_z = None
        self.gamma_at_z = None
        self.mu_at_z = None
        self.xdeflect_at_z = None
        self.ydeflect_at_z = None

        self.lens_redshift = redshift_lens
        self.pixelScale = pixelScale

    def set_lensing_data(self,filename=None,gamma=None,kappa=None,xdeflect=None,ydeflect=None,extent=None):
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
            self.modelname = filename
            self.header = pyfits.getheader("%s_gamma.fits"%(filename))
            self.gamma = pyfits.getdata("%s_gamma.fits"%(filename))
            self.kappa = pyfits.getdata("%s_kappa.fits"%(filename))
            try:
                self.xdeflect = pyfits.getdata("%s_x-arcsec-deflect.fits"%(filename))
                self.ydeflect = pyfits.getdata("%s_y-arcsec-deflect.fits"%(filename))
            except IOError:
                self.xdeflect = None
                self.ydeflect = None

        elif (kappa is not None) and (gamma is not None) and (xdeflect is not None) and (ydeflect is not None):
            self.modelname = None
            self.gamma = gamma
            self.kappa = kappa
            self.xdeflect = xdeflect
            self.ydeflect = ydeflect
            self.modelExtent = extent
        else:
            raise ModelError("Either a filename is given or all 4 components (gamma,kappa,xy-deflection) must be given.")

    def set_bounding_box(self,size,coords,pixelScale=None):
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
        if pixelScale is None:
            pixelScale = self.pixelScale
        if self.modelname is not None:
            return utils.get_bounding_box(pyfits.getheader("%s_gamma.fits"%(self.modelname)),coords,size,\
                                pixelScale)
        elif self.modelExtent is not None:
            ra = np.linspace(self.modelExtent[0],self.modelExtent[1],self.gamma.shape[1])
            dec = np.linspace(self.modelExtent[2],self.modelExtent[3],self.gamma.shape[0])
            ky = np.where(ra>coords.ra.value)[0][0]
            kx = np.where(dec>coords.dec.value)[0][0]
            hsize = (size/pixelScale)/2
            xl = int(np.floor(kx-hsize))
            xu = int(np.ceil(kx+hsize))
            yl = int(np.floor(ky-hsize))
            yu = int(np.ceil(ky+hsize))
            return (xl,xu,yl,yu)
        else:
            raise ModelError("Neither a file nor an extent set is defined for this model to get coordinates from.")

    def get_bounding_box_coordinates(self,bounding_box):
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
        header = self.header
        x0,x1,y0,y1=bounding_box
        wcs=pywcs.WCS(header)
        pixCoords=wcs.wcs_pix2world([[x0,y0],[x1,y1]],1)
        pixLow = pixCoords[0]
        pixHig = pixCoords[1]
        return (pixLow[0],pixHig[0],pixLow[1],pixHig[1])

    def get_image_box_coordinates(self):
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
        header = self.header
        x0,x1,y0,y1=0,self.gamma.shape[1],0,self.gamma.shape[0]
        wcs=pywcs.WCS(header)
        pixCoords=wcs.wcs_pix2world([[x0,y0],[x1,y1]],1)
        pixLow = pixCoords[0]
        pixHig = pixCoords[1]
        return (pixLow[0],pixHig[0],pixLow[1],pixHig[1])



    def set_model_at_z(self,redshift,**kwargs):
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
        dlens = angular_distance(self.lens_redshift)
        dsource = angular_distance(redshift)
        dlens_source = dsource - (1+self.lens_redshift)/(1+redshift)*dlens

        self.kappa_at_z = self.kappa * (dlens_source/dsource)
        self.gamma_at_z = self.gamma * (dlens_source/dsource)
        self.xdeflect_at_z = self.xdeflect * (dlens_source/dsource)
        self.ydeflect_at_z = self.ydeflect * (dlens_source/dsource)

        self.get_magnification(redshift,**kwargs)
        return None

    def compute_shear_angle(self,redshift):
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
        if self.xdeflect_at_z is None or self.ydeflect_at_z is None:
            self.set_model_at_z(redshift)
        dxd_dy,dxd_dx = np.gradient(self.xdeflect_at_z)
        dyd_dy,dyd_dx = np.gradient(self.ydeflect_at_z)
        self.shear_angle = np.degrees(np.arctan2(-0.5*(dxd_dy+dyd_dx),+0.5*(dyd_dy-dxd_dx)))
        return self.shear_angle

    def get_magnification(self,redshift,minMag=5e-4):
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
        if self.kappa_at_z is None or self.gamma_at_z is None:
            self.set_model_at_z(redshift)

        divFactor = np.abs((1-self.kappa_at_z)*(1-self.kappa_at_z) - self.gamma_at_z*self.gamma_at_z)
        divFactor[divFactor==0]=minMag
        self.mu_at_z = 1/divFactor
        return self.mu_at_z

    def draw_cutout(self,size,coords,component="gamma",eixo=None,**kwargs):
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
        if eixo is None:
            fig,ax = mpl.subplots()
        else:
            ax = eixo

        xl,xu,yl,yu = self.set_bounding_box(size,coords)
        if component == "shear":
            data = self.gamma
        elif component == "convergence":
            data = self.kappa
        elif component == "magnification":
            data = self.mu
        elif component == "x-deflect":
            data = self.xdeflect
        elif component == "y-deflect":
            data = self.ydeflect
        else:
            raise ValueError("Invalid value for component. It should be one of: kappa, gamma,magnification x-defelct or y-deflect")
        putils.show_image(ax,data[yl:yu,xl:xu],scale="linear",\
                          extent=(-size/2,size/2,-size/2,size/2),**kwargs)
        putils.draw_cross(ax,0,0,gap=1.0,size=1.5,color="white",lw=3)
        return None

    def get_lensing_parameters_at_position(self,coords,window=5,plot=False):
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
        xl,xu,yl,yu = self.set_bounding_box(window,coords,pixelScale=1.0)

        medKappa = np.median(self.kappa_at_z[yl:yu,xl:xu])
        medGamma = np.median(self.gamma_at_z[yl:yu,xl:xu])
        medMu = np.median(self.mu_at_z[yl:yu,xl:xu])
        medAngle = np.median(self.shear_angle[yl:yu,xl:xu])

        if plot is True:
            fig,ax = mpl.subplots(1,3)
            fig.subplots_adjust(wspace=0)
            ax[0].imshow(self.kappa_at_z,vmin=0,vmax=min(self.kappa_at_z.max(),3))
            ax[0].set_title("convergence")
            ax[1].imshow(self.gamma_at_z,vmin=0,vmax=min(self.gamma_at_z.max(),3))
            ax[1].set_title("shear")
            ax[2].imshow(self.mu_at_z,vmin=1,vmax=min(self.mu_at_z.max(),30))
            ax[2].set_title("magnification")
            # ax[3].imshow(self.shear_angle)
            # ax[3].set_title("shear angle")
            for eixo in ax:
                eixo.tick_params(labelleft=False,labelbottom=False)

        return (medKappa,medGamma,medMu,medAngle)



def regularized_coordinates(xmin,xmax,ymin,ymax,image):
    if ymin<0:
        return False
    if xmin<0:
        return False
    if ymax>image.shape[0]:
        return False
    if xmax>image.shape[1]:
        return False
    return True

def stack_models(models,raExtent,decExtent,size=None,scale=None,modelbbox=None):
    if size is None and scale is None:
        raise ValueError("Either size or pixscale must be defined")

    if scale is not None:
        raGrid = np.arange(raExtent[0],raExtent[1]+scale,scale)
        decGrid = np.arange(decExtent[0],decExtent[1]+scale,scale)
    elif size is not None:
        raGrid = np.linspace(raExtent[0],raExtent[1],size)
        decGrid = np.linspace(decExtent[0],decExtent[1],size)

    # print(raExtent,raGrid.size)
    # print(decExtent,decGrid.size)

    modelGrid = np.zeros([4,decGrid.size,raGrid.size,len(models)])
    for i in range(len(models)):
        # print("Model",i+1)
        if models[i].kappa_at_z is None:
            kappa = models[i].kappa
            gamma = models[i].gamma
            xdeflect = models[i].xdeflect
            ydeflect = models[i].ydeflect
        else:
            kappa = models[i].kappa_at_z
            gamma = models[i].gamma_at_z
            xdeflect = models[i].xdeflect_at_z
            ydeflect = models[i].ydeflect_at_z

        if xdeflect is None:
            #If not present in model files, ignore
            xdeflect = np.ones_like(kappa)*np.nan
            ydeflect = np.ones_like(kappa)*np.nan

        if modelbbox is None:
            modelExtent = models[i].get_image_box_coordinates()
            raModel = np.linspace(modelExtent[0],modelExtent[1],\
                                  models[i].gamma.shape[1])
            decModel = np.linspace(modelExtent[2],modelExtent[3],\
                                   models[i].gamma.shape[0])
        else:
            size,coords = modelbbox
            try:
                xl,xu,yl,yu = models[i].set_bounding_box(size,coords)

                if not regularized_coordinates(xl,xu,yl,yu,kappa):
                    padWidth = int(size/models[i].pixelScale)
                    kappaPadded = np.pad(kappa,padWidth,mode="constant",\
                                         constant_values=np.nan)
                    gammaPadded = np.pad(gamma,padWidth,mode="constant",\
                                         constant_values=np.nan)
                    xdeflectPadded = np.pad(xdeflect,padWidth,mode="constant",\
                                         constant_values=np.nan)
                    ydeflectPadded = np.pad(ydeflect,padWidth,mode="constant",\
                                         constant_values=np.nan)
                    xl += padWidth
                    xu += padWidth
                    yl += padWidth
                    yu += padWidth
                    kappa = kappaPadded[yl:yu,xl:xu]
                    gamma = gammaPadded[yl:yu,xl:xu]
                    xdeflect = xdeflectPadded[yl:yu,xl:xu]
                    ydeflect = ydeflectPadded[yl:yu,xl:xu]
                else:
                    kappa = kappa[yl:yu,xl:xu]
                    gamma = gamma[yl:yu,xl:xu]
                    xdeflect = xdeflect[yl:yu,xl:xu]
                    ydeflect = ydeflect[yl:yu,xl:xu]

                N,M = kappa.shape
                bbox =(xl,xu,yl,yu )
                extentModel = models[i].get_bounding_box_coordinates(bbox)
                # raModel = np.linspace(coords.ra.value-size/(2*3600.),\
                #                       coords.ra.value+size/(2*3600.),M)
                # decModel = np.linspace(coords.dec.value-size/(2*3600.),\
                #                        coords.dec.value+size/(2*3600.),N)
                raModel = np.linspace(extentModel[0],\
                                      extentModel[1],M)
                decModel = np.linspace(extentModel[2],\
                                       extentModel[3],N)

            except IndexError as err:
                kappa = None
                gamma = None
                xdeflect = None
                ydeflect = None
                print(err)


        for k,modelVariable in enumerate([kappa,gamma,xdeflect,ydeflect]):
            # print("Variable",k+1,modelVariable.shape)
            if kappa is None:
                modelGrid[k,:,:,i] = np.nan*np.ones([decGrid.size,raGrid.size])
            else:
                # print(raModel.shape,decModel.shape,modelVariable.shape)
                # modelInterpolator = sip.interp2d(raModel,decModel,modelVariable,\
                #                         bounds_error=False,fill_value=np.nan)
                modelInterpolator = sip.RectBivariateSpline(decModel,raModel[::-1],modelVariable,kx=1,ky=1)
                modelGrid[k,:,:,i] = modelInterpolator(decGrid,raGrid)[:,::-1]

    extentStack = (raGrid[0],raGrid[-1],decGrid[0],decGrid[-1])
    print("extentStack",extentStack)
    # modelstack =  np.nanmedian(modelGrid,axis=-1)
    return np.nanpercentile(modelGrid,[16,50,84],axis=-1),extentStack

def saveMagnifcationCutout(fname,magmap,extent,pixscale):

    w = pywcs.WCS(naxis=2)

    w.wcs.crpix = [1,1]
    w.wcs.cdelt = np.array([pixscale/3600, pixscale/3600])
    w.wcs.crval = [extent[0], extent[2]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # w.wcs.set_pv([(2, 1, 45.0)])

    header = w.to_header()

    hdu = pyfits.PrimaryHDU(magmap,header=header)
    hdu.writeto(fname,overwrite=True)
    return hdu
