import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as pywcs

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
            hsize = int(size/pixelScale)//2
            xl = int(kx-hsize)
            xu = int(kx+hsize)
            yl = int(ky-hsize)
            yu = int(ky+hsize)
            return (xl,xu,yl,yu)
        else:
            raise ModelError("Neither a file nor an extent set is defined for this model to get coordinates from.")

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

    def get_magnification(self,redshift,minMag=5e-2):
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

    def get_lensing_parameters_at_position(self,coords,window=5):
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
        return (medKappa,medGamma,medMu,medAngle)
