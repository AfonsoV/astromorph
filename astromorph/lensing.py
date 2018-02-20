import numpy as np
import astropy.io.fits as pyfits

from . import utils
from . import plot_utils as putils
from .cosmology import angular_distance

class LensingModel(object):

    def __init__(self,model_root,redshift_lens,pixel_scale):
        self.modelname = model_root
        self.header = pyfits.getheader("%s_gamma.fits"%(model_root))
        self.gamma = pyfits.getdata("%s_gamma.fits"%(model_root))
        self.kappa = pyfits.getdata("%s_kappa.fits"%(model_root))
        # self.xdeflect = pyfits.getdata("%s_x-arcsec-deflect.fits"%(model_root))
        # self.ydeflect = pyfits.getdata("%s_y-arcsec-deflect.fits"%(model_root))

        self.lens_redshift = redshift_lens
        self.pixel_scale = pixel_scale


    def set_bounding_box(self,size,coords):
        return utils.get_bounding_box("%s_gamma.fits"%(self.modelname),coords,size,\
                                self.pixel_scale)

    def get_image_box_coordinates(self):
        def fx(x):
            return (x-self.header["CRPIX1"])*self.header["CDELT1"] + self.header["CRVAL1"]
        def fy(y):
            return (y-self.header["CRPIX2"])*self.header["CDELT2"] + self.header["CRVAL2"]
        return (fx(0),fx(self.gamma.shape[1]),fy(0),fy(self.gamma.shape[0]))


    def get_magnification(self,redshift,minMag=5e-2):

        dlens = angular_distance(self.lens_redshift)
        dsource = angular_distance(redshift)
        dlens_source = dsource - (1+self.lens_redshift)/(1+redshift)*dlens

        kappa_at_z = self.kappa * (dlens_source/dsource)
        gamma_at_z = self.kappa * (dlens_source/dsource)

        divFactor = np.abs((1-kappa_at_z)*(1-kappa_at_z) - gamma_at_z*gamma_at_z)
        divFactor[divFactor==0]=minMag
        self.mu = 1/divFactor
        return self.mu

    def draw_cutout(self,size,coords,component="gamma",eixo=None,**kwargs):
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
        # elif component == "x-deflect":
        #     data = self.xdeflect
        # elif component == "y-deflect":
        #     data = self.ydeflect
        else:
            raise ValueError("Invalid value for component. It should be one of: kappa, gamma, x-defelct or y-deflect")
        putils.show_image(ax,data[yl:yu,xl:xu],scale="linear",\
                          extent=(-size/2,size/2,-size/2,size/2),**kwargs)
        putils.draw_cross(ax,0,0,gap=1.0,size=1.5,color="white",lw=3)
        return None

    def get_lensing_parameters_at_position(self,coords,window=5):
        xl,xu,yl,yu = utils.get_bounding_box("%s_gamma.fits"%(self.modelname),coords,\
                                        window,1.0)
        medKappa = np.median(self.kappa[yl:yu,xl:xu])
        medGamma = np.median(self.gamma[yl:yu,xl:xu])
        medMu = np.median(self.mu[yl:yu,xl:xu])
        return (medKappa,medGamma,medMu)
