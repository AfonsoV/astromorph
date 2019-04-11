import numpy as np
from astropy.modeling import Fittable2DModel, Parameter
from .simulation import kappa,total_flux
from scipy.signal import fftconvolve
from scipy.ndimage import shift
from .utils import barycenter
import warnings

import matplotlib.pyplot as mpl

class LensPars:

    def __init__(self, kappa, gamma,mu,shearAngle):
        self.kappa = kappa
        self.gamma = gamma
        self.mu = mu
        self.shearAngle = shearAngle
        return None


    def __str__(self):
        return f"{self.kappa}, {self.gamma}, {self.mu}, {self.shearAngle}"

class Test2D(Fittable2DModel):
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    p=0

    def __init__(self, x_0=x_0.default, y_0=y_0.default,p=0,**kwargs):
        self.p = p
        super().__init__(x_0, y_0, **kwargs)

    def evaluate(self,x,y,x_0,y_0):
        print(self.p)
        r = np.sqrt((x-x_0)*(x-x_0)+(y-y_0)*(y-y_0))
        return np.exp(-r/10)


class PsfModel(Fittable2DModel):

    x_0 = Parameter(default=1,min=0)
    y_0 = Parameter(default=1,min=0)
    I_0 = Parameter(default = 1, min=1e-5)

    def __init__(self,psf,x_0=x_0.default, y_0=y_0.default,I_0=I_0.default, **kwargs):
        self.psf = psf
        super().__init__(x_0, y_0, I_0, **kwargs)
        return None


    def evaluate(self,x,y,x_0, y_0,I_0):

        canvas = np.zeros_like(x)
        Npsf,Mpsf = self.psf.shape
        N,M=x.shape

        if x.shape > self.psf.shape:
            canvas[:Npsf,:Mpsf] = self.psf
        else:
            diffN = Npsf - N
            diffM = Npsf - N
            canvas[diffN//2:-diffN//2,diffM//2:-diffM//2] = self.psf


        xCen,yCen = barycenter(canvas,np.ones_like(canvas))
        dx = (x_0-xCen)
        dy = (y_0-yCen)
        finalPSF = shift(canvas,(dx,dy),order=2)
        return I_0*finalPSF


class LensedSersicModel(Fittable2DModel):

    x_0 = Parameter(default=0,min=-10)
    y_0 = Parameter(default=0,min=-10)
    I_eff = Parameter(default=1,min=0)
    r_eff = Parameter(default=1,min=0.1,max=100)
    n = Parameter(default=1,min=0.1,max=20)
    axratio = Parameter(default=1,min=0.1,max=1.0)
    theta = Parameter(default=0,min=-90,max=90)

    def __init__(self, x_0=x_0.default, y_0=y_0.default, I_eff=I_eff.default,\
                 r_eff=r_eff.default, n=n.default, axratio=axratio.default,\
                 theta=theta.default, psf=None, magZP = 0, exptime =1,\
                 lensPars=None, OverSampling = 5, **kwargs):

        if psf is None:
            warnings.warn("No PSF provided. Convlution not performed.")

        self.psf = psf
        self.magZP = magZP
        self.exptime = exptime
        self.overSampling = np.int(OverSampling)
        assert self.overSampling >= 1, "OverSampling must be equal or greater than 1"

        if lensPars is None:
            raise ValueError("lensPars must be a LensPars object.")
        else:
            self.lensPars = lensPars

        super().__init__(x_0, y_0, I_eff, r_eff, n, axratio, theta, **kwargs)
        return None

    def evaluate(self, x, y, x_0, y_0, I_eff, r_eff, n, axratio, theta):
        """Two dimensional Sersic profile function in a lensed plane."""

        lensKappa = self.lensPars.kappa
        lensGamma = self.lensPars.gamma
        lensMu = self.lensPars.mu
        lensAngle = self.lensPars.shearAngle

        OverSampling = self.overSampling

        shear_factor = (1-lensKappa-lensGamma)/(1-lensKappa+lensGamma)
        ang_rad = np.radians(lensAngle)
        cosLA = np.cos(ang_rad)
        sinLA = np.sin(ang_rad)
        x_center = np.average(x)
        y_center = np.average(y)

        N,M=x.shape
        new_x, new_y = np.mgrid[:N*OverSampling,:M*OverSampling]

        X = (new_x.astype(np.float64) - x_0*OverSampling)/OverSampling
        Y = (new_y.astype(np.float64) - y_0*OverSampling)/OverSampling
        rLensX=(X)*cosLA-(Y)*sinLA
        rLensY=(X)*sinLA+(Y)*cosLA
        if shear_factor<1:
            rLensX *= (1-lensKappa-lensGamma)
            rLensY *= (1-lensKappa+lensGamma)
        else:
            rLensX *= (1-lensKappa+lensGamma)
            rLensY *= (1-lensKappa-lensGamma)
        dmatLens = np.sqrt(rLensX*rLensX + rLensY*rLensY)


        Pa_rad = np.radians(theta)
        rLensGalX=(rLensX)*np.cos(Pa_rad)-(rLensY)*np.sin(Pa_rad)
        rLensGalY=(rLensX)*np.sin(Pa_rad)+(rLensY)*np.cos(Pa_rad)
        dmatGal = np.sqrt(rLensGalX*rLensGalX+(1/(axratio*axratio))*rLensGalY*rLensGalY)
        dmatGal += 1e-5 ## avoid r=0

        k=kappa(n)
        Profile = I_eff * np.exp(-k*((dmatGal/r_eff)**(1/n) - 1))

        if OverSampling != 1:
            Profile = Profile.reshape(x.shape[0],OverSampling,\
                      x.shape[1],OverSampling).sum(3).sum(1)/(OverSampling**2)


        if self.psf is not None:
            Profile = fftconvolve(Profile,self.psf,mode="same")

        return Profile

    @property
    def mag(self):
        totalFlux = total_flux(self.I_eff,self.r_eff,self.n)
        mag = -2.5 * np.log10(totalFlux/self.exptime) + self.magZP
        return mag

    @staticmethod
    def fit_deriv(x, y, x_0, y_0, I_eff, r_eff, n, axratio, theta):
        """ Sersic index derivative with respect to parameters
        """

        ## auxiliary definitions
        k = np.float64(kappa(n))
        T = np.float64(np.radians(theta))
        cosT = np.float64(np.cos(T))
        sinT = np.float64(np.sin(T))
        xC =  np.float64((x-x_0)*cosT - (y-y_0)*sinT)
        yC =  np.float64((x-x_0)*sinT + (y-y_0)*cosT)

        x2 = xC * xC
        y2 = yC * yC
        q2 = np.float64(axratio ** 2)
        q3 = np.float64(axratio ** 3)

        r = np.float64((np.sqrt(x2 + y2/q2) + 1e-5)) ## avoid r=0

        rn = np.float64((r / r_eff) ** (1/n))
        coreFunc = np.float64(np.exp(-k * rn - 1))
        d_r = np.float64(I_eff * coreFunc * ( -(k * rn) / (n * r) ))

        ## parameter derivatives
        d_x0 = d_r * ( xC/r * (-cosT) + yC/ (q2 * r) * (-sinT) )
        d_y0 = d_r * ( xC/r * (+sinT) + yC/ (q2 * r) * (-cosT) )
        d_Ie = coreFunc
        d_re = I_eff * coreFunc * ( (k * rn) / (n * r_eff) )
        d_n = I_eff * coreFunc * ( k/n**2 * np.log(r/r_eff) * rn)
        d_q = d_r * ( -y2 / (r * q3) )
        d_theta = d_r * (xC * yC)/r * (1/q2 - 1)

        # print("===> DEBUG: d_x0",d_x0)
        # print("===> DEBUG: d_y0",d_y0)
        # print("===> DEBUG: d_Ie",d_Ie)
        # print("===> DEBUG: d_re",d_re)
        # print("===> DEBUG: d_n",d_n)
        # print("===> DEBUG: d_q",d_q)
        # print("===> DEBUG: d_theta",d_theta)

        return [d_x0, d_y0, d_Ie, d_re, d_n, d_q, d_theta]


class LensedSersicPSFModel(Fittable2DModel):

    x_0 = Parameter(default=0,min=-10)
    y_0 = Parameter(default=0,min=-10)
    I_eff = Parameter(default=1,min=0,max=1e5)
    r_eff = Parameter(default=1,min=0.1,max=100)
    n = Parameter(default=1,min=0.1,max=20)
    axratio = Parameter(default=1,min=0.1,max=1.0)
    theta = Parameter(default=0,min=-90,max=90)

    xPSF = Parameter(default=0,min=-10)
    yPSF = Parameter(default=0,min=-10)
    I_psf = Parameter(default=1,min=0,max=1e5)

    def __init__(self, x_0=x_0.default, y_0=y_0.default, I_eff=I_eff.default,\
                 r_eff=r_eff.default, n=n.default, axratio=axratio.default,\
                 theta=theta.default, xPSF = xPSF.default, yPSF = yPSF.default,\
                 I_psf=I_psf.default, psf=None, magZP = 0, exptime =1,\
                 lensPars=None, OverSampling = 5, **kwargs):

        if psf is None:
            raise ValueError("PSF value must be provided")

        self.psf = psf
        self.magZP = magZP
        self.exptime = exptime
        self.overSampling = np.int(OverSampling)
        assert self.overSampling >= 1, "OverSampling must be equal or greater than 1"

        if not isinstance(lensPars,LensPars):
            raise ValueError("lensPars must be a LensPars object.")
        else:
            self.lensPars = lensPars

        super().__init__(x_0, y_0, I_eff, r_eff, n, axratio, theta,\
                         xPSF, yPSF, I_psf, **kwargs)
        return None

    def evaluate(self, x, y, x_0, y_0, I_eff, r_eff, n, axratio, theta, xPSF, yPSF, I_psf):
        """Two dimensional Sersic profile function in a lensed plane with an
        added point source."""

        lensKappa = self.lensPars.kappa
        lensGamma = self.lensPars.gamma
        lensMu = self.lensPars.mu
        lensAngle = self.lensPars.shearAngle

        OverSampling = self.overSampling

        shear_factor = (1-lensKappa-lensGamma)/(1-lensKappa+lensGamma)
        ang_rad = np.radians(lensAngle)
        cosLA = np.cos(ang_rad)
        sinLA = np.sin(ang_rad)
        x_center = np.average(x)
        y_center = np.average(y)

        N,M=x.shape
        new_x, new_y = np.mgrid[:N*OverSampling,:M*OverSampling]

        X = (new_x.astype(np.float64) - x_0*OverSampling)/OverSampling
        Y = (new_y.astype(np.float64) - y_0*OverSampling)/OverSampling
        rLensX=(X)*cosLA-(Y)*sinLA
        rLensY=(X)*sinLA+(Y)*cosLA
        if shear_factor<1:
            rLensX *= (1-lensKappa-lensGamma)
            rLensY *= (1-lensKappa+lensGamma)
        else:
            rLensX *= (1-lensKappa+lensGamma)
            rLensY *= (1-lensKappa-lensGamma)
        dmatLens = np.sqrt(rLensX*rLensX + rLensY*rLensY)


        Pa_rad = np.radians(theta)
        rLensGalX=(rLensX)*np.cos(Pa_rad)-(rLensY)*np.sin(Pa_rad)
        rLensGalY=(rLensX)*np.sin(Pa_rad)+(rLensY)*np.cos(Pa_rad)
        dmatGal = np.sqrt(rLensGalX*rLensGalX+(1/(axratio*axratio))*rLensGalY*rLensGalY)
        dmatGal += 1e-5 ## avoid r=0

        k=kappa(n)
        Profile = I_eff * np.exp(-k*((dmatGal/r_eff)**(1/n) - 1))

        if OverSampling != 1:
            Profile = Profile.reshape(x.shape[0],OverSampling,\
                      x.shape[1],OverSampling).sum(3).sum(1)/(OverSampling**2)

        if self.psf is not None:
            ProfileCV = fftconvolve(Profile,self.psf,mode="same")


        psfModel = self.evaluate_PSF(x,y,xPSF,yPSF,I_psf)


        # print(f"I_eff={I_eff[0]:.3f},r_eff={r_eff[0]:.3f},\nProfile (pre-conv) = {np.amax(Profile)}\nProfile (post-conv) = {np.amax(ProfileCV)}")

        # fig,ax = mpl.subplots(1,6,figsize=(20,4))
        # fig.subplots_adjust(wspace=0)
        # ax[0].imshow(dmatLens)
        # ax[1].imshow(dmatGal)
        # ax[2].imshow(Profile)
        # ax[2].contour(Profile,levels=I_eff,colors="white")
        # ax[3].imshow(ProfileCV)
        # ax[4].imshow(psfModel)
        # ax[5].imshow(ProfileCV+psfModel)
        #
        # for eixo in ax:
        #     eixo.tick_params(labelleft="off",labelbottom="off")



        return ProfileCV + psfModel

    @property
    def mag(self):
        totalFlux = total_flux(self.I_eff,self.r_eff,self.n)
        mag = -2.5 * np.log10(totalFlux/self.exptime) + self.magZP
        return mag

    # @staticmethod
    # def fit_deriv(x, y, x_0, y_0, I_eff, r_eff, n, axratio, theta):
    #     """ Sersic index derivative with respect to parameters
    #     """
    #
    #     ## auxiliary definitions
    #     k = np.float64(kappa(n))
    #     T = np.float64(np.radians(theta))
    #     cosT = np.float64(np.cos(T))
    #     sinT = np.float64(np.sin(T))
    #     xC =  np.float64((x-x_0)*cosT - (y-y_0)*sinT)
    #     yC =  np.float64((x-x_0)*sinT + (y-y_0)*cosT)
    #
    #     x2 = xC * xC
    #     y2 = yC * yC
    #     q2 = np.float64(axratio ** 2)
    #     q3 = np.float64(axratio ** 3)
    #
    #     r = np.float64((np.sqrt(x2 + y2/q2) + 1e-5)) ## avoid r=0
    #
    #     rn = np.float64((r / r_eff) ** (1/n))
    #     coreFunc = np.float64(np.exp(-k * rn - 1))
    #     d_r = np.float64(I_eff * coreFunc * ( -(k * rn) / (n * r) ))
    #
    #     ## parameter derivatives
    #     d_x0 = d_r * ( xC/r * (-cosT) + yC/ (q2 * r) * (-sinT) )
    #     d_y0 = d_r * ( xC/r * (+sinT) + yC/ (q2 * r) * (-cosT) )
    #     d_Ie = coreFunc
    #     d_re = I_eff * coreFunc * ( (k * rn) / (n * r_eff) )
    #     d_n = I_eff * coreFunc * ( k/n**2 * np.log(r/r_eff) * rn)
    #     d_q = d_r * ( -y2 / (r * q3) )
    #     d_theta = d_r * (xC * yC)/r * (1/q2 - 1)
    #
    #     # print("===> DEBUG: d_x0",d_x0)
    #     # print("===> DEBUG: d_y0",d_y0)
    #     # print("===> DEBUG: d_Ie",d_Ie)
    #     # print("===> DEBUG: d_re",d_re)
    #     # print("===> DEBUG: d_n",d_n)
    #     # print("===> DEBUG: d_q",d_q)
    #     # print("===> DEBUG: d_theta",d_theta)
    #
    #     return [d_x0, d_y0, d_Ie, d_re, d_n, d_q, d_theta]

    def evaluate_PSF(self,x,y,x_0, y_0,I_0):

        canvas = np.zeros_like(x)
        Npsf,Mpsf = self.psf.shape
        N,M=x.shape

        if x.shape > self.psf.shape:
            canvas[:Npsf,:Mpsf] = self.psf
        else:
            diffN = Npsf - N
            diffM = Npsf - N
            canvas = self.psf[diffN//2:-diffN//2,diffM//2:-diffM//2]


        xCen,yCen = barycenter(canvas,np.ones_like(canvas))
        dx = (x_0-xCen)
        dy = (y_0-yCen)
        finalPSF = shift(canvas,(dx,dy),order=2)
        return I_0*finalPSF



def generate_lensed_sersic_model(shape,modelPars,lensingPars,mag_zeropoint,exposure_time,psf=None,OverSampling = 3,debug=False):
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


    if OverSampling <1:
        raise ValueError
    else:
        OverSampling = int(OverSampling) ## Needs to be integer


    xc,yc,mag,radius,sersic_index,axis_ratio,position_angle = modelPars
    lensKappa,lensGamma,lensMu,lensAngle = lensingPars

    majAxis_Factor = 1/np.abs(1-lensKappa-lensGamma)
    shear_factor = (1-lensKappa-lensGamma)/(1-lensKappa+lensGamma)

    ang_rad = np.radians(lensAngle)
    X,Y = np.meshgrid(np.arange(shape[0]*OverSampling),np.arange(shape[1]*OverSampling))
    X = (X.astype(np.float64) - xc*OverSampling)/OverSampling
    Y = (Y.astype(np.float64) - yc*OverSampling)/OverSampling
    rLensX=(X)*np.cos(ang_rad)-(Y)*np.sin(ang_rad)
    rLensY=(X)*np.sin(ang_rad)+(Y)*np.cos(ang_rad)
    if shear_factor<1:
        rLensX *= (1-lensKappa-lensGamma)
        rLensY *= (1-lensKappa+lensGamma)
    else:
        rLensX *= (1-lensKappa+lensGamma)
        rLensY *= (1-lensKappa-lensGamma)
    dmatLens = np.sqrt(rLensX*rLensX + rLensY*rLensY)


    Pa_rad = np.radians(position_angle)
    rLensGalX=(rLensX-x_0)*np.cos(Pa_rad)-(rLensY-y_0)*np.sin(Pa_rad)
    rLensGalY=(rLensX-x_0)*np.sin(Pa_rad)+(rLensY-y_0)*np.cos(Pa_rad)
    dmatGal = np.sqrt(rLensGalX*rLensGalX+(1/(axis_ratio*axis_ratio))*rLensGalY*rLensGalY)

    totalFlux = exposure_time * 10**( -0.4*(mag-2.5*np.log10(lensMu)-mag_zeropoint) )
    Sigma_e = simulation.effective_intensity(totalFlux,radius,sersic_index)
    Profile = simulation.sersic(dmatGal,Sigma_e,radius,sersic_index)

    if debug is True:
        from .ADD import surface_brightness_profile
        print("magnification",lensMu,"shear",shear_factor)
        # print(X.ravel())
        print("angle",ang_rad)
        print("magFactors - x,y:",(1-lensKappa-lensGamma),(1-lensKappa+lensGamma))
        # fig,ax = mpl.subplots()
        # R,F = surface_brightness_profile(-dmatLens,np.ones_like(dmatLens),rmax=X.shape[0])
        # ax.plot(R,F)
        #
        # fig,ax = mpl.subplots(1,3,sharex=True,sharey=True,figsize=(20,8))
        # fig.subplots_adjust(wspace=0)
        # ax[0].imshow(-dmatLens)
        # ax[1].imshow(-dmatGal)
        # ax[2].imshow(Profile)
        # ax[1].set_title("bruno")

    if OverSampling != 1:
        return Profile.reshape(shape[0],OverSampling,\
                               shape[1],OverSampling).sum(3).sum(1)/(OverSampling**2)
    else:
        return Profile
