import matplotlib.pyplot as mpl
import numpy as np
from . import utils
from . import plot_utils as putils

class Galaxy(object):

    def __init__(self,imgname,coords):
        self.original_name = imgname
        self.coords = coords
        self.cutout = None
        self.psf = None

    def draw_cutout(self,size,pixelscale,eixo=None,**kwargs):
        if self.cutout is None:
            data = get_cutout(self.original_name,self.coords,size,pixelscale)
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

    def set_bounding_box(self,size,pixelscale):
        self.bounding_box = get_bounding_box(self.original_name,self.coords,size,pixelscale)
        self.cutout = get_cutout(self.original_name,self.coords,size,pixelscale)
        return None

    def set_psf(self,filename,resize=None):
        self.psf = pyfits.getdata(filename)
        if self.psf.shape[0]%2==0:
            self.psf = np.pad(self.psf,((0,1),(0,1)),mode="edge")
            self.psf = snd.shift(self.psf,+0.5)

        if resize is not None:
            self.psf = self.psf[50:151,50:151]
        return None

    def fit(self,initPars,lensingPars,mag_zeropoint,exposure_time,oversampling=3):
        xc,yc,mag,radius,sersic_index,axis_ratio,position_angle = initPars
        lensKappa,lensGamma,lensMu = lensingPars

        if self.cutout is None:
            raise ValueError("No cutout for this galaxy is defined")
        else:
            N,M=self.cutout.shape
            majAxis_Factor = 1/np.abs(1-lensKappa-lensGamma)
            shear_factor = (1-lensKappa-lensGamma)/(1-lensKappa+lensGamma)

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
