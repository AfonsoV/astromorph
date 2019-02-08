import numpy as np
import matplotlib
import matplotlib.pyplot as mpl
import matplotlib.ticker as mpt
from astromorph import utils
from astromorph import plot_utils as putils
from astromorph.Galaxy import Galaxy
from astromorph import simulation
import astromorph.cosmology as cosmos
import scipy.ndimage as snd
import h5py
import plotResults
import sys

mpl.style.use("presentationDark")

from astropy.convolution import convolve_fft
from scipy.signal import fftconvolve

import argparse
import configparser

from astromorph.ADD import surface_brightness_profile

class Parm:
  def __init__(self,x,y,mag,rad,ba,pa):
    self.x = x
    self.y = y
    self.mag = mag
    self.rad = rad
    self.ba = ba
    self.pa = pa

def MakeImage(Parm,dimx,dimy,palens,majmagnif,minmagnif,psf):
    import math
    indy,indx = np.indices((10*dimy,10*dimx),np.float32)
    indx = (indx-10*Parm.x)/10.
    indy = (indy-10*Parm.y)/10.

    # print(indx.ravel())
    print("angle",math.pi*palens/180.)
    print("magFactors - x,y:",1/majmagnif,1/minmagnif)
    nindx = indx * math.cos(math.pi*palens/180.) - indy * math.sin(math.pi*palens/180.)
    nindy = indx * math.sin(math.pi*palens/180.) + indy * math.cos(math.pi*palens/180.)
    nindx /= majmagnif
    nindy /= minmagnif
    indx = nindx * math.cos(math.pi*Parm.pa/180.) - nindy * math.sin(math.pi*Parm.pa/180.)
    indy = nindx * math.sin(math.pi*Parm.pa/180.) + nindy * math.cos(math.pi*Parm.pa/180.)
    dist = ((indx/Parm.rad)**2+(indy/Parm.ba/Parm.rad)**2)**0.5
    img = Parm.mag * np.exp(-dist)
    nimg = np.sum(np.sum(np.resize(img,(dimy,10,dimx,10)),3),1)/100
    nimg = fftconvolve(nimg,psf,mode="same")
    #  i=1/0

    # fig,ax = mpl.subplots()
    # R,F = surface_brightness_profile(-np.sqrt(nindx*nindx+nindy*nindy),np.ones_like(nindx),rmax=10*dimx)
    # ax.plot(R,F)
    #
    # fig,ax = mpl.subplots(1,3,sharex=True,sharey=True,figsize=(20,8))
    # fig.subplots_adjust(wspace=0)
    # ax[0].imshow(-np.sqrt(nindx*nindx+nindy*nindy))
    # ax[1].imshow(-np.sqrt(indx*indx+(1/Parm.ba)**2*indy*indy))
    # ax[2].imshow(img)
    # ax[1].set_title("bouwens")
    return nimg


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description = """Main script to fit galaxy
    morphology using MCMC. It takes a pre-formatted hdf5 file containing all
    data relevant to the fit. It is possible to run on individual subsets.""")
    parser.add_argument("configfile",metavar='NAME',type=str)
    parser.add_argument('-c','--clusters',default="",metavar='LIST',type=str)
    parser.add_argument('-i','--indices',default="",type=str,metavar='SET')
    parser.add_argument('--name',default="",type=str,metavar='ID')
    parser.add_argument('-n','--nrun',default=1000,type=int,metavar='NRUN')
    parser.add_argument('-e','--nexclude',default=900,type=int,metavar='NRUN')
    parser.add_argument('--nchain',default=160,type=int,metavar='NRUN')
    parser.add_argument("-P","--plot", action='store_true')
    parser.add_argument("-S","--show", action='store_true')
    parser.add_argument("-L","--nolensing", action="store_true")
    parser.add_argument("-N","--sersicfree", action="store_true")

    args = parser.parse_args()


    ConfigFile = configparser.ConfigParser()
    ConfigFile.read(args.configfile)

    PIXSCALE = float(ConfigFile["datapars"]["pixelscale"])
    # MAGZP = float(ConfigFile["datapars"]["magzeropoint"])
    # EXPTIME = float(ConfigFile["datapars"]["exptime"])
    MASK_RADIUS = float(ConfigFile["datapars"]["mask_radius"])
    # EDGE = int(float(ConfigFile["datapars"]["edgetrim"])/PIXSCALE)
    # GAIN =  float(ConfigFile["datapars"]["gain"])

    if len(args.indices)>0:
        Indexs = args.indices.split(":")
        SUBSET=True
        print("Running:","-".join(Indexs))
    else:
        print("Running all galaxies")
        SUBSET=False

    if len(args.clusters)>0:
        Clusters = args.clusters.split(",")
        print("Runing for clusters: %s"%(", ".join(Clusters)))
    else:
        Clusters = "ALL"
        print("running all clusters")

    PLOT = args.plot
    SHOW_DATA = args.show
    LENS = "_wLensing"
    if args.nolensing is True:
        withLensing = False
        LENS = ""
    else:
        withLensing = True
        LENS = "_wLensing"
    if Clusters!="ALL":
        cluster="_".join(Clusters)
    else:
        cluster = Clusters

    scale="%imas"%(1000*PIXSCALE)

    if SUBSET is True:
        SUFFIX = LENS + "_".join(Indexs)
    else:
        SUFFIX = LENS

    if args.sersicfree is True:
        SUFFIX += "_NFREE"
    else:
        SUFFIX += "_NFIXED"



    # fin = h5py.File("%s_%s_galaxyData.hdf5"%(catname.split(".")[0],scale),"r")
    # fin = h5py.File("A370_275C_abell370_30mas_galaxyData.hdf5","r")
    fin = h5py.File(ConfigFile["data"]["datafile"],"r")

    print("%s_%s_%s_galaxyResults%s.hdf5"%(ConfigFile["data"]["catalog"],scale,cluster,SUFFIX))
    print("%s_mcmcResults_%s%s.txt"%(ConfigFile["data"]["catalog"],cluster,SUFFIX))
    if PLOT is False and SHOW_DATA is False:
        fout = h5py.File("%s_%s_%s_galaxyResults%s.hdf5"%(ConfigFile["data"]["catalog"],scale,cluster,SUFFIX),"w")
        resultsTable = open("%s_mcmcResults_%s%s.txt"%(ConfigFile["data"]["catalog"],cluster,SUFFIX),"w")
        pars = ["xc","yc","mag","re","q","theta","sky"]
        header = "# Name ra dec z mu"
        for p in pars:
            header += " %s %s_errLow %s_errUp "%(p,p,p)
        header += "\n"
        resultsTable.write(header)

    nGals = len(fin)
    nMontecarlo = args.nrun
    nChain = args.nchain
    nExclude = args.nexclude
    for i,galaxySet in enumerate(fin):
        if args.name != "":
            if galaxySet!= args.name:
                continue
        # if galaxySet != "A370336-9522935171": #A370336-9546534134
        #     continue
        print(i+1,galaxySet)
        if not True in [cluster in galaxySet for cluster in Clusters] and Clusters!="ALL":
            continue
        if SUBSET is True:
            if i < int(Indexs[0]) or i > int(Indexs[1]):
                continue
        # if i != 3:
        #     continue
        galImage = fin["%s/galaxy"%(galaxySet)].value
        galMask = fin["%s/mask"%(galaxySet)].value
        galSegmap = fin["%s/labels"%(galaxySet)].value
        galSigma = fin["%s/sigma"%(galaxySet)].value
        galPSF = fin["%s/psf"%(galaxySet)].value
        exptime = fin["/%s"%(galaxySet)].attrs["exptime"]
        mag_zp = fin["/%s"%(galaxySet)].attrs["magzp"]

        if SHOW_DATA is True:
            print("EXPTIME = %.2f"%(exptime))
            print("MAG ZP = %.5f"%(mag_zp))

        if galImage[galImage!=0].size < galImage.size/1.1:
            print("No image coverage for this galaxy. Skipping")
            continue

        sky_level,sky_rms = utils.sky_value(galImage)
        # print(sky_level,sky_rms)
        # galSigma = np.ones_like(galImage)

        detMap = galSegmap.copy()
        detMap[detMap>0]=1
        objectMask = detMap - galMask
        objectMask[objectMask<0] = 0


        galaxyObject = Galaxy(imgdata=galImage)
        galaxyObject.set_segmentation_mask(galMask)
        galaxyObject.set_psf(psfdata = galPSF/np.sum(galPSF),resize=False)
        galaxyObject.set_sigma(galSigma)
        galaxyObject.set_object_mask(objectMask)
        masked_image = np.ma.masked_array(galImage,mask=galMask)



        initPars = list(galaxyObject.estimate_parameters(mag_zp,exptime,PIXSCALE,MASK_RADIUS)) +[sky_level] ## + [sky_level,0,0]

        if withLensing is True:
            mu = fin["/%s"%(galaxySet)].attrs["lensMagnification"]
            kappa = fin["/%s"%(galaxySet)].attrs["lensConvergence"]
            gamma = fin["/%s"%(galaxySet)].attrs["lensShear"]
            shearAngle = -fin["/%s"%(galaxySet)].attrs["lensShearAngle"]

            # mu = 2.2
            # shear = 2.1
            #
            # gamma = 1/(2*shear*mu) - shear/2
            # kappa = 1 - np.sqrt(1/(2*mu) + 1/(4*mu**2*shear**2) +shear**2/4)
            lensPars = (kappa,gamma,mu,shearAngle)

            initPars[2] += 2.5 * np.log10(mu)
            initPars[3] *= np.abs(1-kappa-gamma)
        else:
            lensPars = None
            mu = -99
        # print("LENS PARS:",lensPars)
        # print(initPars)
        #
        # # magnitudes = galaxyPars[2,nMontecarlo//2:]
        # radius = 0.662/(PIXSCALE*da/(180./np.pi*3600.)*1000)
        # print("B_SIZE",radius)
        # # sersic = galaxyPars[4,nMontecarlo//2:]

        if SHOW_DATA is True:
            print("Guess Pars",initPars)
            N,M=galImage.shape
            if lensPars is not None:

                mag,re,n,q,t = 24,3.0,1.0,1.0,0.0
                model1 = simulation.generate_lensed_sersic_model(galImage.shape,\
                        (N/2,M/2,mag,re,n,q,t),lensPars,26,1.0,OverSampling=10,debug=True)

                totalFlux = exptime * 10**( -0.4*(mag-mag_zp) )
                Ie = simulation.effective_intensity(totalFlux,re,n)
                b = simulation.kappa(n)
                rd = re/b
                I0 = Ie * np.exp(b)
                Param = Parm(N/2,M/2,I0,rd,q,t)
                model2 = MakeImage(Param,N,M,shearAngle,1/(1-kappa-gamma),\
                                   1/(1-kappa+gamma),galPSF)
            else:
                model1 = simulation.generate_sersic_model(galImage.shape,\
                        (N/2,M/2,24,5.0,1.0,1.0,0.0),26,1.0)


            fig,ax = mpl.subplots(2,1,figsize = (8,10))
            fig.subplots_adjust(hspace=0)
            ax[0].imshow(galImage)
            ax[1].imshow(masked_image)
            ax[0].set_ylabel("Original")
            ax[1].set_ylabel("with Masking")
            for eixo in ax:
                eixo.tick_params(labelleft=False,labelbottom=False)
            # fig.savefig("debugImages/%s_data.png"%(galaxySet))

            fig,ax=mpl.subplots(1,6,figsize=(22,4))
            fig.subplots_adjust(wspace=0)
            ax[0].imshow(galImage)
            ax[1].imshow(masked_image)
            ax[2].imshow(galMask)
            ax[3].imshow(objectMask)
            ax[4].imshow(galSigma)
            # ax[4].imshow(model2)
            ax[5].imshow(fftconvolve(model1,galPSF,mode="same"))
            break

        # galaxyParsMC = galaxyObject.fit(initPars[:4]+[1.0]+initPars[4:],mag_zp,exptime,nRun = nMontecarlo)
        if args.sersicfree is False:
            galaxyParsMC = galaxyObject.emcee_fit(initPars,mag_zp,exptime,\
                                              lensingPars=lensPars,\
                                              nchain = nChain,\
                                              nsamples = nMontecarlo,\
                                              nexclude = nExclude,\
                                              plot=PLOT,threads = 3,\
                                              ntemps=None)
        else:
            galaxyParsMC = galaxyObject.emcee_fit_MP(initPars[:4]+[1.0]+initPars[4:],mag_zp,exptime,\
                                              lensingPars=lensPars,\
                                              nchain = nChain,\
                                              nsamples = nMontecarlo,\
                                              nexclude = nExclude,\
                                              plot=PLOT,threads = 3,\
                                              ntemps=None)

        da = cosmos.angular_distance(fin["/%s"%(galaxySet)].attrs["z"])
        kn=0
        xc = galaxyParsMC[0,:]
        yc = galaxyParsMC[1,:]
        magnitudes = galaxyParsMC[2,:]
        radius = galaxyParsMC[3,:]*PIXSCALE*da/(180./np.pi*3600.)*1000.
        q = galaxyParsMC[4+kn,:]
        theta = galaxyParsMC[5+kn,:]

        if PLOT is True:
        # if True:
            fig,ax = plotResults.corner_plot_wImage(i+1,galImage*(1-galMask),PIXSCALE,[magnitudes,radius]\
                                            ,[r'$mag$',r'$r_e\ [\mathrm{kpc}]$'],redshift=fin["/%s"%(galaxySet)].attrs["z"])
            # fig.savefig("debugResults/Fit_CoVarSimple_%s.png"%(galaxySet))

            fig,ax = plotResults.corner_plot([xc,yc,magnitudes,radius,q,theta]\
                                ,[r"$x_c$",r"$y_c$",r'$mag$',\
                                r'$r_e\ [\mathrm{kpc}]$',r"$(b/a)$",\
                                r"$\theta_\mathrm{PA}$"])
            # fig.savefig("debugResults/Fit_CoVarFull_%s.png"%(galaxySet))

            bestModelPars = list(np.median(galaxyParsMC,axis=-1))
            if withLensing is True:
                model = simulation.generate_lensed_sersic_model(galImage.shape,\
                                # bestModelPars[:4]+bestModelPars[4:],\
                                bestModelPars[:4]+[1.0]+bestModelPars[4:6],\
                                lensPars,mag_zp,exptime)
            else:
                model = simulation.generate_sersic_model(galImage.shape,\
                                # bestModelPars[:4]+bestModelPars[4:],\
                                bestModelPars[:4]+[1.0]+bestModelPars[4:6],\
                                mag_zp,exptime)
            N,M = model.shape
            x,y = np.meshgrid(range(N),range(M))
            modelSP = fftconvolve(model,galaxyObject.psf,mode="same")\
                    + bestModelPars[-1] #+ bestModelPars[-2]*(x-N/2) + bestModelPars[-1]*(y-M/2)


            fig,ax=mpl.subplots(1,3,figsize=(16,6))
            fig.subplots_adjust(wspace=0)
            ax[0].imshow(galImage*(1-galMask))
            ax[1].imshow(modelSP)
            ax[2].imshow((galImage-modelSP)*(1-galMask))
            for eixo in ax:
                eixo.set_xticks([])
                eixo.set_yticks([])
            # fig.savefig("debugResults/Fit_Model_%s.png"%(galaxySet))
            print("Best Fit Pars",bestModelPars)
            # mpl.close("all")
            break

        results = np.percentile(galaxyParsMC,[16,50,84],axis=-1)
        errors = results[1:,:]-results[:-1,:]
        # print(results.shape)
        # print(errors.shape)

        if PLOT is False and SHOW_DATA is False:
            fout.create_dataset("%s"%(galaxySet),data=galaxyParsMC)

            lineResults = "%s"%(galaxySet)
            ra,dec,z = fin["/%s"%(galaxySet)].attrs["ra"],\
                       fin["/%s"%(galaxySet)].attrs["dec"],\
                       fin["/%s"%(galaxySet)].attrs["z"]
            lineResults += "\t%12.8f\t%12.8f\t%4.2f"%(ra,dec,z)
            lineResults += "\t%4.2f"%(mu)
            for i in range(galaxyParsMC.shape[0]):
                lineResults += "\t%10.6f\t%10.6f\t%10.6f"%(results[1,i],errors[0,i],errors[1,i])
            lineResults +="\n"
            print(lineResults)
            resultsTable.write(lineResults)
        elif PLOT is True:
            lineResults = "%s"%(galaxySet)
            ra,dec,z = fin["/%s"%(galaxySet)].attrs["ra"],\
                       fin["/%s"%(galaxySet)].attrs["dec"],\
                       fin["/%s"%(galaxySet)].attrs["z"]
            lineResults += "\t%12.8f\t%12.8f\t%4.2f"%(ra,dec,z)
            lineResults += "\t%4.2f"%(mu)
            for i in range(galaxyParsMC.shape[0]):
                lineResults += "\t%10.6f\t%10.6f\t%10.6f"%(results[1,i],errors[0,i],errors[1,i])
            lineResults +="\n"
            print(lineResults)


    fin.close()
    if PLOT is False and SHOW_DATA is False:
        fout.close()
        resultsTable.close()
    else:
        mpl.show()
