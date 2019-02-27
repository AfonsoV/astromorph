import sys
import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as mpl
import matplotlib.ticker as mpt
import matplotlib.cm as cm
import matplotlib.patches as mpp
import astropy.io.fits as pyfits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astromorph import utils, galfit,simulation, lensing
from astromorph import plot_utils as putils
from astromorph.lensing import LensingModel
from astromorph.Galaxy import Galaxy
from astromorph.clumps import ClumpFinder
import astromorph.cosmology as cosmos
import scipy.ndimage as snd
import scipy.interpolate as sip
import configparser
import h5py

from astropy.modeling import models, fitting

# from easyOutput import eazyResults

def create_sigma_image(img,gain,ncombine=1):

    img_to_electrons = img*gain*ncombine
    sky_med,sky_std = utils.sky_value(img_to_electrons)

    pre_sigma = np.sqrt(img_to_electrons*img_to_electrons+sky_std*sky_std)
    sigma = np.sqrt(pre_sigma/gain)
    return sigma

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
                raModel = np.linspace(coords.ra.value-size/(2*3600.),\
                                      coords.ra.value+size/(2*3600.),xu-xl)
                decModel = np.linspace(coords.dec.value-size/(2*3600.),\
                                       coords.dec.value+size/(2*3600.),yu-yl)

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
                modelInterpolator = sip.interp2d(raModel,decModel,modelVariable,\
                                        bounds_error=False,fill_value=np.nan)
                modelGrid[k,:,:,i] = modelInterpolator(raGrid,decGrid)[:,::-1]

    # print(modelGrid.shape)
    # return np.nanmedian(modelGrid,axis=-1)
    return np.nanpercentile(modelGrid,[16,50,84],axis=-1)

def stack_images(images,raExtent,decExtent,size,weights=None,norm=None):
    raGrid = np.linspace(raExtent[0],raExtent[1],size)
    decGrid = np.linspace(decExtent[0],decExtent[1],size)
    modelGrid = np.zeros([raGrid.size,decGrid.size,len(images)])
    if weights is not None:
        modelWeights = np.zeros([raGrid.size,decGrid.size,len(images)])


    for i in range(len(images)):
        imageExtent = images[i].get_bounding_box_coordinates()
        raModel = np.linspace(imageExtent[0],imageExtent[1],images[i].cutout.shape[1])
        decModel = np.linspace(imageExtent[2],imageExtent[3],images[i].cutout.shape[0])
        modelInterpolator = sip.interp2d(raModel,decModel,images[i].cutout,bounds_error=False,fill_value=np.nan)

        if weights is not None and len(weights)==len(images):
            weightsInterpolator = sip.interp2d(raModel,decModel,weights[i].cutout,bounds_error=False,fill_value=np.nan)
            modelWeights[:,:,i] = weightsInterpolator(raGrid,decGrid)[:,::-1]
        elif weights is not None:
            raise ValueError("Invalid size for weight list. Must be the same as input images")

        modelGrid[:,:,i] = modelInterpolator(raGrid,decGrid)[:,::-1]
        modelGrid[modelGrid==0]=np.nan

        if norm is not None and len(norm)==len(images):
            modelGrid[:,:,i] *= norm[i]
        elif norm is not None:
            raise ValueError("Invalid size for norm list. Must be the same as input images")

    if weights is None:
        return np.nanmedian(modelGrid,axis=-1)
    else:
        return np.nansum(modelGrid*modelWeights,axis=-1)/np.nansum(modelWeights,axis=-1)


def getRegionData(images,filters,segmap,weights=None,debugRegion=None):

    nregs = np.amax(segmap)

    mask = segmap.copy()
    mask[segmap>0]=1



    img_xc,img_yc = segmap.shape[1]/2,segmap.shape[0]/2

    RegionData = {}
    for region in range(1,nregs+1):
        RegionData[region] = {}
        RegionData[region]["size"] = segmap[segmap==region].size

        regionMask = np.zeros(segmap.shape)
        regionMask[segmap==region]=1

        xc,yc=utils.barycenter(segmap,regionMask)
        RegionData[region]["center"] = (xc,yc)

        RegionData[region]["dist"] = np.sqrt( (xc-img_xc)**2 + (yc-img_yc)**2 )


        for k in range(len(filters)):
            imgData = images[filters[k]].cutout
            _,skyrms = utils.sky_value(imgData)

            if debugRegion is not None:
                if region == debugRegion:
                    fig,ax = mpl.subplots(1,3,figsize=(16,8))
                    ax[0].imshow(imgData)

                    maskRegion = segmap.copy()
                    maskRegion[segmap!=region]=0
                    ax[1].imshow(imgData*maskRegion)

                    ax[2].imshow(imgData,cmap="gray_r")

                    maskedMap = np.ma.masked_array(segmap,mask=(segmap==0))
                    ax[2].imshow(maskedMap,alpha=0.1,cmap=cm.rainbow)

                    fig.suptitle(filters[k])

            if weights is not None:
                weightData = weights[filters[k]].cutout
            else:
                weightData = np.ones_like(imgdata)

            flux = np.nansum(imgData[segmap==region]).astype(np.float64)
            fluxerr = np.sqrt(imgData[segmap==region].size*skyrms**2)
            mag = -2.5*np.log10(flux) + zeropoints[filters[k]]
            magerr = 2.5*fluxerr/flux/np.log(10)

            if np.isnan(mag):
                mag=32
            RegionData[region][filters[k]] = {}
            RegionData[region][filters[k]]["flux"] = flux
            RegionData[region][filters[k]]["fluxerr"] = fluxerr
            RegionData[region][filters[k]]["mag"] = mag
            RegionData[region][filters[k]]["magerr"] = magerr

    return RegionData


def select_regions(regionData,color1,color2,sdist = 50, cdist = 0.25,debug=False):

    if debug:
        print(len(regionData))
    centers = np.zeros([len(regionData),2])
    distances = np.zeros(len(regionData))
    colors1 = np.zeros(len(regionData))
    colors2 = np.zeros(len(regionData))
    err1 = np.zeros(len(regionData))
    err2 = np.zeros(len(regionData))
    for region,data in regionData.items():
        centers[region-1,:] = data["center"]
        distances[region-1] = data["dist"]
        colors1[region-1] = data[color1[0]]["mag"]-data[color1[1]]["mag"]
        err1[region-1] = np.sqrt(data[color1[0]]["magerr"]**2+data[color1[1]]["magerr"]**2)
        colors2[region-1] = data[color2[0]]["mag"]-data[color2[1]]["mag"]
        err2[region-1] = np.sqrt(data[color2[0]]["magerr"]**2+data[color2[1]]["magerr"]**2)

    indexs = [np.argmin(distances)]
    indexs_visited = []
    if debug:
        print("Start search in: region %i"%(indexs[0]+1))

    k=0
    while True:
        if debug:
            print("Start search")
        NewIndices = False
        for i in indexs:

            if i in indexs_visited:
                continue
            else:
                indexs_visited.append(i)
                NewIndices = True
            spatial_distances = np.sqrt( (centers[:,0]-centers[i,0])**2 + (centers[:,1]-centers[i,1])**2 )
            # color_distances = np.sqrt( (colors1[:]-colors1[i])**2 + (colors2[:]-colors2[i])**2 )
            color_distances = (colors1[:]-colors1[i])**2/err1[i]**2 +\
                              (colors2[:]-colors2[i])**2/err2[i]**2


            new_indices = np.argwhere((spatial_distances<sdist)*\
                                      (color_distances<1)*\
                                      (spatial_distances!=0))
            indexs = np.unique(np.append(indexs,new_indices.ravel()))
            if debug:
                print("Searching around region %i"%(i+1))
                print("Color1", colors1[i], "Color2", colors2[i])
                print("Err1", err1[i], "Err2", err2[i])
                print("SPA",spatial_distances)
                print("COL",color_distances)
                print("New matches", new_indices.ravel() + 1)
                print("All matches", indexs + 1, "Matches found previously",indexs_visited)

            # break
        if debug:
            print("New indices -",NewIndices)

        k+=1
        if NewIndices is False:
            break
    return indexs+1

def write_magnification_map(fname,data,raExtent,decExtent):
    fout = h5py.File(fname,"w")/macs1149clu_misc/star_psfs/macs1149_bcgs_out_f105w_psf_69p.fits
    dataset = fout.create_dataset("magnification",data=data[2,:,:])
    dataset = fout.create_dataset("shear",data=data[1,:,:])
    dataset = fout.create_dataset("convergence",data=data[0,:,:])
    dataset.attrs["raLimits"]=raExtent
    dataset.attrs["decLimits"]=decExtent
    fout.close()
    return None



def stack_psf(cluster,filters,weights):

    if cluster =="abells1063":
        cluster = "abell1063"

    nSide = 69
    psfCanvas = np.zeros([nSide,nSide])

    sumWeights = 0
    for i,f in enumerate(filters):
        psfname = "%s/%sclu_misc/star_psfs/%s_bcgs_out_%s_psf_69p.fits"%(rootDataFolder,cluster,cluster,f)
        psfdata = pyfits.getdata(psfname)
        weight = np.nanmedian(weights[f].cutout)
        psfCanvas += psfdata*weight
        sumWeights += weight

    return psfCanvas/sumWeights

def read_catalog(fname):
    return np.loadtxt(fname, dtype = {"names":("Name","RA","DEC","mag"),\
                                      "formats":("U50","U50","U50","f4")})

def pickEvent(event,tableMain,tableZphot,redshift):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind

    idx = np.where(tableMain["ra"]==xdata[ind])[0][0]

    for eixo in axPhot[1:]:
        eixo.cla()

    selDot.set_xdata(xdata[ind])
    selDot.set_ydata(ydata[ind])
    tableZphot.plot_sed(idx,axPhot[1])
    tableZphot.plot_zprob(idx,axPhot[2])
    figPhot.suptitle(r"$z_\mathrm{udrop}$=%.2f"%(redshift))

    figPhot.canvas.draw_idle()
    return None

def find_all_models(dataFolder,rejectList=[]):
    allFiles = glob.glob(f"{dataFolder}/*/v*/*.fits")
    roots = []
    for f in allFiles:
        reject = False
        fname = f.split("/")[-1]
        rootName = "_".join(fname.split("_")[:6])
        for model in rejectList:
            if model in rootName:
                reject = True
                break
        if reject:
            continue
        if not rootName in roots:
            roots.append(rootName)
    return roots

def select_cluster(galName):
    root = galName.split("-")[0]
    if "A370" in root:
        return "abell370"
    elif "A2744" in root:
        return "abell2744"
    elif "AS1063" in root:
        return "abell1063"
    elif "M0416" in root:
        return "macs0416"
    elif "M0717" in root:
        return "macs0717"
    elif "M1149" in root:
        return "macs1149"
    else:
        raise ValueError(f"Invalid galaxy name {galName}:{root}. Cannot find associated cluster.")



if __name__== "__main__":

    SIZE = 4.5
    PIXSCALE = 0.06
    MAGZP = 26.0
    EXPTIME = 1
    MASK_RADIUS = 0.25
    EDGE = int(0.4/PIXSCALE)
    GAIN = 2.5
    PSFTRIM = 41

    catname = sys.argv[1]
    # rootDataFolder ="/data2/bribeiro/HubbleFrontierFields/deepspace"
    rootDataFolder="../../data/deepspace/"

    filters=["f275w","f336w","f435w","f606w","f814w","f105w","f125w","f140w","f160w"]
    filtersStack = ["f105w","f125w","f140w","f160w"]
    allClusters = ["abell370"]#,"abell2744","abell1063","macs0416","macs0717","macs1149"]


    # zphotCatalogs = { "%s"%(c):eazyResults("%s/%sclu_catalogs/eazy/%sclu_v3.9"%(rootDataFolder,c,c),\
    #                  "%sclu_v3.9"%(c)) for c in allClusters}

    photCatalogs = {"%s"%(c):Table.read("%s/%sclu_catalogs/hffds_%sclu_v3.9.cat"%(rootDataFolder,c,c),format="ascii")\
                     for c in allClusters}


    # parent_catalogs = {"A2744275":read_catalog("./A2744_275C.CAT"),\
    #                    "A2744336":read_catalog("./A2744_336C.CAT"),\
    #                    "A370275":read_catalog("./A370_275C.CAT"),\
    #                    "A370336":read_catalog("./A370_336C.CAT"),\
    #                    "AS1063275":read_catalog("./AS1063_275C.CAT"),\
    #                    "AS1063336":read_catalog("./AS1063_336C.CAT"),\
    #                    "M0416275":read_catalog("./M0416_275C.CAT"),\
    #                    "M0416336":read_catalog("./M0416_336C.CAT"),\
    #                    "M0717275":read_catalog("./M0717_275C.CAT"),\
    #                    "M0717336":read_catalog("./M0717_336C.CAT"),\
    #                    "M1149275":read_catalog("./M1149_275C.CAT"),\
    #                    "M1149336":read_catalog("./M1149_336C.CAT")}


    # clusters = {"A2744275":"abell2744",\
    #             "A2744336":"abell2744",\
    #             "A370275":"abell370",\
    #             "A370336":"abell370",\
    #             "AS1063275":"abell1063",\
    #             "AS1063336":"abell1063",\
    #             "M0416275":"macs0416",\
    #             "M0416336":"macs0416",\
    #             "M0717275":"macs0717",\
    #             "M0717336":"macs0717",\
    #             "M1149275":"macs1149",\
    #             "M1149336":"macs1149",\
    #             "M0717":"macs0717",\
    #             "M1149":"macs1149",\
    #             "M0416":"macs0416",\
    #             "A2744":"abell2744",\
    #             "A370":"abell370",\
    #             "AS1063":"abell1063"}

    allEpochs = ["","","","","",""]
    rejectModels = ["williams"]
    # allModels = ["cats","glafic","sharon"]
    # vGLAFIC={"abell2744":"v4","abell370":"v4","abells1063":"v4","macs0416":"v4",\
    #          "macs0717":"v3","macs1149":"v3"}
    # vCATS={"abell2744":"v4","abell370":"v4","abells1063":"v4","macs0416":"v4",\
    #          "macs0717":"v4","macs1149":"v4"}
    # vSHARON={"abell2744":"v4","abell370":"v4","abells1063":"v4","macs0416":"v4",\
    #          "macs0717":"v4","macs1149":"v4"}
    # versions = [vCATS,vGLAFIC,vSHARON]
    # nModels = len(allModels)
    # assert len(allModels)==len(versions),"Number of versions must match number of models"

    cameraInfo = {"f225w":"acs","f275w":"acs","f336w":"acs","f390w":"acs",\
                  "f435w":"acs","f814w":"acs","f606w":"acs",\
                  "f105w":"wfc3","f125w":"wfc3","f140w":"wfc3","f160w":"wfc3"}

    zeropoints = {"f225w":25.665,"f275w":25.665,"f336w":25.665,"f390w":25.665,\
                  "f435w":25.665, "f606w":26.493, "f814w":25.947,\
                  "f105w":26.2687, "f125w":26.2303, "f140w":24.4524,\
                  "f160w":25.9463}

    Colors = [("f606w","f814w"),\
              ("f105w","f125w")]
    allImageInfo = {}

    CLR_DIST = 0.15
    SPC_DIST = 1.5/PIXSCALE



    for cluster,epoch in zip(allClusters,allEpochs):
        imgfolder="%s/%sclu_misc/images/bcgs_out"%(rootDataFolder,cluster)
        imgnames = ["%s/%s_bcgs_out_%s_bkg_drz_masked.fits"%(imgfolder,\
                    cluster,f) for f in filters]
        # imgfolder="%s/%sclu_misc/images/psf_matched"%(rootDataFolder,cluster)
        # imgnames = ["%s/%s_bcgs_out_%s_psf_bkg_drz.fits"%(imgfolder,\
        #             cluster,f) for f in filters]

        # exptimes = ["%s/hlsp_frontier_hst_wfc3-%s_%s_%s_v1.0%s_exp.fits"%(imgfolder,\
        #                                                 scale,cluster,f,epoch) for f in filters]
        imgfolder="%s/%sclu_misc/images/bcgs_out"%(rootDataFolder,cluster)
        imgweights = ["%s/%s_bcgs_out_%s_wht_masked.fits"%(imgfolder,\
                      cluster,f) for f in filters]
        # imgrms = ["%s/hlsp_frontier_hst_wfc3-%s_%s_%s_v1.0%s_rms.fits"%(imgfolder,\
        #                                                 scale,cluster,f,epoch) for f in filters]


        if cluster == "abell1063":
            ### need to update name to match lensing models
            ### image names must be done first
            cluster = "abells1063"

        allImageInfo[cluster] = {}
        allImageInfo[cluster]["imageNames"] = imgnames
        # allImageInfo[cluster]["exposureNames"] = exptimes
        imageFilters = []
        weightsFilters = []
        rmsFilters = []
        for k in range(len(filters)):
            imageFilters.append(Galaxy(imgnames[k]))
            weightsFilters.append(Galaxy(imgweights[k]))
            # rmsFilters.append(Galaxy(imgrms[k]))
        allImageInfo[cluster]["imageData"] = imageFilters
        allImageInfo[cluster]["weightData"] = weightsFilters
        # allImageInfo[cluster]["rmsData"] = rmsFilters



        # lens_models_folder = "/data2/bribeiro/HubbleFrontierFields/LensModels/%s"%(cluster)
        lens_models_folder = "../../data/LensModels/%s"%(cluster)
        lensModelNames = find_all_models(lens_models_folder,rejectModels)
        nModels = len(lensModelNames)
        ConfigFile = configparser.ConfigParser()
        ConfigFile.read("%s/models.cfg"%(lens_models_folder))

        modelsLensing = []
        for i in range(nModels):

            modelName = lensModelNames[i].split("_")[-2]
            version = lensModelNames[i].split("_")[-1]
            redshiftLens = float(ConfigFile[modelName]["redshift"])
            resolution = ConfigFile[modelName]["resolution"]
            if len(resolution.split(","))==1:
                resolutionModel = float(ConfigFile[modelName]["resolution"])
            else:
                resolutionModel = float(ConfigFile[modelName]["resolution"].split(",")[0])

            lensModel = LensingModel(redshiftLens,resolutionModel)
            lensModel.set_lensing_data(filename=f"{lens_models_folder}/{modelName}/{version}/{lensModelNames[i]}")
            # lensModel.set_model_at_z(cat_reshift)
            modelExtent = lensModel.get_image_box_coordinates()
            modelsLensing.append(lensModel)

        allImageInfo[cluster]["lensModels"] = modelsLensing
        allImageInfo[cluster]["lensRedshift"] = redshiftLens


    # if catname == "udrop.cat":
    #     catalog = np.loadtxt(catname,dtype={"names":("ID","z","zconf"),"formats":("U50","f4","f4")})
    # else:
    #     catalog = np.loadtxt(catname,\
    #     dtype={"names":("ID","RA","DEC","mag","dummy1","dummy2","z","zerr"),\
    #     "formats":("U50","U50","U50","f4","f4","f4","f4","f4")})
    catalog = Table.read(catname,format="ascii")


    # testGals = ["A370336-9553634160",\
    #             "A370336-9512334561",\
    #             "A370336-9533834017",\
    #             "A370336-9539455699",\
    #             "A370336-9581434094",\
    #             "A370336-9505833405"]
    # testGals = ["A370336-9488645446","A370336-9541335531",\
    #            "A370336-9531734304","A370275-9495434273",\
    #            "A370275-9493243395","A370336-9576034044",\
    #            "A370275-9520535460","A370336-9484535039",\
    #            "A370336-9495444035","A370336-9489134421",\
    #            "A370336-9505833405"]
    #
    # testGals = ["A370336-9576034044",\
    #            "A370275-9520535460","A370336-9489134421"]

    # testGals = ["A370336-9541335531"]
    # testGals = ["A370336-9489134421"]
    # testGals = ["A370275-9520535460"]
    # testGals = ["A370336-9576034044"]
    # testGals = ["A370336-9541335531","A370275-9520535460"]


    # testGals = ["A370336-9496545794","A370336-9486443883","A370336-9531734304"]
    # testGals = ["A370336-9496545794"]
    # testGals = ["A370336-9486443883"]
    # testGals =["A370336-9531734304"]
    # testGals =["A370275-9506632951"]
    # testGals =["A370336-9476434519"]
    testGals = ["A370336-9505833405"]
    # testGals =["A370336-9505833405","A370336-9539455699"]
    # testGals =["A370336-9539455699"]


    nGals = len(catalog)
    for num in range(nGals):

        galName = catalog["Name"][num]
        if galName not in testGals and len(testGals)>0:
            continue

        if not os.path.isdir(galName):
            os.mkdir(galName)

        fout = h5py.File(f"{galName}/{galName}_deepspace_galaxyData.hdf5","w")


        galRedshift = catalog["redshift"][num]
        parentCat = catalog
        # clusterCat = clusters[galName.split("-")[0]]
        clusterCat = select_cluster(galName)
        print(galName,clusterCat)

        photoCat = photCatalogs[clusterCat]
        # eazyCat = zphotCatalogs[clusterCat]
        if clusterCat == "abell1063":
            ### need to update name to match lensing models
            ### image names must be done first
            clusterCat = "abells1063"

        lensModels = allImageInfo[clusterCat]["lensModels"]
        lensRedshift = allImageInfo[clusterCat]["lensRedshift"]
        galImages = allImageInfo[clusterCat]["imageData"]
        # expImages = allImageInfo[clusterCat]["exposureNames"]
        whtImages = allImageInfo[clusterCat]["weightData"]
        # rmsImages = allImageInfo[clusterCat]["rmsData"]

        kGal = (parentCat["Name"] == galName)

        galPosition = SkyCoord(parentCat["RA"][kGal],parentCat["DEC"][kGal],\
                      frame="fk5",unit=(u.hourangle, u.deg))

        # print(galPosition)
        # print(galName,galRedshift,clusterCat,lensRedshift)
        # print(galPosition)
        # print(galPosition.ra.value[0])
        # continue

        print("Processing image %s - (%i out of %i)"%(galName,num+1,nGals))

        stackRA = (galPosition.ra.value[0]-SIZE/(2*3600.),galPosition.ra.value[0]+SIZE/(2*3600.))
        stackDEC = (galPosition.dec.value[0]-SIZE/(2*3600.),galPosition.dec.value[0]+SIZE/(2*3600.))
        pixScaleLensStack = 0.2

        print("\tStacking Lens Models...")
        modelStack = stack_models(lensModels,stackRA,stackDEC,\
                                  scale=pixScaleLensStack/3600.0,\
                                  modelbbox=(1.1*SIZE,galPosition))
        ExtentModel = stackRA+stackDEC

        print("\tCreating LensModel Object...")
        LensParsGalaxyFull = []
        for i in range(3):
            LensingModelStacked = LensingModel(lensRedshift,pixScaleLensStack)
            LensingModelStacked.set_lensing_data(kappa=modelStack[i,0,:,:],gamma=modelStack[i,1,:,:],\
                                                 xdeflect=modelStack[i,2,:,:],ydeflect=modelStack[i,3,:,:],\
                                                 extent=ExtentModel)

            if i == 1 :
                zTest = np.linspace(-1.0,1.0) + galRedshift
                muTest = np.zeros_like(zTest)
                for j,z in enumerate(zTest):
                    LensingModelStacked.set_model_at_z(z)
                    LensingModelStacked.compute_shear_angle(z)
                    LensParsGalaxy = LensingModelStacked.get_lensing_parameters_at_position(galPosition)
                    muTest[j] = LensParsGalaxy[2]

                LensingModelStacked.set_model_at_z(galRedshift)
                LensingModelStacked.compute_shear_angle(z)
                LensParsGalaxy = LensingModelStacked.get_lensing_parameters_at_position(galPosition)
                MagnificationMap =  LensingModelStacked.mu_at_z

                muUsed = LensParsGalaxy[2]
                fig,ax = mpl.subplots(2,1,sharex=True,figsize=(12,8))
                fig.subplots_adjust(hspace=0)
                ax[0].plot(zTest,muTest,"k-",linewidth=2)
                ax[1].plot(zTest,1 / (2 * np.sqrt(muTest**3)) * (muUsed-muTest),"k-",linewidth=2)
                ax[1].set_xlabel(r"$z$")
                ax[0].set_ylabel(r"$\mu$")
                ax[1].set_ylabel(r"$(\Delta r/r)_{\mu}$")
                for eixo in ax:
                    eixo.vlines(galRedshift,-10,10,linestyle="dashed",color="SteelBlue")
                    eixo.set_xlim(zTest[0],zTest[-1])
                ax[1].set_ylim(-0.05,0.15)
                ax[0].set_ylim(3.01,6.29)
                fig.savefig("debugResults/zphot_uncertainties_magnificantion.png")

            LensingModelStacked.set_model_at_z(galRedshift)
            LensingModelStacked.compute_shear_angle(galRedshift)
            LensParsGalaxy = LensingModelStacked.get_lensing_parameters_at_position(galPosition,plot=False)
            LensParsGalaxyFull.append(LensParsGalaxy)

        LensParsGalaxyFull = np.asarray(LensParsGalaxyFull)
        errorsLensPars = LensParsGalaxyFull[1:]-LensParsGalaxyFull[:-1]
        LensPars = LensParsGalaxyFull[1,:]


        print("\tStacking Galaxy Images...")
        galaxy = dict()
        # exposures = dict()
        weights = dict()
        rmsMap = dict()
        psfImage = dict()
        detImage = dict()
        objImage = dict()
        maskImage = dict()


        figF,axF = mpl.subplots(1,len(filters),figsize=(22,4),sharex=True,sharey=True)
        figF.subplots_adjust(wspace=0,hspace=0)
        fig,ax = mpl.subplots(5,len(filters),figsize=(18,10),sharex=True,sharey=True)
        fig.subplots_adjust(wspace=0,hspace=0)

        figM,axM = mpl.subplots(1,2,figsize=(24,10))
        figM.subplots_adjust(wspace=0)
        for k in range(len(filters)):
            galaxy[filters[k]] = galImages[k]
            galaxy[filters[k]].set_coords(galPosition)
            galaxy[filters[k]].set_bounding_box(SIZE,PIXSCALE)


        #     # exposures[filters[k]] = Galaxy(imgname=expImages[k],coords=galPosition)
        #     # exposures[filters[k]].set_bounding_box(SIZE,PIXSCALE)
            weights[filters[k]] = whtImages[k]
            weights[filters[k]].set_coords(galPosition)
            weights[filters[k]].set_bounding_box(SIZE,PIXSCALE)
            extentGalaxy = galaxy[filters[k]].get_bounding_box_coordinates()
            axM[0].imshow(galaxy[filters[k]].cutout,extent=extentGalaxy,cmap="cividis",vmin=-1e-3,vmax=0.1)
            axM[0].contour(MagnificationMap,levels=[4,4.8,5,5.2,6,10,30],extent=ExtentModel,cmap="Reds")
            axM[1].imshow(MagnificationMap[:,::-1],extent=ExtentModel,cmap="cividis",vmax=30)
            for eixo in axM:
                eixo.tick_params(labelleft=False,labelbottom=False)

            medWeight = np.nanmedian(weights[filters[k]].cutout)
            wht = weights[filters[k]].cutout
            sigma = create_sigma_image(galaxy[filters[k]].cutout*wht,2.5)/medWeight

            rmsMap[filters[k]] = np.sqrt(sigma*sigma + 1/(wht*wht))

            psfImage[filters[k]]  = stack_psf(clusterCat,[filters[k]],weights)

            objImage[filters[k]] = galaxy[filters[k]].create_object_mask(PIXSCALE,MASK_RADIUS)

            mask =  galaxy[filters[k]].create_segmentation_mask(PIXSCALE,MASK_RADIUS)
            detImage[filters[k]],_ = snd.label(mask+objImage[filters[k]])
            maskImage[filters[k]] = snd.binary_dilation(mask,structure=np.ones([5,5]))

            ax[0,k].set_title(filters[k])
            axF[k].set_title(filters[k])
            axF[k].imshow(galaxy[filters[k]].cutout[EDGE:-EDGE,EDGE:-EDGE])
            axF[k].grid(True,color="white",alpha=0.15)
            ax[0,k].imshow(galaxy[filters[k]].cutout[EDGE:-EDGE,EDGE:-EDGE])
            ax[1,k].imshow(weights[filters[k]].cutout[EDGE:-EDGE,EDGE:-EDGE])
            ax[2,k].imshow(rmsMap[filters[k]][EDGE:-EDGE,EDGE:-EDGE])
            ax[3,k].imshow((galaxy[filters[k]].cutout/rmsMap[filters[k]])[EDGE:-EDGE,EDGE:-EDGE])
            # ax[3,k].imshow(psfImage[filters[k]][EDGE:-EDGE,EDGE:-EDGE])
            # ax[3,k].imshow(objImage[filters[k]][EDGE:-EDGE,EDGE:-EDGE])
            ax[4,k].imshow(maskImage[filters[k]][EDGE:-EDGE,EDGE:-EDGE])


        for eixo in ax.ravel():
            eixo.set_xticks([])
            eixo.set_yticks([])

        figF.savefig(f"debugResults/{galName}_multi-wavelength_original.png")

        print("\tStoring in File...")
        for k in range(len(filters)):
            exptime = np.nanmedian(weights[filters[k]].cutout[EDGE:-EDGE,EDGE:-EDGE])
            galaxyData = fout.create_group(f"{filters[k]}")
            galaxyData.create_dataset("galaxy",data=galaxy[filters[k]].cutout[EDGE:-EDGE,EDGE:-EDGE])
            galaxyData.create_dataset("mask",data=maskImage[filters[k]][EDGE:-EDGE,EDGE:-EDGE])
            galaxyData.create_dataset("psf",data=psfImage[filters[k]])
            galaxyData.create_dataset("weights",data=weights[filters[k]].cutout[EDGE:-EDGE,EDGE:-EDGE])
            galaxyData.create_dataset("sigma",data=rmsMap[filters[k]][EDGE:-EDGE,EDGE:-EDGE])
            galaxyData.create_dataset("labels",data=detImage[filters[k]][EDGE:-EDGE,EDGE:-EDGE])


            galaxyData.attrs["ra"] = galPosition.ra.value[0]
            galaxyData.attrs["dec"] = galPosition.dec.value[0]
            galaxyData.attrs["z"] = galRedshift
            galaxyData.attrs["magzp"] = zeropoints[filters[k]]
            galaxyData.attrs["exptime"] = 1#exptime
            galaxyData.attrs["lensMagnification"] = LensPars[2]
            galaxyData.attrs["lensMagnificationErrors"] = errorsLensPars[:,2]
            galaxyData.attrs["lensConvergence"] = LensPars[0]
            galaxyData.attrs["lensConvergenceErrors"] = errorsLensPars[:,0]
            galaxyData.attrs["lensShear"] = LensPars[1]
            galaxyData.attrs["lensShearErrors"] = errorsLensPars[:,1]
            galaxyData.attrs["lensShearAngle"] = LensPars[3]
            galaxyData.attrs["lensShearAngleErrors"] = errorsLensPars[:,3]


        fout.close()
        # break


    if len(testGals)>0:
        mpl.show()
