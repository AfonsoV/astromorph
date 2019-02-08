# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:12:50 2015

@author: bribeiro
"""

import numpy as np
import matplotlib.pyplot as mpl
import cosmology as cosmos
import mod_imports as mi
import argparse
import string
import pyfits
import subprocess as sp
import sys

class Formatter2D():

    def __init__(self,labelx,labely,labelz,array):
        self.labx = labelx
        self.laby = labely
        self.labz = labelz
        self.data=array
        return

    def __call__(self, x, y):
        z = self.data[int(y+0.5),int(x+0.5)]
        return u'%s = %.04f,  %s = %.04f, %s = %.04f'%(self.labx,x,self.laby,y,self.labz,z)


def find_columns(catname):
    f = open(catname)
    header = f.readline()
    f.close()
    keys=header.split()
    idcol = keys.index('ident')-1
    racol = keys.index('alpha')-1
    deccol = keys.index('delta')-1
    zcol = keys.index('z')-1
    zfcol = keys.index('zflags')-1
    return [idcol,racol,deccol,zcol,zfcol]


def load_data(image,ra,dec,hsize):
    try:
        xcen,ycen = mi.get_center_coords(image,ra,dec,hsize)

        hdu=pyfits.open(image)
        image_stamp= hdu[0].data[ycen-hsize:ycen+hsize, xcen-hsize:xcen+hsize]
        hdu.close()
    except IndexError as err:
        if args.verbose:
            print(err)
        image_stamp=np.zeros([hsize*2,hsize*2])

    return image_stamp

def define_structure(size):
    structure = np.zeros([size,size])
    dmat,d= mi.distance_matrix(size/2.-0.5,size/2.-0.5,structure)
    structure[dmat<size/2]=1
    return structure


def compute_fraction_area(image,segmap,frac,flux_sampling=1000,draw_ax=None):
    total_flux = np.sum(image[segmap==1])
    max_flux = np.amax(image[segmap==1])
    min_flux = np.amin(image[segmap==1])

    total_pix = np.size(segmap[segmap==1])

    flux_step = (max_flux-min_flux)/flux_sampling

    area_flux=0
    thresh_flux=max_flux
    npix_flux=0
    while area_flux < frac*total_flux:

        area_flux = np.sum(image[(segmap==1)*(image>=thresh_flux)])
        npix_flux = np.size(image[(segmap==1)*(image>=thresh_flux)])

        if draw_ax!=None:
            draw_ax.plot([thresh_flux],[npix_flux],'r.',mec='red')
        thresh_flux -= flux_step

    if draw_ax!=None:
        draw_ax.set_xlabel(r'$f_t\ [\mathrm{e^{-}/s}]$')
        draw_ax.set_ylabel(r'$N_\mathrm{pix}(f>f_t)$')
    return npix_flux,total_pix,thresh_flux,total_flux


def get_luminosity_image(image,z,pixscale):

    lum_dist = cosmos.luminosity_distance(z)*Mpc #in cm
    ang_dist = cosmos.angular_distance(z)*Mpc ##in cm

    pixel_flux = image/(np.pi*hst_rad**2)*(h*c/l_eff)
    pixel_lum  = pixel_flux*4*np.pi*lum_dist**2
    pixel_area = (pixscale/(180/np.pi*3600)*(ang_dist)*(1e6/Mpc))**2 #in pc

    lum_density = pixel_lum/pixel_area/solar_lum ## image in L_sun/s/pc^2

    return lum_density

def dist(x1,y1,x2,y2):
    "Computes the distance between two points with coordinates (x1,y1) and (x2,y2)"
    return np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

def distance_matrix(xc,yc,img):
    """Compute a matrix with distances and a sorted np.array with unique distance
    values from each pixel position to xc,yc"""
    N,M=img.shape
    Dmat=np.zeros([N,M])
    for i in range(N):
        for j in range(M):
            Dmat[i,j] = dist(xc,yc,i,j)
    dists=np.sort(np.array(list(set(Dmat.reshape(np.size(Dmat))))))
    return Dmat, dists




def gen_segmap_tresh_lum(img,xc,yc,thresh=5.0,Amin=5,pixscale=None,all_detection=False,structure=None):
    """Generate a segmentation map by selecting pixels above sky+thresh*std
    and then select the regions within 1arcsec to belong to the object. Objects
    with a number of pixels less than Amin are non detections and are not present
    in the final segmentation map.
    The final segmentation map is dilated to smooth the detections"""
    N,M=img.shape
    MAP = np.zeros([N,M])

    MAP[img > thresh] = 1
    Regions,Nregions=mi.sci_nd.label(MAP)
    for n in range(1,Nregions+1):
        npix = len(Regions[Regions==n])
        if npix<Amin:
            Regions[Regions==n]=0
        else:
            continue

    if structure==None:
        SIZE=6
        structure = np.zeros([SIZE,SIZE])
        dmat,d= distance_matrix(SIZE/2.-0.5,SIZE/2.-0.5,structure)
        structure[dmat<SIZE/2]=1


    if all_detection:
        return Regions
    else:
        segmap = mi.select_object_map(xc,yc,Regions,pixscale=pixscale)
        return mi.sci_nd.binary_dilation(segmap,structure=structure)


def compute_size(image,ra,dec,redshift,hsize,threshold,fractions,size=6,safedist=1.0):

    Sizes = np.zeros(len(fractions)+1)
    Fluxes = np.zeros(len(fractions)+1)

    image_stamp = load_data(image,ra,dec,hsize)
    if np.amax(image_stamp)==np.amin(image_stamp):
        return Sizes - 99,-9

    image_lum = get_luminosity_image(image_stamp,redshift,args.pixscale)

    dilate = define_structure(size)

    segmap = gen_segmap_tresh_lum(image_lum,hsize,hsize,thresh=args.threshold,Amin=args.areamin,pixscale= args.pixscale,all_detection=True)
    single_source_map = mi.select_object_map(hsize,hsize,segmap,pixscale=args.pixscale)
    image_stamp,single_source_map,imglag,segflag = mi.image_validation(image_stamp,single_source_map,args.pixscale,safedist)


    if segflag==1:
        return Sizes - 99,segflag

    if args.nodilation:
        dilated_map = single_source_map
    else:
        dilated_map = mi.sci_nd.binary_dilation(single_source_map,structure=dilate).astype(np.int32)

    for i in range(len(fractions)):
        Sizes[i],TotalArea,Fluxes[i],TotalFlux = compute_fraction_area(image_stamp,dilated_map,fractions[i])
    Sizes[i+1]=TotalArea
    Fluxes[i+1]=TotalFlux

    if args.nosaving:
        mpl.rcParams['image.cmap']='gist_stern_r'
        segmap[segmap!=0]/=1
        fig,ax=mpl.subplots(2,3,figsize=(20.6,12))
        mpl.subplots_adjust(wspace=0.2,hspace=0.02)
        ax=ax.reshape(np.size(ax))
        ax[0].imshow(np.sqrt(np.abs(image_stamp)),cmap='hot')
        mi.gen_circle(ax[1],hsize,hsize,hsize*(2*np.sqrt(2)*0.5/args.size),color='red')
        ax[1].imshow(segmap)
        ax[2].imshow(single_source_map)

        flux_map = np.zeros(image_stamp.shape)
        flux_map[dilated_map==1]=3
        for j in range(len(Fluxes)-2,-1,-1):
            flux_map[image_stamp>Fluxes[j]]=(len(Fluxes)-j)*4
        flux_map*=dilated_map
        imax,imin,jmax,jmin = mi.find_ij(dilated_map)
        ax[3].imshow(flux_map[imin:imax,jmin:jmax],aspect='equal',cmap='gist_heat_r')
        ax[3].text(12,9,'20',color='white',va='center',ha='center',weight='heavy')
        ax[3].text(10,7.5,'50',color='white',va='center',ha='center',weight='heavy')
        ax[3].text(8.5,5.5,'80',color='white',va='center',ha='center',weight='heavy')
        ax[3].text(6,3,'100',color='white',va='center',ha='center',weight='heavy')


#        masked_neighbours = mi.sci_nd.binary_dilation(segmap - single_source_map,structure=dilate)
        compute_fraction_area(image_stamp,dilated_map,1.0,flux_sampling=500,draw_ax=ax[4])

        ax[5].imshow(dilated_map)

        for i in range(len(ax)):
            if i!=4:
                ax[i].set_xticks([])
                ax[i].set_yticks([])

        fig.text(0.95,0.50,r'$\curvearrowright$',va='center',ha='center',weight='heavy',rotation=-90,fontsize=150)
#        fig.savefig('size_thresholding_example.png')
        def exit_code(event):
            if event.key=='escape':
                sys.exit()
            if event.key=='q':
                mpl.close('all')

        ax[0].format_coord=Formatter2D("x","y","I",image_lum)
        fig.canvas.mpl_connect('key_press_event',exit_code)
        mpl.show()


    return Sizes,segflag

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Computation of non-parametric values for VUDS galaxies.")
    parser.add_argument('-i','--image',metavar='imgname',type=str,help="Field Image on which to compute the non-parametric values")
    parser.add_argument('-c','--catalog',metavar='catname',type=str,help="The input catalog of galaxies to be included in the analysis")
    parser.add_argument('-f','--fractions',metavar='f1,f2,...',type=str,default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",help="Flux fractions to compute galaxy area.")
    parser.add_argument('-t','--tilesmatch',metavar='filename',type=str,help="File containing information on which tile is each galaxy")
    parser.add_argument('-r','--size',metavar='size',type=float,default=10,help="The size (in arcseconds) of the stamp image. Default: 10")
    parser.add_argument('-p','--pixscale',metavar='size',type=float,help="The pixel scale of the image")
    parser.add_argument('-I','--ident',metavar='ID',type=int,help="The galaxy on which to run all computations. If none given, run for all.")
    parser.add_argument('-T','--threshold',metavar='thresh',type=float,default=3.0,help="The threshold of the segmentation map in L_sun/pc^2")
    parser.add_argument('-A','--areamin',metavar='size',type=int,default=5,help="The minimium area above which to consider a positive detection")
    parser.add_argument('-K','--ksky',metavar='kappa',type=float,default=2.5,help="The default sky threshold for segmentation map for images")
    parser.add_argument('-v','--verbose',action='store_true',help="If present outputs values to terminal as well.")
    parser.add_argument('-N','--nosaving',action='store_true',help="If present plots information onto the screen and does not write to any table.")
    parser.add_argument('-F','--fixthresh',action='store_true',help="If present: use fixed threshold, else use variable treshold with redshift.")
    parser.add_argument('-S','--sigma0',metavar='sigma_0',type=float,default=3.0,help="The sigma anchor value at redshift 5")
    parser.add_argument('--nodilation',action='store_true',help="If present: does not perform binary dilation of the segmentation map.")

    args = parser.parse_args()



    fieldimg = args.image
    samplecat = args.catalog
    match_tiles = args.tilesmatch
    startid=args.ident

    colnumbers=find_columns(samplecat)
    ID,RA,DEC,Z,Zflag = np.loadtxt(samplecat,unpack=True,dtype={'names':('a','b','c','d','e'),'formats':('i8','f4','f4','f4','i4')},usecols=colnumbers)
    hsize = int(args.size/args.pixscale)/2

    fractions = [np.float32(f) for f in args.fractions.split(',')]

    check= np.array(['band' in i for i in fieldimg.split('/')])
    if 'CFHTLS' in fieldimg:
        band = string.upper(fieldimg.split('_')[-4])+'band'
        index=3
    else:
        index = np.where(check==True)[0]
        band= fieldimg.split('/')[index]

    field = fieldimg.split('/')[index-1]
    survey = fieldimg.split('/')[index-2]
    if survey =='..':
        survey = fieldimg.split('/')[index-1]

    if args.nodilation:
        prefix='no_dilation_'
    else:
        prefix=''

    if args.fixthresh:
        mode = 'fixthresh'
        Tmode = args.threshold
    else:
        Tmode = args.sigma0
        mode = 'varthresh'

    if args.nosaving:
        if args.verbose:
            print("WARNING: Not saving results to table!")
        pass
    else:
        table_name= "tables_lumlimits/%ssizes_nonparametric_%s_%s_%s_%s_%i.txt"%(prefix,survey,field,band,mode,Tmode)
        if args.verbose:
            print("Saving results to %s"%table_name)
        table_out = open(table_name,"w")

        table_out.write("#ID\tRA\tDEC\tZ\tZflag\t")
        table_out.write("\t".join(["T_%i[pix]"%(round(f*100.0,0)) for f in fractions]))
        table_out.write("\tT_100[pix]\t")
        table_out.write("\t".join(["T_%i[kpc]"%(round(f*100.0,0)) for f in fractions]))
        table_out.write("\tT_100[kpc]\tSegFlag\n")

    start=0
    T=[]
    sblims=np.loadtxt('sblimits.dat',unpack=True,usecols=[1])

    lambda_eff={'I':8140e-10,'H':15369e-10,'J':12486e-10} # cm

    hst_rad=150 #cm
    h=6.626e-27 #ergs.s
    c=2.9979e10 #cm/s
    Mpc=3.0856e24 #cm
    solar_lum = 3.846e33 #ergs/s

    l_eff=lambda_eff[band[:1]]


#    if args.verbose:
#        print('Surface Brightness Limit: %.5f counts/s/pixel**2'%sblims)

    for i in range(len(ID)):

        if not (ID[i] == startid) and (start==0) and (startid!=None):
            continue
        else:
            start=1

        if args.verbose:
            print("---------------------------------------------------------> VUDS %i <---------------------------------------------------------"%ID[i])
        if match_tiles == None:
            imgname = fieldimg
        else:
            tilename=sp.check_output("awk '$1==%s {print $2}' %s"%(ID[i],match_tiles),shell=True)
            try:
                tilename=tilename.split()[0]
            except IndexError as err:
                if args.verbose:
                    print(err)
                continue
            imgname ='%s/%s'%(fieldimg,tilename)

        if args.fixthresh:
            t = args.threshold
        else:
            sigma_0=args.sigma0
            t = sigma_0 * ((1+Z[i])/(1+4.0))**(-3)

        galaxy_sizes,segflag = compute_size(imgname,RA[i],DEC[i],Z[i],hsize,t,fractions)




        da=cosmos.angular_distance(Z[i],pars=None)*1000
        galaxy_physic_sizes = np.array([(N*(args.pixscale**2)*(180/np.pi*3600)**(-2)*(da**2)) for N in galaxy_sizes])
        galaxy_physic_sizes[galaxy_physic_sizes<0] = -99.0

        if args.verbose:
            print(""" threshold = %8.4fL_sun/pc^2 @ z = %.3f d_a=%.3f Mpc)
            Pixel  Sizes %s
            Physic Sizes %s
            Segmentation Flag %i
            """%(t,Z[i],da/1000.,"\t".join(["%10.5f"%(s) for s in galaxy_sizes]),"\t".join(["%10.5f"%(s) for s in galaxy_physic_sizes]) ,segflag ))

        if not args.nosaving:
            table_out.write("%10i\t%12.8f\t%12.8f\t%5.3f\t%2i\t"%(ID[i],RA[i],DEC[i],Z[i],Zflag[i]))
            table_out.write("\t".join(["%10.5f"%(s) for s in galaxy_sizes]))
            table_out.write("\t")
            table_out.write("\t".join(["%10.5f"%(s) for s in galaxy_physic_sizes]))
            table_out.write("\t%i\n"%segflag)

        if args.verbose:
            print("-------------------------------------------------------------------------------------------------------------------------------------"%ID[i])


    if not args.nosaving:
        table_out.close()
