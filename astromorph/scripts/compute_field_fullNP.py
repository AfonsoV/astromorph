# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 12:32:00 2014

@author: Bruno Ribeiro

Code to process a sample of galaxies in a given field and provide a table with the results from non-parametric analysis.
21/07/2014  - First complete functional code without any fatal errors on execution
23/07/2014  - Note the angle correction and the switch of x and y positions in the computation of 
            r_petrosian and C due to the orientation of numpy np.arrays with respect to the image orientation
"""

import TMC
import CAS
import MID
import gm20
import ADD
from mod_imports import *
import argparse
from string import upper



class Galaxy:

    def define_structure(self):
        SIZE=10
        self.structure = np.zeros([SIZE,SIZE])
        dmat,d= distance_matrix(SIZE/2.-0.5,SIZE/2.-0.5,self.structure)
        self.structure[dmat<SIZE/2]=1

    
    def color_correction(self,threshold,color):
        ColorCorretion = 10**(0.4*(color))
        thresh_corrected = threshold*ColorCorretion
        if args.verbose:
            print("Color Correction Term: %.5f"%ColorCorretion)
            print("Old Threshold: %.4f, New Threshold: %.4f"%(threshold,thresh_corrected))
        
        return thresh_corrected
    
    def load_segmap(self,image,safedist=1.0):

        sky_med,sky_std = sky_value(image,args.ksky)

        if args.nodilation:
            image_stamp = sci_nd.gaussian_filter(image,sigma=1.0)

        
        if np.abs(self.color)<10:    
            corrected_thresh = self.color_correction(self.thresh,self.color)
#            new_sblimit = corrected_thresh*self.sblimit
            threshold=corrected_thresh
            added_flag=0
        else:
            threshold=self.thresh
            added_flag=0.5
            
        self.used_threshold = threshold
        self.skyval = sky_med
        self.skystd = sky_std
            
        segmap = gen_segmap_sbthresh(image_stamp-sky_med,self.hsize,self.hsize,self.sblimit,args.pixscale,thresh=threshold,Amin=args.areamin,all_detection=True)
        single_source_map = select_object_map_connected(self.hsize,self.hsize,image_stamp,segmap,pixscale=args.pixscale,radius=args.aperture)
        image_stamp,single_source_map,imflag,segflag = image_validation(image_stamp,single_source_map,args.pixscale,safedist)    


        segflag+=added_flag
        
        mask = segmap.copy()
        mask[mask>0]=1
        mask = mask-single_source_map
        
        return image,imflag,single_source_map,mask,segflag
    
    def load_data(self,image,Xgal,Ygal):
        self.imagename=image
        hdu=pyfits.open(image)
        self.imagecheck= hdu[0].data[Ygal-self.hsize:Ygal+self.hsize, Xgal-self.hsize:Xgal+self.hsize]
        hdu.close()

        self.originalgaldata,self.imageflag,self.originalsegmap,self.mask,self.segmapflag = self.load_segmap(self.imagecheck)



        if self.imageflag==1:
            if args.verbose:
                print "Mostly zero values in image data!"
            self.nodata = True
            return
        
        self.maskeddata=self.originalgaldata.copy()
        nmaskedvalues=len(self.maskeddata[self.mask==1])
        self.sky,self.std=sky_value(self.originalgaldata)
        self.maskeddata[self.mask==1]=npr.normal(self.sky,self.std,nmaskedvalues)
        sky_value(self.originalgaldata)[0]
        self.originalbarx,self.originalbary = barycenter(self.originalgaldata,self.originalsegmap)
        
        self.galdata,self.segmap,self.center=make_stamps(self.originalbarx,self.originalbary,self.maskeddata,self.originalsegmap,pixscale=args.pixscale,fact=args.zoomfactor)
        self.barx,self.bary = self.center[1],self.center[0]
        
        if args.nosaving:
            fig,ax=mpl.subplots(2,3)
            ax=ax.reshape(np.size(ax))
            ax[0].imshow(np.sqrt(abs(self.originalgaldata)))
            ax[1].imshow(self.originalsegmap)
            ax[2].imshow(self.mask)
            ax[3].imshow(self.maskeddata)
            ax[4].imshow(self.galdata)
            ax[5].imshow(self.segmap)
            mpl.show()
        
        
    def __init__(self,image,info_dict):
        self.ra = info_dict['RA']
        self.dec = info_dict['DEC']
        self.ID = info_dict['ID']
        self.redshift = info_dict['z']
        self.redshiftflag = info_dict['zflag']
        self.hsize=info_dict['hsize']
        self.color=info_dict['color']
        self.thresh=info_dict['threshold']
        self.sblimit = info_dict['sblimit']

        self.define_structure()
        try:        
            self.Xoriginal, self.Yoriginal = get_center_coords(image,self.ra,self.dec,self.hsize)
            self.nodata=False
        except (IndexError,IOError) as err:
            if args.verbose:
                print err
            self.nodata=True
            return
            
        self.load_data(image,self.Xoriginal,self.Yoriginal)
        
        if self.nodata:
            return
        XX2,YY2,XY = moments(self.galdata,self.segmap)
        XX2-=self.barx*self.barx
        YY2-=self.bary*self.bary
        XY -=self.barx*self.bary
        self.theta_sky = 180.-theta_sky(XX2,YY2,XY) ## Orientation correction
        self.axis_ratio = axis_ratio(XX2,YY2,XY)
        if np.isnan(self.axis_ratio) and self.segmapflag==1:
            self.nodata=True
            return
        if args.verbose:
            print "q=%.2f\ntheta=%.2f"%(self.axis_ratio,self.theta_sky)
        
    def __str__(self):
        S= "Object %i @ ra,dec=[%.4f,%4f] z=%.3f [%i]"%(self.ID,self.ra,self.dec,self.redshift,self.redshiftflag)
        S+= "\n\tDetected source at X=%.3f Y=%3f"%(self.Xoriginal,self.Yoriginal)
        S+= "\n\tCutout size %i x %i pixel"%self.galdata.shape
        S+= "\n\tPosition Angle: %.3f Axis Ratio: %.2f"%(self.theta_sky,self.axis_ratio)
        return S

    def compute_CAS(self):
        if self.nodata:
            return

        if args.verbose:
            print "-------------------> Computing CAS!"
            
        segmap_for_sky = gen_segmap_tresh(self.originalgaldata,self.hsize,self.hsize,thresh=3.0,Amin=5,k_sky=3,pixscale=args.pixscale,all_detection=True)
        self.sky_patch = extract_sky(self.originalgaldata,self.galdata.shape,segmap_for_sky)

#        fig,ax=mpl.subplots()
#        ax.imshow(self.sky_patch)
#        mpl.show()
        
        self.petrosianradius_circular,self.prcflag = CAS.petrosian_rad(self.maskeddata,self.originalbary,self.originalbarx,verbose=args.verbose)
        self.concentration_circular = CAS.CAS_C(self.maskeddata,self.originalbary,self.originalbarx,rp=self.petrosianradius_circular)[0]
        self.petrosianradius_elliptic,self.preflag  = CAS.petrosian_rad(self.maskeddata,self.originalbary,self.originalbarx,q=self.axis_ratio,ang=self.theta_sky,draw_stuff=args.checkrad,verbose=args.verbose)
        self.concentration_elliptic = CAS.CAS_C(self.maskeddata,self.originalbary,self.originalbarx,ba=self.axis_ratio,theta=self.theta_sky,rp=self.petrosianradius_elliptic)[0]

        if args.verbose:
            print "Computed (C)oncetration           :\t C_c=%.3f, C_e=%.3f"%(self.concentration_circular,self.concentration_elliptic)
        
        self.asymmetry,self.Agalaxy,self.Asky = CAS.CAS_A(self.center,self.galdata,args.nskiesAsymm,self.maskeddata,segmap_for_sky)
        if args.verbose:
            print "Computed (A)symmetry              :\t A=%.3f, Agal=%.3f, Asky=%.3f"%(self.asymmetry,self.Agalaxy,self.Asky)

        if np.isinf(self.petrosianradius_elliptic) and np.isinf(self.petrosianradius_circular):
            rpS = 1.0
        elif np.isinf(self.petrosianradius_elliptic):
            rpS=self.petrosianradius_circular
        else:
            rpS = self.petrosianradius_elliptic
            

        if np.ceil(0.3*rpS)<self.galdata.shape[0]:
            self.clumpiness,self.Sgalaxy,self.Ssky = CAS.CAS_S(self.galdata,self.barx,self.bary,0.3*rpS,self.sky_patch)
            self.clumpflag=0
        else:            
            sky_patch = extract_sky(self.originalgaldata,self.originalgaldata.shape,segmap_for_sky)
            self.clumpiness,self.Sgalaxy,self.Ssky = CAS.CAS_S(self.maskeddata,self.originalbarx,self.originalbary,0.3*rpS,sky_patch)
            self.clumpflag=1

#            galdata, segmap, center = make_stamps(self.originalbarx,self.originalbary,self.maskeddata,self.originalsegmap,pixscale=args.pixscale,fact=2*np.ceil(0.3*rpS/self.galdata.shape[0]))
#            sky_patch = extract_sky(self.originalgaldata,galdata.shape,segmap_for_sky)
#            barx,bary=center[1],center[0]
#            self.clumpiness,self.Sgalaxy,self.Ssky = CAS.CAS_S(galdata,barx,bary,0.3*rpS,sky_patch)
            
        if args.verbose:
            print "Computed (S)Clumpiness            :\t S=%.3f, Sgal=%.3f, Ssky=%.3f"%(self.clumpiness,self.Sgalaxy,self.Ssky)
            print "-------------------> Done computing CAS!"
        
    def compute_GiniM20(self):
        if self.nodata:
            return

        if args.verbose:            
            print "-------------------> Computing Gini-M20!"
        self.gini=gm20.Gini(self.galdata,self.segmap)
        if args.verbose:        
            print "Computed (G)ini                   :\t G=%.3f"%self.gini
        self.momentlight20,self.m20flag = gm20.MomentLight20(self.galdata,self.segmap,self.barx,self.bary,verbose=args.verbose)
        if args.verbose:        
            print "Computed (M20)Moment of 20%% light :\t M20=%.3f"%self.momentlight20
            print "-------------------> Done computing Gini-M20!"
    
    def compute_MID(self):
        if self.nodata:
            return
        if args.verbose:
            print "-------------------> Computing MID!"
        Rs,Qs = MID.multimode(self.galdata,self.segmap)
        self.multimode = max(Rs)
        self.multimode_percentile = Qs[Rs==max(Rs)][0]
        if args.verbose:
            print "Computed (M)ultimode              :\t M=%.3f"%self.multimode
        IntensityMap,LocalMaxima = MID.local_maxims(self.galdata,self.segmap)
        self.intensity,self.intensity_center =MID.intensity(self.galdata,IntensityMap,self.segmap,LocalMaxima)
        if args.verbose:
            print "Computed (I)ntensity              :\t I=%.3f"%self.intensity
        self.deviation = MID.deviation(self.galdata,self.segmap,self.intensity_center)
        if args.verbose:
            print "Computed (D)eviation              :\t D=%.3f"%self.deviation
            print "-------------------> Done computing MID!"
        
    def compute_PsiTXi(self):
        if self.nodata:
            return
        if args.verbose:
            print "-------------------> Computing TPsiXi!"
        self.size = TMC.Size(self.segmap,args.pixscale,self.redshift)
        if args.verbose:
            print "Computed (T)Size                  :\t T=%.3f"%(self.size)
        self.multiplicity = TMC.Multiplicity(self.galdata,self.segmap)[0]
        if args.verbose:
            print "Computed (Psi)Multiplicity        :\t Psi=%.3f"%(self.multiplicity)
#        self.color_dispersion = TMC.color_dispersion()
        self.color_dispersion = -9.999
        if args.verbose:
            print "Computed (Xi)ColorDispersion      :\t Xi=%.3f"%self.color_dispersion
#        #Needed Way of implementation of the Color Dispersion secondary image

        if args.verbose:
            print "-------------------> Done computing TPsiXi!" 
        
    def compute_filamentarity(self):
        if self.nodata:
            return
        if args.verbose:
            print "-------------------> Computing Additional Parameters!"
        self.filamentarity = ADD.filamentarity(self.segmap)
        if args.verbose:
            print "Computed Filamentarity            :\t F=%.3f"%(self.filamentarity)
            print "-------------------> Done computing Filamentarity!"
    
    
    def write_line_results(self,filepointer):
        if self.nodata:
            filepointer.write("%10i\t%12.8f\t%12.8f\t%7.4f\t%4i\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\t-99.0\n"%(self.ID,self.ra,self.dec,self.redshift,self.redshiftflag))
        else:
            filepointer.write("%10i\t%12.8f\t%12.8f\t%7.4f\t%4i\t"%(self.ID,self.ra,self.dec,self.redshift,self.redshiftflag))
            filepointer.write("%7.4f\t%7.4f\t"%(self.axis_ratio,self.theta_sky))       
            filepointer.write("%7.4f\t%7.4f\t%7.4f\t%7.4f\t"%(self.petrosianradius_circular,self.concentration_circular,self.petrosianradius_elliptic,self.concentration_elliptic))
            filepointer.write("%7.4f\t%7.4f\t"%(self.Agalaxy,self.clumpiness))
            filepointer.write("%7.4f\t%7.4f\t"%(self.gini,self.momentlight20))
            filepointer.write("%7.4f\t%7.4f\t%7.4f\t"%(self.multimode,self.intensity,self.deviation))
            filepointer.write("%10.4f\t%7.4f\t%7.4f\t"%(self.size,self.multiplicity,self.color_dispersion))
            filepointer.write("%5.3f\t"%self.filamentarity)
            filepointer.write("%3.1f\t%3i\t%3i\t%3i\t%3i\t"%(self.segmapflag,self.prcflag,self.preflag,self.m20flag,self.clumpflag))
            filepointer.write("%7.4f\t%12.8f\t%12.8f"%(self.used_threshold,self.skyval,self.skystd))

            filepointer.write("\n")
        
#def get_focus(focus_table,name):
#    """"Function helper to get focus value for psf grab"""
#    f=open(focus_table)
#    table=f.readlines()
#    f.close()
#
#    for line in table:
#        if line.split()[0]==name:
#            focus=float(line.split()[-3])
#            break
#        else:
#            continue
#    return focus
# 
#def dist2(x1,y1,x2,y2):
#    "Computes the distance between two points with coordinates (x1,y1) and (x2,y2)"
#    return ((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
#    
#def get_sex_pars(xc,yc,rmax,catfile='test.cat',psf=False):
#    """Returns the SExtractor parameters from the catalog associated with the
#    segmentation map for the source closest to the VUDS catalog coordinates."""
#
#    f=open(catfile,'r')
#    txt=f.readlines()
#    f.close()
#
#    last_line=txt[-1]
#    if last_line[0]=='#':        
#        return -99,-99,-99,-99,-99,-99,-99,-99,-99,-99
#
#    if psf:
#        xs,ys,mag,mum,re,a,kr,e,t,cs,xp,yp=np.loadtxt(catfile,unpack=True)
#    else:
#        xs,ys,mag,mum,re,a,kr,e,t,cs,xp,yp,isoarea=np.loadtxt(catfile,unpack=True)
#
#    n=a*kr/re
#    e=1-e
#    t=t-90
#    if np.size(xs)==1:
#        separation=np.sqrt(dist2(xs,ys,xc,yc))
#        if psf:
#            return xp,yp
#        else:
#            return xs,ys,mag,re,n,e,t,0,separation,isoarea
#    else:
#        dists=(dist2(xs,ys,xc,yc))
#        obj_num=np.where(dists == min(dists))
#        if min(dists)>rmax*rmax:
#            return -99
#        if psf:
#            return xp,yp,obj_num
#        else:
#            return xs,ys,mag,re,n,e,t,obj_num,np.sqrt(min(dists)),isoarea
#            
#def select_PSF(ID,psfdir,focus,xc,yc,hsize=50):
#    """Function helper to get psf image from focus value"""
#    if focus<=-6.5:
#        focus=-6.
#    if focus<=-8.0:
#        focus=-10.
#    psf_cat="%s/TinyTim_f%i.cat"%(psfdir,round(focus,0))
#    psf_img="%s/TinyTim_f%i.fits"%(psfdir,round(focus,0))
#    xs,ys,num=get_sex_pars(xc,yc,rmax=330,catfile=psf_cat,psf=True)
#    X,Y=xs[num],ys[num]
#    if args.ident is None:
#        iraf.imarith("%s[%i:%i,%i:%i]"%(psf_img,X-hsize,X+hsize,Y-hsize,Y+hsize),'*',1.0,'psf.fits')
#    else:
#        iraf.imarith("%s[%i:%i,%i:%i]"%(psf_img,X-hsize,X+hsize,Y-hsize,Y+hsize),'*',1.0,'psf-%i.fits'%(args.ident))
#    
#    return
#
#def get_center_coords(imgname,ra,dec):
#    """ Function ot convert from sky coordinates (RA,DEC) into image coordinates
#    (XC,YC).
#    """
#    import pywcs
#    hdu=pyfits.open(imgname)
#    wcs=pywcs.WCS(hdu[0].header)
#
#    ctype=hdu[0].header["ctype1"]
#    xmax=hdu[0].header["naxis1"]
#    ymax=hdu[0].header["naxis2"]
#    
#    if 'RA' in ctype:
#        sky=np.array([[ra,dec]],np.float_)
#    else:
#        sky=np.array([[dec,ra]],np.float_)
#
#    pixcrd=wcs.wcs_sky2pix(sky,1)
#
#    xc=pixcrd[0,0]
#    yc=pixcrd[0,1]
#
#    return xc,yc


def luminosity_evol(z,zbreak=3.0):
    P23 = -0.360 ## From Reddy&Steidel2009
    P46 = -0.067 ## From Bouwens+2015, see color_and_lum_corrections_report folder for more details
    if z<zbreak:
        return 10**(-0.4*(P23*(z)))
    else:
        return 10**(-0.4*(P46*(z)))*10**(-0.4*(P23*zbreak-P46*zbreak))

def luminosity_evolution(z,zp=2.0,zbreak=3.0):
    return luminosity_evol(z)/luminosity_evol(zp)
    
def find_columns(catname,mag_colname,magref_colname):
    f = open(catname)
    header = f.readline().replace('#','')
    f.close()
    keys=header.split()
    idcol = keys.index('ident')
    racol = keys.index('alpha')
    deccol = keys.index('delta')
    zcol = keys.index('z')
    zfcol = keys.index('zflags')
    magcol = keys.index(mag_colname)
    
    magrefs=magref_colname.split(',')
    
    magrefcol = [keys.index(mc) for mc in magrefs]
    
    return [idcol,racol,deccol,zcol,zfcol,magcol]+magrefcol
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Computation of non-parametric values for VUDS galaxies.")
    parser.add_argument('-i','--image',metavar='imgname',type=str,help="Field Image on which to compute the non-parametric values")
    parser.add_argument('-c','--catalog',metavar='catname',type=str,help="The input catalog of galaxies to be included in the analysis")
    parser.add_argument('-f','--fractions',metavar='f1,f2,...',type=str,default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.0",help="Flux fractions to compute galaxy area.")
    parser.add_argument('-e','--erosions',metavar='f1,f2,...',type=str,default="3,5,7",help="Erosion factors to degrade galaxy segmap.")
    parser.add_argument('-t','--tilesmatch',metavar='filename',type=str,help="File containing information on which tile is each galaxy")
    parser.add_argument('-m','--magnitude',metavar='column',type=str,help="Column name containing galaxy magnitudes to be used for PSF simulation")
    parser.add_argument('-R','--refmagnitude',metavar='name,name',type=str,help="Column name containing galaxy magnitudes to be used for color correction.")
    parser.add_argument('-P','--psf',metavar='filename',type=str,help="Name of the psf filefor the PSF simulation")
    parser.add_argument('-r','--size',metavar='size',type=float,default=8,help="The size (in arcseconds) of the stamp image. Default: 10")
    parser.add_argument('-p','--pixscale',metavar='size',type=float,help="The pixel scale of the image")
    parser.add_argument('-I','--ident',metavar='ID',type=int,help="The galaxy on which to start to run all computations. If none given, run for all.")
    parser.add_argument('--end_ident',metavar='ID',type=int,help="The galaxy on which to send the code.")
    parser.add_argument('-T','--threshold',metavar='thresh',type=float,default=3.0,help="The threshold of te segmentation map")
    parser.add_argument('-a','--aperture',metavar='size',type=float,default=0.5,help="The radius, in arcseconds, of the serach area aperture.")
    parser.add_argument('-A','--areamin',metavar='size',type=int,default=10,help="The minimium area above which to consider a positive detection")
    parser.add_argument('-K','--ksky',metavar='kappa',type=float,default=3.0,help="The default sky threshold for segmentation map for images")
    parser.add_argument('-v','--verbose',action='store_true',help="If present outputs values to terminal as well.")
    parser.add_argument('-N','--nosaving',action='store_true',help="If present plots information onto the screen and does not write to any table.")
    parser.add_argument('-F','--fixthresh',action='store_true',help="If present: use fixed threshold, else use variable treshold with redshift.")
    parser.add_argument('-S','--sigma0',metavar='sigma_0',type=float,default=1.0,help="The sigma anchor value at redshift 4")
    parser.add_argument('-z','--zeropoint',metavar='mag_zp',type=float,help="Magnitude zeropoint of the image")
    parser.add_argument('--evo',action='store_true',help="If present applies luminosity evolution in threshold computation.")
    parser.add_argument('--nodilation',action='store_true',help="If present: does not perform binary dilation of the segmentation map.")
    parser.add_argument('--focus',metavar='NAME',type=str,help="Catalog on which there is the match  of the Tile image and its Focus Value.'")
    parser.add_argument('--error',action='store_true',help="If present: computes the error on the sizes.")
    parser.add_argument('--color',metavar='NAME',type=str,default='',help="suffix name to append to name of the table with the results")

    parser.add_argument('--zoomfactor',metavar='size',type=int,default=2,help="The zoomfactor for images cuts")
    parser.add_argument('--defaultseg',metavar='size',type=float,default=1.0,help="The default segmentation cut for images with no detection")
    parser.add_argument('--nskiesAsymm',metavar='N',type=int,default=20,help="The number of sky realizations to compute A_sky")
    parser.add_argument('--checkrad',action='store_true',help="If present draws the petrosian radius calculation of the first galaxy and then quits.")
    args = parser.parse_args()



    fieldimg = args.image
    samplecat = args.catalog
    match_tiles = args.tilesmatch
    startid=args.ident
    endid=args.end_ident

    
    colnumbers=find_columns(samplecat,args.magnitude,args.refmagnitude)
    ID,RA,DEC,Z,Zflag,Mag,MagRef1,MagRef2 = np.loadtxt(samplecat,unpack=True,dtype={'names':('a','b','c','d','e','f','g','h'),'formats':('i8','f4','f4','f4','i4','f4','f4','f4')},usecols=colnumbers)
    hsize = (args.size/args.pixscale)/2
    
    check= np.array(['band' in i for i in fieldimg.split('/')])
    if 'CFHTLS' in fieldimg:
        band = upper(fieldimg.split('_')[-4])+'band'
        index=3
    else:
        index = np.where(check==True)[0]
        band= fieldimg.split('/')[index]
    
    field = fieldimg.split('/')[index-1]
    survey = fieldimg.split('/')[index-2]
    if survey =='..':
        survey = fieldimg.split('/')[index-1]


    if args.nodilation:
        prefix='no_dilation'
    else:
        prefix=''
    
    if args.evo:
        suffix2='_evolum'
    else:
        suffix2=''
        
    if args.fixthresh:
        mode = 'fixthresh'
        Tmode = args.threshold
    else:
        Tmode = args.sigma0
        mode = 'varthresh'

    if args.nosaving:
        if args.verbose:
            print "WARNING: Not saving results to table!"
        pass
    else:
        if startid is None:
            table_name= "nonparametric_%s_%s_%s_%s_%s_%.2f_ap%.2f_%s%s.txt"%(prefix,survey,field,band,mode,Tmode,args.aperture,args.color,suffix2)
        else:            
            table_name= "nonparametric_%s_%s_%s_%s_%s_%.2f_ap%.2f_%s%s-%i.txt"%(prefix,survey,field,band,mode,Tmode,args.aperture,args.color,suffix2,startid)
        if args.verbose:
            print "Saving results to %s"%table_name
        table_out = open(table_name,"w")
        table_out = open(table_name,"w")
        table_out.write("#ID\tRA\tDEC\tZ\tZflag\tq\ttheta\trpc\tCc\trpe\tCe\tA\tS\tG\tM20\tM\tI\tD\tT\t\\Psi\t\\Xi\tF\tSegFlag\tRpcFlag\tRpeFlag\tM20flag\tClumpFlag\tUsedThresh\tSkyMed\tSkyStd\n")    
    
        if args.verbose:
            print "Saving results to %s"%table_name

    start=0
    T=[]
    sblim=np.amax(np.loadtxt('sblimits.dat',unpack=True,usecols=[2]))
    if args.verbose:
        print('Surface Brightness Limit: %.5f counts/s/arcsec**2'%sblim)
        
    has_psf=False
    if args.focus is None:
        psf_file=args.psf
        has_psf=True
    else:
        pass



    for i in range(len(ID)):

        if not (ID[i] == startid) and (start==0) and (startid!=None):
            continue
        else:
            start=1

        if endid is None:
            pass
        elif ID[i]==endid:
            break

          
        if args.verbose:        
            print "---------------------------------------------------------> VUDS %i (%i out of %i) @ kp=%.2f <---------------------------------------------------------"%(ID[i],i+1,len(ID),args.sigma0)


##        if os.path.isfile('psf.fits') and args.ident is None:
##            sp.call('rm psf.fits',shell=True)
##        elif args.ident is None:
##            pass
##        elif os.path.isfile('psf-%i.fits'%args.ident):
##            sp.call('rm psf-%i.fits'%args.ident,shell=True)
##        else:
##            pass
##        
##        
##        if has_psf and args.ident is None:
##            sp.call('cp %s psf.fits'%psf_file,shell=True)
##        elif (has_psf) and (not args.ident is None):
##            sp.call('cp %s psf-%i.fits'%(psf_file,args.ident),shell=True)


        if match_tiles == None:
            imgname = fieldimg
        else:
            tilename=sp.check_output("awk '$1==%s {print $2}' %s"%(ID[i],match_tiles),shell=True)
            try:
                tilename=tilename.split()[0]
            except IndexError as err:
                if args.verbose:
                    print err
                continue
            imgname ='%s/%s'%(fieldimg,tilename)
            if args.focus is None:
                pass
            else:
                pass
#                focus_value = get_focus(args.focus,tilename)
#                psf_dir='/'.join(match_tiles.split('/')[:-1])+'/PSFs'
#                xc,yc=get_center_coords(imgname,RA[i],DEC[i])
#                select_PSF(ID[i],psf_dir,focus_value,xc,yc)


        zp=2.0
        if args.fixthresh:
            t = args.threshold       
        else:
            sigma_0=args.sigma0
            t = sigma_0 * ((1+Z[i])/(1+zp))**(-3)
        
        if args.evo:
            t*=luminosity_evolution(Z[i],zp)

        if args.fixthresh:
            color=-99
        elif Z[i]<6.0:
            color=MagRef1[i]-Mag[i]
            if MagRef1[i]==Mag[i]==-99:
                color=-99
        else:
            color=MagRef2[i]-Mag[i]
            if MagRef2[i]==Mag[i]==-99:
                color=-99
                

        ObjectGalaxy = Galaxy(imgname,{'ID':ID[i],'RA':RA[i],'DEC':DEC[i],'z':Z[i],'zflag':Zflag[i],'hsize':hsize,'color':color,'threshold':t,'sblimit':sblim})
        ObjectGalaxy.compute_CAS()
        ObjectGalaxy.compute_GiniM20()
        ObjectGalaxy.compute_MID()
        ObjectGalaxy.compute_PsiTXi()
        ObjectGalaxy.compute_filamentarity()
        if args.nosaving:
            break
        else:
            ObjectGalaxy.write_line_results(table_out)
        if args.verbose:
            print "-------------------------------------------------------------------------------------------------------------------------------------"%ID[i] 
        
    
    if args.nosaving:
        pass
    else:
        table_out.close()
    
