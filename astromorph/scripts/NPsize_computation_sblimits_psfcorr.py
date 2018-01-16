import numpy as np
import matplotlib.pyplot as mpl
import cosmology as cosmos
import mod_imports as mi
import argparse
import string
import pyfits
import subprocess as sp
import sys
import galfit_helpers as gfh
from pyraf import iraf
import numpy.random as npr
import os

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

def load_data(image,ra,dec,hsize):
    try:
        xcen,ycen = mi.get_center_coords(image,ra,dec,hsize)
        
        hdu=pyfits.open(image)
        image_stamp= hdu[0].data[int(ycen-hsize):int(ycen+hsize)+1, int(xcen-hsize):int(xcen+hsize)+1]
        hdu.close()

    except IndexError as err:
        if args.verbose:
            print err
        image_stamp=np.zeros([hsize*2,hsize*2])
        
    return image_stamp        



    
    
def define_structure(size):
    basic_structure = np.array([[0,1,0],\
                       [1,1,1],\
                       [0,1,0]],dtype=np.int)
    if size==3:
        return basic_structure
    assert size>=3, 'Minimum size 3 is required!'
    assert size%2==1, 'Structure element needs to by odd!'
    structure = np.zeros([size,size])
    structure[size/2-1:size/2+2,size/2-1:size/2+2]=basic_structure
    for i in range(size/3):    
        structure = mi.sci_nd.binary_dilation(structure,basic_structure).astype(np.int)
    return structure
    
def compute_magnitude(image,xc,yc,aperture,magzero,exptime=1.0):
    dmat = mi.distance_matrix(xc,yc,image)[0]
    in_aperture = (dmat<aperture)
    tot_flux = np.sum(image[in_aperture])
    mag = -2.5*np.log10(tot_flux/exptime)+magzero
    return mag
    
def drop_in_sky_region(img,skydir,nregs,pixscale,drop_radius=1.5):
    
    sky_files = sp.check_output('ls %s/sky*fits'%skydir,shell=True).split()
#    npr.shuffle(sky_files)
    
    drop_size = int(drop_radius/pixscale)
    
    xc,yc = np.array(img.shape)/2
    CM='gray_r'
    imgsky_val,imgsky_std= mi.sky_value(img)

    N,M=img.shape
    StoredStamps = np.zeros([N+1,M+1,nregs])
    for i in range(nregs):
        sky_image= pyfits.getdata(sky_files[i])
        sky_val,sky_std= mi.sky_value(sky_image)
        xs,ys=np.array(sky_image.shape)/2

#        fig,ax=mpl.subplots(1,3,figsize=(25,13))
#        ax[0].imshow(img,cmap=CM,vmax=0.01)
#        ax[1].imshow(sky_image,cmap=CM,vmax=0.01)
        
        
        new_image=sky_image.copy()-sky_val
        new_image[xs-drop_size:xs+drop_size,ys-drop_size:ys+drop_size]+=(img[xc-drop_size:xc+drop_size,yc-drop_size:yc+drop_size]-imgsky_val)
                
#        new_image[xs-drop_size:xs+drop_size,ys-drop_size:ys+drop_size]/=np.sqrt(imgsky_std**2+sky_std**2)/imgsky_std
        for g in np.arange(0.3,0.95,0.025):
            ba=mi.sci_nd.gaussian_filter(new_image[xs-drop_size:xs+drop_size,ys-drop_size:ys+drop_size],g)
            if mi.sky_value(ba)[1]/imgsky_std<1:
#                print g
                break
        new_image[xs-drop_size:xs+drop_size,ys-drop_size:ys+drop_size] = mi.sci_nd.gaussian_filter(new_image[xs-drop_size:xs+drop_size,ys-drop_size:ys+drop_size],g)

#        ax[2].imshow(new_image,cmap=CM,vmax=0.01)
#        fig.canvas.mpl_connect('key_press_event',exit_code)
        StoredStamps[:,:,i]=new_image

        
    return StoredStamps
    
        
def compute_fraction_area(image,segmap,fractions,flux_sampling=1000,draw_ax=None,**kwargs):
    
    Areas = np.zeros(len(fractions))
    Fluxes = np.zeros(len(fractions))

    if np.size(image[segmap==1])==0:
        return Areas,Fluxes
        
    max_flux = np.amax(image[segmap==1])
    min_flux = np.amin(image[segmap==1])

    total_flux = np.sum(image[(segmap==1)*(image>=0)])    
    
    area_flux=0
    npix_flux=0

    
    if draw_ax!=None:  
        draw_ax.set_xlabel(r'$f_t\ [\mathrm{e^{-}/s}]$')
        draw_ax.set_ylabel(r'$F_\mathrm{pix}(f>f_t)$')


    k=0
    for ft in np.linspace(max_flux,min_flux,num=flux_sampling):
#    for ft in np.arange(max_flux,min_flux-flux_step,-flux_step):
        area_flux = np.sum(image[(segmap==1)*(image>=ft)])
        npix_flux = np.size(image[(segmap==1)*(image>=ft)])     
        
        if draw_ax!=None:
            if area_flux<0.5*total_flux and ft>0:
                c='g'
            elif area_flux<0.99*total_flux and ft>0:
                c='y'
            else:
                c='r'
            draw_ax.plot([ft],[area_flux],marker='.',color=c,**kwargs)  
        
        F=fractions[k]
        if area_flux>F*total_flux:
            Areas[k]=npix_flux
            Fluxes[k]=area_flux
            if k==len(fractions)-1:
                continue
            else:    
                k+=1        
        
    Areas[-1]=np.size(image[segmap==1])
    Fluxes[-1]=total_flux
    
    if draw_ax!=None:
        draw_ax.hlines(total_flux,min_flux,max_flux,color='k',linestyle='--')
    return Areas, Fluxes
   
#    flux_step=(max_flux-min_flux)/flux_sampling
#    thresh_flux=max_flux
#    total_pix=np.size(image[segmap==1])
#    while area_flux < frac*total_flux:
#        
#        area_flux = np.sum(image[(segmap==1)*(image>=thresh_flux)])
#        npix_flux = np.size(image[(segmap==1)*(image>=thresh_flux)])     
#        
#        if draw_ax!=None:
#            draw_ax.plot([thresh_flux],[npix_flux],**kwargs)  
#        thresh_flux -= flux_step
#        

#        
#    return npix_flux,total_pix,thresh_flux,total_flux

def simulate_psf(galmag,hsize,sky_med,magzero,ident=None):    
    f1=open('galfit_object.temp','w')
    gfh.write_object(f1,'psf',hsize,hsize,galmag,0,0,0,0,1)
    f1.close()
    
    if ident is None:
        psfname='psf.fits'
        outpsfname='psf_model.fits'
    else:
        psfname='psf-%i.fits'%ident
        outpsfname='psf_model-%i.fits'%ident
        

    f2=open('simulPSF_galfit','w')
    gfh.galfit_input_file(f2,magzero,sky_med,hsize*2+1,hsize*2+1,hsize*2+1,0.03,imgname='none',psfname=psfname,outname=outpsfname)    
    f2.close()
    
    sp.call('galfit simulPSF_galfit > galfit.log ',shell=True,stderr=sp.PIPE)
    psf_data = pyfits.getdata(outpsfname)

    return  psf_data
    
def Anel(img,dmat,pixelscale,rin,rout,draw_stuff=False):
    """Compute the flux with an annular region width width [0.8*r , 1.2*r] """

    stest=np.zeros(img.shape)
    stest[dmat<rout]=1

    ltest=np.zeros(img.shape)
    ltest[dmat>rin]=1

    test=ltest*stest*img

    npix = np.size(test[test!=0])
    ann = np.sum(test)
    
    if draw_stuff:
        fig,ax=mpl.subplots(1,2)
        ax[0].imshow(img)
        ax[1].imshow(test)
        mpl.gcf().canvas.draw()
        fig.canvas.mpl_connect('key_press_event',exit_code)
        mpl.show()

##    small = list(img[dmat<upp*r])
##    large = list(img[dmat>low*r])
##
##    ann=0
##    npix=0
##    for s in small:
##        if s in large:
##            ann+=s
##            npix+=1
##            large.remove(s)
##            
##        else:
##            continue
##    if npix==0:
##        return np.sum(small)
    apix = pixelscale*pixelscale
    return ann/(npix*apix)

def exit_code(event):
    if event.key=='escape':
        sys.exit()
    if event.key=='q':
        mpl.close('all')
        
        
    
def compute_ellipse_distmat(img,xc,yc,q=1.00,ang=0.00):

    X,Y = np.meshgrid(range(img.shape[1]),range(int(img.shape[0])))
    rX=(X-xc)*np.cos(ang)-(Y-yc)*np.sin(ang)
    rY=(X-xc)*np.sin(ang)+(Y-yc)*np.cos(ang)
    dmat = np.sqrt(rX*rX+(1/(q*q))*rY*rY)

    return dmat
    
def compute_sbprofile(image,segmap,pixelscale):
    
    bary,barx = mi.barycenter(image,segmap)
    
    XX2,YY2,XY = mi.moments(image,segmap)

    XX2-=barx*barx
    YY2-=bary*bary
    XY -=barx*bary

    
    theta_sky = 180.-mi.theta_sky(XX2,YY2,XY) ## Orientation correction
    axis_ratio = mi.axis_ratio(XX2,YY2,XY)
    if np.isnan(axis_ratio):
        axis_ratio=1.0
    print("q=%.3f"%axis_ratio)

    dmat = compute_ellipse_distmat(image,barx,bary,q=axis_ratio,ang=np.radians(theta_sky))
        
    
    radius = np.arange(0,3/args.pixscale,1)
    fluxes = np.zeros(len(radius)-1)
    for k in range(len(radius)-1):
        rin=radius[k]
        rout=radius[k+1]
        fluxes[k]=Anel(image,dmat,pixelscale,rin,rout,draw_stuff=False)

    return (radius[1:]+radius[:-1])/2.0,fluxes,barx,bary,axis_ratio,theta_sky


def montecarlo_sky(image,segmap,sky_med,sblimit,hsize,ntries=100):
#    high_segmap = mi.gen_segmap_sbthresh(image-sky_med,hsize,hsize,sblimit,args.pixscale,thresh=3,Amin=args.areamin,all_detection=True)

    N,M=image.shape
    Npixs=[]
    for n in range(ntries):
        xr = npr.randint(0,N)
        yr = npr.randint(0,M)
        single_source_map = mi.select_object_map_connected(yr,xr,image,segmap,pixscale=args.pixscale)
        
#        fig,ax=mpl.subplots()
#        ax.imshow(single_source_map,cmap='YlGnBu_r')
#        fig.canvas.mpl_connect('key_press_event',exit_code)        
#        mpl.show()
        Npixs.append(np.size(single_source_map[single_source_map==1]))
    
    Npixs=np.array(Npixs)
    mpl.hist(Npixs,bins=50)
    return np.mean(Npixs),np.std(Npixs),np.median(Npixs)
    
def get_luminosity_image(image,z,pixscale,segmap):
    
    lum_dist = cosmos.luminosity_distance(z)*Mpc #in cm
    ang_dist = cosmos.angular_distance(z)*Mpc ##in cm
    
    pixel_flux = image/(np.pi*hst_rad**2)*(h*c/l_eff)
    pixel_lum  = pixel_flux*4*np.pi*lum_dist**2
    pixel_area = (pixscale/(180/np.pi*3600)*(ang_dist)*(1e6/Mpc))**2 #in pc
    
    fig,ax=mpl.subplots(1,2,figsize=(22,13))
    ax[0].hist((pixel_flux[(pixel_flux>0)*(segmap!=0)]),bins=50,histtype='stepfilled',color='LightSeaGreen')
    fig.text(0.30,0.93,r"$\sum_{i} F_{i} = %.4e\ \mathrm{ergs\ s^{-1}cm^{-2}}$"%((np.sum(pixel_flux[(pixel_flux>0)*(segmap!=0)]))),fontsize=25,color='Teal',weight='bold',ha='center')
    ax[0].set_xlabel(r'$F_{i}\ [\mathrm{ergs\ s^{-1}cm^{-2}}]$')

    ax[1].hist(np.log10(pixel_lum[(pixel_lum>0)*(segmap!=0)]/solar_lum),bins=50,histtype='stepfilled',color='PowderBlue')
    fig.text(0.70,0.93,r"$\log_{10} \left[ \frac{\sum_{i} L_{i}}{L_\odot} \right] = %.4f$"%(np.log10(np.sum(pixel_lum[(pixel_lum>0)*(segmap!=0)])/solar_lum)),fontsize=25,color='SteelBlue',weight='bold',ha='center')
    ax[1].set_xlabel(r'$\log_{10} \left(L_{i}/L_\odot\right)$')

    fig.canvas.mpl_connect('key_press_event',exit_code)    

    lum_density = pixel_lum/pixel_area/solar_lum ## image in L_sun/s/pc^2
    return lum_density

def color_correction(threshold,color):
    ColorCorretion = 10**(0.4*(color))
    thresh_corrected = threshold*ColorCorretion
#    if args.verbose:
#        print("Color Correction Term: %.5f"%ColorCorretion)
#        print("Old Threshold: %.4f, New Threshold: %.4f"%(threshold,thresh_corrected))
    
    return thresh_corrected


def sb_profile_only(image_stamp,galmag,color,sblimit,threshold,redshift,title='',plot_profile=True):

    radius = np.arange(0,3/args.pixscale,1)
    fluxes = np.zeros(np.size(radius))-99

    sky_med,sky_std = mi.sky_value(image_stamp,args.ksky)

    if np.abs(color)<10:    
        corrected_thresh = color_correction(threshold,color)
        new_sblimit = corrected_thresh*sblimit
        
        threshold=corrected_thresh
    else:
        if args.verbose:
            print("Invalid color value: %.4f"%color)
        return radius,fluxes,-99,-99,-99,-99
 
    if args.nodilation:
        image_stamp = mi.sci_nd.gaussian_filter(image_stamp,sigma=2.0)
        
    segmap = mi.gen_segmap_sbthresh(image_stamp-sky_med,hsize,hsize,sblimit,args.pixscale,thresh=threshold,Amin=args.areamin,all_detection=True)
    single_source_map = mi.select_object_map_connected(hsize,hsize,image_stamp,segmap,pixscale=args.pixscale)
    image_stamp,single_source_map,imglag,segflag = mi.image_validation(image_stamp,single_source_map,args.pixscale,1.0)    

    if segflag==1:
        return radius,fluxes,-99,-99,-99,-99
        
    radius,fluxes,barx,bary,q,theta = compute_sbprofile(image_stamp-sky_med,single_source_map,args.pixscale)
    
    if plot_profile:
        segmap[segmap!=0]/=1    
        fig,ax=mpl.subplots(1,2,figsize=(20.6,16))
        ax=ax.reshape(np.size(ax))
        
        fig.suptitle(title)
        ax[0].set_title(r'$K-I=%.4f\ F_\mathrm{correction} = %.5f$'%(color,10**(0.4*(color))))
        ax[1].set_title(r'$k=%.4f\ k\sigma=%.5f\ \mathrm{[e^{-}s^{-1}arcsec^{-2}]}\ \ k_\mathrm{uncorr}=%.4f$'%(threshold,new_sblimit,args.sigma0*((1+redshift)/(1+2.0))**(-3)))
        
        mpl.subplots_adjust(wspace=0.2,hspace=0.02)
        ax[0].imshow(np.sqrt(np.abs(image_stamp)),cmap='hot')    
        mi.gen_ellipse(ax[0],barx,bary,3*(2*hsize/args.size),q,-theta)

        ax[1].plot(radius,fluxes,'o-',color='CornflowerBlue')
                
        ax[1].hlines(sblimit,min(radius),max(radius),linestyle='--',color='Crimson')
        ax[1].hlines(sblimit*threshold,min(radius),max(radius))
        ax[1].set_ylim(1.1*min(fluxes),1.1*max(fluxes))
#        ax[3].hlines(sblimit*1.5*((1+redshift)/(1+4.0))**(-3),min(rad),max(rad),linestyle='-',color='LimeGreen')
                
        ax[1].set_xlabel(r"$r\ [\mathrm{pix}]$")
        ax[1].set_ylabel(r"$f(r)\ [\mathrm{e^{-}s^{-1}arcsec^{-2}}]$")


        fig.canvas.mpl_connect('key_press_event',exit_code)
        mpl.show()        
    
    return radius,fluxes,barx,bary,q,theta
    

def compute_size(image_stamp,redshift,galmag,color,hsize,threshold,fractions,sblimit,pixelscale,zeropoint,ksky=3.0,Areamin=10,Aperture=0.5,no_dilation=True,size=5,safedist=1.0,title=None,plot_results=False,segmap_output=False,erosion=[3],verbose=False,ident=None):
    
    Sizes = np.zeros(len(fractions))
    Fluxes = np.zeros(len(fractions))

    SizesPSF = np.zeros(len(fractions))
    FluxesPSF = np.zeros(len(fractions))
    ErodedSizes = np.zeros(len(erosion))

    image_original = image_stamp.copy()

    if segmap_output and np.amax(image_stamp)==np.amin(image_stamp):
        return Sizes - 99,Fluxes-99,SizesPSF-99,-9,-9,-9,-9,-9,np.zeros(image_stamp.shape),ErodedSizes-99,-99
    elif np.amax(image_stamp)==np.amin(image_stamp):
        if verbose:
            print("Invalid data values: %.4f,%.4f"%(np.amax(image_stamp),np.amin(image_stamp)))
        return Sizes - 99,Fluxes-99,SizesPSF-99,-9,-9,-9,-9,-9,ErodedSizes-99,-99
    
    dilate = define_structure(size)
    sky_med,sky_std = mi.sky_value(image_stamp,ksky)

    if no_dilation:
        image_stamp = mi.sci_nd.gaussian_filter(image_stamp,sigma=1.0)
        
    if np.abs(color)<10:    
        corrected_thresh = color_correction(threshold,color)
        new_sblimit = corrected_thresh*sblimit
#        if verbose:
#            print("K-I=%.4f\t old sb limit = %.5f\t new sb limit = %.5f counts/s/arcsec**2"%(color,threshold*sblimit,new_sblimit))
        threshold=corrected_thresh
    elif segmap_output and np.abs(color)>10:
        return Sizes - 99,Fluxes-99,SizesPSF-99,-1,-9,-9,-9,-9,np.zeros(image_stamp.shape),ErodedSizes-99,-99
    else:
        if verbose:
            print("Invalid color value: %.4f"%color)
        return Sizes - 99,Fluxes-99,SizesPSF-99,-1,-9,-9,-9,-9,ErodedSizes-99,-99
         
    segmap = mi.gen_segmap_sbthresh(image_stamp-sky_med,hsize,hsize,sblimit,pixelscale,thresh=threshold,Amin=Areamin,all_detection=True)
    single_source_map = mi.select_object_map_connected(hsize,hsize,image_stamp,segmap,pixscale=pixelscale,radius=Aperture)
    image_stamp,single_source_map,imglag,segflag = mi.image_validation(image_stamp,single_source_map,pixelscale,safedist)    
    
#    get_luminosity_image(image_stamp,redshift,pixelscale,single_source_map)

#    W=ncut_graph_matrix(image_stamp)
#    mpl.imshow(W,cmap='YlGnBu_r')
#    mpl.show()
 
    if segmap_output and segflag==1:
        return Sizes - 99,Fluxes-99,SizesPSF-99,segflag,-9,-9,-9,-9,np.zeros(image_stamp.shape),ErodedSizes-99,-99
    elif segflag==1:
        if verbose:
            print("No detection from defined threshold")
        return Sizes - 99,Fluxes-99,SizesPSF-99,segflag,-9,-9,-9,-9,ErodedSizes-99,-99

#    sky_mean,sky_std,sky_median = montecarlo_sky(image_stamp,segmap,sky_med,sblimit,hsize)
    sky_mean,sky_std,sky_median = sky_med,sky_std,0.0
#    if verbose:
#        print "Montecarlo Results:\t mean=%.5f\t std=%.5f\t median=%.5f"%(sky_mean,sky_std,sky_median)


    magflag=0
    if galmag<0:
        if verbose:
            print "Magnitude not found in catalog, computing internally!"
        aperture = 0.5/pixelscale
        galmag = compute_magnitude((image_stamp-sky_med),hsize,hsize,aperture,zeropoint)    
        magflag=0.5

    
    if not segmap_output:
        psf_image = simulate_psf(galmag,hsize,sky_med,zeropoint,ident=ident)
        
        segmap_psf = mi.gen_segmap_sbthresh(psf_image-sky_med,hsize,hsize,sblimit,pixelscale,thresh=threshold,Amin=Areamin,all_detection=True)
        single_source_map_psf = mi.select_object_map(hsize,hsize,segmap_psf,pixscale=pixelscale)

#        fig,ax=mpl.subplots(1,3)
#        ax[0].imshow(psf_image)
#        ax[1].imshow(segmap_psf)
#        ax[2].imshow(single_source_map_psf)
#        mpl.show()

        if no_dilation:
            dilated_map_psf = single_source_map_psf
        else:
            dilated_map_psf = mi.sci_nd.binary_dilation(single_source_map_psf,structure=dilate).astype(np.int32)
    
        SizesPSF,FluxesPSF = compute_fraction_area(psf_image-sky_med,dilated_map_psf,fractions,flux_sampling=500)
    else:
        pass
    

    if no_dilation:
        dilated_map = single_source_map
    else:
        dilated_map = mi.sci_nd.binary_dilation(single_source_map,structure=dilate).astype(np.int32)        

    
    Sizes,Fluxes = compute_fraction_area(image_stamp-sky_med,dilated_map,fractions)

    bary,barx = mi.barycenter(image_stamp,dilated_map)
    centroid_deviation = np.sqrt((hsize-barx)*(hsize-barx)+(hsize-bary)*(hsize-bary))*pixelscale

    if not erosion is None:
        for k in range(len(erosion)):
            eroded_map,labels = mi.sci_nd.label(mi.sci_nd.binary_erosion(segmap,structure=define_structure(erosion[k])).astype(np.int16))
            reselection = mi.select_object_map_connected(hsize,hsize,image_stamp,eroded_map,pixscale=pixelscale,radius=Aperture)
#            fig,ax=mpl.subplots(1,3)
#            ax[0].imshow(single_source_map)
#            ax[1].imshow(eroded_map)
#            ax[2].imshow(reselection)
#            mpl.show()
            ErodedSizes[k] = np.size(reselection[reselection==1])
            
#    for i in range(len(fractions)):
#        SizesPSF[i],TotalAreaPSF,FluxesPSF[i],TotalFluxPSF = compute_fraction_area(psf_image-sky_med,dilated_map_psf,fractions[i])
#        Sizes[i],TotalArea,Fluxes[i],TotalFlux = compute_fraction_area(image_stamp-sky_med,dilated_map,fractions[i])
#
#    Sizes[i+1]=TotalArea
#    Fluxes[i+1]=TotalFlux
#
#    SizesPSF[i+1]=TotalAreaPSF
#    FluxesPSF[i+1]=TotalFluxPSF
#    

    
##    Real_Sizes  = Sizes - SizesPSF
    
    if plot_results:
        print "mag_cat=",galmag,"sky=",sky_med        
        print "sky_threshold = %.8f (sigma = %.8f)"%(sblimit*threshold,threshold)

        print "Pixel Galaxy Sizes",Sizes
        print "Pixel PSF Sizes",SizesPSF
        print "Redshift = %.4f"%redshift
        print "Erosions = %s"%(','.join(['%i'%e for e in ErodedSizes]))
        
        mpl.rcParams['image.cmap']='gist_stern_r'
        mpl.rcParams['axes.labelsize']=12
        mpl.rcParams['xtick.labelsize']=10
        mpl.rcParams['ytick.labelsize']=10
        
        rad,flux,xc,yc,q,theta = compute_sbprofile(image_stamp-sky_med,single_source_map,pixelscale)
        radPSF,fluxPSF,xcPSF,ycPSF,qPSF,thetaPSF = compute_sbprofile(psf_image-sky_med,single_source_map_psf,pixelscale)

#==============================================================================
# PAPER FIGURE
#==============================================================================
        sidecut=25
        import matplotlib.colors as mpc
        import matplotlib.cm as cm
        import img_scale

        fig,ax=mpl.subplots(1,4,figsize=(25,9))
        ax=ax.reshape(np.size(ax))
        mpl.subplots_adjust(wspace=0)
#        jetcmap = cm.get_cmap('viridis', 10) #generate a jet map with 10 values 
#        jet_vals = jetcmap(np.arange(10)) #extract those values as an array 
#        jet_vals[0] = [0., 0, 0., 0.] #change the first value 
#        new_jet = mpc.LinearSegmentedColormap.from_list("newjet", jet_vals)
        
        new_image =(image_stamp[sidecut:-sidecut,sidecut:-sidecut])
        hsize_new = new_image.shape[0]/2
        print 'stamp size exampl:',hsize_new*0.03
        ax[0].imshow(img_scale.sqrt(image_original[sidecut:-sidecut,sidecut:-sidecut],scale_max=0.5*np.amax(new_image)),cmap='YlGnBu_r')
        ax[1].imshow(img_scale.sqrt(new_image,scale_max=0.5*np.amax(new_image)),cmap='YlGnBu_r')
        
        MaskSegmap = np.ma.masked_where(segmap==0,segmap,copy=False)
        ax[2].imshow(MaskSegmap[sidecut:-sidecut,sidecut:-sidecut],cmap='viridis')   
        ax[3].imshow(single_source_map[sidecut:-sidecut,sidecut:-sidecut],cmap='gray_r')
#        ax[3].imshow(mi.sci_nd.binary_erosion(single_source_map[sidecut:-sidecut,sidecut:-sidecut],structure=define_structure(3)),cmap='gray_r')
#        ax[4].imshow(mi.sci_nd.binary_erosion(single_source_map[sidecut:-sidecut,sidecut:-sidecut],structure=define_structure(5)),cmap='gray_r')
#        ax[5].imshow(mi.sci_nd.binary_erosion(single_source_map[sidecut:-sidecut,sidecut:-sidecut],structure=define_structure(7)),cmap='gray_r')
        
        ax[0].set_title('F814W image',fontsize=30)
        ax[1].set_title('Smoothing',fontsize=30)
        ax[2].set_title('Segmentation',fontsize=30)
        ax[3].set_title('Selection',fontsize=30)

        for eixo in ax:
            mi.gen_circle(eixo,hsize_new,hsize_new,(2*hsize/args.size)*args.aperture,color='darkorange',lw=4)
            eixo.tick_params(labelleft='off',labelbottom='off')
#        fig.savefig('../SizeEvolution/images/galaxy_selection_example.png')
#        fig.savefig('../SizeEvolution/images/methodology_galaxy_sizes.pdf')
#        fig.savefig('../SizeEvolution/images/methodology_galaxy_sizes.png')
        mpl.show()
        sys.exit()        
#==============================================================================
# END PAPER FIGURE        
#==============================================================================
        segmap[segmap!=0]/=1    
        fig,ax=mpl.subplots(2,3,figsize=(20.6,16))
        ax=ax.reshape(np.size(ax))
        
        fig.suptitle(title)
        ax[0].set_title(r'$K-I=%.4f\ F_\mathrm{correction} = %.5f$'%(color,10**(0.4*(color))))
        ax[1].set_title(r'$k=%.4f\ k\sigma=%.5f\ \mathrm{[e^{-}s^{-1}arcsec^{-2}]}\ \ k_\mathrm{uncorr}=%.4f$'%(threshold,new_sblimit,args.sigma0*((1+redshift)/(1+2.0))**(-3)))
        
        mpl.subplots_adjust(wspace=0.2,hspace=0.02)
        ax=ax.reshape(np.size(ax))
        ax[0].imshow(np.sqrt(np.abs(image_stamp)),cmap='hot')
        mi.gen_circle(ax[1],hsize,hsize,(2*hsize/args.size)*args.aperture,color='red')
        ax[1].imshow(segmap)   
        ax[2].imshow(single_source_map)
        ax[5].imshow(dilated_map)
    
        mi.gen_ellipse(ax[0],xc,yc,3*(2*hsize/args.size),q,-theta)
        ax[3].plot(rad,flux,'o-',color='CornflowerBlue')
        ax[3].plot(radPSF,fluxPSF/(fluxPSF[0]/np.amax(flux)),'o-',color='Crimson')
        
        TotalArea= Sizes[-1]
        ax[3].vlines(np.sqrt(TotalArea/np.pi),1.1*min(flux),1.1*max(flux),linestyle=':',color='Orchid',linewidth=1.5)
        ax[3].vlines(np.sqrt((TotalArea-sky_mean)/np.pi),1.1*min(flux),1.1*max(flux),linestyle='-',color='ForestGreen',linewidth=3)
        

        ax[3].hlines(sblimit,min(rad),max(rad),linestyle='--',color='Crimson')
        ax[3].hlines(sblimit*threshold,min(rad),max(rad))
        ax[3].set_ylim(1.1*min(flux),1.1*max(flux))
#        ax[3].hlines(sblimit*1.5*((1+redshift)/(1+4.0))**(-3),min(rad),max(rad),linestyle='-',color='LimeGreen')
                
        ax[3].set_xlabel(r"$r\ [\mathrm{pix}]$")
        ax[3].set_ylabel(r"$f(r)\ [\mathrm{e^{-}s^{-1}arcsec^{-2}}]$")
#        ax[6].imshow(np.sqrt(np.abs(psf_image)),aspect='equal',cmap='hot')
#        ax[7].imshow(segmap_psf,aspect='equal',cmap='gist_heat_r')
#        ax[8].imshow(dilated_map_psf,aspect='equal',cmap='gist_heat_r')

        flux_map = np.zeros(image_stamp.shape)
        flux_map[dilated_map==1]=3
        for j in range(len(Fluxes)-2,-1,-1):
            flux_map[image_stamp>Fluxes[4]]=(len(Fluxes)-j)*4
        flux_map*=dilated_map        
        
        imax,imin,jmax,jmin = mi.find_ij(dilated_map)
        ax[5].imshow(flux_map[imin:imax,jmin:jmax],aspect='equal',cmap='gist_heat_r')
#        ax[3].text(12,9,'20',color='white',va='center',ha='center',weight='heavy')
#        ax[3].text(10,7.5,'50',color='white',va='center',ha='center',weight='heavy')
#        ax[3].text(8.5,5.5,'80',color='white',va='center',ha='center',weight='heavy')
#        ax[3].text(6,3,'100',color='white',va='center',ha='center',weight='heavy')
    
    
        masked_neighbours = mi.sci_nd.binary_dilation(segmap - single_source_map,structure=dilate)
        compute_fraction_area(image_stamp-sky_med,dilated_map,[1.0],flux_sampling=500,draw_ax=ax[4])


#        ax[6].set_yscale("log", nonposy='clip')
#        ax[7].set_yscale("log", nonposy='clip')
        
    
        for i in range(len(ax)):
            if i==3 or i==4:
                continue
            else:
                ax[i].set_xticks([])
                ax[i].set_yticks([])
    
        fig.text(0.95,0.50,r'$\curvearrowright$',va='center',ha='center',weight='heavy',rotation=-90,fontsize=150)
        fig.savefig('size_thresholding_%s.png'%(title.split()[1]))      

                
        fig.canvas.mpl_connect('key_press_event',exit_code)
        mpl.show()
        
    if segmap_output:
        return Sizes,Fluxes,SizesPSF,segflag+magflag,sky_mean,sky_std,sky_median,threshold,single_source_map,ErodedSizes,centroid_deviation
            
    return Sizes,Fluxes,SizesPSF,segflag+magflag,sky_mean,sky_std,sky_median,threshold,ErodedSizes,centroid_deviation

def get_focus(focus_table,name):
    """"Function helper to get focus value for psf grab"""
    f=open(focus_table)
    table=f.readlines()
    f.close()

    for line in table:
        if line.split()[0]==name:
            focus=float(line.split()[-3])
            break
        else:
            continue
    return focus
    
def select_PSF(ID,psfdir,focus,xc,yc,hsize=50):
    """Function helper to get psf image from focus value"""
    if focus<=-6.5:
        focus=-6.
    if focus<=-8.0:
        focus=-10.
    psf_cat="%s/TinyTim_f%i.cat"%(psfdir,round(focus,0))
    psf_img="%s/TinyTim_f%i.fits"%(psfdir,round(focus,0))
    xs,ys,num=gfh.get_sex_pars(xc,yc,rmax=330,catfile=psf_cat,psf=True)
    X,Y=xs[num],ys[num]
    if args.ident is None:
        iraf.imarith("%s[%i:%i,%i:%i]"%(psf_img,X-hsize,X+hsize,Y-hsize,Y+hsize),'*',1.0,'psf.fits')
    else:
        iraf.imarith("%s[%i:%i,%i:%i]"%(psf_img,X-hsize,X+hsize,Y-hsize,Y+hsize),'*',1.0,'psf-%i.fits'%(args.ident))
    
    return
    

def sigma(z,sigma,zp=4.0,kp=1.0):
   return sigma * kp * ((1+z)/(1+zp))**(-3)

def luminosity_evol(z,zbreak=3.0):
    P23 = -0.360 ## From Reddy&Steidel2009
    P46 = -0.067 ## From Bouwens+2015, see color_and_lum_corrections_report folder for more details
    if z<zbreak:
        return 10**(-0.4*(P23*(z)))
    else:
        return 10**(-0.4*(P46*(z)))*10**(-0.4*(P23*zbreak-P46*zbreak))

def luminosity_evolution(z,zp=2.0,zbreak=3.0):
    return luminosity_evol(z)/luminosity_evol(zp)


def redefine_colormap(original,ncolors):
    import matplotlib.cm as cm
    import matplotlib.colors as mpc
    jetcmap = cm.get_cmap(original, ncolors) #generate a jet map with 10 values 
    jet_vals = jetcmap(np.arange(ncolors)) #extract those values as an array 
    jet_vals[0] = [1., 1., 1., 1] #change the first value 
    newcmap = mpc.LinearSegmentedColormap.from_list("new", jet_vals)
    return newcmap


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
    parser.add_argument('-r','--size',metavar='size',type=float,default=5,help="The size (in arcseconds) of the stamp image. Default: 10")
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


    args = parser.parse_args()

    fieldimg = args.image
    samplecat = args.catalog
    match_tiles = args.tilesmatch
    startid=args.ident
    endid=args.end_ident
    
    colnumbers=find_columns(samplecat,args.magnitude,args.refmagnitude)
    ID,RA,DEC,Z,Zflag,Mag,MagRef1,MagRef2 = np.loadtxt(samplecat,unpack=True,dtype={'names':('a','b','c','d','e','f','g','h'),'formats':('i8','f4','f4','f4','i4','f4','f4','f4')},usecols=colnumbers)
    hsize = (args.size/args.pixscale)/2
    
    fractions = [np.float32(f) for f in args.fractions.split(',')]
    eroders = [np.int16(e) for e in args.erosions.split(',')]
    
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
            table_name= "tables_sblimits_psfcorr_colorcorr/%ssizes_nonparametric_%s_%s_%s_%s_%.2f_ap%.2f_%s%s.txt"%(prefix,survey,field,band,mode,Tmode,args.aperture,args.color,suffix2)
        else:            
            table_name= "tables_sblimits_psfcorr_colorcorr/%ssizes_nonparametric_%s_%s_%s_%s_%.2f-%i.txt"%(prefix,survey,field,band,mode,Tmode,startid)
        if args.verbose:
            print "Saving results to %s"%table_name
        table_out = open(table_name,"w")
            
        table_out.write("#ID\tRA\tDEC\tZ\tZflag\tColor\t")
        table_out.write("\t".join(["T_%i[pix]\tTpsf_%i[pix]"%(round(f*100.0,0),round(f*100.0,0)) for f in fractions]))   
        table_out.write("\t")
        table_out.write("\t".join(["T_%i[kpc]\tTpsf_%i[kpc]"%(round(f*100.0,0),round(f*100.0,0)) for f in fractions]))    
        table_out.write("\t")
        table_out.write("\t".join(["F_%i[mag]"%(round(f*100.0,0)) for f in fractions]))    
        table_out.write("\t")
        table_out.write("\t".join(["T_eroded_%i"%(e) for e in eroders]))    
        table_out.write("\tSKY_mean\tSKY_std\tMCmedian\tDeviation\tSegFlag\tUsedThreshold\n")

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

    lambda_eff={'I':8140e-10,'H':15369e-10,'J':12486e-10} # cm

    hst_rad=150 #cm
    h=6.626e-27 #ergs.s
    c=2.9979e10 #cm/s
    Mpc=3.0856e24 #cm    
    solar_lum = 3.846e33 #ergs/s

    l_eff=lambda_eff[band[:1]]

    for i in range(len(ID)):
        
        if not (ID[i] == startid) and (start==0) and (startid!=None):
            continue
        else:
            start=1
            
        if endid is None:
            pass
        elif ID[i]==endid:
            break
        
        
        if os.path.isfile('tables_sblimits_psfcorr_colorcorr/%i_randomskysizes.txt'%ID[i]) and args.error:
            print("Skipping %i"%ID[i])
            continue

          
        if Zflag[i]==1 or Zflag[i]<1 or Zflag[i]==21: ### USE ONLY GOOD FLAGS
            continue
        
        if args.verbose:        
            print "---------------------------------------------------------> VUDS %i (%i out of %i) @ kp=%.2f <---------------------------------------------------------"%(ID[i],i+1,len(ID),args.sigma0)

        if os.path.isfile('psf.fits') and args.ident is None:
            sp.call('rm psf.fits',shell=True)
        elif args.ident is None:
            pass
        elif os.path.isfile('psf-%i.fits'%args.ident):
            sp.call('rm psf-%i.fits'%args.ident,shell=True)
        else:
            pass
        
        
        if has_psf and args.ident is None:
            sp.call('cp %s psf.fits'%psf_file,shell=True)
        elif (has_psf) and (not args.ident is None):
            sp.call('cp %s psf-%i.fits'%(psf_file,args.ident),shell=True)
            
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
                focus_value = get_focus(args.focus,tilename)
                psf_dir='/'.join(match_tiles.split('/')[:-1])+'/PSFs'
                xc,yc=gfh.get_center_coords(imgname,RA[i],DEC[i])
                select_PSF(ID[i],psf_dir,focus_value,xc,yc)

        zp=2.0
        if args.fixthresh:
            t = args.threshold       
        else:
            sigma_0=args.sigma0
            t = sigma_0 * ((1+Z[i])/(1+zp))**(-3)
        
        if args.evo:
            t*=luminosity_evolution(Z[i],zp)
            
        if Z[i]<6.0:
            color=MagRef1[i]-Mag[i]
            if MagRef1[i]==Mag[i]==-99:
                color=-99
        else:
            color=MagRef2[i]-Mag[i]
            if MagRef2[i]==Mag[i]==-99:
                color=-99
                
        image_stamp = load_data(imgname,RA[i],DEC[i],hsize)
        
        
#==============================================================================
# SBPROFILES ONLY
#==============================================================================
        
#        radius,fluxes,barx,bary,axis_ratio,theta_sky = sb_profile_only(image_stamp,Mag[i],color,sblim,t,Z[i],title="VUDS %i @ z=%.4f [%i]"%(ID[i],Z[i],Zflag[i]),plot_profile=False)
#        
#        f = open("tables_sblimits_psfcorr_colorcorr/%i_SBProfile.txt"%ID[i],'w')
#        f.write("# %.4f %.4f %.4f %.4f\n"%(barx,bary,axis_ratio,theta_sky))
#        for ra,fl in zip(radius,fluxes):        
#            f.write("%.8f\t%.8f\n"%(ra,fl))
#        f.close()
        

#==============================================================================
# Sizes ONLY
#==============================================================================

        
        galaxy_sizes,galaxy_fluxes,psf_sizes,segflag,sky_mean,sky_std,sky_median,used_threshold,erosions,deviation = compute_size(image_stamp,Z[i],Mag[i],color,hsize,t,fractions,sblim,args.pixscale,args.zeropoint,ksky=args.ksky,Areamin=args.areamin,no_dilation=args.nodilation,Aperture=args.aperture,title="VUDS %i @ z=%.4f [%i]"%(ID[i],Z[i],Zflag[i]),plot_results=args.nosaving,erosion=eroders,verbose=args.verbose,ident=args.ident)
        
 
        da=cosmos.angular_distance(Z[i],pars=None)*1000
        galaxy_physic_sizes = np.array([(N*(args.pixscale**2)*(180/np.pi*3600)**(-2)*(da**2)) for N in galaxy_sizes])
        galaxy_physic_sizes[galaxy_physic_sizes<0] = -99.0

        psf_physic_sizes = np.array([(N*(args.pixscale**2)*(180/np.pi*3600)**(-2)*(da**2)) for N in psf_sizes])
        psf_physic_sizes[psf_physic_sizes<0] = -99.0

        galaxy_mags = np.array([-2.5*np.log10(Fgal)+args.zeropoint for Fgal in galaxy_fluxes])
        galaxy_mags[np.isnan(galaxy_mags)]=-99.0 ## transform nans into -99
#==============================================================================
# ERRORS ONLY
#==============================================================================

        if segflag==0 and args.error:

            skydir='COSMOS_ACS'
            nregs=475
            SS = drop_in_sky_region(image_stamp,skydir,nregs,args.pixscale)
#            mpl.imshow(np.sum(SS,axis=2)/nregs,cmap='YlGnBu_r')
#            mpl.show()
#            pyfits.writeto("SuperSkyTest.fits",np.sum(SS,axis=2)/nregs,clobber=True)
    
    
            AA = []
            N,M = SS[:,:,0].shape
            SMaps = np.zeros([N,M,nregs])
            SFlags = np.zeros(nregs)
            for k in range(nregs):
                GS,GF,PS,SF,SkM,SkD,SkMed,UT,SM,ES,DV = compute_size(SS[:,:,k],Z[i],Mag[i],color,hsize,t,fractions,sblim,args.pixscale,args.zeropoint,ksky=args.ksky,Areamin=args.areamin,plot_results=False,segmap_output=True,erosion=eroders)
                print k+1,GS[-1]            
                AA.append(GS)
                SMaps[:,:,k]=SM
    
#                mpl.close('all')
#                f,a=mpl.subplots(1,3,figsize=(25,12))
#                a[0].imshow(SS[:,:,k],vmax=1e-2,cmap='bone')
#                a[1].imshow(SM)
#                sSM =np.sum(SMaps[:,:,:],axis=2)
#                sSM = np.ma.masked_where(sSM==0,sSM,copy=False)
#                a[2].imshow(SS[:,:,k],vmax=1e-2,cmap='bone')
#                a[2].imshow(sSM,cmap='jet')
#                a[1].set_title('%i:%i'%(k+1,np.amax(np.sum(SMaps[:,:,:],axis=2))))
#                f.canvas.mpl_connect('key_press_event',exit_code)
#                for eixo in a:
#                    eixo.tick_params(labelleft='off',labelbottom='off')
#                mpl.show(block=False)
                
                SFlags[k] = SF
            AA = np.array(AA)

            fig,ax=mpl.subplots(3,4,figsize=(25,14))
            ax=ax.reshape(np.size(ax))
            for jj in range(3*4):
                ax[jj].hist(AA[:,jj],histtype='stepfilled')
                ax[jj].vlines(galaxy_sizes[jj],0,nregs/3.0,color='Red',lw=3)
                ax[jj].set_xlabel(r'$T_{%i}$'%(round(100*fractions[jj],0)),fontsize=12)
                ax[jj].tick_params(labelleft='off',labelbottom='off')
            nfig,nax=mpl.subplots(1,3,figsize=(25,12))
            nbad = np.size(SFlags[SFlags!=0])
            print 'bad flags',nbad
            if nbad==nregs:
                continue
            SuperSegMap=np.sum(SMaps[:,:,SFlags==0],axis=2)
            print np.amax(SuperSegMap)
            SuperSegMapSizes=[np.size(SuperSegMap[SuperSegMap>=n]) for n in range(1,nregs+1-nbad)]
            SuperSegMap=np.ma.masked_where(SuperSegMap==0,SuperSegMap,copy=False)
            nax[0].imshow(image_stamp,cmap='hot')
            nax[0].imshow(SuperSegMap,cmap='YlGnBu_r',alpha=0.65)
            nax[1].plot(np.arange(nregs-nbad)+1.0,SuperSegMapSizes,color='Crimson')
            nax[1].set_xlim(nregs-nbad,0)
            nax[1].hlines(galaxy_sizes[-1],0,nregs,color='ForestGreen')
            print SuperSegMapSizes[-1]*(args.pixscale**2)*(180/np.pi*3600)**(-2)*(da**2), SuperSegMapSizes[-5:]
            SSMselect =  SuperSegMap.copy()
            
            SSMselect[SSMselect<nregs-nbad]=0
            SSMselect=np.ma.masked_where(SSMselect==0,SSMselect,copy=False)
            nax[2].imshow(image_stamp,cmap='hot')
            nax[2].imshow(SSMselect,cmap='YlGnBu_r',alpha=0.65)
            nfig.text(0.5,0.95,'Single Map size: %.4f kpc\n Combined Map size: %.4f kpc'%(galaxy_physic_sizes[-1],SuperSegMapSizes[-1]*(args.pixscale**2)*(180/np.pi*3600)**(-2)*(da**2)),fontsize=12,ha='center',va='center')
        
            np.savetxt('tables_sblimits_psfcorr_colorcorr/%i_randomskysizes.txt'%ID[i],AA,fmt='%10i')
            np.savetxt('tables_sblimits_psfcorr_colorcorr/%i_sizepernumberofSegmaps.txt'%ID[i],SuperSegMapSizes,fmt='%10i')
            fig.savefig('tables_sblimits_psfcorr_colorcorr/%i_randomskysizes.png'%ID[i])
            nfig.savefig('tables_sblimits_psfcorr_colorcorr/%i_sizepernumberofSegmaps.png'%ID[i])
            
        
        if args.verbose:
            print """ threshold = %8.4f @ z = %.3f d_a=%.3f Mpc
            Pixel  Sizes %s
            Physic Sizes %s 
            Fraction Mags %s 
            Segmentation Flag %.1f 
            """%(used_threshold,Z[i],da/1000.,"\t".join(["%10i"%(s) for s in galaxy_sizes]),"\t".join(["%10.5f"%(s) for s in galaxy_physic_sizes]),"\t".join(["%5.3f"%(m) for m in galaxy_mags]) ,segflag )
            
        if not args.nosaving:
            table_out.write("%10i\t%12.8f\t%12.8f\t%5.3f\t%2i\t%.4f\t"%(ID[i],RA[i],DEC[i],Z[i],Zflag[i],color))
            table_out.write("\t".join(["%10i\t%10i"%(s1,s2) for s1,s2 in zip(galaxy_sizes,psf_sizes)]))
            table_out.write("\t")
            table_out.write("\t".join(["%10.5f\t%10.5f"%(s1,s2) for s1,s2 in zip(galaxy_physic_sizes,psf_physic_sizes)]))  
            table_out.write("\t")
            table_out.write("\t".join(["%8.5f"%(mag) for mag in galaxy_mags]))  
            table_out.write("\t")
            table_out.write("\t".join(["%10i"%(e) for e in erosions]))
            table_out.write("\t")
            table_out.write("\t".join(["%10.8f"%s for s in [sky_mean,sky_std,sky_median]]))
            table_out.write("\t%.8f\t%.1f\t%.5f\n"%(deviation,segflag,used_threshold))
            
        if args.verbose:
            print "---------------------------------------------------------------------------------------------------------------------------------------------------------"%ID[i] 

    if not args.nosaving:
        table_out.close()




