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
#from pyraf import iraf
import numpy.random as npr
#import os
import cPickle as pickle
import MID

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
#        fig.canvas.mpl_('key_press_event',exit_code)
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
    
def get_segmap_level(image,segmap,fraction,flux_sampling=1000):
    
    if np.size(image[segmap==1])==0:
        return -99
        
    max_flux = np.amax(image[segmap==1])
    min_flux = np.amin(image[segmap==1])


    total_flux = np.sum(image[(segmap==1)*(image>=0)])    

    
    area_flux=0

    level_map = np.zeros(image.shape)
    
    if fraction==1.0:
        return segmap

    for ft in np.linspace(max_flux,min_flux,num=flux_sampling):
        area_flux = np.sum(image[(segmap==1)*(image>=ft)])
                
        if area_flux>=fraction*total_flux:
            level_map[(segmap==1)*(image>=ft)]=1
            break
    
    return level_map
            

    
   
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
        image_stamp = mi.sci_nd.gaussian_filter(image_stamp,sigma=1.0)
        
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
    

def clean_map(map_label,minarea=10):
    max_label=np.amax(map_label)
    for s in range(1,max_label+1):
        if np.size(map_label[map_label==s])<minarea:
            map_label[map_label==s]=0
        else:
            continue
    
    return mi.sci_nd.label(map_label)

def galaxy_map(image,segmap,zeropoint,sky,factor,mag_lim=26):
    galmap=np.zeros(segmap.shape,dtype=np.int16)
    max_label=np.amax(segmap)
    i=1
    for s in range(1,max_label+1):
        mag = -2.5*np.log10(np.sum(factor*(image-sky)[segmap==s]))+zeropoint
        if mag<mag_lim:
            galmap[segmap==s]=i
            i+=1
        else:
            galmap[segmap==s]=0
    return galmap

def get_clump_stats(image,clump_map,zeropoint):

    nclumps = np.amax(clump_map)
    positions_bar = np.zeros([nclumps,2])
    positions_max = np.zeros([nclumps,2])
    mags=np.zeros([nclumps])
    size=np.zeros([nclumps])
    for i in range(nclumps):
        single_clump_map=clump_map.copy()
        single_clump_map[clump_map!=(i+1)]=0
        single_clump_map[clump_map==(i+1)]=1  
        
        positions_bar[i,:] = mi.barycenter(image,single_clump_map)
        positions_max[i,:] = np.where(image*single_clump_map==np.amax(image*single_clump_map))
        mags[i] = -2.5*np.log10(np.sum(image[single_clump_map==1]))+zeropoint
        size[i] = np.size(single_clump_map[single_clump_map==1])
    
    return mags,positions_bar,positions_max,size
    
    
    
def get_clump_stats_imap(image,intensity_map,zeropoint):
    
    nregs = int(np.amax(intensity_map))
    MAGS,POSM,POSB,SIZES=[],[],[],[]
    for i in range(2,nregs+1):
 
        single_clump_map=intensity_map.copy()
        single_clump_map[intensity_map!=(i)]=0
        single_clump_map[intensity_map==(i)]=1  

       
        mag = -2.5*np.log10(np.sum(image[single_clump_map==1]))+zeropoint
        posimax = np.where(image==np.amax(image[single_clump_map==1]))
        posibar = mi.barycenter(image,single_clump_map)
        size = np.size(single_clump_map[single_clump_map==1])
        
        MAGS.append(mag)
        POSB.append(posibar)
        POSM.append([posimax])
        SIZES.append(size)
    
    return np.array(MAGS),np.array(POSM),np.array(POSB),np.array(SIZES)
    
def find_pairs_and_clumps(image_stamp,redshift,galmag,color,hsize,threshold,fractions,sblimit,pixelscale,zeropoint,ksky=3.0,Areamin=10,Aperture=0.5,no_dilation=True,degrade=None,size=5,safedist=1.0,title=None,plot_results=False,segmap_output=False,erosion=[3],verbose=False,ident=None):
    

    if np.amax(image_stamp)==np.amin(image_stamp):
        if verbose:
            print("Invalid data values: %.4f,%.4f"%(np.amax(image_stamp),np.amin(image_stamp)))
        return {}
    
    dilate = define_structure(size)
    sky_med,sky_std = mi.sky_value(image_stamp,ksky)

    
    if args.error:
        factor=-1.0
    else:
        factor=1.0

    if degrade is not None:
        N,M=image_stamp.shape
        image_stamp = mi.rebin2d(image_stamp,int(N/degrade),int(M/degrade),flux_scale=True)
        pixelscale*=degrade
        
    if no_dilation:
        image_smooth = mi.sci_nd.gaussian_filter(image_stamp,sigma=1.0)
        
    if np.abs(color)<10:    
        corrected_thresh = color_correction(threshold,color)
        new_sblimit = corrected_thresh*sblimit
        if verbose:
            print("Color=%.4f\t old sb limit = %.5f\t new sb limit = %.5f counts/s/arcsec**2"%(color,threshold*sblimit,new_sblimit))
        threshold=corrected_thresh
    elif np.abs(color)>10:
        if verbose:
            print("Invalid color value: %.4f"%color)
        return {}
         
    segmap = mi.gen_segmap_sbthresh(factor*(image_smooth-sky_med),hsize,hsize,sblimit,pixelscale,thresh=threshold,Amin=Areamin,all_detection=True)
    single_source_map = mi.select_object_map_connected(hsize,hsize,factor*image_smooth,segmap,pixscale=pixelscale,radius=Aperture)
    image_smooth,single_source_map,imglag,segflag = mi.image_validation(image_smooth,single_source_map,pixelscale,safedist)    
    

    if no_dilation:
        dilated_map = single_source_map
    else:
        dilated_map = mi.sci_nd.binary_dilation(single_source_map,structure=dilate).astype(np.int32)        
        

    gal_selection = galaxy_map(image_stamp,segmap,zeropoint,sky_med,factor)
    ngals=np.amax(gal_selection)
    FullSet={}
    if verbose:
        print 'Ngals=%i'%ngals
    
    for i in range(ngals):
        single_gal_map=gal_selection.copy()
        single_gal_map[gal_selection!=(i+1)]=0
        single_gal_map[gal_selection==(i+1)]=1


        
        Imap,LM =  MID.local_maxims(factor*image_smooth,single_gal_map)
        Mimap,PMimap,PBimap,Simap=get_clump_stats_imap(factor*image_stamp,Imap,zeropoint)
        
        
        nclumps = len(Mimap)
        nclumpsbright = np.size(Mimap[Mimap<28])
        Xcen,Ycen = mi.barycenter(factor*image_smooth,single_gal_map)
        DistsSingle=np.zeros(nclumps)
        for n in range(nclumps):
            DistsSingle[n] = pixelscale*np.sqrt((PBimap[n,0]-Xcen)*(PBimap[n,0]-Xcen)+(PBimap[n,1]-Ycen)*(PBimap[n,1]-Ycen))

        if verbose:
            print '\t %i ----> \t nclumps=%i (m<28: %i)'%(i,nclumps,nclumpsbright)

        
#        nregs=[]
#        nregs2=[]
##        fractions = [0.2,1.0]#np.linspace(0,1,101)
#        for f in fractions:
#            S = get_segmap_level(image_smooth,single_gal_map,f)
#
#            clump_map_full,nr= mi.sci_nd.label(S)
#            clump_map_clean,nr2 = clean_map(clump_map_full,minarea=Areamin)
#            M,Pb,Pm,Sc=get_clump_stats(image_stamp,clump_map_clean,zeropoint)
#            GalMags[str(f)]=M
#            GalPositionsBar[str(f)]=Pb
#            GalPositionsMax[str(f)]=Pm
#            GalSizes[str(f)]=Sc
#            
#            nregs.append(nr)
#            nregs2.append(nr2)
#            
#    
#        for f in fractions:
#            FP = GalPositionsBar[str(f)]
#            Xcen,Ycen=GalPositionsBar['1.0'][0]
#            nclumps=np.shape(FP)[0]
#            
#            DistsSingle=np.zeros(nclumps)
#            for n in range(nclumps):
#                DistsSingle[n] = pixelscale*np.sqrt((FP[n,0]-Xcen)*(FP[n,0]-Xcen)+(FP[n,1]-Ycen)*(FP[n,1]-Ycen))
#            
#            GalDistances[f]=DistsSingle
#            
#            if verbose:
#                print '\t %i ----> f=%.2f \t nclumps=%i'%(i,f,nclumps)

  


      
        if verbose:
            print 50*'='
        
        
        FullSet[i+1]={}
        FullSet[i+1]['galpos']=(Xcen,Ycen)
        FullSet[i+1]['mags']=Mimap
        FullSet[i+1]['posibar']=PBimap
        FullSet[i+1]['posimax']=PMimap
        FullSet[i+1]['dist']=DistsSingle
        FullSet[i+1]['size']=Simap


##    Real_Sizes  = Sizes - SizesPSF
    
    if plot_results:
        print "mag_cat=",galmag       
        print 'sky median = %.5f +- %.6f'%(sky_med,sky_std)
        print "sky_threshold = %.8f (sigma = %.8f)"%(sblimit*threshold,threshold)
        print "Redshift = %.4f"%redshift
        
        mpl.rcParams['image.cmap']='gist_stern_r'
        mpl.rcParams['axes.labelsize']=12
        mpl.rcParams['xtick.labelsize']=10
        mpl.rcParams['ytick.labelsize']=10
        
#        rad,flux,xc,yc,q,theta = compute_sbprofile(image_smooth-sky_med,single_source_map,pixelscale)
#        radPSF,fluxPSF,xcPSF,ycPSF,qPSF,thetaPSF = compute_sbprofile(psf_image-sky_med,single_source_map_psf,pixelscale)

#==============================================================================
# PAPER FIGURE
#==============================================================================
        sidecut=40
        import matplotlib.colors as mpc
        import matplotlib.cm as cm



        
        fig,ax=mpl.subplots(2,ngals,figsize=(25,15))
        ax=ax.reshape(np.size(ax))
        mpl.subplots_adjust(wspace=0)
        jetcmap = cm.get_cmap('YlGnBu_r', 10) #generate a jet map with 10 values 
        jet_vals = jetcmap(np.arange(10)) #extract those values as an array 
        jet_vals[0] = [0., 0, 0., 0.] #change the first value 
        new_jet = mpc.LinearSegmentedColormap.from_list("newjet", jet_vals)
        
        new_image =(factor*(image_stamp))
        hsize_new = new_image.shape[0]/2


        for i in range(ngals):

            axnum=i
            
            single_gal_map=gal_selection.copy()
            single_gal_map[gal_selection!=(i+1)]=0
            single_gal_map[gal_selection==(i+1)]=1

            ax[axnum].imshow(new_image,cmap='YlGnBu_r',extent=(-hsize_new*pixelscale,hsize_new*pixelscale,-hsize_new*pixelscale,hsize_new*pixelscale),vmin=0)
            
            Imap,LM =  MID.local_maxims(factor*image_smooth,single_gal_map)
            Mimap,PMimap,PBimap,Simap=get_clump_stats_imap(factor*image_stamp,Imap,zeropoint)
            ax[axnum+ngals].imshow(Imap,cmap='viridis',extent=(-hsize_new*pixelscale,hsize_new*pixelscale,-hsize_new*pixelscale,hsize_new*pixelscale))
            
            for k in range(len(Mimap)):
                pos = PMimap[k][0]
                if Mimap[k]<28:
                    ax[axnum+ngals].plot((pos[1]-hsize_new)*pixelscale,(pos[0]-hsize_new)*pixelscale,'x',color='white')
                    ax[axnum+ngals].text((pos[1]-hsize_new)*pixelscale,(pos[0]-hsize_new)*pixelscale,'%.3f'%Mimap[k],color='white',fontsize=8,ha='left',va='bottom')
                    
            Xcen,Ycen = FullSet[i+1]['galpos']
            ax[axnum+ngals].set_ylim((Xcen-hsize_new-50)*pixelscale,(Xcen-hsize_new+50)*pixelscale)
            ax[axnum+ngals].set_xlim((Ycen-hsize_new-50)*pixelscale,(Ycen-hsize_new+50)*pixelscale)
            mi.gen_circle(ax[axnum],(Ycen-hsize_new)*pixelscale,(Xcen-hsize_new)*pixelscale,0.75,color='white',lw=2)
            
        
        for eixo in ax:
            eixo.set_xticks([])
            eixo.set_yticks([])
            
#            nregs=[]
#            nregs2=[]
##            fractions = [0.2,0.51.0]#np.linspace(0,1,101)
#            for f in fractions:
#                S = get_segmap_level(image_smooth,single_gal_map,f)
#
#                clump_map_full,nr= mi.sci_nd.label(S)
#                clump_map_clean,nr2 = clean_map(clump_map_full,minarea=5)
#                M,P,Pm,Sc=get_clump_stats(image_stamp,clump_map_clean,zeropoint)
#                nregs.append(nr)
#                nregs2.append(nr2)
#       
#            for f in fractions:
#                FP = GalPositionsBar[str(f)]
#                FPm = GalPositionsMax[str(f)]
#
#                Xcen,Ycen=GalPositionsBar['1.0'][0]
#                nclumps=np.shape(FP)[0]
#                
#                DistsSingle=np.zeros(nclumps)
#                for n in range(nclumps):
#                    DistsSingle[n] = pixelscale*np.sqrt((FP[n,0]-Xcen)*(FP[n,0]-Xcen)+(FP[n,1]-Ycen)*(FP[n,1]-Ycen))
#                
#                GalDistances[f]=DistsSingle
#                
#            ax[axnum].plot(np.array(fractions),nregs,'s-',color='DarkRed',label='All Disconnected')
#            ax[axnum].plot(np.array(fractions),nregs2,'o-',color='Navy',label='A>10pix Disconnected')
#
#            for f,c in zip([0.2,0.5,1.0],['red','lime','cyan','gold']):
#                S = get_segmap_level(image_smooth,single_gal_map,f)
#                labelmap,nr= mi.sci_nd.label(S)
#                ax[axnum+1].hlines(f*((new_image.shape[0]-2*sidecut)/2),5,15,color=c,lw=3)
#                ax[axnum+1].text(10,f*((new_image.shape[0]-2*sidecut)/2),r'$f=%.2f$'%f,color=c,fontsize=12,va='bottom',ha='center')
#                ax[axnum].vlines(f,0,nr,color=c,lw=3)
#                mi.draw_border(ax[axnum+1],S,c)
#            
##        for eixo in ax[1:]:
##            mi.gen_circle(eixo,hsize_new,hsize_new,(2*hsize/args.size)*args.aperture,color='red')
##            eixo.tick_params(labelleft='off',labelbottom='off')
#            
#            ax[axnum].hlines(1,0,1.1,'k',':')
#            ax[axnum].set_xlim(0,1.1)
#            ax[axnum].set_ylim(0,1.4*max(nregs))
#            ax[axnum].set_xlabel(r'$f$')
#            ax[axnum].set_ylabel(r'$N_c$')
        
#        ax[0].legend(loc='best')
#        fig.savefig('clumps_first_pass_INTESNITY.png')                
        fig.canvas.mpl_connect('key_press_event',exit_code)
        mpl.show()
        sys.exit()        

#==============================================================================
# END PAPER FIGURE        
#==============================================================================        
    return FullSet

class Object:
    def __init__(self,ID):
        self.object_ID=ID
    
    

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
    parser.add_argument('-r','--size',metavar='size',type=float,default=7,help="The size (in arcseconds) of the stamp image. Default: 7")
    parser.add_argument('-p','--pixscale',metavar='size',type=float,help="The pixel scale of the image")
    parser.add_argument('-I','--ident',metavar='ID',type=int,help="The galaxy on which to start to run all computations. If none given, run for all.")
    parser.add_argument('--end_ident',metavar='ID',type=int,help="The galaxy on which to send the code.")
    parser.add_argument('-T','--threshold',metavar='thresh',type=float,default=3.0,help="The threshold of te segmentation map")
    parser.add_argument('-a','--aperture',metavar='size',type=float,default=0.5,help="The radius, in arcseconds, of the serach area aperture.")
    parser.add_argument('-A','--areamin',metavar='size',type=int,default=5,help="The minimium area above which to consider a positive detection")
    parser.add_argument('-K','--ksky',metavar='kappa',type=float,default=3.0,help="The default sky threshold for segmentation map for images")
    parser.add_argument('-v','--verbose',action='store_true',help="If present outputs values to terminal as well.")
    parser.add_argument('-N','--nosaving',action='store_true',help="If present plots information onto the screen and does not write to any table.")
    parser.add_argument('-F','--fixthresh',action='store_true',help="If present: use fixed threshold, else use variable treshold with redshift.")
    parser.add_argument('-S','--sigma0',metavar='sigma_0',type=float,default=1.0,help="The sigma anchor value at redshift 4")
    parser.add_argument('-z','--zeropoint',metavar='mag_zp',type=float,help="Magnitude zeropoint of the image")
    parser.add_argument('-d','--degrade',metavar='downsample',type=float,help="If present: degrades image by the factor given.")
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


    if args.error:
        suffix3='_error'
    else:
        suffix3=''
        
    if args.degrade is not None:
        suffix4='_degraded_%i'%args.degrade
    else:
        suffix4=''
                
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
        table_name= "%sClumpStats_nonparametric_intensity_%s_%s_%s_%s_%.2f_am%i_%s%s%s%s.txt"%(prefix,survey,field,band,mode,Tmode,args.areamin,args.color,suffix2,suffix3,suffix4)
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

    lambda_eff={'I':8140e-10,'H':15369e-10,'J':12486e-10} # cm

    hst_rad=150 #cm
    h=6.626e-27 #ergs.s
    c=2.9979e10 #cm/s
    Mpc=3.0856e24 #cm    
    solar_lum = 3.846e33 #ergs/s

    l_eff=lambda_eff[band[:1]]

    FullCatalog={}

    for i in range(len(ID)):
        
        if not (ID[i] == startid) and (start==0) and (startid!=None):
            continue
        else:
            start=1
            
        if endid is None:
            pass
        elif ID[i]==endid:
            break
          
        if Zflag[i]==1 or Zflag[i]<1 or Zflag[i]==21: ### USE ONLY GOOD FLAGS
            continue
        
        if args.verbose:        
            print "---------------------------------------------------------> VUDS %i (%i out of %i) @ kp=%.2f <---------------------------------------------------------"%(ID[i],i+1,len(ID),args.sigma0)

#        if os.path.isfile('psf.fits') and args.ident is None:
#            sp.call('rm psf.fits',shell=True)
#        elif args.ident is None:
#            pass
#        elif os.path.isfile('psf-%i.fits'%args.ident):
#            sp.call('rm psf-%i.fits'%args.ident,shell=True)
#        else:
#            pass
#        
#        
#        if has_psf and args.ident is None:
#            sp.call('cp %s psf.fits'%psf_file,shell=True)
#        elif (has_psf) and (not args.ident is None):
#            sp.call('cp %s psf-%i.fits'%(psf_file,args.ident),shell=True)
            
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
#            if args.focus is None:
#                pass
#            else:
#                focus_value = get_focus(args.focus,tilename)
#                psf_dir='/'.join(match_tiles.split('/')[:-1])+'/PSFs'
#                xc,yc=gfh.get_center_coords(imgname,RA[i],DEC[i])
#                select_PSF(ID[i],psf_dir,focus_value,xc,yc)

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
        
        FullCatalog[ID[i]] = find_pairs_and_clumps(image_stamp,Z[i],Mag[i],color,hsize,t,fractions,sblim,args.pixscale,args.zeropoint,ksky=args.ksky,Areamin=args.areamin,no_dilation=args.nodilation,degrade=args.degrade,Aperture=args.aperture,title="VUDS %i @ z=%.4f [%i]"%(ID[i],Z[i],Zflag[i]),plot_results=args.nosaving,erosion=eroders,verbose=args.verbose,ident=args.ident) 

    if not args.nosaving:
        with open(table_name, 'wb') as outfile:
            pickle.dump(FullCatalog, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        