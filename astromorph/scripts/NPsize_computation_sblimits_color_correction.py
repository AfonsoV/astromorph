import numpy as np
import matplotlib.pyplot as mpl
import cosmology as cosmos
import mod_imports as mi
import argparse
import string
import pyfits
import subprocess as sp
import sys

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

def exit_code(event):
    if event.key=='escape':
        sys.exit()
    if event.key=='q':
        mpl.close('all')

def compute_size(image_list,pixscales,ra,dec,hsize,threshold,fractions,sblimit,color,size=6,safedist=1.0):

    nimages = len(image_list)
    Sizes = np.zeros([len(fractions)+1,nimages])
    Fluxes = np.zeros([len(fractions)+1,nimages])
    segflags = np.zeros(nimages)

    hsize_pix = [int(hsize/pixscales[k])/2 for k in range(nimages)]

    image_stamps = [load_data(image,ra,dec,hs) for image,hs in zip(image_list,hsize_pix)]
    headers = [pyfits.getheader(image) for image in image_list]

#    if np.amax(image_stamp)==np.amin(image_stamp):
#        return Sizes - 99,-9


    zp = np.zeros(nimages)
    sblimits=np.zeros(nimages)
    for k in range(nimages):
        PHOTFLAM=float(headers[k]['PHOTFLAM'])
        PHOTPLAM=float(headers[k]['PHOTPLAM'])
        zp[k]=-2.5*np.log10(PHOTFLAM)-5*np.log10(PHOTPLAM)-2.408

    sblimits[0]=sblimit
    for k in range(nimages-1):
        ColorCorretion = 10**(0.4*(color[k]-(zp[0]-zp[k+1])))
        sblimits[k+1] = sblimit*ColorCorretion
        if args.verbose:
            print("Color Correction Term: %.5f"%ColorCorretion)
            print("New Surface Brightness Limit: %.5f counts/s/arcsec**2"%sblimits[k+1])

    dilate = define_structure(size)

    for k in range(nimages):
        if np.amax(image_stamps[k])==np.amin(image_stamps[k]):
            segflag= -9
        else:
            segmap = mi.gen_segmap_sbthresh(image_stamps[k],hsize_pix[k],hsize_pix[k],sblimits[k],pixscales[k],thresh=threshold,Amin=args.areamin,all_detection=True)
            single_source_map = mi.select_object_map(hsize_pix[k],hsize_pix[k],segmap,pixscale=pixscales[k])
            image_stamps[k],single_source_map,imglag,segflag = mi.image_validation(image_stamps[k],single_source_map,pixscales[k],safedist)

        if np.abs(segflag)!=0:
            Sizes[:,k]-=99
            segflags[k]=segflag
        else:
            if args.nodilation:
                dilated_map = single_source_map
            else:
                dilated_map = mi.sci_nd.binary_dilation(single_source_map,structure=dilate).astype(np.int32)

            for i in range(len(fractions)):
                Sizes[i,k],TotalArea,Fluxes[i,k],TotalFlux = compute_fraction_area(image_stamps[k],dilated_map,fractions[i])
            Sizes[i+1,k]=TotalArea
            Fluxes[i+1,k]=TotalFlux

            if args.nosaving:
                mpl.rcParams['image.cmap']='gist_stern_r'
                segmap[segmap!=0]/=1
                fig,ax=mpl.subplots(2,3,figsize=(20.6,12))
                mpl.subplots_adjust(wspace=0.2,hspace=0.02)
                ax=ax.reshape(np.size(ax))
                ax[0].imshow(np.sqrt(np.abs(image_stamps[k])),cmap='hot')
                mi.gen_circle(ax[1],hsize_pix[k],hsize_pix[k],hsize_pix[k]*(2*np.sqrt(2)*0.5/args.size),color='red')
                ax[1].imshow(segmap)
                ax[2].imshow(single_source_map)

                flux_map = np.zeros(image_stamps[k].shape)
                flux_map[dilated_map==1]=3
                for j in range(len(Fluxes[:,k])-2,-1,-1):
                    flux_map[image_stamps[k]>Fluxes[j,k]]=(len(Fluxes[:,k])-j)*4
                flux_map*=dilated_map
                imax,imin,jmax,jmin = mi.find_ij(dilated_map)
                ax[3].imshow(flux_map[imin:imax,jmin:jmax],aspect='equal',cmap='gist_heat_r')
                ax[3].text(12,9,'20',color='white',va='center',ha='center',weight='heavy')
                ax[3].text(10,7.5,'50',color='white',va='center',ha='center',weight='heavy')
                ax[3].text(8.5,5.5,'80',color='white',va='center',ha='center',weight='heavy')
                ax[3].text(6,3,'100',color='white',va='center',ha='center',weight='heavy')

        #        masked_neighbours = mi.sci_nd.binary_dilation(segmap - single_source_map,structure=dilate)
                compute_fraction_area(image_stamps[k],dilated_map,1.0,flux_sampling=500,draw_ax=ax[4])

                ax[5].imshow(dilated_map)

                for i in range(len(ax)):
                    if i!=4:
                        ax[i].set_xticks([])
                        ax[i].set_yticks([])

                fig.text(0.95,0.50,r'$\curvearrowright$',va='center',ha='center',weight='heavy',rotation=-90,fontsize=150)
        #        fig.savefig('size_thresholding_example.png')
                fig.canvas.mpl_connect('key_press_event',exit_code)

    if args.nosaving:
        mpl.show()


    return Sizes,segflags

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Computation of non-parametric values for VUDS galaxies.")
    parser.add_argument('-i','--image',metavar='img1,img2,...',type=str,help="Field Image on which to compute the non-parametric values")
    parser.add_argument('-c','--catalog',metavar='catname',type=str,help="The input catalog of galaxies to be included in the analysis")
    parser.add_argument('-f','--fractions',metavar='f1,f2,...',type=str,default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",help="Flux fractions to compute galaxy area.")
    parser.add_argument('-t','--tilesmatch',metavar='filename',type=str,help="File containing information on which tile is each galaxy")
    parser.add_argument('-r','--size',metavar='size',type=float,default=10,help="The size (in arcseconds) of the stamp image. Default: 10")
    parser.add_argument('-p','--pixscale',metavar='p1,p2,...',type=str,help="The pixel scale of the image")
    parser.add_argument('-I','--ident',metavar='ID',type=int,help="The galaxy on which to run all computations. If none given, run for all.")
    parser.add_argument('-T','--threshold',metavar='thresh',type=float,default=3.0,help="The threshold of te segmentation map")
    parser.add_argument('-A','--areamin',metavar='size',type=int,default=5,help="The minimium area above which to consider a positive detection")
    parser.add_argument('-K','--ksky',metavar='kappa',type=float,default=2.5,help="The default sky threshold for segmentation map for images")
    parser.add_argument('-v','--verbose',action='store_true',help="If present outputs values to terminal as well.")
    parser.add_argument('-N','--nosaving',action='store_true',help="If present plots information onto the screen and does not write to any table.")
    parser.add_argument('-F','--fixthresh',action='store_true',help="If present: use fixed threshold, else use variable treshold with redshift.")
    parser.add_argument('-S','--sigma0',metavar='sigma_0',type=float,default=3.0,help="The sigma anchor value at redshift 5")
    parser.add_argument('--nodilation',action='store_true',help="If present: does not perform binary dilation of the segmentation map.")
    parser.add_argument('--colors',metavar='f1,f2,...',type=str,help="Color differences between galaxies in bands.")

    args = parser.parse_args()



    fieldimgs = [n for n in args.image.split(',')]
    pixelscales = [float(p) for p in args.pixscale.split(',')]
#    tilematches = [t for t in args.tilesmatch.split(',')]

    assert len(fieldimgs)==len(pixelscales)

    samplecat = args.catalog
    match_tiles = args.tilesmatch
    startid=args.ident

    colnumbers=find_columns(samplecat)
    ID,RA,DEC,Z,Zflag = np.loadtxt(samplecat,unpack=True,dtype={'names':('a','b','c','d','e'),'formats':('i8','f4','f4','f4','i4')},usecols=colnumbers)
    hsize = args.size

    fractions = [np.float32(f) for f in args.fractions.split(',')]

    fields,surveys,bands=[],[],[]
    for imagename in fieldimgs:
        check= np.array(['band' in i for i in imagename.split('/')])
        if 'CFHTLS' in imagename:
            band = string.upper(imagename.split('_')[-4])+'band'
            index=3
        else:
            index = np.where(check==True)[0]
            band= imagename.split('/')[index]

        field = imagename.split('/')[index-1]
        survey = imagename.split('/')[index-2]
        if survey =='..':
            survey = imagename.split('/')[index-1]

        fields.append(field)
        surveys.append(survey)
        bands.append(band)

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
        table_pointers=[]
        for t in range(len(fieldimgs)):
            table_name="tables_sblimits_CC/%ssizes_nonparametric_%s_%s_%s_%s_%.2f.txt"%(prefix,surveys[t],fields[t],bands[t],mode,Tmode)
            if args.verbose:
                print("Saving results to %s"%table_name)
            table_pointers.append(open(table_name,"w"))

            table_pointers[t].write("#ID\tRA\tDEC\tZ\tZflag\t")
            table_pointers[t].write("\t".join(["T_%i[pix]"%(round(f*100.0,0)) for f in fractions]))
            table_pointers[t].write("\tT_100[pix]\t")
            table_pointers[t].write("\t".join(["T_%i[kpc]"%(round(f*100.0,0)) for f in fractions]))
            table_pointers[t].write("\tT_100[kpc]\tSegFlag\n")

    start=0
    T=[]
    sblim=np.float(sp.check_output("awk '$1=="+'"%s"'%fieldimgs[0]+"{print $3}' sblimits.dat",shell=True))
    if args.verbose:
        print('Surface Brightness Limit: %.5f counts/s/arcsec**2'%sblim)


    colors = [float(c) for c in args.colors.split(',')] #### ECDFS median color difference
#    colors=[1.50] #### COSMOS sensitivity color difference

    imagelist=list(fieldimgs)
    for i in range(len(ID)):

        if not (ID[i] == startid) and (start==0) and (startid!=None):
            continue
        else:
            start=1

        if args.verbose:
            print("---------------------------------------------------------> VUDS %i <---------------------------------------------------------"%ID[i])
        if match_tiles == None:
            pass
        else:
            tilename=sp.check_output("awk '$1==%s {print $2}' %s"%(ID[i],match_tiles),shell=True)

            try:
                tilename=tilename.split()[0]
            except IndexError as err:
                if args.verbose:
                    print(err)
                continue

            for t in range(len(fieldimgs)):
                if not 'fits' in fieldimgs[t]:
                    imagelist[t] ='%s/%s'%(fieldimgs[t],tilename)

        if args.fixthresh:
            thr = args.threshold
        else:
            sigma_0=args.sigma0
            thr = sigma_0 * ((1+Z[i])/(1+4.0))**(-3)



        galaxy_sizes,segflags = compute_size(imagelist,pixelscales,RA[i],DEC[i],hsize,thr,fractions,sblim,colors)
        da=cosmos.angular_distance(Z[i],pars=None)*1000

        for t in range(len(fieldimgs)):

            galaxy_physic_sizes = np.array([(N*(pixelscales[t]**2)*(180/np.pi*3600)**(-2)*(da**2)) for N in galaxy_sizes[:,t]])
            galaxy_physic_sizes[galaxy_physic_sizes<0] = -99.0

            if args.verbose:
                print(""" threshold = %8.4f @ z = %.3f d_a=%.3f Mpc)
                Pixel  Sizes %s
                Physic Sizes %s
                Segmentation Flag %i
                """%(thr,Z[i],da/1000.,"\t".join(["%10i"%(s) for s in galaxy_sizes[:,t]]),"\t".join(["%10.5f"%(s) for s in galaxy_physic_sizes]) ,segflags[t] ))

            if not args.nosaving:
                table_pointers[t].write("%10i\t%12.8f\t%12.8f\t%5.3f\t%2i\t"%(ID[i],RA[i],DEC[i],Z[i],Zflag[i]))
                table_pointers[t].write("\t".join(["%10.5f"%(s) for s in galaxy_sizes[:,t]]))
                table_pointers[t].write("\t")
                table_pointers[t].write("\t".join(["%10.5f"%(s) for s in galaxy_physic_sizes]))
                table_pointers[t].write("\t%i\n"%segflags[t])

        if args.verbose:
            print("-------------------------------------------------------------------------------------------------------------------------------------"%ID[i])


    if not args.nosaving:
        for t in range(len(fieldimgs)):
            table_pointers[t].close()
