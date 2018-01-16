from mod_imports import *

"""
8/10/2013  - covering factor parameter added computes area segemntation map wih respect to a circle enclosing it
21/10/2013  - Correction to the covering factor by asnp.signing the diameter of the circle to the maximum distance between the vertices of the segmentation map
07/11/2013  - Creation of SBprofile on a pixel by pixel basis following the idea of Wuyts et al. 2012
08/11/2013  - Created as a separate file to accomodate additional morphological measures in addition to the 4 systems
"""
def baricenter_image(img,segmap):
    """
    Compute the center of a given image weighted by the flux in each pixel.
    """

    N,M=img.shape
    X,Y=np.meshgrid(range(M),range(N))

    gal=abs(img*segmap)

    X_gal = X[gal>0]
    Y_gal = Y[gal>0]

    Npix = np.size(gal[gal>0])

    xc = 0
    yc = 0

    for i in range(Npix):
        
        xc += X_gal[i]*gal[Y_gal[i],X_gal[i]]
        yc += Y_gal[i]*gal[Y_gal[i],X_gal[i]]

    xc *= (1/np.sum(gal))
    yc *= (1/np.sum(gal))

    return xc,yc

    
def sbprofile(img,segmap):
    """ Code to generate a 'surface brightness profile' based on the position
    of individual spatial bins (by default a spatial bin = 1 pixel). Surface brightness
    values and distance values are normalized to a specific radius (half-ligh, petrosian,...)
    and the SB is normalized by the total surface brightness inside that radius.
    """
    import CAS as c

    xc,yc = baricenter_image(img,segmap)
    xc,yc = np.where(img==np.amax(img))
    
    rp = c.petrosian_rad(img,xc,yc)

    IS=[]
    RS=[]
    gal = img*segmap
    Npix = np.size(gal[gal!=0])
    dmat,ds = distance_matrix(xc,yc,img)
    Ip = np.sum(gal[dmat<rp])/Npix
    imax,imin,jmax,jmin = find_ij(segmap)

    for i in range(imin,imax):
        for j in range(jmin,jmax):
            if gal[i,j]==0:
                continue
            IS.append(gal[i,j]/Ip)
            RS.append(dmat[i,j]/rp)
            
    return RS,IS

def filamentarity(smap):
    """ Defined to compute the area (in pixel) of the galaxy as defined from its
    segmentation map and compare it to the minimum area of the circle that encloses
    the galaxy. (see Matsuda et al. 2011)
    """
    try:
        imax,imin,jmax,jmin=find_ij(smap)
        #sides=[imax-imin,jmax-jmin]

        d2=[]
        border=all_vertices(smap)
        xpos,ypos = np.where(border==1)
        for i in range(len(xpos)):
            p1=(xpos[i],ypos[i])
            for j in range(i+1,len(xpos)):
                p2=(xpos[j],ypos[j])
                d2.append(dist(p1[0],p1[1],p2[0],p2[1]))

        dmax=max(d2)

        area_circle = np.pi*(dmax)**2/4.0
        area_galaxy = np.size(smap[smap>0])
        cf = area_galaxy/area_circle
#        print "Covering Factor = %.2f"%(cf*100)
    except ValueError as err:
        print err
        cf=0.0

    return cf


if __name__=='__main__':
    testvar = 'f'
    
    im1 = 'acs1.fits'
    im2 = 'acs2.fits'

    if testvar == 'sb':
        img1 = pyfits.getdata(im1).astype(float64)
        img2 = pyfits.getdata(im2).astype(float64)
        
        smap1= gen_segmap_sex(im1,3).astype(float64)
        smap2= gen_segmap_sex(im2,3).astype(float64)

        xc1,yc1=np.where(img1==np.amax(img1))
        xc2,yc2=np.where(img2==np.amax(img2))

        IMG1,MAP1,NC1=make_stamps(xc1,yc1,img1,smap1,fact=3)
        IMG2,MAP2,NC2=make_stamps(xc2,yc2,img2,smap2,fact=3)

        xc1,yc1 = baricenter_image(IMG1,MAP1)
        yc2,xc2 = baricenter_image(IMG2,MAP2)
        dmat1,ds = distance_matrix(xc1,yc1,IMG1)
        dmat2,ds = distance_matrix(xc2,yc2,IMG2)
        
        RS1,IS1 = sbprofile(IMG1,MAP1)
        RS2,IS2 = sbprofile(IMG2,MAP2)

        nrows=2
        ncols=4
        ax=make_subplot(nrows,ncols,width=23,height=12)
        ax[0].imshow(IMG1)
        ax[1].imshow(MAP1)
        ax[2].imshow(dmat1)
        ax[3].plot(RS1,IS1,'bo',alpha=0.8)
        ax[3].loglog()
        ax[ncols].imshow(IMG2)
        ax[ncols+1].imshow(MAP2)
        ax[ncols+2].imshow(dmat2)
        ax[3].plot(RS2,IS2,'ro',alpha=0.8)
        ax[3].set_xlim(.03,3)
        show()

    if testvar == 'f':
        field='cosmos'
        imgdir='/Users/bribeiro/Documents/PhD/VUDS/sample_magI_175_25_z3_4_flags_3_4'
        imgcat='cesam_vudsdb_%s.txt'%field
        GAL=153
        ID,RA,DEC,Z = np.loadtxt('%s/%s'%(imgdir,imgcat),unpack=True,usecols=[0,3,4,5],dtype={'names':('IDs','RAs','DECs','Zs'),'formats':('i8','f4','f4','f4')})
        path="%s/%s/%i"%(imgdir,field,ID[GAL])
        test_image=get_name_fits(path,'acs','I')


        ORI = pyfits.getdata(test_image)
        MAP = gen_segmap_sex(test_image).byteswap().newbyteorder()

        ys,xs,es,thetas,obj_num = get_sex_pars(165,165)
        XC,YC,e,theta=xs[obj_num],ys[obj_num],es[obj_num],thetas[obj_num]
        IMG,SMAP,NC=make_stamps(XC,YC,ORI,MAP,fact=2)

        fig,ax=mpl.subplots()
        ax.imshow(IMG*SMAP)
        cf=filamentarity(SMAP)
        imax,imin,jmax,jmin=find_ij(SMAP)

##        rect=np.array([[jmin,imin],[jmin,imax],[jmax,imax],[jmax,imin],[jmin,imin]])-0.5
##        P=Polygon(rect,fill=False,linewidth=2,color='white')
##        ax.add_artist(P)
        
        d2=[]
        border=all_vertices(SMAP)
        xpos,ypos = np.where(border==1)
        for i in range(len(xpos)):
            p1=(xpos[i],ypos[i])
            for j in range(i+1,len(xpos)):
                p2=(xpos[j],ypos[j])
                d2.append(dist(p1[0],p1[1],p2[0],p2[1]))

        dmax=max(d2)
        gen_circle(ax,(jmin+jmax)/2.-0.5,(imin+imax)/2-0.5,dmax/2)
        mpl.show()
