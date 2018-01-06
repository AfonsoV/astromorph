import sys
from . import utils
from . import plot_utils
import numpy as np
import scipy.ndimage as snd
import astropy.io.fits as pyfits

""" CODE TO COMPUTE CAS STATS (Conselice et al. 2003)
26/08/2013  - Creation of separate document, modules imported separately
28/08/2013  - Distance calc made by external function on mod_imports
            - Petrosian radius implementation (elliptical aperture and optimization needed)
            - Segmentation map included in calc of A and S
29/08/2013  - Add description to functions
            - Improve S to take into account background and to zero out negative flux values.
            - IRAF rotate routine for computation of rotated image implemented
            - Sum of absolute values of original image on CAS_A
            - rotate_iraf previously generates segmapped galaxy image
04/09/2013  - compute distance_matrix on petrosian radius using image cuts centered at xc,yc
05/09/2013  - deal with image borders on petrosian_rad
            - Concentration parameter computed with respect to 1.5 times the petrosian radius
06/09/2013  - Elliptical Apeertures within petrosian radius calculations
            - Added code for visualization of elliptical annulus and apertures in each step of the petro_rad calc (now commented)
            - Correction on Ellipse formula (check later dependence on theta)
10/09/2013  - Angle in petrosian radius determination is now the same as input and not 90-theta_in (transformations done prior to this)
17/09/2013  - Petrosian radius computed in fractional steps
            - Default axis ratio value set to different than zero
19/09/2013  - Small change in way to compute S (now using distance matrix to select excluded pixels)
            - Negative values are set to 0 prior to computing clumpiness parameter
20/09/2013  - Added interpolant option to IRAF rotate rotine
            - Alteration on CAS_A to perform Asymmetry minimization (see Conselice 2000 for details)
            - Minimization of the A parameter implemented on find_center_asymmetry
            - Asymmetry function computes A, and CAS_A computes the minimum value for A by minimizing to the best center.
25/09/2013  - Concentration parameter computed using smaller steps
02/10/2013  - change from os.system to subprocess.check_call to avoid printing errors to shell
10/10/2013  - change in CAS_C to compute flux within elliptical apertures using a distance atrix
14/10/2013  - OPTIMIZATION: compute x*x instead of x**2
04/11/2013  - OPTIMIZATION: changes in Anel function for a 200 faster time of computation
05/11/2013  - New method to find minimum of asymmetry based on section 3.3 of Conselice & Bershady (2000)
            - Alteration on A parameter to faster method of minimation mentioned above
            - Created boxcar_filter function to apply to 2d np.arrays by replacing the value of a pixel by the np.mean of its neighbours defined by the box np.size.
            - Alteration on S parameter to properly account for sky subtraction.
06/11/2013  - Small glitch corrected now on petrosian radius due to nan values arising from small radii
07/11/2013  - Do not use segmentation map on computation of A and S
            - Remove segmap from all S and A related functions
20/07/2014  - Added flags on petrosian radius computation for controlling the 'border reach effect'
            - Change in concentraion computation to avoid ZeroDivisionError when r20 is zero. Now r20 defaults to 1e-10
22/07/2014  - Added Asymmetry/Clumpiness galaxy and Asymmetry/Clumpiness sky to output of the CAS
23/07/2014  - Added cycle on Sky assymetry to smooth the value of A_sky and provide a more robust estimation of Asymmetry
17/09/2014  - Added minimum width value of 2 pixel for boxcar_filter
14/11/2014  - Incorporated segmentation map into CAS_A to save time in computing the extracted sky region and avoid multiple computations in the extract_sky function
"""

####################################################################################################
#                                           ASYMMETRY                                                  #
###################################################################################################

def rotate_180(img):
    """ Simple horizontal and vertical flip of the given image."""
    N,M=img.shape
    R=np.zeros([N,M])
    for i in range(N):
        for j in range(M):
            R[N-1-i,M-1-j]=img[i,j]
    return R

def rotate_scipy(img,xc,yc,angle=180.0):
    N,M=img.shape
    X,Y=(N-1)/2,(M-1)/2
    dx=(xc-X)
    dy=(yc-Y)
    gal=img.copy()
    shifted = snd.shift(gal,[-dx,-dy])
    rotated = snd.rotate(shifted,angle)
##    imshow(rotated);show()
    return shifted,rotated

def Asymmetry_scipy(center,img,sky=None,angle=180):
    """Compute the asymmetry statistic from CAS."""
    xc,yc=center
    O = img

    if sky==None:
        S,R = rotate_scipy(img,xc,yc,angle=angle)
        A = np.sum(abs(O-R))/np.sum(abs(O))
    else:
        sky=sky
        S_sky,R_sky=rotate_scipy(sky,xc,yc,angle=angle)
        A_sky=np.sum(abs(sky-R_sky))/np.sum(abs(O))
        A=A_sky

    return A

def find_center_minA(img,angle=180,sky=None):
    N,M=img.shape

    As=2*np.ones(img.shape)
    for i in range(N):
        for j in range(M):
            As[i,j]= Asymmetry_scipy([i,j],img,angle=angle,sky=sky)

    xc,yc=np.where(As==np.amin(As))
    return xc,yc,np.amin(As)

def CAS_A_scipy(img,sky=None,angle=180):
    xc,yc,A=find_center_minA(img,angle=angle,sky=sky)
    return A


def rotate_iraf(img,xc,yc,angle=180):
    """Rotates an image around (xc,yc) by angle using IRAF rotate routine."""
    from pyraf import iraf
    gal = img
    XCen=xc+1
    YCen=yc+1
    imgname = 'sgal.fits'
    hdu=pyfits.PrimaryHDU(gal)
    hdu.writeto(imgname)
    outname='%s_rot.fits'%imgname.split('.')[-2]
    N,M=img.shape
    #print XCen,YCen
    try:
        iraf.images.imgeom.rotate(imgname,outname,angle,xin=XCen,yin=YCen,xout=XCen,yout=YCen,verbose='no',interpolant='linear',ncols=M,nlines=N)
        R = pyfits.getdata(outname)
    except (iraf.IrafError,ValueError) as e:
        print(xc,yc)
        R=np.zeros(gal.shape)
        print(colorred.format('##### %s #####'%e))
    sp.call('rm %s'%outname,shell=True)
    sp.call('rm %s'%imgname,shell=True)
    return R

def Asymmetry(center,img,sky=None):
    """Compute the asymmetry statistic from CAS."""
    xc,yc=center
    O = img

    if sky==None:
        R = rotate_iraf(img,xc,yc)
        A = np.sum(abs(O-R))/np.sum(abs(O))
        #A = A.reshape(A.np.size)
    else:
        S = sky
        R = rotate_iraf(S,xc,yc)
        A = np.sum(abs(S-R))/np.sum(abs(O))
        #A = A.reshape(A.np.size)
    return A

##def CAS_A(center,img,sky):
##    """ Computes the asymmetry statistic from CAS based on the minimum
##    value of the image and the sky patch.
##    """
##    xc,yc,A_img=find_center_asymmetry(img,center)
##    xc_sky,yc_sky,A_sky=find_center_asymmetry(img,center,sky=sky)
##
##    A = A_img-A_sky
##
##    return A

def CAS_A(center,img,nskytries,originalimage,originalsegmap,**kwargs):
    """ Computes the asymmetry statistic from CAS based on the minimum
    value of the image and the sky patch.
    """
    I,J=int(round(center[0])),int(round(center[1]))
#    t1=t.time()
    i,j,A_img = A_conselice(img,[I,J],Asymmetry_scipy)
#    t2=t.time()
#    print t2-t1
    Askies=np.zeros(nskytries)
    for i in range(nskytries):
#        t3=t.time()
        sky = utils.extract_sky(originalimage,img.shape,originalsegmap,**kwargs)
#        t4=t.time()
        x,y,Asky=find_center_asymmetry(img,center,func=Asymmetry_scipy,sky=sky)
#        t5=t.time()
#        print 'sky,A_sky',t4-t3,t5-t4
        Askies[i]=Asky

#    print 'total',t5-t2
    A_sky = np.mean(Askies)
    A = A_img-Asky
    return A,A_img,A_sky

def test_min_A(img,center,sky=None):
    x0,y0=center
    N,M=img.shape
    xs = np.linspace(x0-N/4,x0+N/4)
    ys = np.linspace(y0-M/4,y0+M/4)
    X,Y = np.meshgrid(xs,ys)
    AS = np.zeros(X.shape)
    for i in range(len(xs)):
        for j in range(len(ys)):
            AS[i,j] = Asymmetry([xs[i],ys[j]],img,sky=sky)
    ax = plot_utils.make_subplot(1,3)
    ax[0].imshow(img)
    C2=ax[2].imshow(AS)
##    ax[2].set_xticks(np.linspace(0,AS.shape[0],len(xs[::5])),[str(x) for x in xs[::5]])
##    ax[2].set_yticks(np.linspace(0,AS.shape[1],len(ys[::5])),[str(y) for y in ys[::5]])

    xc,yc,a=find_center_asymmetry(img,center,sky=sky)
    ax[0].plot([xc],[yc],'wx',markersize=10,markeredgewidth=2)
    ax[0].set_xlim(0,N-1)
    ax[0].set_ylim(0,M-1)
    ax[1].imshow(rotate_iraf(img,xc,yc))
    ax[1].set_xlim(0,N-1)
    ax[1].set_ylim(0,M-1)
    cbar=colorbar(C2,ax=ax[2],use_gridspec=True,shrink=0.65)
    cbar.set_label(r'$A$',rotation=0)
    plot_utils.mpl.gcf().savefig('Apar.png',format='png')
    return AS

def find_center_asymmetry(img,center,func=Asymmetry,sky=None):
    """ Finds the center for which the value of A is  minimum.
    """
    res=sci_op.minimize(func,center,args=(img,sky),method='Nelder-Mead')
    if res.success==True:
        xc,yc=res.x
        A=res.fun
    else:
        xc,yc=center
        A=func([xc,yc],img,sky=sky)
        print(utils.colorylw.format('Warning! No miminimum found for center in asymmetry computation.\nSetting center to initial guess.'))

##    res=sci_op.leastsq(func,center,args=(img,sky))
##    print res
##    xc,yc=res[0]
##    A=func([xc,yc],img,sky=sky)
    return xc,yc,A

def find_neighbour_min(AS,img,i,j,sky=None,func=Asymmetry,ssize=1):
    """ To find if any of the 8 adjacent pixels has lower asymmetry than
    A(i,j). If yes, return maximum value coordinate of neighbour pixel. If not,
    (i,j) is returned"""
    if AS==None:
        AS=np.zeros(img.shape)
    for l in range(i-ssize,i+ssize+1):
        for m in range(j-ssize,j+ssize+1):
            try:
                if AS[l,m]==0:
                    AS[l,m]=func([l,m],img,sky=sky)
                else:
                    continue
            except IndexError:
                continue
    Aij=AS[i,j]
    maxi=i
    maxj=j
    for l in range(i-ssize,i+ssize+1):
        for m in range(j-ssize,j+ssize+1):
            try:
                NAij=AS[l,m]
            except IndexError:
                NAij=10.0
            if NAij<Aij:
                maxi=l
                maxj=m
                Aij=NAij
            else:
                continue
##    ax=make_subplot(1,2)
##    ax[0].imshow(AS)
##    ax[1].imshow(img)
##    gcf().canvas.draw_idle()
##    show()
    return maxi,maxj,AS

def A_conselice(img,center,func,sky=None,ssize=1):
    """ Compute the Asymmetry of neighbour pixels until a local minima is
    found. Usually Asymmetry as one global minimum close to the center of the
    galaxy np.meaning that if sterted from a guessed galy center the local minima
    found by this routine shall be the global minima of A for the galaxy.
    """
    found_min=False
    i,j=center
    AS=None
    while not found_min:
        ni,nj,AS=find_neighbour_min(AS,img,i,j,sky=sky,func=func,ssize=ssize)
        if ni==0:
            ni=img.shape[0]-2
        elif nj==0:
            nj=img.shape[1]-2

        if i==ni and j==nj:
            found_min=True
        else:
            i=ni
            j=nj
    return i,j,AS[i,j]


####################################################################################################
#                                           CONCENTRATION                                              #
###################################################################################################


def Anel(img,r,dmat,upp=1.1,low=0.9,draw_stuff=False):
    """Compute the flux with an annular region width width [0.8*r , 1.2*r] """

    stest=np.zeros(img.shape)
    stest[dmat<upp*r]=1

    ltest=np.zeros(img.shape)
    ltest[dmat>low*r]=1

    test=ltest*stest*img

    npix = np.size(test[test!=0])
    ann = np.sum(test)

    if draw_stuff:
        fig,ax=plot_utils.mpl.subplots(1,2)
        ax[0].imshow(img)
        ax[1].imshow(test)
        plot_utils.mpl.gcf().canvas.draw()
        plot_utils.mpl.show()

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

    return ann/npix

def petrosian_rad(img,xc,yc,q=1.00,ang=0.00,eta=0.2,step=0.5,cutfact=2,npix_min=5,full_output=False,draw_stuff=False,verbose=False):
    """Compute the petrosian radius using circular apertures around the galaxy
    center (xc,yc)."""
    ri = np.sqrt(2)
    rat = 1.0
    N,M=img.shape

    if full_output:
        rats=[]
        ris=[]
        ans=[]
        aps=[]
        inte=[]

    X,Y = np.meshgrid(range(img.shape[1]),range(int(img.shape[0])))
    dmat = utils.compute_ellipse_distmat(img,xc,yc,q,ang)

#    import time
#    if full_output:
#        ax=make_subplot(1,2)

    while rat > eta:
        if xc-cutfact*ri<0 or yc-cutfact*ri<0 or\
           int(round(xc+cutfact*ri))+1>M or\
           int(round(yc+cutfact*ri))+1>N:
                if verbose:
                    print(utils.colorylw.format('PETROSIAN RADIUS TRUNCATED DUE TO BORDER REACH!'))
                return ri,1

        dd=0
        xx = X[int(round(xc-cutfact*ri))+dd:int(round(xc+cutfact*ri+1))+dd,int(round(yc-cutfact*ri))+dd:int(round(yc+cutfact*ri+1))+dd]
        yy = Y[int(round(xc-cutfact*ri))+dd:int(round(xc+cutfact*ri+1))+dd,int(round(yc-cutfact*ri))+dd:int(round(yc+cutfact*ri+1))+dd]

        img_cut = img[xx,yy]
        dmat_cut=dmat[xx,yy]

        #ts=time.time()
        annulus = Anel(img_cut,ri,dmat_cut,draw_stuff=draw_stuff)
        #print "Annulus:\t%.6f seconds"%(time.time()-ts)

        aperture = img_cut[dmat_cut<ri]
        ri+=step
        rat = annulus/np.mean(aperture)

        if np.isnan(rat) or (np.size(aperture)<npix_min): #Due to small annulus with no pixel
            rat=1.0
            ri+=step

        if full_output:
            rats.append(rat)
            ris.append(ri)
            ans.append(annulus)
            aps.append(np.mean(aperture))
            inte.append(np.sum(aperture))


#            ax[0].plot(ris,rats,'b-')
#            ax[0].hlines(0.2,0,N,linestyle=':')
#            ax[1].plot(ris,ans,'g-')
#            ax[1].plot(ris,aps,'r-')
#            ax[0].set_xlim(0,N/2)
#            ax[0].set_ylim(0.0,1.0)
#            ax[1].set_xlim(0,N/2)
#            ax[1].set_ylim(0,)
#            fig=gcf()
#            fig.canvas.draw()


    if draw_stuff:
        sys.exit()
    if full_output:
        return ri, np.array(ris),np.array(rats),np.array(ans),np.array(aps),np.array(inte)
    else:
        return ri,0


def flux(img,xc,yc,rad):
    """ Compute the flux within the radius rad."""
    dmat = utils.compute_ellipse_distmat(img,xc,yc)
    return np.sum(img[dmat<rad])


def CAS_C(img,xc,yc,ba=1.00,theta=0.0,rp=None,dr0=0.1,rpstep=0.5,cutfact=1.1,draw_stuff=False):
    """Computes the concentration parameter from CAS based on the petrosian
    radius """
    if rp==None:
        rp,rpflag = petrosian_rad(img,xc,yc,ba,theta,step=rpstep,cutfact=cutfact,draw_stuff=draw_stuff)


    dmat = utils.compute_ellipse_distmat(img,xc,yc,ba,theta)
    F = np.sum(img[dmat<1.5*rp])

    r0 = 0.0
    F0 = 0.0
    r20 = 1e-10

    while F0 < 0.8*F:
        F0 = np.sum(img[dmat<r0])
        if F0 > 0.2*F and r20==1e-10:
            r20 = r0
        r0=r0+dr0

    return 5 * np.log10(r0/r20),rp,r20,r0


####################################################################################################
#                                           SMOOTHNESS                                                 #
###################################################################################################


def CAS_S(img,xc,yc,sigma,skypatch=None):
    """Computes the clumpiness parameter from CAS by smoothin with a gaussian
    kernel with sigma width. Sigma should be around 0.3 the petrosian radius.
    """
    N,M=img.shape

    gal = img.copy()
#    gal[gal<0]=0

#    smoothed = snd.filters.gaussian_filter(gal, sigma, mode='constant')
    smoothed = boxcar_filter2d(gal, sigma)
    dmat,dists=distance_matrix(xc,yc,gal)
    dmat[dmat < sigma]=0
    dmat[dmat >= sigma]=1

    #smoothed[smoothed<0]=0

    S_gal = dmat*abs(gal-smoothed)/np.sum(gal)


#    imshow(S_gal);show()

    if skypatch!=None:
        skypatch=abs(skypatch)
#        smoothed_sky = snd.filters.gaussian_filter(skypatch,sigma,mode='constant')
        smoothed_sky = boxcar_filter2d(skypatch,sigma)

        S_sky = dmat*abs(skypatch-smoothed_sky)/np.sum(gal)
    else:
        S_sky=0


##    fig,ax = mpl.subplots(1,4,figsize=(25,10))
##    ax[0].imshow(gal)
##    ax[1].imshow(smoothed)
##    ax[2].imshow(skypatch)
##    ax[3].imshow(smoothed_sky)
##    mpl.show()


    S = 10*(np.sum(S_gal)-np.sum(S_sky))
    return S,np.sum(S_gal),np.sum(S_sky)


def boxcar_filter2d(in_arr, width,height=None):
    """ Applies a boxcar filter to in_arr with dimensions [width x height]. If
    only width is provided then filter is assumed to be of square shape with size width.
    """

    width=max(width,2)
    if height==None:
        height=width

    width=int(round(width))
    height=int(round(height))

    N,M=in_arr.shape
    if width>N or height>M:
        raise IndexError("Invalid width/height: Greater than dimensions of input np.array. Width,Height=%i,%i and N,M=%i,%i"%(width,height,N,M))

    return snd.uniform_filter(in_arr, size=(width,height), mode="constant")
####################################################################################################
#                                           TESTING                                                    #
###################################################################################################

def test_all_functions():
    t_array = np.ones([50,50])
    boxcar_filter2d(t_array,5,5)

    CAS_C(t_array,25,25)
    CAS_A((5,5),t_array[5:15,5:15],3,t_array,t_array)
    CAS_S(t_array,25,25,3,skypatch=None)

if __name__=='__main__':
    test_all_functions()
