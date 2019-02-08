import sys
import os
import subprocess as sp

import numpy as np
import numpy.random as rdm

import scipy.optimize as scopt
from scipy.special import gamma
from scipy.integrate import simps
from scipy.ndimage import label

import matplotlib.pyplot as mpl
import matplotlib.ticker as mpt
import matplotlib.patches as mpa

from astropy.convolution import Gaussian2DKernel,Moffat2DKernel
from scipy.signal import fftconvolve

from . import utils

import time

from .CAS import Anel


#############################################
############################################# General Sersic
#############################################


def kappa(n):
    r"""

    Parameters
    ----------
        n : float
            the value of the sersic index to compute kappa[b] for.

    Returns
    -------
        kappa : float
            the value of kappa, given the input sersic index n

    References
    ----------
        Ciotti & Bertin 1999 (https://ui.adsabs.harvard.edu/#abs/1999A&A...352..447C/abstract)

    Examples
    --------
        >>> kappa(1)
        1.678...

    """
    from scipy.special import gammaincinv
    return gammaincinv(2*n,1./2)



def gammainc(alfa,x):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    from scipy.special import gammainc,gamma
    return gammainc(alfa,x)*gamma(alfa)

def sersic(r,Ie,Re,n):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    b = kappa(n)
    return Ie*np.exp(-b*(np.abs(r)/Re)**(1./n)+b)

def mu_sersic(r,mu_e,Re,n):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    return mu_e + 2.5*kappa(n) * ( (r/Re)**(1./n) - 1 ) / np.log(10)

def sersic_int(r,Ie,Re,n):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    b = kappa(n)
    return 2*np.pi*Ie*Re*Re*(n/b**(2.*n))*gammainc(2*n,b*(r/Re)**(1./n))*np.exp(b)

def sersic_mean(r,Ie,Re,n,q=1.0):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    return sersic_int(r,Ie,Re,n)/(np.pi*q*r*r)

def total_flux(Ie,re,n):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    b = kappa(n)
    return Ie*re*re*gamma(2*n)*2*np.pi*n/(b**(2.*n))*np.exp(b)

def effective_intensity(total_flux,re,n):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    b = kappa(n)
    return total_flux/(re*re*gamma(2*n)*2*np.pi*n/(b**(2.*n))*np.exp(b))

def test_sersic():
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    ns=[1,2,4,6,8,10]
    Ie=100
    RE=1
    rspace = np.logspace(-2,2,1000)
    fig1,ax= mpl.subplots()
    for n in ns:
        ax.plot(rspace,sersic(rspace,Ie,RE,n),label='n=%i'%n,lw=3)

    ax.legend(loc='best')
    ax.set_ylim(1e-3,)
##    xlim(1e-15,1e2)
    ax.loglog()
    ax.set_xlabel(r'$r/r_e$',fontsize=25)
    ax.set_ylabel(r'$I(r)$',fontsize=25)
    ax.set_xlim(0.02,19)
    ax.set_ylim(0.2,90000)
    ax.yaxis.set_major_formatter(mpt.ScalarFormatter())
    ax.xaxis.set_major_formatter(mpt.ScalarFormatter())
    ax.minorticks_on()

    fig2,ax=mpl.subplots()
    rb,ib=1,20
    rd,d=50,6
    disk=sersic(rspace,d,rd,1)
    bulge=sersic(rspace,ib,rb,4)
    ax.plot(rspace,disk,'k--',rspace,bulge,'k:',rspace,bulge+disk,'k-',lw=2)
    ax.loglog()
    ax.yaxis.set_major_formatter(mpt.ScalarFormatter())
    ax.xaxis.set_major_formatter(mpt.ScalarFormatter())
    ax.set_xlim(0.1,100.)
    ax.set_ylim(1.0,1000.)
    ax.minorticks_on()
    ax.set_xlabel(r'$r/r_e$',fontsize=25)
    ax.set_ylabel(r'$I(r)$',fontsize=25)
    mpl.show()
    return

def generate_lensed_sersic_model(shape,modelPars,lensingPars,mag_zeropoint,exposure_time,psf=None,OverSampling = 5,debug=False):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """


    if OverSampling <1:
        raise ValueError
    else:
        OverSampling = int(OverSampling) ## Needs to be integer


    xc,yc,mag,radius,sersic_index,axis_ratio,position_angle = modelPars
    lensKappa,lensGamma,lensMu,lensAngle = lensingPars

    majAxis_Factor = 1/np.abs(1-lensKappa-lensGamma)
    shear_factor = (1-lensKappa-lensGamma)/(1-lensKappa+lensGamma)

    ang_rad = np.radians(lensAngle)
    X,Y = np.meshgrid(np.arange(shape[0]*OverSampling),np.arange(shape[1]*OverSampling))
    X = (X.astype(np.float64) - xc*OverSampling)/OverSampling
    Y = (Y.astype(np.float64) - yc*OverSampling)/OverSampling
    rLensX=(X)*np.cos(ang_rad)-(Y)*np.sin(ang_rad)
    rLensY=(X)*np.sin(ang_rad)+(Y)*np.cos(ang_rad)
    if shear_factor<1:
        rLensX *= (1-lensKappa-lensGamma)
        rLensY *= (1-lensKappa+lensGamma)
    else:
        rLensX *= (1-lensKappa+lensGamma)
        rLensY *= (1-lensKappa-lensGamma)
    dmatLens = np.sqrt(rLensX*rLensX + rLensY*rLensY)


    Pa_rad = np.radians(position_angle)
    rLensGalX=(rLensX)*np.cos(Pa_rad)-(rLensY)*np.sin(Pa_rad)
    rLensGalY=(rLensX)*np.sin(Pa_rad)+(rLensY)*np.cos(Pa_rad)
    dmatGal = np.sqrt(rLensGalX*rLensGalX+(1/(axis_ratio*axis_ratio))*rLensGalY*rLensGalY)

    totalFlux = exposure_time * 10**( -0.4*(mag-2.5*np.log10(lensMu)-mag_zeropoint) )
    Sigma_e = effective_intensity(totalFlux,radius,sersic_index)
    Profile = sersic(dmatGal,Sigma_e,radius,sersic_index)

    if debug is True:
        from .ADD import surface_brightness_profile
        print("magnification",lensMu,"shear",shear_factor)
        # print(X.ravel())
        print("angle",ang_rad)
        print("magFactors - x,y:",(1-lensKappa-lensGamma),(1-lensKappa+lensGamma))
        # fig,ax = mpl.subplots()
        # R,F = surface_brightness_profile(-dmatLens,np.ones_like(dmatLens),rmax=X.shape[0])
        # ax.plot(R,F)
        #
        # fig,ax = mpl.subplots(1,3,sharex=True,sharey=True,figsize=(20,8))
        # fig.subplots_adjust(wspace=0)
        # ax[0].imshow(-dmatLens)
        # ax[1].imshow(-dmatGal)
        # ax[2].imshow(Profile)
        # ax[1].set_title("bruno")

    if OverSampling != 1:
        return Profile.reshape(shape[0],OverSampling,\
                               shape[1],OverSampling).sum(3).sum(1)/(OverSampling**2)
    else:
        return Profile


def generate_sersic_model(shape,modelPars,mag_zeropoint,exposure_time,psf=None):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    xc,yc,mag,radius,sersic_index,axis_ratio,position_angle = modelPars
    totalFlux = exposure_time * 10**( -0.4*(mag-mag_zeropoint) )
    Sigma_e = effective_intensity(totalFlux,radius,sersic_index)
    return galaxy_creation(shape,xc,yc,Sigma_e,radius,sersic_index,axis_ratio,\
                            position_angle,psf=psf)

##
def galaxy_creation(imsize,xc,yc,I,r,n,q,theta,psf=None,**kwargs):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    R = utils.compute_ellipse_distmat(np.zeros(imsize),xc,yc,q,theta)
    profile = sersic(R,I,r,n)

    if psf is None:
        galaxy = profile
    else:
        galaxy = fftconvolve(profile, psf)

    return galaxy

def profile(img,xc,yc,re,out_fact=5):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """

    R = utils.compute_ellipse_distmat(xc,yc,img)
    rs = np.arange(0,int(out_fact*re))

    def anel(dmat,ri,ro):
        lpix,hpix = np.zeros(dmat.shape),np.zeros(dmat.shape)
        lpix[dmat<ro]=1
        hpix[dmat>ri]=1
        return lpix*hpix

    dr = (rs[1]-rs[0])/2.0
    rmeans = rs[1:]-dr

    Ir=np.zeros(len(rs)-1)
    Ian=np.zeros(len(rmeans))
    for i in range(len(rs)-1):
        A = anel(R,rs[i],rs[i+1])*img
        Ir[i]=np.sum(A)/np.size(A[A!=0])
        Ian[i] = Anel(img,rmeans[i],R)
        continue

    dr = (rs[1]-rs[0])/2.0
    rmeans = rs[1:]-dr
    return rmeans,Ir,Ian

def testing_profile():
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    Ie=1.0
    re=10.0
    q=1.0
    t=0.0
    # fig=mpl.figure();ax1=gca();ax0=gca()
    fig,(ax0,ax1) = mpl.subplots(nrows=2,sharex=True)
    fig.subplots_adjust(hspace=0.0)

    j=0
    for n in [1.0,2.0,3.0,4.0,6.0,8.0,10.0]:
        G=galaxy_creation((200,200),100.5,100.5,Ie,re,n,q,t)
        R,Ir,Ianel=profile(G,100.5,100.5,re,3)
        Ia = sersic(R,Ie,re,n)

##        ax0.plot(R/re,Ir,'o',color=cs[j],label='Numerical')
##        ax0.plot(R/re,Ianel,'x',color=cs[j],label='Numerical')
##        ax0.plot(R/re,Ia,'-',color=cs[j],label='Analytical')

        ax1.plot(R/re,abs(Ir-Ia)/Ia,'-',color=cs[j],label=r'$n=%.1f$'%n)
        ax1.plot(R/re,abs(Ianel-Ia)/Ia,'--',color=cs[j])
        j+=1

    ax0.set_ylabel(r'$I(r)$')
    ax0.semilogy()
    ax0.set_yticks(ax0.get_yticks()[1:-1])
    ax1.set_xlabel(r'$r/r_e$')
    ax1.set_ylabel(r'$\Delta I/I$')
    ax1.semilogy()
    ax1.set_yticks(ax1.get_yticks()[1:-1])
    ax1.legend(loc='best',ncol=2)
    mpl.show()
    return


#############################################
############################################# General Shapes
#############################################

def circ_shape(r):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    I = ellipse_shape(r,1,0)
    return I

def ellipse_shape(r,q,theta):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    if (q>1.01) or (q<=0):
        raise ValueError("Invalid value for axis ratio. It must be between 0<q<=1.")
    if (r<=0):
        raise ValueError("Invalid value for radius. It must be strictly positive.")

    r=int(r)
    I = np.ones([2*r+2,2*r+2])
    dmat = utils.compute_ellipse_distmat(I,r+0.5,r+0.5,q,theta)
    I[dmat>r]=0
    return I


def area_test_ellipse(ntestq=100):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    rs=np.linspace(5,25,5)
    qs=np.linspace(0.15,1.0,ntestq)
    for r in rs:
        Area=np.zeros(ntestq)
        Area_pix=np.zeros(ntestq)
        i=0
        for q in qs:
            E=ellipse_shape(r,q,0.0)
            Area_pix[i] = len(E[E>0])
            Area[i] = np.pi*q*r*r
            i+=1

        mpl.plot(qs,(Area_pix-Area)/Area,label=r'$a=%i$'%r,lw=2)

    mpl.hlines(0,0.1,1,colors='red',linestyle=':',lw=2)
    mpl.xlim(0.15,1.0)
    mpl.ylim(-0.29,0.39)
    mpl.minorticks_on()
    mpl.xlabel(r'$q$')
    mpl.ylabel(r'$(A_{pix} - A)/A$')
    mpl.legend(loc='best')
    mpl.show()
    return

def area_test_ellipse2(ntestq=100):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    rs=np.linspace(5,25,5)
    thetas=np.linspace(-90,90.,ntestq)
    q=0.5
    for r in rs:
        Area=np.zeros(ntestq)
        Area_pix=np.zeros(ntestq)
        i=0
        for theta in thetas:
            E=ellipse_shape(r,q,theta)
            Area_pix[i] = len(E[E>0])
            Area[i] = np.pi*q*r*r
            i+=1

        mpl.plot(thetas,(Area_pix-Area)/Area,label=r'$a=%i$'%r,lw=2)
    mpl.hlines(0,-90,90,colors='red',linestyle=':',lw=2)
    mpl.xlim(-90,90)
    mpl.ylim(-0.15,0.09)
    mpl.minorticks_on()
    mpl.xlabel(r'$\theta$')
    mpl.ylabel(r'$(A_{pix} - A)/A$')
    mpl.legend(loc='lower left')
    mpl.show()
    return

def area_test_circle(rmax=100):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    rs = np.arange(1,rmax)
    Area= np.zeros(len(rs))
    Area_pix= np.zeros(len(rs))
    i=0
    for r in rs:
        C=circ_shape(r)
        Area_pix[i]=len(C[C>0])
        Area[i]=np.pi*r*r
        i+=1
    mpl.plot(rs,(Area_pix-Area),lw=2)
    mpl.hlines(0,0,100,colors='red',linestyle=':',lw=2)
    mpl.xlim(1,99)
    mpl.ylim(-14,29)
    mpl.minorticks_on()
    mpl.xlabel(r'$r$')
    mpl.ylabel(r'$A_{pix} - A$')
    mpl.show()
    return


def rectangular_shape(a,b):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    a=int(a)
    b=int(b)
    side=max([a,b])
    I = np.zeros([side+2,side+2])
    I[(side-a)//2+1:(side+a)//2+1,(side-b)//2+1:(side+b)//2+1]=1
    return I

def square_shape(l):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    return rectangular_shape(l,l)

def angles_from_points(p0,points):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    npoints=len(points)
    angles = np.zeros(npoints)
    for j in range(npoints):
        dx = points[j,0]-p0[0]
        dy = points[j,1]-p0[1]

        if (dx>0) and (dy>0):
            angles[j] = np.degrees(np.arctan(abs(dy/dx)))
        elif (dx<0) and (dy>0):
            angles[j] = 180-np.degrees(np.arctan(abs(dy/dx)))
        elif (dx<0) and (dy<0):
            angles[j] = 180+np.degrees(np.arctan(abs(dy/dx)))
        elif (dx>0) and (dy<0):
            angles[j] = 360-np.degrees(np.arctan(abs(dy/dx)))

    return angles

def pointsInPolygon(points,vertices):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    polygon = mpa.Polygon(vertices)
    return polygon.get_path().contains_points(points)

def polygon_shape(r,nverts,verts=None,clr='red'):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """

    I = np.zeros([2*r+2,2*r+2])

    if verts==None:
        from numpy.random import randint
        verts = np.array(randint(1,2*r-1,[nverts,2]))

        Med = np.mean(verts,0)
        thetas = angles_from_points(Med,verts)
        dtype=[('x',float),('y',float),('A',float)]
        point_list=np.zeros(nverts,dtype=dtype)
        for k in range(nverts):
            point_list[k] = (verts[k,0],verts[k,1],thetas[k])
        ordered_points = np.sort(point_list,order=['A','x','y'])
        poly = np.zeros([nverts+1,2])
        poly[:-1,0] = ordered_points['x']
        poly[:-1,1] = ordered_points['y']
        poly[-1]=poly[0]
    else:
        poly=verts


    XX,YY = np.meshgrid(range(2*r+2),range(2*r+2))

    polMask = pointsInPolygon(np.vstack([XX.ravel(),YY.ravel()]).T,poly)
    mpl.show()
    return I

##P5=polygon_shape(10,5,clr=cs[5])
##P10=polygon_shape(10,10,clr=cs[5])
##C=circ_shape(10)
##E=ellipse_shape(10,0.5,0)
##R=rectangular_shape(20,9)
##Q=square_shape(20)


##ax = make_subplot(2,3)
##ax[0].imshow(C)
##ax[0].set_title(str(C.shape))
##ax[1].imshow(E)
##ax[1].set_title(str(E.shape))
##ax[2].imshow(R)
##ax[2].set_title(str(R.shape))
##ax[3].imshow(Q)
##ax[3].set_title(str(Q.shape))
##ax[4].imshow(P5)
##ax[4].set_title(str(P5.shape))
##ax[5].imshow(P10)
##ax[5].set_title(str(P10.shape))
##show()
##

def gerate_combination(imsize,nshapes,rmax=10,types=['C','E','R','S','P'],intersect=True,nverts_lims=[5,10],Imax=1,profile=False,border=4,rmin=4):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """

    if rmax>min(imsize)/2:
        raise ValueError("Maximum shape size greater than figure size!")

    shapes_func={'C':circ_shape,'E':ellipse_shape,\
                       'R':rectangular_shape,'S':square_shape,'P':polygon_shape}

    canvas = np.zeros(imsize)
    if not intersect:
        pixmap=np.zeros(canvas.shape)
    ntypes=len(types)
    ntries = min(100*nshapes,10000)

    from random import randint,choice,shuffle,uniform
    i=0
    j=0
    pars=[]

    while (i<nshapes) and (j<ntries):

        T = choice(types)
        a,b = randint(rmin,rmax),randint(rmin,rmax)
        r = max(a,b)//2

        x,y = randint(a//2+border,imsize[0]-a//2-border),randint(b//2+border,imsize[1]-b//2-border)
        I = randint(1,Imax)
        dtype=[('Type','S2'),('x',float),('y',float),('r',float),('a',float),('b',float),\
               ('q',float),('t',float),('N',int),('I',float),('n',float)]
        Object=np.array((T,x,y,r,a,b,0,0,0,I,0),dtype=dtype)

        if T == 'E':
            theta=randint(-90,90)
            q = np.pi*min(a,b)/float(max(a,b))
            Shape = shapes_func[T](r,q/np.pi,theta)
            Object['t']=theta
        elif T == 'C':
            Shape = shapes_func[T](r)
            q = np.pi
        elif T == 'S':
            Shape = shapes_func[T](2*r)
            q = 4.0
        elif T == 'R':
            Shape = shapes_func[T](a,b)
            q = 4 * min(a,b)/float(max(a,b))
        elif T == 'P':
            nverts_min,nverts_max=nverts_lims
            nverts=randint(nverts_min,nverts_max)
            Object['N']=nverts
            q=0
            Shape = shapes_func[T](r,nverts)
        else:
            raise ValueError('Invalid type for shape: %s! Choose from C,E,S,R or P'%T)
        Object['q']=q

        if profile:
            xc,yc=Shape.shape[0]/2.,Shape.shape[0]/2.
            n=uniform(0.5,8)
            Object['n']=n
            if T=='E':
                g = galaxy_creation((Shape.shape[0],Shape.shape[0]),xc,yc,I,r,n,q/pi,theta)
            else:
                g = galaxy_creation((Shape.shape[0],Shape.shape[0]),xc,yc,I,r,n,1.0,0.0)
            obj_img = Shape*g
        else:
            obj_img = Shape*I

        if intersect:
            try:
                canvas[x-r-1:x+r+1,y-r-1:y+r+1]+=obj_img
                pars.append(Object)
                i+=1
            except ValueError:
                pass
        else:
            try:
                if 1 in pixmap[x-r-1:x+r+1,y-r-1:y+r+1]*Shape:
                    pass
                else:
                    canvas[x-r-1:x+r+1,y-r-1:y+r+1]+=obj_img
                    pixmap[canvas>0]=1
##                    pixmap[x-r-1:x+r+1,y-r-1:y+r+1]=1
                    pars.append(Object)
                    i+=1
            except ValueError:
                pass

        j+=1

    return canvas,np.array(pars)

##N=1000
##Fr=zeros(N)
##sT=zeros(N)
##Rs=zeros(N)
##for n in range(N):
##    T,pp=gerate_combination(100,1,rmax=45,types='E',intersect=False,profile=True,Imax=100)
##    Fr[n]=sersic_int(pp['r'],pp['I'],100,1.0)*pp['q']
##    sT[n]=sum(T)
##    Rs[n]=pp['r']
##
##SP=scatter(sT,(sT-Fr)/Fr,c=Rs,s=20,alpha=0.7)
##colorbar(SP)
##show()
##sys.exit()

##test,pars = gerate_combination(200,25,rmax=50,types='CESRP',intersect=False,Imax=100)
##ax = make_subplot(2,3,width=25,height=15)
##ax[0].imshow(test,cmap='spectral')
##ax[1].imshow(test,cmap='jet')
##ax[1].set_title(str(len(pars)))
##ax[2].imshow(test,cmap='cool')
##ax[3].imshow(test,cmap='hot')
##ax[4].imshow(test,cmap='ocean')
##ax[5].imshow(test,cmap='terrain')
##show()
##sys.exit()

def test_max_intensity():
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    ns=np.linspace(0.01,10.,100)
    gs1=[]
    gs2=[]

    for n in ns:
        g=galaxy_creation((100,100),50.5,50.5,10.,10.,n,1.,0.)
        Gmax=np.amax(g)
        gs1.append(Gmax/sersic(0.5,10,10,n))
        gs2.append(Gmax/sersic(0.,10,10,n))

    mpl.plot(ns,gs1)
    mpl.plot(ns,gs2)
    mpl.xlabel(r'$n$')
    mpl.ylabel(r'$I_f$')
    mpl.show()
    return

##test_max_intensity()

######################################################################
###################################################################### General PSF
######################################################################

def create_gaussian_PSF(fwhm,**kwargs):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    return Gaussian2DKernel(sigma,**kwargs)

def create_moffat_PSF(gamma,alpha,**kwargs):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    return Moffat2DKernel(gamma,alpha,**kwargs)

######################################################################
###################################################################### General Noise
######################################################################

def create_gaussian_sky(shape,stddev,mean=0):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    return rdm.normal(mean*np.ones(shape),stddev)

def poissonFilter(model,gain=1,exposure_time=1):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    return rdm.poisson(model*exposure_time*gain)/gain


######################################################################
###################################################################### General Test
######################################################################

def plot_results(An,Num,var,absolute=True):
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    fig,(ax0,ax1) = mpl.subplots(nrows=2,sharex=True)
    fig.subplots_adjust(hspace=0.0)
    ax0.scatter(An[Num>0],Num[Num>0],c='DodgerBlue',s=40,alpha=0.5)
    ax0.plot(sort(An),sort(An),'r--')
    ax0.set_ylabel(r'$%s\ [\mathrm{numerical}]$'%var)

    if absolute:
        ax1.scatter(An[Num>0], abs(Num[Num>0]-An[Num>0])/An[Num>0],c='DodgerBlue',s=40,alpha=0.5)
        ax1.set_ylabel(r'$|\Delta %s|/%s$'%(var,var))
    else:
        ax1.scatter(An[Num>0], (Num[Num>0]-An[Num>0])/An[Num>0],c='DodgerBlue',s=40,alpha=0.5)
        ax1.set_ylabel(r'$\Delta %s/%s$'%(var,var))

    ax1.hlines(0,0,1.05*max(An))
    ax1.set_xlim(0,1.05*max(An))
    ax1.set_xlabel(r'$%s [\mathrm{analytical}]$'%var)
    ax0.set_yticks(ax0.get_yticks()[1:-1])
    ax1.set_yticks(ax1.get_yticks()[1:-1])
    return fig,ax0,ax1

def test_kappa():
    r"""

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    "Compare the obtained values with the Ciotti & Bertin Approximation"

    n=np.linspace(0.1,10)
    def b_ciotti(n):
        return 2*n-1./3+4./(405*n) + 46./(25515*n*n)

    tstart = time.time()
    bc=b_ciotti(n)
    print("Ciotti & Bertin:\t %.4f millisec"%((time.time()-tstart)*1000))
    tstart=time.time()
    kn=kappa(n)
    print("This work:\t %.4f millisec"%((time.time()-tstart)*1000))

    fig,(ax0,ax1)=mpl.subplots(nrows=2,sharex=True)
    fig.subplots_adjust(hspace=0.0)
    ax0.plot(n,bc,'r-',lw=2.5,label='Ciotti & Bertin (1999)')
    ax0.plot(n,kn,'b--',label='This work',lw=2)
    ax0.set_ylabel(r'$\kappa$')
    ax0.legend(loc='lower right')
    ax0.semilogy()
    ax0.yaxis.set_major_formatter(mpt.ScalarFormatter())
    ax0.set_yticks(ax0.get_yticks()[2:-2])
    diff = (kn-bc)/bc
    ax1.plot(n,abs(diff),'k',lw=2)
    ax1.hlines(0,1,10)
    ax1.set_xlabel(r'$n$')
    ax1.set_ylabel(r'$|\Delta \kappa|/\kappa$')
    ax1.semilogy()
##    ax1.yaxis.set_major_formatter(mpt.ScalarFormatter())
    ax1.set_yticks(ax1.get_yticks()[2:-2])
    for eixo in [ax0,ax1]:
        eixo.minorticks_on()
    mpl.show()
    return
