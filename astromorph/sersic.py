import sys
import os
import subprocess as sp
sys.path.append('/Users/bribeiro/Documents/PhD/simulations_images')

from matplotlib.pyplot import *
from scipy.optimize import *
from numpy import *
import mod_imports as MI
import numpy.random as rdm
from scipy.special import gamma
from scipy.integrate import simps
from scipy.ndimage import label
import time
import matplotlib.ticker as mpt

imsavedir = '/Users/bribeiro/Dropbox/ThesisLAM/figures/methods'


#############################################
############################################# General Sersic
#############################################
def plot_results(An,Num,var,absolute=True):
    fig,(ax0,ax1) = subplots(nrows=2,sharex=True)
    subplots_adjust(hspace=0.0)
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
##
##def I2(r,I0,r0,m=2):
##    Is = I0*(1-(r/float(r0))**m)
##    try:
##        Is[Is<0]=0
##    except TypeError:
##        if Is<0:
##            Is=0
##    return Is
##
##def test_I2():
##    I0=1
##    r0=1
##    rs=linspace(0,1.1*r0)
##    for m in [1,2,3,4,5]:
##        plot(rs,I2(rs,I0,r0,m),'-')
##    return
##
##
##
##def Fr_I2(r,I0,r0,m=2):
##    Ftot = I0*pi*r0**2*(m/(m+2.))
 ##    Fr = B * r**2 *(0.5 - (r/float(r0))**m / (m+2.))
##    try:
##        Fr[r>=r0]=Ftot
##    except TypeError:
##        if r>=r0:
##            Fr=Ftot
##    return Fr
##
##def test_FrI2():
##    I0=1
##    r0=1
##    rs=linspace(0,1.1*r0)
##    for m in [1,2,3,4,5]:
##        Ftot = I0*pi*r0**2*(m/(m+2.))
##        plot(rs,Fr_I2(rs,I0,r0,m),'-',color=cs[m])
##        hlines(Ftot,0,1.1*r0,color=cs[m],linestyle='--')
##    return
##
##
##
##def invI2(F,I0,r0):
##    B = 2*pi*r0**2
##    m=2.
##    Ftot = pi*r0**2*(m/(m+2.))
##    iI2 = sqrt(r0**2*(1-sqrt(B*(B-4*F))/B))
##    try:
##        iI2[F>=Ftot]=r0
##    except TypeError:
##        if F>=Ftot:
##            iI2=r0
##    return iI2
##
##def test_invI2():
##    I0=1.
##    r0=4.
##    m=2.
##    Ftot = pi*r0**2*(m/(m+2.))
##    fs = linspace(0,1.2*Ftot,1000)
##    plot(fs,invI2(fs,I0,r0))
##    return
##
##def Lp_I2(p,I0,r0):
##    m=2.
##    Ftot = pi*r0**2*(m/(m+2.))
##    rF = invI2(p,I0,r0)
##    lp = 2*pi*I0*rF**3/(15*r0**2) *(3*rF**2+5*r0**2)
##    return lp/(mean(I2(linspace(0,r0,1000),I0,r0)))
##
##def test_Lp():
##    I0=10.
##    r0=10.
##    ps = linspace(0,1,1000)
##    plot(ps,Lp_I2(ps,I0,r0))
##    return
##
####test_I2()
####test_FrI2()
####test_invI2()
##test_Lp()
##show()
##
def kappa(n):
    from scipy.special import gammaincinv
    return gammaincinv(2*n,1./2)

def test_kappa():
    "Compare the obtained values with the Ciotti & Bertin Approximation"

    n=linspace(0.1,10)
    def b_ciotti(n):
        return 2*n-1./3+4./(405*n) + 46./(25515*n*n)

    tstart = time.time()
    bc=b_ciotti(n)
    print "Ciotti & Bertin:\t %.4f millisec"%((time.time()-tstart)*1000)
    tstart=time.time()
    kn=kappa(n)
    print "This work:\t %.4f millisec"%((time.time()-tstart)*1000)

    fig,(ax0,ax1)=subplots(nrows=2,sharex=True)
    subplots_adjust(hspace=0.0)
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
    savefig('%s/kappa.png'%imsavedir)
    show()
    return

##test_kappa()

def gammainc(alfa,x):
    from scipy.special import gammainc,gamma
    return gammainc(alfa,x)*gamma(alfa)

def sersic(r,Ie,Re,n):
    "Exponential disk profile"
    b = kappa(n)
    return Ie*exp(-b*(abs(r)/Re)**(1./n)+b)

def mu_sersic(r,mu_e,Re,n):
    return mu_e + 2.5*kappa(n) * ( (r/Re)**(1./n) - 1 ) / log(10)


def sersic_int(r,Ie,Re,n):
    b = kappa(n)
    return 2*pi*Ie*Re*Re*(n/b**(2.*n))*gammainc(2*n,b*(r/Re)**(1./n))*exp(b)

def sersic_mean(r,Ie,Re,n,q=1.0):
    return sersic_int(r,Ie,Re,n)/(pi*q*r*r)

def total_flux(Ie,re,n):
    b = kappa(n)
    return Ie*re*re*gamma(2*n)*2*pi*n/(b**(2.*n))*exp(b)

def test_sersic():
    ns=[1,2,4,6,8,10]
    Ie=100
    RE=1
    rspace = logspace(-2,2,1000)
    fig1,ax=subplots()
    for n in ns:
        plot(rspace,sersic(rspace,Ie,RE,n),label='n=%i'%n,lw=3)

    legend(loc='best')
    ylim(1e-3,)
##    xlim(1e-15,1e2)
    loglog()
    xlabel(r'$r/r_e$',fontsize=25)
    ylabel(r'$I(r)$',fontsize=25)
    ax.set_xlim(0.02,19)
    ax.set_ylim(0.2,90000)
    ax.yaxis.set_major_formatter(mpt.ScalarFormatter())
    ax.xaxis.set_major_formatter(mpt.ScalarFormatter())
    ax.minorticks_on()
    savefig('%s/sersic.png'%imsavedir)
    fig2,ax=subplots()
    rb,ib=1,20
    rd,d=50,6
    disk=sersic(rspace,d,rd,1)
    bulge=sersic(rspace,ib,rb,4)
    plot(rspace,disk,'k--',rspace,bulge,'k:',rspace,bulge+disk,'k-',lw=2)
    ax.loglog()
    ax.yaxis.set_major_formatter(mpt.ScalarFormatter())
    ax.xaxis.set_major_formatter(mpt.ScalarFormatter())
    ax.set_xlim(0.1,100.)
    ax.set_ylim(1.0,1000.)
    ax.minorticks_on()
    xlabel(r'$r/r_e$',fontsize=25)
    ylabel(r'$I(r)$',fontsize=25)
    show()
##    fig1.savefig("%s/sersic.png"%(imsavedir))
    fig2.savefig("/Users/bribeiro/Dropbox/components.png")

    return

##test_sersic()
##sys.exit()


##
def galaxy_maker(imsize,xc,yc,I,r,n,q,theta):

    X,Y=np.meshgrid(range(imsize),range(imsize))
    ang=np.radians(theta)
    rX=(X-xc)*np.cos(ang)-(Y-yc)*np.sin(ang)
    rY=(X-xc)*np.sin(ang)+(Y-yc)*np.cos(ang)
    R = np.sqrt( rX**2 + (1/q)**2*rY**2 )

    galaxy=sersic(R,I,r,n)

    return galaxy

def profile(img,xc,yc,re,out_fact=5):

    R,ds = MI.distance_matrix(xc,yc,img)

    rs = arange(0,int(out_fact*re))

    def anel(dmat,ri,ro):
        lpix,hpix = zeros(dmat.shape),zeros(dmat.shape)
        lpix[dmat<ro]=1
        hpix[dmat>ri]=1
        return lpix*hpix
    dr = (rs[1]-rs[0])/2.0
    rmeans = rs[1:]-dr

    Ir=zeros(len(rs)-1)
    Ian=zeros(len(rmeans))
    for i in range(len(rs)-1):
        A = anel(R,rs[i],rs[i+1])*img
        Ir[i]=sum(A)/size(A[A!=0])
        Ian[i] = CAS.Anel(img,rmeans[i],R)
        continue

    dr = (rs[1]-rs[0])/2.0
    rmeans = rs[1:]-dr
    return rmeans,Ir,Ian

def testing_profile():
    Ie=1.0
    re=10.0
    q=1.0
    t=0.0
    fig=figure();ax1=gca();ax0=gca()
##    fig,(ax0,ax1) = subplots(nrows=2,sharex=True)
##    subplots_adjust(hspace=0.0)

    j=0
    for n in [1.0,2.0,3.0,4.0,6.0,8.0,10.0]:
        G=galaxy_maker(200,100.5,100.5,Ie,re,n,q,t)
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
    savefig('%s/profiles.png'%imsavedir)
    show()
    return

##testing_profile()

#############################################
############################################# General Shapes
#############################################

def circ_shape(r):
    I = ellipse_shape(r,1,0)
    return I

def ellipse_shape(r,q,theta):
    if (q>1.01) or (q<=0):
        raise ValueError("Invalid value for axis ratio. It must be between 0<q<=1.")
    if (r<=0):
        raise ValueError("Invalid value for radius. It must be strictly positive.")

    r=int(r)
    I = ones([2*r+2,2*r+2])
    ang=radians(theta)
    X,Y = meshgrid(range(I.shape[1]),range(int(I.shape[0])))
    rX=(X-r-0.5)*cos(ang)-(Y-r-0.5)*sin(ang)
    rY=(X-r-0.5)*sin(ang)+(Y-r-0.5)*cos(ang)
    dmat = sqrt(rX*rX+(1/(q*q))*rY*rY)
    I[dmat>r]=0
    return I


def area_test_ellipse(ntestq=100):
    rs=linspace(5,25,5)
    qs=linspace(0.15,1.0,ntestq)
    for r in rs:
        Area=zeros(ntestq)
        Area_pix=zeros(ntestq)
        i=0
        for q in qs:
            E=ellipse_shape(r,q,0.0)
            Area_pix[i] = len(E[E>0])
            Area[i] = pi*q*r*r
            i+=1

        plot(qs,(Area_pix-Area)/Area,label=r'$a=%i$'%r,lw=2)
    hlines(0,0.1,1,colors='red',linestyle=':',lw=2)
    xlim(0.15,1.0)
    ylim(-0.29,0.39)
    minorticks_on()
    xlabel(r'$q$')
    ylabel(r'$(A_{pix} - A)/A$')
    legend(loc='best')
    savefig('%s/area_ellipse.png'%imsavedir)
    show()
    return

def area_test_ellipse2(ntestq=100):
    rs=linspace(5,25,5)
    thetas=linspace(-90,90.,ntestq)
    q=0.5
    for r in rs:
        Area=zeros(ntestq)
        Area_pix=zeros(ntestq)
        i=0
        for theta in thetas:
            E=ellipse_shape(r,q,theta)
            Area_pix[i] = len(E[E>0])
            Area[i] = pi*q*r*r
            i+=1

        plot(thetas,(Area_pix-Area)/Area,label=r'$a=%i$'%r,lw=2)
    hlines(0,-90,90,colors='red',linestyle=':',lw=2)
    xlim(-90,90)
    ylim(-0.15,0.09)
    minorticks_on()
    xlabel(r'$\theta$')
    ylabel(r'$(A_{pix} - A)/A$')
    legend(loc='lower left')
    savefig('%s/area_ellipse_theta.png'%imsavedir)
    show()
    return

def area_test_circle(rmax=100):
    rs = arange(1,rmax)
    Area=zeros(len(rs))
    Area_pix=zeros(len(rs))
    i=0
    for r in rs:
        C=circ_shape(r)
        Area_pix[i]=len(C[C>0])
        Area[i]=pi*r*r
        i+=1
    plot(rs,(Area_pix-Area),lw=2)
    hlines(0,0,100,colors='red',linestyle=':',lw=2)
    xlim(1,99)
    ylim(-14,29)
    minorticks_on()
    xlabel(r'$r$')
    ylabel(r'$A_{pix} - A$')
    savefig('%s/area_circle.png'%imsavedir)
    show()
    return

##area_test_ellipse()
##area_test_ellipse2()
##area_test_circle()

def rectangular_shape(a,b):
    a=int(a)
    b=int(b)
    side=max([a,b])
    I = zeros([side+2,side+2])
    I[(side-a)/2+1:(side+a)/2+1,(side-b)/2+1:(side+b)/2+1]=1
    return I

def square_shape(l):
    return rectangular_shape(l,l)

def angles_from_points(p0,points):
    npoints=len(points)
    angles = zeros(npoints)
    for j in range(npoints):
        dx = points[j,0]-p0[0]
        dy = points[j,1]-p0[1]

        if (dx>0) and (dy>0):
            angles[j] = degrees(arctan(abs(dy/dx)))
        elif (dx<0) and (dy>0):
            angles[j] = 180-degrees(arctan(abs(dy/dx)))
        elif (dx<0) and (dy<0):
            angles[j] = 180+degrees(arctan(abs(dy/dx)))
        elif (dx>0) and (dy<0):
            angles[j] = 360-degrees(arctan(abs(dy/dx)))

    return angles

def polygon_shape(r,nverts,verts=None,clr='b'):
    from PiP import PiP

    I = ones([2*r+2,2*r+2])

    if verts==None:
##        from scipy.spatial import ConvexHull
        from numpy.random import randint
        verts = array(randint(1,2*r-1,[nverts,2]))
##        hull = ConvexHull(verts)
##        vertices = list(set([simplex for simplex in hull.simplices.reshape(hull.simplices.size)]))
        Med = mean(verts,0)
        thetas = angles_from_points(Med,verts)
        dtype=[('x',float),('y',float),('A',float)]
        point_list=zeros(nverts,dtype=dtype)
        for k in range(nverts):
            point_list[k] = (verts[k,0],verts[k,1],thetas[k])
        ordered_points = sort(point_list,order=['A','x','y'])
        poly = zeros([nverts+1,2])
        poly[:-1,0] = ordered_points['x']
        poly[:-1,1] = ordered_points['y']
        poly[-1]=poly[0]
    else:
        poly=verts


    XX,YY = meshgrid(range(2*r+2),range(2*r+2))

    for i in range(2*r+2):
        for j in range(2*r+2):
            if PiP(XX[i,j],YY[i,j],poly):
                continue
            else:
                I[i,j]=0
##    plot(poly[:,0],poly[:,1],'o-',color=clr);xlim(0,2*r+1);ylim(0,2*r+1)
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

    if rmax>imsize/2:
        raise ValueError("Maximum shape size greater than figure size!")

    shapes_func={'C':circ_shape,'E':ellipse_shape,\
                       'R':rectangular_shape,'S':square_shape,'P':polygon_shape}

    canvas = zeros([imsize,imsize])
    if not intersect:
        pixmap=zeros(canvas.shape)
    ntypes=len(types)
    ntries = min(100*nshapes,10000)

    from random import randint,choice,shuffle,uniform
    i=0
    j=0
    pars=[]

    while (i<nshapes) and (j<ntries):

        T = choice(types)
        a,b = randint(rmin,rmax),randint(rmin,rmax)
        r = max(a,b)/2

        x,y = randint(a/2+border,imsize-a/2-border),randint(b/2+border,imsize-b/2-border)
        I = randint(1,Imax)
        dtype=[('Type','S2'),('x',float),('y',float),('r',float),('a',float),('b',float),\
               ('q',float),('t',float),('N',int),('I',float),('n',float)]
        Object=array((T,x,y,r,a,b,0,0,0,I,0),dtype=dtype)

        if T == 'E':
            theta=randint(-90,90)
            q = pi*min(a,b)/float(max(a,b))
            Shape = shapes_func[T](r,q/pi,theta)
            Object['t']=theta
        elif T == 'C':
            Shape = shapes_func[T](r)
            q = pi
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
                g = galaxy_maker(Shape.shape[0],xc,yc,I,r,n,q/pi,theta)
            else:
                g = galaxy_maker(Shape.shape[0],xc,yc,I,r,n,1.0,0.0)
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

    return canvas,array(pars)

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
    ns=linspace(0.01,10.,100)
    gs1=[]
    gs2=[]

    for n in ns:
        g=galaxy_maker(100,50.5,50.5,10.,10.,n,1.,0.)
        Gmax=amax(g)
        gs1.append(Gmax/sersic(0.5,10,10,n))
        gs2.append(Gmax/sersic(0.,10,10,n))

    plot(ns,gs1)
    plot(ns,gs2)
    xlabel(r'$n$')
    ylabel(r'$I_f$')
    show()
    return

##test_max_intensity()

######################################################################
###################################################################### General GALFIT
######################################################################

def write_object(f,model,x,y,m,re,n,ba,pa,num):
    f.write("#Object number: %i\n"%num)
    f.write(' 0) %s             # Object type\n'%model)
    f.write(' 1) %6.4f %6.4f  1 1    # position x, y        [pixel]\n'%(x,y))
    f.write(' 3) %4.4f      1       # total magnitude\n' %m)
    f.write(' 4) %4.4f       1       #     R_e              [Pixels]\n'%re)
    f.write(' 5) %4.4f       1       # Sersic exponent (deVauc=4, expdisk=1)\n'%n)
    f.write(' 9) %4.4f       1       # axis ratio (b/a)   \n'%ba)
    f.write('10) %4.4f       1       # position angle (PA)  [Degrees: Up=0, Left=90]\n'%pa)
    f.write(' Z) 0                  #  Skip this model in output image?  (yes=1, no=0)\n')
    f.write(' \n')
    return

def galfit_input_file(f,magzpt,sky,xsize,ysize,sconvbox,imgname='galaxy',mask=False):
    root=os.getcwd()
    f.write("================================================================================\n")
    f.write("# IMAGE and GALFIT CONTROL PARAMETERS\n")
    f.write("A) none         # Input data image (FITS file)\n")
    f.write("B) %s.fits        # Output data image block\n"%imgname)
    f.write("C) none                # Sigma image name (made from data if blank or 'none' \n")
    f.write("D) psf.fits          # Input PSF image and (optional) diffusion kernel\n")
    f.write("E) 1                   # PSF fine sampling factor relative to data \n")
    if mask:
        f.write("F) mask.txt            # Bad pixel mask (FITS image or ASCII coord list)\n")
    else:
        f.write("F) none                # Bad pixel mask (FITS image or ASCII coord list)\n")
    f.write("G) none                # File with parameter constraints (ASCII file) \n")
    f.write("H) 1    %i   1    %i # Image region to fit (xmin xmax ymin ymax)\n"%(xsize+1,ysize+1))
    f.write("I) %i    %i          # Size of the convolution box (x y)\n"%(sconvbox,sconvbox))
    f.write("J) %7.5f             # Magnitude photometric zeropoint \n"%magzpt)
    f.write("K) 0.396 0.396         # Plate scale (dx dy)   [arcsec per pixel]\n")
    f.write("O) regular             # Display type (regular, curses, both)\n")
    f.write("P) 1                   # Options: 0=normal run; 1,2=make model/imgblock and quit\n")
    f.write("\n")
    f.write("# INITIAL FITTING PARAMETERS\n")
    f.write("#\n")
    f.write("#For object type, the allowed functions are:\n")
    f.write("#nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat,\n")
    f.write("#ferrer, and sky.\n")
    f.write("#\n")
    f.write("#Hidden parameters will only appear when theyre specified:\n")
    f.write("#C0 (diskyness/boxyness),\n")
    f.write("#Fn (n=integer, Azimuthal Fourier Modes).\n")
    f.write("#R0-R10 (PA rotation, for creating spiral structures).\n")
    f.write("#\n")
    f.write("# ------------------------------------------------------------------------------\n")
    f.write("#  par)    par value(s)    fit toggle(s)   parameter description\n")
    f.write("# ------------------------------------------------------------------------------\n")
    f.write("\n")

    obj=open('%s/galfit_object.temp'%root,'r')
    objects=obj.readlines()
    for line in objects:
        f.write(line)
    obj.close()

    f.write("# Object: Sky\n")
    f.write(" 0) sky                    #  object type\n")
    f.write(" 1) %7.4f      1          #  sky background at center of fitting region [ADUs]\n"%sky)
    f.write(" 2) 0.0000      0          #  dsky/dx (sky gradient in x)\n")
    f.write(" 3) 0.0000      0          #  dsky/dy (sky gradient in y)\n")
    f.write(" Z) 0                      #  output option (0 = resid., 1 = Dont subtract)")
    f.close()
    return


def galaxy_maker_galfit(imsize,model,xc,yc,Ie,re,n,ba,theta,mag_zpt=30.0,exptime=1.0,sky_val=0.0,PSF=True,FWHM=3.):
    import pyfits
    root=os.getcwd()
    f=open('%s/galfit_object.temp'%root,'w')
    try:
        N=len(imsize)
        if N==2:
            xsize,ysize=imsize
        else:
            raise ValueError("Invalid Dimensions on imsize!")
    except TypeError:
        xsize=imsize
        ysize=imsize

    Ftot = total_flux(Ie,re,n)
    mag = -2.5*log10(Ftot/exptime)+mag_zpt
    if model in ['sersic','expdisk','devauc']:
        write_object(f,model,xc,yc,mag,re,n,ba,theta,1)
    else:
        raise ValueError("Invalid Model! Choose from: sersic, expdisk, devauc ")
    f.close()
    f1=open('%s/GALFIT_input'%root,'w')
    galfit_input_file(f1,mag_zpt,sky_val,xsize,ysize,xsize)
    f1.close()
    if PSF:
        make_psf(mag_zpt,FWHM)
    sp.call('galfit -o1 GALFIT_input >> galfit.log',shell=True,stderr=sp.PIPE)
    galaxy=pyfits.getdata('%s/galaxy.fits'%root)
    sp.call('rm galaxy.fits galfit_object.temp GALFIT_input galfit.log',shell=True,stderr=sp.PIPE)
    if PSF:
        sp.call('rm %s/psf.fits'%root,shell=True,stderr=sp.PIPE)
    return galaxy


def make_psf(mzpt,FWHM=3):
    root=os.getcwd()

    fwhm_pix = FWHM
    imsize = int(40 * fwhm_pix/2.0)

    xc,yc=imsize/2.0+1,imsize/2.0+1

    f=open('%s/galfit_object.temp'%root,'w')

    f.write('0) gaussian           # object type\n')
    f.write('1) %.2f  %.2f  1 1  # position x, y        [pixel]\n'%(xc,yc))
    f.write('3) 14.0       1       # total magnitude\n')
    f.write('4) %.4f        0       #   FWHM               [pixels]\n'%(fwhm_pix))
    f.write('9) 1.0        1       # axis ratio (b/a)\n')
    f.write('10) 0.0         1       # position angle (PA)  [Degrees: Up=0, Left=90]\n')
    f.write('Z) 0                  # leave in [1] or subtract [0] this comp from data?\n')
    f.close()
    root=os.getcwd()

    f=open('%s/galfit_psf.txt'%root,'w')
    galfit_input_file(f,mzpt,0.0,imsize,imsize,0,imgname='psf',mask=False)
    f.close()

    sp.call('galfit -o1 galfit_psf.txt >> galfit.log',shell=True,stderr=sp.PIPE)
    sp.call('rm galfit_object.temp galfit_psf.txt',shell=True,stderr=sp.PIPE)

    return
