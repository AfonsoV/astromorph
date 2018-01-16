from mod_imports import *
import cosmology as cosmos
import hope

""" COMPUTE I,Psi,Xi parameters Law et al. 2007
27/08/2013  - Beggining
            - Reduced light_potential calculations by half (better optimization needed)
28/08/2013  - rearrange_light -> rearrange_light_spiral
            - new algorithm based on distances matrix to rearrange pixels
            - distance calc in a new function on mod_imports
            - distance_matrix moved to mod_imports
27/09/2013  - Dealed with diference in np.sizes from differente segmentation maps to color dispersion calc
02/10/2013  - No segmentation map applied to calc of color dispersion
14/10/2013  - Corretion to light_potential due to IndexErrors induced by changes in find_ij
            - Light Potential now only computes pixels from segmap, skipping all zero values
17/10/2013  - Correction on light potential to deal with sources clos to the border by settin X[gal==0]=-1
14/11/2014  - Added hope.jit to optimize the computation time of the light_potential function through speeding the sum_all call
"""

def Size(segmap,pixscale,z,pars=None):
    "Compute I parameter using pixscale in arcsec. Areas are inb kpc2"
    da=cosmos.angular_distance(z,pars=pars)*1000
    N = np.size(segmap[segmap>0])
    I = N * pixscale**2 * 2.4e-11 * da**2
    return I

@hope.jit
def sum_all(gal,X,Y,init,npoints):
    Sum=0.0
    for i in range(init,npoints):
        for j in range(i+1,npoints):
            r12 = np.sqrt((X[i]-X[j])*(X[i]-X[j])+(Y[i]-Y[j])*(Y[i]-Y[j]))
            Sum += gal[X[i],Y[i]]*gal[X[j],Y[j]]/r12
    return Sum

def light_potential(img,segmap):
    "Compute the light potential of a galaxy within the segmentation map"
    gal=img*segmap
    N,M=gal.shape
    Y,X = np.meshgrid(range(M),range(N))
    
    X[gal==0]=-1
    Y[gal==0]=-1

    X=X[X>=0]
    Y=Y[Y>=0]

##    import pp
##    import time
##
##    ts=time.time()
##
##    job_server = pp.Server(ppservers=())
##    Sum=0
##    nparts=32
##    for s in range(nparts):
##        Sum+=job_server.submit(sum_all,(gal,X,Y,s,nparts))()
##    print "Time elapsed: %f"%(time.time()-ts)
##
##    ts=time.time()    
##    Sum=sum_all(gal,X,Y,0,2)+sum_all(gal,X,Y,1,2)
##    print "Time elapsed: %f"%(time.time()-ts)

    Sum=sum_all(gal,X,Y,1,len(X))
    return Sum

##root=os.getcwd()
##
##test_image="%s/galfit_stamps/stamp110.fits"%root
##
##ORI = pyfits.getdata(test_image)
##MAP = gen_segmap_sex(test_image)
##ys,xs,es,thetas,obj_num = get_sex_pars(165,165)
##XC,YC,e,theta=xs[obj_num],ys[obj_num],es[obj_num],thetas[obj_num]
##
##IMG,SMAP,NC=make_stamps(XC,YC,ORI,MAP,fact=2)
##
##light_potential(IMG,SMAP)


def rearrange_light_spiral(img,segmap):
    "Rearrange pixels by decreasing intensity in an outward spiral pattern"
    N=np.size(img,0)
    M=np.size(img,1)
    gal=img*segmap
    Nimg=np.zeros([N,M])
    intensities=np.sort(gal.reshape(np.size(gal)))
    i=-2
    k=1
    x=np.size(Nimg,0)/2
    y=np.size(Nimg,1)/2
    Nimg[x-1,y-1]=intensities[-1]
    X=[x]
    Y=[y]
    while i>-len(intensities):
        if intensities[i]==0:
            break
        for j in range(1,k+1):
            x=x+(-1)**(k+1)
            X.append(x)
            Y.append(y)
            Nimg[x-1,y-1]=intensities[i]
            i-=1
        for j in range(1,k+1):
            y=y+(-1)**(k+1)
            X.append(x)
            Y.append(y)
            Nimg[x-1,y-1]=intensities[i]
            i-=1
        k+=1
    return X,Y,Nimg

def rearrange_light_distance(img,segmap):
    "Rearrange pixels by decreasing intensity using a distance matrix"
    N,M=img.shape
    gal=img*segmap
    Nimg=np.zeros([N,M])
    intensities=np.sort(gal.reshape(np.size(gal)))
    x=np.size(Nimg,0)/2-1
    y=np.size(Nimg,1)/2-1
    Nimg[x,y]=intensities[-1]
    Dmat, dists = distance_matrix(x,y,Nimg)
    i=-2
    j=1
    while i>-len(intensities):
        if intensities[i]==0:
            break
        pix = np.where(Dmat == dists[j])
        Nimg[pix] = intensities[i-len(pix[0])+1:i+1]
        i-= len(pix[0])
        j+=1
    return Nimg

def Multiplicity(img,segmap):
    "Compute the multiplicity statistic of Law et al. (2007)"
    psi_a=light_potential(img,segmap)

    compact=rearrange_light_distance(img,segmap)
    csegmap=np.zeros([np.size(compact,0),np.size(compact,1)])
    csegmap[compact>0]=1.0
    psi_c=light_potential(compact,csegmap)

##    ax=make_subplot(1,2)
##    ax[0].imshow(img)
##    ax[1].imshow(compact)
##    show()
##    print psi_a,psi_c
    
    Psi = 100*np.log10(psi_c/psi_a)
    return Psi,psi_a


def min_stat_pars(pars,img1,img2,sig1=1,sig2=1):
    """Function to minimize and find the alpha and beta parameters to compute xi.
    Images have to be of the same np.size, determined by image 1.See Papovich et al. 2003"""
    alpha,beta=pars
    sigma=(sig1+sig2)/2.0 #try to find a better np.meaning of sigma (why just one on the original equation)
    N1,N2=np.size(img1,0),np.size(img2,0)
    M1,M2=np.size(img1,1),np.size(img2,1)
    if N1!=N2 or M1!=M2:
        raise ValueError("Shape mismatch. Both Images have to be of the same np.size")
    Sum=0
    for i in range(N1):
        for j in range(M1):
            try:
                Sum+= ((img2[i,j]-alpha*img1[i,j]-beta)/sigma[i,j])**2
            except TypeError:
                Sum+= ((img2[i,j]-alpha*img1[i,j]-beta)/sigma)**2
    return Sum

def test_minimize_ab(img1,img2):
    "3D plot of the function min_stat_pars to be minimized"
    alphas=np.linspace(-10,10,50)
    betas = np.linspace(-10,10,50)
    X,Y = np.meshgrid(alphas,betas)
    fig=figure()
    ax=Axes3D(fig)
    Z = min_stat_pars([X,Y],img1,img2)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
    ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=cm.hot)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    ax.set_zlabel(r'$\chi^2$')
    show()
    return None

def find_alpha_beta(img1,img2,a0=1.0,b0=0):
    "Find alpha and beta parameter by minimizing min_stats_pars"
    res=sci_op.minimize(min_stat_pars,[a0,b0],args=(img1,img2),method='Nelder-Mead')
    if res.success == True:
        a,b=res.x
    else:
        a,b=a0,b0
    return a,b

def color_dispersion(img1,img2,segmap1,segmap2,bg1,bg2,cen1,cen2):
    "Compute the color dispersion parameter given constant backgrounds bg1 and bg2"
    
    GAL1=img1*segmap1
    GAL2=img2*segmap2

##    imax1,imin1,jmax1,jmin1=find_ij(segmap1)
##    imax2,imin2,jmax2,jmin2=find_ij(segmap2)

    x1,y1=cen1
    x2,y2=cen2
    
    r1=np.size_segmentation_map(x1,y1,segmap1)
    r2=np.size_segmentation_map(x2,y2,segmap2)
    rmax=max(r1,r2)
    
    GAL1=GAL1[int(x1-rmax):int(x1+rmax)+1,int(y1-rmax):int(y1+rmax)+1]
    GAL2=GAL2[int(x2-rmax):int(x2+rmax)+1,int(y2-rmax):int(y2+rmax)+1]

    alpha,beta=find_alpha_beta(GAL1,GAL2)

##    ax=make_subplot(1,3,width=12)
##    ax[0].imshow(GAL1)
##    ax[1].imshow(GAL2)
##    ax[2].imshow(GAL2-alpha*GAL1)
##    show()
    
    #test_minimize_ab(GAL1,GAL2)    
    Sum1 = np.sum((GAL2-alpha*GAL1-beta)**2)
    Sum2 = np.sum((GAL2-beta)**2)
    
    Npix = max([len(segmap1[segmap1>0]),len(segmap2[segmap2>0])])

    #Sum3 += np.sum(bg2-alpha*bg1)**2 ### Use if not constant background
    Sum3 = Npix*(bg2-alpha*bg1)**2 
    
    Xi = (Sum1-Sum3)/(Sum2-Sum3)

    return Xi

