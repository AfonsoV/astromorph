from . import utils

####################################################################################################
#                                              GINI                                                    #
###################################################################################################


def Gini(img,segmap):
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
    "Compute the gini coefficient for a given image with a segmentation map provided"
    gal=img*segmap
    fluxs=gal[segmap>0]
    fluxs=np.sort(fluxs)
    AveFlux=np.mean(abs(fluxs))

    Npix=len(fluxs)

    Sum=0
    for i in range(Npix):
        Sum+=(2*(i+1)-Npix-1)*abs(fluxs[i])

    G = Sum/(AveFlux*Npix*(Npix-1))
    return G

def test_gini():
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
    """Code to test values of G in ideal cases: all light in one pixel G=1
    and light uniformly distributed G=0"""
    N=50
    M=50
    Timg=np.zeros([N,M])
    Tmap=np.zeros([N,M])
    Dmat,dist=distance_matrix(N/2-1,M/2-1,Tmap)
    Timg[24,24]=1
    Tmap[Dmat<5]=1
    print("Single pixel galaxy: G=%.2f"%Gini(Timg,Tmap))
    Timg[Dmat<5]=1
    print("Uniform pixel galaxy: G=%.2f"%Gini(Timg,Tmap))
    return


####################################################################################################
#                                               M_20                                                   #
###################################################################################################

def Mtot(cen,img,segmap):
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
    """Compute the second order total momentum of a galaxy given xc and yc"""
    N,M=img.shape
    gal = img*segmap
    imax,imin,jmax,jmin=find_ij(segmap)
    xc,yc=cen

    XX,YY=np.meshgrid(range(M),range(N))
    dX=XX-xc
    dY=YY-yc
    Mtot_image = gal * (dX*dX+dY*dY)

    return np.sum(Mtot_image)

def find_center_mtot(img,segmap,x0,y0,verbose=False):
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
    """Find the galaxy center by minimizing Mtot"""
    res=sci_op.minimize(Mtot,[x0,y0],args=(img,segmap),method='Nelder-Mead')
    if res.success == True:
        xc,yc=res.x
        mtot=res.fun
        mtot_flag=0
    else:
        xc,yc=x0,y0
        mtot=Mtot([x0,y0],img,segmap)
        mtot_flag=1
        if verbose:
            print('Warning! No miminimum found for center in total momentum computation.\n Setting center to initial guess.')
    return xc,yc,mtot,mtot_flag

def MomentLight20(img,segmap,x0=100,y0=100,verbose=False):
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
    """Compute the M20 index based on Initial guess for galactic center (x0,y0)
    Mtot is minimized prior to calc of M20 by finding the best pair (xc,yc)"""
    xc,yc,mtot,mtot_flag=find_center_mtot(img,segmap,x0,y0,verbose=verbose)
    gal = img*segmap

    fluxes=gal[segmap>0]
    fluxes=np.sort(fluxes)
    ftot=np.sum(fluxes)

    sumF=fluxes[-1]
    sumM=0
    i=-1
    while sumF < 0.2 * ftot:
        x1,y1=np.where(gal==fluxes[i])
        temp=0
        if len(x1)>1:
            temp = fluxes[i]*((x1-xc)**2+(y1-yc)**2)
            sumM += np.sum(temp)
        else:
            sumM += fluxes[i]*((x1-xc)**2+(y1-yc)**2)
        i-=len(x1)
        sumF+=len(x1)*fluxes[i]

    if sumM==0:
        sumM=np.array([sumF])

    m20=np.log10(sumM/mtot)
    try:
        return m20[0],mtot_flag
    except IndexError as err:
        return m20,mtot_flag

def test_minimize_m(img,segmap):
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
    """Function to plot the 2D distribution of Mtot for different pairs of (x,y)
    to investigate if minimize is picking the right solution"""
    N=np.size(img,0)
    M=np.size(img,1)
    xc = np.linspace(N/4,3*N/4,50)
    yc = np.linspace(M/4,3*M/4,50)
    X,Y = np.meshgrid(xc,yc)
    fig=figure()
    ax=Axes3D(fig)
    Z =Mtot([X,Y],img,segmap)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
    ax.contourf(X, Y, Z, zdir='z',offset=-2, cmap=cm.hot)
    ax.set_xlabel(r'$x_c$')
    ax.set_ylabel(r'$y_c$')
    ax.set_zlabel(r'$M_{tot}$')
    show()
    return None
