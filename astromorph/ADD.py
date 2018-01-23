from . import utils
import numpy as np
import matplotlib.pyplot as mpl
from .CAS import petrosian_rad,Anel

def sbprofiler(img,segmap):
    r"""Code to generate a 'surface brightness profile' based on the position
    of individual spatial bins (by default a spatial bin = 1 pixel). Surface brightness
    values and distance values are normalized to a specific radius (half-ligh, petrosian,...)
    and the SB is normalized by the total surface brightness inside that radius.

    Parameters
    ----------

    Returns
    -------

    References
    ----------

    Examples
    --------

    """
    xc,yc = utils.barycenter(img,segmap)
    # xc,yc = np.where(img==np.amax(img))

    rp,_ = petrosian_rad(img,xc,yc)

    gal = img*segmap
    Npix = np.size(gal[gal!=0])
    dmat = utils.compute_ellipse_distmat(img,xc,yc)
    Ip = np.sum(gal[dmat<rp])/Npix

    IS = (gal[gal!=0]/Ip).ravel()
    RS = (dmat[gal!=0]/rp).ravel()
    return RS,IS

def surface_brightness_profile(image,segmap,q=1.0,angle=0.0,rmax=None,rstep=3):
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
    xc,yc = utils.barycenter(image,segmap)

    dmat = utils.compute_ellipse_distmat(image,xc,yc,q,angle)
    if rmax is None:
        rmax = maxSegMapDistance(segmap)

    radius = np.arange(1,rmax,rstep)
    fluxes = [Anel(image,r,dmat) for r in radius]
    return radius,fluxes

def vertices(smap):
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
    assert np.size(smap[smap>0])>0,"There are no non-zero elements on the segmentation map"
    fig,ax=mpl.subplots()
    outline = ax.contour(smap,levels=[0.5])
    outlinePath = outline.collections[0].get_paths()
    mpl.close(fig)
    # mpl.show()
    return [p.vertices for p in outlinePath]

def maxSegMapDistance(smap):
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
    border = vertices(smap)
    if len(border)==0:
        return 0
    if len(border)>1:
        raise ValueError("This method only works with single contiguous regions")

    points = border[0]
    dmax = 0
    for p in points:
        distances = utils.dist(p[0],p[1],points[:,0],points[:,1])
        mdist = distances.max()
        if mdist > dmax:
            dmax = mdist
    return dmax

def filamentarity(smap):
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
    """ Defined to compute the area (in pixel) of the galaxy as defined from its
    segmentation map and compare it to the minimum area of the circle that encloses
    the galaxy. (see Matsuda et al. 2011)
    """
    dmax = maxSegMapDistance(smap)
    area_circle = np.pi*(dmax)**2/4.0
    area_galaxy = np.size(smap[smap>0])
    return area_galaxy/area_circle
