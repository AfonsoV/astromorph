from . import utils
import numpy as np

def quantile(img,segmap,q):
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
    """Computes the flux associated with the quantile value so that
    q% of the pixels are below that value"""
    gal_fluxes=img[segmap>0]
    Npix = np.size(gal_fluxes)
    F=0
    N=0
    dF=abs(np.amax(gal_fluxes)-np.amin(gal_fluxes))/100
    if dF==0:
        dF=0.01
    while N < q*Npix:
        N = np.size(gal_fluxes[gal_fluxes < F])
        F=F+dF
    return F

def grouping(img,segmap,q):
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
    """Generates a group map where all the pixels above the Flux F associated
    with the qauntile q are asnp.signed a value of one. All other pixels are left
    with zero value"""
    N=np.size(img,0)
    M=np.size(img,1)
    GROUPS = np.zeros([N,M])
    F = quantile(img,segmap,q)
    img=img*segmap
    GROUPS[img>=F]=1
    GROUPS = GROUPS*img
##    imshow(GROUPS)
##    colorbar()
##    text(0,0,str(F),color='white')
    return GROUPS

def find_groups(grp):
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
    """Finds the diferent non-contiguous groups and computes the area associated
    with each one."""
    Regions,Nregions=sci_nd.label(grp) #selects the non-contiguous regions asnp.signing different sequential values to each pixel belonging to a given group
    Areas=np.zeros(Nregions)
    #Fmax=np.zeros(Nregions)
    for i in range(Nregions):
        Areas[i]=np.size(grp[Regions==i+1])
        #Fmax[i]=max(grp[Regions==i+1])
##    title(str(Areas));show()
    return Areas

def sort_areas(Areas):
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
    """Sort the values in Areas in a decreasing way"""
    A=np.zeros(np.size(Areas))
    MAX = max(Areas)
    for i in range(len(A)):
        try:
            A[i] = Areas[Areas==MAX]
        except ValueError:
            A[i:i+len(Areas[Areas==MAX])] = Areas[Areas==MAX]
        try:
            MAX = max(Areas[Areas<MAX])
        except ValueError:
            return A
    return A

def multimode(img,segmap):
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
    """ Computes the multimode statistic from MID"""
    Qs=np.linspace(0.0,1.0,100,endpoint=True)
    Rs=np.zeros(len(Qs))
    for i in range(len(Qs)):
        Grp=grouping(img,segmap,Qs[i])
        Areas=find_groups(Grp)
        if len(Areas)>1:
            SortArea = sort_areas(Areas)
            Rs[i] = SortArea[1]**2/float(SortArea[0])
        else:
            continue
    return Rs,Qs

def find_neighbour_max(img,segmap,i,j):
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
    """ To find if any of the 8 adjacent pixels has higher fluxes that
    F(i,j). If yes, return maximum value coordinates of neighbour pixel.
    If not, (i,j) is returned"""
    gal = img * segmap
    if gal[i,j]==0:
        return -9,-9

#    if i>0 and j>0:
#        neighbors = gal[i-1:i+2,j-1:j+2]
#    if i>0 and j==0:
#        neighbors = gal[i-1:i+2,:j+2]
#    if i==0 and j>0:
#        neighbors = gal[:i+2,j-1:j+2]
#    if i==0 and j==0:
#        neighbors = gal[:i+2,:j+2]
#
#    maxi,maxj =  np.where(neighbors==np.amax(neighbors))
#    if len(maxi)>1:
#        raise ValueError("Two equal maxima!")
#
#    return maxi[0],maxj[0]
    maxi=i
    maxj=j
    Fij=gal[i,j]
    for l in range(i-1,i+2):
        for m in range(j-1,j+2):
            try:
                Nfij=gal[l,m]
            except IndexError:
                Nfij=-1e10
            if Nfij>Fij:
                maxi=l
                maxj=m
                Fij=Nfij
            else:
                continue
    return maxi,maxj

def find_local_max(img,segmap,i,j):
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
    """To find the local maximum associated with the pixel (i,j) by the maximum
    gradient path method."""
    ni,nj=find_neighbour_max(img,segmap,i,j)
    while i!=ni or j!=nj:
        i,j=ni,nj
        ni,nj=find_neighbour_max(img,segmap,i,j)
    return ni,nj

def local_maxims(img,segmap):
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
    """ Find all the local maxima associated with every pixel within the
    segmentation map and constructs the Group Intensity Map (a map describing)
    all the groups of pixels associated with every local maxima."""

    imax,imin,jmax,jmin=find_ij(segmap)
#    imax-=1
#    jmax-=1
    N=np.size(segmap,0)
    M=np.size(segmap,1)
    Imap=np.zeros([N,M])
    LM=[]
    mi,mj=find_local_max(img,segmap,imin,jmin)

    LM.append([mi,mj])
    for i in range(imin,imax):
        for j in range(jmin,jmax):
            mi,mj=find_local_max(img,segmap,i,j)
            if [mi,mj] not in LM:
                LM.append([mi,mj])
            Imap[i,j] = LM.index([mi,mj])+1
    return Imap,np.array(LM)

def intensity(Img,Imap,Segmap,LM):
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
    """Computes the intensity statistic from MID"""
    Nregions = int(np.amax(Imap))
    Fmap = Imap * Img * Segmap
    Is=np.zeros(Nregions)
    for n in range(Nregions):
        Is[n]= np.sum(Fmap[Imap==n+1])/(n+1)

    if Nregions>1:
        I=max(Is[Is<max(Is)])/max(Is)
    else:
        I=0

    return I,LM[Is==max(Is)][0]

def find_centroid(img,segmap):
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
    """ Computes the centroid of the light distribution as np.averaged by
    the pixel flux."""
    return utils.barycenter(img,segmap)

def deviation(img,segmap,imax_center):
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
    """ Computes the deviation statistic from MID """
    xcen,ycen=find_centroid(img,segmap)
    #print xcen,ycen
    x1,y1=imax_center
    nseg=np.size(segmap[segmap>0])
    return np.sqrt(np.pi/nseg)*np.sqrt((x1-xcen)**2+(y1-ycen)**2)
