import numpy as np
import matplotlib
import matplotlib.pyplot as mpl
import matplotlib.colors as clr
from matplotlib.patches import Rectangle,Ellipse,Circle,Polygon

def make_subplot(nrows,ncols,fignum=10,width=20,height=8):
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
    fig=mpl.figure(fignum,figsize=(width,height))
    gs=mpl.GridSpec(nrows,ncols)
    ax=[]
    for i in range(ncols*nrows):
        ax.append(mpl.subplot(gs[i]))
    return ax


def all_vertices(segmap):
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
    border = np.zeros(segmap.shape)
    for i in range(segmap.shape[0]):
        indices=np.where(segmap[i,:]==1)[0]
        k=0
        ks=0
        if len(indices)>0:
            start_min=min(indices)
            end_max=max(indices)
            border[i,start_min]=1
            border[i,end_max]=1
            while k <len(indices):
                if indices[k]==start_min+ks:
                    k+=1
                    ks+=1
                else:
                    start_min=indices[k]
                    end_max=indices[k-1]
                    border[i,end_max]=1
                    border[i,start_min]=1
                    k+=1
                    ks=1

    for j in range(segmap.shape[1]):
        indices=np.where(segmap[:,j]==1)[0]
        k=0
        ks=0
        if len(indices)>0:
            start_min=min(indices)
            end_max=max(indices)
            border[start_min,j]=1
            border[end_max,j]=1
            while k <len(indices):
                if indices[k]==start_min+ks:
                    k+=1
                    ks+=1
                else:
                    start_min=indices[k]
                    end_max=indices[k-1]
                    border[start_min,j]=1
                    border[end_max,j]=1
                    k+=1
                    ks=1

    return border

def draw_border(eixo,segmap,color,dilate=False,**kwargs):
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
    if dilate:
        V = all_vertices(sci_nd.binary_dilation(segmap,structure=define_structure(5)))
    else:
        V = all_vertices(segmap)
    Vselect=np.ma.masked_where(V==0,V,copy=True)
    eixo.imshow(Vselect,cmap=border_color(color),alpha=1.0,vmin=0.1,vmax=0.9,**kwargs)
    return

def border_color(color):
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
    r,g,b = clr.hex2color(clr.cnames[color])
    colormap = {'red':((0.,0.,0.),\
                       (1.0,r,1.0)),\

                'green':((0.,0.,0.),\
                       (1.0,g,1.0)),\

                'blue':((0.,0.,0.),\
                       (1.0,b,1.0))}

    my_cmap = clr.LinearSegmentedColormap(color,colormap)
    return my_cmap


def gen_circle(eixo,xc,yc,r,color='white',**kwargs):
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
    "Draws a white circle on plot eixo with center at (xc,yc) ad radius r"
    C = Circle((xc,yc),radius=r,color=color,fill=False,**kwargs)
    eixo.add_artist(C)
    return None

def gen_ellipse(eixo,xc,yc,r,q,theta,color='white',**kwargs):
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
    """Draws a white ellipse on plot eixo with center at (xc,yc)
    and radius r, axis ratio q and angle theta"""
    E = Ellipse((xc,yc),width=2*r,height=2*q*r,angle=theta,color=color,fill=False,**kwargs)
    eixo.add_artist(E)
    return None

def gen_rectangle(eixo,xc,yc,dx,dy,color='white',lw=2,angle=0.):
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
    """Draws a color rectangle on plot eixo with center at (xc,yc)
    and sides dx and dy."""
    R = Rectangle((xc-dx/2.,yc-dy/2.),dx,dy,angle=angle,color=color,fill=False,linewidth=lw)
    eixo.add_artist(R)
    return None

def vertices(segmap,direction):
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
    "Returns the polygon vertices of the segmentation map"

    mins=[]
    maxs=[]
    if direction=='x':
        for i in range(segmap.shape[0]):
            indices=np.where(segmap[i,:]==1)[0]
            if len(indices)>0:
                mins.append([min(indices)-0.5,i])
                maxs.append([max(indices)+0.5,i])

    elif direction=='y':
        for j in range(segmap.shape[1]):
            indices=np.where(segmap[:,j]==1)[0]
            if len(indices)>0:
                mins.append([j,min(indices)-0.5])
                maxs.append([j,max(indices)+0.5])


    if len(mins)==1:
        mins=[copy(mins[0]),copy(mins[0])]
        maxs=[copy(maxs[0]),copy(maxs[0])]

    mins[0][0]-=0.5
    mins[-1][0]+=0.5
    maxs[0][0]-=0.5
    maxs[-1][0]+=0.5

    for p in maxs[::-1]:
        mins.append(p)

    return mins



def draw_segmap_border(eixo,segmap,pixscale=-99,color='red',lw=2,distinct=False,direction='y'):
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
    """ Rotine to draw the borders of the segmentation map in axes eixo.
    """
    Regions,Nregions=sci_nd.label(segmap)
    if distinct:
        a=np.linspace(0,1,np.amax(segmap))
        import random as rdm
        rdm.shuffle(a)
        segvals=[]

    for n in range(1,Nregions+1):
        A=np.zeros(Regions.shape)
        A[Regions==n]=1
        ymins=np.array(vertices(A,direction=direction))

        if pixscale!=-99:
            ymins-=Regions.shape[0]/2.0
            ymins*=pixscale

        if distinct:
            segval=np.amax(segmap[Regions==n])
            if segval in segvals:
                pass
            else:
                segvals.append(segval)

            P=Polygon(ymins,fill=False,color=cm.hsv(a[segval-1]),linewidth=lw)
        else:
            P=Polygon(ymins,fill=False,color=color,linewidth=lw)
        eixo.add_artist(P)

    return

def draw_cross(eixo,x,y,gap=0.75,size=1.0,**kwargs):
    eixo.plot([x,x],[y-gap,y-gap-size],**kwargs)
    eixo.plot([x,x],[y+gap,y+gap+size],**kwargs)
    eixo.plot([x-gap,x-gap-size],[y,y],**kwargs)
    eixo.plot([x+gap,x+gap+size],[y,y],**kwargs)
    return None


def show_image(axes,data,scale="linear",minflux=1e-10,**kwargs):
    if scale == "linear":
        axes.imshow(data,**kwargs)
    elif scale == "sqrt":
        axes.imshow(np.sqrt(data-data.min()+minflux),**kwargs)
    elif scale == "log":
        axes.imshow(np.log10(data-data.min()+minflux),**kwargs)
    else:
        raise ValueError("scale must be one of: linear, sqrt or log")
    return
