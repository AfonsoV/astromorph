from scipy.integrate import simps
import astropy.constants as const
import astropy.units as u
import numpy as np
from .config import ConfigFile


Obar = float(ConfigFile["cosmology"]["Obar"])
Omat = float(ConfigFile["cosmology"]["Omat"])
Ok = float(ConfigFile["cosmology"]["Ok"])
Orad = float(ConfigFile["cosmology"]["Orad"])
Ow = float(ConfigFile["cosmology"]["Ow"])
w = float(ConfigFile["cosmology"]["w"])
H0 = float(ConfigFile["cosmology"]["H0"])


Msun=const.M_sun              #kg
Mpc=(1 * u.Mpc).to("km")     #km
c=const.c.to("km/s")                #km/s


def hubble(z,pars=None):
    r"""Compute the value of the Hubble constant at redshift z.

    This will return the value of the Hubble constant as computed in terms
    of the Universe density parameters.

    Parameters
    ----------
    z : float, array
        The redshift at which to compute the value of the equation.
    pars : dictionary, optional
        A dictionary containing custom cosmological parameters to use in the
        equation. Accepted keys are  {'h','r','m','k','l','w'}. Default values
        are set in the pyfado.cfg file.
    Returns
    -------
    value : float, array
        The value of the Hubble constant at the requested redshifts computed
        with the given cosmological parameters.

    References
    ----------
    See https://en.wikipedia.org/wiki/Friedmann_equations for an overview

    Examples
    --------

    >>> friedmann(0.5)
    88.881171616412942
    >>> friedmann(0.5,pars={"h":0.7,"m":0.3,"l":0.7})
    91.60376629811681

    """
    z = np.asarray(z)
    P={'h':H0/100,'r':Orad,'m':Omat,'k':Ok,'l':Ow,'w':w}
    if not (pars==None):
        for p in pars:
            P[p] = pars[p]
    return 100*P['h']*np.sqrt(P['r']*(1+z)**4.+P['m']*(1+z)**3.+P['k']*(1+z)**2.+P['l']*(1+z)**(3*(1.+P['w'])))

def comov_rad(z,pars=None,npoints=10000):
    r""" Compute the comoving distance to redshift z.

    This will return the comoving distance as by integrating the hubble equation
    from 0 to z using npoints as the resolution.

    Parameters
    ----------
    z : float, array
        The redshift at which to compute the value of the equation.
    pars : dictionary, optional
        A dictionary containing custom cosmological parameters to use in the
        equation. Accepted keys are  {'h','r','m','k','l','w'}. Default values
        are set in the pyfado.cfg file.
    npoints : int, optional
        The number of points between 0 and z used to estimate the integral.

    Returns
    -------
    out : float, array
        The comoving distance to redshit z in Mpc.

    Raises
    ------
        ValueError if the redshift is negative

    See Also
    --------
        angular_distance, luminosity_distance

    References
    ----------
    See e.g. https://en.wikipedia.org/wiki/Comoving_and_proper_distances
    for details

    Examples
    --------

    >>> comov_rad(0.5)
    1958.1055013937253
    >>> comov_rad([0.5,0.7])
    array([ 1958.10550139,  2592.95086931])

    """
    z_arr = np.asarray(z)
    if z_arr.size==1:
        if type(z)==list:
            z=z[0]
        if z<0:
            raise ValueError("The redshift must be a positive value")
        if z==0:
            z=1e-4
        radius=[]
        z_points=np.linspace(1e-5,z,npoints)
        H=hubble(z_points,pars=pars)
        invH=1./H
        radius=simps(invH,z_points)
        return (c*radius).value
    else:
        radius_arr = np.empty_like(z_arr)
        for i,redshift in enumerate(z_arr):
            radius_arr[i] = comov_rad(redshift,pars=pars,npoints=npoints)
        return radius_arr

def comoving_volume(z1,z2,area=4*np.pi,pars=None,npoints=50):
    r""" Computes the comoving volume between two redshifts.


    Parameters
    ----------
    z1 : float
        Lower redshift limit.
    z2 : float
        Upper redshift limit.
    area : float, optional
        Sky area used to compute the volume, in radians. Default to the entire
        sky.
    pars : dictionary, optional
        A dictionary containing custom cosmological parameters to use in the
        equation. Accepted keys are  {'h','r','m','k','l','w'}. Default values
        are set in the pyfado.cfg file.
    npoints : int, optional
        The number of intermediate redshifts to estimate the total volume.
        Default: 50 points.

    Returns
    -------
        out: float
            The comoving volume beween z1 and z2 over the defined area.

    Other Parameters
    ----------------

    Raises
    ------
        ValueError if the redshift is negative

    See Also
    --------

    Notes
    -----

    References
    ----------

    Examples
    --------
    """

    assert z1<z2, "z1 must be smaller than z2"
    if z1<0:
        raise ValueError("Redshift must be positive")

    if z1==0:
        z1=1e-4
    z_points=np.linspace(z1,z2,npoints)
    H=Hubble(z_points,pars=pars)
    DA=np.array([angular_distance(zed) for zed in z_points])
    Zterm = (1+z_points)*(1+z_points)
    func_z = Zterm*DA*DA/H
    volume = area*simps(func_z,z_points)
    return (c*volume).value



def lookback_time(z,pars=None,npoints=10000):
    r""" Computes the lookback_time to redshift z.

    Parameters
    ----------
    z : float, array
        The redshift for which to compute the lookback time.
    pars : dictionary, optional
        A dictionary containing custom cosmological parameters to use in the
        equation. Accepted keys are  {'h','r','m','k','l','w'}. Default values
        are set in the pyfado.cfg file.
    npoints : int, optional
        The number of points between 0 and z used to estimate the integral.

    Returns
    -------
    out : float, array
        The lookback times (in Gyr) for the provided redshift(s).

    Other Parameters
    ----------------

    Raises
    ------
        ValueError if the redshift is negative

    See Also
    --------
        find_z_tL
    Notes
    -----

    References
    ----------

    Examples
    --------
    """
    z_arr = np.asarray(z)
    if z_arr.size==1:
        if type(z)==list:
            z=z[0]
        if z<0:
            raise ValueError("The redshift must be a positive value")
        if z==0:
            z=1e-4
        radius=[]
        z_points=np.linspace(1e-5,z,npoints)
        H=hubble(z_points,pars=pars)
        I = 1./((1+z_points)*H)
        tl = simps(I,z_points)*Mpc/(365.25*24*60*60*1e9)
        return tl.value
    else:
        t_arr = np.empty_like(z_arr)
        for i,redshift in enumerate(z_arr):
            t_arr[i] = lookback_time(redshift,pars=pars,npoints=npoints)
        return t_arr


def find_z_tL(time,pars=None):
    r""" Computes the redshift corresponding to the given lookback time (in Gyr).
    Parameters
    ----------
    time : float, array
        The lookback time for which we want to compute the redshift.
    pars : dictionary, optional
        A dictionary containing custom cosmological parameters to use in the
        equation. Accepted keys are  {'h','r','m','k','l','w'}. Default values
        are set in the pyfado.cfg file.

    Returns
    -------
    out : float, array
        The redshift corresponding to the given lookback time.

    Other Parameters
    ----------------

    Raises
    ------

    See Also
    --------
        lookback_time
    Notes
    -----

    References
    ----------

    Examples
    --------
    """
    from scipy.optimize import bisect
    def minim(z):
        return lookback_time(z,pars=None)-time
    res=bisect(minim,0,1e4)
    return res

def angular_distance(z,pars=None):
    r""" Computes the angular diameter distance to z.

    Parameters
    ----------
    z : float, array
        The redshift for which to compute the angular diameter distance.
    pars : dictionary, optional
        A dictionary containing custom cosmological parameters to use in the
        equation. Accepted keys are  {'h','r','m','k','l','w'}. Default values
        are set in the pyfado.cfg file.

    Returns
    -------
    out : float, array
        The angular diameter distance (in Mpc) for redshift z.

    Other Parameters
    ----------------

    Raises
    ------

    See Also
    --------
    luminosity_distance

    Notes
    -----

    References
    ----------

    Examples
    --------
    """
    return comov_rad(z,pars=pars)/(1+z)

def luminosity_distance(z,pars=None):
    r""" Computes the luminosity distance to z.

    Parameters
    ----------
    z : float, array
        The redshift for which to compute the angular diameter distance.
    pars : dictionary, optional
        A dictionary containing custom cosmological parameters to use in the
        equation. Accepted keys are  {'h','r','m','k','l','w'}. Default values
        are set in the pyfado.cfg file.

    Returns
    -------
    out : float, array
        The luminosity distance (in Mpc) for redshift z.

    Other Parameters
    ----------------

    Raises
    ------

    See Also
    --------
    angular_distance

    Notes
    -----

    References
    ----------

    Examples
    --------
    """
    return comov_rad(z,pars=pars)*(1+z)


def test_volume():
    r""" Testing volume computation.
    Parameters
    ----------
    None
    Returns
    -------
    None
    """
    z1=0
    z2=np.arange(0.1,20.1,0.1)
    volumes = np.array([comoving_volume(z1,z,area=(np.pi/180)**2) for z in z2])
    fig,ax=mpl.subplots()
    ax.plot(z2,volumes/1e9,'-',linestyle='-',linewidth=3,color='red')
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$V(<z)\ [\mathrm{Gpc^{3}deg^{-2}}]$')
    ax.set_xlim(0,20)
    ax.minorticks_on()
    mpl.show()
    return

def plot_lookback():
    r""" Testing lookback time computation.
    Parameters
    ----------
    None
    Returns
    -------
    None
    """
    Zs = np.linspace(1e-5,10,1000)
    times=np.zeros(len(Zs))
    for i in range(len(Zs)):
        times[i]=lookback_time(Zs[i])
    fig,ax=mpl.subplots()
    ax.plot(Zs,times,'k',lw=3)
    ax.hlines(13.8,0,10,'k',':')
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$t_L\ [\mathrm{Gyr}]$')
    aux = fig.add_axes([0.5,0.2,0.3,0.3])
    aux.plot(Zs,times,'k',lw=3)
    aux.set_xlim(2,6)
    aux.set_ylim(10,13)
    aux.grid(True)
    mpl.show()
    return None

def test_distances():
    r""" Testing distances computation.
    Parameters
    ----------
    None
    Returns
    -------
    None
    """
    Zs = np.linspace(1e-5,10,1000)
    distsA=np.zeros(len(Zs))
    distsA2=np.zeros(len(Zs))
    distsL=np.zeros(len(Zs))
    distsL2=np.zeros(len(Zs))
    for i in range(len(Zs)):
        distsA[i]=angular_distance(Zs[i])
        distsA2[i]=angular_distance(Zs[i],pars={'l':0.0,'m':1.0})
        distsL[i]=luminosity_distance(Zs[i])
        distsL2[i]=luminosity_distance(Zs[i],pars={'l':0.0,'m':1.0})
    fig,(ax1,ax2)=mpl.subplots(1,2,figsize=(15,8))
    ax1.plot(Zs,distsA,'b-',Zs,distsA2,'r-')
    ax1.set_xlabel(r'$z$')
    ax1.set_ylabel(r'$d_A$')
    ax2.plot(Zs,distsL,'b-',Zs,distsL2,'r-')
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$d_L$')
    mpl.show()
    return None

def plot_conversion_arcsec_to_kpc():
    r""" Testing arcsec to kpc conversion.
    Parameters
    ----------
    None
    Returns
    -------
    None
    """
    Zs = np.linspace(0.1,6.9,1000)
    dists_Planck=np.array([angular_distance(z) for z in Zs])
    kpc_per_arcsec = 1*dists_Planck/(180/np.pi*3600)*1000
    fig,ax=mpl.subplots(figsize=(15,8))
    ax.plot(Zs,1.0/kpc_per_arcsec,'-',color='RoyalBlue',lw=3)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$S\ [\mathrm{arcsec/kpc}]$')
    ax.minorticks_on()
    ax.set_xlim(0.1,6.5)
    fig.savefig('Scale_arcsec_per_kpc.png')
    mpl.show()
    return None

def compare_WMAP_Planck_cosmologies():
    r""" Comparing WMAP to Planck cosmology impact on angular_distance
    Parameters
    ----------
    None
    Returns
    -------
    None
    """
    Zs = np.linspace(0.1,6.9,1000)
    dists_Planck=np.array([angular_distance(z) for z in Zs])
    dists_WMAP=np.array([angular_distance(z,pars={'h':0.7,'m':0.3,'l':0.7}) for z in Zs])
    diff = dists_Planck/dists_WMAP
    fig,ax=mpl.subplots(2,1,sharex=True,figsize=(15,13))
    mpl.subplots_adjust(hspace=0.0)
    ax[1].plot(Zs,diff,'k-',lw=2)
    ax[1].set_xlabel(r'$z$')
    ax[1].set_ylabel(r'$D_A^\mathrm{Planck}/D_A^\mathrm{STD}$')
    ax[1].hlines(1,0,7,'r','--',alpha=0.5,lw=2)
    ax[1].set_xlim(0.1,6.9)
    ax[1].set_ylim(0.985,1.045)
    ax[0].plot(Zs,dists_WMAP,'r-',lw=2,label=r'STD: $H_0=70\mathrm{km\ s^{-1}Mpc^{-1}}; \Omega_m=0.3; \Omega_\Lambda=0.7$')
    ax[0].plot(Zs,dists_Planck,'g-',lw=2,label='Planck')
    ax[0].set_ylabel(r'$D_A\ [\mathrm{Mpc}]$')
    ax[0].legend(loc='lower right')
    ax[0].set_ylim(250,1900)
    # fig.savefig('Impact_Cosmology_Choice.png')
    mpl.show()
    return None

if __name__=='__main__':
    import matplotlib.pyplot as mpl
