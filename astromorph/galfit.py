import subprocess as sp
import astropy.io.fits as pyfits

def get_fixpars_default():
    r""" Returns the default dictionary containing the information on whether
    or not to fix any parameter of the fit. By default, all parameters are
    not fixed.

    Parameters
    ----------

    Returns
    -------
    fixpars : dict
        A dictionary for each of the sersic parameters setting the fix/free key.

    References
    ----------

    Examples
    --------

    """
    return {'x':1,'y':1,'m':1,'re':1,'n':1,'q':1,'pa':1,'sky':1}

def write_object(model,x,y,m,re,n,ba,pa,num,fixpars=None):
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
    if fixpars is None:
        fixpars=get_fixpars_default()

    objString = ""
    objString += "#Object number: %i\n"%(num)
    objString += " 0) %s             # Object type\n"%(model)
    objString += " 1) %6.4f %6.4f  %i %i    # position x, y        [pixel]\n"%(x,y,fixpars['x'],fixpars['y'])
    objString += " 3) %4.4f      %i       # total magnitude\n"%(m,fixpars['m'])
    objString += " 4) %4.4f       %i       #     R_e              [Pixels]\n"%(re,fixpars['re'])
    objString += " 5) %4.4f       %i       # Sersic exponent (deVauc=4, expdisk=1)\n"%(n,fixpars['n'])
    objString += " 9) %4.4f       %i       # axis ratio (b/a)   \n"%(ba,fixpars['q'])
    objString += "10) %4.4f       %i       # position angle (PA)  [Degrees: Up=0, Left=90]\n"%(pa,fixpars['pa'])
    objString += " Z) 0                  #  Skip this model in output image?  (yes=1, no=0)\n"
    objString += " \n"
    return objString


def input_file(f,modelsString,magzpt,sky,x_range,y_range,sconvbox,pixscale,imgname='input.fits',outname="output.fits",psfname='none',maskname="none",signame='none',fixpars=None):
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
    if fixpars is None:
        fixpars=get_fixpars_default()

    assert len(x_range)==len(y_range)==2,"x_range,y_range must have two elements"
    assert x_range[1]>x_range[0],"x_range must be sorted in ascendent order"
    assert y_range[1]>y_range[0],"y_range must be sorted in ascendent order"

    f.write("================================================================================\n")
    f.write("# IMAGE and GALFIT CONTROL PARAMETERS\n")
    f.write("A) %s         # Input data image (FITS file)\n"%imgname)
    f.write("B) %s        # Output data image block\n"%outname)
    f.write("C) %s                # Sigma image name (made from data if blank or 'none' \n"%signame)
    f.write("D) %s         # Input PSF image and (optional) diffusion kernel\n"%psfname)
    f.write("E) 1                   # PSF fine sampling factor relative to data \n")
    f.write("F) %s                # Bad pixel mask (FITS image or ASCII coord list)\n"%maskname)
    f.write("G) none                # File with parameter constraints (ASCII file) \n")
    f.write("H) %i    %i   %i    %i # Image region to fit (xmin xmax ymin ymax)\n"%(x_range[0],x_range[1],y_range[0],y_range[1]))
    f.write("I) %i    %i          # Size of the convolution box (x y)\n"%(sconvbox,sconvbox))
    f.write("J) %7.5f             # Magnitude photometric zeropoint \n"%magzpt)
    f.write("K) %.3f %.3f        # Plate scale (dx dy)   [arcsec per pixel]\n"%(pixscale,pixscale))
    f.write("O) regular             # Display type (regular, curses, both)\n")
    f.write("P) 0                   # Options: 0=normal run; 1,2=make model/imgblock and quit\n")
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

    f.write(modelsString)

    f.write("# Object: Sky\n")
    f.write(" 0) sky                    #  object type\n")
    f.write(" 1) %7.4f      %i          #  sky background at center of fitting region [ADUs]\n"%(sky,fixpars['sky']))
    f.write(" 2) 0.0000      0          #  dsky/dx (sky gradient in x)\n")
    f.write(" 3) 0.0000      0          #  dsky/dy (sky gradient in y)\n")
    f.write(" Z) 0                      #  output option (0 = resid., 1 = Dont subtract)")
    f.close()
    return


def galaxy_maker(mag_zpt,xsize,ysize,model,xc,yc,pars,theta,sky=0,psfname="none"):
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

    if model=='sersic':
        mag,re,n,q = pars
        modelPars = write_object(model,xc,yc,mag,re,n,q,theta,1)
    elif model=='expdisk':
        mag_d,h,q=pars
        modelPars = write_object(f,model,xc,yc,mag_d,h,1.0,q,theta,1)
    elif model=='devauc':
        mag_b,re,q=pars
        modelPars = write_object(f,model,xc,yc,mag_b,re,4.0,q,theta,1)
    elif model=='compost':
        mag_d,rd,mag_b,rb,ba=pars
        theta_b,theta_d=theta
        modelPars = write_object(f,'expdisk',xc,yc,mag_d,rd,1.0,ba,theta_d,1)
        modelPars += write_object(f,'devauc',xc,yc,mag_b,rb,4.0,ba,theta_b,2)
    else:
        raise ValueError('model %s does not exist, please choose from: sersic,expdisk, devauc or compost.')
        return None

    f1=open('GALFIT_input','w')
    input_file(f1,modelPars,mag_zpt,sky,(1,xsize),(1,ysize),xsize,1,\
                imgname="none",outname="model.fits",psfname=psfname)
    f1.close()
    sp.call('galfit -o1 GALFIT_input >> galfit.log',shell=True)
    galaxy=pyfits.getdata('model.fits')
    sp.call('rm model.fits GALFIT_input galfit.log',shell=True)

    return galaxy

def write_gaussian(xc,yc,mag,fwhm,axis_ratio,theta):
    gaussPars = ""
    gaussPars+= "0) gaussian           # object type\n"
    gaussPars+= "1) %.4f  %.4f  1 1  # position x, y        [pixel]\n"%(xc,yc)
    gaussPars+= "3) %.4f      1       # total magnitude\n"%(mag)
    gaussPars+= "4) %.4f        1       #   FWHM               [pixels]\n"%(fwhm)
    gaussPars+= "9) %.4f       1       # axis ratio (b/a)\n"%(axis_ratio)
    gaussPars+= "10) %.4f         1       # position angle (PA)  [Degrees: Up=0, Left=90]\n"%(theta)
    gaussPars+= "Z) 0                  # leave in [1] or subtract [0] this comp from data?\n"
    return gaussPars

def make_psf(mzpt,FWHM=3,psfname="psf.fits"):
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
    fwhm_pix = FWHM
    imsize = int(20 * fwhm_pix)


    xc,yc=imsize/2.0+1,imsize/2.0+1
    psfPars = write_gaussian(xc,yc,14.0,fwhm_pix,1.0,0.0)

    f=open('galfit_psf.txt','w')
    input_file(f,psfPars,mzpt,0.0,(1,imsize),(1,imsize),0,1,imgname="none",outname=psfname)
    f.close()

    sp.call('galfit -o1 galfit_psf.txt >> galfit.log',shell=True)
    sp.call('rm galfit_psf.txt galfit.log',shell=True)
    return pyfits.getdata("psf.fits")

def read_results_file(fname):
    r""" Reads a galfit HDU cube to get the results for the best fit model,
    which are stored in the header of the cube extension 2. It is assumed that
    a single sersic model was fit.

    Parameters
    ----------
    fname : str
        The name of the result file to read the model parameters from.

    Returns
    -------
    xc : str
        The model X center and its error (separated by whitespace)
    yc : str
        The model Y center and its error (separated by whitespace)
    M : str
        The model magnitude and its error (separated by whitespace)
    R : str
        The model effective radius and its error (separated by whitespace)
    N : str
        The model sersic index and its error (separated by whitespace)
    Q : str
        The model axis ratio and its error (separated by whitespace)
    T : str
        The model position angle and its error (separated by whitespace)
    chi : str
        The reduced chi square value of the fit
    F : str
        A galfit flag indicating the final status of the fit. If 0 everything
        is ok. If 1 it means that GALFIT finished 100 iterations without
        converging. If 2, it means GALFIT results diverged and they cannot
        be trusted,


    References
    ----------

    Examples
    --------

    """
    try:
        hdu=pyfits.open(fname)
        chi=hdu[2].header['CHI2NU']
        xc=hdu[2].header['1_XC'].translate(None,'[*').replace('+/-','').replace(']',' -99.00')
        yc=hdu[2].header['1_YC'].translate(None,'[*').replace('+/-','').replace(']',' -99.00')
        M=hdu[2].header['1_MAG'].translate(None,'[*').replace('+/-','').replace(']',' -99.00')
        R=hdu[2].header['1_RE'].translate(None,'[*').replace('+/-','').replace(']',' -99.00')
        N=hdu[2].header['1_N'].translate(None,'[*').replace('+/-','').replace(']',' -99.00')
        Q=hdu[2].header['1_AR'].translate(None,'[*').replace('+/-','').replace(']',' -99.00')
        T=hdu[2].header['1_PA'].translate(None,'[*').replace('+/-','').replace(']',' -99.00')
        F= hdu[2].header['FLAGS'].replace(' ',',')
        if ('1' in F.split(',')):
            F=1
        elif ('2' in F.split(',')):
            F=2
        else:
            F=0
    except IOError:
        xc,yc,M,R,N,Q,T,chi,F="-99 -99","-99 -99","-99 -99","-99 -99","-99 -99","-99 -99","-99 -99","-99",-9
    return xc,yc,M,R,N,Q,T,chi,F