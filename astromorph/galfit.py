import subprocess as sp
import astropy.io.fits as pyfits

def write_object(f,model,x,y,m,re,n,ba,pa,num,fixpars=None):
    if fixpars==None:
        fixpars={'x':1,'y':1,'m':1,'re':1,'n':1,'q':1,'pa':1}
    f.write("#Object number: %i\n"%num)
    f.write(' 0) %s             # Object type\n'%model)
    f.write(' 1) %6.4f %6.4f  %i %i    # position x, y        [pixel]\n'%(x,y,fixpars['x'],fixpars['y']))
    f.write(' 3) %4.4f      %i       # total magnitude\n' %(m,fixpars['m']))
    f.write(' 4) %4.4f       %i       #     R_e              [Pixels]\n'%(re,fixpars['re']))
    f.write(' 5) %4.4f       %i       # Sersic exponent (deVauc=4, expdisk=1)\n'%(n,fixpars['n']))
    f.write(' 9) %4.4f       %i       # axis ratio (b/a)   \n'%(ba,fixpars['q']))
    f.write('10) %4.4f       %i       # position angle (PA)  [Degrees: Up=0, Left=90]\n'%(pa,fixpars['pa']))
    f.write(' Z) 0                  #  Skip this model in output image?  (yes=1, no=0)\n')
    f.write(' \n')
    return


def galfit_input_file(f,magzpt,sky,xsize,ysize,sconvbox,pixscale,imgname='galaxy.fits',outname="results.fits",psfname='psf.fits',maskname="none",signame='none',fixpars=None):
    if fixpars==None:
        fixpars={'sky':1}
    f.write("================================================================================\n")
    f.write("# IMAGE and GALFIT CONTROL PARAMETERS\n")
    f.write("A) %s         # Input data image (FITS file)\n"%imgname)
    f.write("B) %s        # Output data image block\n"%outname)
    f.write("C) %s                # Sigma image name (made from data if blank or 'none' \n"%signame)
    f.write("D) %s         # Input PSF image and (optional) diffusion kernel\n"%psfname)
    f.write("E) 1                   # PSF fine sampling factor relative to data \n")
    f.write("F) %s                # Bad pixel mask (FITS image or ASCII coord list)\n"%maskname)
    f.write("G) none                # File with parameter constraints (ASCII file) \n")
    f.write("H) 1    %i   1    %i # Image region to fit (xmin xmax ymin ymax)\n"%(xsize+1,ysize+1))
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

    obj=open('galfit_object.temp','r')
    objects=obj.readlines()
    for line in objects:
        f.write(line)
    obj.close()

    f.write("# Object: Sky\n")
    f.write(" 0) sky                    #  object type\n")
    f.write(" 1) %7.4f      %i          #  sky background at center of fitting region [ADUs]\n"%(sky,fixpars['sky']))
    f.write(" 2) 0.0000      0          #  dsky/dx (sky gradient in x)\n")
    f.write(" 3) 0.0000      0          #  dsky/dy (sky gradient in y)\n")
    f.write(" Z) 0                      #  output option (0 = resid., 1 = Dont subtract)")
    f.close()
    return


def galaxy_maker_galfit(mag_zpt,xsize,ysize,model,xc,yc,pars,theta,sky=0):
    f=open('galfit_object.temp','w')
    if model=='sersic':
        mag,re,n,q = pars
        write_object(f,model,xc,yc,mag,re,n,q,theta,1)
    elif model=='expdisk':
        mag_d,h,q=pars
        write_object(f,model,xc,yc,mag_d,h,1.0,q,theta,1)
    elif model=='devauc':
        mag_b,re,q=pars
        write_object(f,model,xc,yc,mag_b,re,4.0,q,theta,1)
    elif model=='compost':
        mag_d,rd,mag_b,rb,ba=pars
        theta_b,theta_d=theta
        write_object(f,'expdisk',xc,yc,mag_d,rd,1.0,ba,theta_d,1)
        write_object(f,'devauc',xc,yc,mag_b,rb,4.0,ba,theta_b,2)
    else:
        raise ValueError('model %s does not exist, please choose from: sersic,expdisk, devauc or compost.')
        return None

    f.close()
    f1=open('GALFIT_input','w')
    galfit_input_file(f1,mag_zpt,sky,xsize,ysize,xsize,1,imgname="model.fits")
    f1.close()
    sp.call('galfit -o1 GALFIT_input >> galfit.log',shell=True)
    galaxy=pyfits.getdata('model.fits')
    sp.call('rm model.fits galfit_object.temp GALFIT_input galfit.log',shell=True)

    return galaxy


def make_psf(mzpt,FWHM=3):


    fwhm_pix = FWHM
    imsize = int(20 * fwhm_pix)


    xc,yc=imsize/2.0+1,imsize/2.0+1

    f=open('galfit_object.temp','w')

    f.write('0) gaussian           # object type\n')
    f.write('1) %.2f  %.2f  1 1  # position x, y        [pixel]\n'%(xc,yc))
    f.write('3) 14.0       1       # total magnitude\n')
    f.write('4) %.4f        0       #   FWHM               [pixels]\n'%(fwhm_pix))
    f.write('9) 1.0        1       # axis ratio (b/a)\n')
    f.write('10) 0.0         1       # position angle (PA)  [Degrees: Up=0, Left=90]\n')
    f.write('Z) 0                  # leave in [1] or subtract [0] this comp from data?\n')
    f.close()
    root=os.getcwd()

    f=open('galfit_psf.txt','w')
    galfit_input_file(f,mzpt,0.0,imsize,imsize,0,1,imgname='psf',maskname=False)
    f.close()

    sp.call('galfit -o1 galfit_psf.txt >> galfit.log',shell=True)
    sp.call('rm galfit_object.temp galfit_psf.txt galfit.log',shell=True)

    return

def read_results_file(fname):
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
