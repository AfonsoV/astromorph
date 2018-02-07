import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import numpy as np
import scipy.ndimage as snd
import matplotlib.patches as mpa
import matplotlib.pyplot as mpl
import argparse



def get_pixel_coords(imgname,x,y,hsize=1,verify_limits=True):
    hdu=pyfits.open(imgname)
    wcs=pywcs.WCS(hdu[0].header)

    ctype=hdu[0].header["ctype1"]
    xmax=hdu[0].header["naxis1"]
    ymax=hdu[0].header["naxis2"]

    if 'RA' in ctype:
        pixPos=np.array([[x,y]],np.float_)
    else:
        pixPos=np.array([[x,y]],np.float_)

    skycrd=wcs.wcs_pix2world(pixPos,1)

    ra=skycrd[0,0]
    dec=skycrd[0,1]

    return (ra,dec)


def get_image_extent(imgname):
    imgheader = pyfits.getheader(imgname)
    x0,y0 = get_pixel_coords(imgname,0,0)
    x1,y1 = get_pixel_coords(imgname,imgheader["NAXIS1"],imgheader["NAXIS2"])
    return (x0,x1,y0,y1)

parser = argparse.ArgumentParser(description='Process an image and produce a footprint file in ascii format')
parser.add_argument('images', metavar='filename', type=str, nargs='+',
                    help='a list of image names to be processed')
parser.add_argument("-p","--pixel", action="store_true", help="If given, footprint will be output in pixel coordnates: Default, ra,dec coordinates.")
parser.add_argument("-o","--output", help='The name of the output file. Default: filename.fpt')


if __name__ == "__main__":
    args = parser.parse_args()

    nimages = len(args.images)

    print("PROCESSING A TOTAL OF %i IMAGES"%(nimages))

    for i in range(nimages):
        print("\t%i:Processing image %s"%(i+1,args.images[i]))
        imgdata = pyfits.getdata(args.images[i])
        
        # if imgdata.shape[0]>1000:
        #     imgdata =snd.zoom(imgdata,1000/imgdata.shape[0])
        smap = np.zeros_like(imgdata)
        smap[np.isnan(imgdata)]=0
        smap[(imgdata!=0)]=1


        # fig,ax=mpl.subplots(1,1)

        if not args.pixel:
            extent = get_image_extent(args.images[i])
            # mpl.imshow(imgdata,vmin=-0.001,vmax=0.01,extent=extent)
            cnt = mpl.contour(smap,levels=[0.5],extent=extent)
        else:
            cnt = mpl.contour(smap,levels=[0.5])

        outlines = cnt.collections[0].get_paths()

        fname = args.images[i][:-5]
        print("\tWriting to %s.fpt"%(fname))
        f_out = open("%s.fpt"%(fname),"w")
        for i,path in enumerate(outlines):
            f_out.write("# Region %i\n"%(i+1))
            coords = path.vertices
            f_out.write(" ".join(["%9.6f"%c for c in coords.ravel()]))
            f_out.write("\n")
        f_out.close()

        # mpl.show()
        mpl.close("all")
