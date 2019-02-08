import numpy as np
import matplotlib
import matplotlib.pyplot as mpl
import matplotlib.ticker as mpt
import astromorph.cosmology as cosmos
import scipy.ndimage as snd
import h5py

def exit_code(event):
    if event.key=='escape':
        sys.exit()
    if event.key=='q':
        mpl.close('all')


def find_level(histogram,level,nslices=1000):

    HTOT = np.sum(histogram)
    HMAX = np.amax(histogram)
    HMIN = np.amin(histogram)

    slices = np.linspace(HMAX,HMIN,nslices)

    for i in range(nslices):
        if np.sum(histogram[histogram>slices[i]]) > level*HTOT:
            break
        else:
            continue

    return slices[i]


def draw_image(eixo,ID,image,pixscale,redshift):

    N,M=image.shape

    eixo.tick_params(labelbottom='off',labelleft='off',left='off',right='off',top='off',bottom='off')

    imax = 0.75*np.amax(image)
    imin = np.amin(image)

    eixo.imshow(image,extent=(-N/2.*pixscale,N/2.*pixscale,-M/2.*pixscale,M/2.*pixscale),aspect='equal',vmin=imin,vmax=imax)

    dpc = 5
    anchor = -1.45
    darcsec = dpc*(180./np.pi*3600.)/1000./cosmos.angular_distance(redshift)
    eixo.hlines(anchor,anchor,anchor+darcsec,color='white',linewidth=2,linestyle='solid')
    # eixo.set_title(imgname.split('/')[0])

    eixo.text(anchor+darcsec/2,anchor,'%i kpc'%(dpc),color='white',weight='bold',ha='center',va='bottom')

    eixo.set_xlim(-1.6,1.6)
    eixo.set_ylim(-1.6,1.6)
    eixo.minorticks_off()

    return eixo



def corner_plot_wImage(ID,image,pixscale,variables,labels,nbins=75,redshift=2):

    nvars=len(variables)
    fig,ax = mpl.subplots(nvars,nvars,figsize=(5*len(variables),5*len(variables)),sharex='col')
    fig.subplots_adjust(hspace=0.025,wspace=0.025)

    fig,ax = corner_plot(variables,labels,nbins=nbins,figax=(fig,ax))

    Bbox = ax[0,nvars-1].get_position()
    axIm = fig.add_axes([Bbox.x0,Bbox.y0,Bbox.width,Bbox.height])
    axIm.set_visible(True)

    axIm = draw_image(axIm,ID,image,pixscale,redshift)

    return fig,ax

def corner_plot(variables,labels,nbins=75,figax=None):

    nvars=len(variables)
    if figax==None:
        fig,ax = mpl.subplots(nvars,nvars,figsize=(5*len(variables),5*len(variables)),sharex='col')
        fig.subplots_adjust(hspace=0.025,wspace=0.025)
    else:
        fig,ax=figax

    cmap=matplotlib.cm.get_cmap("magma")
    lineColor = "PowderBlue"
    errorBarColor = "RoyalBlue"

    bin_vars = [np.linspace(min(var),max(var),nbins) for var in variables]
    vals_and_errs = np.array([np.percentile(var,[16,50,84]) for var in variables])
    for i in range(nvars):
        for j in range(nvars):

            if j>i:
                ax[i,j].set_visible(False)
            elif i==j:
                hy,ey,py=ax[i,j].hist(variables[i],bins=bin_vars[i],color=cmap(0.99),histtype='stepfilled')
                ax[i,j].vlines(vals_and_errs[i,1],0,1.1*np.amax(hy),color=lineColor,linewidth=2)
                # if "r_e" in labels[j]:
                #     ax[i,j].vlines(0.662,0,1.1*np.amax(hy),color="GoldenRod",linewidth=3)
                ax[i,j].set_ylim(0,1.1*np.amax(hy))
            else:
                H,Xe,Ye = np.histogram2d(variables[j],variables[i],bins=nbins)
                Z = snd.gaussian_filter(H,2)

                levels_contour = [find_level(Z,l) for l in [0.99,0.95,0.68]]
                Zshow = np.ma.masked_where(Z<levels_contour[0],Z)
                ax[i,j].imshow(Zshow.T,cmap=cmap,extent=(min(Xe),max(Xe),min(Ye),max(Ye)),aspect='auto')
                # ax[i,j].plot(variables[j],variables[i],",",color=cmap(0),zorder=-1)
                ax[i,j].contour(Z.T,levels=levels_contour[1:],extent=(min(Xe),max(Xe),min(Ye),max(Ye)),colors="white")
                ax[i,j].vlines(vals_and_errs[j,1],min(Ye),max(Ye),color=lineColor,linewidth=2)
                ax[i,j].hlines(vals_and_errs[i,1],min(Xe),max(Xe),color=lineColor,linewidth=2)
                ax[i,j].errorbar([vals_and_errs[j,1]],[vals_and_errs[i,1]],\
                        xerr=[[vals_and_errs[j,1]-vals_and_errs[j,0]],[vals_and_errs[j,2]-vals_and_errs[j,1]]],\
                        yerr=[[vals_and_errs[i,1]-vals_and_errs[i,0]],[vals_and_errs[i,2]-vals_and_errs[i,1]]],\
                        ecolor=errorBarColor,elinewidth=2,capthick=2,markersize=15)


                # if "r_e" in labels[i]:
                #     ax[i,j].hlines(0.662,min(Xe),max(Xe),color="GoldenRod",linewidth=3)
                # elif "r_e" in labels[j]:
                #     ax[i,j].vlines(0.662,min(Ye),max(Ye),color="GoldenRod",linewidth=3)

            ax[i,j].xaxis.set_major_formatter(mpt.ScalarFormatter(useOffset=False))
            ax[i,j].xaxis.set_major_locator(mpt.MaxNLocator(3))
            ax[i,j].minorticks_on()

            if i==nvars-1:
                ax[i,j].set_xlabel(labels[j])

            if j==0 and i>0:
                ax[i,j].set_ylabel(labels[i])


            if j>0:
                ax[i,j].tick_params(labelleft='off')
            if j==0 and i==0:
                ax[i,j].tick_params(labelleft='off')
            if i<nvars-1:
                ax[i,j].tick_params(labelbottom='off')


    return fig,ax

def simpler_mag_rad_plot(ID,z,zflag,imgname,pixscale):

    image = pyfits.getdata(imgname)
    N,M=image.shape

    fig = mpl.figure(figsize=(12.5,14))

    axM = fig.add_axes([0.1,0.1,0.65,0.65])
    axHX = fig.add_axes([0.1,0.75,0.65,0.20],sharex=axM)
    axHY = fig.add_axes([0.75,0.1,0.20,0.65],sharey=axM)
    axIm = fig.add_axes([0.75,0.75,0.20,0.20])


    axHX.tick_params(labelbottom='off')
    axHY.tick_params(labelleft='off')

    axIm = draw_image(axIm,ID,z,zflag,imgname,pixscale)

    eixos = [axM,axHX,axHY]

    H,Xe,Ye = np.histogram2d(magnitudes,radius,bins=100)
    Z = snd.gaussian_filter(H,2.5)

    levels_contour = [find_level(Z,l) for l in [0.99,0.95,0.68]]

    if levels_contour[0]== levels_contour[1]:
        levels_contour[1]+=1e-3

    hx,ex,px=axHX.hist(magnitudes,bins=Xe,color='SteelBlue',histtype='stepfilled')
    hy,ey,py=axHY.hist(radius,bins=Ye,orientation='horizontal',color='SteelBlue',histtype='stepfilled')

    axHX.vlines(magmeanval,0,1.1*np.amax(hx),color='indianred',lw=1.5)
    axHY.hlines(radmeanval,0,1.1*np.amax(hy),color='indianred',lw=1.5)

    axHX.set_ylim(0,1.1*np.amax(hx))
    axHY.set_xlim(0,1.1*np.amax(hy))


    CMAP = matplotlib.cm.get_cmap('Blues')

##        CMAP.set_under('white')

    X1S,Y1S = np.where(Z>levels_contour[-1])
    magmax,magmin=Xe[np.amax(X1S)],Xe[np.amin(X1S)]
    remax,remin=Ye[np.amax(Y1S)],Ye[np.amin(Y1S)]


    img = axM.imshow(Z.T,cmap=CMAP,extent=(min(Xe),max(Xe),min(Ye),max(Ye)),aspect='auto')
    axM.contour(Z.T,levels=levels_contour,extent=(min(Xe),max(Xe),min(Ye),max(Ye)),cmap='viridis')
    axM.vlines(magmeanval,min(Ye),max(Ye),color='indianred',lw=1.5)
    axM.hlines(radmeanval,min(Xe),max(Xe),color='indianred',lw=1.5)

##    axM.errorbar([magmeanval],[radmeanval],xerr=[[magmeanval-magmax],[magmin-magmeanval]],yerr=[[radmeanval-remin],[remax-radmeanval]],ecolor='Black',elinewidth=2,capthick=2,markersize=15)
    axM.errorbar([magmeanval],[radmeanval],xerr=[[magmeanval-maglowbound],[maghighbound-magmeanval]],yerr=[[radmeanval-radlowbound],[radhighbound-radmeanval]],ecolor='Crimson',elinewidth=2,capthick=2,markersize=15)

    axM.xaxis.set_major_formatter(mpt.ScalarFormatter(useOffset=False))
    axM.set_xlabel(r'$mag$')
    axM.set_ylabel(r'$r_e\ [\mathrm{pc}]$')

    fig.canvas.mpl_connect('key_press_event',exit_code)

    axHY.xaxis.set_major_locator(mpt.MaxNLocator(3))
    axHX.yaxis.set_major_locator(mpt.MaxNLocator(3))
    axM.xaxis.set_major_locator(mpt.MaxNLocator(4))
    for eixo in eixos:
        eixo.minorticks_on()

    return fig,eixos


if __name__ == "__main__":
    PIXSCALE = 0.03

    catalog = np.loadtxt("udrop.cat",dtype={"names":("ID","z","zconf"),"formats":("U50","f4","f4")})

    COLS = ("ID","z","mag","mag_errL","mag_errU","re","re_errL","re_errU","q")
    FMTS = ("U50","f4","f4","f4","f4","f4","f4","f4","f4")
    CLNM = (0,3,10,11,12,13,14,15,16)


    CLRS = ["DodgerBlue","ForestGreen","Crimson","DarkOrange","Indigo","Goldenrod"]
    MRKRS = ["s","D","o","p","d","8"]
    fig,ax = mpl.subplots()
    for i,cluster in enumerate(["A370"]):
        tableResults = np.loadtxt("mcmcResults_%s_wLensing.txt"%(cluster),\
                                  dtype={"names":COLS,"formats":FMTS},usecols=CLNM)

        IDs = [str(name) for name in tableResults["ID"]]
        redshifts = tableResults["z"]
        dAng = np.asarray([cosmos.angular_distance(z) for z in redshifts])
        conversionPixToKpc = PIXSCALE*dAng/(180./np.pi*3600.)*1000.
        rePhys = tableResults["re"]*conversionPixToKpc* np.sqrt(tableResults["q"])
        rePhysEL = tableResults["re_errL"]*conversionPixToKpc* np.sqrt(tableResults["q"])
        rePhysEU = tableResults["re_errU"]*conversionPixToKpc* np.sqrt(tableResults["q"])
        mag = tableResults["mag"]
        magEL = tableResults["mag_errL"]
        magEU = tableResults["mag_errU"]

        ax.plot(mag,rePhys,MRKRS[i],label=cluster,color=CLRS[i],zorder=1+i)
        ax.errorbar(mag,rePhys,xerr=[magEL,magEU],yerr=[rePhysEL,rePhysEU],fmt=",",\
                    color="k",alpha=0.25,elinewidth=0.5,zorder=-1)




        table = open("results%s_udrop_sizes_Bruno_11Apr18.txt"%(cluster),"w")
        table.write("# ID z mag magErrDown magErrUp re reErrDown reErrUp\n")
        for i in range(len(IDs)):
            line = "%s %4.2f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f\n"%(IDs[i],\
                    redshifts[i],mag[i],magEL[i],magEU[i],\
                    rePhys[i],rePhysEL[i],rePhysEU[i])
            table.write(line)
        table.close()

    ax.semilogy()
    ax.yaxis.set_major_formatter(mpt.ScalarFormatter())
    ax.set_yticks([0.1,0.3,1,3,10])
    ax.legend()
    ax.set_ylabel(r"$r_{e,c}\ [\mathrm{kpc}]$")
    ax.set_xlabel(r"$m_\mathrm{GALFIT}$")
    fig.savefig("summaryPlot_udrop_sizesGalaxies_A370.png")


    tableResults = np.loadtxt("mcmcResults_A370_wLensing.txt",\
                            dtype={"names":COLS,"formats":FMTS},usecols=CLNM)

    rePhys = tableResults["re"]*conversionPixToKpc * np.sqrt(tableResults["q"])
    rePhysEL = tableResults["re_errL"]*conversionPixToKpc * np.sqrt(tableResults["q"])
    rePhysEU = tableResults["re_errU"]*conversionPixToKpc * np.sqrt(tableResults["q"])

    IDs = tableResults["ID"][(rePhys>1)*(rePhys<3)]
    res = rePhys[(rePhys>1)*(rePhys<3)]
    for i,name in enumerate(IDs):
        print(name,res[i])

    mag = tableResults["mag"]
    magEL = tableResults["mag_errL"]
    magEU = tableResults["mag_errU"]

    tableResultsBouwens = np.loadtxt("resultsA370_Bouwens.txt",\
                            dtype={"names":("mag","re","ID"),\
                            "formats":("f4","f4","U50")},usecols=[1,2,5])

    reBins = np.logspace(-2,1,16)

    clrM = "Navy"
    clrB = "DarkOrange"
    mB = tableResultsBouwens["mag"]
    rB = tableResultsBouwens["re"]
    fig,ax = mpl.subplots(1,2,sharey=True,figsize=(12,6))
    fig.subplots_adjust(wspace=0.0)
    ax[0].plot(mag,rePhys,"o",color=clrM,label="Bruno")
    ax[0].errorbar(mag,rePhys,xerr=[magEL,magEU],yerr=[rePhysEL,rePhysEU],\
                fmt=",",color="k",alpha=0.15,elinewidth=0.5,zorder=-1)
    ax[0].plot(mB,rB,"o",color=clrB,label="Bouwens")
    ax[0].semilogy()
    ax[0].yaxis.set_major_formatter(mpt.ScalarFormatter())
    ax[0].set_yticks([0.1,0.3,1,3,10])
    ax[0].set_ylabel(r"$r_{e,c}\ [\mathrm{kpc}]$")
    ax[0].set_xlabel(r"$m_\mathrm{YJH}$")
    Hm,_,_=ax[1].hist(rePhys,bins=reBins,orientation="horizontal",histtype="step",\
                hatch=r"\\",color=clrM)
    Hb,_,_=ax[1].hist(rB,bins = reBins,orientation="horizontal",histtype="step",\
                hatch=r"//",color=clrB)

    maxHist = max(np.amax(Hb),np.amax(Hm))
    ax[1].hlines(np.median(rePhys),0,1.1*maxHist,color=clrM)
    ax[1].hlines(np.median(rB),0,1.1*maxHist,color=clrB)
    ax[1].set_xlim(0,1.05*maxHist)
    ax[1].set_xlabel(r"$N_\mathrm{gal}$")
    ax[1].semilogy()
    ax[1].text(0.45*maxHist,10,r"$\widetilde{r_{e,c}}=%.2f\ \mathrm{kpc}$"\
                                %(np.median(rePhys)),color=clrM,fontsize=18)
    ax[1].text(0.45*maxHist,6,r"$\widetilde{r_{e,c}}=%.2f\ \mathrm{kpc}$"\
                                %(np.median(rB)),color=clrB,fontsize=18)
    ax[1].yaxis.set_major_formatter(mpt.ScalarFormatter())
    ax[1].set_yticks([0.1,0.3,1,3,10])
    ax[0].legend(fontsize=14,markerscale=1.5)
    ax[0].set_xlim(13,32)
    fig.savefig("comparisonPlot_udrop_sizesMagGalaxies_A370.png")


    fig,ax = mpl.subplots()
    for i in range(rePhys.size):
        match = (tableResultsBouwens["ID"]==tableResults["ID"][i])
        # print(tableResults["ID"][i],tableResultsBouwens["ID"][match])
        if (rB[match].size == 1):
            ax.plot(rePhys[i],rB[match],"ko")

    ax.plot([0.01,30],[0.01,30],"--",color="red")
    ax.loglog()
    ax.yaxis.set_major_formatter(mpt.ScalarFormatter())
    ax.xaxis.set_major_formatter(mpt.ScalarFormatter())
    ax.set_xticks([0.1,0.3,1,3,10])
    ax.set_yticks([0.1,0.3,1,3,10])
    ax.set_xlim(0.02,12)
    ax.set_ylim(0.02,12)
    ax.set_ylabel(r"$r_{e,c}\ [\mathrm{Bouwens}]$")
    ax.set_xlabel(r"$r_{e,c}\ [\mathrm{Bruno}]$")
    fig.savefig("comparisonPlot_udrop_sizesGalaxies_A370.png")

    mpl.show()
