import TMC
import CAS
import MID
import gm20
import ADD
import mod_imports as GEN #every module imported within mod_imports needs the GEN. prefix
import analysis as AN
"""
ALL IN ONE
29/08/2013  - XC,YC from galaxy brightest pixel
03/09/2013  - SExtractor segmentation map implemented in fucntion on mod_imports
05/09/2013  - galSVM case added for comparison between parameters
"""

import argparse

parser = argparse.ArgumentParser(description="Widget to inspect galaxies from a catalog")
group = parser.add_mutually_exclusive_group()
group.add_argument('-c','--case',metavar='C',type=str,default="all",choices=("tmc", "gm20", "cas", "mid", "test", "all","none"),help="available cases: tmc, gm20, cas, mid, test, all or none")
parser.add_argument('-f','--field',metavar='Field',type=str,default='cosmos',choices=('cosmos','ecdfs','vvds2h'),help="Field from which to select galaxy. Availble choices: cosmos, ecdfs or vvds2h.")
parser.add_argument('-t','--threshold',metavar='THRESH',type=float,default=5.0,help="Segmentation Map threshold value.")
parser.add_argument('-m','--maps',metavar='MAP',type=int,default=2,choices=(0,1,2),help="Choice of segmentation map. 0 - Simple Threshold\n1 - Petrosian Radius defined Threshold (Lotz et al. 2004)\n2 - SExtractor segmentation map.")
parser.add_argument('-d','--directory',metavar='dir',type=str,default='/Users/bribeiro/Documents/PhD/VUDS/sample_magI_175_25_z3_4_flags_3_4',help="Directory np.where all images are stored inside a folder with the field name.")
parser.add_argument('-ib','--insband',metavar='I,B',type=str,default='acs,I',help="Instrument,Band on which to perform calculations of morphological parameters.")
parser.add_argument('-ib2','--insband2',metavar='I2,B2',type=str,default='acs,f606w',help="Instrument,Band on which to perform calculations of morphological color dipsersion.")

args = parser.parse_args()


case=args.case   #available cases: tmc, gm20, cas, mid, test, all or none
THRESH=args.threshold
MAPS=args.maps
field=args.field
imgdir=args.directory
instrumento,banda=[s for s in args.insband.split(',')]
instrumento2,banda2=[s for s in args.insband2.split(',')]

##################################  GENERAL  ###################################
root=GEN.os.getcwd()

test_image="%s/acs2.fits"%root
test_image2="%s/disturbed_faint.fits"%root

################################################################################

imgcat='cesam_vudsdb_%s.txt'%field

ID,RA,DEC,Z = GEN.np.loadtxt('%s/%s'%(imgdir,imgcat),unpack=True,usecols=[0,3,4,5],dtype={'names':('IDs','RAs','DECs','Zs'),'formats':('i8','f4','f4','f4')})

###
### acs1.fits is the path="%s/%i"%(imgdir,ID[3])
### acs2.fits is the path="%s/%i"%(imgdir,ID[4])
###

def runit(GAL,case,instrument,band,segmap=1,thresh=5.0,graph=True):
    global IMG,SEGMAP,MAP_s,MAP_r,MAP_t,ORIGINAL,image1
    path="%s/%s/%i"%(imgdir,field,ID[GAL])
    ra = RA[GAL]
    dec = DEC[GAL]
    file_name = GEN.get_name_fits(path,instrument,band)
    image1=file_name

    try:
        ORIGINAL = GEN.pyfits.getdata(image1)
    except (ValueError,IOError) as err:
        print GEN.colorred.format(err.message)
        return GEN.np.zeros([1,1]),GEN.np.zeros([1,1]),{}
#    IMG=np.zeros([200,200])
#    IMG=random.random(200*200).reshape(200,200)
#    IMG[75:125,75:125]+=2
#    IMG[90:110,90:110]=5

    N,M=ORIGINAL.shape
    pixscale=10.0/N
       
    MAP_s = GEN.gen_segmap_sex(image1,thresh=thresh)
    x_field,y_field=GEN.get_center_coords(image1,ra,dec)
    sky1,sig1= GEN.background_estimate(ORIGINAL,MAP_s)

    try:
        ys,xs,es,thetas,obj_num = GEN.get_sex_pars(x_field,y_field)
        GEN.sp.call('rm %s/test.cat'%root,shell=True)
    except ValueError as err:
        print GEN.colorred.format("No detection from SExtractor!")
        return ORIGINAL,MAP_s,{}

    MAP_r=GEN.np.zeros(ORIGINAL.shape)
    MAP_t=GEN.np.zeros(ORIGINAL.shape)
    if segmap==1:
        MAP_r = GEN.gen_segmap_rp(ORIGINAL,xs,ys,1-es,thetas,thresh=thresh)
    elif segmap==0:
        MAP_t = GEN.gen_segmap_tresh(ORIGINAL,thresh,sig1,sky1)

    XC,YC,e,theta=xs[obj_num],ys[obj_num],es[obj_num],thetas[obj_num]

#    XC1,YC1 = GEN.get_center_coords(image1,RA[3],DEC[3])


### FOR MODEL
#    MAP_t = GEN.gen_segmap_tresh(IMG,thresh,0.5,0.0)
#    MAP_r = GEN.gen_segmap_rp(IMG,XC,YC,0.3,45.0,thresh=3.0)
###

=
### FOR ACS1 SOURCE
#    MAP_t = GEN.gen_segmap_tresh(IMG,thresh,0.003928,8.65e-6)
#    MAP_r = GEN.gen_segmap_rp(IMG,XC,YC,0.397,-16.1,thresh=5.0)
###

### FOR ACS2 SOURCE
#    MAP_t = GEN.gen_segmap_tresh(IMG,thresh,0.004397,2e-5)
#    MAP_r = GEN.gen_segmap_rp(IMG,XC,YC,0.157,-19.3,thresh=5.0)
###

    MAPS=[MAP_t,MAP_r,MAP_s]
    PARS=dict()

    SEGMAP=MAPS[segmap]


##    XC2,YC2 = MID.find_centroid(IMG,MAP_r)
##    XC3,YC3 = gm20.find_center_mtot(IMG,MAP_r,XC1,YC1)

    IMG,SEGMAP,NC=GEN.make_stamps(XC,YC,ORIGINAL,SEGMAP,fact=2,pixscale=pixscale)
    NXC,NYC = NC

##    if case == 'cas' or case == 'all':
##        SKYPATCH,OUTMAP = GEN.sky_patch(ORIGINAL,MAPS[segmap],IMG.shape)
    
    npix = SEGMAP[SEGMAP>0].shape[0]
    PARS[r'$N_{pix}$']=npix
    PARS[r'$b/a$']=1-e
    PARS[r'$\theta$']=theta
    PARS[r'$F$']=ADD.filamentarity(SEGMAP)
    PARS[r'$d_A$']=TMC.angular_distance(Z[GAL])*1000
    PARS[r'$pix_s$']=pixscale
##    GEN.imshow(sky_patch,cmap='bone');GEN.colorbar();GEN.show()

##    nside = 50
##    GEN.imshow(SEGMAP[int(XC)-nside:int(XC)+nside+1,int(YC)-nside:int(YC)+nside+1],cmap='hot',extent=(int(XC)-nside,int(XC)+nside,int(YC)-nside,int(YC)+nside))
##    GEN.plot(int(XC),int(YC),'x',color='red',ms=10,mew=2)
##    GEN.show()


##################################  TMC  ######################################

    if case == 'tmc' or case == 'all':
       

        #print TMC.light_potential(IMG,SEGMAP)
        #print TMC.light_potential(BLOB,BSEGMAP)
    
        T=TMC.Size(SEGMAP,pixscale,Z[GAL])
        Mu=TMC.Multiplicity(IMG,SEGMAP)
        try:
            image2=GEN.get_name_fits(path,instrumento2,banda2)
            ORIGINAL_2 = GEN.pyfits.getdata(image2)
            MAP_2s = GEN.gen_segmap_sex(image1,thresh=thresh)
            x_field,y_field=GEN.get_center_coords(image1,ra,dec)
            ys,xs,es,thetas,obj_num = GEN.get_sex_pars(x_field,y_field)
            GEN.sp.call('rm %s/test.cat'%root,shell=True)
            XC2,YC2=xs[obj_num],ys[obj_num]
            IMG2,MAP2,NC2=GEN.make_stamps(XC2,YC2,ORIGINAL_2,MAP_2s,pixscale=pixscale)
            sky2,sig2= GEN.background_estimate(ORIGINAL,MAP_2s)
            CD=TMC.color_dispersion(IMG,IMG2,SEGMAP,MAP2,sky1,sky2,NC,NC2)
#            CD2=TMC.color_dispersion(IMG2,IMG,MAP2,SEGMAP,sky2,sky1)
        except ValueError as err:
            if 'Empty' not in err.message:
                print  GEN.colorred.format("No detection from SExtractor!")
            CD=GEN.nan
#            CD2=GEN.nan

        print "T=%.2f"%T
        print "Psi=%.2f"%Mu
        print "xi=%.4f"%CD
#        print "xi_inv=%.4f"%CD2

        PARS[r'$T$']=T
        PARS[r'$\Psi$']=Mu
        PARS[r'$\xi$']=CD       


    
##################################  gm20  #####################################

    if case == 'gm20' or case == 'all':

        G=gm20.Gini(IMG,SEGMAP)
        M20=gm20.MomentLight20(IMG,SEGMAP,NXC,NYC)
        
        print "G=%.2f"%G
        print "M_20=%.2f"%M20

        PARS[r'$G$']=G
        PARS[r'$M_{20}$']=M20


##################################  CAS  ######################################

    if case == 'cas' or case == 'all':
        C_ell, RP_ell,R20,R80=CAS.CAS_C(ORIGINAL,YC,XC,1-e,-theta,dr0=0.05,rpstep=0.1) #Compute using elliptical apertures
        C_circ, RP_circ,R20,R80=CAS.CAS_C(ORIGINAL,YC,XC,1.0,0.0,dr0=0.05,rpstep=0.1)    #Compute using circular apertures
        sky_patch=GEN.sky_region(image1,IMG.shape,1.5,nmax=100)
        A = CAS.CAS_A([NXC,NYC],IMG,sky_patch)
        S, Simg=CAS.CAS_S(IMG,NXC,NYC,(1.5*RP_circ/5),sky_patch)
   
        print "C=%.2f"%C_ell
        print "A=%.2f"%A
        print "S=%.2f"%S

        PARS[r'$C_{circ}$']=C_circ
        PARS[r'$C_{ell}$']=C_ell
        PARS[r'$A$']=A
        PARS[r'$S$']=S
        PARS[r'$r_{p,c}$']=RP_circ
        PARS[r'$r_{p,e}$']=RP_ell
        
##################################  MID  ######################################

    if case == 'mid' or case == 'all':
        Rs,Qs=MID.multimode(IMG,SEGMAP)
        Mm,qmax=max(Rs),Qs[Rs==max(Rs)][0]
        IMAP,LM=MID.local_maxims(IMG,SEGMAP)
        I,Icen=MID.intensity(IMG,IMAP,SEGMAP,LM)
        D=MID.deviation(IMG,SEGMAP,Icen)

##        GEN.figure(10)
##        GEN.plot(Qs,Rs,'-')
##        GEN.show()
    #    Xc,Yc = MID.find_centroid(IMG,SEGMAP)
    #    X1,Y1 = Icen


        print "M=%.2f"%Mm
        print "I=%.2f"%I
        print "D=%.2f"%D

        PARS[r'$M$']=Mm
        PARS[r'$I$']=I
        PARS[r'$D$']=D


    #        print Icen


##################################  PLOTTING  #################################

    
    if graph:
        COLOR='spectral'
        imax,imin,jmax,jmin=GEN.find_ij(MAP_r)

        ax=GEN.make_subplot(2,2,width=10,height=10)
        C1=ax[0].imshow(ORIGINAL,cmap=COLOR,extent=(jmin,jmax,imin,imax))
        ax[0].plot([YC],[XC],'kx',markersize=10,markeredgewidth=2,label=r'$I_{max}$')
##        ax[0].plot([YC1],[XC1],'wx',markersize=10,markeredgewidth=2,label=r'$\mathrm{[RA,DEC]}\rightarrow [x_c,y_c]$')
##        ax[0].plot([YC2],[XC2],'x',color='gray',markersize=10,markeredgewidth=2,label=r'$\mathrm{Flux\ weighted}$')
##        ax[0].plot([YC3],[XC3],'x',color='maroon',markersize=10,markeredgewidth=2,label=r'$\min(M_{tot})$')
##        ax[0].legend(bbox_to_anchor = (1.80, 1.0),numpoints=1)

        C2=ax[1].imshow(MAP_t*ORIGINAL,cmap=COLOR,extent=(jmin,jmax,imin,imax))
        C3=ax[2].imshow(MAP_s*ORIGINAL,cmap=COLOR,extent=(jmin,jmax,imin,imax))
        C4=ax[3].imshow(MAP_r*ORIGINAL,cmap=COLOR,extent=(jmin,jmax,imin,imax))
        ax[1].text(jmin-2,imax+2,r'$I_{i,j} > \langle sky\rangle+%i \sigma_{sky}$'%thresh,color='white',np.size=15)
        ax[3].text(jmin-2,imax+2,r'$G(r_p/5) * I_{i,j} > %i\mu (r_p)$'%thresh,color='white',np.size=15)
        ax[2].text(jmin-2,imax+2,'SExtractor',color='white')
        GEN.show()

###############################   TESTING AREA  ###############################


    if case == 'test':

### CAS testing ###
##        C,RP=CAS_C(IMG,NXC,NYC)
##        gen_circle(ax[0],NXC,NYC,RP)
##        SEGMAP2 = gen_segmap_rp(IMG,NXC,NYC,RP,thresh=thresh)
##        C4=ax[5].imshow(SEGMAP2*IMG,cmap=COLOR)
##        print C
##        R=CAS.rotate_iraf(IMG,SEGMAP,NYC+1,NXC+1) ### run this command only from terminal
##        ax[1].imshow(R,cmap=COLOR)
##        R=R[:,1:]
##        CC=ax[2].imshow(abs(SEGMAP*IMG-R),cmap=COLOR)
##        GEN.colorbar(CC,ax=ax[2])
##        print np.sum(abs(SEGMAP*IMG-R))/np.sum(abs(SEGMAP*IMG))

### TMC testing ###   
##        DMAT,dists=distance_matrix(NXC,NYC,IMG)
##        CBLOB=rearrange_light_distance(IMG,SEGMAP)
##        X,Y,SBLOB=rearrange_light_spiral(IMG,SEGMAP)
##        C3 = ax[2].imshow(DMAT,cmap=COLOR)
##        C4 = ax[3].imshow(CBLOB,cmap=COLOR)
##        C5 = ax[4].imshow(SBLOB,cmap=COLOR)
##        BSEGMAP = np.zeros([np.size(CBLOB,0),np.size(CBLOB,1)])
##        BSEGMAP[CBLOB>0]=1.0
##        C6 = ax[5].imshow(BSEGMAP,cmap=COLOR)


### SKY PATCH TESTING S and A
        rp=CAS.petrosian_rad(ORIGINAL,XC,YC,1-e,theta)
        T=TMC.Size(SEGMAP,pixscale,Z)
        PARS[r'$r_p$']=rp
        PARS[r'$T$']=T
        #S,Simg= CAS.CAS_S(IMG,SEGMAP,XC,YC,(1.5*rp)/5)
        #print S,rp,(1.5*rp)/5.

    if case=='none':
        pass

    return IMG,SEGMAP,PARS

##############################################################################
##############################################################################
##############################################################################


if __name__=='__main__':
#    fig=GEN.figure(figsize=(12,10))
#    GEN.show()

    P=[]
    gal_test=311
    for i in range(len(ID)):
##    for i in range(gal_test,gal_test+1):
##    for i in [19,23,60,90,100]:
        print GEN.bold_bri_colorblu.format('\n%i:\tAssessing galaxy %i'%(i,ID[i]))
        IMG,MAP,PARS = runit(i,case,instrumento,banda,segmap=MAPS,thresh=THRESH,graph=False)
        P.append(PARS)

##        GEN.imshow(IMG,cmap='bone');ax=GEN.gca();GEN.draw_segmap_border(ax,MAP,color='red',distinct=False)
##        fig.canvas.draw_idle()
##        GEN.clf()



##Cmax=GEN.np.zeros((2,len(ID)))
##for i in range(len(ID)):
##    print GEN.bold_bri_colorblu.format('\n%i:\tAssessing galaxy %i'%(i,ID[i]))
##    Cmax[0,i]=ID[i]
##    IMG,MAP,PARS = runit(i,case,instrumento,banda,segmap=2,thresh=5.0,graph=False)
##    TMAP=MAP
##    try:
##        imax,imin,jmax,jmin=GEN.find_ij(TMAP)
##    except ValueError as err:
##        print err
##        continue
##    N,M=TMAP.shape
##    Cs=GEN.np.zeros((N,M,4))
##    for k in range(imin,imax):
##        print "%.2f%%"%((k-imin)/float(imax-imin)*100)
##        for j in range(jmin,jmax):
##            if TMAP[k,j]==1:
##                Cs[k,j,:]=CAS.CAS_C(IMG,k,j)
##            else:
##                continue
##    cpars=Cs[:,:,0]
##    cpars[GEN.np.isinf(cpars)==True]=0.0
##    Cmax[1,i]=GEN.np.amax(cpars)




if __name__=='__main__':
    NoK=max([len(p) for p in P])
    ALL = GEN.np.zeros([NoK,len(ID)])
    for i in range(len(ID)):
        if P[i].values()!=[]:
            ALL[:,i]=P[i].values()
        else:
            ALL[:,i]=GEN.nan


    AN.dump_txt(P,instrumento,banda,{'ID':ID,'RA':RA,'DEC':DEC,'Z':Z},fname='results_%s_%i'%(case,MAPS))


##    P=[]
##    for i in [0,1,2]:
##        IMG,MAP,PARS=runit(case,test_image,test_image2,ra,dec,segmap=i,graph=False)
##        P.append(PARS)
##    
##    X = range(len(PARS))
##    GEN.plot(X,P[0].values(),'o',markersize=15,mec='red',mfc='None',label='Threshold',mew=1.5)
##    GEN.plot(X,P[1].values(),'o',markersize=10,mec='blue',mfc='None',label='Petrosian',mew=1.5)
##    GEN.plot(X,P[2].values(),'o',markersize=5,mec='green',mfc='None',label='SExtractor',mew=1.5)
##    GEN.xticks(X,P[0].keys())
##    GEN.xlim(-1,len(X))
##    GEN.legend(loc='best',numpoints=1)
##    GEN.show()    


#test_azed('cosmos',139,10.0,nplots=25)


