from mod_imports import *
import gm20 as g


img=pyfits.getdata("acs1.fits")[130:210,120:200]
smap = gen_segmap_tresh(img,37,47,1.5)

##fig,(ax1,ax2)=subplots(1,2)
##ax1.imshow(img)
##ax2.imshow(smap)
##show()

imgcat='/Users/bribeiro/Documents/PhD/ImagingFields/sample_cosmos_zgt2_flags2349_22232429.txt'
imgdir='/Users/bribeiro/Documents/PhD/ImagingFields/CANDELS/COSMOS/Hband/GALFIT'

ID,RA,DEC,Z,ZFlag = np.loadtxt('%s'%(imgcat),unpack=True,usecols=[0,1,2,3,4],dtype={'names':('IDs','RAs','DECs','Zs','Zflags'),'formats':('i8','f4','f4','f4','i4')})


def gen_entropy_index(img,alpha):

    data = abs(img.copy())
    data=data[data>0]
    media=nannp.mean(data)
    Npop = np.size(data)

    if alpha==0:
        gei = np.sum(log(media/data))/Npop
    elif alpha==1:
        gei = np.sum((data/media)*log(data/media))/Npop
    else:
        gei = np.sum((data/media)**alpha-1)/(Npop*alpha*(alpha-1))
    return gei


##
##for alpha in [0,0.5,1,2,3]:
##    print gen_entropy_index(img*smap,alpha)
##
##print "G"
##print g.Gini(img,smap)



alphas=[0,0.5,1,2,3]

IDs=[]
for i in range(len(ID)):
    if os.path.exists("%s/%i/galaxy_stamp.fits"%(imgdir,ID[i])):
##        print "Found image for VUDS %i"%ID[i]
        IDs.append(ID[i])
    else:
        continue



ginis=np.zeros(len(IDs))
geis=np.zeros([len(IDs),len(alphas)])

i=0
for VID in IDs:
    img=pyfits.getdata("%s/%i/galaxy_stamp.fits"%(imgdir,VID))
    N,M=img.shape
    smap=gen_segmap_tresh(img,N/2,M/2,thresh=3)
    ginis[i]=g.Gini(img,smap)
    for j in range(len(alphas)):
        geis[i,j]=gen_entropy_index(img*smap,alphas[j])
    
    i+=1
    
for i in range(len(alphas)):
    plot(ginis,geis[:,i],'s',label=r'$\alpha=%.1f$'%alphas[i])
    
legend(loc='best')
show()
