from mod_imports import *


tabledir='/Users/bruno/Documents/PhD/Presentations/VUDS_milano_12_12_13/'

"""
03/10/2013  - added dump_txt to save the morphological information on an ascii table that can be read from topcat or python or else
"""

def dump_txt(table,instrument,band,catinfo,fname='results'):
    """ Function to dump the results from running ALL.py into an ascii table
    with all the morphological information.
    """
    
    f=open("%s_%s_%s.txt"%(fname,instrument,band),'w')
    f.write('# ')
    for n in catinfo.keys():
        f.write('%s\t'%n)
    nvals=array([len(p) for p in table])
    valid=where(nvals>0)[0]
    index=valid[0]
    for p in table[index].keys():
        f.write("%s\t"%p.replace('$',''))
    f.write("\n")
    for i in range(len(table)):
        for v in catinfo.values():
            f.write("%.3f\t"%v[i])
        if table[i].values()==[]:
            for k in range(max(nvals)):
                f.write('""\t')
        else:
            for val in table[i].values():
                f.write("%.3f\t"%val)
        f.write("\n")
    f.close()
    return 

def bin_results(table_name,nbins=4):
    global zbins
    """ Compresses the input table to a set of mean values in bins of redshift
    to inspect if there is any evolution of a given parameter with z.  It returns
    a table with all the mean values for the input parameters per redshift bin (first column)
    and the standard deviations of each parameter in that bin.
    """
    tab=genfromtxt(table_name,filling_values=nan)
    Z=tab[:,0]
    
    zbins=linspace(min(Z),max(Z)+1e-10,num=nbins+1)
    indexs=[]
    for i in range(nbins):
        testarray=(Z>=zbins[i])*(Z<zbins[i+1])
        indices = where(testarray==True)
        indexs.append(indices)

    zs=[]

    skip_cols=4
    medpars=zeros([nbins,tab.shape[1]-skip_cols])
    devpars=zeros([nbins,tab.shape[1]-skip_cols])
    for i in range(nbins):
        zs.append(mean(Z[indexs[i]]))
        for j in range(skip_cols,tab.shape[-1]):
            Sample=float64(tab[:,j][indexs[i]])
            CleanSample=Sample[isnan(Sample)==False]
            CleanSample=CleanSample[isinf(CleanSample)==False]
            medpars[i,j-skip_cols]=mean(CleanSample)
            devpars[i,j-skip_cols]=std(CleanSample)
   
    return zs,medpars,devpars,tab

def morph_fraction(table):
    
    zed = loadtxt(table,unpack=True,usecols=[3])
    gtype = loadtxt(table,unpack=True,usecols=[-1])
    compact=zed[gtype==0]
    cometary=zed[gtype==1]
    pair=zed[gtype==2]
    elongated=zed[gtype==3]
    multicore=zed[gtype==4]
    diffuse=zed[gtype==5]
    no_detection=zed[gtype==-9]

    labels=['Compact','Cometary','Pair','Elongated','Multicore','Diffuse']

    ntypes=6
    
    nbins=8
    zbins=linspace(min(zed),max(zed)+1e-10,num=nbins+1)
    zs=[]
    indexs=[]
    for i in range(nbins):
        testarray=(zed>=zbins[i])*(zed<zbins[i+1])
        indices = where(testarray==True)
        print zbins[i],zbins[i+1]
        print i,len(indices[0])
        indexs.append(indices)
        zs.append(mean(zed[indices]))

    a=zeros([nbins,ntypes])

    for k in range(ntypes):
        for j in range(nbins):
            a[j,k]=len(zed[gtype[indexs[j]]==k])

    ntot=zeros(nbins)
    for l in range(nbins):
        ntot[l]=sum(a[l,:])

    
    for i in range(1):
        plot(zs,a[:,i]/ntot,'o-',label=labels[i])

##    ylim(0,0.6)
    xlabel(r'$z$')
    ylabel(r'$N_f$')
    legend(loc='best',numpoints=1)
    ylim(0,0.4)
    xlim(2,6.5)
    savefig('%s/number_fraction.png'%tabledir,format='png')
    fill_between(linspace(4.2,8),1.0,0.0,color='gray',alpha=0.1)
    ylim(0,0.4)
    xlim(2,6.5)
    savefig('%s/number_fraction_shaded.png'%tabledir,format='png')
    show()

 
def test_binning_plots():
    zbins,MP,DP,tab=bin_results('results_acs_I.txt')
    for k in range(1,15):
        plot(MP[:,0],MP[:,k],'o-')
        plot(tab[:,3],tab[:,3+k],'o',markersize=6,alpha=0.2)
        errorbar(MP[:,0],MP[:,k],yerr=DP[:,k],fmt='o',markersize=10)
        vlines(zbins,-100,100,linestyle=':')
        ylim(0.9*min(tab[:,3+k]),1.1*max(tab[:,3+k]))
        show()

##test_binning_plots()
##if __name__=='__main__':
##    import os,sys
##    imgdir="%s/"%(os.getcwd())
##    imgcat="%s/results_galfit_stamp.txt"%imgdir
##
##    parcols=[0,1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
##    table = np.genfromtxt(imgcat,usecols=parcols)
##
##
##    f=open(imgcat,'r')
##    hdr=f.readline()
##    keywords=hdr.split()[1:]
##    f.close()
##
##    i=1
##    labels=dict()
##    k=0
##    for j in parcols:
##        labels[k]=r'$%s$'%keywords[j]
##        k+=1
##
##
##    ntypes = int(max(table[:,16])+1)
##
##    Test=zeros([len(labels),ntypes])
##    for i in range(len(labels)):
##        for j in range(ntypes):
##            subset = table[:,i][table[:,16]==j]
##            Test[i,j]=mean(subset)
##
##
##    ##for i in range(len(labels)):
##    ##    plot(range(4),Test[i,:],'o',markersize=10)
##    ##    xticks(range(4),['spiral','sersic','irregular','multi'])
##    ##    xlabel('Galaxy Type')
##    ##    ylabel(labels[i])
##    ##    xlim(-1,4)
##    ##    show()
##    ##    clf()
##
##    subset=[3,4,6,9,15]
##
##    new_type=[]
##    for j in range(100):
##        W=[]
##        gal_test=table[j,:]
##        for i in range(4):
##            diff = abs(gal_test[subset]-Test[:,i][subset])
##            W.append(sum(diff))
##        try:
##            new_type.append(where(W==min(W))[0][0])
##        except IndexError as err:
##            new_type.append(nan)
##            print err
##
##    As=[]
##    for i in range(100):
##        As.append([new_type[i],table[i,16]])
##    sizes=[]
##    for a in As:
##        sizes.append(len(where(array(As)==a)[0]))            
##
##
##    scatter(table[:,16],new_type,s=sizes)
##    show()
        
table='%s/vuds_milano_table.txt'%tabledir
morph_fraction(table)

def plane_plot(labels,col1,col2,parameters,colors,log=False):
    for label in labels[::-1]:
        table=eval(label.lower())
        if label=='Compact':
            plot(table[:,col1],table[:,col2],'o',markersize=7,color=colors[label],alpha=0.75,label=label)
        else:
            plot(table[:,col1],table[:,col2],'o',markersize=7,color=colors[label],alpha=0.50,label=label)
        xlabel(parameters[col1])
        ylabel(parameters[col2])
    if log:
        loglog()
        xlim(1e-1,1e2)
    legend(loc='best',numpoints=1)
    savefig("%s/%s_%s.png"%(tabledir,parameters[col1].translate(None,'$/{}_'),parameters[col2].translate(None,'$/{}_')),format='png')
    clf()
    return None

def plane_plot_mean(labels,col1,col2,parameters,colors,log=False):
    for label in labels:
        table=eval(label.lower())
        plot(table[:,col1],table[:,col2],'o',markersize=7,color=colors[label],alpha=0.20)
        plot([mean(table[:,col1][isinf(table[:,col1])==False])],[mean(table[:,col2][isinf(table[:,col2])==False])],'o',markersize=10,color=colors[label],label=label)
        xlabel(pars[col1])
        ylabel(pars[col2])
    if log:
        loglog()
        xlim(1e-1,1e2)
    legend(loc='best',numpoints=1)
    savefig("%s/%s_%s_mean.png"%(tabledir,parameters[col1].translate(None,'$/\{}_'),parameters[col2].translate(None,'$/{}_')),format='png')
    clf()
    return None

if __name__=='__main__':
    results_table='%s/vuds_milano_table.txt'%tabledir
##    results_table_2='results_all_wfc3_f160w.txt'
##    eye_class='eye_class_acs_I.txt'

    f=open(results_table,'r')
    header=f.readline()
    keywords=header.split()[1:]
    f.close()

    k=0
    pars=dict()
    for j in range(len(keywords)):
        pars[k]=r'$%s$'%keywords[j]
        print j,keywords[j]
        k+=1

    table = genfromtxt(results_table)
    gtype = loadtxt(results_table,unpack=True,usecols=[-1])
    compact=table[gtype==0]
    cometary=table[gtype==1]
    pair=table[gtype==2]
    elongated=table[gtype==3]
    multicore=table[gtype==4]
    diffuse=table[gtype==5]
    no_detection=table[gtype==-9]


##    table2 = genfromtxt(results_table_2)
##    compact2=table2[gtype==0]
##    cometary2=table2[gtype==1]
##    pair2=table2[gtype==2]
##    elongated2=table2[gtype==3]
##    multicore2=table2[gtype==4]
##    diffuse2=table2[gtype==5]
##    no_detection2=table2[gtype==-9]

    labels=['Compact','Cometary','Pair','Elongated','Multicore','Diffuse']
    colors={'Compact':'Blue','Cometary':'DarkViolet','Pair':'Red',\
            'Elongated':'Gold','Multicore':'Green','Diffuse':'Gray'}



#################################################################################
##    plane_plot(labels[:-1],4,8,pars,colors)
##    plane_plot_mean(labels[:-1],4,8,pars,colors)
###############################################################################
##    plane_plot(labels[:-1],10,11,pars,colors)
##    plane_plot_mean(labels[:-1],10,11,pars,colors)
#################################################################################
##    plane_plot(labels[:-1],17,18,pars,colors)
##    plane_plot_mean(labels[:-1],17,18,pars,colors)
#################################################################################
##    plane_plot(labels[:-1],12,14,pars,colors,log=True)
##    plane_plot_mean(labels[:-1],12,14,pars,colors,log=True)
###################################################################################    
##    plane_plot(labels[:-1],19,16,pars,colors,log=True)
##    plane_plot_mean(labels[:-1],19,16,pars,colors,log=True)
###################################################################################    
    plane_plot(labels[:-1],14,16,pars,colors,log=True)
    plane_plot_mean(labels[:-1],14,16,pars,colors,log=True)
###############################################################################    


##    n=7;plot(table[:,n],table2[:,n],'o');print n;plot(table2[:,n],table2[:,n],'k--');xlabel(pars[n]+' I-band');ylabel(pars[n]+' H-band');show()


