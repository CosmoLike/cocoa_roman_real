#Make mask here
import numpy as np

def get_linear_gal_bias_bitmask(Ntomo_source, Ntomo_lens, ggl_exclude, Nbins,
    type_2pcf = "fourier"):
    ''' get bit-mask matrix for linear galaxy bias 
    '''
    if type_2pcf=="fourier":
        N2pcf_ss = int((Ntomo_source+1)*Ntomo_source/2)
    elif type_2pcf=="real":
        N2pcf_ss = int((Ntomo_source+1)*Ntomo_source)
    else:
        print(f'[{rank}/{size}] Invalid value {type_2pcf} for type_2pcf')
        exit(-1)
    N2pcf_gs = Ntomo_source*Ntomo_lens - len(ggl_exclude)
    N2pcf_gg = Ntomo_lens
    Ndata = (N2pcf_ss+N2pcf_gs+N2pcf_gg)*Nbins
    bitmask = np.zeros([Ntomo_lens, Ndata])
    # cosmic shear 2pcf, pass
    # galaxy-galaxy lensing
    ct = N2pcf_ss
    for i in range(Ntomo_lens):
        for j in range(Ntomo_source):
            skip_this_ggl = False
            for ggl_pair in ggl_exclude:
                if (i==ggl_pair[0]) and (j==ggl_pair[1]):
                    skip_this_ggl = True
                    break
            if not skip_this_ggl:
                bitmask[i][ct*Nbins:(ct+1)*Nbins] += 1
                ct += 1
    # galaxy clustering
    for i in range(Ntomo_lens):
        bitmask[i][ct*Nbins:(ct+1)*Nbins] += 2
        ct += 1
    assert ct == N2pcf_ss + N2pcf_gs + N2pcf_gg
    
    return bitmask
bitmask = get_linear_gal_bias_bitmask(Ntomo_source=8, Ntomo_lens=8, ggl_exclude=[[6,0],[7,0],[7,1]], Nbins=15, type_2pcf="real")

np.save('linear_galbias_mask.npy', bitmask)