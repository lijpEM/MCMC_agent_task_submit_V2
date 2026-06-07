'''
    File name: MT.py
    Authors: Hoël Seillé / Gerhard Visser
    Date created: 01/06/2020
    Date last modified: 14/04/2021
    Python Version: 3.6
'''
__author__ = "Hoël Seillé / Gerhard Visser"
__copyright__ = "Copyright 2020, CSIRO"
__credits__ = ["Hoël Seillé / Gerhard Visser"]
__license__ = "GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Hoël Seillé"
__email__ = "hoel.seille@csiro.au"
__status__ = "Beta"



import numpy as np
import pandas as pd
import re as _re
from types import SimpleNamespace


def readEDI(file_name):
    
    """
    function that reads EDI files
    Z in linear scale; format: ohm
    
    Input:
    - complete path to the EDI file
    
    Output:
    - data (EDI pandas DF format)
    - site_id (string)
    - coord (panda DF of 7 values (['Lat_deg']['Lat_min']['Lat_sec']['Long_deg']['Long_min']['Long_sec']['Elev'])
             
    Pandas DF format:        
    # ['FREQ',
    # 'ZXXR','ZXXI','ZXX.VAR',
    # 'ZXYR','ZXYI','ZXY.VAR',
    # 'ZYXR','ZYXI','ZYX.VAR',
    # 'ZYYR','ZYYI','ZYY.VAR',
    # 'TXR.EXP','TXI.EXP','TXVAR.EXP',
    # 'TYR.EXP','TYI.EXP','TYVAR.EXP']
    
    """
    
    import numpy as np
    import re
    import pandas as pd

    coord = pd.DataFrame({'Lat_deg':0,'Lat_min':0,'Lat_sec':0,'Long_deg':0,'Long_min':0,'Long_sec':0,'Elev':0},index=[0])

    with open(file_name, 'r') as f:
        data = f.readlines()
        for i,line in enumerate(data):
            line = data[i]
            words = line.split()
            
            #READ SITE NAME
            if any("DATAID" in s for s in words):
                words = ''.join(words)
                m = re.search(r'DATAID\s*=\s*"?([^"\s]+)"?', words)
                if m:
                    site_id = m.group(1)
                else:
                    site_id = 'UNKNOWN'

            #READ NUMBER OF FREQUENCIES
            if any("NFREQ" in s for s in words):
                words = ''.join(words)
                nfreq_str = (re.findall(r'\d+', words))
                nfreq = int(nfreq_str[0])

             #READ LATITUDE
            if any("REFLAT" in s for s in words):
                words = ''.join(words)
                reflat_str = (re.findall(r'\-?\d+', words))
                coord['Lat_deg'] = str(reflat_str[0])
                coord['Lat_min'] = str(reflat_str[1])
                coord['Lat_sec'] = str('.'.join(reflat_str[2:]))				

            #READ LONGITUDE  (accept both REFLON and REFLONG)
            if any(("REFLON" in s) for s in words):
                words = ''.join(words)
                reflong_str = (re.findall(r'\-?\d+', words))
                coord['Long_deg'] = str(reflong_str[0])
                coord['Long_min'] = str(reflong_str[1])
                coord['Long_sec'] = str('.'.join(reflong_str[2:]))	

            #READ ELEVATION  (accept REFELEV or ELEV)
            if any(("REFELEV" in s) or ("ELEV" in s) for s in words):
                words = ''.join(words)
                refelev_str = (re.findall(r'\-?\d+', words))
                coord['Elev'] = str('.'.join(refelev_str[0:]))				

    #READ MT DATA
    param = ['FREQ',
             'ZXXR','ZXXI','ZXX.VAR',
             'ZXYR','ZXYI','ZXY.VAR',
             'ZYXR','ZYXI','ZYX.VAR',
             'ZYYR','ZYYI','ZYY.VAR',
             'TXR.EXP','TXI.EXP','TXVAR.EXP',
             'TYR.EXP','TYI.EXP','TYVAR.EXP',]
    
    edi_data = np.empty((nfreq, len(param)))
    
    with open(file_name, 'r') as f:
        data = f.readlines()
        for i,line in enumerate(data):
            line = data[i]
            words = line.split()
            
            for col, data_type in enumerate(param):
                aa=[]            
                if ('>%s' %data_type) in words:
                    for k in range (1,1000):                   
                        if any(">" in s for s in data[i+k].split()):
                                break
                        else:
                            a = data[i+k].split()
                            aa += a
                    edi_data[:,col] = aa
    
    
    # write to Pandas format
    edi_pd = pd.DataFrame(edi_data)
    edi_pd.columns = param

    return (edi_pd, site_id, coord)





def phaseTensor(Z):
    
    # MAKE THAT FUNCTION MORE GENERAL AND SHORTER
    """
    Function that calculates phase tensor and parameters
    Following Caldwell et al. (2004 GJI) / Bibby et al. (2005 GJI)
    
    Input:
    - data (pandas DF format for Z)
    
    Output:
    - phase tensor matrix
    - phase tensor parameters (phmax,phmin,alpha,beta,ellip,azimuth)
    
    """   
    def names(name):
        return  Z[name].values

    if isinstance(Z, pd.DataFrame):
        freq = Z['FREQ'].values
        nf = len(freq)
        X11 = Z['ZXXR'].values
        Y11 = Z['ZXXI'].values
        X12 = Z['ZXYR'].values
        Y12 = Z['ZXYI'].values
        X21 = Z['ZYXR'].values
        Y21 = Z['ZYXI'].values
        X22 = Z['ZYYR'].values
        Y22 = Z['ZYYI'].values
        
    else:
        freq = Z[:,0]
        nf = len(freq)
        
        X11 = Z[:,1]
        Y11 = Z[:,2]
        X12 = Z[:,4]
        Y12 = Z[:,5]
        X21 = Z[:,7]
        Y21 = Z[:,8]
        X22 = Z[:,10]
        Y22 = Z[:,11]


    #loop over each frequency 
    #for f in range(0, nf):
    det_x = (X11*X22 - X21*X12)
    ph11 = (X22*Y11 - X12*Y21)/det_x
    ph12 = (X22*Y12 - X12*Y22)/det_x 
    ph21 = (X11*Y21 - X21*Y11)/det_x
    ph22 = (X11*Y22 - X21*Y12)/det_x
    
    # Bibby et al. 2005 GJI
    pi1 = 0.5 * np.sqrt((ph11-ph22)**2 + (ph12+ph21)**2)
    pi2 = 0.5 * np.sqrt((ph11+ph22)**2 + (ph12-ph21)**2)
    
    phmax = np.degrees(np.arctan(pi2+pi1))
    phmin = np.degrees(np.arctan(pi2-pi1))
    
    ellip = (phmax-phmin) / (phmax+phmin)
    alpha = np.degrees(0.5 * np.arctan2((ph12+ph21) , (ph11-ph22)))
    beta  = np.degrees(0.5 * np.arctan2((ph12-ph21) , (ph11+ph22)))
    azimuth = alpha - beta
    
    ph_tens = np.zeros((nf, 5))
    ph_params = np.zeros((nf, 7))	
    ph_tens[:,0] = ph_params[:,0] = freq
    
    ph_tens[:,1] = ph11
    ph_tens[:,2] = ph12
    ph_tens[:,3] = ph21
    ph_tens[:,4] = ph22
    
    ph_params[:,1] = phmax
    ph_params[:,2] = phmin
    ph_params[:,3] = alpha
    ph_params[:,4] = beta
    ph_params[:,5] = ellip
    ph_params[:,6] = azimuth
    
    return(ph_tens, ph_params)
    



def phaseTensorErr(Z,MC_realizations = 10000):
    
    """
    Function that calculates phase tensor parameters errors
    It uses a Monte Carlo simulaiton 
    Following Booker (2014 Survey in Geophysics)
    
    
    Input:
    - data (pandas DF format for Z)
    - optional: number of MC realizations to perform (default =10000, should be enough)
    
    Output:
    - phase tensor matrix std. dev.
    - phase tensor parameters std. dev. (phmax,phmin,alpha,beta,ellip,azimuth)
    
    """   
        
    from scipy.stats import norm
    
    freq = Z['FREQ'].values
    nf = len(freq)
    
    Z_MC = np.zeros((MC_realizations,12, len(freq)))
    ph_tens_MC = np.zeros((MC_realizations,4,len(freq)))
    ph_params_MC  = np.zeros((MC_realizations,6,len(freq)))
    
    ph_tens_std = np.zeros((len(freq),4))
    ph_params_std  = np.zeros((len(freq),6))
    
    #For each frequency:
    for f in range(nf):
    
        Z_MC[:,0,f] = freq[f]
        
        Z_MC[:,1,f] = np.random.normal(Z['ZXXR'][f], (Z['ZXX.VAR'][f])**0.5, MC_realizations)
        Z_MC[:,2,f] = np.random.normal(Z['ZXXI'][f], (Z['ZXX.VAR'][f])**0.5, MC_realizations)
        Z_MC[:,4,f] = np.random.normal(Z['ZXYR'][f], (Z['ZXY.VAR'][f])**0.5, MC_realizations)
        Z_MC[:,5,f] = np.random.normal(Z['ZXYI'][f], (Z['ZXY.VAR'][f])**0.5, MC_realizations)
        Z_MC[:,7,f] = np.random.normal(Z['ZYXR'][f], (Z['ZYX.VAR'][f])**0.5, MC_realizations)
        Z_MC[:,8,f] = np.random.normal(Z['ZYXI'][f], (Z['ZYX.VAR'][f])**0.5, MC_realizations)
        Z_MC[:,10,f] = np.random.normal(Z['ZYYR'][f], (Z['ZYY.VAR'][f])**0.5, MC_realizations)
        Z_MC[:,11,f] = np.random.normal(Z['ZYYI'][f], (Z['ZYY.VAR'][f])**0.5, MC_realizations)

        ph_tens, ph_params = phaseTensor(Z_MC[:,:,f])

        ph_tens_MC[:,:,f] = ph_tens[:,1:]
        ph_params_MC[:,:,f] = ph_params[:,1:]

        mean, ph_tens_std[f,0] = norm.fit(ph_tens_MC[:,0,f])
        mean, ph_tens_std[f,1] = norm.fit(ph_tens_MC[:,1,f])
        mean, ph_tens_std[f,2] = norm.fit(ph_tens_MC[:,2,f])
        mean, ph_tens_std[f,3] = norm.fit(ph_tens_MC[:,3,f])
        
        mean, ph_params_std[f,0] = norm.fit(ph_params_MC[:,0,f])
        mean, ph_params_std[f,1] = norm.fit(ph_params_MC[:,1,f])
        mean, ph_params_std[f,2] = norm.fit(ph_params_MC[:,2,f])
        mean, ph_params_std[f,3] = norm.fit(ph_params_MC[:,3,f])
        mean, ph_params_std[f,4] = norm.fit(ph_params_MC[:,4,f])
        mean, ph_params_std[f,5] = norm.fit(ph_params_MC[:,5,f])

    return(ph_tens_std, ph_params_std)



def medianFilter(freq, param, freq_sp):

    """
    Function that filters out outliers using a median filter ()
    """   
    
    param_filt = np.zeros(len(freq))
    # freq_sp = 0.5 # interval value (log10) where the points can be considered for calculating the median
    for f1, f1_val in enumerate(freq):
        paramMed = []
        if f1 == 0:         # consider 0 beyond the beginning extremity (lim->0 = 0)
            paramMed = [0]
        for f2, f2_val in enumerate(freq):
            if np.log10(f1_val)- freq_sp/2 < np.log10(f2_val) < np.log10(f1_val)+ freq_sp/2:
                paramMed.append(param[f2])
        param_filt[f1] = np.median(np.array(paramMed))
    return param_filt




def z2rhophy(freq,Zr,Zi,dZ=0):
    
    import numpy as np

    FREQ = freq
    ZR = Zr
    ZI = Zi
    Z_VAR = dZ
    
    # calcul of apparent resistivity and phases
    rho = ((ZR**2+ZI**2)*0.2/(FREQ))
    phy = np.degrees(np.arctan2(ZI,ZR))
    
    # calcul of errors
    drho = ((ZR**2+ZI**2)**0.5)*(np.sqrt(Z_VAR))*0.4/(FREQ) 
    arcsin_arg = np.sqrt(Z_VAR)/((ZR**2+ZI**2)**0.5)
    arcsin_arg = np.clip(arcsin_arg, -1.0, 1.0)
    dphy= np.degrees(np.arcsin(arcsin_arg))

    return (rho, phy, drho, dphy)




def Zinv(Z):
    
    """
    Compute the invariant of Z: mean of the TE and TM modes
    """
    
    ZinvR = (Z['ZXYR'].values - Z['ZYXR'].values)/2
    ZinvI = (Z['ZXYI'].values - Z['ZYXI'].values)/2
    
    return ZinvR ,ZinvI 




def Zdet(Z):
    
    """
    Compute the determinant of Z
    
    Zdet = Zxx*Zyy - Zxy*Zyx = Zdet_1 - Zdet_2
    """
    nF = len(Z)
    # Calculate determinant 
    mat = np.zeros((2,2,nF), complex)
    mat[0,0,:] = Z['ZXXR'].values + (Z['ZXXI'].values * 1j)
    mat[0,1,:] = Z['ZXYR'].values + (Z['ZXYI'].values * 1j)
    mat[1,0,:] = Z['ZYXR'].values + (Z['ZYXI'].values * 1j)
    mat[1,1,:] = Z['ZYYR'].values + (Z['ZYYI'].values * 1j)
    
    ZdetR = np.zeros((nF))
    ZdetI = np.zeros((nF))
    sd = np.zeros((nF))
    lnSd = np.zeros((nF))
    
    for freq_det in range(len(Z)):
        det = np.linalg.det(mat[:,:,freq_det])**0.5
        ZdetR[freq_det] = det.real
        ZdetI[freq_det] = det.imag
    
    
    # Calculate determinant std dev.
    # (log transform of) x + dx --> log(x) + (log(1+dx/x) - log(1-dx/x))/(2*sqrt(2))
    cent = (ZdetR**2 + ZdetI**2)**0.5
    sd = (Z['ZXX.VAR'] + Z['ZXY.VAR'] + Z['ZYX.VAR'] + Z['ZYY.VAR'])**0.5
    lnSd = (np.log(1+sd/cent) - np.log(1-sd/cent))/(2*np.sqrt(2))
    
    return ZdetR ,ZdetI, sd, lnSd



def keepMax(x):
    # For the attributes we use the highest value encountered 
    # as a minimum value (see Seille & Visser GJI 2020)
    for i in range(1,len(x)):
        if abs(x[i])>abs(x[i-1]):
            x[i] = abs(x[i])
        else:
            x[i] = abs(x[i-1])
    return x




def getData(edi_file_path, medfiltsp, StSh=False):
    import warnings
    warnings.warn(
        "MT.getData() is DEPRECATED and contains a units bug that makes "
        "rho_a ~633,000x too small. Use MT.read_edi() + MT.edi_to_S_MTdat() "
        "instead, as demonstrated in Step1_initialize.py.",
        FutureWarning, stacklevel=2)
    
    # read edi file
    Z_orig, site_id, coord = readEDI(edi_file_path)
    
    # Conversion to appropriate units: we use ohms
    #   1 ohm = 10000(4*pi) [mV/km/nT]
    C = 10000/(4*np.pi)
    Z = Z_orig.copy(deep=True)
    Z.loc[:,['ZXXR','ZXXI','ZXYR','ZXYI','ZYXR','ZYXI','ZYYR','ZYYI']] = Z_orig.loc[:,['ZXXR','ZXXI','ZXYR','ZXYI','ZYXR','ZYXI','ZYYR','ZYYI']] / C
    Z.loc[:,['ZXX.VAR','ZXY.VAR','ZYX.VAR','ZYY.VAR']] = Z_orig.loc[:,['ZXX.VAR','ZXY.VAR','ZYX.VAR','ZYY.VAR']] / (C**2)

    freq = Z['FREQ']
    
    # calculate phase tensor parameters            
    ph_tens, ph_params = phaseTensor(Z)
    ph_tensERR, ph_paramsERR = phaseTensorErr(Z)
    beta = ph_params[:,4]
    ellip = ph_params[:,5]
    beta_err = ph_paramsERR[:,3]
    ellip_err = ph_paramsERR[:,4]

    # Calculate determinant of Z and error 
    ZdetR ,ZdetI, ZdetSd, ZdetLnSd = Zdet(Z)
    
    # Calculate differences between Zxy and Zyx
    difPol = abs(np.log(abs(Z['ZXYR']))-np.log(abs(Z['ZYXR']))) + abs(np.log(abs(Z['ZXYI']))-np.log(abs(Z['ZYXI'])))

    #Filter out outliers using a median filter
    ellip_filt = medianFilter(freq, ellip, medfiltsp)
    beta_filt = medianFilter(freq, beta, medfiltsp)
    difPol_filt = medianFilter(freq, difPol, medfiltsp)

    # For the attributes we use the highest value encountered as a minimum value (see paper)
    ellipM = keepMax(ellip_filt)
    betaM = keepMax(beta_filt)
    difPolM = keepMax(difPol_filt)

    # Create dataframes for inversion (data_inv) and general (data_Z)
    df_inv = pd.DataFrame(columns=['freq','ZdetRLn','ZdetILn','ZdetLnSd','elipM','betaM','difPolM'])
    df_inv['freq'] = freq
    df_inv['ZdetRLn'] = np.log(ZdetR)
    df_inv['ZdetILn'] = np.log(ZdetI)
    df_inv['ZdetLnSd'] = ZdetLnSd
    df_inv['elipM'] = ellipM
    df_inv['betaM'] = betaM
    if StSh:
        df_inv['difPolM'] = difPolM 
    else:
        df_inv['difPolM'] = difPolM * 0.3
    
    dat = [Z_orig, ph_params, ph_paramsERR]
    
    return site_id, df_inv, dat 











def fwd1D(model, f, layers = 'depth'):

    
    """
    Compute the MT response of a 1D conductivity model
    
    Input:
    - model (nLayers x 2 array) 
        --> resistivity ohm.m
        --> depth in meters (last depth assumed to be the bottom of the model)
    - frequencies array at which the solution is computed (in Hz)
    
    Output:
    - impedance Z
    - apparent resistivity
    - phase
    
    """
    
    import numpy as np
    import math
   
    mu = 4*math.pi*1E-7; #Magnetic Permeability (H/m)
    
    res = model[:,0] 
    
    if layers == 'depth':
        depth = model[:,1]  #depth
        # Calculate thickness of each layers
        th = np.zeros((len(depth)))
        th[0] = depth[0]
        th[-1] = depth[-1]
        for i in range(1,len(depth[:-1])):
            th[i] = depth[i]-depth[i-1]
        #th[-2] =  th[-1] / 2
    else:
        th = model[:,0]  #thicknesses

    nlayers = len(res)
    
    nf=len(f)
    
    ares=[]
    ares = np.array(ares, dtype = np.float32)
    phy=[]
    phy = np.array(phy, dtype = np.float32)
    
    #Define arrays
    w = 2*np.pi*f
    
    Cm = np.empty(nlayers, complex)
    Z = np.empty(nf, complex)
    
    #Récurrence de Wait(1954)
    
    #Loop over each frequency
    for i in range(0, nf):    
        #1  Calculate bottom half space impedance
        gm = np.sqrt((w[i] * mu * (1.0/res[-1]))*1j);
        Cm[-1] = 1 / gm
        
        for k in range(nlayers-2, -1, -1):
            #2  Calculate impedance of layer k
            gm = np.sqrt((w[i] * mu * (1.0/res[k]))*1j);
            rm = (1 - gm*Cm[k+1]) / (1 + gm*Cm[k+1])
            Cm[k] = (1 -rm*np.exp(-2*gm*th[k])) / (gm*(1 +rm*np.exp(-2*gm*th[k])))
        Z[i] = Cm[0] *1j * w[i] * mu     #Cm[0] last Cm to be calculated: the one in surface
        # Units of Z in international standards units E/B(V/m/A/H)(EDI) from E/H(mV/km/nT) == factor mu
    
        # Step 3. Compute apparent resistivity and phase
        apparentResistivity = (abs(Z[i]) * abs(Z[i]))/(mu*w[i])
        phase = math.atan2(Z[i].imag, Z[i].real)
    
        ares=np.append(ares, apparentResistivity)
        phy=np.append(phy,phase)
    phy_deg=phy*180/math.pi #from radian to degrees
    
    return f, ares, phy_deg, Z




def niblettBostick_depthTransform(rho, phy, T):
    
    """
    "Immediate transformation of apparent resistivity and phase data and 
    presentation of an approximate resistivity and depth data. Specifically, 
    this transformation is based on the simple asymptotic expressions 
    introduced by Bostick (1977)"   
    
    --> GOLDBERG, S. and ROTSTEIN, Y. 1982, A Simple Form of Presentation of 
    Magnetotelluric Data Using the Bostick Transform,
    Geophysical Prospecting 30,211-216.

    Input:
    - apparent resistivity (Ohm meters)
    - phase angle (degrees)
    - period (seconds)
    
    Output:
    - resistivity estimate (Ohm meters)
    - depth (meters) 

    """
    
    mu0 = 4*np.pi*10**-7

    rho_nb = rho * (np.pi/(2*np.deg2rad(phy%90)) - 1)
    depth_nb = np.sqrt(rho*T/(2*np.pi*mu0))
    
    return rho_nb, depth_nb



def compute_x2(residual,std):
    if type(residual) == np.float64:
        x2 = (((residual/std)*(residual/std))) 
    else:
        x2 = (((residual/std)@(residual/std).T)) 
    return x2

def rms(obs_dat, resp):
    residuals = np.r_[(obs_dat[:,1] - resp[:,0]), (obs_dat[:,2] - resp[:,1])]
    std = np.r_[obs_dat[:,3],obs_dat[:,3]]
    x2 = ((residuals/std)@(residuals/std).T) 
    rms = (np.sqrt(x2 / len(residuals)))

    return rms


def read_edi(filename, water_depth=None, units='field', te_mode='xy',
             app_res_err_floor=0.05, phase_err_floor=2.86,
             unwrap_tm_phase=True, verbose=True):
    """
    Parse an SEG EDI file and return an MTdata dict compatible with the
    SERPENT Bayesian MT inversion.

    Returns a dict with keys:
      Freqs, FreqID, DatID, Data, DataErr, WaterDepth.

    DatID codes:
       104 = TE phase (deg)
       123 = TE log10(rho_a)  [Ohm-m]
       106 = TM phase (deg)
       125 = TM log10(rho_a)  [Ohm-m]
    """
    with open(filename, 'r') as fh:
        txt = fh.read()

    m = _re.search(r'EMPTY\s*=\s*([\-\+\dEe\.]+)', txt)
    empty_val = float(m.group(1)) if m else 1e32

    elev_val = None
    for pat in (r'REFELEV\s*=\s*(-?\d+\.?\d*)',
                r'(?<!REF)ELEV\s*=\s*(-?\d+\.?\d*)',
                r'ELEVATION\s*=\s*(-?\d+\.?\d*)'):
        m = _re.search(pat, txt, flags=_re.IGNORECASE)
        if m:
            elev_val = float(m.group(1))
            break
    if water_depth is None:
        water_depth = abs(elev_val) if elev_val is not None else 0.0

    def _block(name):
        pat = r'>\s*' + _re.escape(name) + r'[^\n]*\n([\s\S]*?)(?=\n\s*>)'
        mm = _re.search(pat, txt)
        if not mm:
            return np.array([])
        return np.fromstring(mm.group(1), sep=' ')

    freqs  = _block('FREQ')
    zxyR   = _block('ZXYR');   zxyI = _block('ZXYI')
    zyxR   = _block('ZYXR');   zyxI = _block('ZYXI')
    zxyVar = _block('ZXY.VAR')
    zyxVar = _block('ZYX.VAR')

    if freqs.size == 0:
        raise ValueError(f"read_edi: no >FREQ block in {filename}")

    def _sanitize(v):
        v = v.astype(float).copy()
        v[np.abs(v - empty_val) < 1e-3 * abs(empty_val)] = np.nan
        return v

    zxyR, zxyI = _sanitize(zxyR), _sanitize(zxyI)
    zyxR, zyxI = _sanitize(zyxR), _sanitize(zyxI)
    zxyVar = _sanitize(zxyVar); zxyVar[np.isnan(zxyVar)] = 0.0
    zyxVar = _sanitize(zyxVar); zyxVar[np.isnan(zyxVar)] = 0.0

    Zxy = zxyR + 1j * zxyI
    Zyx = zyxR + 1j * zyxI

    if te_mode.lower() == 'xy':
        Z_TE, var_TE = Zxy, zxyVar
        Z_TM, var_TM = Zyx, zyxVar
    elif te_mode.lower() == 'yx':
        Z_TE, var_TE = Zyx, zyxVar
        Z_TM, var_TM = Zxy, zxyVar
    else:
        raise ValueError("te_mode must be 'xy' or 'yx'.")

    T = 1.0 / freqs
    if units.lower() == 'field':
        rho_TE = 0.2 * T * np.abs(Z_TE) ** 2
        rho_TM = 0.2 * T * np.abs(Z_TM) ** 2
    else:
        mu0 = 4 * np.pi * 1e-7
        w   = 2 * np.pi * freqs
        rho_TE = np.abs(Z_TE) ** 2 / (mu0 * w)
        rho_TM = np.abs(Z_TM) ** 2 / (mu0 * w)

    phi_TE = np.degrees(np.arctan2(Z_TE.imag, Z_TE.real))
    phi_TM = np.degrees(np.arctan2(Z_TM.imag, Z_TM.real))
    if unwrap_tm_phase:
        phi_TM = np.where(phi_TM < 0, phi_TM + 180.0, phi_TM)

    sigZ_TE = np.sqrt(np.maximum(var_TE, 0))
    sigZ_TM = np.sqrt(np.maximum(var_TM, 0))
    eRho_TE = 2 * rho_TE * sigZ_TE / np.abs(Z_TE)
    eRho_TM = 2 * rho_TM * sigZ_TM / np.abs(Z_TM)
    ePhi_TE = np.degrees(sigZ_TE / np.abs(Z_TE))
    ePhi_TM = np.degrees(sigZ_TM / np.abs(Z_TM))

    logRho_TE = np.log10(rho_TE);  logRho_TM = np.log10(rho_TM)
    elogR_TE  = eRho_TE / (rho_TE * np.log(10))
    elogR_TM  = eRho_TM / (rho_TM * np.log(10))
    log_floor = app_res_err_floor / np.log(10)
    elogR_TE  = np.maximum(elogR_TE, log_floor)
    elogR_TM  = np.maximum(elogR_TM, log_floor)
    ePhi_TE   = np.maximum(ePhi_TE, phase_err_floor)
    ePhi_TM   = np.maximum(ePhi_TM, phase_err_floor)

    nF = len(freqs)
    Data, DataErr, DatID, FreqID = [], [], [], []
    for i in range(nF):
        for val, err, code in [
            (phi_TE[i],    ePhi_TE[i],  104),
            (logRho_TE[i], elogR_TE[i], 123),
            (phi_TM[i],    ePhi_TM[i],  106),
            (logRho_TM[i], elogR_TM[i], 125),
        ]:
            if np.isnan(val) or np.isnan(err):
                continue
            Data.append(val);   DataErr.append(err)
            DatID.append(code); FreqID.append(i + 1)

    MTdata = dict(
        Freqs      = freqs,
        FreqID     = np.asarray(FreqID, int),
        DatID      = np.asarray(DatID,  int),
        Data       = np.asarray(Data),
        DataErr    = np.asarray(DataErr),
        WaterDepth = water_depth,
    )
    if verbose:
        flip = te_mode[::-1]
        print(f"read_edi: {filename}")
        print(f"  {nF} frequencies, {freqs.min():.3g}..{freqs.max():.3g} Hz "
              f"(T = {T.min():.0f}..{T.max():.0f} s)")
        print(f"  WaterDepth = {water_depth} m    units={units}    "
              f"TE<-Z{te_mode}, TM<-Z{flip}")
        print(f"  {len(Data)} data points (TE & TM, rho + phase; rho stored as log10)")
    return MTdata


def edi_to_S_MTdat(MTdata, mode='TM'):
    """
    Given the MTdata dict from read_edi, build an S.MTdat SimpleNamespace
    exactly as the inversion code expects, selecting either TE or TM.
    """
    if mode.upper() == 'TM':
        rho_code, phase_code = 125, 106
    elif mode.upper() == 'TE':
        rho_code, phase_code = 123, 104
    else:
        raise ValueError("mode must be 'TE' or 'TM'")

    app, appE, phase, phaseE, freqs = [], [], [], [], []
    for val, err, code, fid in zip(MTdata['Data'], MTdata['DataErr'],
                                   MTdata['DatID'], MTdata['FreqID']):
        if code == rho_code:
            app.append(val);   appE.append(err)
            freqs.append(MTdata['Freqs'][fid - 1])
        elif code == phase_code:
            phase.append(val); phaseE.append(err)

    return SimpleNamespace(
        TEappRes     = np.asarray(app),
        TEappResErr  = np.asarray(appE),
        TEphase      = np.asarray(phase),
        TEphaseErr   = np.asarray(phaseE),
        freqs        = np.asarray(freqs),
    )


# ---------------------------------------------------------------------
def read_mt_mare2dem(path, station_name,
                    app_res_err_floor=0.05,
                    phase_err_floor=2.0,
                    unwrap_tm_phase=True,
                    verbose=True):
    """
    Parse a Mare2DEM/SIO-style multi-station MT data file (the kind
    distributed with Blatter et al. 2022 / SERPENT,
    e.g. SERPENT_fullMTdataSet.txt) and return ONE station's data in
    the same dict structure that read_edi() returns -- so the rest of
    Step1_initialize.py works unchanged.

    File layout
    -----------
        Format: EMData_2.x
        # MT Frequencies:    nFreq
        <one freq per line, Hz>
        # MT Receivers:      nRx
        ! header line (starts with !)
        <nRx rows: X Y Z Theta Alpha Beta SolveStatic Name>
        # Data:       nData
        ! header line
        <nData rows: Type Freq# Tx# Rx# Data StdErr>

    Data-type codes used by this format (different from EDI's
    Mare2DEM-marine codes!):
        103 = LINEAR apparent resistivity TE (ohm-m)
        104 = phase TE (deg)
        105 = LINEAR apparent resistivity TM (ohm-m)
        106 = phase TM (deg)

    We convert 103/105 to log10 in-place and propagate the error
    (sigma_log10 = sigma_linear / (rho * ln 10)), then map
        103 -> 123   (so edi_to_S_MTdat sees TE log10 rho_a)
        105 -> 125   (TM log10 rho_a)
        104, 106  unchanged.

    Parameters
    ----------
    path : str
        Path to the .txt / .resp file.
    station_name : str
        Which receiver to extract (e.g. 's08', 's10').  Case-sensitive
        match against the Name column.
    app_res_err_floor : float
        Minimum fractional uncertainty in apparent resistivity.  In log10
        space the floor is app_res_err_floor / ln(10).
    phase_err_floor : float
        Minimum phase uncertainty (deg).
    unwrap_tm_phase : bool
        If True (default), phases that arrive in the (-180, -90) quadrant
        are mapped to (0, 90) by adding 180.  Marine TM data are often
        stored that way because of the sign convention on Zyx; the 1D
        inversion expects [0, 90].
    verbose : bool
        Print a one-line summary at the end.

    Returns
    -------
    MTdata : dict
        Same keys as read_edi():
        'Freqs', 'FreqID', 'DatID', 'Data', 'DataErr', 'WaterDepth'.
        WaterDepth is taken from the Z column of the selected receiver.
    """
    with open(path, 'r') as fh:
        lines = fh.readlines()

    # ---- pass 1: locate section headers --------------------------
    n_freq = n_rx = n_data = None
    i_freq_hdr = i_rx_hdr = i_dat_hdr = -1
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s.startswith('#'):
            continue
        lhs, _, rhs = s.partition(':')
        try:
            cnt = int(rhs.strip())
        except ValueError:
            continue
        key = lhs.lstrip('#').strip().lower()
        if key == 'mt frequencies':
            n_freq, i_freq_hdr = cnt, i
        elif key == 'mt receivers':
            n_rx, i_rx_hdr = cnt, i
        elif key == 'data':
            n_data, i_dat_hdr = cnt, i

    if n_freq is None: raise ValueError(f'{path}: # MT Frequencies header not found')
    if n_rx   is None: raise ValueError(f'{path}: # MT Receivers   header not found')
    if n_data is None: raise ValueError(f'{path}: # Data           header not found')

    # ---- frequencies ----------------------------------------------
    freqs = np.array([float(lines[i_freq_hdr + 1 + j].strip())
                      for j in range(n_freq)], dtype=float)

    # ---- receivers (skip any '!' header rows) ---------------------
    j = i_rx_hdr + 1
    while j < len(lines) and lines[j].lstrip().startswith('!'):
        j += 1
    rx_X = np.zeros(n_rx); rx_Y = np.zeros(n_rx); rx_Z = np.zeros(n_rx)
    rx_Name = []
    for k in range(n_rx):
        toks = lines[j + k].split()
        rx_X[k] = float(toks[0]); rx_Y[k] = float(toks[1]); rx_Z[k] = float(toks[2])
        # Theta, Alpha, Beta, SolveStatic, Name
        rx_Name.append(toks[7] if len(toks) > 7 else f'Rx{k+1}')

    if station_name not in rx_Name:
        raise ValueError(f"Station '{station_name}' not in file.  Available: {rx_Name}")
    sta_idx0 = rx_Name.index(station_name)            # 0-based
    sta_idx1 = sta_idx0 + 1                            # 1-based for file
    water_depth = float(rx_Z[sta_idx0])

    # ---- data block ------------------------------------------------
    j = i_dat_hdr + 1
    while j < len(lines) and lines[j].lstrip().startswith('!'):
        j += 1

    DatID, FreqID, Data, DataErr = [], [], [], []
    for k in range(n_data):
        toks = lines[j + k].split()
        rx_id = int(toks[3])
        if rx_id != sta_idx1:
            continue
        typ  = int(toks[0])
        fid  = int(toks[1])
        val  = float(toks[4])
        err  = float(toks[5])
        DatID.append(typ); FreqID.append(fid)
        Data.append(val);  DataErr.append(err)

    DatID   = np.asarray(DatID, dtype=int)
    FreqID  = np.asarray(FreqID, dtype=int)
    Data    = np.asarray(Data, dtype=float)
    DataErr = np.asarray(DataErr, dtype=float)

    if DatID.size == 0:
        raise ValueError(f"No data rows found for station {station_name}.")

    # ---- convert 103/105 (LINEAR rho_a) to log10 with error propagation
    ln10 = np.log(10.0)
    for code_lin, code_log in [(103, 123), (105, 125)]:
        sel = (DatID == code_lin)
        if not sel.any():
            continue
        rho_lin = Data[sel].copy()
        sig_lin = DataErr[sel].copy()
        # apply fractional-error floor BEFORE log conversion
        floor   = app_res_err_floor * rho_lin
        sig_lin = np.maximum(sig_lin, floor)
        log_rho   = np.log10(np.maximum(rho_lin, 1e-300))
        log_rho_e = sig_lin / (np.maximum(rho_lin, 1e-300) * ln10)
        Data[sel]    = log_rho
        DataErr[sel] = log_rho_e
        DatID[sel]   = code_log

    # ---- phase error floor and (optional) TM quadrant unwrap -----
    for code_ph in [104, 106]:
        sel = (DatID == code_ph)
        if not sel.any():
            continue
        DataErr[sel] = np.maximum(DataErr[sel], phase_err_floor)
        if unwrap_tm_phase and code_ph == 106:
            v = Data[sel]
            v[v < -90] += 180.0
            v[v > 270] -= 360.0
            Data[sel] = v

    # ---- summary --------------------------------------------------
    if verbose:
        T = 1.0 / freqs
        n_te = int(np.sum(DatID == 123))
        n_tm = int(np.sum(DatID == 125))
        print(f'read_mt_mare2dem: {path}')
        print(f'  Station: {station_name}  (X={rx_X[sta_idx0]:.0f}, '
              f'Y={rx_Y[sta_idx0]:.0f}, Z={water_depth:.1f} m)')
        print(f'  {n_freq} frequencies (T = {T.min():.1f} .. {T.max():.0f} s)')
        print(f'  {n_te} TE log10(rho_a), {n_tm} TM log10(rho_a) data points')
        if n_te:
            sel = (DatID == 123)
            print(f'  TE: log10(rho_a) = [{Data[sel].min():.2f}, '
                  f'{Data[sel].max():.2f}]')
        if n_tm:
            sel = (DatID == 125)
            print(f'  TM: log10(rho_a) = [{Data[sel].min():.2f}, '
                  f'{Data[sel].max():.2f}]')

    return {
        'Freqs':      freqs,
        'FreqID':     FreqID,
        'DatID':      DatID,
        'Data':       Data,
        'DataErr':    DataErr,
        'WaterDepth': water_depth,
    }
