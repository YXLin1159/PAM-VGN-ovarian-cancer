import os
import numpy as np
from scipy import signal
import cv2
from skimage import restoration
from tqdm import tqdm
from numba import njit
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

#%%
class scanParam:
    def __init__(self , aLineDepth , aLinesPerBScan , bScansPerCScan):
        self.depth = aLineDepth
        self.width = aLinesPerBScan
        self.length = bScansPerCScan

class saftParam:
    def __init__(self , N_ele_bf , N_zero_pad , scanParam):
        self.N_bf = N_ele_bf
        self.N_zero_pad = N_zero_pad
        self.c = 1500;  self.fs = 2e8
        self.ut_NA = 0.1
        self.dx = 5e-6; self.dy = 5e-6; self.dz = self.c / self.fs
        self.z_sample = (1 + np.arange(N_zero_pad , N_zero_pad + scanParam.depth))*self.dz
        self.t_focus = 4e-6; self.zf = self.t_focus*self.c
        self.r_bf = np.round((self.N_bf-1)/2).astype(np.int16)
        w = np.zeros((scanParam.depth , self.N_bf))
        for zi in range(scanParam.depth):
            r = np.floor(0.1 + (self.ut_NA * self.dz / self.dx)*np.abs(zi + 1 + N_zero_pad - self.t_focus*self.fs))
            r = np.minimum(r , self.r_bf).astype(np.int16)
            w[zi , (self.r_bf-r):(self.r_bf+r)] = 1
        self.SIR = w

#%% LOAD BINARY RAW DATA
def loadBin(scanDir , sampleIdx , scanParam):
    # RAW_DATA: #B X #A X DEPTH
    dirList = [x for x in os.listdir(scanDir) if x.endswith('.bin')]
    if sampleIdx < 1 or sampleIdx > len(dirList):
        print("Invalid sample index.")
        return
    else:
        F = os.path.join(scanDir , dirList[sampleIdx-1])
        Ntot = scanParam.depth * scanParam.width * scanParam.length
        with open(F , "rb") as fid:
            raw_data = np.fromfile(fid , dtype='>f8', count = Ntot)
            if raw_data.size != Ntot:
                raise ValueError(f"Expected {Ntot} elements, but got {raw_data.size}")
            raw_data = np.reshape(raw_data , (scanParam.length , scanParam.width , scanParam.depth))           
            for scan_idx in tqdm(np.arange(1 , scanParam.length , 2)):
                raw_data[scan_idx,:,:] = np.flipud(raw_data[scan_idx,:,:])
            print(">>>>>>>> RAW DATA LOADED")
            return raw_data.astype(np.double)

#%% PROCESS RAW DATA
@njit(parallel=True)
def saft1Bscan(bscan , Lz , La , r_bf , dx , zf , fs , N_zero_pad , z_sample , SIR):
    rf_sum = np.zeros((Lz , La))
    for idx_aline in range(La):
        scan_idx_left = np.maximum(0 , idx_aline - r_bf)
        scan_idx_right = np.minimum(La , idx_aline + r_bf)
        das_raw_data = np.zeros((Lz , scan_idx_right - scan_idx_left));
        for idx_saft in np.arange(scan_idx_left , scan_idx_right):
            rf_fil = bscan[: , idx_saft]
            diff_x = dx * np.abs(idx_saft - idx_aline)
            tof = (zf + np.multiply(np.sign(z_sample - zf) , np.sqrt(np.power(diff_x,2) + np.power((zf - z_sample),2)))) / 1500.0
            rxReadPtr = (np.rint(tof * fs) - N_zero_pad).astype(np.int16)
            idx1 = np.where(rxReadPtr > Lz) ; idx2 = np.where(rxReadPtr < 1)
            rxReadPtr[idx1] = Lz; rxReadPtr[idx2] = 1
            rf_tmp = rf_fil[rxReadPtr-1]
            rf_tmp[idx1] = 0; rf_tmp[idx2] = 0
            das_raw_data[:,idx_saft - scan_idx_left] = rf_tmp
            
        w_tmp = SIR[: , r_bf - idx_aline + scan_idx_left : r_bf - idx_aline + scan_idx_right]
        area_z = np.maximum(np.sum(w_tmp , 1) , 1 + 1e-3)
        das_raw_data = np.multiply(das_raw_data , w_tmp)          
        rf_sum0 = np.sum(das_raw_data , axis = 1);
        rf_sum_squared = np.square(np.divide(rf_sum0 , area_z))
        rf_squared_sum = np.divide(np.sum(np.square(das_raw_data) , axis = 1) , area_z)
        cf_saft = np.divide(rf_sum_squared , rf_squared_sum)
        cf_var_saft = np.nan_to_num(np.sqrt(np.divide(cf_saft , 1+1e-3-cf_saft)))
        rf_sum[:,idx_aline] = np.multiply(rf_sum0 , cf_var_saft)    
    return rf_sum


def computeSAFT(raw_data , saftParam):
    [Lb , La , Lz] = raw_data.shape
    rf_saft = np.zeros((Lz , La , Lb))
    env_saft = np.zeros((Lz , La , Lb))
    
    hpFilt = signal.firwin2(63 , [0, 0.03, 0.03, 1] , [0, 0, 1, 1]).reshape((63,1))
    lpFilt = signal.firwin2(64 , [0, 0.7, 0.7, 1] , [1, 0.9, 0.1, 0]).reshape((64,1))
    
    # NJIT DOES NOT SUPPORT CLASS OBJECT AS INPUT, SO HAVE TO EXTRACT THE PARAMETERS
    r_bf = saftParam.r_bf
    dx = saftParam.dx
    zf = saftParam.zf
    fs = saftParam.fs
    N_zero_pad = saftParam.N_zero_pad
    z_sample = saftParam.z_sample
    SIR = saftParam.SIR
    
    for idx_bscan in tqdm(range(Lb) , desc="Computing SAFT"):
        bscan = np.transpose(np.squeeze(raw_data[idx_bscan , : , :]))
        bscan = signal.convolve(signal.convolve(bscan , hpFilt ,'same') , lpFilt , 'same')      
        rf_sum = saft1Bscan(bscan , Lz , La , r_bf , dx , zf , fs , N_zero_pad , z_sample , SIR)
        rf_saft[:,:,idx_bscan] = rf_sum
        env_saft[:,:,idx_bscan] = np.abs(signal.hilbert(rf_sum , axis = 0))   
    return rf_saft , env_saft

def _computeLog(env_saft , dB_PA):
    log_data = env_saft / np.max(env_saft)
    min_dBdas = np.power(10,(-dB_PA/20));
    log_data = np.where(log_data < min_dBdas, 0, log_data)
    log_data = (20/dB_PA)*np.log10(log_data+1);
    return log_data

def logCompress(env_saft , dB_PA , projMethod):
    [Lz , La , Lb] = env_saft.shape 
    log_data = _computeLog(env_saft , dB_PA)
    signal_offset = 15
    if (projMethod == 'MIP'):
        mip_odd  = np.squeeze( np.max(log_data[ signal_offset:Lz , : , np.arange(0,Lb,2) ] , axis=0) );
        mip_even = np.squeeze( np.max(log_data[ signal_offset:Lz , : , np.arange(1,Lb,2) ] , axis=0) );
    elif (projMethod == "PIP"):
        mip_odd  = np.squeeze( np.sum(log_data[ signal_offset:Lz , : , np.arange(0,Lb,2) ] , axis=0) );
        mip_even = np.squeeze( np.sum(log_data[ signal_offset:Lz , : , np.arange(1,Lb,2) ] , axis=0) );
    else:
        raise Exception("Projection method not recognized.")
        
    mip_odd = mip_odd/np.max(mip_odd)
    mip_even = mip_even/np.max(mip_even)
    return log_data , mip_odd , mip_even

def mipTouchUp(mip):
    [La , Lb] = mip.shape
    mip = cv2.resize(mip, (2*Lb , La) , interpolation = cv2.INTER_LANCZOS4)  
    l_psf = 15
    sig_psf = 5
    ax = np.linspace(-(l_psf - 1) / 2., (l_psf - 1) / 2., l_psf)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig_psf))
    psf = np.outer(gauss, gauss)
    mip = restoration.richardson_lucy(signal.wiener(mip,(3,3)), psf, 5)
    return mip

def plotMIP(mip , lb , up):
    plt.imshow(mip, clim=(lb , up), cmap='gray')
    a = plt.gca()
    xax = a.axes.get_xaxis()
    xax = xax.set_visible(False)
    yax = a.axes.get_yaxis()
    yax = yax.set_visible(False)
    plt.tight_layout(pad=0.01)
    plt.show() 
    plt.gcf().set_dpi(400)
    return