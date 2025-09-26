import numpy as np
from typing import Tuple, Sequence, Optional
import warnings
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.interpolate import PchipInterpolator
from scipy.signal import medfilt, hilbert, find_peaks, fftconvolve
from tqdm import tqdm
from skimage.morphology import disk, binary_opening, binary_dilation, closing
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.exposure import equalize_adapthist
from skimage.restoration import denoise_bilateral

DAQ_DT = 2.5 # [ns]
N_ALINES_PER_BSCAN = 2000
N_BV = 5

def compute_freq_features(rf_signal: np.ndarray , rf_ref: np.ndarray , sensitivity: float = 20.0 , halfband: float = 8.0) -> Tuple[float, float, float]:
    '''
    Compute FREQUENCY domain features from the RF signal and reference spectrum.
    '''
    Nz, Nalines = rf_signal.shape
    Nz_ref  = rf_ref.size
    if Nz < 2*Nz_ref:
        rf_signal_aug = np.vstack((rf_signal , np.zeros((2 * Nz_ref - Nz, Nalines))))
    else:
        rf_signal_aug = rf_signal
    max_ref = np.max(np.abs(rf_ref))
    rf_ref = rf_ref / max_ref

    A = np.abs(np.fft.fft(rf_signal_aug , axis=0))
    pos_freq = A[:Nz_ref,:].copy()
    if A.shape[0] >= 2*Nz_ref:
        pos_part += A[Nz_ref:2*Nz_ref,:][::-1,:]
    spec_mean = np.mean(pos_freq , axis=1)
    spec_norm = spec_mean / rf_ref
    max_spec = np.max(spec_norm)
    spec_norm = spec_norm / max_spec

    half_len = Nz_ref // 2
    spec_50  = spec_norm[:half_len]
    freq_list = np.linspace(1.0,50.0,half_len+1)[1:]
    idx_max = int(np.nanargmax(spec_50))
    fc = float(freq_list[idx_max])
    lower_bound = sensitivity - halfband
    upper_bound = sensitivity + halfband
    idx_lower = max(np.searchsorted(freq_list , lower_bound, side='right')-1 , 0)
    idx_upper = min(np.searchsorted(freq_list , upper_bound, side='right') , len(freq_list)-1)
    if idx_upper <= idx_lower:
        warnings.warn("Upper frequency bound is less than or equal to lower frequency bound. Adjusting bounds.")
        idx_upper = min(idx_lower + 1, len(freq_list)-1)
    
    freq_midband = freq_list[idx_lower:(idx_upper+1)]
    spec_midband = spec_50[idx_lower:(idx_upper+1)]
    X = np.vstack((np.ones_like(freq_midband) , freq_midband)).T
    coef, *_ = np.linalg.lstsq(X , spec_midband , rcond=None)
    fintercept = float(coef[0])
    fslope = float(coef[1])

    return fc, fslope, fintercept

def compute_delay_features(rf_alines: np.ndarray, 
                           deriv_kernel: Optional[np.ndarray] = None, 
                           thresh_scale: float = 1.25, 
                           min_threshold: float = 0.1) -> Tuple[float, float]:
    '''
    Compute TIME domain features from the RF signal
    '''
    Nz, N_alines = rf_alines.shape
    if deriv_kernel is None:
        deriv_kernel = np.array([0.104550, 0.292315, 0.0, -0.292315, -0.104550], dtype=float)

    hf_filter = np.hanning(65)
    hf_filter = hf_filter / hf_filter.sum()

    Nz_interp = Nz * int(2)
    xi = np.linspace(0, Nz - 1, Nz_interp)  # original x positions are 0..Nz-1 (0-based)
    t1_all = np.full((N_alines,), np.nan, dtype=float)
    t2_all = np.full((N_alines,), np.nan, dtype=float)

    # helper to process 1 A-line
    def _process_aline(aline: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        # 1. remove high freq noise
        aline_f = np.convolve(uniform_filter1d(aline, size=3, mode="nearest"), hf_filter, mode="same")
        # 2. interpolate using pchip
        try:
            interpolator = PchipInterpolator(np.arange(Nz), aline_f)
            aline_interp = interpolator(xi)
        except Exception:
            aline_interp = np.interp(xi, np.arange(Nz), aline_f)
        # 3. normalize amplitude
        max_abs = np.max(np.abs(aline_interp))
        aline_interp = aline_interp / max_abs

        # 4. calculate rf derivative and smoothing + normalization
        aline_deriv = medfilt(np.convolve(aline_interp, deriv_kernel, mode="same") , kernel_size=3)
        max_abs_der = np.max(np.abs(aline_deriv))
        if max_abs_der > 0:
            aline_deriv = aline_deriv / max_abs_der

        aline_deriv2 = medfilt(np.convolve(aline_deriv, deriv_kernel, mode="same") , kernel_size=5)
        max_abs_der2 = np.max(np.abs(aline_deriv2))
        if max_abs_der2 > 0:
            aline_deriv2 = aline_deriv2 / max_abs_der2

        # 5. threshold and candidate detection
        early_window = min(30, aline_interp.size)
        threshold = max(thresh_scale * np.max(np.abs(aline_interp[:early_window])), min_threshold)

        # candidate indices: amplitude > threshold AND derivative small
        candidate_mask = (np.abs(aline_interp) > threshold) & (np.abs(aline_deriv) < threshold)
        candidate_idx = np.nonzero(candidate_mask)[0]
        if candidate_idx.size == 0:
            return np.nan, np.nan

        # t1: first candidate (Python indices are 0-based)
        t1_idx = int(candidate_idx[0])

        # t2: first peak of Hilbert envelope
        env = np.abs(hilbert(aline_interp))
        max_env = np.max(env)
        env = env / max_env

        t2_idx = None
        if env is not None:
            try:
                # try to find peaks with minimal height; height parameter expects envelope units [0,1]
                peaks, properties = find_peaks(env, height=threshold * 0.5)  # more permissive
                if peaks.size > 0:
                    # choose first peak (lowest index) whose height >= threshold*0.5
                    # but prefer peaks with height > threshold if any
                    heights = properties.get("peak_heights", None)
                    if heights is None:
                        t2_idx = int(peaks[0])
                    else:
                        t2_idx = int(peaks[above_thresh[0]] if (above_thresh := np.where(heights>threshold)[0]).size > 0 else int(peaks[0]))

                else:
                    idx_env = np.nonzero(env > threshold)[0]
                    if idx_env.size > 0:
                        # choose the index of the maximum value within the first contiguous run
                        # find the run:
                        run_start = idx_env[0]
                        run_end = run_start
                        for ii in range(1, idx_env.size):
                            if idx_env[ii] - idx_env[ii - 1] > 1:
                                break
                            run_end = idx_env[ii]
                        run_slice = env[run_start:run_end + 1]
                        if run_slice.size > 0:
                            t2_idx_local = run_start + int(np.argmax(run_slice))
                            t2_idx = int(t2_idx_local)
            except Exception:
                t2_idx = None

        # if we found both t1 and t2, compute t0 and return times
        if (t1_idx is not None) and (t2_idx is not None):
            # baseline search: indices before t1 where amplitude and derivatives < threshold
            search_end = max(0, t1_idx)
            if search_end >= 1:
                indices = np.arange(0, search_end + 1)
                k0_mask = (np.abs(aline_interp[indices]) < threshold) & \
                          (np.abs(aline_deriv[indices]) < threshold) & \
                          (np.abs(aline_deriv2[indices]) < threshold)
                k0_idx = np.nonzero(k0_mask)[0]
                if k0_idx.size > 0:
                    t0_idx = int(k0_idx[-1]) - 1
                    if t0_idx < 0:
                        t0_idx = 0
                else:
                    t0_idx = 0
            else:
                t0_idx = 0

            t1_ns = (t1_idx - t0_idx) * float(DAQ_DT)
            t2_ns = (t2_idx - t0_idx) * float(DAQ_DT)
            return t1_ns, t2_ns

        return np.nan, np.nan

    for i in range(N_alines):
        t1v, t2v = _process_aline(rf_alines[:, i])
        t1_all[i] = t1v
        t2_all[i] = t2v

    # medians ignoring NaNs
    t1_med = float(np.nanmedian(t1_all)) if np.any(np.isfinite(t1_all)) else float(np.nan)
    t2_med = float(np.nanmedian(t2_all)) if np.any(np.isfinite(t2_all)) else float(np.nan)
    return t1_med, t2_med

def compute_lrf(mip: np.ndarray,
                rho_1: int = 10,
                rho_2_test: Sequence[int] = tuple(range(2,11)),
                rho_3: int = 2,
                n_angle: int = 36,
                orientation_mode: str = 'fast',
                ) -> Tuple[np.ndarray , np.ndarray]:
    H , W  = mip.shape
    mip_norm = (mip - mip.min()) / (mip.max() - mip.min() + 1e-12)
    img_enhance = equalize_adapthist(denoise_bilateral(mip_norm, sigma_color=0.05, sigma_spatial=2, win_size=5, multichannel=False), clip_limit=0.01)
    
    # K_rho2: z-direction derivative kernel
    k = np.array([0.030320, 0.249724, 0.439911, 0.249724, 0.030320], dtype=float)
    d = np.array([0.104550, 0.292315, 0.0, -0.292315, -0.104550], dtype=float)
    K_rho2 = np.convolve(np.convolve(k, d), d)  # 1D pattern used as K_rho2 basis
    L_k_rho2 = K_rho2.size

    # K_rho3: gaussian blob kernel
    x3 = np.arange(-2 * rho_3, 2 * rho_3 + 1)
    K_rho3_1d = (1.0 / rho_3) * np.exp(- (x3 ** 2) / (2.0 * rho_3 ** 2))
    K_rho3 = np.outer(K_rho3_1d, K_rho3_1d)

    angle_list = np.linspace(90.0 , -90.0 , n_angle+1)[:-1]
    x1 = np.arange(-2*rho_1 , 2*rho_1+1)
    N_thickness = len(rho_2_test)

    LRT_all = np.zeros((H,W,n_angle,N_thickness) , dtype=float)
    img_pad = np.pad(img_enhance , pad_width=2*rho_1, mode='edge')

    for idx_rho2 , rho2 in tqdm(enumerate(list(rho_2_test)) , desc='Looping over rho 2'):
        y2 = np.linspace(-2*rho2 , 2*rho2 , num = L_k_rho2)
        xx , yy = np.meshgrid(x1 , y2)
        xy = np.column_stack((xx.ravel() , yy.ravel()))
        for idx_angle , theta in enumerate(angle_list):
            theta_rad = np.deg2rad(theta)
            R = np.array([[np.cos(theta_rad) , -np.sin(theta_rad)] , [np.sin(theta_rad) , np.cos(theta_rad)]] , dtype=float)
            rot_xy = xy @ R
            xx_qr = rot_xy[:,0].reshape(xx.shape)
            yy_qr = rot_xy[:,1].reshape(yy.shape)
            K_rho1 = (1.0/rho_1) * np.exp( -(xx_qr ** 2) / (2.0 * rho_1 ** 2) )
            K_rho2_interp = np.interp(yy_qr.ravel(), y2, K_rho2).reshape(yy_qr.shape)
            mask_oob = (yy_qr < y2[0]) | (yy_qr > y2[-1])
            if mask_oob.any():
                K_rho2_interp[mask_oob] = 0.0

            h_eta = K_rho1 * K_rho2_interp
            e_eta = fftconvolve(h_eta, K_rho3, mode="same")
            conv_theta = -fftconvolve(img_pad, e_eta, mode="same")
            conv_theta[conv_theta < 0] = 0.0
            pad = 2*rho_1
            LRT_all[:, :, idx_angle, idx_rho2] = conv_theta[pad: pad + H, pad: pad + W]
    
    LRT_max_thickness = np.max(LRT_all , axis=3) # shape (H, W, n_angle)
    if orientation_mode == "fast":
        argmax_idx = np.argmax(LRT_max_thickness, axis=2)   # shape (H, W)
        angle_values = angle_list  # mapping index->angle
        idx_orient = np.asarray(angle_values)[argmax_idx]
        segbv_raw = np.max(LRT_max_thickness, axis=2)
    else:
        angle_template = np.linspace(180.0, -180.0, 2 * n_angle, endpoint=False)
        cosine_template = np.cos(np.deg2rad(angle_template))
        cosine_template[cosine_template < 0] = 0.0

        # flatten pixel vectors (M, n_angle) where M = H*W
        M = H * W
        T = median_filter(LRT_max_thickness.reshape(M, n_angle) , size=(1,3)) # shape (M, n_angle)

        # For each pixel compute xcorr(T[i], cosine_template, maxlag=n_angle)
        idx_orient_flat = np.zeros(M, dtype=float)
        # precompute lag vector for xcorr result length 2*n_angle+1
        lags = np.arange(-n_angle, n_angle + 1)
        for i in range(M):
            test_vec = T[i, :]
            r = np.correlate(test_vec, cosine_template, mode="full")   # length = n_angle + len(cosine_template) -1 = 3*n_angle -1
            center = (r.size - 1) // 2
            r_segment = r[center - n_angle: center + n_angle + 1]  # length 2*n_angle+1
            max_idx = np.argmax(r_segment)
            lag_max = lags[max_idx]
            orient_deg = -(-lag_max * 180.0 / n_angle - 90.0)
            idx_orient_flat[i] = orient_deg
        idx_orient = idx_orient_flat.reshape(H, W)
        segbv_raw = np.max(LRT_max_thickness, axis=2)
    
    segbv_norm = (segbv_raw - segbv_raw.min()) / (segbv_raw.max() - segbv_raw.min() + 1e-12)
    segbv = equalize_adapthist(denoise_bilateral(segbv_norm, sigma_color=0.05, sigma_spatial=1, win_size=3, multichannel=False), clip_limit=0.03)
    return segbv , idx_orient

def generate_graph_data(RF_raw_all: np.ndarray, 
                        RF_DAS_env: np.ndarray, 
                        idx_bscan_all: np.ndarray, 
                        rf_freq_ref: np.ndarray,
                        im_mip_odd_correct: np.ndarray, 
                        im_mip_even_correct: np.ndarray,
                        signal_offset: int = 15,
                        N_neighbor: int = 21) -> np.ndarray:
    graph_data_list = []

    for _, idx_bscan_tmp in enumerate(tqdm(idx_bscan_all, desc="Generating Graphs")):
        # Determine x-range depending on odd/even B-scan
        x_range = (121, N_ALINES_PER_BSCAN) if idx_bscan_tmp % 2 == 1 else (1, N_ALINES_PER_BSCAN - 120)
        x_range = np.arange(*x_range)

        # Extract RF B-scan and neighbors
        rf_bscan = RF_raw_all[:, x_range, idx_bscan_tmp].copy()
        rf_bscan[:signal_offset, :] = 0

        rf_neighbor = RF_raw_all[:, x_range, idx_bscan_tmp - 2*N_neighbor : idx_bscan_tmp + 2*N_neighbor : 2].copy()
        rf_neighbor[:signal_offset, :, :] = 0

        env_bscan = median_filter(RF_DAS_env[:, x_range, idx_bscan_tmp], size=(1,5))
        env_bscan = median_filter(env_bscan, size=(3,3))
        env_bscan[:signal_offset, :] = 0

        env_neighbor = RF_DAS_env[:, x_range, idx_bscan_tmp - 2*N_neighbor : idx_bscan_tmp + 2*N_neighbor : 2].copy()
        env_neighbor[:signal_offset, :, :] = 0

        # Binarize and morphological opening
        env_bw = binary_opening(env_bscan > 1e-2, selem=disk(3))
        labeled_cc = label(env_bw)
        regions = regionprops(labeled_cc)

        if len(regions) >= N_BV and all([r.area > 50 for r in sorted(regions, key=lambda r: r.area, reverse=True)[:N_BV]]):
            # Compute vessel diameters and directionalities
            if idx_bscan_tmp % 2 == 1:
                idx_mip_tmp = int(np.ceil(idx_bscan_tmp / 2))
                mip_neighbor = im_mip_odd_correct[x_range, idx_mip_tmp-N_neighbor : idx_mip_tmp+N_neighbor+1]
            else:
                idx_mip_tmp = int(idx_bscan_tmp / 2)
                mip_neighbor = im_mip_even_correct[x_range, idx_mip_tmp-N_neighbor : idx_mip_tmp+N_neighbor+1]

            mip_neighbor = resize(mip_neighbor, (mip_neighbor.shape[0], 2*mip_neighbor.shape[1]))
            _ , idx_orient = compute_lrf(mip_neighbor)
            idx_orient_bscan = idx_orient[:, 2*N_neighbor : 2*(N_neighbor+2)].mean(axis=1)

            # Segment top N_BV vessels
            graph_bscan = np.zeros((N_BV, 8))
            top_regions = sorted(regions, key=lambda r: r.area, reverse=True)[:N_BV]

            for idx_bv, r in enumerate(top_regions):
                x_coord = int(round(r.centroid[1]))
                x_coord_um = x_coord * 3
                bv_bb = r.bbox
                bv_diameter = abs((bv_bb[3] - bv_bb[1]) * 3 * np.sin(np.deg2rad(90 - abs(idx_orient_bscan[x_coord-2 : x_coord+3].mean()))))

                x_range_tmp = np.arange(int(x_coord - (bv_bb[3]-bv_bb[1])/4), int(x_coord + (bv_bb[3]-bv_bb[1])/4)+1)
                rf_alines_tmp = rf_bscan[:, x_range_tmp]

                t1, t2 = compute_delay_features(rf_alines_tmp)
                fc, fslope, fintercept = compute_freq_features(rf_alines_tmp, rf_freq_ref)

                graph_bscan[idx_bv, :] = [x_coord_um, bv_diameter, idx_orient_bscan[x_coord], t1, t2, fc, fslope, fintercept]

            graph_bscan_aug = np.hstack([np.full((N_BV, 1), idx_bscan_tmp), graph_bscan])
            graph_data_list.append(graph_bscan_aug)

    # Concatenate all graph data at once
    graph_data = np.vstack(graph_data_list) if graph_data_list else np.empty((0, 9))
    return graph_data
