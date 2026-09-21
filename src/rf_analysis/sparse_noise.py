"""
Receptive field reconstruction and fitting from sparse noise stimuli.

Based on Lindeberg (2025c,d) Gaussian derivative model for visual receptive fields.
Extracts spatial parameters: σ (scale), θ (orientation), κ (elongation).

RF reconstruction uses **ridge regression with a lagged stimulus design matrix**
(replacing the earlier spike-triggered average / STA approach). A signed stimulus
encoding (+1 ON, −1 OFF, 0 grey) is regressed against the averaged ΔF/F response
using cross-validated L2 regularisation.  The fitted weights are reshaped into a
spatiotemporal RF (n_lags × grid_h × grid_w); the spatial RF at the lag with
maximum response energy is passed to the Gaussian fitter.

Orientation estimation
----------------------
Two orientation estimators are provided and recorded separately:

  θ (theta)  — major axis of the fitted Gaussian envelope (all orders).
               Reliable for m=0 when κ > ~1.3; poorly constrained for m=1
               (κ ≈ 1) because the envelope is nearly circular.

  φ (phi)    — differentiation direction estimated directly from lobe geometry
               (m=1 and m=2 only; NaN for m=0 where no lobes exist).
               For m=1: vector from negative-lobe centroid → positive-lobe centroid.
               For m=2: principal axis of the two same-sign flanking regions.
               Reliable even when κ ≈ 1 because it exploits lobe *positions*,
               not envelope shape.

  theta_hybrid — recommended single orientation column:
               φ for m=1 and m=2 (lobe geometry, more reliable);
               θ for m=0 (envelope, only reliable estimator available).

Timing rationale (Allen Brain Observatory 2P / GCaMP6f, ~30 Hz imaging):
  - Mouse V1 neural response latency:  ~25–50 ms  (electrophysiology)
  - GCaMP6f rise time (single AP):      ~45 ms
  - GCaMP6f half-decay (single AP):    ~142 ms
  - Visual stimulus → Ca²⁺ peak:       ~100–200 ms
  → response_delay = 4 frames (≈ 133 ms): starts measuring after Ca²⁺ rise begins
  → response_window = 5 frames (≈ 165 ms): averages through Ca²⁺ peak
  → n_lags = 8 presentations (≈ 267 ms at 30 Hz): covers GCaMP6f transient history

References
----------
Chen TW et al. (2013) Ultrasensitive fluorescent proteins for imaging neuronal
    activity. *Nature* 499:295–300. https://doi.org/10.1038/nature12354
de Vries SEJ et al. (2020) A large-scale standardized physiological survey reveals
    functional organization of the mouse visual cortex. *Nat Neurosci* 23:138–151.
    https://doi.org/10.1038/s41593-019-0550-9
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import Ridge, RidgeCV
import warnings


# Allen SDK locally sparse noise template values (0-255 encoding)
LSN_ON = 255       # White stimulus
LSN_OFF = 0        # Black stimulus
LSN_GREY = 127     # Gray background
LSN_OFF_SCREEN = 64

# Default ridge regularisation grid (log-spaced, covers 7 decades)
_LAMBDA_GRID = np.logspace(-2, 5, 25)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reconstruct_rf(dataset, cell_id, stimulus_name='locally_sparse_noise_4deg',
                   response_delay=4, response_window=5,
                   n_lags=8, lambda_values=None, cv_folds=5):
    """
    Reconstruct 2D receptive field using ridge regression with a lagged
    stimulus design matrix.

    For each stimulus presentation p, the response is the mean ΔF/F in a
    window starting ``response_delay`` frames after onset.  A design matrix
    X of shape (n_presentations, n_pixels × n_lags) encodes the signed
    stimulus pattern (ON=+1, OFF=−1, grey=0) at presentation p and the
    ``n_lags − 1`` preceding presentations.  Ridge regression with
    cross-validated λ is solved; the coefficients are reshaped into a
    spatiotemporal RF (n_lags, grid_h, grid_w).  The spatial RF at the lag
    with maximum absolute energy is returned as a signed map, split into ON
    and OFF subfields for downstream Gaussian fitting.

    Parameters
    ----------
    dataset : allensdk OphysExperimentData
        From ``boc.get_ophys_experiment_data(exp_id)``.
    cell_id : int
        Cell specimen ID.
    stimulus_name : str
        Falls back to any available sparse noise stimulus if not found.
    response_delay : int, default=4
        Imaging frames to skip after stimulus onset before measuring response.
        At ~30 Hz, 4 frames ≈ 133 ms — accounts for visual latency (~25–50 ms)
        and early GCaMP6f rise (~45 ms rise time; Chen et al. 2013).
    response_window : int, default=5
        Imaging frames to average for the response measurement.
        At ~30 Hz, 5 frames ≈ 165 ms — covers the GCaMP6f Ca²⁺ peak
        (~100–200 ms post-stimulus; de Vries et al. 2020).
    n_lags : int, default=8
        Number of presentation-level lags included in the design matrix.
        At ~30 Hz with one-frame presentations, 8 lags ≈ 267 ms of stimulus
        history — captures the full GCaMP6f transient (half-decay ~142 ms).
    lambda_values : array-like or None
        Ridge regularisation values to search over.  Defaults to
        np.logspace(-2, 5, 25).
    cv_folds : int, default=5
        Number of cross-validation folds for λ selection.

    Returns
    -------
    rf_on : ndarray, shape (n_y, n_x)
        Positive part of the signed RF map at the best lag (ON subfield).
    rf_off : ndarray, shape (n_y, n_x)
        Negative part of the signed RF map, sign-flipped (OFF subfield).
    metadata : dict
        Keys: grid_size, pixel_size_deg, stim_name, cell_id,
        n_presentations, response_delay, response_window,
        method, n_lags, best_lag, best_lag_ms, ridge_lambda, cv_folds.
    """
    lambdas = _LAMBDA_GRID if lambda_values is None else np.asarray(lambda_values)

    # --- Find sparse noise stimulus ---
    stimuli = dataset.list_stimuli()
    if stimulus_name not in stimuli:
        sparse_options = [s for s in stimuli if 'sparse_noise' in s]
        if not sparse_options:
            raise ValueError(
                f"No sparse noise stimulus found. Available: {stimuli}")
        stimulus_name = sparse_options[0]

    # --- Load stimulus template and presentation table ---
    stim_table = dataset.get_stimulus_table(stimulus_name)
    template_result = dataset.get_locally_sparse_noise_stimulus_template(
        stimulus=stimulus_name)
    if isinstance(template_result, tuple):
        template, _ = template_result
    else:
        template = template_result
    n_template_frames, grid_h, grid_w = template.shape

    unique_vals = np.unique(template)
    on_val  = int(unique_vals.max())   # 255
    off_val = int(unique_vals.min())   # 0

    # --- Load ΔF/F trace for this cell ---
    _, dff = dataset.get_dff_traces(cell_specimen_ids=[cell_id])
    dff = dff[0]
    n_timepoints = len(dff)

    # --- Response per presentation (vectorised) ---
    starts        = stim_table['start'].values.astype(int)
    frame_indices = stim_table['frame'].values.astype(int)
    n_pres        = len(starts)

    window_offsets = np.arange(response_window)
    window_idx     = (starts + response_delay)[:, np.newaxis] + window_offsets

    valid_frames    = (window_idx >= 0) & (window_idx < n_timepoints)
    window_idx_safe = np.clip(window_idx, 0, n_timepoints - 1)

    dff_windows = dff[window_idx_safe]
    dff_windows[~valid_frames] = np.nan
    responses = np.nanmean(dff_windows, axis=1)  # (n_pres,)

    valid_mask = (
        ~np.isnan(responses)
        & (frame_indices >= 0)
        & (frame_indices < n_template_frames)
    )

    if valid_mask.sum() < max(20, cv_folds):
        raise ValueError(
            f"Cell {cell_id}: only {valid_mask.sum()} valid presentations "
            f"(need ≥ {max(20, cv_folds)}).")

    # --- Build lagged stimulus design matrix ---
    X = _build_design_matrix(template, frame_indices, on_val, off_val,
                              grid_h, grid_w, n_lags)

    X_fit = X[valid_mask]
    y_fit = responses[valid_mask]

    # --- Ridge regression with cross-validated λ ---
    ridge = RidgeCV(alphas=lambdas, cv=cv_folds, fit_intercept=True)
    ridge.fit(X_fit, y_fit)

    # --- Spatiotemporal RF: (n_lags, grid_h, grid_w) ---
    strf = ridge.coef_.reshape(n_lags, grid_h, grid_w)

    # Best lag = lag with maximum absolute peak response
    lag_energy = np.array([np.abs(strf[lag]).max() for lag in range(n_lags)])
    best_lag   = int(np.argmax(lag_energy))
    rf_map     = strf[best_lag]   # signed spatial RF

    # Separate ON (+) and OFF (−) subfields
    rf_on  = np.maximum(rf_map,  0.0)
    rf_off = np.maximum(-rf_map, 0.0)

    # Estimate presentation interval for lag→ms conversion
    if n_pres > 1:
        diffs = np.diff(starts[:min(100, n_pres)])
        median_interval_frames = float(np.median(diffs[diffs > 0]))
    else:
        median_interval_frames = 1.0
    frame_rate_hz = 30.0
    ms_per_frame  = 1000.0 / frame_rate_hz
    best_lag_ms   = best_lag * median_interval_frames * ms_per_frame

    metadata = {
        'grid_size':        (grid_h, grid_w),
        'pixel_size_deg':   _pixel_size(stimulus_name),
        'stim_name':        stimulus_name,
        'cell_id':          cell_id,
        'n_presentations':  int(valid_mask.sum()),
        'response_delay':   response_delay,
        'response_window':  response_window,
        'method':           'ridge',
        'n_lags':           n_lags,
        'best_lag':         best_lag,
        'best_lag_ms':      round(best_lag_ms, 1),
        'ridge_lambda':     float(ridge.alpha_),
        'cv_folds':         cv_folds,
    }

    return rf_on, rf_off, metadata


def fit_gaussian_derivative(rf_on, rf_off, pixel_size_deg=4.65,
                            smooth_sigma=0.75, order=1):
    """
    Fit 2D Gaussian to the dominant subfield and extract Lindeberg parameters.

    Strategy:
    1. Combine ON − OFF to get a signed RF map.
    2. Identify the dominant polarity (largest absolute peak).
    3. Flip if needed so the dominant peak is positive.
    4. Optionally smooth, then fit ``gaussian_2d`` via ``curve_fit``.
    5. Convert (σ_x, σ_y, θ) → Lindeberg (σ, θ, κ).

    Parameters
    ----------
    rf_on, rf_off : ndarray, shape (n_y, n_x)
        Subfield maps from ``reconstruct_rf()``.
    pixel_size_deg : float, default=4.65
    smooth_sigma : float, default=0.75
    order : int, default=1
        Unused — kept for API compatibility.

    Returns
    -------
    params : dict
        sigma, theta, kappa, x0, y0, amplitude, sigma_x, sigma_y,
        dominant_subfield.
    fit_quality : dict
        r_squared, rmse, converged, message.
    """
    rf_combined = rf_on - rf_off
    grid_h, grid_w = rf_combined.shape

    if smooth_sigma > 0:
        rf_work = gaussian_filter(rf_combined, sigma=smooth_sigma)
    else:
        rf_work = rf_combined.copy()

    # --- Determine dominant polarity ---
    if abs(rf_work.max()) >= abs(rf_work.min()):
        dominant = 'ON'
    else:
        dominant = 'OFF'
        rf_work = -rf_work

    # --- Initial parameter estimates from image moments ---
    Y, X = np.mgrid[0:grid_h, 0:grid_w]
    baseline = np.median(rf_work)
    rf_pos   = np.maximum(rf_work - baseline, 0)
    total    = rf_pos.sum()

    if total < 1e-10:
        return _nan_params(dominant, pixel_size_deg), _fail_quality("No signal")

    w       = rf_pos / total
    x0_init = float(np.clip((X * w).sum(), 0, grid_w - 1))
    y0_init = float(np.clip((Y * w).sum(), 0, grid_h - 1))

    dx  = X - x0_init
    dy  = Y - y0_init
    Mxx = (dx ** 2 * w).sum()
    Myy = (dy ** 2 * w).sum()
    Mxy = (dx * dy * w).sum()

    eigvals, eigvecs = np.linalg.eigh([[Mxx, Mxy], [Mxy, Myy]])
    eigvals = np.maximum(eigvals, 0.25)

    sigma_major_init = float(np.sqrt(eigvals[1]))
    sigma_minor_init = float(np.sqrt(eigvals[0]))
    theta_init       = float(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))

    amp_init    = float(rf_work.max() - baseline)
    offset_init = float(baseline)

    p0 = [amp_init, x0_init, y0_init,
          max(sigma_major_init, 0.5), max(sigma_minor_init, 0.5),
          theta_init, offset_init]

    lower = [0,      -2,         -2,          0.3, 0.3, -np.pi, -np.inf]
    upper = [np.inf, grid_w + 2, grid_h + 2,
             grid_w / 2, grid_h / 2, np.pi, np.inf]

    xy_data = (X, Y)
    z_data  = rf_work.ravel()

    try:
        popt, _ = curve_fit(gaussian_2d, xy_data, z_data, p0=p0,
                            bounds=(lower, upper), maxfev=10000)
        converged = True
        message   = 'Converged'
    except (RuntimeError, ValueError) as e:
        popt      = np.array(p0, dtype=float)
        converged = False
        message   = str(e)

    amp, x0, y0, sx, sy, theta_rad, offset = popt
    sigma, theta_deg, kappa = _to_lindeberg(sx, sy, theta_rad)

    params = {
        'sigma':             float(sigma * pixel_size_deg),
        'theta':             float(theta_deg),
        'kappa':             float(kappa),
        'x0':                float(x0 * pixel_size_deg),
        'y0':                float(y0 * pixel_size_deg),
        'amplitude':         float(amp),
        'sigma_x':           float(sx * pixel_size_deg),
        'sigma_y':           float(sy * pixel_size_deg),
        'dominant_subfield': dominant,
    }

    fitted    = gaussian_2d(xy_data, *popt)
    ss_res    = float(np.sum((z_data - fitted) ** 2))
    ss_tot    = float(np.sum((z_data - z_data.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse      = float(np.sqrt(ss_res / len(z_data)))

    fit_quality = {
        'r_squared': r_squared,
        'rmse':      rmse,
        'converged': converged,
        'message':   message,
    }

    return params, fit_quality


def extract_spatial_parameters(dataset, cell_id, quality_threshold=0.3,
                                **reconstruct_kwargs):
    """
    End-to-end: sparse noise → σ, θ, κ.

    Returns dict with all params + quality + metadata, or ``None``
    if the fit quality is below threshold.
    """
    try:
        rf_on, rf_off, metadata = reconstruct_rf(dataset, cell_id,
                                                  **reconstruct_kwargs)
        params, quality = fit_gaussian_derivative(
            rf_on, rf_off, pixel_size_deg=metadata['pixel_size_deg'])

        if quality['r_squared'] < quality_threshold:
            warnings.warn(
                f"Cell {cell_id}: R²={quality['r_squared']:.3f} "
                f"below {quality_threshold}")
            return None

        return {**params, **quality, **metadata}

    except Exception as e:
        warnings.warn(f"Cell {cell_id}: {e}")
        return None


# ===========================================================================
# Batch extraction (fast path for full-container processing)
# ===========================================================================

def batch_extract_spatial_parameters(dataset, cell_ids,
                                      stimulus_name='locally_sparse_noise_4deg',
                                      quality_threshold=0.3,
                                      response_delay=4, response_window=5,
                                      n_lags=8, lambda_values=None):
    """
    Fast batch RF extraction for a full container (m=0 dominant-subfield fit).

    Speedup over calling ``extract_spatial_parameters()`` per neuron:
    * ``get_dff_traces()`` called once for all neurons.
    * Design matrix X built once (vectorised).
    * ``RidgeCV(cv=None)`` uses GCV with SVD shared across neurons.

    Note: this function uses the original m=0 dominant-subfield fit.
    For order-corrected parameters use classify_rf_order() per neuron.
    """
    lambdas  = _LAMBDA_GRID if lambda_values is None else np.asarray(lambda_values)
    cell_ids = list(cell_ids)

    # --- Find stimulus ---
    stimuli = dataset.list_stimuli()
    if stimulus_name not in stimuli:
        sparse = [s for s in stimuli if 'sparse_noise' in s]
        if not sparse:
            raise ValueError(f"No sparse noise stimulus. Available: {stimuli}")
        stimulus_name = sparse[0]

    stim_table    = dataset.get_stimulus_table(stimulus_name)
    tr            = dataset.get_locally_sparse_noise_stimulus_template(stimulus=stimulus_name)
    template      = tr[0] if isinstance(tr, tuple) else tr
    n_tf, grid_h, grid_w = template.shape
    on_val, off_val = int(template.max()), int(template.min())
    pix_size      = _pixel_size(stimulus_name)
    n_pixels      = grid_h * grid_w

    starts        = stim_table['start'].values.astype(int)
    frame_indices = stim_table['frame'].values.astype(int)
    n_pres        = len(starts)

    # --- Single NWB read for all neurons ---
    _, all_dff = dataset.get_dff_traces(cell_specimen_ids=cell_ids)
    n_cells, n_tp = all_dff.shape

    # --- Build X once (vectorised) ---
    X = _build_design_matrix(template, frame_indices, on_val, off_val,
                              grid_h, grid_w, n_lags)

    # --- Response matrix Y: (n_pres, n_cells) ---
    win_idx  = (starts + response_delay)[:, None] + np.arange(response_window)
    in_bnds  = (win_idx >= 0) & (win_idx < n_tp)
    win_safe = np.clip(win_idx, 0, n_tp - 1)

    dff_wins = all_dff[:, win_safe]
    dff_wins[:, ~in_bnds] = np.nan
    Y = np.nanmean(dff_wins, axis=2).T          # (n_pres, n_cells)
    Y = np.where(np.isnan(Y), 0.0, Y).astype(np.float64)

    frame_valid = (frame_indices >= 0) & (frame_indices < n_tf)
    X_fit = X[frame_valid].astype(np.float64)
    Y_fit = Y[frame_valid]
    n_valid = X_fit.shape[0]

    if n_valid < 20:
        raise ValueError(f"Only {n_valid} valid presentations — cannot fit.")

    # --- GCV ridge ---
    try:
        rcv = RidgeCV(alphas=lambdas, cv=None,
                      fit_intercept=True, alpha_per_target=True)
        rcv.fit(X_fit, Y_fit)
        W            = rcv.coef_
        lambdas_used = np.atleast_1d(rcv.alpha_)
    except TypeError:
        rcv = RidgeCV(alphas=lambdas, cv=None, fit_intercept=True)
        rcv.fit(X_fit, Y_fit.mean(axis=1, keepdims=True))
        lam = float(rcv.alpha_)
        r   = Ridge(alpha=lam, fit_intercept=True)
        r.fit(X_fit, Y_fit)
        W            = r.coef_
        lambdas_used = np.full(n_cells, lam)

    diffs      = np.diff(starts[:min(100, n_pres)])
    ms_per_lag = float(np.median(diffs[diffs > 0])) * (1000.0 / 30.0)

    results = []
    for i, cid in enumerate(cell_ids):
        try:
            strf       = W[i].reshape(n_lags, grid_h, grid_w)
            lag_energy = np.array([np.abs(strf[lag]).max() for lag in range(n_lags)])
            best_lag   = int(np.argmax(lag_energy))
            rf_map     = strf[best_lag]

            rf_on  = np.maximum(rf_map,  0.0)
            rf_off = np.maximum(-rf_map, 0.0)

            params, quality = fit_gaussian_derivative(
                rf_on, rf_off, pixel_size_deg=pix_size)

            if quality['r_squared'] < quality_threshold:
                continue

            lam_i = float(lambdas_used[i] if i < len(lambdas_used) else lambdas_used[0])
            results.append({
                **params, **quality,
                'cell_id':          cid,
                'grid_size':        (grid_h, grid_w),
                'pixel_size_deg':   pix_size,
                'stim_name':        stimulus_name,
                'n_presentations':  n_valid,
                'response_delay':   response_delay,
                'response_window':  response_window,
                'method':           'ridge_batch',
                'n_lags':           n_lags,
                'best_lag':         best_lag,
                'best_lag_ms':      round(best_lag * ms_per_lag, 1),
                'ridge_lambda':     lam_i,
            })
        except Exception as e:
            warnings.warn(f"Cell {cid}: {e}")

    return results


# ===========================================================================
# Helper / model functions
# ===========================================================================

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """Rotated 2D Gaussian, returns flattened array for ``curve_fit``."""
    x, y   = xy
    cos_t  = np.cos(theta)
    sin_t  = np.sin(theta)

    a = cos_t**2 / (2 * sigma_x**2) + sin_t**2 / (2 * sigma_y**2)
    b = sin_t * cos_t * (1 / (2 * sigma_x**2) - 1 / (2 * sigma_y**2))
    c = sin_t**2 / (2 * sigma_x**2) + cos_t**2 / (2 * sigma_y**2)

    dx = x - x0
    dy = y - y0
    g  = offset + amplitude * np.exp(-(a * dx**2 + 2 * b * dx * dy + c * dy**2))
    return g.ravel()


def _build_design_matrix(template, frame_indices, on_val, off_val,
                          grid_h, grid_w, n_lags):
    """
    Build a (n_pres, n_pixels × n_lags) signed stimulus design matrix.

    Encoding: ON pixel = +1, OFF pixel = −1, grey/other = 0.
    """
    n_pres   = len(frame_indices)
    n_tf     = template.shape[0]
    n_pixels = grid_h * grid_w

    valid_f    = (frame_indices >= 0) & (frame_indices < n_tf)
    fi_clipped = np.clip(frame_indices, 0, n_tf - 1)
    frames     = template[fi_clipped].reshape(n_pres, n_pixels).astype(np.float32)
    stim_enc   = np.where(frames == on_val,  np.float32(1.0),
                 np.where(frames == off_val, np.float32(-1.0), np.float32(0.0)))
    stim_enc[~valid_f] = 0.0

    X = np.zeros((n_pres, n_pixels * n_lags), dtype=np.float32)
    for lag in range(n_lags):
        n_v = n_pres - lag
        if n_v > 0:
            X[lag:, lag * n_pixels:(lag + 1) * n_pixels] = stim_enc[:n_v]

    return X


def _to_lindeberg(sigma_x, sigma_y, theta_rad):
    """Convert (σ_x, σ_y, θ_rad) → Lindeberg (σ, θ°, κ)."""
    sigma = float(np.sqrt(sigma_x * sigma_y))
    if sigma_x >= sigma_y:
        theta_deg = float(np.rad2deg(theta_rad) % 180)
    else:
        theta_deg = float((np.rad2deg(theta_rad) + 90) % 180)
    kappa = float(max(sigma_x, sigma_y) / min(sigma_x, sigma_y))
    return sigma, theta_deg, kappa


def _pixel_size(stimulus_name):
    """Approximate pixel size in visual degrees for a sparse noise stimulus."""
    if '8deg' in stimulus_name:
        return 9.3
    elif '4deg' in stimulus_name:
        return 4.65
    return 4.65


def _nan_params(dominant, pixel_size_deg):
    return {
        'sigma': np.nan, 'theta': np.nan, 'kappa': np.nan,
        'x0': np.nan, 'y0': np.nan, 'amplitude': np.nan,
        'sigma_x': np.nan, 'sigma_y': np.nan,
        'dominant_subfield': dominant,
    }


def _fail_quality(message):
    return {
        'r_squared': 0.0, 'rmse': np.inf,
        'converged': False, 'message': message,
    }


# ===========================================================================
# Gaussian derivative order classification
# ===========================================================================
#
# Lindeberg (2025) treats V1 simple cells as 2D Gaussian *derivatives* of
# integer order m.  In the rotated frame (u along major axis, v perpendicular):
#
#   G(u,v) = exp( -u²/(2σ_u²) - v²/(2σ_v²) )
#
#   f_0 = A · G                       + b   [m=0, blob]
#   f_1 = A · (u/σ_u) · G             + b   [m=1, edge]
#   f_2 = A · (u²/σ_u² − 1) · G      + b   [m=2, flanked]
#
# All three share 7 parameters; model selection is by R² on the full signed map.
# m=1 and m=2 MUST be fit to rf_on − rf_off (not the dominant lobe alone).
# ===========================================================================


def _uv_coords(xy, x0, y0, theta):
    """Rotate (x, y) into the RF-aligned frame (u along θ, v perpendicular)."""
    x, y  = xy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    u =  (x - x0) * cos_t + (y - y0) * sin_t
    v = -(x - x0) * sin_t + (y - y0) * cos_t
    return u, v


def _gaussian_envelope(u, v, su, sv):
    return np.exp(-u**2 / (2.0 * su**2) - v**2 / (2.0 * sv**2))


def gaussian_deriv_m0(xy, A, x0, y0, su, sv, theta, b):
    """m=0: anisotropic Gaussian blob (plain envelope)."""
    u, v = _uv_coords(xy, x0, y0, theta)
    return (b + A * _gaussian_envelope(u, v, su, sv)).ravel()


def gaussian_deriv_m1(xy, A, x0, y0, su, sv, theta, b):
    """m=1: first Gaussian derivative — two opposite lobes.

    f = A · (u/σ_u) · G(u,v) + b
    A > 0: ON lobe at u > 0, OFF at u < 0.  A < 0: reversed.
    """
    u, v = _uv_coords(xy, x0, y0, theta)
    G    = _gaussian_envelope(u, v, su, sv)
    return (b + A * (u / su) * G).ravel()


def gaussian_deriv_m2(xy, A, x0, y0, su, sv, theta, b):
    """m=2: second Gaussian derivative — central lobe + two flanks.

    f = A · (u²/σ_u² − 1) · G(u,v) + b   [He_2(u/σ_u)]
    A > 0: OFF-centre / ON-flanks.  A < 0: ON-centre / OFF-flanks.
    """
    u, v = _uv_coords(xy, x0, y0, theta)
    G    = _gaussian_envelope(u, v, su, sv)
    He2  = (u / su)**2 - 1.0
    return (b + A * He2 * G).ravel()


# Map order → callable model
_MODEL_FN = {0: gaussian_deriv_m0, 1: gaussian_deriv_m1, 2: gaussian_deriv_m2}


# ===========================================================================
# Display-model orientation: seed-then-refine
# ===========================================================================
#
# The orientation we *report* (phi from moments, theta from the envelope) is
# continuous.  The orientation at which the idealised model panel is *drawn*
# was previously chosen by a coarse 10 deg grid search (correlation of the
# rendered model with the smoothed RF).  That coarse step is what made the
# illustrations look quantised / over-aligned with the coordinate grid.
#
# `best_display_orientation` keeps the coarse grid's job — picking the correct
# basin, robust to optimiser local minima — but then refines the *angle only*
# continuously within that basin, so the drawn map is no longer snapped to
# 10 deg.  It deliberately does NOT re-fit all parameters, which would let the
# optimiser wander back into the local minimum the grid search rescued.
#
# All spatial inputs (x0, y0, sigma_u, sigma_v) are in PIXEL/grid units, to
# match the model functions above.
# ===========================================================================

def _corr(a, b):
    """Pearson correlation of two arrays (flattened, mean-removed)."""
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def render_model_map(rf_smoothed, x0, y0, sigma_u, sigma_v, theta_deg, m):
    """Render a unit-amplitude, zero-baseline Gaussian-derivative model on the
    same grid as ``rf_smoothed``.

    The differentiation axis is ``theta_deg`` with scale ``sigma_u`` along it
    and ``sigma_v`` perpendicular.  Polarity is flipped if needed so the model
    correlates positively with ``rf_smoothed`` (cosmetic, matches the existing
    post-fit polarity convention).
    """
    grid_h, grid_w = rf_smoothed.shape
    Y, X = np.mgrid[0:grid_h, 0:grid_w]
    xy   = (X.astype(float), Y.astype(float))
    th   = np.deg2rad(theta_deg)
    model = _MODEL_FN[m](xy, 1.0, x0, y0, sigma_u, sigma_v, th, 0.0).reshape(grid_h, grid_w)
    if _corr(rf_smoothed, model) < 0:
        model = -model
    return model


def best_display_orientation(rf_smoothed, x0, y0, sigma_u, sigma_v, m,
                             extra_angles_deg=(), coarse_step_deg=10.0,
                             refine=True, refine_window_deg=10.0):
    """Pick the orientation at which to DRAW the idealised model panel.

    Two-stage seed-then-refine (see module note above):
      1. COARSE — scan candidate angles 0..180 in ``coarse_step_deg`` for BOTH
         sigma orderings, plus any ``extra_angles_deg`` (e.g. theta_envelope,
         phi); keep the (angle, ordering) with the highest |correlation|.
      2. REFINE — hold that ordering fixed and refine the ANGLE only by a
         bounded continuous search within +/- ``refine_window_deg`` of the
         coarse winner (0.25 deg tol = visually continuous).

    Parameters
    ----------
    rf_smoothed : ndarray (n_y, n_x)   smoothed RF (the correlation target)
    x0, y0      : float                RF centre in pixel/grid units
    sigma_u, sigma_v : float           the two fitted scales in pixels
    m           : int {0,1,2}
    extra_angles_deg : iterable of float   optional seeds (theta, phi) in deg

    Returns
    -------
    dict: theta_display (deg, continuous), sigma_u_used, sigma_v_used,
          model_map, corr, angle_corr_profile (coarse |corr| for the polar plot)
    """
    orderings = [(sigma_u, sigma_v), (sigma_v, sigma_u)]
    coarse    = np.arange(0.0, 180.0, coarse_step_deg)

    best     = {'corr': -np.inf}
    profiles = {}
    for oi, (su, sv) in enumerate(orderings):
        prof = []
        for ang in coarse:
            c = abs(_corr(rf_smoothed,
                          render_model_map(rf_smoothed, x0, y0, su, sv, ang, m)))
            prof.append(c)
            if c > best['corr']:
                best = {'corr': c, 'angle': float(ang), 'su': su, 'sv': sv, 'oi': oi}
        profiles[oi] = prof

    for ang in extra_angles_deg:                       # theta_env, phi, ...
        if ang is None or (isinstance(ang, float) and np.isnan(ang)):
            continue
        for oi, (su, sv) in enumerate(orderings):
            c = abs(_corr(rf_smoothed,
                          render_model_map(rf_smoothed, x0, y0, su, sv, ang, m)))
            if c > best['corr']:
                best = {'corr': c, 'angle': float(ang) % 180,
                        'su': su, 'sv': sv, 'oi': oi}

    su, sv, a0 = best['su'], best['sv'], best['angle']

    if refine:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(
            lambda ang: -abs(_corr(
                rf_smoothed, render_model_map(rf_smoothed, x0, y0, su, sv, ang, m))),
            bounds=(a0 - refine_window_deg, a0 + refine_window_deg),
            method='bounded', options={'xatol': 0.25},
        )
        a_final = float(res.x) % 180
    else:
        a_final = a0 % 180

    model_map = render_model_map(rf_smoothed, x0, y0, su, sv, a_final, m)
    return {
        'theta_display':      a_final,
        'sigma_u_used':       float(su),
        'sigma_v_used':       float(sv),
        'model_map':          model_map,
        'corr':               _corr(rf_smoothed, model_map),
        'angle_corr_profile': profiles[best['oi']],
    }


def _init_params_for_order(rf_signed, m):
    """Estimate p0 from image moments of |RF| — valid for all orders."""
    grid_h, grid_w = rf_signed.shape
    Y, X = np.mgrid[0:grid_h, 0:grid_w]

    rf_abs = np.abs(rf_signed)
    total  = rf_abs.sum()
    if total < 1e-12:
        return [1.0, grid_w / 2, grid_h / 2, 2.0, 1.0, 0.0, 0.0]

    w       = rf_abs / total
    x0_init = float(np.clip((X * w).sum(), 0, grid_w - 1))
    y0_init = float(np.clip((Y * w).sum(), 0, grid_h - 1))

    dx  = X - x0_init
    dy  = Y - y0_init
    Mxx = (dx**2 * w).sum()
    Myy = (dy**2 * w).sum()
    Mxy = (dx * dy * w).sum()

    evals, evecs = np.linalg.eigh([[Mxx, Mxy], [Mxy, Myy]])
    evals        = np.maximum(evals, 0.25)

    su_init    = float(np.sqrt(evals[1]))
    sv_init    = float(np.sqrt(evals[0]))
    theta_init = float(np.arctan2(evecs[1, 1], evecs[0, 1]))

    baseline = float(np.median(rf_signed))
    if m == 0:
        amp_init = float(rf_signed.max() - baseline)
    elif m == 1:
        amp_init = float(rf_signed.max() - baseline)
    else:
        peak_pos = rf_signed.max() - baseline
        peak_neg = -(rf_signed.min() - baseline)
        amp_init = -float(peak_neg) if peak_neg > peak_pos else float(peak_pos)

    return [amp_init, x0_init, y0_init,
            max(su_init, 0.5), max(sv_init, 0.5),
            theta_init, baseline]


def fit_rf_by_order(rf_on, rf_off, m, pixel_size_deg=4.65, smooth_sigma=0.75,
                    n_random_starts=15, seed=None):
    """Fit a single Gaussian derivative order to the full signed RF map.

    Uses multi-start optimisation to guard against Levenberg–Marquardt
    convergence to local minima.  One moment-based initialisation is always
    run first (from ``_init_params_for_order``), followed by
    ``n_random_starts`` random initialisations that sample the parameter
    space more broadly.  The start yielding the highest R² is returned.

    The trust-region reflective (TRF) solver is used throughout because it
    handles box constraints natively and is more robust than the default
    unconstrained LM when sigma bounds are active.

    Parameters
    ----------
    rf_on, rf_off : ndarray (n_y, n_x)
    m : int {0, 1, 2}
    pixel_size_deg : float
    smooth_sigma : float
    n_random_starts : int, default=15
        Number of random initialisations in addition to the moment-based one.
        Set to 0 to disable multi-start (faster, but more prone to local minima).
    seed : int or None, default=None
        Random seed for reproducible random starts.

    Returns
    -------
    params : dict
        sigma, theta, kappa, x0, y0, amplitude, sigma_x, sigma_y,
        dominant_subfield, derivative_order,
        n_starts_succeeded, r2_improvement.
    quality : dict
        r_squared, rmse, converged, message, aic, n_pixels.

    Notes
    -----
    ``r2_improvement`` is the difference between the best R² found across all
    starts and the R² of the moment-based init alone.  A large positive value
    (> 0.05) indicates the moment basin was a local minimum — flag these
    neurons for visual inspection.
    """
    if m not in (0, 1, 2):
        raise ValueError(f"Derivative order must be 0, 1 or 2; got {m}")

    rng = np.random.default_rng(seed)

    rf_signed = rf_on - rf_off
    grid_h, grid_w = rf_signed.shape

    if smooth_sigma > 0:
        rf_work = gaussian_filter(rf_signed, sigma=smooth_sigma)
    else:
        rf_work = rf_signed.copy()

    Y, X    = np.mgrid[0:grid_h, 0:grid_w]
    xy_data = (X.astype(float), Y.astype(float))
    if np.abs(rf_work).max() < 1e-12:
        dominant = 'ON' if rf_on.max() >= rf_off.max() else 'OFF'
        return _nan_params(dominant, pixel_size_deg), _fail_quality("No signal")

    model_fn = _MODEL_FN[m]

    # -- m=0 polarity fix ------------------------------------------------
    # For m=0 (Gaussian blob) A > 0 by convention (G is always positive).
    # If the dominant blob is *negative* (OFF-centre cell), the A >= 0
    # constraint forces the optimizer to fit a positive Gaussian to a
    # negative map -- it collapses to near-zero amplitude regardless of
    # multi-start, yielding R^2 ~ 0 and making m=0 look artificially worse
    # than m=1/2 (which have unconstrained A).  Fix: pre-flip the signed
    # map for m=0 so the dominant blob is always positive before fitting.
    # sigma/theta/kappa are polarity-invariant; dominant_subfield records
    # the true polarity.
    if m == 0:
        rf_fit = rf_work if abs(rf_work.max()) >= abs(rf_work.min()) else -rf_work
        lower_A, upper_A = 0.0, np.inf
    else:
        rf_fit = rf_work
        lower_A, upper_A = -np.inf, np.inf

    z_data = rf_fit.ravel().astype(float)
    ss_tot = float(np.sum((z_data - z_data.mean()) ** 2))

    sig_min   = 0.3
    sig_max_u = grid_w / 2.0
    sig_max_v = grid_h / 2.0

    lower = [lower_A, -2,         -2,          sig_min, sig_min, -np.pi, -np.inf]
    upper = [upper_A, grid_w + 2, grid_h + 2,
             sig_max_u, sig_max_v, np.pi, np.inf]

    # ── moment-based initialisation ──────────────────────────────────
    moment_p0 = _init_params_for_order(rf_fit, m)

    # ── random initialisations ────────────────────────────────────────
    def _random_p0():
        x0r  = rng.uniform(grid_w * 0.1, grid_w * 0.9)
        y0r  = rng.uniform(grid_h * 0.1, grid_h * 0.9)
        sur  = np.exp(rng.uniform(np.log(sig_min), np.log(sig_max_u / 2)))
        svr  = np.exp(rng.uniform(np.log(sig_min), np.log(sig_max_v / 2)))
        thr  = rng.uniform(-np.pi, np.pi)
        sign = rng.choice([-1.0, 1.0]) if m > 0 else 1.0
        Ar   = sign * float(np.abs(rf_fit).max())
        br   = float(np.median(rf_fit))
        return [Ar, x0r, y0r, sur, svr, thr, br]

    all_p0s = [moment_p0] + [_random_p0() for _ in range(n_random_starts)]


    # ── run all starts, track best by R² ─────────────────────────────────
    best_r2     = -np.inf
    best_popt   = None
    moment_r2   = None
    n_succeeded = 0

    for i, p0 in enumerate(all_p0s):
        try:
            popt, _ = curve_fit(
                model_fn, xy_data, z_data,
                p0=p0, bounds=(lower, upper),
                maxfev=15000, method='trf',
            )
        except (RuntimeError, ValueError):
            continue

        ss_res = float(np.sum((z_data - model_fn(xy_data, *popt)) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        n_succeeded += 1

        if i == 0:
            moment_r2 = r2

        if r2 > best_r2:
            best_r2   = r2
            best_popt = popt

    # ── fall back to moment p0 if every start diverged ───────────────────
    converged = best_popt is not None
    if not converged:
        best_popt = np.array(moment_p0, dtype=float)
        best_r2   = 0.0
        message   = "All initialisations failed to converge"
    else:
        r2_improvement = best_r2 - (moment_r2 if moment_r2 is not None else best_r2)
        if r2_improvement > 0.05:
            message = (
                f"Multi-start improved R² by {r2_improvement:+.3f} — "
                f"moment basin was a local minimum."
            )
        else:
            message = f"Converged ({n_succeeded}/{len(all_p0s)} starts succeeded)."

    r2_improvement = (
        best_r2 - moment_r2
        if (moment_r2 is not None and converged)
        else 0.0
    )

    # ── unpack best-fit parameters ────────────────────────────────────────
    A, x0, y0, su, sv, theta_rad, b = best_popt

    # Capture the RAW differentiation-frame scales BEFORE the major/minor swap
    # below.  In the m=1/m=2 model the derivative prefactor is (u/su) with u
    # along theta_rad, so:
    #   su = scale ALONG the differentiation axis      = sigma_phi
    #   sv = scale PERPENDICULAR to it                 = sigma_orth
    # Tony's direction-aware elongation keeps the sign of which axis is larger:
    #   kappa_dir = sigma_orth / sigma_phi = sv / su
    #     > 1  -> elongated perpendicular to differentiation (Hildegard case)
    #     < 1  -> elongated along the differentiation axis   (Wendell case)
    # This is the (un-swapped) signed counterpart of the max/min `kappa` below.
    # Meaningful only for m>=1; m=0 has no differentiation direction.
    su_raw, sv_raw = float(su), float(sv)
    if m >= 1 and su_raw > 0:
        kappa_dir      = sv_raw / su_raw
        sigma_phi_deg  = su_raw * pixel_size_deg
        sigma_orth_deg = sv_raw * pixel_size_deg
    else:
        kappa_dir      = float('nan')
        sigma_phi_deg  = float('nan')
        sigma_orth_deg = float('nan')

    sigma_px  = float(np.sqrt(su * sv))
    kappa     = float(su / sv) if su >= sv else float(sv / su)
    sigma_deg = sigma_px * pixel_size_deg

    if su >= sv:
        theta_deg = float(np.rad2deg(theta_rad) % 180)
    else:
        theta_deg = float((np.rad2deg(theta_rad) + 90) % 180)
        su, sv    = sv, su

    # ── post-fit polarity correction (m>0 only) ───────────────────────────
    # Theta normalisation maps theta_rad into [0°,180°), which for odd-order
    # derivatives can implicitly flip the model polarity (rotating by π is
    # equivalent to negating A for m=1).  Re-evaluate the model with the
    # *normalised* theta and check correlation with the data; if negative,
    # flip A so that params['amplitude'] is always polarity-consistent with
    # the raw signed RF.  σ, κ, θ are unchanged — this is a display fix.
    if m > 0:
        popt_norm  = [A, x0, y0, su, sv, np.deg2rad(theta_deg), float(b)]
        model_norm = model_fn(xy_data, *popt_norm)
        b_val      = float(b)
        if np.dot(z_data - b_val, model_norm - b_val) < 0:
            A = -A

    dominant = 'ON' if rf_on.max() >= rf_off.max() else 'OFF'

    params = {
        'sigma':              sigma_deg,
        'theta':              theta_deg,
        'kappa':              kappa,           # max/min, always >= 1
        'kappa_dir':          kappa_dir,       # sigma_orth/sigma_phi, signed (may be < 1)
        'sigma_phi':          sigma_phi_deg,   # scale along differentiation axis (deg)
        'sigma_orth':         sigma_orth_deg,  # scale perpendicular to it (deg)
        'x0':                 float(x0 * pixel_size_deg),
        'y0':                 float(y0 * pixel_size_deg),
        'amplitude':          float(A),
        'sigma_x':            float(su * pixel_size_deg),
        'sigma_y':            float(sv * pixel_size_deg),
        'dominant_subfield':  dominant,
        'derivative_order':   m,
        'n_starts_succeeded': n_succeeded,
        'r2_improvement':     float(r2_improvement),
    }

    fitted    = model_fn(xy_data, *best_popt)
    n_pix     = len(z_data)
    ss_res    = float(np.sum((z_data - fitted) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse      = float(np.sqrt(ss_res / n_pix))
    aic       = n_pix * np.log(ss_res / n_pix + 1e-30) + 2 * 7

    quality = {
        'r_squared': float(r_squared),
        'rmse':      rmse,
        'converged': converged,
        'message':   message,
        'aic':       float(aic),
        'n_pixels':  n_pix,
    }

    return params, quality


# ===========================================================================
# φ from lobe geometry  (order-appropriate orientation estimator)
# ===========================================================================

def estimate_phi_from_lobes(rf_on, rf_off, m, pixel_size_deg=4.65,
                             smooth_sigma=0.75):
    """Estimate orientation φ from lobe spatial positions for m=1 and m=2.

    This is the theoretically preferred orientation estimator for m≥1 because
    it uses the *positions* of the lobes rather than the shape of the
    envelope.  For m=1 the envelope is nearly circular (κ ≈ 1), making the
    ellipse major axis (θ) unreliable; φ from lobe centroids is well-defined
    even for perfectly isotropic envelopes.

    For m=0 this function returns (None, 0.0) — no lobes exist so lobe
    geometry is undefined.  Use θ from the envelope fit instead.

    Parameters
    ----------
    rf_on, rf_off : ndarray (n_y, n_x)
        Subfield maps.
    m : int {0, 1, 2}
        Derivative order of this neuron (from classify_rf_order).
    pixel_size_deg : float
        Degrees per pixel — used to convert pixel distances to degrees.
    smooth_sigma : float
        Pre-smoothing in pixels (same value used in fitting).

    Returns
    -------
    phi_deg : float or None
        Differentiation direction in [0, 180°), or None if m=0 or
        estimation fails (e.g. no signal in one polarity).
    phi_confidence : float
        Lobe separation / spread in visual degrees.  Larger = more reliable.
        Use as a quality weight; typical useful threshold ≈ σ of the neuron.
        0.0 if m=0 or failed.

    Notes
    -----
    m=1: phi = direction of maximum antisymmetry, estimated from the first-order
         signed spatial moments of the smoothed RF (same smooth_sigma as fit).
         For a pure m=1 Gaussian derivative this is analytically exact.
         phi_confidence = normalised moment magnitude in visual degrees;
         values > ~5 deg are reliable.

    m=2: phi = principal axis of the two same-sign flank regions.
         Flanks are the majority-sign pixels; principal axis from 2nd-moment
         eigendecomposition of their spatial distribution.
         phi_confidence = RMS spread of flanks along principal axis × pixel_size_deg.
    """
    if m == 0:
        return None, 0.0

    rf_signed = rf_on - rf_off
    if smooth_sigma > 0:
        rf_signed = gaussian_filter(rf_signed, sigma=smooth_sigma)

    if np.abs(rf_signed).max() < 1e-12:
        return None, 0.0

    grid_h, grid_w = rf_signed.shape
    Y, X = np.mgrid[0:grid_h, 0:grid_w]

    if m == 1:
        # ── First-order signed spatial moments ───────────────────────────────
        # More robust than centroid-separation for noisy, low-pixel-count RFs.
        # Every pixel contributes in proportion to its signed response, so noise
        # near zero has negligible influence while genuine lobe signal dominates.
        # Derivation: for a pure m=1 map f = A*(u/sigma_u)*G, the signed moments
        # reduce analytically to the differentiation axis, making this estimator
        # exact in the noiseless limit regardless of kappa.
        #
        # Use the same smoothing as the fit (smooth_sigma=0.75 by default).
        # Heavier smoothing was tried (floor at 1.5) but destroys the dipole
        # on small RFs in the 9x16 sparse-noise grid -- the sigma=0.75
        # smoothed RF is where the dipole is actually visible, so moments
        # must be computed at the same scale.
        sigma_m = smooth_sigma
        rf_m = rf_signed   # already smoothed at smooth_sigma above

        # Threshold mask: only include pixels with |rf| > 15% of peak.
        # On small (9x16) noisy grids background noise dominates the signed
        # moments and overwhelms the genuine dipole signal.  Masking to
        # signal-rich pixels near the lobe cores makes the estimator robust.
        peak = float(np.abs(rf_m).max())
        if peak < 1e-12:
            return None, 0.0
        rf_m = np.where(np.abs(rf_m) > 0.15 * peak, rf_m, 0.0)

        abs_total = float(np.abs(rf_m).sum())
        if abs_total < 1e-12:
            return None, 0.0

        # Centre at the |rf|-weighted centroid.
        # Matters for the non-square 9x16 sparse-noise grid where the geometric
        # centre differs from the RF centre.
        w   = np.abs(rf_m) / abs_total
        x_c = float((X * w).sum())
        y_c = float((Y * w).sum())

        # Signed first moments = direction of maximum antisymmetry
        mx = float((rf_m * (X - x_c)).sum())
        my = float((rf_m * (Y - y_c)).sum())

        mag = float(np.sqrt(mx**2 + my**2))
        if mag < 1e-12:
            return None, 0.0

        phi_deg = float(np.rad2deg(np.arctan2(my, mx)) % 360)

        # Confidence: mean absolute projection of RF onto the dipole axis,
        # in visual degrees.  Reliable dipoles give values ~sigma_deg of the cell
        # (typically 5–40°); near-zero indicates no detectable dipole structure.
        # Rule of thumb: > ~5° = reliable.
        phi_confidence = (mag / (abs_total + 1e-12)) * pixel_size_deg

        return phi_deg, phi_confidence

    else:  # m == 2
        # ── principal axis of the two same-sign flanking regions ────────────
        # Flanks = majority-sign pixels (minority sign is the central lobe).
        pos_sum = float(rf_signed[rf_signed > 0].sum())
        neg_sum = float((-rf_signed)[rf_signed < 0].sum())

        if pos_sum >= neg_sum:
            flank_map = np.maximum(rf_signed, 0.0)   # flanks are positive
        else:
            flank_map = np.maximum(-rf_signed, 0.0)  # flanks are negative

        total = flank_map.sum()
        if total < 1e-12:
            return None, 0.0

        # Weighted centroid of flanks
        w   = flank_map / total
        x_c = float((X * w).sum())
        y_c = float((Y * w).sum())

        # Second moments → principal axis
        dx  = X - x_c
        dy  = Y - y_c
        Mxx = float((dx**2 * w).sum())
        Myy = float((dy**2 * w).sum())
        Mxy = float((dx * dy * w).sum())

        _, evecs = np.linalg.eigh([[Mxx, Mxy], [Mxy, Myy]])
        # Major eigenvector (index 1) = axis along which flanks are spread = φ
        phi_deg = float(np.rad2deg(np.arctan2(evecs[1, 1], evecs[0, 1])) % 180)

        # Confidence = RMS spread of flanks along φ axis
        proj   = dx * evecs[0, 1] + dy * evecs[1, 1]
        spread = float(np.sqrt((proj**2 * w).sum()))
        phi_confidence = spread * pixel_size_deg

        return phi_deg, phi_confidence


# ===========================================================================
# classify_rf_order — best-model selection + hybrid orientation
# ===========================================================================

def classify_rf_order(rf_on, rf_off, pixel_size_deg=4.65, smooth_sigma=0.75,
                      orders=(0, 1, 2), delta_r2_threshold=0.05,
                      n_random_starts=15, seed=None):
    """Fit derivative orders m=0, 1, 2 and select the best-fitting model.

    Model selection uses a **parsimonious upgrade rule**: the raw best order
    by R² is accepted only if it improves over m=0 by at least
    ``delta_r2_threshold``.  If the improvement is marginal, the neuron is
    classified as m=0 (simpler model preferred).  This prevents noisy low-SNR
    maps from being spuriously assigned m=1 or m=2 when the R² gain is driven
    by fitting noise rather than genuine lobe structure.

    Orientation is handled by two estimators recorded separately:

      theta         — major axis of the fitted Gaussian envelope (all orders).
                      Unreliable for m=1 because κ ≈ 1 in that case.

      phi           — differentiation direction from lobe geometry (m=1, m=2).
                      NaN for m=0 (no lobes).  See estimate_phi_from_lobes().

      theta_hybrid  — recommended single orientation column:
                      phi  for m=1 and m=2 (lobe geometry, more reliable);
                      theta for m=0 (envelope, only available estimator).

    Parameters
    ----------
    rf_on, rf_off : ndarray (n_y, n_x)
    pixel_size_deg : float
    smooth_sigma : float
    orders : tuple of int
    delta_r2_threshold : float, default=0.05
        Minimum R² improvement over m=0 required to accept a higher-order
        model.  Only applied when 0 is in ``orders``.
        Setting to 0.0 restores the previous pure-maximum-R² behaviour.
    n_random_starts : int, default=15
        Passed to ``fit_rf_by_order``.  Number of random initialisations in
        addition to the moment-based one.  Set to 0 to disable multi-start.
    seed : int or None, default=None
        Random seed for reproducible random starts.  Each order receives a
        deterministic child seed derived from this value.

    Returns
    -------
    best_order : int
    best_params : dict
        Lindeberg params from the winning fit, plus:
          'phi'           — lobe-geometry orientation: [0,360°) for m=1,
                            [0,180°) for m=2, NaN for m=0
          'phi_confidence'— lobe separation / spread in degrees
          'theta_hybrid'  — phi if m≥1, theta if m=0
          'delta_r2_vs_m0'— R²(best) − R²(m=0); NaN if m=0 not evaluated
    best_quality : dict
    all_results : dict
        {m: {'params': ..., 'quality': ...}} for every evaluated order.
    """
    # Derive fixed child seeds so results are reproducible regardless of
    # the order in which models are evaluated.
    rng = np.random.default_rng(seed)
    order_seeds = {m: int(rng.integers(0, 2**31)) for m in orders}

    all_results = {}
    for m in orders:
        params, quality = fit_rf_by_order(
            rf_on, rf_off, m=m,
            pixel_size_deg=pixel_size_deg,
            smooth_sigma=smooth_sigma,
            n_random_starts=n_random_starts,
            seed=order_seeds[m],
        )
        all_results[m] = {'params': params, 'quality': quality}

    # ── parsimonious model selection ─────────────────────────────────────
    # Raw best by R²
    best_raw = max(orders, key=lambda m: all_results[m]['quality']['r_squared'])

    if best_raw != 0 and 0 in orders and delta_r2_threshold > 0:
        r2_best = all_results[best_raw]['quality']['r_squared']
        r2_m0   = all_results[0]['quality']['r_squared']
        if r2_best - r2_m0 < delta_r2_threshold:
            # Marginal improvement — prefer the parsimonious m=0
            best_order = 0
        else:
            best_order = best_raw
    else:
        best_order = best_raw

    # Compute ΔR² vs m=0 for reporting (NaN if m=0 not evaluated)
    if 0 in orders:
        delta_r2_vs_m0 = (
            all_results[best_order]['quality']['r_squared']
            - all_results[0]['quality']['r_squared']
        )
    else:
        delta_r2_vs_m0 = float('nan')

    best_params  = dict(all_results[best_order]['params'])   # copy
    best_quality = all_results[best_order]['quality']
    best_params['delta_r2_vs_m0'] = delta_r2_vs_m0

    # ── orientation: compute φ from lobe geometry for m=1 and m=2 ──────────
    phi, phi_conf = estimate_phi_from_lobes(
        rf_on, rf_off,
        m=best_order,
        pixel_size_deg=pixel_size_deg,
        smooth_sigma=smooth_sigma,
    )

    best_params['phi']            = phi if phi is not None else np.nan
    best_params['phi_confidence'] = phi_conf

    # theta_hybrid: φ where available (m≥1), θ from envelope for m=0
    # Fixed — φ only trusted for m=1:
    if phi is not None and best_order == 1:
        best_params['theta_hybrid'] = phi
    else:
        best_params['theta_hybrid'] = best_params['theta']

    return best_order, best_params, best_quality, all_results
