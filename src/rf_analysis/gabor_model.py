"""
Gabor comparison model for the m=1 simple-cell analysis.

Companion to ``sparse_noise.py``.  The Gaussian derivative side of the
comparison uses ``sparse_noise.fit_rf_by_order`` unchanged; this module supplies
only the Gabor, written to mirror that function line for line so the two
families are optimised identically and any difference between them is a
difference between models rather than between fitting procedures.

Everything that ``fit_rf_by_order`` does, ``fit_gabor`` does the same way:

  * the same signed map, ``rf_on - rf_off``, pre-smoothed by ``smooth_sigma``;
  * the same ``(X, Y)`` pixel grid and the same ``xy``-tuple / ``.ravel()``
    model signature;
  * the same moment-based initialisation from
    ``sparse_noise._init_params_for_order(rf_fit, 1)``, extended with the two
    carrier parameters;
  * the same random-start scheme, including the ``sign = rng.choice([-1, 1])``
    amplitude draw, extended with the two carrier parameters;
  * unconstrained amplitude, as ``fit_rf_by_order`` uses for m >= 1 — see the
    polarity note in that function;
  * the same bounds on the seven shared parameters;
  * the same ``curve_fit(..., method='trf', maxfev=15000)`` call and the same
    best-by-R-squared selection with a moment-basin improvement diagnostic.

The only differences are the two extra parameters, carrier spatial frequency
``f`` (cycles per pixel) and carrier phase ``psi``, and the AIC parameter count,
which is 9 rather than 7.

This is the ALIGNED Gabor: the carrier runs along ``u``, the same axis along
which the Gaussian derivative's lobes run.  Both families are therefore
restricted in the same way, and neither is given a shape the other cannot make.
The full affine forms, which let the carrier or the lobes run at an angle to
the envelope, are not fitted here.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

from .sparse_noise import (
    _uv_coords,
    _gaussian_envelope,
    _init_params_for_order,
    _nan_params,
    _fail_quality,
)

# Carrier frequency bounds, cycles per stimulus pixel.  The upper bound is the
# Nyquist limit of the stimulus grid; above it the carrier is not resolvable and
# the Gabor would be fitting pixel noise.
F_MIN = 1e-3
F_MAX = 0.5

K_GABOR = 9   # A, x0, y0, su, sv, theta, b, f, psi


def gabor_2d(xy, A, x0, y0, su, sv, theta, b, f, psi):
    """Aligned 2D Gabor.

    f = b + A * cos(2*pi*f*u + psi) * G(u, v)

    Same envelope and same rotated frame as ``sparse_noise.gaussian_deriv_m1``;
    the carrier runs along u, which is the differentiation axis of the Gaussian
    derivative.  A < 0 is equivalent to a phase shift of pi, so amplitude is
    left unconstrained exactly as it is for m >= 1 in ``fit_rf_by_order``.
    """
    u, v = _uv_coords(xy, x0, y0, theta)
    G = _gaussian_envelope(u, v, su, sv)
    return (b + A * np.cos(2.0 * np.pi * f * u + psi) * G).ravel()


def fit_gabor(rf_on, rf_off, pixel_size_deg=4.65, smooth_sigma=0.75,
              n_random_starts=15, seed=None):
    """Fit an aligned 2D Gabor to the full signed RF map.

    Mirrors ``sparse_noise.fit_rf_by_order(..., m=1, ...)`` in every respect
    except the two carrier parameters and the AIC parameter count.

    Parameters
    ----------
    rf_on, rf_off : ndarray (n_y, n_x)
    pixel_size_deg : float
    smooth_sigma : float
    n_random_starts : int, default=15
        As in ``fit_rf_by_order``: total starts are this plus one moment-based
        initialisation, so the default of 15 gives 16 starts.
    seed : int or None
        Pass the same seed used for the Gaussian derivative fit so the two
        families draw their random starts from the same stream position.

    Returns
    -------
    params : dict
        sigma, theta, kappa, kappa_dir, sigma_phi, sigma_orth, x0, y0,
        amplitude, sigma_x, sigma_y, carrier_freq_cpd, carrier_period_deg,
        phase, dominant_subfield, n_starts_succeeded, r2_improvement.
    quality : dict
        r_squared, rmse, converged, message, aic, n_pixels, n_params.
    """
    rng = np.random.default_rng(seed)

    rf_signed = rf_on - rf_off
    grid_h, grid_w = rf_signed.shape

    rf_work = (gaussian_filter(rf_signed, sigma=smooth_sigma)
               if smooth_sigma > 0 else rf_signed.copy())

    Y, X = np.mgrid[0:grid_h, 0:grid_w]
    xy_data = (X.astype(float), Y.astype(float))

    if np.abs(rf_work).max() < 1e-12:
        dominant = 'ON' if rf_on.max() >= rf_off.max() else 'OFF'
        return _nan_params(dominant, pixel_size_deg), _fail_quality("No signal")

    # Amplitude unconstrained, as fit_rf_by_order does for m >= 1.
    rf_fit = rf_work
    z_data = rf_fit.ravel().astype(float)
    ss_tot = float(np.sum((z_data - z_data.mean()) ** 2))

    sig_min   = 0.3
    sig_max_u = grid_w / 2.0
    sig_max_v = grid_h / 2.0

    lower = [-np.inf, -2,         -2,         sig_min,   sig_min,   -np.pi, -np.inf,
             F_MIN, -np.pi]
    upper = [ np.inf, grid_w + 2, grid_h + 2, sig_max_u, sig_max_v,  np.pi,  np.inf,
             F_MAX,  np.pi]

    # -- moment-based initialisation ---------------------------------------
    # The seven shared parameters come from the same estimator the Gaussian
    # derivative uses.  The carrier period is seeded to match the lobe spacing
    # of the equivalent first-order derivative, whose lobes peak at u = +/- su,
    # and the phase to the antisymmetric case, which is the Gabor closest to a
    # first derivative.  Seeding the carrier at a HIGH frequency is what made
    # the superseded comparison fail.
    moment_p0 = list(_init_params_for_order(rf_fit, 1))
    su_init = moment_p0[3]
    moment_p0 += [float(np.clip(1.0 / (4.0 * su_init), F_MIN, F_MAX)), -np.pi / 2.0]

    # -- random initialisations --------------------------------------------
    def _random_p0():
        x0r  = rng.uniform(grid_w * 0.1, grid_w * 0.9)
        y0r  = rng.uniform(grid_h * 0.1, grid_h * 0.9)
        sur  = np.exp(rng.uniform(np.log(sig_min), np.log(sig_max_u / 2)))
        svr  = np.exp(rng.uniform(np.log(sig_min), np.log(sig_max_v / 2)))
        thr  = rng.uniform(-np.pi, np.pi)
        sign = rng.choice([-1.0, 1.0])
        Ar   = sign * float(np.abs(rf_fit).max())
        br   = float(np.median(rf_fit))
        fr   = np.exp(rng.uniform(np.log(1.0 / (4.0 * sig_max_u)), np.log(F_MAX)))
        psir = rng.uniform(-np.pi, np.pi)
        return [Ar, x0r, y0r, sur, svr, thr, br, fr, psir]

    all_p0s = [moment_p0] + [_random_p0() for _ in range(n_random_starts)]

    # -- run all starts, track best by R-squared ----------------------------
    best_r2, best_popt, moment_r2, n_succeeded = -np.inf, None, None, 0

    for i, p0 in enumerate(all_p0s):
        try:
            popt, _ = curve_fit(
                gabor_2d, xy_data, z_data,
                p0=p0, bounds=(lower, upper),
                maxfev=15000, method='trf',
            )
        except (RuntimeError, ValueError):
            continue

        ss_res = float(np.sum((z_data - gabor_2d(xy_data, *popt)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        n_succeeded += 1

        if i == 0:
            moment_r2 = r2
        if r2 > best_r2:
            best_r2, best_popt = r2, popt

    converged = best_popt is not None
    if not converged:
        best_popt = np.array(moment_p0, dtype=float)
        best_r2   = 0.0
        message   = "All initialisations failed to converge"
        r2_improvement = 0.0
    else:
        r2_improvement = best_r2 - (moment_r2 if moment_r2 is not None else best_r2)
        message = (
            f"Multi-start improved R² by {r2_improvement:+.3f} — "
            f"moment basin was a local minimum."
            if r2_improvement > 0.05 else
            f"Converged ({n_succeeded}/{len(all_p0s)} starts succeeded)."
        )

    A, x0, y0, su, sv, theta_rad, b, f, psi = best_popt

    # Same raw-then-swap convention as fit_rf_by_order: su is the scale along
    # the carrier axis (the counterpart of sigma_phi), sv perpendicular to it.
    su_raw, sv_raw = float(su), float(sv)
    kappa_dir      = sv_raw / su_raw if su_raw > 0 else float('nan')
    sigma_phi_deg  = su_raw * pixel_size_deg
    sigma_orth_deg = sv_raw * pixel_size_deg

    sigma_deg = float(np.sqrt(su * sv)) * pixel_size_deg
    kappa     = float(su / sv) if su >= sv else float(sv / su)

    if su >= sv:
        theta_deg = float(np.rad2deg(theta_rad) % 180)
    else:
        theta_deg = float((np.rad2deg(theta_rad) + 90) % 180)
        su, sv    = sv, su

    dominant = 'ON' if rf_on.max() >= rf_off.max() else 'OFF'

    f_cpd = float(f) / pixel_size_deg           # cycles per degree
    params = {
        'sigma':              sigma_deg,
        'theta':              theta_deg,
        'kappa':              kappa,
        'kappa_dir':          kappa_dir,
        'sigma_phi':          sigma_phi_deg,
        'sigma_orth':         sigma_orth_deg,
        'x0':                 float(x0 * pixel_size_deg),
        'y0':                 float(y0 * pixel_size_deg),
        'amplitude':          float(A),
        'sigma_x':            float(su * pixel_size_deg),
        'sigma_y':            float(sv * pixel_size_deg),
        'carrier_freq_cpd':   f_cpd,
        'carrier_period_deg': float(1.0 / f_cpd) if f_cpd > 0 else float('nan'),
        'phase':              float(np.rad2deg(psi)),
        'dominant_subfield':  dominant,
        'model':              'gabor',
        'n_starts_succeeded': n_succeeded,
        'r2_improvement':     float(r2_improvement),
    }

    fitted    = gabor_2d(xy_data, *best_popt)
    n_pix     = len(z_data)
    ss_res    = float(np.sum((z_data - fitted) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse      = float(np.sqrt(ss_res / n_pix))
    # Same form as fit_rf_by_order, with k = 9 rather than 7.
    aic       = n_pix * np.log(ss_res / n_pix + 1e-30) + 2 * K_GABOR

    quality = {
        'r_squared': float(r_squared),
        'rmse':      rmse,
        'converged': converged,
        'message':   message,
        'aic':       float(aic),
        'n_pixels':  n_pix,
        'n_params':  K_GABOR,
    }

    return params, quality
