# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 10:08:50 2026

@author: amirt
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit

@dataclass
class TTCAnalyzer:
    """
    Utility class for extracting log-binned g2 curves from a TTC (time-time correlation) matrix.

    Parameters
    ----------
    eps : float
        Small value added to std to avoid division-by-zero in weighted averaging.
    """
    eps: float = 1e-15

    # ---------- Simple model functions (kept because they were in your notebook) ----------

    @staticmethod
    def linear(x: np.ndarray, m: float, c: float) -> np.ndarray:
        """Linear function: m*x + c."""
        return m * x + c

    @staticmethod
    def exp_fun(x: np.ndarray, beta: float, A: float, tau: float, kww: float) -> np.ndarray:
        """
        Exponential/KWW-style function often used for g2 fitting:
        g2(x) = A + beta * exp( -2 * (x/tau)^kww )
        """
        return A + beta * np.exp(-2 * (x / tau) ** kww)

    # ---------- Core helpers ----------

    @staticmethod
    def weighted_nanaverage(A: np.ndarray, weights: np.ndarray, axis=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Weighted mean ignoring NaNs.
        Returns (mean, err) where err follows: 1/sqrt(sum(weights over valid points)).
        """
        valid = ~np.isnan(A)
        wsum = np.nansum(valid * weights, axis=axis)
        mean = np.nansum(A * weights, axis=axis) / wsum
        err = 1.0 / np.sqrt(wsum)
        return mean, err

    @staticmethod
    def ttc_vertical(ttc: np.ndarray, t_delay: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert TTC into a "vertical" representation:
        - each row i contains diagonal(ttc, offset=i)
        - remove self-correlation (t_delay = 0) by dropping first row/col

        Returns
        -------
        ttc_v : np.ndarray
            (N-1, N-1) array with NaNs where diagonals run out.
        time_us : np.ndarray
            delay time axis in microseconds, excluding 0
        """
        n = ttc.shape[0]
        ttc_v = np.full((n, n), np.nan, dtype=float)

        for i in range(n):
            m = n - i
            ttc_v[i, 0:m] = np.diagonal(ttc, offset=i)

        time = np.arange(0, n) * t_delay  # in same unit as t_delay
        # remove self correlation (delay=0)
        return ttc_v[1:, :-1], time[1:] / 1e6  # convert to microseconds

    @staticmethod
    def logbin_tdelay(t: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build log-spaced binning indices for a delay-time vector t.

        Input
        -----
        t : array
            delay times (must be positive increasing; first element should be >0)
        n : int
            approximate number of log bins

        Output
        ------
        log_binned : np.ndarray
            indices into the original t array, defining bin edges in index-space.
        t_binned : np.ndarray
            mean t per bin
        dt_binned : np.ndarray
            std(t) per bin
        """
        if len(t) < 2:
            raise ValueError("t must contain at least two time points for log-binning.")
        if t[0] <= 0:
            raise ValueError("t[0] must be > 0 for logspace binning.")

        # log edges across the time range
        bins = np.logspace(np.log10(t[0] - 1e-8), np.log10(t[-1]), n)

        t_mean = binned_statistic(t, t, statistic="mean", bins=bins)
        t_std = binned_statistic(t, t, statistic="std", bins=bins)

        t_binned = t_mean.statistic
        dt_binned = t_std.statistic

        binnumber = t_mean.binnumber
        unique, counts = np.unique(binnumber, return_counts=True)

        # Convert "how many points fall in each bin" into cumulative indices
        log_binned = [0]
        C = 0
        for ct in counts:
            C += ct
            log_binned.append(C)

        # remove NaNs that come from empty bins
        mask = ~np.isnan(t_binned)
        t_binned = t_binned[mask]
        dt_binned = dt_binned[mask]

        return np.array(log_binned, dtype=int), t_binned, dt_binned

    # ---------- Main public API ----------

    def g2_section_area(
        self,
        ttc: np.ndarray,
        std: np.ndarray,
        t_delay: float,
        num: int,
        log_avg_n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Weighted version (uses std as uncertainty, weights = 1/std^2).

        Slices the TTC into 'num' horizontal waiting-time sections, then for each section:
        - creates log-spaced bins in delay time
        - averages g2 over a 2D area (delay range × waiting-time range)
        - uses weighted average to produce g2(t_delay) with uncertainties

        Returns
        -------
        tdelay_section, dtdelay_section, g2_cuts, g2_cuts_err, t_w
        """
        ttc_v, time_us = self.ttc_vertical(ttc, t_delay)
        std_ttc_v, _ = self.ttc_vertical(std, t_delay)

        if num < 1:
            raise ValueError("num must be >= 1.")
        if log_avg_n < 2:
            raise ValueError("log_avg_n must be >= 2.")

        no_lines = max(int(ttc_v.shape[1] / num), 1)
        section_edges = np.asarray(np.arange(0, ttc_v.shape[1], no_lines), dtype=int)
        section_edges = section_edges[: num + 1]

        if num == 1:
            section_edges = np.array([0, ttc_v.shape[1]], dtype=int)

        # We'll allocate after we know how many points in first computed cut
        g2_cuts = g2_cuts_err = tdelay_section = dtdelay_section = None

        for i, ii in enumerate(section_edges):
            if i == len(section_edges) - 1:
                break

            # determine valid length in this diagonal column
            g2_length = np.count_nonzero(~np.isnan(ttc_v[:, ii]))
            if g2_length < 2:
                continue

            log_idx, tdelay_cut, dtdelay_cut = self.logbin_tdelay(time_us[0:g2_length], log_avg_n)

            g2_points, dg2_points = [], []
            for j in range(len(log_idx) - 1):
                a, b = log_idx[j], log_idx[j + 1]
                g2_area = ttc_v[a:b, ii: section_edges[i + 1]]
                std_area = std_ttc_v[a:b, ii: section_edges[i + 1]]

                weights = 1.0 / np.power(std_area + self.eps, 2)
                g2_point, dg2_point = self.weighted_nanaverage(g2_area, weights, axis=None)

                g2_points.append(g2_point)
                dg2_points.append(dg2_point)

            g2_points = np.array(g2_points, dtype=float)
            dg2_points = np.array(dg2_points, dtype=float)

            if g2_cuts is None:
                # allocate full arrays (num sections × len(g2_points))
                g2_cuts = np.full((num, len(g2_points)), np.nan, dtype=float)
                g2_cuts_err = np.full((num, len(g2_points)), np.nan, dtype=float)
                tdelay_section = np.full((num, len(g2_points)), np.nan, dtype=float)
                dtdelay_section = np.full((num, len(g2_points)), np.nan, dtype=float)

            # fill this section (truncate if needed)
            L = min(len(g2_points), g2_cuts.shape[1])
            tdelay_section[i, :L] = tdelay_cut[:L]
            dtdelay_section[i, :L] = dtdelay_cut[:L]
            g2_cuts[i, :L] = g2_points[:L]
            g2_cuts_err[i, :L] = dg2_points[:L]

        # waiting time centers (in same unit as t_delay; often microseconds or detector clock units)
        t_w = t_delay * (section_edges[:-1] + section_edges[1:]) / 2.0
        return tdelay_section, dtdelay_section, g2_cuts, g2_cuts_err, t_w

    def g2_section_area_new(
        self,
        ttc: np.ndarray,
        t_delay: float,
        num: int,
        log_avg_n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Unweighted/simple version:
        g2 point = nanmean(area)
        g2 err   = nanstd(area)/sqrt(N_valid)  (standard error)

        Returns same outputs as g2_section_area, but without using 'std' weights.
        """
        ttc_v, time_us = self.ttc_vertical(ttc, t_delay)

        if num < 1:
            raise ValueError("num must be >= 1.")
        if log_avg_n < 2:
            raise ValueError("log_avg_n must be >= 2.")

        no_lines = max(int(ttc_v.shape[1] / num), 1)
        section_edges = np.asarray(np.arange(0, ttc_v.shape[1], no_lines), dtype=int)
        section_edges = section_edges[: num + 1]

        if num == 1:
            section_edges = np.array([0, ttc_v.shape[1]], dtype=int)

        g2_cuts = g2_cuts_err = tdelay_section = dtdelay_section = None

        for i, ii in enumerate(section_edges):
            if i == len(section_edges) - 1:
                break

            g2_length = np.count_nonzero(~np.isnan(ttc_v[:, ii]))
            if g2_length < 2:
                continue

            log_idx, tdelay_cut, dtdelay_cut = self.logbin_tdelay(time_us[0:g2_length], log_avg_n)

            g2_points, dg2_points = [], []
            for j in range(len(log_idx) - 1):
                a, b = log_idx[j], log_idx[j + 1]
                g2_area = ttc_v[a:b, ii: section_edges[i + 1]]

                n_valid = np.count_nonzero(~np.isnan(g2_area))
                mean = np.nanmean(g2_area)
                se = np.nanstd(g2_area) / np.sqrt(n_valid) if n_valid > 0 else np.nan

                g2_points.append(mean)
                dg2_points.append(se)

            g2_points = np.array(g2_points, dtype=float)
            dg2_points = np.array(dg2_points, dtype=float)

            if g2_cuts is None:
                g2_cuts = np.full((num, len(g2_points)), np.nan, dtype=float)
                g2_cuts_err = np.full((num, len(g2_points)), np.nan, dtype=float)
                tdelay_section = np.full((num, len(g2_points)), np.nan, dtype=float)
                dtdelay_section = np.full((num, len(g2_points)), np.nan, dtype=float)

            L = min(len(g2_points), g2_cuts.shape[1])
            tdelay_section[i, :L] = tdelay_cut[:L]
            dtdelay_section[i, :L] = dtdelay_cut[:L]
            g2_cuts[i, :L] = g2_points[:L]
            g2_cuts_err[i, :L] = dg2_points[:L]

        t_w = t_delay * (section_edges[:-1] + section_edges[1:]) / 2.0
        return tdelay_section, dtdelay_section, g2_cuts, g2_cuts_err, t_w
    
    @staticmethod
    def g2_model(x, beta, A, tau, kww): 
        """exponential function for fitting the g2 [contrast, baseline, time constant, kww exponent]"""
        return A + beta*np.exp( -2*(x/tau)**kww )

    def g2_fitting(
        self,
        g2: np.ndarray,
        tau: np.ndarray,
        sigma: np.ndarray | None,
        p0,
        bound_lower,
        bound_higher,
        maxfev: int = 5000,
        absolute_sigma: bool = True,
        eps_sigma: float = 1e-15):
        
        """
        Fit g2(tau_axis) using g2_model.
 
        Returns
        -------
        g2_fitted : np.ndarray (same length as input arrays after masking)
        popt : np.ndarray [beta, A, tau, kww]
        perr : np.ndarray uncertainties of parameters
        mask : np.ndarray boolean mask used for fitting (valid points)
        success : bool
        message : str
        """
        
        #g2 = np.asarray(g2, dtype=float)
        #tau = np.asarray(tau, dtype=float)

        if sigma is None:
            sigma = np.ones_like(g2, dtype=float)
        else:
            sigma = np.asarray(sigma, dtype=float)
    
        # Mask invalid points
        mask = (
            ~np.isnan(g2)
            & ~np.isnan(tau)
            & ~np.isnan(sigma)
            & (tau > 0)
            & (sigma > 0)
        )
 
        # If too few points, fail gracefully
        if np.count_nonzero(mask) < 5:
            return (
                np.full_like(g2, np.nan),
                np.full(4, np.nan),
                np.full(4, np.nan),
                mask,
                False,
                "Not enough valid points to fit.",
            )
 
        x = tau[mask]
        y = g2[mask]
        s = sigma[mask]
 
        # Avoid sigma being zero/too small
        s = np.maximum(s, eps_sigma)
 
        try:
            popt, pcov = curve_fit(
                self.g2_model,
                x,
                y,
                sigma=s,
                p0=p0,
                bounds=(bound_lower, bound_higher),
                maxfev=maxfev,
                absolute_sigma=absolute_sigma,
            )
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(4, np.nan)
 
            # Return fitted curve on the *full* tau_axis (not only masked)
            g2_fitted_full = np.full_like(g2, np.nan, dtype=float)
            g2_fitted_full[mask] = self.g2_model(x, *popt)
 
            return g2_fitted_full, popt, perr, mask, True, "OK"
 
        except Exception as e:
            return (
                np.full_like(g2, np.nan),
                np.full(4, np.nan),
                np.full(4, np.nan),
                mask,
                False,
                f"Fit failed: {e}",
            )
        
    def ttc_full_pipeline(
        self,
        ttc: np.ndarray,
        std: np.ndarray,
        t_delay: float,
        ttc_slices_num: int,
        log_avg_n: int,
        p0=None,
        bound_lower=None,
        bound_higher=None,
        fit_all_slices: bool = False,
        ):
        """
        Full pipeline:
        1) Extract g2(tau) for each waiting-time slice (sections)
        2) Fit each extracted g2 with KWW model
        3) Return decay constant gamma (= 1/tau_fit) and metadata

        Returns a dict for clarity (recommended for pipelines).
        """
        # normalize ttc
        
        # --- Step 1: extract g2 sections ---
        tdelay_section, dtdelay_section, g2_cuts, g2_cuts_err, t_w = self.g2_section_area(
            ttc, std, t_delay, ttc_slices_num, log_avg_n
        )

        # Defaults 
        if p0 is None:
            p0 = [0.2, 1.0, 1.0, 1.0]  # beta, A, tau, kww
        if bound_lower is None:
            bound_lower = [0.0, -0.1, 1e-12, 0.1]
        if bound_higher is None:
            bound_higher = [10.0, 2.0, 1e12, 2.0]

        num_slices = g2_cuts.shape[0]
        # Decide which slices to fit
        slices_to_fit = range(num_slices) if fit_all_slices else [0]

        # Prepare outputs
        fitted_g2 = np.full_like(g2_cuts, np.nan, dtype=float)
        params = np.full((num_slices, 4), np.nan, dtype=float)     # beta, A, tau, kww
        params_err = np.full((num_slices, 4), np.nan, dtype=float)
        gamma = np.full(num_slices, np.nan, dtype=float)
        gamma_err = np.full(num_slices, np.nan, dtype=float)
        fit_success = np.zeros(num_slices, dtype=bool)
        fit_message = np.array([""] * num_slices, dtype=object)
        fit_mask = np.full_like(g2_cuts, False, dtype=bool)

        # --- Step 2: fit each slice ---
        for i in slices_to_fit:
            g2_i = g2_cuts[i]
            #normalize to baseline=1
            g2_i = (g2_i - np.nanmin(g2_i)) / (np.nanmax(g2_i) - np.nanmin(g2_i))
            tau_i = tdelay_section[i]      # µs (based on your earlier conversion)
            sig_i = g2_cuts_err[i]

            g2_fit_i, popt, perr, mask, ok, msg = self.g2_fitting(
                g2=g2_i,
                tau=tau_i,
                sigma=sig_i,
                p0=p0,
                bound_lower=bound_lower,
                bound_higher=bound_higher,
            )
            
            fitted_g2[i] = g2_fit_i
            params[i] = popt
            params_err[i] = perr
            fit_success[i] = ok
            fit_message[i] = msg

            # store mask back into same shape
            fit_mask[i, : len(mask)] = mask

            # --- Step 3: extract decay constant gamma (= 1/tau_fit) ---
            tau_fit = popt[2]
            dtau_fit = perr[2]

            if ok and np.isfinite(tau_fit) and tau_fit > 0:
                gamma[i] = 1.0 / tau_fit
                gamma_err[i] = dtau_fit / (tau_fit ** 2) if np.isfinite(dtau_fit) else np.nan
        
        # Helpful metadata bundle
        result = {
            # extracted (measured) curves
            "g2_extracted": g2_cuts,
            "g2_extracted_err": g2_cuts_err,
            "tau_axis_us": tdelay_section,          # µs
            "tau_axis_us_err": dtdelay_section,     # µs
            "t_w": t_w,                              # waiting-time centers (units of t_delay)
            # fitted curves + fit outputs
            "g2_fitted": fitted_g2,
            "fit_params": params,                    # columns: beta, A, tau, kww
            "fit_params_err": params_err,
            "tau_fitted": tau_fit,             # fitted tau (µs)
            "tau_fitted_err": dtau_fit,
            "gamma": gamma,                          # 1/tau_fit
            "gamma_err": gamma_err,
            "fit_success": fit_success,
            "fit_message": fit_message,
            "fit_mask": fit_mask,
            # some useful bookkeeping
            "model_param_names": ["beta", "A", "tau", "kww"],
            "gamma_definition": "gamma = 1/tau_fit (tau_fit is the fitted relaxation time in same units as tau_axis_us)",
        }

        return result

        
             
            
            
    
    
    










