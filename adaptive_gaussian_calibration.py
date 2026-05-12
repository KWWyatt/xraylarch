# Adaptive Gaussian Calibration - Automatically adjusts to data

def calibrate_cs_edge_adaptive_gaussian(projects, target_e0=35985.0, fit_window=15.0, 
                                       smooth_points=5, peak_height_factor=0.1):
    """
    Adaptive Gaussian calibration that automatically adjusts peak height thresholds
    based on the actual data characteristics.
    
    Parameters:
    -----------
    projects : dict
        Dictionary of loaded projects
    target_e0 : float
        Target Cs K edge energy in eV (default: 35985.0 eV)
    fit_window : float
        Energy window around peak for Gaussian fitting (default: 15.0 eV)
    smooth_points : int
        Number of points for smoothing (default: 5)
    peak_height_factor : float
        Fraction of max derivative to use as minimum peak height (default: 0.1)
    """
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    from larch.xafs import pre_edge
    
    def gaussian(x, amplitude, center, sigma, offset):
        return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) + offset
    
    print(f"Adaptive Gaussian calibration: Auto-adjusting to data characteristics")
    print("="*70)
    
    for proj_name, project in projects.items():
        print(f"\\nProcessing project: {proj_name}")
        
        for group_name, group in project.groups.items():
            if not (hasattr(group, 'energy') and hasattr(group, 'mu')):
                continue
                
            print(f"\\n  Processing group: {group_name}")
            
            # Store original energy if not already done
            if not hasattr(group, 'energy_original'):
                group.energy_original = group.energy.copy()
            
            energy = group.energy_original
            mu = group.mu
            
            # Calculate and smooth derivative
            dmu_de = np.gradient(mu, energy)
            dmu_de_smooth = uniform_filter1d(dmu_de, size=smooth_points)
            
            # Search window around expected Cs K-edge
            expected_edge_min = target_e0 - 15  # Slightly wider for initial analysis
            expected_edge_max = target_e0 + 15
            
            strict_mask = (energy >= expected_edge_min) & (energy <= expected_edge_max)
            
            if not strict_mask.any():
                print(f"    Warning: No data in search window")
                continue
            
            search_energies = energy[strict_mask]
            search_derivative = dmu_de_smooth[strict_mask]
            
            # ADAPTIVE THRESHOLD: Base on actual data characteristics
            max_derivative = np.max(search_derivative)
            min_derivative = np.min(search_derivative)
            derivative_range = max_derivative - min_derivative
            
            # Calculate adaptive minimum peak height
            adaptive_min_height = min_derivative + peak_height_factor * derivative_range
            
            print(f"    Data analysis:")
            print(f"      Max derivative: {max_derivative:.6f}")
            print(f"      Min derivative: {min_derivative:.6f}")
            print(f"      Range: {derivative_range:.6f}")
            print(f"      Adaptive threshold: {adaptive_min_height:.6f}")
            
            # Find peaks with adaptive threshold
            peak_indices, peak_properties = find_peaks(search_derivative, 
                                                      height=adaptive_min_height,
                                                      prominence=derivative_range * 0.05)
            
            if len(peak_indices) == 0:
                # If still no peaks, lower the threshold further
                adaptive_min_height = min_derivative + 0.05 * derivative_range
                print(f"    No peaks found, lowering threshold to: {adaptive_min_height:.6f}")
                
                peak_indices, peak_properties = find_peaks(search_derivative, 
                                                          height=adaptive_min_height)
            
            if len(peak_indices) == 0:
                # Final fallback: just use the maximum
                max_idx = np.argmax(search_derivative)
                peak_energy = search_energies[max_idx]
                peak_value = search_derivative[max_idx]
                print(f"    Using absolute maximum at {peak_energy:.2f} eV (value: {peak_value:.6f})")
            else:
                # Find the peak closest to the expected edge energy
                peak_energies = search_energies[peak_indices]
                peak_distances = np.abs(peak_energies - target_e0)
                closest_peak_idx = peak_indices[np.argmin(peak_distances)]
                
                peak_energy = search_energies[closest_peak_idx]
                peak_value = search_derivative[closest_peak_idx]
                
                print(f"    Found {len(peak_indices)} peaks")
                print(f"    Using closest to target: {peak_energy:.2f} eV (value: {peak_value:.6f})")
                print(f"    Distance from target: {abs(peak_energy - target_e0):.2f} eV")
            
            # Proceed with Gaussian fitting
            fit_min = peak_energy - fit_window/2
            fit_max = peak_energy + fit_window/2
            fit_mask = (energy >= fit_min) & (energy <= fit_max)
            
            if fit_mask.sum() < 8:
                print(f"    Insufficient points for fitting ({fit_mask.sum()} points)")
                # Apply simple calibration
                energy_shift = target_e0 - peak_energy
                group.energy = group.energy_original + energy_shift
                pre_edge(group)
                
                group.energy_shift_applied = energy_shift
                group.derivative_center_fitted = peak_energy
                group.calibration_method = 'simple_peak_insufficient_points'
                group.fit_validation_passed = False
                
                print(f"    Simple calibration: {energy_shift:+.2f} eV shift")
                print(f"    New e0: {group.e0:.2f} eV")
                continue
            
            fit_energies = energy[fit_mask]
            fit_derivative = dmu_de_smooth[fit_mask]
            
            # Improved parameter estimation
            amplitude_guess = peak_value - np.min(fit_derivative)
            center_guess = peak_energy
            
            # Estimate sigma from data width
            half_max = min_derivative + 0.5 * (peak_value - min_derivative)
            half_max_indices = np.where(fit_derivative >= half_max)[0]
            if len(half_max_indices) > 1:
                estimated_fwhm = fit_energies[half_max_indices[-1]] - fit_energies[half_max_indices[0]]
                sigma_guess = max(1.0, estimated_fwhm / 2.355)  # Ensure minimum reasonable sigma
            else:
                sigma_guess = 2.0
            
            offset_guess = np.min(fit_derivative)
            
            print(f"    Initial fit parameters:")
            print(f"      Amplitude: {amplitude_guess:.6f}")
            print(f"      Center: {center_guess:.2f} eV")
            print(f"      Sigma: {sigma_guess:.2f} eV")
            
            # Reasonable bounds
            bounds = ([0, fit_energies[0], 0.5, -np.inf],
                     [5*amplitude_guess, fit_energies[-1], 10.0, np.inf])
            
            try:
                # Perform Gaussian fit
                popt, pcov = curve_fit(gaussian, fit_energies, fit_derivative, 
                                     p0=[amplitude_guess, center_guess, sigma_guess, offset_guess],
                                     bounds=bounds, maxfev=5000)
                
                fitted_amplitude, fitted_center, fitted_sigma, fitted_offset = popt
                
                # Calculate fit quality
                fitted_curve = gaussian(fit_energies, *popt)
                ss_res = np.sum((fit_derivative - fitted_curve) ** 2)
                ss_tot = np.sum((fit_derivative - np.mean(fit_derivative)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # More lenient validation criteria
                fit_is_good = True
                reasons = []
                
                # Check if center is within reasonable range
                if abs(fitted_center - peak_energy) > fit_window/2:
                    fit_is_good = False
                    reasons.append(f"Center too far ({abs(fitted_center - peak_energy):.2f} eV)")
                
                # More lenient sigma check
                if fitted_sigma < 0.3 or fitted_sigma > 15.0:
                    fit_is_good = False
                    reasons.append(f"Unrealistic sigma ({fitted_sigma:.2f} eV)")
                
                # More lenient R² check
                if r_squared < 0.3:
                    fit_is_good = False
                    reasons.append(f"Poor fit (R²={r_squared:.3f})")
                
                # Check amplitude is reasonable
                if fitted_amplitude < amplitude_guess * 0.05:
                    fit_is_good = False
                    reasons.append("Amplitude too small")
                
                if fit_is_good:
                    print(f"    ✓ Successful Gaussian fit:")
                    print(f"      Center: {fitted_center:.3f} eV")
                    print(f"      Sigma: {fitted_sigma:.3f} eV")
                    print(f"      FWHM: {2.355 * fitted_sigma:.3f} eV")
                    print(f"      R²: {r_squared:.4f}")
                    
                    energy_shift = target_e0 - fitted_center
                    group.energy = group.energy_original + energy_shift
                    pre_edge(group)
                    
                    # Store successful fit metadata
                    group.energy_shift_applied = energy_shift
                    group.derivative_center_fitted = fitted_center
                    group.derivative_center_original = peak_energy
                    group.gaussian_fit_params = popt
                    group.gaussian_fit_r2 = r_squared
                    group.gaussian_fwhm = 2.355 * fitted_sigma
                    group.calibration_method = 'adaptive_gaussian_fit'
                    group.fit_validation_passed = True
                    group.adaptive_threshold_used = adaptive_min_height
                    
                    print(f"      Energy shift: {energy_shift:+.2f} eV")
                    print(f"      New e0: {group.e0:.2f} eV")
                    
                else:
                    print(f"    ⚠ Fit validation failed: {', '.join(reasons)}")
                    print(f"    Using simple peak finding")
                    
                    # Fallback to simple peak
                    energy_shift = target_e0 - peak_energy
                    group.energy = group.energy_original + energy_shift
                    pre_edge(group)
                    
                    group.energy_shift_applied = energy_shift
                    group.derivative_center_fitted = peak_energy
                    group.derivative_center_original = peak_energy
                    group.calibration_method = 'simple_peak_fallback'
                    group.fit_validation_passed = False
                    group.adaptive_threshold_used = adaptive_min_height
                    
                    print(f"      Fallback shift: {energy_shift:+.2f} eV")
                    print(f"      New e0: {group.e0:.2f} eV")
                
            except Exception as e:
                print(f"    ✗ Gaussian fitting failed: {e}")
                
                # Fallback to simple peak
                energy_shift = target_e0 - peak_energy
                group.energy = group.energy_original + energy_shift
                pre_edge(group)
                
                group.energy_shift_applied = energy_shift
                group.derivative_center_fitted = peak_energy
                group.derivative_center_original = peak_energy
                group.calibration_method = 'simple_peak_error_fallback'
                group.fit_validation_passed = False
                group.adaptive_threshold_used = adaptive_min_height
                
                print(f"      Error fallback shift: {energy_shift:+.2f} eV")
                print(f"      New e0: {group.e0:.2f} eV")
    
    return projects

def analyze_derivative_characteristics(projects):
    """
    Analyze the derivative characteristics of your data to help set appropriate thresholds
    """
    import numpy as np
    from scipy.ndimage import uniform_filter1d
    
    print("Analyzing derivative characteristics across all groups:")
    print("="*60)
    
    all_max_values = []
    all_ranges = []
    
    for proj_name, project in projects.items():
        print(f"\\nProject: {proj_name}")
        
        for group_name, group in project.groups.items():
            if not (hasattr(group, 'energy') and hasattr(group, 'mu')):
                continue
            
            energy = group.energy_original if hasattr(group, 'energy_original') else group.energy
            dmu_de = np.gradient(group.mu, energy)
            dmu_de_smooth = uniform_filter1d(dmu_de, size=5)
            
            # Focus on edge region
            edge_mask = (energy >= 35970) & (energy <= 36000)
            if edge_mask.any():
                edge_derivative = dmu_de_smooth[edge_mask]
                max_val = np.max(edge_derivative)
                min_val = np.min(edge_derivative)
                range_val = max_val - min_val
                
                all_max_values.append(max_val)
                all_ranges.append(range_val)
                
                print(f"  {group_name}:")
                print(f"    Max derivative: {max_val:.6f}")
                print(f"    Min derivative: {min_val:.6f}")
                print(f"    Range: {range_val:.6f}")
                print(f"    10% of range: {0.1 * range_val:.6f}")
    
    if all_max_values:
        print(f"\\nOverall statistics:")
        print(f"  Average max derivative: {np.mean(all_max_values):.6f}")
        print(f"  Average range: {np.mean(all_ranges):.6f}")
        print(f"  Suggested min_peak_height: {0.1 * np.mean(all_ranges):.6f}")
    
    return all_max_values, all_ranges

print("Adaptive Gaussian calibration functions loaded:")
print("- calibrate_cs_edge_adaptive_gaussian(): Auto-adjusts to your data")
print("- analyze_derivative_characteristics(): Analyzes your data characteristics")
