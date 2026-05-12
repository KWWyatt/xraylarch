# Improved Gaussian Fitting with Noise Rejection

def calibrate_cs_edge_robust_gaussian(projects, target_e0=35985.0, fit_window=15.0, 
                                     smooth_points=5, min_peak_height=0.0005):
    """
    Robust Gaussian calibration that avoids fitting to noise by:
    1. Using a stricter search window around the expected edge
    2. Requiring minimum peak height to avoid noise
    3. Better smoothing to reduce noise impact
    4. Multiple peak detection to find the strongest peak
    5. Validation of fit results
    
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
    min_peak_height : float
        Minimum peak height to consider valid (default: 0.0005)
    """
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    from larch.xafs import pre_edge
    
    def gaussian(x, amplitude, center, sigma, offset):
        """Gaussian function for fitting"""
        return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) + offset
    
    print(f"Robust Gaussian calibration: Avoiding noise fits")
    print("="*60)
    
    for proj_name, project in projects.items():
        print(f"\nProcessing project: {proj_name}")
        
        for group_name, group in project.groups.items():
            if not (hasattr(group, 'energy') and hasattr(group, 'mu')):
                continue
                
            print(f"\n  Processing group: {group_name}")
            
            # Store original energy if not already done
            if not hasattr(group, 'energy_original'):
                group.energy_original = group.energy.copy()
            
            energy = group.energy_original
            mu = group.mu
            
            # Calculate first derivative
            dmu_de = np.gradient(mu, energy)
            
            # Apply more aggressive smoothing to reduce noise
            dmu_de_smooth = uniform_filter1d(dmu_de, size=smooth_points)
            
            # Define a strict search window around expected Cs K-edge
            # This prevents fitting to noise far from the edge
            expected_edge_min = target_e0 - 10  # ±10 eV around expected edge
            expected_edge_max = target_e0 + 10
            
            strict_mask = (energy >= expected_edge_min) & (energy <= expected_edge_max)
            
            if not strict_mask.any():
                print(f"    Warning: No data in strict search window [{expected_edge_min}, {expected_edge_max}]")
                continue
            
            # Extract data in strict search window
            search_energies = energy[strict_mask]
            search_derivative = dmu_de_smooth[strict_mask]
            
            # Find peaks in the derivative (not just maximum)
            # This helps identify the true edge peak vs noise
            peak_indices, peak_properties = find_peaks(search_derivative, 
                                                      height=min_peak_height,
                                                      prominence=min_peak_height/2)
            
            if len(peak_indices) == 0:
                print(f"    Warning: No significant peaks found above threshold {min_peak_height}")
                # Fallback to simple maximum finding with lower threshold
                max_idx = np.argmax(search_derivative)
                peak_energy = search_energies[max_idx]
                peak_value = search_derivative[max_idx]
                print(f"    Using simple maximum at {peak_energy:.2f} eV")
            else:
                # Find the strongest peak (highest prominence or height)
                if 'prominences' in peak_properties:
                    strongest_peak_idx = peak_indices[np.argmax(peak_properties['prominences'])]
                else:
                    strongest_peak_idx = peak_indices[np.argmax(peak_properties['peak_heights'])]
                
                peak_energy = search_energies[strongest_peak_idx]
                peak_value = search_derivative[strongest_peak_idx]
                
                print(f"    Found {len(peak_indices)} peaks, using strongest at {peak_energy:.2f} eV")
            
            # Validate that the peak is reasonable
            if peak_value < min_peak_height:
                print(f"    Warning: Peak too small ({peak_value:.6f} < {min_peak_height})")
                print(f"    Skipping Gaussian fit for this group")
                continue
            
            # Define fitting region around the identified peak
            fit_min = peak_energy - fit_window/2
            fit_max = peak_energy + fit_window/2
            fit_mask = (energy >= fit_min) & (energy <= fit_max)
            
            if fit_mask.sum() < 8:
                print(f"    Warning: Insufficient points for fitting ({fit_mask.sum()} points)")
                continue
            
            fit_energies = energy[fit_mask]
            fit_derivative = dmu_de_smooth[fit_mask]
            
            # Better initial parameter estimation
            amplitude_guess = peak_value - np.min(fit_derivative)
            center_guess = peak_energy
            # Estimate sigma from the width of the peak
            half_max = (peak_value + np.min(fit_derivative)) / 2
            half_max_indices = np.where(fit_derivative >= half_max)[0]
            if len(half_max_indices) > 1:
                estimated_fwhm = fit_energies[half_max_indices[-1]] - fit_energies[half_max_indices[0]]
                sigma_guess = estimated_fwhm / 2.355  # Convert FWHM to sigma
            else:
                sigma_guess = 2.0  # Default guess
            
            offset_guess = np.min(fit_derivative)
            
            initial_params = [amplitude_guess, center_guess, sigma_guess, offset_guess]
            
            # Set reasonable bounds to prevent unrealistic fits
            bounds = ([0, fit_energies[0], 0.5, -np.inf],  # Lower bounds
                     [10*amplitude_guess, fit_energies[-1], 10.0, np.inf])  # Upper bounds
            
            try:
                # Perform Gaussian fit with bounds
                popt, pcov = curve_fit(gaussian, fit_energies, fit_derivative, 
                                     p0=initial_params, bounds=bounds, maxfev=3000)
                
                fitted_amplitude, fitted_center, fitted_sigma, fitted_offset = popt
                
                # Calculate fit quality
                fitted_curve = gaussian(fit_energies, *popt)
                ss_res = np.sum((fit_derivative - fitted_curve) ** 2)
                ss_tot = np.sum((fit_derivative - np.mean(fit_derivative)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Validate fit results
                fit_is_good = True
                validation_messages = []
                
                # Check if center is reasonable
                if abs(fitted_center - peak_energy) > fit_window/3:
                    fit_is_good = False
                    validation_messages.append(f"Center too far from peak ({abs(fitted_center - peak_energy):.2f} eV)")
                
                # Check if sigma is reasonable (not too narrow or too wide)
                if fitted_sigma < 0.5 or fitted_sigma > 8.0:
                    fit_is_good = False
                    validation_messages.append(f"Unrealistic sigma ({fitted_sigma:.2f} eV)")
                
                # Check fit quality
                if r_squared < 0.5:
                    fit_is_good = False
                    validation_messages.append(f"Poor fit quality (R² = {r_squared:.3f})")
                
                # Check if amplitude is reasonable
                if fitted_amplitude < amplitude_guess * 0.1:
                    fit_is_good = False
                    validation_messages.append(f"Amplitude too small")
                
                if fit_is_good:
                    print(f"    ✓ Successful Gaussian fit:")
                    print(f"      Center: {fitted_center:.3f} eV")
                    print(f"      Sigma: {fitted_sigma:.3f} eV")
                    print(f"      FWHM: {2.355 * fitted_sigma:.3f} eV")
                    print(f"      R²: {r_squared:.4f}")
                    
                    # Calculate energy shift
                    energy_shift = target_e0 - fitted_center
                    print(f"      Required shift: {energy_shift:+.2f} eV")
                    
                    # Apply calibration
                    group.energy = group.energy_original + energy_shift
                    pre_edge(group)
                    
                    # Store metadata
                    group.energy_shift_applied = energy_shift
                    group.derivative_center_fitted = fitted_center
                    group.derivative_center_original = peak_energy
                    group.gaussian_fit_params = popt
                    group.gaussian_fit_r2 = r_squared
                    group.gaussian_fwhm = 2.355 * fitted_sigma
                    group.calibration_method = 'robust_gaussian_fit'
                    group.fit_validation_passed = True
                    
                    print(f"      New e0: {group.e0:.2f} eV")
                    
                else:
                    print(f"    ✗ Gaussian fit failed validation:")
                    for msg in validation_messages:
                        print(f"      - {msg}")
                    
                    # Use simple peak finding as fallback
                    energy_shift = target_e0 - peak_energy
                    group.energy = group.energy_original + energy_shift
                    pre_edge(group)
                    
                    group.energy_shift_applied = energy_shift
                    group.derivative_center_fitted = peak_energy
                    group.derivative_center_original = peak_energy
                    group.calibration_method = 'simple_peak_fallback'
                    group.fit_validation_passed = False
                    
                    print(f"      Using simple peak: {energy_shift:+.2f} eV shift")
                    print(f"      New e0: {group.e0:.2f} eV")
                
            except Exception as fit_error:
                print(f"    ✗ Gaussian fitting failed: {fit_error}")
                
                # Fallback to simple peak finding
                energy_shift = target_e0 - peak_energy
                group.energy = group.energy_original + energy_shift
                pre_edge(group)
                
                group.energy_shift_applied = energy_shift
                group.derivative_center_fitted = peak_energy
                group.derivative_center_original = peak_energy
                group.calibration_method = 'simple_peak_fallback'
                group.fit_validation_passed = False
                
                print(f"      Fallback applied: {energy_shift:+.2f} eV shift")
                print(f"      New e0: {group.e0:.2f} eV")
    
    return projects

def plot_robust_calibration_results(projects):
    """
    Enhanced plotting to show the robust calibration results
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    
    def gaussian(x, amplitude, center, sigma, offset):
        return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) + offset
    
    for proj_name, project in projects.items():
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Robust Gaussian Calibration: {proj_name}', fontsize=16)
        
        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]
        
        for group_name, group in project.groups.items():
            if not hasattr(group, 'energy_original'):
                continue
            
            # Calculate derivatives
            dmu_orig = np.gradient(group.mu, group.energy_original)
            dmu_smooth = uniform_filter1d(dmu_orig, size=5)
            dmu_cal = np.gradient(group.mu, group.energy)
            
            # Plot 1: Calibrated absorption
            ax1.plot(group.energy_original, group.norm, 
                    label=f'{group_name} (original)', linestyle='--', alpha=0.7)
            ax1.plot(group.energy, group.norm, 
                    label=f'{group_name} (calibrated)', linewidth=2)
            ax1.axvline(35985.0, color='red', linestyle=':', alpha=0.8, label='Target Cs K-edge')
            
            # Plot 2: Derivatives with search window
            ax2.plot(group.energy_original, dmu_orig, alpha=0.3, label=f'{group_name} (raw)')
            ax2.plot(group.energy_original, dmu_smooth, linewidth=2, label=f'{group_name} (smoothed)')
            
            # Show search window
            ax2.axvspan(35975, 35995, alpha=0.2, color='yellow', label='Search window')
            ax2.axvline(35985.0, color='red', linestyle=':', alpha=0.8)
            
            # Mark peaks found
            search_mask = (group.energy_original >= 35975) & (group.energy_original <= 35995)
            if search_mask.any():
                search_energies = group.energy_original[search_mask]
                search_deriv = dmu_smooth[search_mask]
                peaks, _ = find_peaks(search_deriv, height=0.0005, prominence=0.0002)
                if len(peaks) > 0:
                    ax2.scatter(search_energies[peaks], search_deriv[peaks], 
                              color='red', s=50, zorder=5, label='Detected peaks')
            
            # Plot 3: Gaussian fit detail
            if hasattr(group, 'gaussian_fit_params') and hasattr(group, 'fit_validation_passed'):
                fit_center = group.derivative_center_fitted
                fit_window = 15.0
                fit_mask = (group.energy_original >= fit_center - fit_window/2) & \
                          (group.energy_original <= fit_center + fit_window/2)
                
                if fit_mask.any():
                    fit_energies = group.energy_original[fit_mask]
                    fit_data = dmu_smooth[fit_mask]
                    
                    ax3.plot(fit_energies, fit_data, 'o', alpha=0.7, markersize=4, 
                            label=f'{group_name} data')
                    
                    if group.fit_validation_passed:
                        fit_curve = gaussian(fit_energies, *group.gaussian_fit_params)
                        ax3.plot(fit_energies, fit_curve, '-', linewidth=3, 
                                label=f'Valid Gaussian fit (R²={group.gaussian_fit_r2:.3f})')
                        ax3.axvline(group.derivative_center_fitted, color='green', 
                                   linestyle='-', alpha=0.8, linewidth=2,
                                   label=f'Fitted center ({group.derivative_center_fitted:.2f} eV)')
                    else:
                        ax3.axvline(group.derivative_center_fitted, color='orange', 
                                   linestyle='--', alpha=0.8, linewidth=2,
                                   label=f'Fallback center ({group.derivative_center_fitted:.2f} eV)')
            
            ax3.axvline(35985.0, color='red', linestyle=':', alpha=0.8)
            
            # Plot 4: Summary
            method = getattr(group, 'calibration_method', 'unknown')
            shift = getattr(group, 'energy_shift_applied', 0)
            validation = getattr(group, 'fit_validation_passed', False)
            
            status_color = 'lightgreen' if validation else 'lightyellow'
            status_text = '✓ Robust fit' if validation else '⚠ Fallback used'
            
            summary_text = f"{group_name}:\n{status_text}\n"
            summary_text += f"Method: {method}\n"
            summary_text += f"Shift: {shift:+.2f} eV\n"
            
            if hasattr(group, 'gaussian_fwhm') and validation:
                summary_text += f"FWHM: {group.gaussian_fwhm:.2f} eV\n"
            if hasattr(group, 'gaussian_fit_r2') and validation:
                summary_text += f"R²: {group.gaussian_fit_r2:.3f}\n"
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.8))
        
        # Format plots
        ax1.set_ylabel('Normalized μ(E)')
        ax1.set_title('Calibrated XAFS Data')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_ylabel('dμ/dE')
        ax2.set_title('Derivative with Peak Detection')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_xlabel('Energy (eV)')
        ax3.set_ylabel('dμ/dE')
        ax3.set_title('Gaussian Fit Validation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.set_title('Calibration Status')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()

print("Robust Gaussian calibration functions loaded:")
print("- calibrate_cs_edge_robust_gaussian(): Avoids fitting to noise")
print("- plot_robust_calibration_results(): Shows validation results")
