# Energy Calibration for Cs K Edge
# The Cs K edge should be at ~35985 eV, aligned with the maximum of the first derivative

def calibrate_cs_k_edge(projects, target_e0=35985.0):
    """
    Calibrate Cs K edge energy by aligning the maximum of the first derivative 
    with the expected e0 value of 35985 eV.
    
    Parameters:
    -----------
    projects : dict
        Dictionary of loaded projects from batch_process_athena_projects()
    target_e0 : float, optional
        Target Cs K edge energy in eV (default: 35985.0 eV)
    
    Returns:
    --------
    dict
        Dictionary of calibrated projects with updated energy arrays
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from larch.xafs import pre_edge
    
    print(f"Calibrating Cs K edge to {target_e0:.1f} eV")
    print("="*50)
    
    calibrated_projects = {}
    
    for proj_name, project in projects.items():
        print(f"\nProcessing project: {proj_name}")
        calibrated_projects[proj_name] = project
        
        for group_name, group in project.groups.items():
            if not (hasattr(group, 'energy') and hasattr(group, 'mu')):
                continue
                
            print(f"  Processing group: {group_name}")
            
            # Store original energy
            if not hasattr(group, 'energy_original'):
                group.energy_original = group.energy.copy()
            
            # Calculate first derivative
            dmu_de = np.gradient(group.mu, group.energy)
            
            # Find current e0 region (±10 eV around current e0)
            current_e0 = getattr(group, 'e0', 35985.0)
            search_mask = (group.energy >= current_e0 - 10) & (group.energy <= current_e0 + 10)
            
            if search_mask.any():
                # Find maximum of first derivative in search region
                search_energies = group.energy[search_mask]
                search_derivative = dmu_de[search_mask]
                max_idx = np.argmax(search_derivative)
                derivative_max_energy = search_energies[max_idx]
                
                # Calculate energy shift
                energy_shift = target_e0 - derivative_max_energy
                
                print(f"    Original derivative max: {derivative_max_energy:.2f} eV")
                print(f"    Energy shift needed: {energy_shift:+.2f} eV")
                
                # Apply calibration
                group.energy = group.energy_original + energy_shift
                
                # Re-run pre-edge analysis
                pre_edge(group)
                
                print(f"    New e0 after calibration: {group.e0:.2f} eV")
                
                # Store calibration info
                group.energy_shift_applied = energy_shift
                group.derivative_max_original = derivative_max_energy
            
    return calibrated_projects

def plot_calibration_comparison(projects_original, projects_calibrated):
    """
    Plot comparison of original vs calibrated energy scales
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    for proj_name in projects_original.keys():
        if proj_name not in projects_calibrated:
            continue
            
        orig_project = projects_original[proj_name]
        cal_project = projects_calibrated[proj_name]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Energy Calibration Results: {proj_name}', fontsize=14)
        
        for group_name in orig_project.groups.keys():
            if group_name not in cal_project.groups:
                continue
                
            orig_group = orig_project.groups[group_name]
            cal_group = cal_project.groups[group_name]
            
            if not (hasattr(orig_group, 'energy') and hasattr(orig_group, 'norm')):
                continue
            
            # Get original energy array
            orig_energy = orig_group.energy
            if hasattr(cal_group, 'energy_original'):
                orig_energy = cal_group.energy_original
            
            # Plot normalized absorption
            ax1.plot(orig_energy, orig_group.norm, 
                    label=f'{group_name} (original)', linestyle='--', alpha=0.7)
            ax1.plot(cal_group.energy, cal_group.norm, 
                    label=f'{group_name} (calibrated)', linewidth=2)
            
            # Plot derivatives
            dmu_orig = np.gradient(orig_group.mu, orig_energy)
            dmu_cal = np.gradient(cal_group.mu, cal_group.energy)
            
            ax2.plot(orig_energy, dmu_orig, 
                    label=f'{group_name} (original)', linestyle='--', alpha=0.7)
            ax2.plot(cal_group.energy, dmu_cal, 
                    label=f'{group_name} (calibrated)', linewidth=2)
            
            # Mark target e0 and original derivative maximum
            ax1.axvline(35985.0, color='red', linestyle=':', alpha=0.8, label='Target e0 (35985 eV)')
            ax2.axvline(35985.0, color='red', linestyle=':', alpha=0.8, label='Target e0 (35985 eV)')
            
            if hasattr(cal_group, 'derivative_max_original'):
                ax1.axvline(cal_group.derivative_max_original, color='orange', 
                           linestyle=':', alpha=0.6, label='Original deriv max')
                ax2.axvline(cal_group.derivative_max_original, color='orange', 
                           linestyle=':', alpha=0.6, label='Original deriv max')
        
        # Format plots
        ax1.set_ylabel('Normalized μ(E)')
        ax1.set_title('Normalized XAFS Data - Energy Calibration')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Energy (eV)')
        ax2.set_ylabel('dμ/dE')
        ax2.set_title('First Derivative - Shows Calibration Alignment')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Cs K-edge energy calibration functions loaded:")
    print("- calibrate_cs_k_edge(projects, target_e0=35985.0)")
    print("- plot_calibration_comparison(projects_original, projects_calibrated)")
