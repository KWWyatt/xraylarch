# Improved Calibration for CsPbI3_100mM - Focus on derivative alignment

def fix_cs100mM_calibration(projects, target_e0=35985.0):
    """
    Specifically fix the CsPbI3_100mM calibration to align derivative maximum with e0
    """
    import numpy as np
    from larch.xafs import pre_edge
    import matplotlib.pyplot as plt
    
    print("Fixing CsPbI3_100mM calibration - aligning derivative maximum with e0")
    print("="*60)
    
    for proj_name, project in projects.items():
        if 'CsPbI3_100mM' not in proj_name:
            continue
            
        print(f"\nProcessing project: {proj_name}")
        
        for group_name, group in project.groups.items():
            if 'CsPbI3_100mM' not in group_name:
                continue
                
            print(f"  Fixing group: {group_name}")
            
            # Store original if not already done
            if not hasattr(group, 'energy_original'):
                group.energy_original = group.energy.copy()
                group.mu_original = group.mu.copy()
            
            # Use original data for calculation
            energy = group.energy_original
            mu = group.mu_original
            
            # Calculate first derivative
            dmu_de = np.gradient(mu, energy)
            
            # Find the actual maximum of the first derivative in the edge region
            # Look in a broader window around the expected edge
            edge_window = (energy >= 35970) & (energy <= 36000)
            
            if edge_window.any():
                edge_energies = energy[edge_window]
                edge_derivative = dmu_de[edge_window]
                
                # Find the absolute maximum
                max_idx = np.argmax(edge_derivative)
                actual_deriv_max = edge_energies[max_idx]
                max_value = edge_derivative[max_idx]
                
                print(f"    Current derivative maximum: {actual_deriv_max:.2f} eV")
                print(f"    Maximum derivative value: {max_value:.6f}")
                print(f"    Target e0: {target_e0:.2f} eV")
                
                # Calculate shift to align derivative max with target e0
                energy_shift = target_e0 - actual_deriv_max
                print(f"    Required shift: {energy_shift:+.2f} eV")
                
                # Apply calibration
                group.energy = group.energy_original + energy_shift
                
                # Re-run pre-edge analysis
                pre_edge(group)
                
                print(f"    New e0 after calibration: {group.e0:.2f} eV")
                
                # Store calibration info
                group.energy_shift_applied = energy_shift
                group.derivative_max_original = actual_deriv_max
                group.calibration_target = target_e0
                
                # Verify the fix
                new_dmu_de = np.gradient(group.mu, group.energy)
                verify_window = (group.energy >= 35980) & (group.energy <= 35990)
                if verify_window.any():
                    verify_energies = group.energy[verify_window]
                    verify_derivative = new_dmu_de[verify_window]
                    new_max_idx = np.argmax(verify_derivative)
                    new_deriv_max = verify_energies[new_max_idx]
                    print(f"    Verification: New derivative max at {new_deriv_max:.2f} eV")
                    print(f"    Alignment error: {abs(new_deriv_max - target_e0):.2f} eV")
    
    return projects

def plot_cs100mM_fix(projects):
    """Plot before/after for CsPbI3_100mM specifically"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    for proj_name, project in projects.items():
        if 'CsPbI3_100mM' not in proj_name:
            continue
            
        for group_name, group in project.groups.items():
            if 'CsPbI3_100mM' not in group_name or not hasattr(group, 'energy_original'):
                continue
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            fig.suptitle(f'CsPbI3_100mM Calibration Fix: {group_name}', fontsize=16)
            
            # Calculate derivatives
            dmu_orig = np.gradient(group.mu_original, group.energy_original)
            dmu_cal = np.gradient(group.mu, group.energy)
            
            # Plot normalized absorption
            ax1.plot(group.energy_original, group.norm, 
                    label='Before calibration', linestyle='--', linewidth=2, alpha=0.7)
            ax1.plot(group.energy, group.norm, 
                    label='After calibration', linewidth=2)
            
            # Plot derivatives
            ax2.plot(group.energy_original, dmu_orig, 
                    label='Before calibration', linestyle='--', linewidth=2, alpha=0.7)
            ax2.plot(group.energy, dmu_cal, 
                    label='After calibration', linewidth=2)
            
            # Mark key positions
            target_e0 = 35985.0
            ax1.axvline(target_e0, color='red', linestyle=':', linewidth=2, alpha=0.8, 
                       label='Target Cs K-edge (35985 eV)')
            ax2.axvline(target_e0, color='red', linestyle=':', linewidth=2, alpha=0.8, 
                       label='Target Cs K-edge (35985 eV)')
            
            if hasattr(group, 'derivative_max_original'):
                ax2.axvline(group.derivative_max_original, color='orange', 
                           linestyle=':', linewidth=2, alpha=0.6, 
                           label=f'Original deriv max ({group.derivative_max_original:.1f} eV)')
            
            # Add shift info
            if hasattr(group, 'energy_shift_applied'):
                shift_text = f'Energy shift applied: {group.energy_shift_applied:+.2f} eV'
                ax1.text(0.02, 0.98, shift_text, transform=ax1.transAxes, 
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            # Format plots
            ax1.set_ylabel('Normalized μ(E)')
            ax1.set_title('Normalized XAFS Data')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(35960, 36010)  # Focus on edge region
            
            ax2.set_xlabel('Energy (eV)')
            ax2.set_ylabel('dμ/dE')
            ax2.set_title('First Derivative - Showing Alignment')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(35960, 36010)  # Focus on edge region
            
            plt.tight_layout()
            plt.show()
            
            break  # Only process the first matching group
        break  # Only process the first matching project

# Usage instructions
print("To fix CsPbI3_100mM calibration:")
print("1. projects_fixed = fix_cs100mM_calibration(projects)")
print("2. plot_cs100mM_fix(projects_fixed)")
