import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
# Import the functions you need:
from larch.io import read_athena, read_ascii
from larch.xafs import (pre_edge, autobk, xftf, ff2chi, feffpath, feffit_transform, feffit_dataset, feffit, feffit_report, TransformGroup, path2chi)
from larch.fitting import param, guess, param_group
from larch.utils import group2dict
from larch.wxlib import plotlabels as plab 
import matplotlib.cm as cm

# Include any other required imports that your methods reference, such as:
try:
    from scipy.integrate import simpson as integrate_method
except ImportError:
    try:
        from scipy.integrate import simps as integrate_method
    except ImportError:
        from scipy.integrate import trapz as integrate_method
        print("Note: Using trapz for integration instead of simpson/simps")

# Ensure that Group is imported as well
from larch import Group

# -----------------------------------------------------------------------
# NOTE: The following helper functions/classes are assumed to be defined:
#   - read_athena(file): reads an Athena project file.
#   - pre_edge(group): performs a pre-edge subtraction.
#   - autobk(group, rbkg=..., kweight=...): background subtraction.
#   - xftf(group, kmin=..., kmax=..., dk=..., kwindow=..., kweight=...): FT processing.
#   - integrate_method(data, x): integrates data over x.
#   - Group(): returns an empty group-like object to hold attributes.
#   - ftwindow(...): from larch.xafs.xafsft, for generating window functions.
#   - (Optionally) plab: for a customized k-axis label.
#
# Make sure these are imported or defined before using XASmu2r.
# -----------------------------------------------------------------------

class XASmu2r:
    def __init__(self, prj_folder):
        """
        Initialize XASmu2r with a folder containing Athena project (.prj) files.
        
        Parameters:
            prj_folder (str or Path): Folder path with .prj files.
        """
        self.prj_folder = Path(prj_folder)
        self.projects = {}
        self.process_athena_projects()  # Load projects during initialization

    def process_athena_projects(self):
        """
        Process Athena project files in self.prj_folder.
        
        This method:
          - Reads all .prj files (using read_athena) and stores the projects in a dictionary.
          - Prints available groups for each project.
          - Prints each group's public attributes.
          - Plots the raw absorption data (mu vs. energy) for each group.
        """
        projects = {}

        # Load projects from all .prj files (search recursively in folder and subfolders)
        if not self.prj_folder.exists():
            raise FileNotFoundError(f"Folder not found: {self.prj_folder}")
        prj_files = list(self.prj_folder.glob("**/*.prj"))
        if not prj_files:
            print(f"WARNING: No .prj files found in {self.prj_folder}")
            print(f"  Searched recursively in: {self.prj_folder}")
        for prj_file in prj_files:
            print(f"Loading project file: {prj_file}")
            project = read_athena(prj_file)  # Assumes read_athena is defined elsewhere
            # Use relative path as key if multiple .prj with same stem exist in subfolders
            key = str(prj_file.relative_to(self.prj_folder)) if prj_file.is_relative_to(self.prj_folder) else prj_file.stem
            key = key.replace("\\", "_").replace("/", "_").replace(".prj", "")
            projects[key] = project

        # List available groups for each project and inspect public attributes
        total_groups = 0
        for prj_name, project in projects.items():
            n_grp = len(project.groups) if hasattr(project, 'groups') else 0
            total_groups += n_grp
            print(f"\nAvailable Groups in Project '{prj_name}': ({n_grp} groups)")
            if n_grp == 0:
                print("  (none - check that the .prj file contains data groups)")
            for group_name in getattr(project, 'groups', {}) or {}:
                print("  ", group_name)
        if total_groups == 0 and projects:
            print("\nWARNING: Projects loaded but 0 groups found. Check .prj file format and content.")

        for proj_name, project in projects.items():
            print(f"\nProcessing Project: {proj_name}")
            for group_name, group in project.groups.items():
                print(f"\nProcessing Group: {group_name}")
                # Print public (non-private) attributes of the group
                for attr in dir(group):
                    if not attr.startswith('_'):
                        print("  ", attr, "->", type(getattr(group, attr)))
                # Plot the raw absorption curve
                # plt.figure()
                # plt.plot(group.energy, group.mu, label='mu')
                # plt.xlabel('Energy (eV)')
                # plt.ylabel(r'$\mu(E)$')
                # plt.legend()
                # plt.title(f"Raw Absorption Data: {proj_name} - {group_name}")
                # plt.show()

        # ===== SUMMARY: process_athena_projects =====
        print("\n" + "="*80)
        print("PROCESS_ATHENA_PROJECTS SUMMARY")
        print("="*80)
        print(f"  Projects loaded:  {len(projects)}")
        print(f"  Total groups:     {total_groups}")
        print("-"*80)
        print("  Group listing:")
        for prj_name, project in projects.items():
            grp_names = list(project.groups.keys()) if hasattr(project, 'groups') else []
            for grp_name in grp_names:
                grp = project.groups[grp_name]
                emin = grp.energy.min() if hasattr(grp, 'energy') else float('nan')
                emax = grp.energy.max() if hasattr(grp, 'energy') else float('nan')
                print(f"    {prj_name}.{grp_name:40s}  E: {emin:.1f} - {emax:.1f} eV")
        print("="*80 + "\n")

        self.projects = projects

    # def batch_process_projects(self):
    #     """
    #     Batch process all Athena projects stored in self.projects.
        
    #     This method:
    #       - Applies pre-edge subtraction on each group.
    #       - Prints e0 and edge_step values.
    #       - Determines a common energy range.
    #       - Plots the normalized absorption curve μ(E) and its derivative.
    #     """
    #     for proj_name, project in self.projects.items():
    #         print(f"\nProject: {proj_name}")
    #         print("Available groups:", list(project.groups.keys()))
            
    #         for name, group in project.groups.items():
    #             pre_edge(group)  # Assumes pre_edge is defined elsewhere
    #             print(f"{name}: e0 = {group.e0}, edge_step = {group.edge_step}")
            
    #         energy_min_list = [group.energy.min() for group in project.groups.values() if hasattr(group, 'energy')]
    #         energy_max_list = [group.energy.max() for group in project.groups.values() if hasattr(group, 'energy')]
    #         global_energy_min = min(energy_min_list) if energy_min_list else None
    #         global_energy_max = max(energy_max_list) if energy_max_list else None
            
    #         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
    #         for name, group in project.groups.items():
    #             if hasattr(group, 'energy') and hasattr(group, 'mu') and hasattr(group, 'norm'):
    #                 # Plot normalized absorption
    #                 ax1.plot(group.energy, group.norm, label=f"{name}")
    #                 ax1.axvline(x=group.e0, color='gray', linestyle='--', alpha=0.7)
    #                 ax1.text(group.e0, np.max(group.norm), f"e0={group.e0:.2f}",
    #                          rotation=90, va='bottom', fontsize=8)
    #                 # Plot first derivative of mu vs energy
    #                 dmu = np.gradient(group.mu, group.energy)
    #                 ax2.plot(group.energy, dmu, label=f"{name}")
    #                 ax2.axvline(x=group.e0, color='gray', linestyle='--', alpha=0.7)
    #                 ax2.text(group.e0, np.max(dmu), f"e0={group.e0:.2f}",
    #                          rotation=90, va='bottom', fontsize=8)
    #             else:
    #                 print(f"Missing data for {name}")
            
    #         ax1.set_ylabel(r'$\mu(E)$ (normalized)')
    #         ax1.set_title(f"Normalized XAFS Data for Project: {proj_name}")
    #         ax1.legend(loc='upper right', fontsize=8)
    #         ax2.set_xlabel('Energy (eV)')
    #         ax2.set_ylabel(r'$d\mu/dE$')
    #         ax2.set_title("First Derivative of μ(E)")
    #         ax2.legend(loc='upper right', fontsize=8)
            
    #         if global_energy_min is not None and global_energy_max is not None:
    #             ax1.set_xlim(global_energy_min, global_energy_max)
    #             ax2.set_xlim(global_energy_min, global_energy_max)
            
    #         plt.tight_layout()
    #         plt.show()
    def batch_process_projects(self, smooth=False, window_length=15, polyorder=3,
                               optimize_preedge=True, pre1=None, pre2=None,
                               shift=-1.4, average=False):
        """
        Batch process all Athena projects stored in self.projects with energy calibration.
        
        This method:
        - Applies pre-edge subtraction on each group with optional pre1/pre2 optimization.
        - Optionally applies Savitzky-Golay filter to smooth normalized absorption data.
        - Applies energy calibration via global shift and/or e0 averaging:
          - If average=True: aligns each group's e0 to the global average e0
          - If shift!=0: applies an additional fixed shift to all groups
          - Both can be combined for e0 alignment plus a global offset
        - Creates a visualization that shows before and after shifting.
        - Plots normalized absorption curves and their derivatives.
        
        Parameters:
        -----------
        smooth : bool, optional
            Whether to apply Savitzky-Golay smoothing to the normalized absorption data. Default is False.
        window_length : int, optional
            Window length for Savitzky-Golay filter (must be odd). Default is 15.
        polyorder : int, optional
            Polynomial order for Savitzky-Golay filter. Default is 3.
        optimize_preedge : bool, optional
            Whether to automatically optimize pre1/pre2 values based on the energy range. Default is True.
        pre1 : float, optional
            Lower bound of pre-edge region (relative to e0, in eV). If None and optimize_preedge is True,
            will be automatically calculated. Typical values: -150 to -200 eV.
        pre2 : float, optional
            Upper bound of pre-edge region (relative to e0, in eV). If None and optimize_preedge is True,
            will be automatically calculated. Typical values: -30 to -50 eV.
        shift : float, optional
            Global energy shift (in eV) to apply to all groups. Default is -1.4 eV.
            This shift is applied in addition to any e0 averaging alignment.
        average : bool, optional
            Whether to align each group's e0 to the global average e0. Default is False.
            If True, each group is shifted so its e0 matches the global average.
            If False, only the global shift is applied to all groups.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from scipy.signal import savgol_filter
    
        # First pass: process every group and collect e₀ values globally.
        all_e0_values = []
        all_groups = []
        
        for proj_name, project in self.projects.items():
            print(f"\nProject: {proj_name}")
            print("Available groups:", list(project.groups.keys()))
            for name, group in project.groups.items():
                # Determine pre1 and pre2 values for pre_edge
                group_pre1 = pre1
                group_pre2 = pre2
                
                if optimize_preedge and (pre1 is None or pre2 is None):
                    # Estimate e0 first for optimization (rough estimate from data midpoint or derivative max)
                    if hasattr(group, 'energy') and hasattr(group, 'mu'):
                        energy = group.energy
                        mu = group.mu
                        
                        # Use derivative maximum as rough e0 estimate
                        dmu = np.gradient(mu, energy)
                        rough_e0_idx = np.argmax(dmu)
                        rough_e0 = energy[rough_e0_idx]
                        
                        # Calculate energy range available before the edge
                        energy_min = energy.min()
                        pre_edge_range = rough_e0 - energy_min
                        
                        if pre1 is None:
                            # Set pre1 to capture most of pre-edge region, but leave some margin
                            # Use 90% of available range or at least -150 eV, whichever is smaller
                            group_pre1 = max(-pre_edge_range * 0.9, -200)
                            # Round to nearest 5 eV for cleaner values
                            group_pre1 = round(group_pre1 / 5) * 5
                        
                        if pre2 is None:
                            # Set pre2 to be close enough to edge to capture trend
                            # but far enough to avoid edge features (typically -30 to -50 eV)
                            # Use a fraction of the pre-edge range, minimum -30 eV
                            group_pre2 = min(-30, group_pre1 / 3)
                            # Round to nearest 5 eV
                            group_pre2 = round(group_pre2 / 5) * 5
                        
                        print(f"  Optimized pre-edge for {name}: pre1={group_pre1:.0f} eV, pre2={group_pre2:.0f} eV")
                
                # Call pre_edge with optimized or user-specified parameters
                if group_pre1 is not None and group_pre2 is not None:
                    pre_edge(group, pre1=group_pre1, pre2=group_pre2)
                else:
                    pre_edge(group)
                
                # Optionally apply Savitzky-Golay filter to normalized data
                if smooth and hasattr(group, 'norm') and hasattr(group, 'energy'):
                    # Store original normalized data if not already stored
                    if not hasattr(group, 'norm_original'):
                        group.norm_original = group.norm.copy()
                    
                    # Ensure window_length is odd (Savitzky-Golay requires odd window length)
                    window_length_adj = window_length
                    if window_length % 2 == 0:
                        window_length_adj = window_length + 1
                        print(f"  Warning: window_length must be odd. Using {window_length_adj} instead of {window_length}.")
                    
                    # Check if data length is sufficient for filtering
                    if len(group.norm) < window_length_adj:
                        print(f"  Warning: Data length ({len(group.norm)}) is shorter than window length ({window_length_adj}). "
                              f"Skipping filter for {name}.")
                    else:
                        # Apply Savitzky-Golay filter
                        group.norm = savgol_filter(group.norm, window_length_adj, polyorder)
                        print(f"  Applied Savitzky-Golay filter (window={window_length_adj}, polyorder={polyorder}) to {name}")
                
                e0_src = "(from Athena file)" if getattr(group, '_e0_from_athena_file', False) else "(auto-detected)"
                print(f"{name}: e0 = {group.e0:.2f} eV {e0_src}, edge_step = {group.edge_step}")
                if hasattr(group, 'e0'):
                    all_e0_values.append(group.e0)
                    # Store reference to group for visualization
                    group._proj_name = proj_name
                    group._group_name = name
                    all_groups.append(group)
        
        # Compute the global average e₀
        if all_e0_values:
            global_avg_e0 = np.mean(all_e0_values)
            print(f"Number of e0 values found: {len(all_e0_values)}")
            print(f"\nGlobal average e0: {global_avg_e0:.2f} eV")
        else:
            global_avg_e0 = None
            if average:
                print("No e0 values found, cannot perform e0 averaging alignment.")
                return
        
        # Report energy calibration mode
        if average and shift != 0:
            print(f"Energy calibration: e0 averaging + global shift of {shift:.2f} eV")
        elif average:
            print(f"Energy calibration: e0 averaging only (align to global average)")
        elif shift != 0:
            print(f"Energy calibration: global shift of {shift:.2f} eV (no e0 averaging)")
        else:
            print(f"Energy calibration: none (no averaging, no shift)")
            
        # Create a special visualization to show the e0 alignment effect
        if all_groups:
            fig_align = plt.figure(figsize=(14, 10))
            
            # Create 2x2 grid: before/after for norm and derivative
            ax1 = plt.subplot(2, 2, 1)  # Before alignment - norm
            ax2 = plt.subplot(2, 2, 2)  # After alignment - norm
            ax3 = plt.subplot(2, 2, 3)  # Before alignment - derivative
            ax4 = plt.subplot(2, 2, 4)  # After alignment - derivative
            
            # Generate color map for groups
            cmap = cm.viridis
            colors = [cmap(i/len(all_groups)) for i in range(len(all_groups))]
            
            # Draw both original and shifted data
            for i, group in enumerate(all_groups):
                color = colors[i]
                label = f"{group._proj_name}.{group._group_name}"
                
                # Calculate the total shift amount
                if average and global_avg_e0 is not None:
                    alignment_shift = global_avg_e0 - group.e0
                else:
                    alignment_shift = 0
                total_shift = alignment_shift + shift
                shifted_energy = group.energy + total_shift
                
                # Draw vertical lines for e0 positions
                if hasattr(group, 'norm') and hasattr(group, 'energy'):
                    # Original data plots
                    ax1.plot(group.energy, group.norm, color=color, alpha=0.7, lw=1.5)
                    ax1.axvline(x=group.e0, color=color, linestyle='--', alpha=0.5)
                    
                    # Derivative before alignment
                    dmu = np.gradient(group.mu, group.energy)
                    ax3.plot(group.energy, dmu, color=color, alpha=0.7, lw=1.5)
                    ax3.axvline(x=group.e0, color=color, linestyle='--', alpha=0.5)
                    
                    # Shifted data plots
                    ax2.plot(shifted_energy, group.norm, color=color, label=label, alpha=0.7, lw=1.5)
                    
                    # Derivative after alignment
                    dmu_shifted = np.gradient(group.mu, shifted_energy)
                    ax4.plot(shifted_energy, dmu_shifted, color=color, label=label, alpha=0.7, lw=1.5)
            
            # Add a vertical line for the global average e0
            for ax in [ax1, ax2, ax3, ax4]:
                if ax == ax1 or ax == ax3:
                    # For "before" plots, show the global average as reference
                    if global_avg_e0 is not None:
                        ax.axvline(x=global_avg_e0, color='red', linestyle='-', linewidth=2,
                                  label=f'Global avg e0: {global_avg_e0:.2f} eV')
                else:
                    # For "after" plots, show where e0s are after calibration
                    if average and global_avg_e0 is not None:
                        target_e0 = global_avg_e0 + shift
                        ax.axvline(x=target_e0, color='red', linestyle='-', linewidth=2,
                                  label=f'Calibrated e0: {target_e0:.2f} eV')
                    elif global_avg_e0 is not None:
                        # No averaging, just show where average landed after shift
                        target_e0 = global_avg_e0 + shift
                        ax.axvline(x=target_e0, color='red', linestyle='-', linewidth=2,
                                  label=f'Avg e0 + shift: {target_e0:.2f} eV')
            
            # Set titles and labels
            calib_type = "Alignment" if average else "Shift"
            ax1.set_title(f'Before {calib_type} - Normalized μ(E)')
            ax2.set_title(f'After {calib_type} - Normalized μ(E)')
            ax3.set_title(f'Before {calib_type} - dμ/dE')
            ax4.set_title(f'After {calib_type} - dμ/dE')
            
            ax1.set_ylabel('Normalized μ(E)')
            ax3.set_ylabel('dμ/dE')
            ax3.set_xlabel('Energy (eV)')
            ax4.set_xlabel('Energy (eV)')
            
            # Find common energy range for zoom
            e0_min = min(all_e0_values)
            e0_max = max(all_e0_values)
            range_window = 50  # eV before and after the e0 range
            
            # Set identical zoom windows around the e0 region
            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_xlim(e0_min - range_window, e0_max + range_window)
                ax.grid(True, alpha=0.3)
            
            # Add legend only to one subplot to avoid clutter
            handles, labels = ax2.get_legend_handles_labels()
            fig_align.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
                           ncol=min(6, len(all_groups) + 1), fontsize=9)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            if average and shift != 0:
                plt.suptitle(f'Energy Calibration: e0 Averaging + {shift:.2f} eV Shift', fontsize=16)
            elif average:
                plt.suptitle('Visualization of e0 Alignment Effect', fontsize=16)
            else:
                plt.suptitle(f'Visualization of {shift:.2f} eV Global Energy Shift', fontsize=16)
            plt.show()
            
        # Continue with standard processing (the existing code for project-by-project plots)
        for proj_name, project in self.projects.items():
            energy_min_list = [group.energy.min() for group in project.groups.values() if hasattr(group, 'energy')]
            energy_max_list = [group.energy.max() for group in project.groups.values() if hasattr(group, 'energy')]
            global_energy_min = min(energy_min_list) if energy_min_list else None
            global_energy_max = max(energy_max_list) if energy_max_list else None
    
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
            for name, group in project.groups.items():
                if hasattr(group, 'energy') and hasattr(group, 'mu') and hasattr(group, 'norm'):
                    # Calculate the total shift: alignment (if average=True) + global shift
                    if average and global_avg_e0 is not None and hasattr(group, 'e0'):
                        alignment_shift = global_avg_e0 - group.e0
                    else:
                        alignment_shift = 0
                    total_shift = alignment_shift + shift
                    group.energy = group.energy + total_shift
                    
                    # Determine the reference e0 for plotting
                    if average and global_avg_e0 is not None:
                        ref_e0 = global_avg_e0 + shift  # Where all e0s should align after calibration
                    else:
                        ref_e0 = group.e0 + shift  # Each group's e0 after shift

                    # Plot the normalized absorption curve using the shifted energy
                    ax1.plot(group.energy, group.norm, label=f"{name}")
                    ax1.axvline(x=ref_e0, color='gray', linestyle='--', alpha=0.7)
                    ax1.text(ref_e0, np.max(group.norm), f"e0={group.e0:.2f}",
                            rotation=90, va='bottom', fontsize=8)

                    # Compute and plot the first derivative with respect to the shifted energy
                    dmu = np.gradient(group.mu, group.energy)
                    ax2.plot(group.energy, dmu, label=f"{name}")
                    ax2.axvline(x=ref_e0, color='gray', linestyle='--', alpha=0.7)
                    ax2.text(ref_e0, np.max(dmu), f"e0={group.e0:.2f}",
                            rotation=90, va='bottom', fontsize=8)
                else:
                    print(f"Missing data for {name}")
            
            ax1.set_ylabel(r'$\mu(E)$ (normalized)')
            calib_label = "e0-aligned" if average else f"shifted {shift:.1f} eV"
            ax1.set_title(f"Normalized XAFS Data for Project: {proj_name} ({calib_label})")
            ax1.legend(loc='upper right', fontsize=8)
            ax2.set_xlabel('Energy (eV)')
            ax2.set_ylabel(r'$d\mu/dE$')
            ax2.set_title(f"First Derivative of μ(E) ({calib_label})")
            ax2.legend(loc='upper right', fontsize=8)
            
            if global_energy_min is not None and global_energy_max is not None:
                ax1.set_xlim(global_energy_min, global_energy_max)
                ax2.set_xlim(global_energy_min, global_energy_max)
            
            plt.tight_layout()
            plt.show()

        # ===== SUMMARY: batch_process_projects =====
        print("\n" + "="*80)
        print("BATCH_PROCESS_PROJECTS SUMMARY")
        print("="*80)
        print(f"  Projects processed:     {len(self.projects)}")
        print(f"  Total groups processed: {len(all_e0_values)}")
        print(f"  Global average e0:      {global_avg_e0:.2f} eV" if global_avg_e0 else "  Global average e0:      N/A")
        print(f"  Energy shift applied:   {shift:.2f} eV")
        print(f"  e0 averaging enabled:   {average}")
        print(f"  Smoothing applied:      {smooth}")
        print("-"*80)
        print("  Group e0 values (after calibration):")
        for group in all_groups:
            proj = group._proj_name
            name = group._group_name
            orig_e0 = group.e0
            if average and global_avg_e0 is not None:
                alignment_shift = global_avg_e0 - orig_e0
            else:
                alignment_shift = 0
            total_shift = alignment_shift + shift
            calibrated_e0 = orig_e0 + total_shift
            print(f"    {proj}.{name:35s}  e0_orig: {orig_e0:.2f} eV  ->  e0_calibrated: {calibrated_e0:.2f} eV  (shift: {total_shift:+.2f})")
        print("="*80 + "\n")
    
    


    def _self_absorption_correction(self, group, thickness, incident_angle, emission_angle):
        """
        Apply a self-absorption correction (for a single group) using the Booth & Bridges approach.
        This method modifies the passed group in place.
        
        Parameters:
          group          : An object containing normalized data with attributes norm, energy, mu, edge_step (and optionally e0)
          thickness      : Effective sample thickness in centimeters.
          incident_angle : Incident angle of the X-ray beam relative to the sample (degrees).
          emission_angle : Emission angle for fluorescence (degrees).
          
        Returns:
          The modified group with new attributes:
              - norm_corrected  : Corrected and normalized absorption data
              - self_abs_correction: The computed correction factor CF
              - mu_pe, mu_other : The estimated absorption components
              - self_abs_thickness, self_abs_incident_angle, self_abs_emission_angle
        """
        import numpy as np

        # Convert angles to radians
        incident_rad = np.radians(incident_angle)
        emission_rad = np.radians(emission_angle)

        # Get the edge energy (use group.e0 if it exists; otherwise, choose the first energy value)
        e0 = getattr(group, 'e0', group.energy[0])

        # Total absorption coefficient
        mu_total = group.mu.copy()

        # Estimate the photoelectric absorption component (μ_pe)
        edge_idx = np.argmin(np.abs(group.energy - e0))
        pre_edge_region = (group.energy < e0 - 30)
        post_edge_region = (group.energy > e0 + 50)

        if np.any(pre_edge_region) and np.any(post_edge_region):
            mu_pe = np.zeros_like(mu_total)
            # Before the edge: a small nonzero value
            mu_pe[group.energy < e0] = 0.01 * group.edge_step
            # After the edge: use norm * edge_step to capture the oscillatory part
            mu_pe[group.energy >= e0] = group.norm[group.energy >= e0] * group.edge_step
            mu_other = mu_total - mu_pe
            mu_other = np.maximum(mu_other, 0.01 * mu_total.min())
        else:
            print("  Warning: Cannot determine pre- or post-edge regions. Using simplified model.")
            mu_pe = group.norm * group.edge_step
            mu_other = mu_total - mu_pe

        # Approximate μ_total(Ef) by μ_other
        mu_at_ef = mu_other

        sin_incident = np.sin(incident_rad) if np.sin(incident_rad) >= 1e-6 else 1e-6
        sin_emission = np.sin(emission_rad) if np.sin(emission_rad) >= 1e-6 else 1e-6

        # Compute the correction factor using the Booth & Bridges formula
        with np.errstate(divide='ignore', invalid='ignore'):
            incident_factor = mu_total / sin_incident
            emission_factor = mu_at_ef / sin_emission
            numerator = mu_pe * (incident_factor + emission_factor)
            denominator = mu_total * mu_pe + emission_factor * mu_total
            CF = numerator / denominator
            CF = np.where(np.isfinite(CF), CF, 1.0)

        # Save original normalized data if not already saved
        if not hasattr(group, 'norm_raw'):
            group.norm_raw = group.norm.copy()

        # Save computed absorption components for later inspection
        group.mu_pe = mu_pe
        group.mu_other = mu_other

        # Apply the correction and renormalize the result
        group.norm_corrected = group.norm * CF
        group.norm_corrected = group.norm_corrected / group.norm_corrected[-1]
        group.self_abs_correction = CF

        # Store parameters used in the correction
        group.self_abs_thickness = thickness
        group.self_abs_incident_angle = incident_angle
        group.self_abs_emission_angle = emission_angle

        return group

    def _parse_formula_from_group_name(self, group_name, project_name=None):
        """
        Parse chemical formula and concentration from project/group name.
        
        First attempts to extract concentration from project_name. If that fails,
        falls back to extracting from group_name.
        
        Explicit concentration patterns (in order of priority):
        - 1p2M, 0p9M (decimal with 'p' instead of '.')
        - 1.2M, 0.9M (decimal with '.')
        - d_1_2_M, d_0_9_M (underscore-separated with 'd' prefix)
        - 100mM (millimolar)
        - 100uM, 100μM (micromolar)
        
        Parameters:
        -----------
        group_name : str
            Name of the group (used for formula and fallback concentration)
        project_name : str, optional
            Name of the project (evaluated first for concentration)
        
        Returns:
        --------
        tuple: (formula, concentration) where formula is a string and concentration is a float
               Concentration is in molar (M)
        
        Raises:
        -------
        ValueError: If no explicit concentration pattern is found in either name
        """
        import re
        
        def _extract_concentration_explicit(name):
            """
            Extract concentration using only explicit molar patterns.
            Returns concentration value in M, or None if not found.
            """
            if not name:
                return None
            
            # Explicit concentration patterns - in order of priority
            # Note: Use (?![A-Za-z0-9]) instead of \b because underscore is a word char
            patterns = [
                # 1p2M, 0p9M - decimal with 'p' instead of '.'
                (r'(\d+)p(\d+)\s*M(?![A-Za-z0-9])', lambda m: float(f"{m.group(1)}.{m.group(2)}")),
                
                # 1.2M, 0.9M - standard decimal notation
                (r'(\d+\.\d+)\s*M(?![A-Za-z0-9])', lambda m: float(m.group(1))),
                
                # d_1_2_M, d_0_9_M - underscore-separated with 'd' prefix
                (r'd[_-](\d+)[_-](\d+)[_-]?M(?![A-Za-z0-9])', lambda m: float(f"{m.group(1)}.{m.group(2)}")),
                
                # 100mM - millimolar
                (r'(\d+(?:\.\d+)?)\s*mM(?![A-Za-z0-9])', lambda m: float(m.group(1)) / 1000.0),
                
                # 100uM, 100μM - micromolar
                (r'(\d+(?:\.\d+)?)\s*[uμ]M(?![A-Za-z0-9])', lambda m: float(m.group(1)) / 1e6),
            ]
            
            for pattern, func in patterns:
                match = re.search(pattern, name, re.IGNORECASE)
                if match:
                    return func(match)
            
            return None
        
        # Common perovskite and lead compound formulas to search for
        known_formulas = [
            'CsPbI3', 'CsPbBr3', 'CsPbCl3',
            'MAPbI3', 'MAPbBr3', 'MAPbCl3',
            'FAPbI3', 'FAPbBr3', 'FAPbCl3',
            'PbI2', 'PbBr2', 'PbCl2', 'PbO', 'PbO2', 'PbS', 'PbSe', 'PbTe',
            'Pb3O4', 'PbSO4', 'PbCO3', 'PbCrO4',
            'Cs4PbI6', 'Cs4PbBr6', 'CsPb2I5', 'CsPb2Br5'
        ]
        
        formula = None
        concentration = None
        
        # Normalize group name for formula matching
        name_upper = group_name.upper().replace(' ', '').replace('-', '').replace('_', '')
        
        # Try to find a known formula in group name
        for f in known_formulas:
            if f.upper() in name_upper:
                formula = f
                break
        
        # If no formula in group name, try project name
        if formula is None and project_name is not None:
            proj_upper = project_name.upper().replace(' ', '').replace('-', '').replace('_', '')
            for f in known_formulas:
                if f.upper() in proj_upper:
                    formula = f
                    break
        
        # If still no known formula, try to extract any chemical formula pattern
        if formula is None:
            formula_pattern = r'([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)'
            matches = re.findall(formula_pattern, group_name)
            pb_matches = [m for m in matches if 'Pb' in m and len(m) > 2]
            if pb_matches:
                formula = max(pb_matches, key=len)
        
        # --- Concentration extraction: project_name first, then group_name ---
        
        # Try project_name first
        if project_name is not None:
            concentration = _extract_concentration_explicit(project_name)
        
        # If not found in project_name, try group_name
        if concentration is None:
            concentration = _extract_concentration_explicit(group_name)
        
        # Raise error if no explicit concentration found
        if concentration is None:
            raise ValueError(
                f"Could not parse explicit concentration from project_name='{project_name}' "
                f"or group_name='{group_name}'. "
                f"Expected patterns: 0p9M, 0.9M, d_0_9_M, 100mM, 100uM"
            )
        
        # Default formula if nothing found
        if formula is None:
            formula = 'PbI2'  # default assumption
            print(f"  Warning: Could not parse formula from '{group_name}', using default: {formula}")
        
        return formula, concentration

    def apply_self_absorption_correction_all(self, elem='Pb', edge='L3', line='La',
                                              anginp=45.0, angout=45.0):
        """
        Apply the self-absorption correction to every group within every project 
        using the larch fluo_corr() function (FLUO algorithm by D. Haskel).
        
        This implementation is designed for solution XAS and accounts for:
        - Actual solution density (concentration-dependent)
        - Solvent contribution to X-ray attenuation (DMSO by default)
        - Solute concentration parsed from project/group name
        
        Parameters:
        -----------
        elem : str, optional
            Atomic symbol of absorbing element. Default is 'Pb'.
        edge : str, optional
            Name of absorption edge. Default is 'L3' for Pb L-III edge.
        line : str, optional
            Name of fluorescence line. Default is 'La' for L-alpha.
        anginp : float, optional
            Input angle in degrees. Default is 45.0.
        angout : float, optional
            Output angle in degrees. Default is 45.0.
        
        The chemical formula is parsed from the project/group name. Project names
        are checked first for concentration, then group names as fallback.
        Group names should contain a chemical formula (e.g., CsPbI3, PbI2) and 
        optionally a concentration (e.g., CsPbI3_10pct, PbI2_0.5M).
        
        Solution density is determined using an empirical linear relationship
        for Pb-halide solutions in DMSO: density = 0.5877 * concentration + 1.1
        
        An effective bulk solution formula is constructed that accounts for
        both solute and solvent contributions to X-ray attenuation.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from larch.xafs.fluo import (fluo_corr, get_density_from_concentration,
                                     build_effective_solution_formula)

        print("Applying self-absorption correction (FLUO algorithm) to all groups...")
        print(f"  Settings: elem={elem}, edge={edge}, line={line}, anginp={anginp}°, angout={angout}°")
        print(f"  Using solution model: accounting for solute + DMSO solvent")

        processed_groups = []
        skipped_groups = []

        # Loop through all projects and groups
        for proj_name, project in self.projects.items():
            for group_name, group in project.groups.items():
                required_attrs = ['energy', 'mu']
                has_required = all(hasattr(group, attr) for attr in required_attrs)
                
                if has_required:
                    try:
                        print(f"\n  Processing '{proj_name}/{group_name}'...")
                        
                        # Parse formula from project/group name (project_name evaluated first)
                        formula, concentration = self._parse_formula_from_group_name(group_name, proj_name)
                        print(f"    Parsed formula: {formula}, concentration: {concentration:.3f} M")
                        
                        # Get solution density from concentration
                        solution_density = get_density_from_concentration(concentration)
                        print(f"    Solution density: {solution_density:.4f} g/mL")
                        
                        # Build effective bulk solution formula (solute + solvent)
                        effective_formula, _ = build_effective_solution_formula(
                            solute_formula=formula,
                            concentration_M=concentration,
                            solution_density=solution_density
                        )
                        print(f"    Effective solution formula: {effective_formula}")
                        
                        # Store original mu and norm for comparison
                        if hasattr(group, 'norm'):
                            group.norm_raw = group.norm.copy()
                        group.mu_raw = group.mu.copy()
                        
                        # Apply fluo_corr with effective formula and solution density
                        fluo_corr(group.energy, group.mu, effective_formula, elem, 
                                  group=group, edge=edge, line=line,
                                  anginp=anginp, angout=angout,
                                  density=solution_density)
                        
                        # Store correction metadata (original solute info)
                        group.self_abs_formula = formula
                        group.self_abs_concentration = concentration
                        group.self_abs_elem = elem
                        group.self_abs_edge = edge
                        group.self_abs_anginp = anginp
                        group.self_abs_angout = angout
                        
                        # Store solution-specific metadata
                        group.solution_concentration_M = concentration
                        group.solution_density_g_ml = solution_density
                        group.solution_formula_effective = effective_formula
                        
                        # Also create norm_corrected for compatibility with visualization
                        if hasattr(group, 'norm_corr'):
                            group.norm_corrected = group.norm_corr.copy()
                        
                        # Calculate a correction factor for visualization (ratio of corrected to original)
                        if hasattr(group, 'mu_corr'):
                            # Avoid division by zero
                            with np.errstate(divide='ignore', invalid='ignore'):
                                group.self_abs_correction = np.where(
                                    group.mu_raw != 0,
                                    group.mu_corr / group.mu_raw,
                                    1.0
                                )
                        
                        processed_groups.append((proj_name, group_name, formula, concentration, solution_density))
                        print(f"    ✓ Applied FLUO correction")
                        
                    except Exception as e:
                        import traceback
                        print(f"    ✗ Error: {str(e)}")
                        traceback.print_exc()
                        skipped_groups.append((proj_name, group_name, f"Error: {str(e)}"))
                else:
                    missing = [attr for attr in required_attrs if not hasattr(group, attr)]
                    skipped_groups.append((proj_name, group_name, f"Missing attributes: {', '.join(missing)}"))
                    print(f"  ✗ Skipped '{group_name}' - missing: {', '.join(missing)}")

        # Summary of processing
        print("\n" + "="*60)
        print("Self-absorption correction summary (FLUO algorithm - solution model):")
        print(f"  Processed {len(processed_groups)} groups")
        print(f"  Skipped {len(skipped_groups)} groups")
        
        if processed_groups:
            print("\nProcessed groups:")
            for proj_name, group_name, formula, conc, density in processed_groups:
                print(f"    {proj_name}/{group_name} -> formula: {formula}, "
                      f"conc: {conc:.3f} M, density: {density:.3f} g/mL")

        # Visualize results for up to 3 groups
        if processed_groups:
            viz_groups = processed_groups[:min(3, len(processed_groups))]
            for proj_name, group_name, formula, conc, density in viz_groups:
                group = self.projects[proj_name].groups[group_name]

                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                
                # Plot 1: Raw and corrected μ(E)
                ax1 = axes[0, 0]
                ax1.plot(group.energy, group.mu_raw, 'b-', label='Original μ(E)', lw=1.5)
                if hasattr(group, 'mu_corr'):
                    ax1.plot(group.energy, group.mu_corr, 'r-', label='Corrected μ(E)', lw=1.5)
                if hasattr(group, 'e0'):
                    ax1.axvline(x=group.e0, color='gray', linestyle='--', label='Edge (E₀)')
                ax1.set_xlabel('Energy (eV)')
                ax1.set_ylabel('μ(E)')
                ax1.set_title('Self-Absorption Correction Effect on μ(E)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Plot 2: Normalized μ(E) before and after correction
                ax2 = axes[0, 1]
                if hasattr(group, 'norm_raw'):
                    ax2.plot(group.energy, group.norm_raw, 'b-', label='Original norm', lw=1.5)
                if hasattr(group, 'norm_corr'):
                    ax2.plot(group.energy, group.norm_corr, 'r-', label='Corrected norm', lw=1.5)
                if hasattr(group, 'e0'):
                    ax2.axvline(x=group.e0, color='gray', linestyle='--')
                ax2.set_xlabel('Energy (eV)')
                ax2.set_ylabel('Normalized μ(E)')
                ax2.set_title('Normalized Absorption')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Plot 3: Correction factor vs. energy
                ax3 = axes[1, 0]
                if hasattr(group, 'self_abs_correction'):
                    ax3.plot(group.energy, group.self_abs_correction, 'g-', lw=2)
                    if hasattr(group, 'e0'):
                        ax3.axvline(x=group.e0, color='gray', linestyle='--')
                    ax3.axhline(y=1.0, color='black', linestyle=':', alpha=0.5)
                ax3.set_xlabel('Energy (eV)')
                ax3.set_ylabel('Correction Factor (μ_corr/μ_raw)')
                ax3.set_title('Self-Absorption Correction Factor')
                ax3.grid(True, alpha=0.3)

                # Plot 4: XANES region comparison
                ax4 = axes[1, 1]
                if hasattr(group, 'e0'):
                    e0 = group.e0
                    xanes_mask = (group.energy > e0 - 20) & (group.energy < e0 + 100)
                    if np.any(xanes_mask):
                        energy_xanes = group.energy[xanes_mask]
                        if hasattr(group, 'norm_raw'):
                            ax4.plot(energy_xanes, group.norm_raw[xanes_mask], 'b-', 
                                     label='Original', lw=1.5, alpha=0.7)
                        if hasattr(group, 'norm_corr'):
                            ax4.plot(energy_xanes, group.norm_corr[xanes_mask], 'r-', 
                                     label='Corrected', lw=1.5, alpha=0.7)
                        ax4.axvline(x=e0, color='gray', linestyle='--')
                ax4.set_xlabel('Energy (eV)')
                ax4.set_ylabel('Normalized μ(E)')
                ax4.set_title('XANES Region Detail')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                eff_formula = getattr(group, 'solution_formula_effective', formula)
                plt.suptitle(f"FLUO Correction: {proj_name}/{group_name}\n"
                            f"Solute: {formula}, Conc: {conc:.3f} M, Density: {density:.3f} g/mL\n"
                            f"Effective formula: {eff_formula[:50]}{'...' if len(eff_formula) > 50 else ''}", 
                            fontsize=10, y=1.05)
                plt.show()

        # ===== SUMMARY: apply_self_absorption_correction_all =====
        print("\n" + "="*80)
        print("APPLY_SELF_ABSORPTION_CORRECTION_ALL SUMMARY")
        print("="*80)
        print(f"  Processed groups:        {len(processed_groups)}")
        print(f"  Skipped groups:          {len(skipped_groups)}")
        print(f"  Absorbing element:       {elem} ({edge} edge)")
        print(f"  Fluorescence line:       {line}")
        print(f"  Input/output angles:     {anginp}° / {angout}°")
        print("-"*80)
        print("  Correction details per group:")
        for proj_name, group_name, formula, conc, density in processed_groups:
            group = self.projects[proj_name].groups[group_name]
            eff_formula = getattr(group, 'solution_formula_effective', 'N/A')
            avg_corr = np.mean(getattr(group, 'self_abs_correction', [1.0]))
            print(f"    {proj_name}.{group_name:35s}")
            print(f"      Formula: {formula}, Conc: {conc:.3f} M, Density: {density:.3f} g/mL")
            print(f"      Effective formula: {eff_formula[:60]}{'...' if len(eff_formula) > 60 else ''}")
            print(f"      Avg correction factor: {avg_corr:.4f}")
        if skipped_groups:
            print("-"*80)
            print("  Skipped groups:")
            for proj_name, group_name, reason in skipped_groups:
                print(f"    {proj_name}.{group_name}: {reason}")
        print("="*80 + "\n")


    def _determine_autobk_clamps(self, group, rbkg=1.0, kmax=12.0, kmin=0.0,
                                  noise_fraction=0.15, plot_diagnostic=False):
        """
        Determine if high/low clamps are needed in autobk and their magnitude.
        
        This method analyzes the χ(k) data to detect conditions requiring clamps:
        - High-k noise dominance → clamp_hi needed
        - Background spline "chasing" noise → higher clamps needed
        - Low-k artifacts or edge effects → clamp_lo needed
        
        Clamp values in autobk typically range:
        - 0: No constraint (spline follows data closely)
        - 2-5: Light clamping (minor constraint)
        - 10-20: Moderate clamping (noticeable constraint)
        - 40-100: Strong clamping (significant stiffness)
        
        Parameters:
        -----------
        group : larch Group
            Group with energy, mu, norm_corrected attributes
        rbkg : float
            R_bkg value to use for testing (default 1.0)
        kmax : float
            Maximum k value to consider (default 12.0)
        kmin : float
            Minimum k value (default 0.0)
        noise_fraction : float
            Fraction of high-k region to use for noise estimation (default 0.15)
        plot_diagnostic : bool
            If True, generate diagnostic plot showing clamp analysis
            
        Returns:
        --------
        dict : Contains:
            - clamp_hi : recommended high-k clamp value (0-100)
            - clamp_lo : recommended low-k clamp value (0-100)
            - needs_hi_clamp : bool indicating if high clamp is recommended
            - needs_lo_clamp : bool indicating if low clamp is recommended
            - snr_hi : SNR in high-k region
            - snr_lo : SNR in low-k region
            - noise_sigma : estimated noise level
            - oscillation_ratio : ratio of signal amplitude to noise
            - justification : text explanation
        """
        import numpy as np
        
        # Create temporary group for analysis
        temp_group = type(group)()
        for attr in ['energy', 'mu', 'norm_corrected', 'e0', 'edge_step']:
            if hasattr(group, attr):
                setattr(temp_group, attr, getattr(group, attr))
        temp_group.norm = temp_group.norm_corrected
        
        # Run baseline autobk WITHOUT clamps to see natural behavior
        try:
            autobk(temp_group, rbkg=rbkg, kmin=kmin, kmax=kmax, kweight=2, 
                   clamp_lo=0, clamp_hi=0)
        except Exception as e:
            print(f"    Warning: autobk failed in clamp analysis: {e}")
            return {
                'clamp_hi': 0, 'clamp_lo': 0,
                'needs_hi_clamp': False, 'needs_lo_clamp': False,
                'justification': f'autobk failed: {e}'
            }
        
        if not hasattr(temp_group, 'k') or not hasattr(temp_group, 'chi'):
            return {
                'clamp_hi': 0, 'clamp_lo': 0,
                'needs_hi_clamp': False, 'needs_lo_clamp': False,
                'justification': 'No k/chi data available'
            }
        
        k = temp_group.k
        chi = temp_group.chi
        k_max_data = np.max(k)
        
        # ========================================================
        # STEP 1: Estimate noise from high-k region
        # ========================================================
        noise_start_k = k_max_data * (1 - noise_fraction)
        noise_mask_hi = k >= noise_start_k
        
        if np.sum(noise_mask_hi) < 5:
            noise_mask_hi = np.zeros_like(k, dtype=bool)
            noise_mask_hi[-10:] = True
        
        chi_noise_hi = chi[noise_mask_hi]
        sigma_hi = np.std(chi_noise_hi)
        if sigma_hi < 1e-12:
            sigma_hi = 1e-12
        
        # ========================================================
        # STEP 2: Compute SNR in different k regions
        # ========================================================
        # High-k region: last 20% of data
        hi_k_threshold = k_max_data * 0.8
        hi_k_mask = k >= hi_k_threshold
        
        # Mid-k region: 40-60% of data (should have good signal)
        mid_k_min = k_max_data * 0.4
        mid_k_max = k_max_data * 0.6
        mid_k_mask = (k >= mid_k_min) & (k <= mid_k_max)
        
        # Low-k region: first 20% of usable data (above kmin)
        lo_k_threshold = max(kmin + 1.0, k_max_data * 0.2)
        lo_k_mask = (k >= kmin) & (k <= lo_k_threshold)
        
        # Compute k-weighted chi for SNR calculation
        kw = 2
        weighted_chi = chi * k**kw
        
        # SNR in each region
        if np.any(hi_k_mask):
            snr_hi = np.mean(np.abs(weighted_chi[hi_k_mask])) / (sigma_hi * np.mean(k[hi_k_mask])**kw + 1e-12)
        else:
            snr_hi = 0
        
        if np.any(mid_k_mask):
            signal_mid = np.mean(np.abs(weighted_chi[mid_k_mask]))
            snr_mid = signal_mid / (sigma_hi * np.mean(k[mid_k_mask])**kw + 1e-12)
        else:
            snr_mid = snr_hi
        
        if np.any(lo_k_mask) and len(chi[lo_k_mask]) > 3:
            # Estimate low-k noise differently (may have edge artifacts)
            chi_lo = chi[lo_k_mask]
            # Use difference from smooth trend as noise estimate
            from scipy.signal import savgol_filter
            try:
                if len(chi_lo) >= 7:
                    chi_lo_smooth = savgol_filter(chi_lo, min(7, len(chi_lo)//2*2+1), 2)
                    sigma_lo = np.std(chi_lo - chi_lo_smooth)
                else:
                    sigma_lo = np.std(chi_lo)
            except:
                sigma_lo = np.std(chi_lo)
            sigma_lo = max(sigma_lo, 1e-12)
            snr_lo = np.mean(np.abs(weighted_chi[lo_k_mask])) / (sigma_lo * np.mean(k[lo_k_mask])**kw + 1e-12)
        else:
            snr_lo = snr_mid
            sigma_lo = sigma_hi
        
        # ========================================================
        # STEP 3: Check for oscillation coherence (noise chasing)
        # ========================================================
        # Compare zero-crossing density in high-k vs mid-k regions
        # More zero crossings in high-k suggests noise dominance
        
        zc_hi = np.sum(np.diff(np.sign(chi[hi_k_mask])) != 0) if np.any(hi_k_mask) else 0
        zc_mid = np.sum(np.diff(np.sign(chi[mid_k_mask])) != 0) if np.any(mid_k_mask) else 0
        
        # Normalize by number of points
        n_hi = np.sum(hi_k_mask)
        n_mid = np.sum(mid_k_mask)
        
        if n_hi > 1 and n_mid > 1:
            zc_density_hi = zc_hi / (n_hi - 1)
            zc_density_mid = zc_mid / (n_mid - 1)
        else:
            zc_density_hi = 0
            zc_density_mid = 0.5  # Assume moderate density
        
        # High-k has more zero crossings than expected → noise dominated
        zc_ratio = zc_density_hi / (zc_density_mid + 1e-6)
        
        # ========================================================
        # STEP 4: Compute background "chasing" metric
        # ========================================================
        # Run autobk with different clamp values and compare residuals
        clamp_test_values = [0, 5, 20, 50]
        residual_changes = []
        
        for clamp_val in clamp_test_values:
            test_group = type(group)()
            for attr in ['energy', 'mu', 'norm_corrected', 'e0', 'edge_step']:
                if hasattr(group, attr):
                    setattr(test_group, attr, getattr(group, attr))
            test_group.norm = test_group.norm_corrected
            
            try:
                autobk(test_group, rbkg=rbkg, kmin=kmin, kmax=kmax, kweight=2,
                       clamp_lo=clamp_val, clamp_hi=clamp_val)
                if hasattr(test_group, 'chi'):
                    # Compute high-k residual amplitude
                    chi_test = test_group.chi
                    if len(chi_test) == len(chi) and np.any(hi_k_mask):
                        residual_hi = np.std(chi_test[hi_k_mask])
                        residual_changes.append((clamp_val, residual_hi))
            except:
                pass
        
        # Large change in high-k residual when adding clamps → chasing behavior
        chasing_metric = 0
        if len(residual_changes) >= 2:
            res_no_clamp = residual_changes[0][1]
            res_with_clamp = residual_changes[-1][1]
            if res_no_clamp > 1e-12:
                chasing_metric = abs(res_with_clamp - res_no_clamp) / res_no_clamp
        
        # ========================================================
        # STEP 5: Determine recommended clamp values
        # ========================================================
        # High clamp decision criteria:
        # 1. Low SNR in high-k region (< 1.5)
        # 2. High zero-crossing ratio (> 1.5 times mid-k)
        # 3. Significant chasing metric (> 0.2)
        
        needs_hi_clamp = False
        clamp_hi = 0
        hi_reasons = []
        
        # SNR-based
        if snr_hi < 0.5:
            needs_hi_clamp = True
            clamp_hi = max(clamp_hi, 40)  # Strong clamp
            hi_reasons.append(f"very low SNR_hi={snr_hi:.2f}")
        elif snr_hi < 1.0:
            needs_hi_clamp = True
            clamp_hi = max(clamp_hi, 20)  # Moderate clamp
            hi_reasons.append(f"low SNR_hi={snr_hi:.2f}")
        elif snr_hi < 1.5:
            needs_hi_clamp = True
            clamp_hi = max(clamp_hi, 5)  # Light clamp
            hi_reasons.append(f"marginal SNR_hi={snr_hi:.2f}")
        
        # Zero-crossing ratio based
        if zc_ratio > 2.0:
            needs_hi_clamp = True
            clamp_hi = max(clamp_hi, 40)
            hi_reasons.append(f"high ZC ratio={zc_ratio:.2f}")
        elif zc_ratio > 1.5:
            needs_hi_clamp = True
            clamp_hi = max(clamp_hi, 20)
            hi_reasons.append(f"elevated ZC ratio={zc_ratio:.2f}")
        
        # Chasing metric based
        if chasing_metric > 0.5:
            needs_hi_clamp = True
            clamp_hi = max(clamp_hi, 30)
            hi_reasons.append(f"significant chasing={chasing_metric:.2f}")
        elif chasing_metric > 0.2:
            needs_hi_clamp = True
            clamp_hi = max(clamp_hi, 10)
            hi_reasons.append(f"moderate chasing={chasing_metric:.2f}")
        
        # SNR ratio (high-k vs mid-k)
        snr_ratio = snr_hi / (snr_mid + 1e-6)
        if snr_ratio < 0.3:
            needs_hi_clamp = True
            clamp_hi = max(clamp_hi, 30)
            hi_reasons.append(f"SNR drop ratio={snr_ratio:.2f}")
        elif snr_ratio < 0.5:
            needs_hi_clamp = True
            clamp_hi = max(clamp_hi, 15)
            hi_reasons.append(f"SNR decline ratio={snr_ratio:.2f}")
        
        # Low clamp decision criteria:
        # 1. Low SNR in low-k region
        # 2. Artifacts near edge (high variance)
        
        needs_lo_clamp = False
        clamp_lo = 0
        lo_reasons = []
        
        if snr_lo < 1.0:
            needs_lo_clamp = True
            clamp_lo = max(clamp_lo, 10)
            lo_reasons.append(f"low SNR_lo={snr_lo:.2f}")
        elif snr_lo < 1.5:
            needs_lo_clamp = True
            clamp_lo = max(clamp_lo, 5)
            lo_reasons.append(f"marginal SNR_lo={snr_lo:.2f}")
        
        # Check for edge artifacts (unusually high amplitude near kmin)
        if np.any(lo_k_mask) and np.any(mid_k_mask):
            amp_lo = np.max(np.abs(chi[lo_k_mask]))
            amp_mid = np.max(np.abs(chi[mid_k_mask]))
            if amp_lo > 2 * amp_mid:  # Low-k amplitude much higher than expected
                needs_lo_clamp = True
                clamp_lo = max(clamp_lo, 20)
                lo_reasons.append(f"edge artifact (amp_ratio={amp_lo/amp_mid:.2f})")
        
        # Cap clamps at reasonable values
        clamp_hi = min(clamp_hi, 100)
        clamp_lo = min(clamp_lo, 100)
        
        # ========================================================
        # STEP 6: Generate justification
        # ========================================================
        justification_parts = []
        
        if needs_hi_clamp:
            justification_parts.append(f"clamp_hi={clamp_hi}: {'; '.join(hi_reasons)}")
        else:
            justification_parts.append("clamp_hi=0: Good high-k SNR")
        
        if needs_lo_clamp:
            justification_parts.append(f"clamp_lo={clamp_lo}: {'; '.join(lo_reasons)}")
        else:
            justification_parts.append("clamp_lo=0: Good low-k data")
        
        justification = " | ".join(justification_parts)
        
        result = {
            'clamp_hi': clamp_hi,
            'clamp_lo': clamp_lo,
            'needs_hi_clamp': needs_hi_clamp,
            'needs_lo_clamp': needs_lo_clamp,
            'snr_hi': snr_hi,
            'snr_mid': snr_mid,
            'snr_lo': snr_lo,
            'snr_ratio_hi_mid': snr_ratio,
            'noise_sigma_hi': sigma_hi,
            'noise_sigma_lo': sigma_lo,
            'zc_ratio': zc_ratio,
            'chasing_metric': chasing_metric,
            'oscillation_ratio': snr_mid / (snr_hi + 1e-6),
            'justification': justification,
            'hi_reasons': hi_reasons,
            'lo_reasons': lo_reasons
        }
        
        # ========================================================
        # STEP 7: Diagnostic plot (optional)
        # ========================================================
        if plot_diagnostic:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Panel 1: k-weighted chi(k) with regions highlighted
            ax1 = axes[0, 0]
            ax1.plot(k, weighted_chi, 'b-', lw=1.5, label=f'k^{kw}·χ(k)')
            
            # Shade different k regions
            if np.any(lo_k_mask):
                ax1.axvspan(k[lo_k_mask][0], k[lo_k_mask][-1], 
                           alpha=0.2, color='green', label='Low-k region')
            if np.any(mid_k_mask):
                ax1.axvspan(k[mid_k_mask][0], k[mid_k_mask][-1], 
                           alpha=0.2, color='blue', label='Mid-k region')
            if np.any(hi_k_mask):
                ax1.axvspan(k[hi_k_mask][0], k[hi_k_mask][-1], 
                           alpha=0.2, color='red', label='High-k region')
            
            ax1.set_xlabel('k (Å⁻¹)')
            ax1.set_ylabel(f'k^{kw}·χ(k)')
            ax1.set_title('χ(k) with Region Analysis')
            ax1.legend(loc='upper right', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Panel 2: SNR profile
            ax2 = axes[0, 1]
            snr_k = np.abs(weighted_chi) / (sigma_hi * k**kw + 1e-12)
            ax2.plot(k, snr_k, 'b-', lw=1.5, label='SNR(k)')
            ax2.axhline(y=1.5, color='orange', linestyle='--', label='SNR=1.5 threshold')
            ax2.axhline(y=1.0, color='red', linestyle=':', label='SNR=1.0 threshold')
            
            # Mark region SNRs
            ax2.axhline(y=snr_hi, color='red', alpha=0.5, linestyle='-', 
                        label=f'SNR_hi={snr_hi:.2f}')
            ax2.axhline(y=snr_mid, color='blue', alpha=0.5, linestyle='-', 
                        label=f'SNR_mid={snr_mid:.2f}')
            
            ax2.set_xlabel('k (Å⁻¹)')
            ax2.set_ylabel('SNR')
            ax2.set_title('Signal-to-Noise Ratio Profile')
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, min(np.max(snr_k) * 1.1, 20))
            
            # Panel 3: Effect of clamps on chi(k)
            ax3 = axes[1, 0]
            colors = plt.cm.viridis(np.linspace(0, 1, len(clamp_test_values)))
            
            for i, (clamp_val, residual) in enumerate(residual_changes):
                test_group = type(group)()
                for attr in ['energy', 'mu', 'norm_corrected', 'e0', 'edge_step']:
                    if hasattr(group, attr):
                        setattr(test_group, attr, getattr(group, attr))
                test_group.norm = test_group.norm_corrected
                
                try:
                    autobk(test_group, rbkg=rbkg, kmin=kmin, kmax=kmax, kweight=2,
                           clamp_lo=clamp_val, clamp_hi=clamp_val)
                    if hasattr(test_group, 'chi'):
                        ax3.plot(test_group.k, test_group.chi * test_group.k**kw, 
                                color=colors[i], alpha=0.7, 
                                label=f'clamp={clamp_val}')
                except:
                    pass
            
            ax3.set_xlabel('k (Å⁻¹)')
            ax3.set_ylabel(f'k^{kw}·χ(k)')
            ax3.set_title('Effect of Clamps on χ(k)')
            ax3.legend(loc='upper right', fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            # Panel 4: Summary metrics
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            summary_text = (
                f"CLAMP ANALYSIS SUMMARY\n"
                f"{'='*40}\n\n"
                f"RECOMMENDED CLAMPS:\n"
                f"  clamp_hi = {clamp_hi}  ({'NEEDED' if needs_hi_clamp else 'not needed'})\n"
                f"  clamp_lo = {clamp_lo}  ({'NEEDED' if needs_lo_clamp else 'not needed'})\n\n"
                f"DIAGNOSTIC METRICS:\n"
                f"  SNR (high-k):  {snr_hi:.3f}\n"
                f"  SNR (mid-k):   {snr_mid:.3f}\n"
                f"  SNR (low-k):   {snr_lo:.3f}\n"
                f"  SNR ratio (hi/mid): {snr_ratio:.3f}\n\n"
                f"  Zero-crossing ratio: {zc_ratio:.3f}\n"
                f"  Chasing metric: {chasing_metric:.3f}\n"
                f"  Noise σ (high-k): {sigma_hi:.2e}\n\n"
                f"REASONING:\n"
            )
            
            if hi_reasons:
                summary_text += f"  High clamp: {'; '.join(hi_reasons)}\n"
            else:
                summary_text += f"  High clamp: Not needed (good SNR)\n"
            
            if lo_reasons:
                summary_text += f"  Low clamp: {'; '.join(lo_reasons)}\n"
            else:
                summary_text += f"  Low clamp: Not needed (good data)\n"
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            fig.suptitle(f'Autobk Clamp Analysis (rbkg={rbkg:.2f}, kmax={kmax:.2f})', 
                        fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
        
        return result

    def _optimize_kmax(self, group, rbkg=1.0, kmin=1.0, kmax_range=(9.0, 14.0), kmax_step=0.25,
                       snr_threshold=1.5, noise_fraction=0.15):
        """
        Optimize kmax using a composite score that balances:
        1. SNR criterion (must be above threshold)
        2. First-shell amplitude preservation  
        3. Low-R artifact suppression
        4. High-R noise minimization
        5. Peak position stability
        
        STRICTLY ENFORCES the SNR cutoff: kmax <= k_cutoff_snr + margin
        
        Uses a composite score for selection among acceptable candidates,
        not just "largest acceptable kmax".
        
        Parameters:
        -----------
        group : larch Group
            Group with energy, mu, norm_corrected attributes
        rbkg : float
            Current rbkg value to use during kmax optimization
        kmin : float
            Current kmin value to use during kmax optimization
        kmax_range : tuple
            (min_kmax, max_kmax) range to search (default 9.0 to 14.0 Å⁻¹)
        kmax_step : float
            Step size for kmax sweep (default 0.25)
        snr_threshold : float
            Minimum acceptable SNR (default 1.5)
        noise_fraction : float
            Fraction of high-k data to use for noise estimation (default 0.15)
            
        Returns:
        --------
        dict : Contains optimal_kmax, composite_score, acceptable_range, snr_at_kmax, justification
        """
        import numpy as np
        
        # Create a temporary group for testing
        temp_group = type(group)()
        for attr in ['energy', 'mu', 'norm_corrected', 'e0', 'edge_step']:
            if hasattr(group, attr):
                setattr(temp_group, attr, getattr(group, attr))
        temp_group.norm = temp_group.norm_corrected
        
        # First, do a baseline autobk to get chi(k) for noise analysis
        autobk(temp_group, rbkg=rbkg, kmin=kmin, kweight=1)
        
        if not hasattr(temp_group, 'k') or not hasattr(temp_group, 'chi'):
            return {'optimal_kmax': 11.0, 'acceptable_range': (10.0, 12.0),
                    'snr_at_kmax': None, 'justification': 'Could not compute chi(k)'}
        
        k = temp_group.k
        chi = temp_group.chi
        
        # Step 1: Estimate noise floor from high-k region
        k_max_data = k.max()
        noise_start_k = k_max_data * (1 - noise_fraction)
        noise_mask = k >= noise_start_k
        
        if np.sum(noise_mask) < 5:
            noise_mask = np.zeros_like(k, dtype=bool)
            noise_mask[-10:] = True
        
        chi_noise_region = chi[noise_mask]
        sigma = np.std(chi_noise_region)
        if sigma < 1e-10:
            sigma = 1e-10
        
        # Compute SNR(k)
        snr_k = np.abs(chi) / sigma
        
        # Find the k value where SNR consistently drops below threshold
        snr_below_threshold = snr_k < snr_threshold
        k_cutoff_snr = k_max_data
        consecutive_low = 0
        for i in range(len(k)):
            if snr_below_threshold[i]:
                consecutive_low += 1
                if consecutive_low >= 3:
                    k_cutoff_snr = k[max(0, i - 2)]
                    break
            else:
                consecutive_low = 0
        
        # Step 2: Sweep kmax values and compute composite score for each
        kmax_values = np.arange(kmax_range[0], min(kmax_range[1], k_max_data - 0.5) + kmax_step, kmax_step)
        
        kmax_results = []
        first_shell_positions = []
        
        # Scoring weights for composite score
        w_firstshell = 1.0   # Reward first-shell amplitude
        w_lowR = 0.5         # Penalize low-R artifacts
        w_highR = 0.3        # Penalize high-R noise
        w_stability = 0.2    # Penalize peak position instability
        
        for kmax_test in kmax_values:
            test_group = type(group)()
            for attr in ['energy', 'mu', 'norm_corrected', 'e0', 'edge_step']:
                if hasattr(group, attr):
                    setattr(test_group, attr, getattr(group, attr))
            test_group.norm = test_group.norm_corrected
            
            try:
                autobk(test_group, rbkg=rbkg, kmin=kmin, kmax=kmax_test, kweight=1)
                
                # Use actual kmin/kmax in FT
                ft_kmin = max(kmin, 2.0)
                ft_kmax = min(kmax_test, 13.0)
                xftf(test_group, kmin=ft_kmin, kmax=ft_kmax, dk=1, kwindow='hanning', kweight=3)
                
                if hasattr(test_group, 'chir_mag') and hasattr(test_group, 'r'):
                    r = test_group.r
                    chir_mag = test_group.chir_mag
                    
                    # First-shell peak (R = rbkg to rbkg+2.0 Å)
                    first_shell_mask = (r > rbkg) & (r < rbkg + 2.0)
                    peak_amp = 0.0
                    peak_r = 0.0
                    if np.any(first_shell_mask):
                        r_fs = r[first_shell_mask]
                        chir_fs = chir_mag[first_shell_mask]
                        peak_idx = np.argmax(chir_fs)
                        peak_r = r_fs[peak_idx]
                        peak_amp = chir_fs[peak_idx]
                    
                    first_shell_positions.append(peak_r)
                    
                    # Low-R artifact (R = 0 to rbkg)
                    low_r_mask = (r > 0) & (r < rbkg)
                    low_r_intensity = 0.0
                    if np.any(low_r_mask):
                        low_r_intensity = integrate_method(chir_mag[low_r_mask], r[low_r_mask])
                    
                    # High-R noise (R = 5.5 to 7.0)
                    high_r_mask = (r > 5.5) & (r < 7.0)
                    high_r_noise = 0.0
                    if np.any(high_r_mask):
                        high_r_noise = np.std(chir_mag[high_r_mask])
                    
                    # Compute SNR at this kmax
                    k_test = test_group.k
                    chi_test = test_group.chi
                    k_mask = k_test <= kmax_test
                    avg_snr = np.mean(np.abs(chi_test[k_mask]) / sigma) if np.any(k_mask) else 0
                    
                    kmax_results.append({
                        'kmax': kmax_test,
                        'peak_r': peak_r,
                        'peak_amp': peak_amp,
                        'low_r_intensity': low_r_intensity,
                        'high_r_noise': high_r_noise,
                        'avg_snr': avg_snr,
                        'group': test_group
                    })
            except Exception as e:
                continue
        
        if not kmax_results:
            return {'optimal_kmax': 11.0, 'acceptable_range': (10.0, 12.0),
                    'snr_at_kmax': None, 'justification': 'No valid kmax results'}
        
        # Step 3: Determine rollover point and stability
        snr_values = [r['avg_snr'] for r in kmax_results]
        kmax_values_arr = [r['kmax'] for r in kmax_results]
        
        # Find where SNR peaks (before rollover)
        if len(snr_values) > 0:
            max_snr_idx = np.argmax(snr_values)
            k_at_max_snr = kmax_values_arr[max_snr_idx]
        else:
            k_at_max_snr = k_cutoff_snr
        
        # STRICT rollover limit: min of (peak SNR k, SNR cutoff + small margin)
        safety_margin = 0.25  # Conservative margin for fluorescence data
        rollover_k = min(k_at_max_snr, k_cutoff_snr + safety_margin)
        
        # Compute peak position stability
        median_pos = np.median(first_shell_positions) if first_shell_positions else 2.5
        
        if len(first_shell_positions) > 1:
            positions = np.array(first_shell_positions)
            stable_mask = np.ones(len(kmax_results), dtype=bool)
            if len(positions) > 3:
                for i in range(2, len(positions)):
                    local_std = np.std(positions[max(0, i-2):i+1])
                    if local_std > 0.05:
                        stable_mask[i] = False
        else:
            stable_mask = np.ones(len(kmax_results), dtype=bool)
        
        # Step 4: Normalize and compute composite score
        max_amp = max(r['peak_amp'] for r in kmax_results) or 1.0
        max_lowR = max(r['low_r_intensity'] for r in kmax_results) or 1.0
        max_highR = max(r['high_r_noise'] for r in kmax_results) or 1.0
        
        acceptable_results = []
        
        for i, result in enumerate(kmax_results):
            kmax_val = result['kmax']
            avg_snr = result['avg_snr']
            
            # STRICT enforcement: must pass all criteria
            snr_ok = avg_snr >= snr_threshold * 0.5
            below_cutoff = kmax_val <= rollover_k  # ENFORCED
            stable = stable_mask[i] if i < len(stable_mask) else True
            
            if snr_ok and stable and below_cutoff:
                # Compute composite score  
                norm_amp = result['peak_amp'] / max_amp
                norm_lowR = result['low_r_intensity'] / max_lowR
                norm_highR = result['high_r_noise'] / max_highR
                peak_shift = abs(result['peak_r'] - median_pos)
                
                # Score: higher is better (maximize first-shell, minimize artifacts/noise)
                score = (w_firstshell * norm_amp - 
                        w_lowR * norm_lowR - 
                        w_highR * norm_highR - 
                        w_stability * peak_shift)
                
                result['composite_score'] = score
                acceptable_results.append(result)
        
        # Step 5: Select best-scoring acceptable candidate
        if acceptable_results:
            acceptable_results.sort(key=lambda x: x['composite_score'], reverse=True)
            best = acceptable_results[0]
            optimal_kmax = best['kmax']
            optimal_snr = best['avg_snr']
            composite_score = best['composite_score']
        else:
            # No acceptable candidates - use conservative fallback
            optimal_kmax = max(kmax_range[0], rollover_k - 0.5)
            optimal_snr = 0
            composite_score = None
            # Find closest result for SNR estimate
            for result in kmax_results:
                if result['kmax'] <= optimal_kmax:
                    optimal_snr = result['avg_snr']
        
        # Find acceptable range
        acceptable_min = kmax_range[0]
        acceptable_max = max([r['kmax'] for r in acceptable_results]) if acceptable_results else kmax_range[0]
        for result in kmax_results:
            if result['avg_snr'] >= snr_threshold * 0.3:
                acceptable_min = result['kmax']
                break
        
        justification = (f"SNR cutoff: {k_cutoff_snr:.2f} Å⁻¹, "
                        f"rollover limit: {rollover_k:.2f} Å⁻¹, "
                        f"SNR at optimal: {optimal_snr:.2f}")
        
        # Check boundary
        if optimal_kmax <= kmax_range[0] + kmax_step:
            justification += " ⚠ AT LOWER BOUNDARY"
        elif optimal_kmax >= rollover_k - kmax_step:
            justification += " (at rollover limit)"
        
        return {
            'optimal_kmax': optimal_kmax,
            'composite_score': composite_score,
            'acceptable_range': (acceptable_min, acceptable_max),
            'snr_at_kmax': optimal_snr,
            'k_cutoff_snr': k_cutoff_snr,
            'rollover_k': rollover_k,
            'noise_sigma': sigma,
            'justification': justification,
            'kmax_results': kmax_results,
            'n_acceptable': len(acceptable_results)
        }
    
    def _optimize_kmin(self, group, rbkg=1.0, kmax=12.0, 
                       kmin_range=(0.8, 2.0), kmin_step=0.02):
        """
        Optimize kmin using a composite score that balances:
        1. Low-R artifact suppression
        2. High-R noise minimization  
        3. First-shell amplitude preservation
        4. Peak position stability
        
        Uses the ACTUAL candidate kmin in the FT window, not a hard-coded value.
        
        Parameters:
        -----------
        group : larch Group
            Group with energy, mu, norm_corrected attributes
        rbkg : float
            rbkg value to use during optimization
        kmax : float
            kmax value to use during optimization
        kmin_range : tuple
            (min_kmin, max_kmin) range to search (default 0.8 to 2.0 Å⁻¹)
        kmin_step : float
            Step size for kmin sweep (default 0.02)
            
        Returns:
        --------
        dict : Contains optimal_kmin, composite_score, low_r_metric, all_results
        """
        import numpy as np
        
        kmin_values = np.arange(kmin_range[0], kmin_range[1] + kmin_step, kmin_step)
        kmin_results = []
        
        # Weights for composite score (tuned for Pb L3 fluorescence data)
        w_lowR = 1.0        # Weight for low-R artifact suppression
        w_highR = 0.3       # Weight for high-R noise
        w_firstshell = 0.5  # Negative weight - reward preserving first shell
        w_stability = 0.2   # Weight for peak position instability
        
        # First pass: collect metrics for all candidates
        first_shell_positions = []
        
        for kmin_test in kmin_values:
            # Create temporary group
            temp_group = type(group)()
            for attr in ['energy', 'mu', 'norm_corrected', 'e0', 'edge_step']:
                if hasattr(group, attr):
                    setattr(temp_group, attr, getattr(group, attr))
            temp_group.norm = temp_group.norm_corrected
            
            try:
                # Apply autobk with test kmin
                autobk(temp_group, rbkg=rbkg, kmin=kmin_test, kmax=kmax, kweight=1)
                
                # Use ACTUAL candidate kmin in FT window (not hard-coded kmin=3)
                ft_kmin = max(kmin_test, 2.0)  # FT kmin should be >= candidate kmin, but at least 2.0
                ft_kmax = min(kmax, 13.0)  # Use actual kmax
                xftf(temp_group, kmin=ft_kmin, kmax=ft_kmax, dk=1, kwindow='hanning', kweight=3)
                
                if hasattr(temp_group, 'chir_mag') and hasattr(temp_group, 'r'):
                    r = temp_group.r
                    chir_mag = temp_group.chir_mag
                    
                    # Compute low-R artifact metric (R = 0 to rbkg)
                    low_r_mask = (r > 0) & (r < rbkg)
                    low_r_intensity = 0.0
                    if np.any(low_r_mask):
                        low_r_intensity = integrate_method(chir_mag[low_r_mask], r[low_r_mask])
                    
                    # Compute high-R noise metric (R = 5.5 to 7.0 Å)
                    high_r_mask = (r > 5.5) & (r < 7.0)
                    high_r_noise = 0.0
                    if np.any(high_r_mask):
                        high_r_noise = np.std(chir_mag[high_r_mask])
                    
                    # Find first-shell peak (R = rbkg to rbkg+2.0 Å)
                    first_shell_mask = (r > rbkg) & (r < rbkg + 2.0)
                    first_shell_amp = 0.0
                    first_shell_pos = 0.0
                    if np.any(first_shell_mask):
                        r_fs = r[first_shell_mask]
                        chir_fs = chir_mag[first_shell_mask]
                        peak_idx = np.argmax(chir_fs)
                        first_shell_amp = chir_fs[peak_idx]
                        first_shell_pos = r_fs[peak_idx]
                    
                    first_shell_positions.append(first_shell_pos)
                    
                    kmin_results.append({
                        'kmin': kmin_test,
                        'low_r_intensity': low_r_intensity,
                        'high_r_noise': high_r_noise,
                        'first_shell_amp': first_shell_amp,
                        'first_shell_pos': first_shell_pos,
                        'group': temp_group
                    })
            except Exception as e:
                continue
        
        if not kmin_results:
            return {'optimal_kmin': 1.0, 'composite_score': None, 'low_r_metric': None,
                    'kmin_results': [], 'justification': 'No valid results'}
        
        # Second pass: compute peak instability relative to median position
        median_pos = np.median(first_shell_positions) if first_shell_positions else 2.5
        
        # Normalize metrics and compute composite score
        # Lower score is better
        max_lowR = max(r['low_r_intensity'] for r in kmin_results) or 1.0
        max_highR = max(r['high_r_noise'] for r in kmin_results) or 1.0
        max_amp = max(r['first_shell_amp'] for r in kmin_results) or 1.0
        
        for result in kmin_results:
            # Normalized metrics (0-1 range)
            norm_lowR = result['low_r_intensity'] / max_lowR
            norm_highR = result['high_r_noise'] / max_highR
            norm_amp = result['first_shell_amp'] / max_amp
            peak_shift = abs(result['first_shell_pos'] - median_pos)
            
            # Composite score: minimize low-R, high-R noise, peak shift; maximize first-shell amp
            result['composite_score'] = (w_lowR * norm_lowR + 
                                         w_highR * norm_highR + 
                                         w_stability * peak_shift - 
                                         w_firstshell * norm_amp)
        
        # Find kmin that minimizes composite score
        kmin_results.sort(key=lambda x: x['composite_score'])
        best = kmin_results[0]
        
        # Build justification
        justification = (f"Composite score: {best['composite_score']:.4f}, "
                        f"low-R: {best['low_r_intensity']:.4f}, "
                        f"1st shell: {best['first_shell_amp']:.3f}")
        
        # Warn if optimal kmin is at boundary
        if best['kmin'] >= kmin_range[1] - kmin_step:
            justification += " ⚠ AT UPPER BOUNDARY"
        elif best['kmin'] <= kmin_range[0] + kmin_step:
            justification += " ⚠ AT LOWER BOUNDARY"
        
        return {
            'optimal_kmin': best['kmin'],
            'composite_score': best['composite_score'],
            'low_r_metric': best['low_r_intensity'],
            'first_shell_amp': best['first_shell_amp'],
            'kmin_results': kmin_results,
            'justification': justification
        }
    
    def _optimize_rbkg(self, group, kmin=1.0, kmax=12.0,
                       rbkg_range=(0.9, 1.5), n_rbkg=50):
        """
        Optimize rbkg using a composite score that balances:
        1. Low-R artifact suppression
        2. First-shell amplitude preservation
        3. First-shell peak position stability
        4. Background smoothness
        
        Uses the ACTUAL candidate kmin/kmax in the FT window.
        
        Parameters:
        -----------
        group : larch Group
            Group with energy, mu, norm_corrected attributes
        kmin : float
            kmin value to use (from prior optimization)
        kmax : float
            kmax value to use (from prior optimization)
        rbkg_range : tuple
            (min_rbkg, max_rbkg) range to search
        n_rbkg : int
            Number of rbkg values to test
            
        Returns:
        --------
        dict : Contains optimal_rbkg, composite_score, low_r_metric, all_results
        """
        import numpy as np
        
        rbkg_values = np.linspace(rbkg_range[0], rbkg_range[1], n_rbkg)
        rbkg_results = []
        
        # Weights for composite score
        w_lowR = 1.0         # Weight for low-R artifact suppression
        w_firstshell = 0.5   # Negative weight - reward preserving first shell (amplitude)
        w_stability = 0.3    # Weight for peak position instability
        w_smoothness = 0.2   # Weight for background non-smoothness
        
        first_shell_positions = []
        
        for rbkg_test in rbkg_values:
            # Create temporary group
            temp_group = type(group)()
            for attr in ['energy', 'mu', 'norm_corrected', 'e0', 'edge_step']:
                if hasattr(group, attr):
                    setattr(temp_group, attr, getattr(group, attr))
            temp_group.norm = temp_group.norm_corrected
            
            try:
                # Apply autobk with test rbkg and current kmin/kmax
                autobk(temp_group, rbkg=rbkg_test, kmin=kmin, kmax=kmax, kweight=1)
                
                # Use ACTUAL kmin/kmax in FT window (not hard-coded)
                ft_kmin = max(kmin, 2.0)  # At least 2.0 for FT
                ft_kmax = min(kmax, 13.0)
                xftf(temp_group, kmin=ft_kmin, kmax=ft_kmax, dk=1, kwindow='hanning', kweight=3)
                
                if hasattr(temp_group, 'chir_mag') and hasattr(temp_group, 'r'):
                    r = temp_group.r
                    chir_mag = temp_group.chir_mag
                    
                    # Compute low-R artifact metric (R = 0 to rbkg_test)
                    low_r_mask = (r > 0) & (r < rbkg_test)
                    low_r_intensity = 0.0
                    if np.any(low_r_mask):
                        low_r_intensity = integrate_method(chir_mag[low_r_mask], r[low_r_mask])
                    
                    # Find first-shell peak (R = rbkg_test to rbkg_test+2.0 Å)
                    first_shell_mask = (r > rbkg_test) & (r < rbkg_test + 2.0)
                    first_shell_amp = 0.0
                    first_shell_pos = 0.0
                    if np.any(first_shell_mask):
                        r_fs = r[first_shell_mask]
                        chir_fs = chir_mag[first_shell_mask]
                        peak_idx = np.argmax(chir_fs)
                        first_shell_amp = chir_fs[peak_idx]
                        first_shell_pos = r_fs[peak_idx]
                    
                    first_shell_positions.append(first_shell_pos)
                    
                    # Compute background smoothness metric
                    # (standard deviation of second derivative of background in k-space)
                    bkg_smoothness = 0.0
                    if hasattr(temp_group, 'bkg') and hasattr(temp_group, 'k'):
                        k = temp_group.k
                        bkg = temp_group.bkg
                        if len(bkg) > 4:
                            # Second derivative of background
                            d2bkg = np.diff(np.diff(bkg))
                            bkg_smoothness = np.std(d2bkg)
                    
                    rbkg_results.append({
                        'rbkg': rbkg_test,
                        'low_r_intensity': low_r_intensity,
                        'first_shell_amp': first_shell_amp,
                        'first_shell_pos': first_shell_pos,
                        'bkg_smoothness': bkg_smoothness,
                        'group': temp_group
                    })
            except Exception:
                continue
        
        if not rbkg_results:
            return {'optimal_rbkg': 1.0, 'composite_score': None, 'low_r_metric': None,
                    'rbkg_results': [], 'justification': 'No valid results'}
        
        # Compute peak instability relative to median position
        median_pos = np.median(first_shell_positions) if first_shell_positions else 2.5
        
        # Normalize metrics and compute composite score
        max_lowR = max(r['low_r_intensity'] for r in rbkg_results) or 1.0
        max_amp = max(r['first_shell_amp'] for r in rbkg_results) or 1.0
        max_smooth = max(r['bkg_smoothness'] for r in rbkg_results) or 1.0
        
        for result in rbkg_results:
            norm_lowR = result['low_r_intensity'] / max_lowR
            norm_amp = result['first_shell_amp'] / max_amp
            peak_shift = abs(result['first_shell_pos'] - median_pos)
            norm_smooth = result['bkg_smoothness'] / max_smooth
            
            # Composite score: minimize low-R, peak shift, non-smoothness; maximize first-shell amp
            result['composite_score'] = (w_lowR * norm_lowR + 
                                         w_stability * peak_shift +
                                         w_smoothness * norm_smooth -
                                         w_firstshell * norm_amp)
        
        # Find rbkg that minimizes composite score
        rbkg_results.sort(key=lambda x: x['composite_score'])
        best = rbkg_results[0]
        
        # Build justification
        justification = (f"Composite score: {best['composite_score']:.4f}, "
                        f"low-R: {best['low_r_intensity']:.4f}, "
                        f"1st shell: {best['first_shell_amp']:.3f}")
        
        # Check boundary
        rbkg_step = (rbkg_range[1] - rbkg_range[0]) / n_rbkg
        if best['rbkg'] >= rbkg_range[1] - rbkg_step:
            justification += " ⚠ AT UPPER BOUNDARY"
        elif best['rbkg'] <= rbkg_range[0] + rbkg_step:
            justification += " ⚠ AT LOWER BOUNDARY"
        
        return {
            'optimal_rbkg': best['rbkg'],
            'composite_score': best['composite_score'],
            'low_r_metric': best['low_r_intensity'],
            'first_shell_amp': best['first_shell_amp'],
            'rbkg_results': rbkg_results,
            'justification': justification
        }

    def background_optimizer(self, fixed_kmax=15.0, kmin_range=(0.8, 2.0),
                            rbkg_range=(0.9, 1.5), plot_results=True,
                            auto_clamp=True, plot_clamp_diagnostic=False,
                            tol_rbkg=0.01, tol_kmin=0.02):
        """
        Perform alternating rbkg/kmin background optimization for all groups.
        
        Uses fixed kmax with alternating optimization: rbkg → kmin → rbkg → kmin
        
        This two-parameter optimization is simpler and more stable than three-parameter
        optimization, since kmax is typically determined by data quality (SNR) rather
        than background fitting considerations.
        
        Parameters:
        -----------
        fixed_kmax : float
            Fixed kmax value for all groups (default 15.0 Å⁻¹)
        kmin_range : tuple
            (min, max) range for kmin search (default 0.8 to 2.0 Å⁻¹)
        rbkg_range : tuple
            (min, max) range for rbkg search (default 0.9 to 1.5)
        plot_results : bool
            Whether to generate diagnostic plots (default True)
        auto_clamp : bool
            Whether to automatically determine clamp_hi/clamp_lo values (default True)
        plot_clamp_diagnostic : bool
            Whether to generate detailed clamp analysis plots (default False)
        tol_rbkg : float
            Convergence tolerance for rbkg (default 0.01)
        tol_kmin : float
            Convergence tolerance for kmin in Å⁻¹ (default 0.02)
            
        Convergence is checked after each rbkg→kmin cycle.
            
        For each group, stores:
        - group.optimalBkg_kmax : kmax value (fixed)
        - group.optimalBkg_kmin : optimal kmin value  
        - group.optimalBkg_rbkg : optimal rbkg value
        - group.optimalBkg_clamp_hi : recommended high-k clamp (0 if not needed)
        - group.optimalBkg_clamp_lo : recommended low-k clamp (0 if not needed)
        - group.optimalBkg_justification : dict with optimization details
        - group.optimalBkg_iteration_history : list of (rbkg, kmin) per step
        - group.optimalBkg_converged : bool indicating if convergence was achieved
        """
        import matplotlib.cm as cm
        import numpy as np

        print("="*70)
        print("ALTERNATING RBKG/KMIN BACKGROUND OPTIMIZATION")
        print("="*70)
        print(f"Optimization sequence: rbkg → kmin → rbkg → kmin")
        print(f"Fixed kmax = {fixed_kmax:.2f} Å⁻¹ (not optimized)")
        print(f"Convergence tolerances: rbkg={tol_rbkg:.3f}, kmin={tol_kmin:.3f} Å⁻¹")
        print(f"kmin range: {kmin_range}")
        print(f"rbkg range: {rbkg_range}")
        print("="*70)

        for proj_name, project in self.projects.items():
            print(f"\n{'='*50}")
            print(f"Processing Project: {proj_name}")
            print(f"{'='*50}")
            
            for group_name, group in project.groups.items():
                if not (hasattr(group, 'energy') and hasattr(group, 'mu') and 
                        hasattr(group, 'norm_corrected')):
                    missing = [attr for attr in ['energy', 'mu', 'norm_corrected'] 
                              if not hasattr(group, attr)]
                    print(f"\n  ✗ Skipping '{group_name}': Missing {', '.join(missing)}")
                    continue
                
                print(f"\n  Processing group: {group_name}")
                print(f"  {'-'*40}")
                
                # ============================================================
                # Initialize with fixed kmax and reasonable starting values
                # ============================================================
                current_kmax = fixed_kmax  # Fixed, not optimized
                current_rbkg = 1.0
                current_kmin = 1.0  # Safer than 0.0 - avoid edge artifacts
                
                # Track iteration history: (rbkg, kmin) tuples
                iteration_history = [(current_rbkg, current_kmin)]
                converged = False
                
                justification = {}
                
                # ============================================================
                # Collect baseline diagnostics
                # ============================================================
                diagnostics = {
                    'edge_step': getattr(group, 'edge_step', float('nan')),
                    'pre_edge_slope': float('nan'),
                    'post_edge_slope': float('nan'),
                    'fixed_kmax': fixed_kmax,
                    'n_kmin_tested': 0,
                    'n_rbkg_tested': 0,
                    'kmin_at_boundary': False,
                    'rbkg_at_boundary': False,
                    'boundary_details': [],
                    'top3_kmin': [],
                    'top3_rbkg': [],
                    'n_steps': 0,
                    'converged': False,
                }
                
                # Compute pre/post edge slopes if available
                if hasattr(group, 'pre_edge') and hasattr(group, 'energy'):
                    try:
                        e0 = getattr(group, 'e0', group.energy[len(group.energy)//2])
                        pre_mask = group.energy < e0 - 20
                        post_mask = (group.energy > e0 + 50) & (group.energy < e0 + 200)
                        if np.sum(pre_mask) > 5:
                            pre_fit = np.polyfit(group.energy[pre_mask], group.mu[pre_mask], 1)
                            diagnostics['pre_edge_slope'] = pre_fit[0]
                        if np.sum(post_mask) > 5:
                            post_fit = np.polyfit(group.energy[post_mask], group.mu[post_mask], 1)
                            diagnostics['post_edge_slope'] = post_fit[0]
                    except Exception:
                        pass
                
                # ============================================================
                # ALTERNATING OPTIMIZATION: rbkg → kmin → rbkg → kmin
                # ============================================================
                print(f"\n  Fixed kmax = {fixed_kmax:.2f} Å⁻¹")
                print(f"  Starting alternating optimization...")
                print(f"  Initial: rbkg={current_rbkg:.3f}, kmin={current_kmin:.2f}")
                
                # Step 1: Optimize rbkg (first pass)
                print(f"\n  [Step 1/4] Optimizing rbkg...")
                prev_rbkg = current_rbkg
                rbkg_result_1 = self._optimize_rbkg(
                    group,
                    kmin=current_kmin,
                    kmax=current_kmax,
                    rbkg_range=rbkg_range,
                    n_rbkg=50
                )
                current_rbkg = rbkg_result_1['optimal_rbkg']
                justification['rbkg_1'] = rbkg_result_1
                print(f"    rbkg: {prev_rbkg:.3f} → {current_rbkg:.3f}")
                iteration_history.append((current_rbkg, current_kmin))
                
                # Step 2: Optimize kmin (first pass)
                print(f"\n  [Step 2/4] Optimizing kmin...")
                prev_kmin = current_kmin
                kmin_result_1 = self._optimize_kmin(
                    group,
                    rbkg=current_rbkg,
                    kmax=current_kmax,
                    kmin_range=kmin_range,
                    kmin_step=0.02
                )
                current_kmin = kmin_result_1['optimal_kmin']
                justification['kmin_1'] = kmin_result_1
                print(f"    kmin: {prev_kmin:.3f} → {current_kmin:.3f}")
                iteration_history.append((current_rbkg, current_kmin))
                
                # Check convergence after first cycle
                d_rbkg_1 = abs(current_rbkg - 1.0)  # Change from initial
                d_kmin_1 = abs(current_kmin - 1.0)  # Change from initial
                
                # Step 3: Optimize rbkg (second pass)
                print(f"\n  [Step 3/4] Optimizing rbkg...")
                prev_rbkg = current_rbkg
                rbkg_result_2 = self._optimize_rbkg(
                    group,
                    kmin=current_kmin,
                    kmax=current_kmax,
                    rbkg_range=rbkg_range,
                    n_rbkg=50
                )
                current_rbkg = rbkg_result_2['optimal_rbkg']
                justification['rbkg_2'] = rbkg_result_2
                justification['rbkg'] = rbkg_result_2  # Store final as main
                d_rbkg = abs(current_rbkg - prev_rbkg)
                print(f"    rbkg: {prev_rbkg:.3f} → {current_rbkg:.3f} (Δ={d_rbkg:.4f})")
                iteration_history.append((current_rbkg, current_kmin))
                
                # Step 4: Optimize kmin (second pass)
                print(f"\n  [Step 4/4] Optimizing kmin...")
                prev_kmin = current_kmin
                kmin_result_2 = self._optimize_kmin(
                    group,
                    rbkg=current_rbkg,
                    kmax=current_kmax,
                    kmin_range=kmin_range,
                    kmin_step=0.02
                )
                current_kmin = kmin_result_2['optimal_kmin']
                justification['kmin_2'] = kmin_result_2
                justification['kmin'] = kmin_result_2  # Store final as main
                d_kmin = abs(current_kmin - prev_kmin)
                print(f"    kmin: {prev_kmin:.3f} → {current_kmin:.3f} (Δ={d_kmin:.4f})")
                iteration_history.append((current_rbkg, current_kmin))
                
                # Check convergence (second pass changes should be small)
                converged = (d_rbkg < tol_rbkg and d_kmin < tol_kmin)
                if converged:
                    print(f"  ✓ CONVERGED (Δrbkg={d_rbkg:.4f} < {tol_rbkg}, Δkmin={d_kmin:.4f} < {tol_kmin})")
                else:
                    print(f"  ⚠ Not fully converged (Δrbkg={d_rbkg:.4f}, Δkmin={d_kmin:.4f})")
                
                # Collect diagnostics
                diagnostics['n_steps'] = 4
                diagnostics['converged'] = converged
                
                if rbkg_result_2.get('rbkg_results'):
                    diagnostics['n_rbkg_tested'] = len(rbkg_result_2['rbkg_results'])
                    sorted_rbkg = sorted(rbkg_result_2['rbkg_results'], 
                                        key=lambda x: x.get('composite_score', float('inf')))[:3]
                    diagnostics['top3_rbkg'] = [
                        {'rbkg': r['rbkg'], 'score': r.get('composite_score', float('nan'))} 
                        for r in sorted_rbkg
                    ]
                
                if kmin_result_2.get('kmin_results'):
                    diagnostics['n_kmin_tested'] = len(kmin_result_2['kmin_results'])
                    sorted_kmin = sorted(kmin_result_2['kmin_results'], 
                                        key=lambda x: x.get('composite_score', float('inf')))[:3]
                    diagnostics['top3_kmin'] = [
                        {'kmin': r['kmin'], 'score': r.get('composite_score', float('nan'))} 
                        for r in sorted_kmin
                    ]
                
                # Check boundary conditions (kmin and rbkg only, kmax is fixed)
                if abs(current_kmin - kmin_range[0]) < 0.03:
                    diagnostics['kmin_at_boundary'] = True
                    diagnostics['boundary_details'].append('kmin at lower bound')
                if abs(current_kmin - kmin_range[1]) < 0.03:
                    diagnostics['kmin_at_boundary'] = True
                    diagnostics['boundary_details'].append('kmin at upper bound')
                if abs(current_rbkg - rbkg_range[0]) < 0.02:
                    diagnostics['rbkg_at_boundary'] = True
                    diagnostics['boundary_details'].append('rbkg at lower bound')
                if abs(current_rbkg - rbkg_range[1]) < 0.02:
                    diagnostics['rbkg_at_boundary'] = True
                    diagnostics['boundary_details'].append('rbkg at upper bound')
                
                # ============================================================
                # Determine clamp values (after convergence)
                # ============================================================
                current_clamp_hi = 0
                current_clamp_lo = 0
                
                if auto_clamp:
                    print(f"\n  Determining clamp values...")
                    clamp_result = self._determine_autobk_clamps(
                        group,
                        rbkg=current_rbkg,
                        kmax=current_kmax,
                        kmin=current_kmin,
                        plot_diagnostic=plot_clamp_diagnostic
                    )
                    
                    current_clamp_hi = clamp_result['clamp_hi']
                    current_clamp_lo = clamp_result['clamp_lo']
                    justification['clamps'] = clamp_result
                    
                    if clamp_result['needs_hi_clamp']:
                        print(f"    clamp_hi = {current_clamp_hi}")
                    if clamp_result['needs_lo_clamp']:
                        print(f"    clamp_lo = {current_clamp_lo}")
                
                # ============================================================
                # Apply optimal parameters to group
                # ============================================================
                print(f"\n  Applying final parameters...")
                
                # Store optimal values
                group.optimalBkg_kmax = current_kmax
                group.optimalBkg_kmin = current_kmin
                group.optimalBkg_rbkg = current_rbkg
                group.optimalBkg_clamp_hi = current_clamp_hi
                group.optimalBkg_clamp_lo = current_clamp_lo
                group.optimalBkg_justification = justification
                group.optimalBkg_iteration_history = iteration_history
                group.optimalBkg_converged = converged
                
                # Also keep old attribute names for compatibility
                group.optimal_rbkg = current_rbkg
                
                # Apply the optimal background subtraction WITH clamps
                group.norm = group.norm_corrected
                autobk(group, rbkg=current_rbkg, kmin=current_kmin, kmax=current_kmax, 
                       kweight=3, clamp_hi=current_clamp_hi, clamp_lo=current_clamp_lo)
                
                # Use actual kmin in FT (not hard-coded kmin=3)
                ft_kmin = max(current_kmin, 2.0)
                ft_kmax = min(current_kmax, 13.0)
                xftf(group, kmin=ft_kmin, kmax=ft_kmax, dk=1, kwindow='hanning', kweight=3)
                
                # Collect first-shell diagnostics after autobk
                if hasattr(group, 'r') and hasattr(group, 'chir_mag'):
                    r = group.r
                    chir = group.chir_mag
                    first_shell_mask = (r > current_rbkg) & (r < current_rbkg + 2.0)
                    if np.any(first_shell_mask):
                        peak_idx = np.argmax(chir[first_shell_mask])
                        r_fs = r[first_shell_mask]
                        diagnostics['first_shell_peak_amp'] = chir[first_shell_mask][peak_idx]
                        diagnostics['first_shell_peak_pos'] = r_fs[peak_idx]
                    diagnostics['low_r_metric'] = justification.get('rbkg', {}).get('low_r_metric', float('nan'))
                
                # Store diagnostics on group
                group.optimalBkg_diagnostics = diagnostics
                
                convergence_status = "✓ CONVERGED" if converged else "⚠ not converged"
                print(f"\n  {convergence_status} for {group_name}")
                print(f"    kmax = {current_kmax:.2f} Å⁻¹ (fixed)")
                print(f"    kmin = {current_kmin:.3f} Å⁻¹")
                print(f"    rbkg = {current_rbkg:.3f}")
                print(f"    steps = {diagnostics['n_steps']}")
                if current_clamp_hi > 0 or current_clamp_lo > 0:
                    print(f"    clamp_hi = {current_clamp_hi}, clamp_lo = {current_clamp_lo}")
                
                # ============================================================
                # Generate diagnostic plots
                # ============================================================
                if plot_results:
                    self._plot_background_optimization(
                        group, group_name, proj_name,
                        current_kmax, current_kmin, current_rbkg,
                        justification
                    )
        
        # ===== SUMMARY: background_optimizer =====
        print("\n" + "="*100)
        print("ALTERNATING RBKG/KMIN BACKGROUND_OPTIMIZER SUMMARY")
        print("="*100)
        print(f"  Optimization settings:")
        print(f"    fixed_kmax:     {fixed_kmax:.2f} Å⁻¹")
        print(f"    kmin_range:     {kmin_range}")
        print(f"    rbkg_range:     {rbkg_range}")
        print(f"    tol: rbkg={tol_rbkg}, kmin={tol_kmin}")
        print("="*100)
        
        # Detailed per-group summary
        for proj_name, project in self.projects.items():
            for group_name, group in project.groups.items():
                if not hasattr(group, 'optimalBkg_kmax'):
                    continue
                    
                print(f"\n{'─'*100}")
                print(f"  GROUP: {proj_name}.{group_name}")
                print(f"{'─'*100}")
                
                # Get stored values
                kmin = getattr(group, 'optimalBkg_kmin', float('nan'))
                kmax = getattr(group, 'optimalBkg_kmax', float('nan'))
                rbkg = getattr(group, 'optimalBkg_rbkg', float('nan'))
                clamp_hi = getattr(group, 'optimalBkg_clamp_hi', 0)
                clamp_lo = getattr(group, 'optimalBkg_clamp_lo', 0)
                diag = getattr(group, 'optimalBkg_diagnostics', {})
                conv = getattr(group, 'optimalBkg_converged', False)
                history = getattr(group, 'optimalBkg_iteration_history', [])
                
                # Convergence status
                conv_status = "✓ CONVERGED" if conv else "⚠ NOT CONVERGED"
                print(f"  STATUS: {conv_status} in {diag.get('n_steps', 0)} steps")
                
                # Data quality metrics
                print(f"\n  DATA QUALITY:")
                print(f"    edge_step: {diag.get('edge_step', float('nan')):.4f}")
                
                # Optimal parameters
                print(f"\n  OPTIMAL PARAMETERS:")
                print(f"    kmax = {kmax:.2f} Å⁻¹ (fixed)  |  kmin = {kmin:.3f} Å⁻¹  |  rbkg = {rbkg:.3f}")
                if clamp_hi > 0 or clamp_lo > 0:
                    print(f"    clamp_hi = {clamp_hi}  |  clamp_lo = {clamp_lo}")
                
                # Iteration history
                if len(history) > 1:
                    print(f"\n  OPTIMIZATION HISTORY:")
                    print(f"    {'step':>4s}  {'rbkg':>7s}  {'kmin':>7s}")
                    for i, (r, kn) in enumerate(history):
                        print(f"    {i:>4d}  {r:>7.3f}  {kn:>7.3f}")
                
                # Optimization metrics
                print(f"\n  FINAL METRICS:")
                print(f"    low-R artifact metric: {diag.get('low_r_metric', float('nan')):.4f}")
                print(f"    first-shell peak: amp = {diag.get('first_shell_peak_amp', float('nan')):.3f}, "
                      f"pos = {diag.get('first_shell_peak_pos', float('nan')):.2f} Å")
                
                # Boundary status
                boundary_hit = (diag.get('kmin_at_boundary', False) or 
                               diag.get('rbkg_at_boundary', False))
                print(f"\n  BOUNDARY STATUS: {'⚠ BOUNDARY HIT' if boundary_hit else '✓ Not at boundary'}")
                if diag.get('boundary_details'):
                    for detail in diag['boundary_details']:
                        print(f"    • {detail}")
                
                # Top 3 candidates
                top3 = diag.get('top3_rbkg', [])
                if len(top3) >= 2:
                    best_score = top3[0].get('score', float('nan'))
                    second_score = top3[1].get('score', float('nan'))
                    delta = abs(second_score - best_score) if not (np.isnan(best_score) or np.isnan(second_score)) else float('nan')
                    print(f"\n  SOLUTION CONFIDENCE (rbkg):")
                    print(f"    best = {best_score:.4f}, 2nd = {second_score:.4f}, Δ = {delta:.4f}")
                    if delta < 0.01:
                        print(f"    ⚠ Low score separation - optimization may be weakly informative")
        
        print("\n" + "="*100)
        print("QUICK REFERENCE TABLE")
        print("="*100)
        print(f"  {'Group':<40s} {'kmin':>6s} {'kmax':>6s} {'rbkg':>6s} {'steps':>5s} {'conv':>6s} {'bound':>8s}")
        print("-"*100)
        for proj_name, project in self.projects.items():
            for group_name, group in project.groups.items():
                if not hasattr(group, 'optimalBkg_kmax'):
                    continue
                kmin = getattr(group, 'optimalBkg_kmin', float('nan'))
                kmax = getattr(group, 'optimalBkg_kmax', float('nan'))
                rbkg = getattr(group, 'optimalBkg_rbkg', float('nan'))
                diag = getattr(group, 'optimalBkg_diagnostics', {})
                conv = getattr(group, 'optimalBkg_converged', False)
                n_steps = diag.get('n_steps', 0)
                boundary = 'YES' if (diag.get('kmin_at_boundary') or 
                                    diag.get('rbkg_at_boundary')) else 'no'
                conv_str = '✓' if conv else '⚠'
                gname = f"{proj_name}.{group_name}"[:40]
                print(f"  {gname:<40s} {kmin:>6.2f} {kmax:>6.2f} {rbkg:>6.3f} {n_steps:>5d} {conv_str:>6s} {boundary:>8s}")
        print("="*100 + "\n")
    
    def _plot_background_optimization(self, group, group_name, proj_name,
                                       optimal_kmax, optimal_kmin, optimal_rbkg,
                                       justification):
        """
        Generate comprehensive diagnostic plots for background optimization.
        """
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid: 3 rows x 3 cols
        # Row 1: kmax optimization results
        # Row 2: kmin and rbkg optimization results
        # Row 3: Final χ(k) and χ(R)
        
        ax1 = plt.subplot(3, 3, 1)  # SNR analysis
        ax2 = plt.subplot(3, 3, 2)  # kmax sweep χ(R)
        ax3 = plt.subplot(3, 3, 3)  # First shell stability
        ax4 = plt.subplot(3, 3, 4)  # kmin sweep metric
        ax5 = plt.subplot(3, 3, 5)  # rbkg sweep metric
        ax6 = plt.subplot(3, 3, 6)  # rbkg sweep χ(R)
        ax7 = plt.subplot(3, 3, 7)  # Final χ(k)
        ax8 = plt.subplot(3, 3, 8)  # Final χ(R)
        ax9 = plt.subplot(3, 3, 9)  # Parameter summary
        
        # Plot 1: SNR analysis (if kmax was optimized)
        if 'kmax' in justification and 'kmax_results' in justification['kmax']:
            kmax_data = justification['kmax']
            if kmax_data.get('kmax_results'):
                kmax_vals = [r['kmax'] for r in kmax_data['kmax_results']]
                snr_vals = [r['avg_snr'] for r in kmax_data['kmax_results']]
                ax1.plot(kmax_vals, snr_vals, 'b-o', markersize=4)
                ax1.axhline(y=1.5, color='r', linestyle='--', label='SNR threshold')
                ax1.axvline(x=optimal_kmax, color='g', linestyle='-', lw=2, 
                           label=f'Optimal: {optimal_kmax:.2f}')
                if kmax_data.get('k_cutoff_snr'):
                    ax1.axvline(x=kmax_data['k_cutoff_snr'], color='orange', 
                               linestyle=':', label=f"SNR cutoff: {kmax_data['k_cutoff_snr']:.2f}")
                ax1.set_xlabel('kmax (Å⁻¹)')
                ax1.set_ylabel('Average SNR')
                ax1.set_title('kmax: SNR Analysis')
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'kmax optimization\nnot performed', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('kmax: SNR Analysis')
        
        # Plot 2: kmax sweep χ(R) comparison
        if 'kmax' in justification and 'kmax_results' in justification['kmax']:
            kmax_data = justification['kmax']
            if kmax_data.get('kmax_results'):
                cmap = cm.viridis
                results = kmax_data['kmax_results']
                kmax_vals = [r['kmax'] for r in results]
                norm_c = plt.Normalize(min(kmax_vals), max(kmax_vals))
                for result in results[::max(1, len(results)//8)]:  # Plot subset
                    if hasattr(result['group'], 'r') and hasattr(result['group'], 'chir_mag'):
                        color = cmap(norm_c(result['kmax']))
                        ax2.plot(result['group'].r, result['group'].chir_mag, 
                                color=color, alpha=0.7, lw=1)
                ax2.set_xlim(0, 6)
                ax2.set_xlabel('R (Å)')
                ax2.set_ylabel('|χ(R)|')
                ax2.set_title('kmax sweep: χ(R)')
                ax2.axvspan(0, 1.5, color='lightgray', alpha=0.3)
        
        # Plot 3: First shell peak stability
        if 'kmax' in justification and 'kmax_results' in justification['kmax']:
            kmax_data = justification['kmax']
            if kmax_data.get('kmax_results'):
                results = kmax_data['kmax_results']
                kmax_vals = [r['kmax'] for r in results]
                peak_rs = [r['peak_r'] for r in results]
                peak_amps = [r['peak_amp'] for r in results]
                ax3.plot(kmax_vals, peak_rs, 'b-o', markersize=4, label='Peak position')
                ax3.axvline(x=optimal_kmax, color='g', linestyle='-', lw=2)
                ax3.set_xlabel('kmax (Å⁻¹)')
                ax3.set_ylabel('First shell peak R (Å)', color='b')
                ax3.tick_params(axis='y', labelcolor='b')
                ax3_twin = ax3.twinx()
                ax3_twin.plot(kmax_vals, peak_amps, 'r-s', markersize=4, label='Peak amplitude')
                ax3_twin.set_ylabel('Peak amplitude', color='r')
                ax3_twin.tick_params(axis='y', labelcolor='r')
                ax3.set_title('First Shell Stability')
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: kmin sweep χ(R) comparison
        if 'kmin' in justification and 'kmin_results' in justification['kmin']:
            kmin_data = justification['kmin']
            if kmin_data.get('kmin_results'):
                cmap = cm.coolwarm
                results = kmin_data['kmin_results']
                # Sort by kmin for consistent coloring
                results_sorted = sorted(results, key=lambda x: x['kmin'])
                kmin_vals = [r['kmin'] for r in results_sorted]
                norm_c = plt.Normalize(min(kmin_vals), max(kmin_vals))
                # Plot subset of χ(R) curves to show the progression
                for result in results_sorted[::max(1, len(results_sorted)//10)]:
                    if hasattr(result['group'], 'r') and hasattr(result['group'], 'chir_mag'):
                        color = cmap(norm_c(result['kmin']))
                        ax4.plot(result['group'].r, result['group'].chir_mag,
                                color=color, alpha=0.7, lw=1)
                # Highlight optimal kmin result
                for result in results_sorted:
                    if abs(result['kmin'] - optimal_kmin) < 0.02:
                        if hasattr(result['group'], 'r') and hasattr(result['group'], 'chir_mag'):
                            ax4.plot(result['group'].r, result['group'].chir_mag,
                                    'k-', lw=2, label=f'Optimal: {optimal_kmin:.3f}')
                        break
                ax4.set_xlim(0, 4)
                ax4.set_xlabel('R (Å)')
                ax4.set_ylabel('|χ(R)|')
                ax4.set_title('kmin sweep: χ(R)')
                ax4.axvspan(0, 1.5, color='lightgray', alpha=0.3, label='Low-R region')
                ax4.legend(fontsize=8, loc='upper right')
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
                sm.set_array([])
                plt.colorbar(sm, ax=ax4, label='kmin (Å⁻¹)')
        else:
            ax4.text(0.5, 0.5, 'kmin optimization\nnot performed',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('kmin sweep: χ(R)')
        
        # Plot 5: rbkg sweep χ(R) comparison (low-R focus)
        if 'rbkg' in justification and 'rbkg_results' in justification['rbkg']:
            rbkg_data = justification['rbkg']
            if rbkg_data.get('rbkg_results'):
                cmap = cm.plasma
                results = rbkg_data['rbkg_results']
                # Sort by rbkg for consistent coloring
                results_sorted = sorted(results, key=lambda x: x['rbkg'])
                rbkg_vals = [r['rbkg'] for r in results_sorted]
                norm_c = plt.Normalize(min(rbkg_vals), max(rbkg_vals))
                # Plot subset of χ(R) curves
                for result in results_sorted[::max(1, len(results_sorted)//10)]:
                    if hasattr(result['group'], 'r') and hasattr(result['group'], 'chir_mag'):
                        color = cmap(norm_c(result['rbkg']))
                        ax5.plot(result['group'].r, result['group'].chir_mag,
                                color=color, alpha=0.7, lw=1)
                # Highlight optimal rbkg result
                for result in results_sorted:
                    if abs(result['rbkg'] - optimal_rbkg) < 0.02:
                        if hasattr(result['group'], 'r') and hasattr(result['group'], 'chir_mag'):
                            ax5.plot(result['group'].r, result['group'].chir_mag,
                                    'k-', lw=2, label=f'Optimal: {optimal_rbkg:.3f}')
                        break
                ax5.set_xlim(0, 4)
                ax5.set_xlabel('R (Å)')
                ax5.set_ylabel('|χ(R)|')
                ax5.set_title('rbkg sweep: χ(R) (low-R focus)')
                ax5.axvspan(0, 1.5, color='lightgray', alpha=0.3, label='Low-R region')
                ax5.legend(fontsize=8, loc='upper right')
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
                sm.set_array([])
                plt.colorbar(sm, ax=ax5, label='rbkg')
        
        # Plot 6: rbkg sweep χ(k) comparison (shows effect on background subtraction)
        if 'rbkg' in justification and 'rbkg_results' in justification['rbkg']:
            rbkg_data = justification['rbkg']
            if rbkg_data.get('rbkg_results'):
                cmap = cm.plasma
                results = rbkg_data['rbkg_results']
                results_sorted = sorted(results, key=lambda x: x['rbkg'])
                rbkg_vals = [r['rbkg'] for r in results_sorted]
                norm_c = plt.Normalize(min(rbkg_vals), max(rbkg_vals))
                for result in results_sorted[::max(1, len(results_sorted)//10)]:
                    if hasattr(result['group'], 'k') and hasattr(result['group'], 'chi'):
                        color = cmap(norm_c(result['rbkg']))
                        # Plot k²-weighted χ(k) for visibility
                        chi_k2 = result['group'].chi * result['group'].k**2
                        ax6.plot(result['group'].k, chi_k2,
                                color=color, alpha=0.7, lw=1)
                # Highlight optimal rbkg result
                for result in results_sorted:
                    if abs(result['rbkg'] - optimal_rbkg) < 0.02:
                        if hasattr(result['group'], 'k') and hasattr(result['group'], 'chi'):
                            chi_k2 = result['group'].chi * result['group'].k**2
                            ax6.plot(result['group'].k, chi_k2,
                                    'k-', lw=2, label=f'Optimal: {optimal_rbkg:.3f}')
                        break
                ax6.set_xlim(0, optimal_kmax + 1)
                ax6.set_xlabel('k (Å⁻¹)')
                ax6.set_ylabel('χ(k)·k²')
                ax6.set_title('rbkg sweep: χ(k)')
                ax6.legend(fontsize=8, loc='upper right')
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
                sm.set_array([])
                plt.colorbar(sm, ax=ax6, label='rbkg')
        
        # Plot 7: Final χ(k)
        if hasattr(group, 'k') and hasattr(group, 'chi'):
            for kw in [1, 2, 3]:
                chi_kw = group.chi * group.k**kw
                label = f'k={kw}' if kw > 1 else f'k¹'
                alpha = 0.5 + 0.15 * kw
                ax7.plot(group.k, chi_kw, label=f'k^{kw} weighted', alpha=alpha)
            ax7.set_xlabel('k (Å⁻¹)')
            ax7.set_ylabel('χ(k)·kⁿ')
            ax7.set_title('Final χ(k) - Multiple k-weights')
            ax7.legend(fontsize=8)
            ax7.grid(True, alpha=0.3)
            ax7.axvline(x=optimal_kmax, color='r', linestyle='--', alpha=0.5,
                       label=f'kmax={optimal_kmax:.1f}')
        
        # Plot 8: Final χ(R)
        if hasattr(group, 'r') and hasattr(group, 'chir_mag'):
            ax8.plot(group.r, group.chir_mag, 'b-', lw=2, label='|χ(R)|')
            if hasattr(group, 'chir_re'):
                ax8.plot(group.r, group.chir_re, 'g-', lw=1, alpha=0.7, label='Re[χ(R)]')
            ax8.set_xlim(0, 6)
            ax8.set_xlabel('R (Å)')
            ax8.set_ylabel('|χ(R)|')
            ax8.set_title('Final χ(R)')
            ax8.axvspan(0, 1.5, color='lightgray', alpha=0.3)
            ax8.legend(fontsize=8)
            ax8.grid(True, alpha=0.3)
        
        # Plot 9: Parameter summary text
        ax9.axis('off')
        summary_text = (
            f"OPTIMIZATION SUMMARY\n"
            f"{'='*30}\n\n"
            f"Optimal Parameters:\n"
            f"  kmax = {optimal_kmax:.2f} Å⁻¹\n"
            f"  kmin = {optimal_kmin:.3f} Å⁻¹\n"
            f"  rbkg = {optimal_rbkg:.3f}\n\n"
        )
        
        if 'kmax' in justification:
            kmax_j = justification['kmax']
            if kmax_j.get('acceptable_range'):
                summary_text += (f"kmax Details:\n"
                               f"  Range: {kmax_j['acceptable_range'][0]:.2f} - "
                               f"{kmax_j['acceptable_range'][1]:.2f}\n")
            if kmax_j.get('snr_at_kmax'):
                summary_text += f"  SNR: {kmax_j['snr_at_kmax']:.2f}\n"
        
        if 'kmin' in justification:
            kmin_j = justification['kmin']
            if kmin_j.get('low_r_metric'):
                summary_text += f"\nkmin Low-R metric: {kmin_j['low_r_metric']:.4f}\n"
        
        if 'rbkg' in justification:
            rbkg_j = justification['rbkg']
            if rbkg_j.get('low_r_metric'):
                summary_text += f"rbkg Low-R metric: {rbkg_j['low_r_metric']:.4f}\n"
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.suptitle(f"Background Optimization: {proj_name} / {group_name}",
                    fontsize=14, y=1.02)
        plt.show()

    def visualize_kw(self):
        """
        Visualize χ(k) for different k-weight values (1, 2, and 3) for each group.
        
        Creates a figure with three side-by-side subplots for the k-weighted data.
        """
        try:
            k_label = plab.k
        except NameError:
            k_label = "k"

        for proj_name, project in self.projects.items():
            print(f"\nProcessing Project: {proj_name}")
            for group_name, group in project.groups.items():
                if hasattr(group, 'k') and hasattr(group, 'chi'):
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=False, sharex=True)
                    for i, kweight in enumerate([1, 2, 3]):
                        axs[i].plot(group.k, group.chi * group.k**kweight, 
                                    color=f'C{i}', linewidth=2)
                        axs[i].set_xlabel(k_label, fontsize=11)
                        axs[i].set_ylabel(f"$χ(k) \\cdot k^{kweight}$", fontsize=11)
                        axs[i].set_title(f"k-weight = {kweight}", fontsize=12)
                        axs[i].grid(True, linestyle='--', alpha=0.6)
                        if hasattr(group, 'k') and len(group.k) > 0:
                            axs[i].set_xlim(min(group.k), max(group.k))
                    plt.suptitle(f"χ(k) with Different k-Weights: {proj_name} - {group_name}", 
                                 fontsize=14, y=1.02)
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"Skipping group {group_name} in {proj_name}: Missing 'k' or 'chi' attribute")

    def smooth_and_taper_chi_all(self, window_length=11, polyorder=3, taper_delta=1.0):
        """
        Loop through all projects and groups to smooth and taper chi(k) data.
        
        Parameters:
            window_length (int): Window length for the Savitzky–Golay filter (must be odd).
            polyorder (int): Polynomial order for the Savitzky–Golay filter.
            taper_delta (float): Width (in k units) of the taper region at the high-k end.
        
        For each group having attributes 'k' and 'chi':
          - Smooth chi(k) using a Savitzky–Golay filter (if the data length permits),
          - Create and apply a taper at the high‑k end,
          - Save the processed arrays (chi_smoothed and chi_tapered) to the group along with the smoothing parameters,
          - Generate a diagnostic plot comparing original, smoothed, and tapered χ(k),
          - And print out the magnitude changes due to smoothing and tapering.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.signal import savgol_filter

        # Define an inner helper function to smooth and taper a single (k, chi) array pair.
        def smooth_and_taper_chi(k, chi, window_length, polyorder, taper_delta):
            # Check if data length is sufficient for smoothing
            if len(chi) < window_length:
                print(f"Warning: Data length ({len(chi)}) is shorter than window length ({window_length}). Using original data.")
                chi_smoothed = chi.copy()
            else:
                chi_smoothed = savgol_filter(chi, window_length, polyorder)
            
            # Create a taper array that forces the signal to go to zero at the high-k edge.
            kmax = np.max(k)
            taper = np.ones_like(k)
            mask = k >= (kmax - taper_delta)
            if np.any(mask):
                taper[mask] = 0.5 * (1 + np.cos(np.pi * (k[mask] - (kmax - taper_delta)) / taper_delta))
            
            chi_tapered = chi_smoothed * taper
            return chi_smoothed, chi_tapered

        # Loop over all projects and groups stored in self.projects
        for proj_name, project in self.projects.items():
            print(f"\nProcessing chi(k) smoothing and tapering for Project: {proj_name}")
            
            # Find groups that have the required chi(k) data
            valid_groups = [name for name, grp in project.groups.items() 
                            if hasattr(grp, 'k') and hasattr(grp, 'chi')]
            print(f"Found {len(valid_groups)} groups with chi(k) data")
            
            for group_name, group in project.groups.items():
                if hasattr(group, 'k') and hasattr(group, 'chi'):
                    print(f"  Processing group: {group_name}")
                    k = group.k
                    chi = group.chi
                    
                    try:
                        chi_smoothed, chi_tapered = smooth_and_taper_chi(
                            k, chi, window_length, polyorder, taper_delta)
                        
                        # Store the processed data and parameters in the group
                        group.chi_smoothed = chi_smoothed
                        group.chi_tapered = chi_tapered
                        group.taper_parameters = {
                            'window_length': window_length,
                            'polyorder': polyorder,
                            'taper_delta': taper_delta
                        }
                        
                        # Plot the original, smoothed, and tapered chi(k)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(k, chi, label="Original χ(k)", alpha=0.6, color='black', linestyle='--', zorder=3)
                        ax.plot(k, chi_smoothed, label="Smoothed χ(k)", color='blue', linestyle='-', linewidth=1.5)
                        ax.plot(k, chi_tapered, label="Smoothed & Tapered χ(k)", color='red', linestyle='-', linewidth=2)
                        
                        # Highlight the taper region
                        kmax = np.max(k)
                        taper_start = kmax - taper_delta
                        ax.axvspan(taper_start, kmax, alpha=0.15, color='red', label="Taper region")
                        
                        # Use plab.k for a nice label if available; otherwise, default to "k"
                        try:
                            k_label = plab.k
                        except NameError:
                            k_label = "k"
                        
                        ax.set_xlabel(k_label, fontsize=12)
                        ax.set_ylabel("χ(k)", fontsize=12)
                        ax.set_title(f"χ(k) Smoothing and Tapering\nProject: {proj_name} - Group: {group_name}", fontsize=14)
                        ax.grid(True, alpha=0.3)
                        ax.legend(loc='best')
                        plt.tight_layout()
                        plt.show()
                        
                        # Print out diagnostic information
                        chi_max_diff = np.max(np.abs(chi - chi_smoothed))
                        taper_effect = np.max(np.abs(chi_smoothed - chi_tapered))
                        print(f"    Max change from smoothing: {chi_max_diff:.6f}")
                        print(f"    Max change from tapering: {taper_effect:.6f}")
                    except Exception as e:
                        print(f"    Error processing {group_name}: {str(e)}")
                else:
                    print(f"  Skipping group {group_name}: Missing 'k' or 'chi' attribute")
                    

    

    def ft_processing_batch(self, kmax_range=(10.0, 14.0), kmin_zc_range=(1.0, 3.0),
                            kweight=3, dk=3, n_kmax=10, plot_results=True):
        """
        Perform batch Fourier Transform (FT) processing with simplified kmin/kmax optimization.
        
        Parameters:
            kmax_range : tuple (default (10.0, 14.0))
                (kmax_lower, kmax_upper) range for kmax candidates.
            kmin_zc_range : tuple (default (2.0, 4.0))
                (kmin_lower, kmin_upper) range for zero-crossing kmin candidates.
            kweight : int (default 3)
                k-weighting for FT.
            dk : float (default 3)
                Window parameter for FT.
            n_kmax : int (default 10)
                Number of kmax candidates to evaluate.
            plot_results : bool (default True)
                If True, plot FT results.
          
        Output attributes stored on each group:
            best_kmin, best_kmax: Best FT parameters
            best_r, best_chir_mag, best_chir_re: FT results
        """
        kmax_lower, kmax_upper = kmax_range
        kmin_zc_lower, kmin_zc_upper = kmin_zc_range
        
        for proj_name, project in self.projects.items():
            for group_name, group in project.groups.items():
                if not hasattr(group, 'energy') or not hasattr(group, 'mu'):
                    print(f"Skipping group '{group_name}': Missing 'energy' or 'mu'")
                    continue
                
                # Get background parameters
                rbkg_value = getattr(group, 'optimalBkg_rbkg', getattr(group, 'rbkg', 1.2))
                optimal_kmin_bkg = getattr(group, 'optimalBkg_kmin', 0.0)
                optimal_kmax_bkg = getattr(group, 'optimalBkg_kmax', 14.0)
                
                # Run autobk if needed
                if not hasattr(group, 'k') or not hasattr(group, 'chi'):
                    autobk(group, rbkg=rbkg_value, kmin=optimal_kmin_bkg, kmax=optimal_kmax_bkg)
                
                k = group.k
                chi = group.chi
                k_max_data = np.max(k)
                weighted_chi = chi * k**kweight
                
                print(f"\nProcessing FT for '{proj_name}.{group_name}'")
                print(f"  kmax range: [{kmax_lower:.1f}, {kmax_upper:.1f}] Å⁻¹")
                
                # Initialize diagnostics for this group
                ft_diagnostics = {
                    'snr_cutoff_k': optimal_kmax_bkg,
                    'kmin_candidates': [],
                    'kmax_candidates': [],
                    'n_zero_crossings_found': 0,
                    'chosen_zc_index': None,
                    'kmin_at_boundary': False,
                    'kmax_at_boundary': False,
                    'selection_reason': '',
                    'first_shell_peak_amp': float('nan'),
                    'low_r_artifact_area': float('nan'),
                    'high_r_noise_area': float('nan'),
                    'nyquist_estimate': float('nan'),
                    'top_candidates': [],
                }
                
                # --- Find kmin candidates from zero crossings in specified range ---
                zc_indices = np.where(np.diff(np.sign(weighted_chi)))[0]
                zc_k = k[zc_indices] if zc_indices.size > 0 else np.array([])
                ft_diagnostics['n_zero_crossings_found'] = len(zc_k)
                
                # Filter to only zero crossings in the specified range
                candidate_kmin_list = zc_k[(zc_k >= kmin_zc_lower) & (zc_k <= kmin_zc_upper)]
                
                if len(candidate_kmin_list) == 0:
                    # Fallback if no zero crossings in range
                    candidate_kmin_list = np.array([max(kmin_zc_lower, optimal_kmin_bkg)])
                    ft_diagnostics['selection_reason'] = 'fallback (no ZC in range)'
                
                candidate_kmin_list = np.sort(np.unique(candidate_kmin_list))
                ft_diagnostics['kmin_candidates'] = candidate_kmin_list.tolist()
                print(f"  kmin candidates (zero-crossings in [{kmin_zc_lower}, {kmin_zc_upper}]): "
                      f"{np.round(candidate_kmin_list, 2)}")
                
                # --- Generate kmax candidates ---
                effective_kmax_upper = min(kmax_upper, k_max_data, optimal_kmax_bkg)
                candidate_kmax_list = np.linspace(kmax_lower, effective_kmax_upper, n_kmax)
                candidate_kmax_list = candidate_kmax_list[candidate_kmax_list <= k_max_data]
                ft_diagnostics['kmax_candidates'] = candidate_kmax_list.tolist()
                
                print(f"  kmax candidates: {len(candidate_kmax_list)} values in "
                      f"[{np.min(candidate_kmax_list):.2f}, {np.max(candidate_kmax_list):.2f}] Å⁻¹")
                
                # --- Evaluate all (kmin, kmax) combinations ---
                ft_results = []
                best_score = -np.inf
                best_result = None
                
                for cand_kmin in candidate_kmin_list:
                    for cand_kmax in candidate_kmax_list:
                        if cand_kmin >= cand_kmax - 1.0:
                            continue
                        
                        temp = Group()
                        temp.k = k.copy()
                        temp.chi = chi.copy()
                        xftf(temp, kweight=kweight, kmin=cand_kmin, kmax=cand_kmax, 
                             dk=dk, kwindow='hanning')
                        
                        # Quality metric: first shell height / noise
                        first_shell_mask = (temp.r > rbkg_value) & (temp.r < rbkg_value + 1.5)
                        noise_mask = (temp.r > 0) & (temp.r < rbkg_value)
                        low_r_mask = (temp.r > 0) & (temp.r < 1.2)  # low-R artifact region
                        high_r_mask = (temp.r > 5.5) & (temp.r < 7.0)  # high-R noise region
                        
                        if np.any(first_shell_mask):
                            first_shell_height = np.max(temp.chir_mag[first_shell_mask])
                            noise = np.mean(temp.chir_mag[noise_mask]) if np.any(noise_mask) else 0.001
                            low_r_area = np.trapz(temp.chir_mag[low_r_mask], temp.r[low_r_mask]) if np.any(low_r_mask) else 0
                            high_r_noise = np.std(temp.chir_mag[high_r_mask]) if np.any(high_r_mask) else 0
                            snr = first_shell_height / max(noise, 1e-10)
                            
                            result = {
                                'kmin': cand_kmin, 'kmax': cand_kmax,
                                'snr': snr, 'temp': temp,
                                'peak': first_shell_height,
                                'lowR': low_r_area,
                                'noise': high_r_noise
                            }
                            ft_results.append(result)
                            
                            if snr > best_score:
                                best_score = snr
                                best_result = result
                
                if best_result is None:
                    print(f"  Warning: No valid FT results")
                    continue
                
                # Store best results
                group.best_kmin = best_result['kmin']
                group.best_kmax = best_result['kmax']
                group.best_r = best_result['temp'].r
                group.best_chir_mag = best_result['temp'].chir_mag
                group.best_chir_re = best_result['temp'].chir_re
                if hasattr(best_result['temp'], 'chir_im'):
                    group.best_chir_im = best_result['temp'].chir_im
                
                # Compute Nyquist estimate: N_ind ≈ (2 * Δk * ΔR) / π
                delta_k = best_result['kmax'] - best_result['kmin']
                delta_r = 6.0  # typical R range used
                ft_diagnostics['nyquist_estimate'] = (2 * delta_k * delta_r) / np.pi
                
                # Store diagnostics
                ft_diagnostics['first_shell_peak_amp'] = best_result.get('peak', float('nan'))
                ft_diagnostics['low_r_artifact_area'] = best_result.get('lowR', float('nan'))
                ft_diagnostics['high_r_noise_area'] = best_result.get('noise', float('nan'))
                
                # Check boundary conditions
                if abs(best_result['kmin'] - kmin_zc_lower) < 0.1:
                    ft_diagnostics['kmin_at_boundary'] = True
                if abs(best_result['kmax'] - effective_kmax_upper) < 0.2:
                    ft_diagnostics['kmax_at_boundary'] = True
                
                # Determine selection reason
                if not ft_diagnostics['selection_reason']:
                    if best_result['kmax'] >= effective_kmax_upper - 0.2:
                        ft_diagnostics['selection_reason'] = 'highest stable below cutoff'
                    else:
                        ft_diagnostics['selection_reason'] = 'best SNR in range'
                
                # Find which zero crossing was chosen
                if len(candidate_kmin_list) > 0:
                    chosen_idx = np.argmin(np.abs(candidate_kmin_list - best_result['kmin']))
                    ft_diagnostics['chosen_zc_index'] = int(chosen_idx + 1)  # 1-indexed
                
                # Store top 5 candidates sorted by SNR
                sorted_results = sorted(ft_results, key=lambda x: x['snr'], reverse=True)[:5]
                ft_diagnostics['top_candidates'] = [
                    {'kmin': r['kmin'], 'kmax': r['kmax'], 'score': r['snr'], 
                     'peak': r.get('peak', 0), 'lowR': r.get('lowR', 0), 'noise': r.get('noise', 0)}
                    for r in sorted_results
                ]
                
                # Store diagnostics on group
                group.ft_diagnostics = ft_diagnostics
                
                print(f"  Best: kmin={best_result['kmin']:.2f}, kmax={best_result['kmax']:.2f}, SNR={best_score:.1f}")
                
                # --- Plot results ---
                if plot_results and ft_results:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # R-space plot
                    ax1 = axes[0]
                    ax1.plot(best_result['temp'].r, best_result['temp'].chir_mag, 
                             'b-', lw=2, label='|χ(R)|')
                    ax1.plot(best_result['temp'].r, best_result['temp'].chir_re, 
                             'r--', lw=1.5, label='Re[χ(R)]')
                    ax1.axvspan(0, rbkg_value, color='lightgray', alpha=0.3)
                    ax1.set_xlabel('R (Å)', fontsize=12)
                    ax1.set_ylabel('|χ(R)|', fontsize=12)
                    ax1.set_title(f'Best FT: kmin={best_result["kmin"]:.2f}, kmax={best_result["kmax"]:.2f}')
                    ax1.legend()
                    ax1.set_xlim(0, 6)
                    ax1.grid(True, alpha=0.3)
                    
                    # k-space plot with window
                    ax2 = axes[1]
                    ax2.plot(k, weighted_chi, 'b-', lw=1.5, label=f'k^{kweight}·χ(k)')
                    ax2.axvline(best_result['kmin'], color='g', ls='--', label=f'kmin={best_result["kmin"]:.2f}')
                    ax2.axvline(best_result['kmax'], color='g', ls='-', lw=2, label=f'kmax={best_result["kmax"]:.2f}')
                    ax2.set_xlabel('k (Å⁻¹)', fontsize=12)
                    ax2.set_ylabel(f'k^{kweight}·χ(k)', fontsize=12)
                    ax2.set_title('k-space with FT window')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    fig.suptitle(f"FT Processing: {proj_name}.{group_name}", fontsize=12)
                    plt.tight_layout()
                    plt.show()

        # ===== SUMMARY: ft_processing_batch =====
        print("\n" + "="*80)
        print("FT_PROCESSING_BATCH SUMMARY")
        print("="*80)
        print(f"\n  PARAMETERS:")
        print(f"    kmax_range: {kmax_range}")
        print(f"    kmin_zc_range: {kmin_zc_range}")
        print(f"    kweight: {kweight}")
        print(f"    dk: {dk}")
        
        # Detailed per-group diagnostics
        print("\n" + "-"*80)
        print("  PER-GROUP DIAGNOSTICS:")
        print("-"*80)
        
        for proj_name, project in self.projects.items():
            for group_name, group in project.groups.items():
                if not (hasattr(group, 'best_kmin') and hasattr(group, 'best_kmax')):
                    print(f"\n  {proj_name}.{group_name}: (no FT data)")
                    continue
                
                print(f"\n  {proj_name}.{group_name}:")
                kmin = group.best_kmin
                kmax = group.best_kmax
                
                # Get diagnostics if available
                diag = getattr(group, 'ft_diagnostics', {})
                
                # Main results
                snr_cutoff = diag.get('snr_cutoff_k', float('nan'))
                nyquist = diag.get('nyquist_estimate', float('nan'))
                num_kmin_cand = diag.get('kmin_candidates', 0)
                num_kmax_cand = diag.get('kmax_candidates', 0)
                n_zc = diag.get('n_zero_crossings_found', 0)
                chosen_zc = diag.get('chosen_zc_index', 0)
                
                print(f"    Selected: kmin={kmin:.2f}, kmax={kmax:.2f}")
                print(f"    SNR cutoff k: {snr_cutoff:.2f} Å⁻¹")
                print(f"    Candidates tested: {len(num_kmin_cand)} kmin × {len(num_kmax_cand)} kmax = {len(num_kmin_cand) * len(num_kmax_cand)} combinations")
                print(f"    Zero crossings found: {n_zc}, selected ZC #{chosen_zc}")
                
                # Get reason and boundaries
                reason = diag.get('selection_reason', 'unknown')
                kmin_boundary = diag.get('kmin_at_boundary', False)
                kmax_boundary = diag.get('kmax_at_boundary', False)
                
                print(f"    Selection reason: {reason}")
                
                if kmin_boundary or kmax_boundary:
                    boundary_flags = []
                    if kmin_boundary:
                        boundary_flags.append("kmin at lower bound")
                    if kmax_boundary:
                        boundary_flags.append("kmax at upper bound")
                    print(f"    ⚠ BOUNDARY: {', '.join(boundary_flags)}")
                
                # Metrics
                first_shell = diag.get('first_shell_peak_amp', float('nan'))
                low_r_area = diag.get('low_r_artifact_area', float('nan'))
                high_r_noise = diag.get('high_r_noise_area', float('nan'))
                
                print(f"    First-shell peak: {first_shell:.3f}")
                print(f"    Low-R artifact area (0-1.2 Å): {low_r_area:.4f}")
                print(f"    High-R noise (5.5-7 Å): {high_r_noise:.4f}")
                print(f"    Nyquist estimate: {nyquist:.2f}")
                
                # Top candidates table
                top_candidates = diag.get('top_candidates', [])
                if top_candidates:
                    print(f"    Top {len(top_candidates)} candidates by SNR:")
                    for i, cand in enumerate(top_candidates, 1):
                        prefix = "  →" if (abs(cand['kmin'] - kmin) < 0.01 and abs(cand['kmax'] - kmax) < 0.01) else "   "
                        print(f"    {prefix} #{i}: kmin={cand['kmin']:.2f}, kmax={cand['kmax']:.2f}, "
                              f"SNR={cand['score']:.1f}, peak={cand['peak']:.3f}, "
                              f"lowR={cand['lowR']:.4f}, noise={cand['noise']:.4f}")
        
        # Quick reference table
        print("\n" + "-"*80)
        print("  QUICK REFERENCE:")
        print(f"  {'Group':<45s}  {'kmin':>6s}  {'kmax':>6s}  {'Peak':>6s}  {'LowR':>7s}  {'Nyquist':>7s}")
        print("-"*80)
        for proj_name, project in self.projects.items():
            for group_name, group in project.groups.items():
                if hasattr(group, 'best_kmin') and hasattr(group, 'best_kmax'):
                    kmin = group.best_kmin
                    kmax = group.best_kmax
                    diag = getattr(group, 'ft_diagnostics', {})
                    peak = diag.get('first_shell_peak_amp', float('nan'))
                    low_r = diag.get('low_r_artifact_area', float('nan'))
                    nyq = diag.get('nyquist_estimate', float('nan'))
                    print(f"    {proj_name}.{group_name:<40s}  {kmin:>6.2f}  {kmax:>6.2f}  "
                          f"{peak:>6.3f}  {low_r:>7.4f}  {nyq:>7.2f}")
                else:
                    print(f"    {proj_name}.{group_name:<40s}  (no FT data)")
        print("="*80 + "\n")

    def _ft_processing_batch_original(self, snr_threshold=1.5, plot_snr_diagnostic=True):
        """
        [DEPRECATED - kept for reference] Original complex FT processing method.
        Use ft_processing_batch() instead.
        """
        for proj_name, project in self.projects.items():
            for group_name, group in project.groups.items():
                if not hasattr(group, 'energy') or not hasattr(group, 'mu'):
                    print(f"Skipping group '{group_name}' in Project '{proj_name}': "
                          f"Missing 'energy' or 'mu' attributes")
                    continue
                
                # ----------------------------------------------------------------
                # PART 1: Retrieve optimized background parameters and run autobk
                # ----------------------------------------------------------------
                rbkg_value = getattr(group, 'optimalBkg_rbkg', getattr(group, 'rbkg', 1.2))
                optimal_kmin_bkg = getattr(group, 'optimalBkg_kmin', None)
                optimal_kmax_bkg = getattr(group, 'optimalBkg_kmax', None)
                
                # Track whether we have valid optimized parameters
                use_optimized_params = (optimal_kmin_bkg is not None and 
                                        optimal_kmax_bkg is not None and
                                        optimal_kmin_bkg < optimal_kmax_bkg)
                
                print(f"\nProcessing FT for Project '{proj_name}', Group '{group_name}'")
                print(f"  Using rbkg = {rbkg_value:.2f} Å")
                
                if use_optimized_params:
                    print(f"  Optimized background kmin = {optimal_kmin_bkg:.2f} Å⁻¹")
                    print(f"  Optimized background kmax = {optimal_kmax_bkg:.2f} Å⁻¹")
                    
                    # Run autobk with optimized parameters to get physically correct χ(k)
                    autobk(group, rbkg=rbkg_value, kmin=optimal_kmin_bkg, kmax=optimal_kmax_bkg)
                    print(f"  → Ran autobk with optimized parameters")
                else:
                    print(f"  Warning: Optimized background parameters not available")
                    print(f"  → Using default autobk parameters")
                    # Fallback: use default autobk
                    autobk(group, rbkg=rbkg_value)
                    # Set fallback values for kmax constraint
                    optimal_kmax_bkg = min(14.0, np.max(group.k) - 0.5) if hasattr(group, 'k') else 14.0
                    optimal_kmin_bkg = 0.0
                
                # Get k and chi from autobk (NOT chi_tapered)
                k = group.k
                chi = group.chi
                k_max_data = np.max(k)
                
                print(f"  Using χ(k) from autobk (k range: 0 to {k_max_data:.2f} Å⁻¹)")
                
                for kw in [3]:  # You can add other k-weights if desired
                    print(f"\n  Analyzing with k-weight {kw}:")
                    weighted_chi = chi * k**kw
                    
                    # ================================================================
                    # SMOOTH SNR(k) USING ROLLING RMS ENVELOPE
                    # ================================================================
                    # This replaces the oscillatory pointwise SNR with a smooth,
                    # physically meaningful signal envelope that decays monotonically.
                    
                    # Step 1: Define window size (~0.5 Å⁻¹ in k-space)
                    dk = np.mean(np.diff(k))
                    window_size = max(1, int(0.5 / dk))
                    
                    # Step 2: Compute rolling RMS (local signal envelope)
                    # This captures the amplitude envelope of χ(k) oscillations
                    chi_squared = weighted_chi**2
                    kernel = np.ones(window_size) / window_size
                    rms_envelope = np.sqrt(np.convolve(chi_squared, kernel, mode='same'))
                    
                    # Step 3 (optional): Apply mild smoothing to further stabilize envelope
                    rms_envelope = np.convolve(rms_envelope, kernel, mode='same')
                    
                    # Step 4: Estimate noise level using RMS in high-k region
                    k_noise_min = np.percentile(k, 80)
                    noise_region_mask = k >= k_noise_min
                    
                    if np.sum(noise_region_mask) >= 5:
                        # Use RMS (not std) for consistency with envelope calculation
                        sigma_noise = np.sqrt(np.mean(weighted_chi[noise_region_mask]**2))
                    else:
                        # Fallback to last ~10 points
                        sigma_noise = np.sqrt(np.mean(weighted_chi[-10:]**2))
                    
                    sigma_noise = max(sigma_noise, 1e-12)  # Prevent division by zero
                    
                    # Step 5: Compute smooth SNR(k) from envelope
                    snr_k = rms_envelope / sigma_noise
                    
                    print(f"  k-space noise σ (RMS) = {sigma_noise:.4e} (from k ≥ {k_noise_min:.2f} Å⁻¹)")
                    print(f"  SNR(k) envelope range: [{np.min(snr_k):.2f}, {np.max(snr_k):.2f}]")
                    print(f"  Rolling window size: {window_size} points (~0.5 Å⁻¹)")
                    
                    # ================================================================
                    # SUSTAINED-THRESHOLD KMAX SELECTION
                    # ================================================================
                    # Require consecutive points above threshold to avoid isolated spikes
                    min_run = 3  # Number of consecutive points required
                    
                    valid = snr_k > snr_threshold
                    # Convolution counts how many of the previous min_run points are valid
                    runs = np.convolve(valid.astype(int), np.ones(min_run, dtype=int), mode='same')
                    
                    # Points where at least min_run consecutive points exceed threshold
                    valid_indices = np.where(runs >= min_run)[0]
                    
                    if len(valid_indices) > 0:
                        kmax_snr_limit = k[valid_indices[-1]]
                    else:
                        # Fallback if no sustained region exceeds threshold
                        kmax_snr_limit = 11.0
                    
                    print(f"  SNR threshold = {snr_threshold} (sustained over {min_run} points)")
                    print(f"  kmax_snr_limit = {kmax_snr_limit:.2f} Å⁻¹ "
                          f"(last k with sustained SNR > {snr_threshold})")
                    
                    # Store for later reference
                    setattr(group, f"kmax_snr_limit_kw{kw}", kmax_snr_limit)
                    setattr(group, f"snr_k_kw{kw}", snr_k)
                    setattr(group, f"sigma_noise_kw{kw}", sigma_noise)
                    setattr(group, f"rms_envelope_kw{kw}", rms_envelope)  # Store envelope for diagnostics
                    
                    # ----------------------------------------------------------------
                    # PART 2: Candidate kmin from zero crossings of weighted χ(k)
                    # ----------------------------------------------------------------
                    zc_indices = np.where(np.diff(np.sign(weighted_chi)))[0]
                    if zc_indices.size > 0:
                        zc_k = k[zc_indices]
                        print(f"  Found {len(zc_k)} zero crossings in weighted χ(k)")
                    else:
                        zc_k = np.array([])
                        print("  No zero crossings found in weighted χ(k)")
                    
                    # Select kmin candidates from zero crossings in range ~2.0–4.0 Å⁻¹
                    kmin_region = zc_k[(zc_k >= 2.0) & (zc_k <= 4.0)]
                    
                    if len(kmin_region) >= 2:
                        # Take up to 4 candidates where oscillations are stable
                        candidate_kmin_list = np.sort(kmin_region[:4])
                    elif len(kmin_region) == 1:
                        # Single zero crossing in range
                        candidate_kmin_list = kmin_region
                    else:
                        # Fallback: use optimalBkg_kmin as basis
                        candidate_kmin_list = np.array([optimal_kmin_bkg, optimal_kmin_bkg + 0.5])
                    
                    # Enforce: kmin >= optimalBkg_kmin (if available)
                    if use_optimized_params:
                        candidate_kmin_list = candidate_kmin_list[candidate_kmin_list >= optimal_kmin_bkg]
                        if len(candidate_kmin_list) == 0:
                            candidate_kmin_list = np.array([optimal_kmin_bkg])
                    
                    # Ensure kmin is not in pre-edge region (< ~1.5 Å⁻¹ unless justified)
                    candidate_kmin_list = candidate_kmin_list[candidate_kmin_list >= 1.5]
                    if len(candidate_kmin_list) == 0:
                        candidate_kmin_list = np.array([max(1.5, optimal_kmin_bkg)])
                    
                    candidate_kmin_list = np.sort(np.unique(candidate_kmin_list))
                    print(f"  kmin candidates (from zero crossings): {np.round(candidate_kmin_list, 2)}")
                    
                    # ================================================================
                    # MODIFIED PART 3: Constrain kmax using SNR limit
                    # ================================================================
                    # kmax range: [11.0, min(optimalBkg_kmax, kmax_snr_limit, k_max_data)]
                    kmax_lower = 11.0
                    kmax_upper = min(optimal_kmax_bkg, kmax_snr_limit, k_max_data)
                    
                    print(f"  kmax upper bound: min(bkg={optimal_kmax_bkg:.2f}, "
                          f"snr_limit={kmax_snr_limit:.2f}, data={k_max_data:.2f}) = {kmax_upper:.2f} Å⁻¹")
                    
                    # Ensure valid range
                    if kmax_upper <= kmax_lower:
                        kmax_upper = min(k_max_data, 14.0, kmax_snr_limit)
                        kmax_lower = max(9.0, np.min(candidate_kmin_list) + 2.0)
                    
                    # Generate dense candidate list (~20 points)
                    n_kmax_candidates = 20
                    candidate_kmax_list = np.linspace(kmax_lower, kmax_upper, n_kmax_candidates)
                    
                    # Enforce: kmax <= max(k) and kmax > kmin + 1.0
                    min_kmin = np.min(candidate_kmin_list)
                    candidate_kmax_list = candidate_kmax_list[candidate_kmax_list <= k_max_data]
                    candidate_kmax_list = candidate_kmax_list[candidate_kmax_list > min_kmin + 1.0]
                    
                    if len(candidate_kmax_list) == 0:
                        candidate_kmax_list = np.array([min(optimal_kmax_bkg, k_max_data, kmax_snr_limit)])
                    
                    print(f"  kmax candidates (SNR-constrained): {len(candidate_kmax_list)} values in "
                          f"[{np.min(candidate_kmax_list):.2f}, {np.max(candidate_kmax_list):.2f}] Å⁻¹")
                    
                    # ----------------------------------------------------------------
                    # Evaluate ALL candidates with SNR penalty (PART 4)
                    # ----------------------------------------------------------------
                    ft_results = []
                    best_score = -float('inf')
                    best_kmin = None
                    best_kmax = None
                    best_temp = None
                    
                    print(f"  Evaluating {len(candidate_kmin_list) * len(candidate_kmax_list)} "
                          f"(kmin, kmax) combinations...")
                    
                    for cand_kmin in candidate_kmin_list:
                        for cand_kmax in candidate_kmax_list:
                            if cand_kmin >= cand_kmax - 1.0:
                                continue  # Minimum window width of 1.0 Å⁻¹
                            
                            temp = Group()
                            temp.k = k.copy()
                            temp.chi = chi.copy()
                            
                            # FT call (unchanged)
                            xftf(temp, kweight=kw, kmin=cand_kmin, kmax=cand_kmax, 
                                 dk=3, kwindow='hanning')
                            
                            # First shell definition: (rbkg_value, rbkg_value + 1.5)
                            first_shell_mask = (temp.r > rbkg_value) & (temp.r < rbkg_value + 1.5)
                            if np.any(first_shell_mask):
                                first_shell_height = np.max(temp.chir_mag[first_shell_mask])
                                first_shell_pos = temp.r[first_shell_mask][
                                    np.argmax(temp.chir_mag[first_shell_mask])]
                                
                                # Noise level: R < rbkg (unchanged)
                                noise_mask = (temp.r > 0) & (temp.r < rbkg_value)
                                noise = np.mean(temp.chir_mag[noise_mask]) if np.any(noise_mask) else 0.001
                                
                                # R-space SNR (unchanged)
                                snr = first_shell_height / noise if noise > 0 else 0
                                position_factor = 1.0 - 0.2 * abs(first_shell_pos - 2.5) / 1.5
                                
                                # ============================================
                                # NEW PART 4: SNR-based penalty for kmax
                                # ============================================
                                # Interpolate SNR(k) at candidate kmax
                                kmax_snr_value = np.interp(cand_kmax, k, snr_k)
                                
                                # Compute penalty: scales from 0.5 to 1.0 based on SNR
                                # Higher SNR at kmax → penalty closer to 1.0 (less penalty)
                                # Lower SNR at kmax → penalty closer to 0.5 (more penalty)
                                kmax_penalty = kmax_snr_value / 2.0
                                kmax_penalty = np.clip(kmax_penalty, 0.5, 1.0)
                                
                                # Modified quality score with kmax penalty
                                quality_score = snr * position_factor * first_shell_height * kmax_penalty
                                
                                ft_results.append({
                                    'kmin': cand_kmin,
                                    'kmax': cand_kmax,
                                    'snr': snr,
                                    'quality_score': quality_score,
                                    'quality_score_raw': snr * position_factor * first_shell_height,
                                    'kmax_penalty': kmax_penalty,
                                    'kmax_snr_value': kmax_snr_value,
                                    'temp': temp,
                                    'first_shell_height': first_shell_height,
                                    'noise': noise,
                                    'constrained': use_optimized_params
                                })
                                
                                if quality_score > best_score:
                                    best_score = quality_score
                                    best_kmin = cand_kmin
                                    best_kmax = cand_kmax
                                    best_temp = Group()
                                    best_temp.k = temp.k.copy()
                                    best_temp.chi = temp.chi.copy()
                                    best_temp.r = temp.r.copy()
                                    best_temp.chir_mag = temp.chir_mag.copy()
                                    best_temp.chir_re = temp.chir_re.copy()
                                    best_temp.chir_im = temp.chir_im.copy() if hasattr(temp, 'chir_im') else None
                    
                    # ----------------------------------------------------------------
                    # PART 5: Handle boundary condition - kmax stuck at lower bound
                    # ----------------------------------------------------------------
                    if ft_results and best_kmax is not None:
                        if np.isclose(best_kmax, np.min(candidate_kmax_list)):
                            print("  Expanded kmax search downward due to boundary selection")
                            
                            # Expand search downward: [9.0, original_kmax_lower]
                            expanded_kmax_lower = 9.0
                            expanded_kmax_upper = np.min(candidate_kmax_list)
                            
                            if expanded_kmax_lower < expanded_kmax_upper - 0.5:
                                n_expanded = 15
                                expanded_kmax_list = np.linspace(
                                    expanded_kmax_lower, expanded_kmax_upper, n_expanded)
                                expanded_kmax_list = expanded_kmax_list[
                                    expanded_kmax_list > min_kmin + 1.0]
                                
                                print(f"    Expanding to include {len(expanded_kmax_list)} "
                                      f"additional kmax values in [{expanded_kmax_lower:.2f}, "
                                      f"{expanded_kmax_upper:.2f}] Å⁻¹")
                                
                                # Evaluate expanded candidates with SNR penalty
                                for cand_kmin in candidate_kmin_list:
                                    for cand_kmax in expanded_kmax_list:
                                        if cand_kmin >= cand_kmax - 1.0:
                                            continue
                                        
                                        temp = Group()
                                        temp.k = k.copy()
                                        temp.chi = chi.copy()
                                        
                                        xftf(temp, kweight=kw, kmin=cand_kmin, kmax=cand_kmax, 
                                             dk=3, kwindow='hanning')
                                        
                                        first_shell_mask = (temp.r > rbkg_value) & (temp.r < rbkg_value + 1.5)
                                        if np.any(first_shell_mask):
                                            first_shell_height = np.max(temp.chir_mag[first_shell_mask])
                                            first_shell_pos = temp.r[first_shell_mask][
                                                np.argmax(temp.chir_mag[first_shell_mask])]
                                            
                                            noise_mask = (temp.r > 0) & (temp.r < rbkg_value)
                                            noise = np.mean(temp.chir_mag[noise_mask]) if np.any(noise_mask) else 0.001
                                            
                                            snr = first_shell_height / noise if noise > 0 else 0
                                            position_factor = 1.0 - 0.2 * abs(first_shell_pos - 2.5) / 1.5
                                            
                                            # Apply SNR penalty
                                            kmax_snr_value = np.interp(cand_kmax, k, snr_k)
                                            kmax_penalty = np.clip(kmax_snr_value / 2.0, 0.5, 1.0)
                                            quality_score = snr * position_factor * first_shell_height * kmax_penalty
                                            
                                            ft_results.append({
                                                'kmin': cand_kmin,
                                                'kmax': cand_kmax,
                                                'snr': snr,
                                                'quality_score': quality_score,
                                                'quality_score_raw': snr * position_factor * first_shell_height,
                                                'kmax_penalty': kmax_penalty,
                                                'kmax_snr_value': kmax_snr_value,
                                                'temp': temp,
                                                'first_shell_height': first_shell_height,
                                                'noise': noise,
                                                'constrained': use_optimized_params,
                                                'expanded': True
                                            })
                                            
                                            if quality_score > best_score:
                                                best_score = quality_score
                                                best_kmin = cand_kmin
                                                best_kmax = cand_kmax
                                                best_temp = Group()
                                                best_temp.k = temp.k.copy()
                                                best_temp.chi = temp.chi.copy()
                                                best_temp.r = temp.r.copy()
                                                best_temp.chir_mag = temp.chir_mag.copy()
                                                best_temp.chir_re = temp.chir_re.copy()
                                                best_temp.chir_im = temp.chir_im.copy() if hasattr(temp, 'chir_im') else None
                                
                                # Update candidate list for plotting
                                candidate_kmax_list = np.sort(np.unique(
                                    np.concatenate([candidate_kmax_list, expanded_kmax_list])))
                    
                    # ================================================================
                    # NEW PART 5: Final safety check - enforce SNR-based upper bound
                    # ================================================================
                    if best_kmax is not None and best_kmax > kmax_snr_limit:
                        print(f"  Warning: Selected kmax ({best_kmax:.2f}) exceeds SNR limit "
                              f"({kmax_snr_limit:.2f})")
                        print(f"  → Adjusting kmax to SNR limit: {kmax_snr_limit:.2f} Å⁻¹")
                        
                        # Find best result with kmax <= kmax_snr_limit
                        valid_results = [r for r in ft_results if r['kmax'] <= kmax_snr_limit]
                        if valid_results:
                            valid_results.sort(key=lambda x: x['quality_score'], reverse=True)
                            best_result_adjusted = valid_results[0]
                            best_kmax = best_result_adjusted['kmax']
                            best_kmin = best_result_adjusted['kmin']
                            best_score = best_result_adjusted['quality_score']
                            best_temp = best_result_adjusted['temp']
                        else:
                            # If no valid results, just cap kmax
                            best_kmax = kmax_snr_limit
                    
                    # ----------------------------------------------------------------
                    # Plot only representative subset
                    # ----------------------------------------------------------------
                    if ft_results:
                        # Select 5 representative kmax values for plotting:
                        # min, 25th percentile, median, 75th percentile, max
                        all_kmax_evaluated = np.unique([r['kmax'] for r in ft_results])
                        all_kmax_evaluated = np.sort(all_kmax_evaluated)
                        
                        if len(all_kmax_evaluated) > 5:
                            plot_kmax_indices = [
                                0,  # minimum
                                int(len(all_kmax_evaluated) * 0.25),  # 25th percentile
                                int(len(all_kmax_evaluated) * 0.5),   # median
                                int(len(all_kmax_evaluated) * 0.75),  # 75th percentile
                                len(all_kmax_evaluated) - 1  # maximum
                            ]
                            plot_kmax_list = all_kmax_evaluated[plot_kmax_indices]
                        else:
                            plot_kmax_list = all_kmax_evaluated
                        
                        # Always include best_kmax in plot
                        if best_kmax is not None and best_kmax not in plot_kmax_list:
                            plot_kmax_list = np.sort(np.append(plot_kmax_list, best_kmax))
                        
                        plot_kmax_list = np.unique(plot_kmax_list)
                        
                        n_rows = len(candidate_kmin_list)
                        n_cols = len(plot_kmax_list)
                        
                        print(f"  Plotting {n_rows} × {n_cols} representative combinations")
                        
                        if n_rows * n_cols >= 1:
                            fig, axs = plt.subplots(n_rows, n_cols, 
                                                    figsize=(4 * n_cols, 3 * n_rows), sharex=True)
                            # Ensure axs is 2D
                            if n_rows == 1 and n_cols == 1:
                                axs = np.array([[axs]])
                            elif n_rows == 1:
                                axs = np.expand_dims(axs, axis=0)
                            elif n_cols == 1:
                                axs = np.expand_dims(axs, axis=1)
                            
                            for i, cand_kmin in enumerate(candidate_kmin_list):
                                for j, cand_kmax in enumerate(plot_kmax_list):
                                    ax = axs[i, j]
                                    
                                    # Find matching result
                                    matching = [r for r in ft_results 
                                               if np.isclose(r['kmin'], cand_kmin) and 
                                                  np.isclose(r['kmax'], cand_kmax)]
                                    
                                    if matching:
                                        result = matching[0]
                                        temp = result['temp']
                                        
                                        ax.plot(temp.r, temp.chir_mag, label="|χ(R)|", color='blue')
                                        ax.plot(temp.r, temp.chir_re, label="Re[χ(R)]", 
                                                linestyle='--', color='red')
                                        
                                        ax.set_title(f"kmin={cand_kmin:.2f}, kmax={cand_kmax:.2f}\n"
                                                    f"Score={result['quality_score']:.1f} "
                                                    f"(pen={result.get('kmax_penalty', 1.0):.2f})")
                                    else:
                                        ax.text(0.5, 0.5, "Not evaluated", ha='center', va='center')
                                        ax.set_title(f"kmin={cand_kmin:.2f}, kmax={cand_kmax:.2f}")
                                    
                                    ax.axvspan(0, rbkg_value, color='lightgray', alpha=0.3)
                                    ax.set_xlabel('R (Å)')
                                    ax.grid(True, alpha=0.3)
                                    ax.legend(fontsize=8)
                                    ax.set_xlim(0, 6)
                            
                            # Highlight best result
                            if best_kmin is not None and best_kmax is not None:
                                try:
                                    best_i = np.where(np.isclose(candidate_kmin_list, best_kmin))[0][0]
                                    best_j = np.where(np.isclose(plot_kmax_list, best_kmax))[0][0]
                                    for spine in axs[best_i, best_j].spines.values():
                                        spine.set_edgecolor('red')
                                        spine.set_linewidth(3)
                                except (IndexError, Exception) as e:
                                    print(f"  Warning: Couldn't highlight best parameter pair: {e}")
                            
                            constraint_label = "SNR-Constrained" if use_optimized_params else "Default"
                            fig.suptitle(f"FT Results ({constraint_label}): Project '{proj_name}', "
                                        f"Group '{group_name}', k-weight {kw}\n"
                                        f"Evaluated {len(ft_results)} combinations, "
                                        f"kmax_snr_limit={kmax_snr_limit:.2f} Å⁻¹",
                                        fontsize=12)
                            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
                            plt.show()
                    
                    # ================================================================
                    # SNR DIAGNOSTIC PLOT (showing envelope-based SNR)
                    # ================================================================
                    if plot_snr_diagnostic and ft_results:
                        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                        
                        # Panel 1 (top-left): SNR(k) envelope vs k
                        ax1 = axes[0, 0]
                        ax1.plot(k, snr_k, 'b-', lw=2, label='SNR(k) envelope')
                        ax1.axhline(y=snr_threshold, color='orange', linestyle='--', 
                                   lw=2, label=f'SNR threshold = {snr_threshold}')
                        ax1.axvline(x=kmax_snr_limit, color='red', linestyle=':', 
                                   lw=2, label=f'kmax_snr_limit = {kmax_snr_limit:.2f}')
                        if best_kmax is not None:
                            ax1.axvline(x=best_kmax, color='green', linestyle='--', 
                                       lw=2, label=f'Chosen kmax = {best_kmax:.2f}')
                            best_kmax_snr = np.interp(best_kmax, k, snr_k)
                            ax1.scatter([best_kmax], [best_kmax_snr], color='green', s=100, zorder=5)
                        
                        ax1.set_xlabel('k (Å$^{-1}$)', fontsize=12)
                        ax1.set_ylabel('SNR(k)', fontsize=12)
                        ax1.set_title('Smooth SNR(k) from RMS Envelope')
                        ax1.legend(loc='upper right', fontsize=9)
                        ax1.grid(True, alpha=0.3)
                        ax1.set_xlim(0, k_max_data)
                        ax1.set_ylim(0, min(np.max(snr_k) * 1.1, 50))
                        
                        # Panel 2 (top-right): RMS envelope vs |χ(k)|
                        ax2 = axes[0, 1]
                        ax2.plot(k, np.abs(weighted_chi), 'b-', lw=1, alpha=0.5, 
                                label=f'|k^{kw}·χ(k)|')
                        ax2.plot(k, rms_envelope, 'r-', lw=2, label='RMS envelope')
                        ax2.axhline(y=sigma_noise, color='orange', linestyle='--', 
                                   lw=1.5, label=f'Noise σ = {sigma_noise:.2e}')
                        ax2.axvline(x=kmax_snr_limit, color='red', linestyle=':', 
                                   lw=2, alpha=0.7)
                        
                        ax2.set_xlabel('k (Å$^{-1}$)', fontsize=12)
                        ax2.set_ylabel('Amplitude', fontsize=12)
                        ax2.set_title('Signal Envelope vs Noise Level')
                        ax2.legend(loc='upper right', fontsize=9)
                        ax2.grid(True, alpha=0.3)
                        ax2.set_xlim(0, k_max_data)
                        
                        # Panel 3 (bottom-left): weighted χ(k) with SNR regions
                        ax3 = axes[1, 0]
                        ax3.plot(k, weighted_chi, 'b-', lw=1.5, label=f'k^{kw}·χ(k)')
                        
                        # Shade noise-dominated region
                        ax3.axvspan(kmax_snr_limit, k_max_data, color='red', alpha=0.15, 
                                   label=f'Noise-dominated (k > {kmax_snr_limit:.2f})')
                        
                        if best_kmin is not None:
                            ax3.axvline(x=best_kmin, color='green', linestyle='--', 
                                       label=f'kmin = {best_kmin:.2f}')
                        if best_kmax is not None:
                            ax3.axvline(x=best_kmax, color='green', linestyle='-', 
                                       lw=2, label=f'kmax = {best_kmax:.2f}')
                        
                        ax3.set_xlabel('k (Å$^{-1}$)', fontsize=12)
                        ax3.set_ylabel(f'k^{kw}·χ(k)', fontsize=12)
                        ax3.set_title('k-weighted χ(k) with SNR-based limits')
                        ax3.legend(loc='upper right', fontsize=9)
                        ax3.grid(True, alpha=0.3)
                        
                        # Panel 4 (bottom-right): Summary text
                        ax4 = axes[1, 1]
                        ax4.axis('off')
                        
                        summary_text = (
                            f"SNR ENVELOPE ANALYSIS SUMMARY\n"
                            f"{'='*45}\n\n"
                            f"METHOD: Rolling RMS envelope\n"
                            f"  Window size: {window_size} points (~0.5 Å⁻¹)\n"
                            f"  Noise σ (RMS): {sigma_noise:.4e}\n\n"
                            f"RESULTS:\n"
                            f"  SNR(k) range: [{np.min(snr_k):.2f}, {np.max(snr_k):.2f}]\n"
                            f"  SNR threshold: {snr_threshold}\n"
                            f"  Sustained run requirement: {min_run} points\n\n"
                            f"KMAX SELECTION:\n"
                            f"  kmax_snr_limit: {kmax_snr_limit:.2f} Å⁻¹\n"
                            f"  Chosen kmax: {best_kmax:.2f} Å⁻¹\n"
                            f"  SNR at chosen kmax: {np.interp(best_kmax, k, snr_k):.2f}\n\n"
                            f"COMPARISON:\n"
                            f"  Background kmax: {optimal_kmax_bkg:.2f} Å⁻¹\n"
                            f"  Data limit: {k_max_data:.2f} Å⁻¹"
                        )
                        
                        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                                fontsize=10, verticalalignment='top', fontfamily='monospace',
                                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                        
                        fig.suptitle(f"SNR Envelope Diagnostic: Project '{proj_name}', Group '{group_name}'",
                                    fontsize=14)
                        plt.tight_layout(rect=[0, 0, 1, 0.95])
                        plt.show()
                    
                    # ----------------------------------------------------------------
                    # PART 7: Output and Storage
                    # ----------------------------------------------------------------
                    if ft_results:
                        ft_results.sort(key=lambda x: x.get('quality_score', x.get('snr', 0)), 
                                       reverse=True)
                        best_result = ft_results[0]
                        
                        # Store per k-weight results
                        setattr(group, f"best_kmin_kw{kw}", best_result['kmin'])
                        setattr(group, f"best_kmax_kw{kw}", best_result['kmax'])
                        
                        if kw == 2:
                            group.best_kmin = best_result['kmin']
                            group.best_kmax = best_result['kmax']
                        
                        # Store background-aware FT parameters
                        group.best_kmin_constrained = best_result['kmin']
                        group.best_kmax_constrained = best_result['kmax']
                        group.ft_constrained_by_bkg = use_optimized_params
                        
                        # Store FT results
                        best_temp = best_result['temp']
                        setattr(group, f"best_r_kw{kw}", best_temp.r)
                        setattr(group, f"best_chir_mag_kw{kw}", best_temp.chir_mag)
                        setattr(group, f"best_chir_re_kw{kw}", best_temp.chir_re)
                        if hasattr(best_temp, 'chir_im'):
                            setattr(group, f"best_chir_im_kw{kw}", best_temp.chir_im)
                        
                        # Store quality metrics
                        setattr(group, f"ft_quality_score_kw{kw}", best_result['quality_score'])
                        setattr(group, f"ft_snr_kw{kw}", best_result['snr'])
                        setattr(group, f"ft_kmax_penalty_kw{kw}", best_result.get('kmax_penalty', 1.0))
                        
                        print(f"\n  Best parameters (k-weight {kw}):")
                        print(f"    kmin = {best_result['kmin']:.3f} Å⁻¹")
                        print(f"    kmax = {best_result['kmax']:.3f} Å⁻¹")
                        print(f"    kmax_snr_limit = {kmax_snr_limit:.3f} Å⁻¹")
                        print(f"    SNR (R-space) = {best_result['snr']:.1f}")
                        print(f"    SNR at kmax = {best_result.get('kmax_snr_value', 'N/A'):.2f}")
                        print(f"    kmax penalty = {best_result.get('kmax_penalty', 1.0):.3f}")
                        print(f"    Quality Score = {best_result.get('quality_score', 'N/A'):.2f}")
                        if use_optimized_params:
                            print(f"    Constrained by background optimization: Yes")
                            print(f"    Background kmin/kmax: [{optimal_kmin_bkg:.2f}, "
                                  f"{optimal_kmax_bkg:.2f}] Å⁻¹")
                        
                        # Final summary plot
                        plt.figure(figsize=(12, 5))
                        plt.subplot(1, 2, 1)
                        plt.plot(best_temp.r, best_temp.chir_mag, label="|χ(R)|", linewidth=2)
                        plt.plot(best_temp.r, best_temp.chir_re, label="Re[χ(R)]", 
                                linestyle='--', linewidth=2)
                        plt.axvspan(0, rbkg_value, color='lightgray', alpha=0.3, 
                                    label=f"Unphysical (< {rbkg_value:.2f} Å)")
                        plt.axvspan(rbkg_value, rbkg_value + 1.5, color='lightgreen', alpha=0.1, 
                                    label=f"First shell ({rbkg_value:.2f}-{rbkg_value+1.5:.2f} Å)")
                        plt.xlabel('R (Å)', fontsize=12)
                        plt.ylabel('|χ(R)|', fontsize=12)
                        plt.title(f"Best R-space Transform (k-weight {kw})")
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.xlim(0, 6)
                        
                        plt.subplot(1, 2, 2)
                        plt.plot(k, chi * k**kw, 'b-', lw=2, label=f'k^{kw}·χ(k)')
                        plt.axvline(x=best_result['kmin'], color='g', linestyle='--', 
                                    label=f'kmin={best_result["kmin"]:.2f}')
                        plt.axvline(x=best_result['kmax'], color='g', linestyle='-', 
                                    lw=2, label=f'kmax={best_result["kmax"]:.2f}')
                        
                        # Show SNR limit
                        plt.axvline(x=kmax_snr_limit, color='red', linestyle=':', 
                                    alpha=0.7, label=f'kmax_snr_limit={kmax_snr_limit:.2f}')
                        
                        # Show optimized background bounds if available
                        if use_optimized_params:
                            plt.axvline(x=optimal_kmin_bkg, color='g', linestyle=':', 
                                        alpha=0.5, label=f'bkg_kmin={optimal_kmin_bkg:.2f}')
                            plt.axvline(x=optimal_kmax_bkg, color='orange', linestyle=':', 
                                        alpha=0.5, label=f'bkg_kmax={optimal_kmax_bkg:.2f}')
                        
                        # Shade noise region
                        plt.axvspan(kmax_snr_limit, k_max_data, color='red', alpha=0.1)
                        
                        try:
                            from larch.xafs.xafsft import ftwindow
                            win = ftwindow(k, xmin=best_result['kmin'], xmax=best_result['kmax'], 
                                           dx=3, window='hanning')
                            plt.plot(k, win * np.max(np.abs(chi * k**kw)) * 0.8, 
                                     'k:', label='Window')
                        except Exception as e:
                            print(f"  Warning: Could not generate window function: {e}")
                        
                        plt.xlabel('k (Å$^{-1}$)', fontsize=12)
                        plt.ylabel(f'k^{kw}·χ(k)', fontsize=12)
                        plt.title('k-space data with SNR-constrained window')
                        plt.legend(fontsize=9, loc='upper right')
                        plt.grid(True, alpha=0.3)
                        
                        constraint_note = " (SNR-Constrained)" if use_optimized_params else ""
                        plt.suptitle(
                            f"Best FT Result{constraint_note}: Project '{proj_name}', "
                            f"Group '{group_name}', k-weight {kw}\n"
                            f"kmin={best_result['kmin']:.3f}, kmax={best_result['kmax']:.3f}, "
                            f"kmax_snr_limit={kmax_snr_limit:.2f}",
                            fontsize=12
                        )
                        plt.tight_layout(rect=[0, 0, 1, 0.93])
                        plt.show()
                    else:
                        print(f"  Warning: No valid FT results for k-weight {kw}")

    def enhanced_verify_modifications(self):
        """
        Verify modifications by comparing original and processed data.
        
        Generates a 2x3 grid of plots comparing original vs. modified χ(k) (with different k-weighting)
        as well as FT results, including markers for the rbkg value.
        """
        for proj_name, project in self.projects.items():
            for group_name, group in project.groups.items():
                if hasattr(group, 'k') and hasattr(group, 'chi'):
                    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                    
                    # Plot 1: Original vs Tapered χ(k)
                    axs[0, 0].plot(group.k, group.chi, label='Original χ(k)', color='blue')
                    if hasattr(group, 'chi_tapered'):
                        axs[0, 0].plot(group.k, group.chi_tapered, label='Tapered χ(k)', color='green')
                    axs[0, 0].set_xlabel('k (Å$^{-1}$)')
                    axs[0, 0].set_ylabel('χ(k)')
                    axs[0, 0].legend()
                    axs[0, 0].set_title('Original vs Modified χ(k)')
                    
                    # Plot 2: k²-weighted χ(k)
                    axs[0, 1].plot(group.k, group.chi * group.k**2, label='Original k²χ(k)', color='blue')
                    if hasattr(group, 'chi_tapered'):
                        axs[0, 1].plot(group.k, group.chi_tapered * group.k**2, label='Modified k²χ(k)', color='green')
                    axs[0, 1].set_xlabel('k (Å$^{-1}$)')
                    axs[0, 1].set_ylabel('k²χ(k)')
                    axs[0, 1].legend()
                    axs[0, 1].set_title('k²-weighted χ(k)')
                    
                    # Plot 3: k³-weighted χ(k)
                    axs[0, 2].plot(group.k, group.chi * group.k**3, label='Original k³χ(k)', color='blue')
                    if hasattr(group, 'chi_tapered'):
                        axs[0, 2].plot(group.k, group.chi_tapered * group.k**3, label='Modified k³χ(k)', color='green')
                    axs[0, 2].set_xlabel('k (Å$^{-1}$)')
                    axs[0, 2].set_ylabel('k³χ(k)')
                    axs[0, 2].legend()
                    axs[0, 2].set_title('k³-weighted χ(k)')
                    
                    # Plot 4: FT from rbkg optimization (if available)
                    rbkg_ft_available = hasattr(group, 'r') and hasattr(group, 'chir_mag')
                    rbkg_value = getattr(group, 'optimal_rbkg', getattr(group, 'rbkg', None))
                    
                    if rbkg_ft_available:
                        axs[1, 0].plot(group.r, group.chir_mag, 
                                       label=f'rbkg={rbkg_value:.2f}' if rbkg_value else 'rbkg FT', 
                                       color='blue')
                        axs[1, 0].set_xlabel('R (Å)')
                        axs[1, 0].set_ylabel('|χ(R)|')
                        axs[1, 0].legend()
                        axs[1, 0].set_title('FT from rbkg optimization')
                        if rbkg_value:
                            axs[1, 0].axvline(x=rbkg_value, color='gray', linestyle='--', alpha=0.7)
                            axs[1, 0].text(rbkg_value + 0.05, 0.9 * axs[1, 0].get_ylim()[1], 
                                         f'rbkg={rbkg_value:.2f}', rotation=90, va='top')
                            axs[1, 0].axvspan(0, rbkg_value, color='lightgray', alpha=0.3, 
                                              label='Unphysical region')
                    else:
                        axs[1, 0].text(0.5, 0.5, 'No rbkg optimization FT data', 
                                      horizontalalignment='center', verticalalignment='center')
                    
                    # Plot 5: Best FT with k-weight=2 (if available)
                    if hasattr(group, 'best_r_kw2') and hasattr(group, 'best_chir_mag_kw2'):
                        axs[1, 1].plot(group.best_r_kw2, group.best_chir_mag_kw2, 
                                       label='|χ(R)| (k-weight=2)', color='green')
                        axs[1, 1].set_xlabel('R (Å)')
                        axs[1, 1].set_ylabel('|χ(R)|')
                        if rbkg_value:
                            axs[1, 1].axvline(x=rbkg_value, color='gray', linestyle='--', alpha=0.7)
                            axs[1, 1].axvspan(0, rbkg_value, color='lightgray', alpha=0.3)
                        axs[1, 1].legend()
                        axs[1, 1].set_title(f'FT with k²: kmin={getattr(group, "best_kmin_kw2", "N/A"):.2f}, ' +
                                            f'kmax={getattr(group, "best_kmax_kw2", "N/A"):.2f}')
                    else:
                        axs[1, 1].text(0.5, 0.5, 'No k²-weighted FT data', 
                                      horizontalalignment='center', verticalalignment='center')
                    
                    # Plot 6: Best FT with k-weight=3 (if available)
                    if hasattr(group, 'best_r_kw3') and hasattr(group, 'best_chir_mag_kw3'):
                        axs[1, 2].plot(group.best_r_kw3, group.best_chir_mag_kw3, 
                                       label='|χ(R)| (k-weight=3)', color='red')
                        axs[1, 2].set_xlabel('R (Å)')
                        axs[1, 2].set_ylabel('|χ(R)|')
                        if rbkg_value:
                            axs[1, 2].axvline(x=rbkg_value, color='gray', linestyle='--', alpha=0.7)
                            axs[1, 2].axvspan(0, rbkg_value, color='lightgray', alpha=0.3)
                        axs[1, 2].legend()
                        axs[1, 2].set_title(f'FT with k³: kmin={getattr(group, "best_kmin_kw3", "N/A"):.2f}, ' +
                                            f'kmax={getattr(group, "best_kmax_kw3", "N/A"):.2f}')
                    else:
                        axs[1, 2].text(0.5, 0.5, 'No k³-weighted FT data', 
                                      horizontalalignment='center', verticalalignment='center')
                    
                    r_max = 10
                    for i in range(3):
                        axs[1, i].set_xlim(0, r_max)
                    
                    plt.tight_layout()
                    plt.suptitle(f"Verification for Project '{proj_name}', Group '{group_name}'", 
                                fontsize=16, y=1.02)
                    plt.show()
                else:
                    print(f"Cannot verify group '{group_name}' in Project '{proj_name}': Missing 'k' or 'chi' attribute")

    def save_processed_exafs(self, output_path, kweights=None, include_imaginary=False,
                              use_tapered_chi=True, verbose=True):
        """
        Save the processed EXAFS data as .dat files.
        
        This method saves the optimized FT results from ft_processing_batch as ASCII .dat files.
        For each group and k-weight, it produces two files:
          1. chi_R file: Contains R, |χ(R)|, and Re[χ(R)] (and optionally Im[χ(R)])
          2. chi_K file: Contains k and k^weight * χ(k)
        
        Parameters
        ----------
        output_path : str or Path
            Directory where the .dat files will be saved. Created if it doesn't exist.
        kweights : list of int, optional
            List of k-weights to save (default: [1, 2, 3]).
        include_imaginary : bool, optional
            If True, include the imaginary component Im[χ(R)] in the chi_R files (default: False).
        use_tapered_chi : bool, optional
            If True, use chi_tapered for k-weighted chi(k); otherwise use raw chi (default: True).
        verbose : bool, optional
            If True, print progress messages (default: True).
        
        Returns
        -------
        dict
            Dictionary with saved file paths organized by project/group/kweight.
        
        Output File Naming
        ------------------
        Files are named as:
          - {project}_{group}_chiR_kw{kweight}.dat  for R-space data
          - {project}_{group}_chiK_kw{kweight}.dat  for k-space data
        
        File Format
        -----------
        Each file contains:
          - Header lines (starting with #) with metadata
          - Column headers
          - Tab-separated data columns
        
        Example Usage
        -------------
        >>> cspbi3.save_processed_exafs(r'C:/Users/data/exafs_output')
        >>> cspbi3.save_processed_exafs(r'C:/Users/data/exafs_output', kweights=[2, 3])
        """
        from pathlib import Path
        import os
        from datetime import datetime
        
        # Set default kweights
        if kweights is None:
            kweights = [1, 2, 3]
        
        # Ensure output path exists
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"Saving processed EXAFS data to: {output_path}")
            print(f"K-weights to save: {kweights}")
        
        # Track saved files
        saved_files = {}
        
        for proj_name, project in self.projects.items():
            saved_files[proj_name] = {}
            
            for group_name, group in project.groups.items():
                saved_files[proj_name][group_name] = {}
                
                # Check for required data
                has_k = hasattr(group, 'k')
                has_chi = hasattr(group, 'chi_tapered') if use_tapered_chi else hasattr(group, 'chi')
                
                if not has_k:
                    if verbose:
                        print(f"  Skipping {proj_name}/{group_name}: Missing k data")
                    continue
                
                # Get the chi data
                k = group.k
                if use_tapered_chi and hasattr(group, 'chi_tapered'):
                    chi = group.chi_tapered
                elif hasattr(group, 'chi'):
                    chi = group.chi
                else:
                    if verbose:
                        print(f"  Skipping {proj_name}/{group_name}: Missing chi data")
                    continue
                
                # Sanitize names for filenames (remove special characters)
                safe_proj_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in proj_name)
                safe_group_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in group_name)
                
                for kw in kweights:
                    saved_files[proj_name][group_name][kw] = {}
                    
                    # ----------------------------------------------------------------
                    # Save chi(R) file
                    # ----------------------------------------------------------------
                    r_attr = f"best_r_kw{kw}"
                    chir_mag_attr = f"best_chir_mag_kw{kw}"
                    chir_re_attr = f"best_chir_re_kw{kw}"
                    chir_im_attr = f"best_chir_im_kw{kw}"
                    
                    has_r_data = (hasattr(group, r_attr) and 
                                  hasattr(group, chir_mag_attr) and 
                                  hasattr(group, chir_re_attr))
                    
                    if has_r_data:
                        r_data = getattr(group, r_attr)
                        chir_mag = getattr(group, chir_mag_attr)
                        chir_re = getattr(group, chir_re_attr)
                        
                        # Get optional parameters
                        kmin = getattr(group, f"best_kmin_kw{kw}", None)
                        kmax = getattr(group, f"best_kmax_kw{kw}", None)
                        quality_score = getattr(group, f"ft_quality_score_kw{kw}", None)
                        
                        # Build header
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        header_lines = [
                            f"# EXAFS Chi(R) Data",
                            f"# Generated by XASmu2r.save_processed_exafs",
                            f"# Date: {timestamp}",
                            f"# Project: {proj_name}",
                            f"# Group: {group_name}",
                            f"# K-weight: {kw}",
                        ]
                        if kmin is not None:
                            header_lines.append(f"# kmin: {kmin:.4f} Angstrom^-1")
                        if kmax is not None:
                            header_lines.append(f"# kmax: {kmax:.4f} Angstrom^-1")
                        if quality_score is not None:
                            header_lines.append(f"# Quality Score: {quality_score:.4f}")
                        header_lines.append(f"#")
                        
                        # Prepare data columns
                        if include_imaginary and hasattr(group, chir_im_attr):
                            chir_im = getattr(group, chir_im_attr)
                            header_lines.append(f"# Columns: R(Angstrom)  |Chi(R)|  Re[Chi(R)]  Im[Chi(R)]")
                            data_stack = np.column_stack([r_data, chir_mag, chir_re, chir_im])
                        else:
                            header_lines.append(f"# Columns: R(Angstrom)  |Chi(R)|  Re[Chi(R)]")
                            data_stack = np.column_stack([r_data, chir_mag, chir_re])
                        
                        # Write file
                        chir_filename = f"{safe_proj_name}_{safe_group_name}_chiR_kw{kw}.dat"
                        chir_filepath = output_path / chir_filename
                        
                        with open(chir_filepath, 'w') as f:
                            f.write('\n'.join(header_lines) + '\n')
                            for row in data_stack:
                                f.write('\t'.join(f"{val:.8e}" for val in row) + '\n')
                        
                        saved_files[proj_name][group_name][kw]['chiR'] = str(chir_filepath)
                        if verbose:
                            print(f"  Saved: {chir_filename}")
                    else:
                        if verbose:
                            print(f"  No R-space data for {proj_name}/{group_name} kw={kw}")
                    
                    # ----------------------------------------------------------------
                    # Save k-weighted chi(k) file
                    # ----------------------------------------------------------------
                    # Compute k-weighted chi
                    weighted_chi = chi * (k ** kw)
                    
                    # Get best kmin/kmax for this k-weight if available
                    kmin = getattr(group, f"best_kmin_kw{kw}", None)
                    kmax = getattr(group, f"best_kmax_kw{kw}", None)
                    quality_score = getattr(group, f"ft_quality_score_kw{kw}", None)
                    
                    # Build header
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    header_lines = [
                        f"# EXAFS k-weighted Chi(k) Data",
                        f"# Generated by XASmu2r.save_processed_exafs",
                        f"# Date: {timestamp}",
                        f"# Project: {proj_name}",
                        f"# Group: {group_name}",
                        f"# K-weight: {kw}",
                        f"# Chi source: {'chi_tapered' if use_tapered_chi else 'chi'}",
                    ]
                    if kmin is not None:
                        header_lines.append(f"# Optimal kmin: {kmin:.4f} Angstrom^-1")
                    if kmax is not None:
                        header_lines.append(f"# Optimal kmax: {kmax:.4f} Angstrom^-1")
                    if quality_score is not None:
                        header_lines.append(f"# Quality Score: {quality_score:.4f}")
                    header_lines.append(f"#")
                    header_lines.append(f"# Columns: k(Angstrom^-1)  k^{kw}*Chi(k)")
                    
                    # Write file
                    chik_filename = f"{safe_proj_name}_{safe_group_name}_chiK_kw{kw}.dat"
                    chik_filepath = output_path / chik_filename
                    
                    data_stack = np.column_stack([k, weighted_chi])
                    
                    with open(chik_filepath, 'w') as f:
                        f.write('\n'.join(header_lines) + '\n')
                        for row in data_stack:
                            f.write('\t'.join(f"{val:.8e}" for val in row) + '\n')
                    
                    saved_files[proj_name][group_name][kw]['chiK'] = str(chik_filepath)
                    if verbose:
                        print(f"  Saved: {chik_filename}")
        
        # Print summary
        if verbose:
            total_files = sum(
                len(group_data.get(kw, {}))
                for proj_data in saved_files.values()
                for group_data in proj_data.values()
                for kw in kweights
            )
            print(f"\nTotal files saved: {total_files}")
            print(f"Output directory: {output_path}")
        
        return saved_files

    def save_corrected_projects(self, output_dir=r'C:\Users\kwill\Keenan_UCB-O365\OneDrive - UCB-O365\Data\XAS\NSLS-II\AthenaProjectFiles\fromNotebook'):
        """
        For every group in every project stored in self.projects that has 'chi' and 'norm_corrected':
          - Save a copy of the original norm as group.norm_original,
          - Overwrite group.norm with group.norm_corrected,
          - And add the group to a combined Athena project file.
        
        The combined project file is saved to output_dir with a timestamp in its filename.
        
        Parameters:
            output_dir (str or Path): Directory where the combined Athena project file will be saved.
        """
        from larch.io import create_athena
        import os
        from datetime import datetime
        from pathlib import Path
        import copy

        # Ensure the output directory exists
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
            print(f"Created output directory: {output_dir}")

        # Create a timestamp for a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_file = output_dir / f"all_projects_combined_{timestamp}.prj"
        print(f"\nCreating combined project file with all groups: {project_file}")

        # Create a new Athena project
        athena_project = create_athena()

        processed_count = 0
        projects_with_groups = set()

        # Loop over each project and group in self.projects
        for proj_name, project in self.projects.items():
            for group_name, group in project.groups.items():
                # Check if the group contains the required data
                if hasattr(group, 'chi') and hasattr(group, 'norm_corrected'):
                    # Deep copy the group so that we don't modify the original in memory.
                    group_copy = copy.deepcopy(group)
                    
                    # Save a copy of the original norm, if available.
                    if hasattr(group_copy, 'norm'):
                        group_copy.norm_original = group_copy.norm.copy()
                    else:
                        group_copy.norm_original = None
                    
                    # Overwrite norm with the corrected data.
                    if hasattr(group_copy, 'norm_corrected'):
                        group_copy.norm = group_copy.norm_corrected.copy()
                    
                    # Create a combined name from the project and group names.
                    combined_name = f"{proj_name}_{group_name}"
                    
                    # Add the group to the new Athena project.
                    try:
                        athena_project.add_group(group_copy)
                        print(f"  Adding group '{combined_name}' to the combined file")
                        processed_count += 1
                        projects_with_groups.add(proj_name)
                    except Exception as e:
                        print(f"  Error adding group '{combined_name}': {str(e)}")
                else:
                    missing = []
                    if not hasattr(group, 'chi'):
                        missing.append('chi')
                    if not hasattr(group, 'norm_corrected'):
                        missing.append('norm_corrected')
                    print(f"  Skipping group '{group_name}': Missing {', '.join(missing)}")
        
        # Save the Athena project if any groups have been added.
        if processed_count > 0:
            try:
                athena_project.save(str(project_file))
                print(f"  Successfully saved {processed_count} groups to {project_file}")
            except Exception as e:
                import traceback
                print(f"  Error saving combined project: {str(e)}")
                traceback.print_exc()
        else:
            print("  No processable groups found in any project, no file created")
        
        print(f"\nSaved {processed_count} groups across {len(projects_with_groups)} projects")
        print(f"File saved at: {os.path.abspath(str(project_file))}")
    def build_params_for_iteration(self, iteration, prev_result=None, So_2=0.94):
        """
        Build a parameter group for the given iteration.
        
        Parameters:
            iteration (int): The iteration number
            prev_result (object, optional): Previous fit result with parameters to use as starting points
            So_2 (float, optional): Amplitude reduction factor (default: 0.94)
            
        Returns:
            param_group: Group of parameters configured for fitting
        """
        print(f"Building parameter group for iteration {iteration}...")
        from larch.fitting import param, param_group
        
        # Create parameters group with a descriptive name
        params = param_group(name=f"fit_params_iter{iteration}")
        
        # Define default parameters (if no previous results available)
        params.amp = param(So_2, vary=False, min=0.5, max=1.2)
        params.del_e0 = param(0.0, vary=True, min=-15, max=15)
        params.n_PbI = param(3.0, vary=True, min=1.0, max=6.0)
        params.pbi_sig2 = param(0.01, vary=True, min=0.001, max=0.03)
        params.pbi_delr = param(0.0, vary=True, min=-0.2, max=0.2)
        
        # Update from previous results if available
        if prev_result is not None and hasattr(prev_result, 'params'):
            print(f"  Using values from previous fit result")
            for name in ['amp', 'del_e0', 'n_PbI', 'pbi_sig2', 'pbi_delr']:
                if name in prev_result.params:
                    # Keep the same parameter properties (vary, min, max) but update value
                    value = prev_result.params[name].value
                    if hasattr(params, name) and hasattr(params[name], 'value'):
                        params[name].value = value
                    else:
                        setattr(params, name, param(value, 
                                                vary=params[name].vary, 
                                                min=params[name].min, 
                                                max=params[name].max))
        
        # Add debugging to verify params isn't None
        print(f"  Created parameters for iteration {iteration}: {params}")
        
        return params

    def build_linked_params(self, iteration, prev_result=None, s02_value=0.78, 
                            fix_s02=True, fix_e0=False, available_elements=None):
        """
        Build a parameter group with intelligent linking for multi-path EXAFS fits.
        
        Based on the strategy from CsPbI3_1p0M_DMSO_RelaxedFit.txt:
        - Core parameters: N_I, N_O for coordination (S and C link to N_O)
        - sigma2_I, sigma2_O for Debye-Waller (S and C link to sigma2_O)
        - delr_I, delr_O, delr_S for distance shifts (delr_C is fixed)
        - e0 is shared across all paths
        - s02 is fixed at 0.78
        
        Parameters:
            iteration (int): Current iteration number
            prev_result (object): Previous fit result with parameters
            s02_value (float): S0^2 amplitude reduction factor (default: 0.78)
            fix_s02 (bool): Whether to fix s02 (default: True)
            fix_e0 (bool): Whether to fix e0 (default: False)
            available_elements (list): Elements present in paths (e.g., ['I', 'O', 'S', 'C'])
        
        Returns:
            param_group: Parameter group configured for linked fitting
        """
        from larch.fitting import param, param_group
        
        if available_elements is None:
            available_elements = ['I', 'O']  # Default to just I and O
        
        print(f"\nBuilding linked parameter group for iteration {iteration}")
        print(f"  Elements included: {available_elements}")
        
        params = param_group(name=f"linked_params_iter{iteration}")
        
        # --- Core amplitude and energy shift parameters ---
        params.s02 = param(s02_value, vary=not fix_s02, min=0.5, max=1.0)
        params.e0 = param(0.0, vary=not fix_e0, min=-15, max=15)
        
        # --- Coordination number parameters ---
        # N_I: iodine coordination (independent)
        params.N_I = param(3.0, vary=True, min=0.5, max=8.0)
        # N_O: oxygen/DMSO coordination (independent, S and C link to this)
        params.N_O = param(4.0, vary=True, min=0.5, max=8.0)
        
        # --- Sigma2 (Debye-Waller) parameters ---
        # sigma2_I: for Pb-I paths (independent)
        params.sigma2_I = param(0.010, vary=True, min=0.001, max=0.03)
        # sigma2_O: for Pb-O paths (S and C link to this)
        params.sigma2_O = param(0.035, vary=True, min=0.001, max=0.06)
        
        # --- Delta R (distance shift) parameters ---
        # delr_I: for Pb-I paths (independent)
        params.delr_I = param(0.0, vary=True, min=-0.15, max=0.15)
        # delr_O: for Pb-O paths (independent)
        params.delr_O = param(0.0, vary=True, min=-0.15, max=0.15)
        
        # Additional delr parameters for S and C (if these elements are present)
        if 'S' in available_elements:
            params.delr_S = param(-0.05, vary=True, min=-0.20, max=0.10)
        if 'C' in available_elements:
            # C is typically fixed in the reference fit
            params.delr_C = param(-0.057, vary=False, min=-0.15, max=0.15)
        
        # --- Update from previous results if available ---
        if prev_result is not None and hasattr(prev_result, 'params'):
            print(f"  Updating from previous fit result")
            prev_params = prev_result.params
            
            # Map old parameter names to new names
            param_mapping = {
                'amp': 's02',
                'del_e0': 'e0', 
                'n_PbI': 'N_I',
                'n_PbO': 'N_O',
                'pbi_sig2': 'sigma2_I',
                'pbo_sig2': 'sigma2_O',
                'pbi_delr': 'delr_I',
                'pbo_delr': 'delr_O',
                # Direct mappings for new names
                's02': 's02',
                'e0': 'e0',
                'N_I': 'N_I',
                'N_O': 'N_O',
                'sigma2_I': 'sigma2_I',
                'sigma2_O': 'sigma2_O',
                'delr_I': 'delr_I',
                'delr_O': 'delr_O',
                'delr_S': 'delr_S',
                'delr_C': 'delr_C'
            }
            
            for old_name, new_name in param_mapping.items():
                if old_name in prev_params and hasattr(params, new_name):
                    try:
                        params[new_name].value = prev_params[old_name].value
                        print(f"    {new_name} = {prev_params[old_name].value:.4f} (from {old_name})")
                    except:
                        pass
        
        # --- Reset N_O if non-physical for iteration >= 3 ---
        if iteration >= 3 and params.N_O.value < 1:
            print(f"    N_O = {params.N_O.value:.4f} is non-physical (<1), resetting to 3.0")
            params.N_O.value = 3.0
        
        # Print parameter summary
        print(f"  Parameter summary:")
        print(f"    s02 = {params.s02.value:.3f} (vary={params.s02.vary})")
        print(f"    e0 = {params.e0.value:.3f} (vary={params.e0.vary})")
        print(f"    N_I = {params.N_I.value:.3f}, N_O = {params.N_O.value:.3f}")
        print(f"    sigma2_I = {params.sigma2_I.value:.4f}, sigma2_O = {params.sigma2_O.value:.4f}")
        print(f"    delr_I = {params.delr_I.value:.4f}, delr_O = {params.delr_O.value:.4f}")
        
        return params

    def identify_path_elements(self, path):
        """
        Identify the scattering elements in a FEFF path from its geometry.
        
        Parameters:
            path: FeffPath object with geometry information
        
        Returns:
            dict: {'primary': element, 'all_scatterers': [elements], 
                   'is_ms': bool, 'nleg': int, 'reff': float}
        """
        result = {
            'primary': None,
            'all_scatterers': [],
            'is_ms': False,
            'nleg': 2,
            'reff': getattr(path, 'reff', 0.0)
        }
        
        # Try to get geometry from feffdat
        geom = None
        if hasattr(path, '_feffdat') and hasattr(path._feffdat, 'geom'):
            geom = path._feffdat.geom
            result['nleg'] = getattr(path._feffdat, 'nleg', 2)
        elif hasattr(path, 'geom'):
            geom = path.geom
            result['nleg'] = getattr(path, 'nleg', 2)
        
        result['is_ms'] = result['nleg'] > 2
        
        if geom is not None:
            # Extract scattering atoms (skip absorber at ipot=0)
            for atom in geom:
                if len(atom) >= 4:  # (symbol, x, y, z, ipot) or similar
                    atom_symbol = atom[0]
                    if isinstance(atom[-1], int) and atom[-1] != 0:  # Not the absorber
                        result['all_scatterers'].append(atom_symbol)
            
            # Primary scatterer is the first one for SS, or first unique for MS
            if result['all_scatterers']:
                result['primary'] = result['all_scatterers'][0]
        
        # Fallback: try to identify from label
        if result['primary'] is None and hasattr(path, 'label'):
            label = path.label.upper()
            for elem in ['I', 'O', 'S', 'C', 'N', 'Cl', 'Br']:
                if f'PB-{elem}' in label or f'PB_{elem}' in label:
                    result['primary'] = elem
                    break
        
        return result

    def assign_linked_params_to_path(self, path, iteration, fix_delr=False):
        """
        Assign linked parameter expressions to a FEFF path based on the reference fit strategy.
        
        Parameter Assignment Rules (from CsPbI3_1p0M_DMSO_RelaxedFit.txt):
        - Pb-I SS: s02='N_I * s02', sigma2='sigma2_I', deltar='delr_I'
        - Pb-O SS: s02='N_O * s02', sigma2='sigma2_O', deltar='delr_O'
        - Pb-S SS: s02='N_O * s02' (linked!), sigma2='sigma2_O' (linked!), deltar='delr_S'
        - Pb-C SS: s02='N_O * 2 * s02', sigma2='sigma2_O * 1.1', deltar='delr_C' (fixed)
        - Pb-O-I MS: s02='N_O * s02', sigma2='(sigma2_I + sigma2_O)/2', deltar='delr_I + delr_O'
        - Pb-S-O MS: s02='N_O * s02', sigma2='sigma2_O', deltar='delr_O'
        
        Parameters:
            path: FeffPath object to configure
            iteration (int): Current iteration number
            fix_delr (bool): If True, use fixed delr values for problematic paths
        
        Returns:
            path: Configured path with parameter assignments
        """
        # Store original degeneracy
        path.orig_degen = getattr(path, 'degen', 1.0)
        path.degen = 1.0  # Always set to 1 for coordination number fitting
        
        # Shared e0
        path.e0 = 'e0'
        
        # Identify path type
        path_info = self.identify_path_elements(path)
        primary = path_info['primary']
        all_scatterers = path_info['all_scatterers']
        is_ms = path_info['is_ms']
        
        # Assign based on element and path type
        if is_ms:
            # --- Multiple Scattering paths ---
            scatterer_set = set(all_scatterers)
            
            if 'I' in scatterer_set and 'O' in scatterer_set:
                # Pb-O-I type MS path
                path.s02 = '1 * s02'
                path.sigma2 = '(sigma2_I + sigma2_O) / 2'
                path.deltar = 'delr_I + delr_O' if not fix_delr else 0.0
                path.param_type = 'MS_O_I'
                
            elif 'S' in scatterer_set and 'O' in scatterer_set:
                # Pb-S-O type MS path
                path.s02 = 'N_O * s02'
                path.sigma2 = '(sigma2_O + sigma2_S) / 2'  # Average of O and S sigma2
                path.deltar = 'delr_O' if not fix_delr else 0.0
                path.param_type = 'MS_S_O'
                
            elif 'I' in scatterer_set:
                # Pb-I-I or other I-containing MS
                path.s02 = '1 * s02'
                path.sigma2 = 'sigma2_I * 1.5'
                path.deltar = 'delr_I' if not fix_delr else 0.0
                path.param_type = 'MS_I'
                
            elif 'O' in scatterer_set:
                # Pb-O-O or other O-containing MS
                path.s02 = '1 * s02'
                path.sigma2 = 'sigma2_O * 1.3'
                path.deltar = 'delr_O' if not fix_delr else 0.0
                path.param_type = 'MS_O'
                
            else:
                # Generic MS - link to O parameters
                path.s02 = '1 * s02'
                path.sigma2 = 'sigma2_O * 1.5'
                path.deltar = '0' if not fix_delr else 0.0
                path.param_type = 'MS_other'
                
        else:
            # --- Single Scattering paths ---
            if primary == 'I':
                path.s02 = 'N_I * s02'
                path.sigma2 = 'sigma2_I'
                path.deltar = 'delr_I' if not fix_delr else 0.0
                path.param_type = 'SS_I'
                
            elif primary == 'O':
                path.s02 = 'N_O * s02'
                path.sigma2 = 'sigma2_O'
                path.deltar = 'delr_O' if not fix_delr else 0.0
                path.param_type = 'SS_O'
                
            elif primary == 'S':
                # S links to O for coordination and sigma2
                path.s02 = 'N_O * s02'  # LINKED to N_O!
                path.sigma2 = 'sigma2_O'  # LINKED to sigma2_O!
                path.deltar = 'delr_S' if not fix_delr else 0.0
                path.param_type = 'SS_S'
            
            elif primary == 'Cs':
                # Cs is often present in the reference fit but not in our samples, so link to O but with a scaling factor
                path.s02 = '1 * s02'  # Assume Cs contributes less than O
                path.sigma2 = 'sigma2_Cs'  # Assume more disorder for Cs
                path.deltar = 'delr_Cs' if not fix_delr else 0.0
                path.param_type = 'SS_Cs'
                
            elif primary == 'C':
                # C uses 2*N_O and scaled sigma2_O
                path.s02 = 'N_O * 2 * s02'
                path.sigma2 = 'sigma2_O * 1.1'
                path.deltar = 'delr_C'  # Always use delr_C (typically fixed)
                path.param_type = 'SS_C'
                
            elif primary == 'N':
                # N is similar to O (nitrogen in ligands)
                path.s02 = 'N_O * s02'
                path.sigma2 = 'sigma2_O'
                path.deltar = 'delr_O' if not fix_delr else 0.0
                path.param_type = 'SS_N'
                
            else:
                # Unknown - default to O-like behavior
                path.s02 = 'N_O * s02'
                path.sigma2 = 'sigma2_O'
                path.deltar = 'delr_O' if not fix_delr else 0.0
                path.param_type = 'SS_other'
        
        # Store path info for later reference
        path.linked_params_info = path_info
        
        return path

    def robust_feffit(self, params, datasets, max_retries=3, verbose=True):
        """
        Perform feffit with automatic fallback to fixed parameters if fit fails.
        
        If the fit fails to converge, this method progressively fixes parameters
        starting with those that have high uncertainties or correlations:
        1. First try normal fit
        2. If fails, fix s02 (if not already fixed)
        3. If fails, fix e0
        4. If fails, fix delr parameters
        5. If fails, fix sigma2 parameters
        
        Parameters:
            params: Parameter group
            datasets: Dataset or list of datasets for fitting
            max_retries (int): Maximum number of retry attempts
            verbose (bool): Print diagnostic messages
        
        Returns:
            fit_result: Fit result (may have some fixed parameters)
            success (bool): Whether fit converged
            fixed_params (list): List of parameters that were fixed
        """
        from larch.xafs import feffit
        import copy
        
        # Ensure datasets is a list
        if not isinstance(datasets, list):
            datasets = [datasets]
        
        fixed_params = []
        
        # Parameters to fix in order of priority
        fix_order = [
            ('s02', 0.78),
            ('e0', 0.0),
            ('delr_C', -0.057),
            ('delr_S', -0.05),
            ('sigma2_O', 0.035),
            ('sigma2_I', 0.010)
        ]
        
        for retry in range(max_retries + 1):
            try:
                # Make a copy of params to avoid modifying the original
                fit_params = copy.deepcopy(params)
                
                # Apply previous fixes
                for pname, pval in fixed_params:
                    if hasattr(fit_params, pname):
                        fit_params[pname].vary = False
                        fit_params[pname].value = pval
                
                # Try the fit
                result = feffit(fit_params, datasets)
                
                # Check if fit is reasonable
                if result.rfactor < 0.5 and not np.isnan(result.chi_square):
                    if verbose:
                        print(f"  Fit converged: R-factor = {result.rfactor:.6f}")
                        if fixed_params:
                            print(f"  Fixed parameters: {[p[0] for p in fixed_params]}")
                    return result, True, fixed_params
                else:
                    raise ValueError(f"Fit result unreasonable: R-factor={result.rfactor}")
                    
            except Exception as e:
                if verbose:
                    print(f"  Fit attempt {retry + 1} failed: {str(e)[:50]}...")
                
                if retry < len(fix_order):
                    # Fix next parameter in order
                    pname, pval = fix_order[retry]
                    if hasattr(params, pname) and params[pname].vary:
                        fixed_params.append((pname, params[pname].value if params[pname].value != 0 else pval))
                        if verbose:
                            print(f"    Fixing {pname} = {fixed_params[-1][1]:.4f}")
        
        # Final fallback - try with all parameters fixed except N_I and N_O
        if verbose:
            print("  Final fallback: fixing all parameters except N_I and N_O")
        
        try:
            fit_params = copy.deepcopy(params)
            for pname in ['s02', 'e0', 'delr_I', 'delr_O', 'delr_S', 'delr_C', 'sigma2_I', 'sigma2_O']:
                if hasattr(fit_params, pname):
                    fit_params[pname].vary = False
            
            result = feffit(fit_params, datasets)
            return result, True, [('all_except_N', 'fixed')]
            
        except Exception as e:
            if verbose:
                print(f"  All fit attempts failed: {e}")
            return None, False, fixed_params

    def fit_with_linked_params(self, group, paths, iteration, prev_result=None,
                                available_elements=None, verbose=True):
        """
        Perform EXAFS fit using the linked parameter strategy.
        
        This is a convenience method that:
        1. Builds linked parameters
        2. Assigns parameters to paths based on their element type
        3. Performs robust fitting with automatic fallback
        
        Parameters:
            group: Data group with k and chi attributes
            paths: List of FEFF paths to fit
            iteration (int): Current iteration number  
            prev_result: Previous fit result for parameter initialization
            available_elements (list): Elements in paths (auto-detected if None)
            verbose (bool): Print diagnostic messages
        
        Returns:
            fit_result: Fit result
            success (bool): Whether fit converged
            params: Final parameter group used
        """
        from larch.xafs import feffit_dataset
        from larch import Group
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Fitting with linked parameters - Iteration {iteration}")
            print(f"{'='*60}")
            print(f"  Paths: {len(paths)}")
        
        # Auto-detect elements if not provided
        if available_elements is None:
            available_elements = set()
            for path in paths:
                info = self.identify_path_elements(path)
                if info['primary']:
                    available_elements.add(info['primary'])
                available_elements.update(info['all_scatterers'])
            available_elements = list(available_elements)
            if verbose:
                print(f"  Auto-detected elements: {available_elements}")
        
        # Build linked parameters
        params = self.build_linked_params(
            iteration=iteration,
            prev_result=prev_result,
            s02_value=0.78,
            fix_s02=True,
            available_elements=available_elements
        )
        
        # Assign parameters to each path
        configured_paths = []
        for path in paths:
            configured_path = self.assign_linked_params_to_path(path, iteration)
            configured_paths.append(configured_path)
            if verbose:
                ptype = getattr(configured_path, 'param_type', 'unknown')
                print(f"    Path: {getattr(path, 'label', 'unknown')[:30]:30s} -> {ptype}")
        
        # Create dataset
        # --- FIT RANGES DEFINED HERE ---
        # get_transform_for_group() returns TransformGroup with:
        #   k-range: kmin, kmax from group's best_kmin/best_kmax or defaults (3.0, 12.0)
        #   R-range: rmin=rbkg (default 0.9), rmax=4.0 Å
        trans = self.get_transform_for_group(group)
        temp_data = Group(k=group.k, chi=group.chi)
        dataset = feffit_dataset(data=temp_data, pathlist=configured_paths, transform=trans)
        
        # Perform robust fit
        result, success, fixed_params = self.robust_feffit(params, [dataset], verbose=verbose)
        
        if success and result is not None:
            if verbose:
                print(f"\n  Final fit results:")
                print(f"    R-factor = {result.rfactor:.6f}")
                print(f"    Chi-square = {result.chi_square:.2f}")
                for pname in ['N_I', 'N_O', 'sigma2_I', 'sigma2_O', 'e0']:
                    if pname in result.params:
                        p = result.params[pname]
                        err = getattr(p, 'stderr', None)
                        err_str = f" +/- {err:.4f}" if err else " (no error)"
                        print(f"    {pname} = {p.value:.4f}{err_str}")
        else:
            print(f"\n  WARNING: Fit did not converge for iteration {iteration}")
        
        return result, success, params
    def iterative_fit_linked_all(self, feff_category, iteration, feff_base_dir, 
                                include_ms=False, max_paths_to_add=3, verbose=True,
                                plot_results=True, aicc_improvement_threshold=2.0):
        """
        Perform iterative EXAFS fitting for all groups using linked parameter strategy.
        
        Use this starting from iteration 3 (after Pb-I and Pb-O are fit).
        
        Parameters:
            feff_category (str): Category of paths to add (e.g., 'Pb-S', 'Pb-C', 'MS')
            iteration (int): Current iteration number (should be >= 3)
            feff_base_dir (str): Directory containing FEFF calculations
            include_ms (bool): If True, also evaluate MS paths for the category
            max_paths_to_add (int): Maximum number of new paths to add per group
            verbose (bool): Print detailed progress messages
            plot_results (bool): If True, plot comparison with previous iteration
        
        Returns:
            dict: Results dictionary with fit parameters for each group
        """
        import os
        from pathlib import Path
        import copy
        import matplotlib.pyplot as plt
        from larch.xafs import feffpath, feffit_dataset
        from larch import Group
        
        print(f"\n{'='*70}")
        print(f"ITERATIVE FIT WITH LINKED PARAMETERS - Iteration {iteration}")
        print(f"Adding {feff_category} paths")
        print(f"{'='*70}")
        
        feff_base_dir = Path(feff_base_dir)
        results = {}
        total_processed = 0
        total_improved = 0
        
        # --- Helper to analyze FEFF paths ---
        def analyze_feff_path(file_path, category):
            path_info = {'is_match': False, 'path': file_path, 
                        'filename': os.path.basename(file_path)}
            try:
                fp_obj = feffpath(filename=file_path)
                if hasattr(fp_obj._feffdat, 'absorber') and fp_obj._feffdat.absorber != 'Pb':
                    return path_info
                
                target_element = category.split('-')[1] if '-' in category else category
                
                scatterers = []
                target_found = False
                for atom in fp_obj._feffdat.geom:
                    if atom[-1] != 0:
                        scatterers.append(atom[0])
                        if atom[0].upper() == target_element.upper():
                            target_found = True
                
                if target_found or (category.upper() == 'MS' and fp_obj._feffdat.nleg > 2):
                    path_info['is_match'] = True
                    path_info['reff'] = fp_obj._feffdat.reff
                    path_info['nleg'] = fp_obj._feffdat.nleg
                    path_info['degen'] = fp_obj._feffdat.degen
                    path_info['feffpath'] = fp_obj
                    path_info['scatterers'] = scatterers
                    path_info['path_type'] = 'ss' if fp_obj._feffdat.nleg == 2 else 'ms'
                    path_info['label'] = f"Pb-{'-'.join(scatterers)}: {path_info['reff']:.3f}Å"
                return path_info
            except Exception as e:
                return path_info
        
        # --- Collect candidate paths ---
        print(f"\nScanning {feff_base_dir} for {feff_category} paths...")
        candidate_paths = []
        for root, dirs, files in os.walk(feff_base_dir):
            for file in files:
                if file.lower().endswith('.dat') and 'feff' in file.lower():
                    full_path = os.path.join(root, file)
                    info = analyze_feff_path(full_path, feff_category)
                    if info.get('is_match'):
                        if not include_ms and info.get('path_type') == 'ms':
                            continue
                        candidate_paths.append(info)
        
        candidate_paths.sort(key=lambda x: x.get('reff', float('inf')))
        print(f"  Found {len(candidate_paths)} candidate paths")
        
        if not candidate_paths:
            return results
        
        # --- Process each group ---
        for proj_name, project in self.projects.items():
            print(f"\nProject: {proj_name}")
            
            for group_name, group in project.groups.items():
                group_id = f"{proj_name}.{group_name}"
                print(f"\n  Group: {group_id}")
                
                if not (hasattr(group, 'k') and hasattr(group, 'chi')):
                    print(f"    Skipping - missing k or chi")
                    continue
                
                # Get previous results
                prev_fit = getattr(group, f'fit_result_iter{iteration-1}', None)
                prev_paths = getattr(group, f'paths_iter{iteration-1}', [])
                
                if prev_fit is None:
                    print(f"    Skipping - no previous result")
                    continue
                
                prev_rfactor = prev_fit.rfactor
                print(f"    Previous R-factor: {prev_rfactor:.6f}, paths: {len(prev_paths)}")
                
                # --- FIT RANGES DEFINED HERE ---
                # get_transform_for_group() returns TransformGroup with:
                #   k-range: kmin, kmax from group's best_kmin/best_kmax or defaults (3.0, 12.0)
                #   R-range: rmin=rbkg (default 0.9), rmax=4.0 Å
                trans = self.get_transform_for_group(group)
                filtered = [p for p in candidate_paths 
                        if trans.rmin <= p.get('reff', 0) <= trans.rmax + 0.5]
                
                # --- MS CONSTRAINT: Only include MS paths whose atoms have SS counterparts ---
                if feff_category.upper() == 'MS':
                    # Extract elements from existing SS paths in the model
                    existing_ss_elements = set()
                    for path in prev_paths:
                        if hasattr(path, '_feffdat') and hasattr(path._feffdat, 'geom'):
                            if path._feffdat.nleg == 2:  # Single scattering
                                for atom in path._feffdat.geom:
                                    if atom[-1] != 0:  # Not absorber
                                        existing_ss_elements.add(atom[0].upper())
                    
                    if existing_ss_elements:
                        print(f"    Elements with SS paths in model: {existing_ss_elements}")
                        
                        # Filter MS paths: all scatterers must have SS counterparts
                        ms_filtered = []
                        for cand in filtered:
                            if cand.get('path_type') == 'ms':
                                scatterers = set(s.upper() for s in cand.get('scatterers', []))
                                if scatterers.issubset(existing_ss_elements):
                                    ms_filtered.append(cand)
                                else:
                                    missing = scatterers - existing_ss_elements
                                    if verbose:
                                        print(f"    Skipping {cand.get('label', 'MS path')}: "
                                              f"missing SS for {missing}")
                            else:
                                ms_filtered.append(cand)  # Keep SS paths
                        filtered = ms_filtered
                        print(f"    {len(filtered)} MS paths remain after SS constraint")
                    else:
                        print(f"    Warning: No SS elements found in model, skipping MS paths")
                        filtered = []
                
                if not filtered:
                    print(f"    No candidates in R-range")
                    continue
                
                # Compute AICc for the current (previous-iteration) model
                prev_aicc_info = self._compute_aicc(prev_fit)
                if verbose:
                    print(f"    Current model AICc: {prev_aicc_info['aicc']:.2f} "
                          f"(Nind={prev_aicc_info['nind']:.1f}, k={prev_aicc_info['k']}, "
                          f"RSS={prev_aicc_info['rss']:.4f})")

                # Test each candidate
                test_results = []
                for cand in filtered[:max_paths_to_add * 2]:
                    cand_record = {
                        'path_info': cand, 'path': None,
                        'rfactor': np.inf, 'aicc': np.inf,
                        'delta_aicc': np.inf, 'sanity_pass': False,
                        'rejection_reason': '', 'result': None,
                    }
                    try:
                        new_path = feffpath(cand['feffpath'].filename)
                        new_path.label = cand.get('label', 'new')
                        cand_record['path'] = new_path
                        test_paths = copy.deepcopy(prev_paths) + [new_path]
                        
                        result, success, _ = self.fit_with_linked_params(
                            group, test_paths, iteration, prev_fit, verbose=False)
                        
                        if not success or result is None:
                            cand_record['rejection_reason'] = 'fit did not converge'
                            test_results.append(cand_record)
                            continue

                        cand_record['result'] = result
                        cand_record['rfactor'] = result.rfactor

                        # Physical sanity check
                        sane, reason = self._fit_is_physically_sane(
                            result, paths=test_paths, verbose=verbose)
                        cand_record['sanity_pass'] = sane
                        if not sane:
                            cand_record['rejection_reason'] = f'physical sanity failed: {reason}'
                            test_results.append(cand_record)
                            if verbose:
                                print(f"      {cand['label'][:35]:35s} SANITY FAIL: {reason}")
                            continue

                        # Compute AICc for candidate model
                        cand_aicc = self._compute_aicc(result)
                        cand_record['aicc'] = cand_aicc['aicc']
                        if not cand_aicc['valid']:
                            cand_record['rejection_reason'] = f'AICc invalid: {cand_aicc["reason"]}'
                            test_results.append(cand_record)
                            continue

                        delta = cand_aicc['aicc'] - prev_aicc_info['aicc']
                        cand_record['delta_aicc'] = delta
                        improvement = prev_rfactor - result.rfactor
                        print(f"      {cand['label'][:35]:35s} "
                              f"AICc={cand_aicc['aicc']:+.2f}  "
                              f"ΔAICc={delta:+.2f}  "
                              f"Rfactor ΔR={improvement:+.6f}")
                    except Exception as e:
                        cand_record['rejection_reason'] = f'exception: {e}'
                    test_results.append(cand_record)
                
                # Rank candidates by AICc among sane, valid fits
                sane_candidates = [
                    tr for tr in test_results
                    if tr['sanity_pass'] and np.isfinite(tr['aicc'])
                ]
                sane_candidates.sort(key=lambda x: x['aicc'])

                # Accept only if AICc improves beyond threshold
                paths_to_add = []
                for sc in sane_candidates:
                    if not prev_aicc_info['valid']:
                        # If old model AICc is invalid, accept any valid new model
                        paths_to_add.append(sc)
                    elif (prev_aicc_info['aicc'] - sc['aicc']) >= aicc_improvement_threshold:
                        paths_to_add.append(sc)
                    if len(paths_to_add) >= max_paths_to_add:
                        break

                if not paths_to_add:
                    best_reason = 'no candidates passed AICc threshold'
                    if sane_candidates:
                        best_delta = sane_candidates[0]['delta_aicc']
                        best_reason = (f'best ΔAICc={best_delta:+.2f}, '
                                       f'threshold={aicc_improvement_threshold}')
                    elif test_results:
                        reasons = [tr['rejection_reason'] for tr in test_results if tr['rejection_reason']]
                        best_reason = '; '.join(set(reasons[:3]))
                    print(f"    No paths accepted: {best_reason}")
                    results[group_id] = {'rfactor': prev_rfactor, 'improved': False,
                                         'reason': best_reason}
                    continue
                
                # Final fit with selected paths
                final_paths = copy.deepcopy(prev_paths)
                for pa in paths_to_add:
                    final_paths.append(pa['path'])
                    print(f"    + {pa['path_info']['label']}  (ΔAICc={pa['delta_aicc']:+.2f})")
                
                final_result, success, _ = self.fit_with_linked_params(
                    group, final_paths, iteration, prev_fit, verbose=True)
                
                if not success or final_result is None:
                    print(f"    WARNING: Final fit did not converge for {group_id}")
                    results[group_id] = {'rfactor': prev_rfactor, 'improved': False,
                                         'converged': False}
                    continue

                # Final sanity and AICc gate
                final_sane, final_reason = self._fit_is_physically_sane(
                    final_result, paths=final_paths, verbose=verbose)
                final_aicc = self._compute_aicc(final_result)
                metrics = self._evaluate_model_selection_metrics(prev_fit, final_result)

                if not final_sane:
                    print(f"    REJECTED final fit: sanity failed ({final_reason})")
                    results[group_id] = {'rfactor': prev_rfactor, 'improved': False,
                                         'reason': f'final sanity: {final_reason}'}
                    continue

                if final_aicc['valid'] and prev_aicc_info['valid']:
                    if (prev_aicc_info['aicc'] - final_aicc['aicc']) < aicc_improvement_threshold:
                        print(f"    REJECTED final fit: ΔAICc={metrics['delta_aicc']:+.2f} "
                              f"(threshold={aicc_improvement_threshold})")
                        results[group_id] = {'rfactor': prev_rfactor, 'improved': False,
                                             'reason': f"ΔAICc={metrics['delta_aicc']:+.2f}"}
                        continue

                # Accept the final fit
                setattr(group, f'fit_result_iter{iteration}', final_result)
                setattr(group, f'paths_iter{iteration}', final_paths)
                improvement = prev_rfactor - final_result.rfactor
                print(f"\n    ACCEPTED: R-factor {prev_rfactor:.6f} -> {final_result.rfactor:.6f}")
                print(f"    AICc {prev_aicc_info['aicc']:.2f} -> {final_aicc['aicc']:.2f}  "
                      f"(ΔAICc={metrics['delta_aicc']:+.2f})")
                results[group_id] = {
                    'rfactor': final_result.rfactor, 'improvement': improvement,
                    'improved': True, 'paths_added': len(paths_to_add),
                    'aicc_old': prev_aicc_info['aicc'],
                    'aicc_new': final_aicc['aicc'],
                    'delta_aicc': metrics['delta_aicc'],
                }
                total_processed += 1
                if improvement > 0:
                    total_improved += 1
                
                # --- Plot comparison with previous iteration ---
                if plot_results:
                    self._plot_fit_comparison(group, group_id, iteration, 
                                             prev_fit, final_result, 
                                             prev_paths, final_paths,
                                             feff_category)
        
        # ===== SUMMARY: iterative_fit_linked_all =====
        print("\n" + "="*80)
        print(f"ITERATIVE_FIT_LINKED_ALL SUMMARY (Iteration {iteration}, Category: {feff_category})")
        print(f"  Model selection: AICc (threshold={aicc_improvement_threshold})")
        print("="*80)
        print(f"  Groups processed: {total_processed}")
        print(f"  Groups improved:  {total_improved}")
        print("-"*80)
        print("  Results per group:")
        print(f"  {'Group':<45s}  {'R_prev':>10s}  {'R_new':>10s}  {'AICc_old':>10s}  {'AICc_new':>10s}  {'ΔAICc':>8s}  {'#p':>3s}")
        print("-"*80)
        for group_id, res in results.items():
            if 'rfactor' in res:
                r_new = res['rfactor']
                improvement = res.get('improvement', 0)
                r_prev = r_new + improvement
                paths_added = res.get('paths_added', 0)
                aicc_old = res.get('aicc_old', float('nan'))
                aicc_new = res.get('aicc_new', float('nan'))
                d_aicc = res.get('delta_aicc', float('nan'))
                improved = "✓" if res.get('improved') else ""
                print(f"    {group_id:<40s}  {r_prev:>10.6f}  {r_new:>10.6f}  "
                      f"{aicc_old:>10.2f}  {aicc_new:>10.2f}  {d_aicc:>+8.2f}  "
                      f"{paths_added:>3d} {improved}")
            else:
                reason = res.get('reason', 'skipped')
                print(f"    {group_id:<40s}  (not improved: {reason})")
        print("="*80 + "\n")
        return results

    def _evaluate_path_delr(self, path, fit_result):
        """
        Evaluate the fitted ΔR used by a specific path.

        Works for numeric deltar, parameter names, or simple expressions.
        """
        import re

        deltar = getattr(path, 'deltar', 0.0)

        if isinstance(deltar, (int, float)):
            return float(deltar)

        if fit_result is None or not hasattr(fit_result, 'params'):
            return None

        expr = str(deltar).strip()
        if not expr:
            return 0.0

        params = fit_result.params

        if expr in params:
            try:
                return float(params[expr].value)
            except Exception:
                pass

        names = sorted(set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', expr)))
        safe_locals = {}
        for name in names:
            if name in params:
                safe_locals[name] = params[name].value

        try:
            return float(eval(expr, {"__builtins__": {}}, safe_locals))
        except Exception:
            return None   


    def _path_signature(self, path):
        """
        Return a robust signature for matching replacement candidates.
        """
        info = self.identify_path_elements(path)
        role = self._infer_path_role(path)

        return {
            'role': role,
            'primary': (info.get('primary') or 'UNKNOWN').upper(),
            'scatterers': tuple(sorted(s.upper() for s in info.get('all_scatterers', []))),
            'nleg': info.get('nleg', 2),
            'is_ms': info.get('is_ms', False),
        }


    def _extract_candidate_signature(self, fp_obj, filename=None):
        """
        Build a candidate signature from a FEFF path object.
        """
        scatterers = []
        nleg = getattr(fp_obj._feffdat, 'nleg', getattr(fp_obj, 'nleg', 2))
        reff = getattr(fp_obj._feffdat, 'reff', getattr(fp_obj, 'reff', None))

        if hasattr(fp_obj, '_feffdat') and hasattr(fp_obj._feffdat, 'geom'):
            for atom in fp_obj._feffdat.geom:
                try:
                    if atom[-1] != 0:
                        scatterers.append(str(atom[0]).upper())
                except Exception:
                    continue

        scatterers = tuple(sorted(scatterers))
        is_ms = nleg > 2
        primary = scatterers[0] if scatterers else 'UNKNOWN'

        # Candidate role inference
        if not is_ms:
            if primary == 'I':
                role = 'SS_I'
            elif primary == 'O':
                role = 'SS_O'
            elif primary == 'S':
                role = 'SS_S'
            elif primary == 'CS':
                role = 'SS_Cs'
            else:
                role = f'SS_{primary}'
        else:
            if 'S' in scatterers and 'O' in scatterers:
                role = 'MS_S_O'
            elif 'I' in scatterers and 'O' in scatterers:
                role = 'MS_O_I'
            elif 'I' in scatterers:
                role = 'MS_I'
            elif 'O' in scatterers:
                role = 'MS_O'
            elif 'S' in scatterers:
                role = 'MS_S'
            else:
                role = 'MS_other'

        return {
            'filename': filename,
            'feffpath': fp_obj,
            'reff': reff,
            'nleg': nleg,
            'scatterers': scatterers,
            'primary': primary,
            'is_ms': is_ms,
            'role': role,
        }


    def _candidate_matches_signature(self, candidate_info, signature, strict_scatterers=True):
        """
        Match FEFF candidates to the current path in a way that supports
        stage-specific replacement.

        For your use case:
        - SS_I replacement should match any candidate with role SS_I
        - SS_O replacement should match any candidate with role SS_O
        """
        if candidate_info.get('role') != signature.get('role'):
            return False

        if candidate_info.get('is_ms', False) != signature.get('is_ms', False):
            return False

        if strict_scatterers and candidate_info.get('scatterers', ()) != signature.get('scatterers', ()):
            return False

        return True


    def _parameter_is_near_bound(self, par, frac=0.02):
        """
        Return True if an lmfit Parameter is close to either bound.
        """
        try:
            if par is None or par.value is None:
                return False
            if par.min is None or par.max is None:
                return False
            span = par.max - par.min
            if span <= 0:
                return False
            return (
                abs(par.value - par.min) <= frac * span
                or abs(par.max - par.value) <= frac * span
            )
        except Exception:
            return False


    def _fit_has_bad_bounds(self, fit_result, delr_names=None, sig2_names=None, n_names=None):
        """
        Lightweight sanity screen for pathological fits.
        """
        if fit_result is None or not hasattr(fit_result, 'params'):
            return True

        params = fit_result.params
        delr_names = delr_names or []
        sig2_names = sig2_names or []
        n_names = n_names or []

        # Check obvious bound-hugging
        for name in delr_names + sig2_names + n_names:
            if name in params and self._parameter_is_near_bound(params[name], frac=0.02):
                return True

        # Check obviously nonphysical coordination collapse
        for name in n_names:
            if name in params:
                try:
                    if params[name].value < 0.75:
                        return True
                except Exception:
                    pass

        return False

    # ------------------------------------------------------------------
    # AICc-based model selection helpers
    # ------------------------------------------------------------------

    # Configurable ΔR sanity thresholds per path type (warn, reject)
    DELR_SANITY_THRESHOLDS = {
        'SS_I':  {'warn': 0.05, 'reject': 0.08},
        'SS_O':  {'warn': 0.08, 'reject': 0.12},
        'SS_S':  {'warn': 0.08, 'reject': 0.12},
        'SS_Cs': {'warn': 0.08, 'reject': 0.12},
        'MS':    {'warn': 0.12, 'reject': 0.15},
    }

    # Default physical-sanity coordination-number floors
    COORDINATION_FLOORS = {
        'N_I': 1.0,
        'N_O': 0.75,
        'N_S': 0.0,
        'N_Cs': 0.0,
    }

    def _get_fit_residual_array(self, fit_result):
        """Return the residual array from a fit result.

        Prefers ``fit_result.residual`` when available; otherwise reconstructs
        the residual from the first dataset's data and model chi(R).

        Returns
        -------
        numpy.ndarray or None
        """
        if fit_result is None:
            return None
        # Direct residual from lmfit / larch feffit
        if hasattr(fit_result, 'residual') and fit_result.residual is not None:
            return np.asarray(fit_result.residual)
        # Reconstruct from dataset
        try:
            dset = fit_result.datasets[0]
            data_re = np.asarray(dset.data.chir_re)
            model_re = np.asarray(dset.model.chir_re)
            data_im = np.asarray(dset.data.chir_im)
            model_im = np.asarray(dset.model.chir_im)
            residual = np.concatenate([data_re - model_re, data_im - model_im])
            return residual
        except Exception:
            return None

    def _count_varying_parameters(self, fit_result_or_params):
        """Count the number of varying (free) parameters.

        Accepts either a fit result object (with ``.params``) or a parameter
        group directly.

        Returns
        -------
        int
        """
        params = getattr(fit_result_or_params, 'params', fit_result_or_params)
        count = 0
        try:
            for pname in params:
                p = params[pname]
                if getattr(p, 'vary', False):
                    count += 1
        except Exception:
            pass
        return count

    def _get_transform_ranges(self, fit_result=None, transform=None, dataset=None):
        """Extract k-range and R-range widths from the transform context.

        Returns
        -------
        dict  with keys 'kmin', 'kmax', 'dk', 'rmin', 'rmax', 'dr'
        """
        trans = transform
        if trans is None and dataset is not None:
            trans = getattr(dataset, 'transform', None)
        if trans is None and fit_result is not None:
            try:
                trans = fit_result.datasets[0].transform
            except Exception:
                pass
        if trans is None:
            return None
        kmin = getattr(trans, 'kmin', 3.0)
        kmax = getattr(trans, 'kmax', 12.0)
        rmin = getattr(trans, 'rmin', 0.9)
        rmax = getattr(trans, 'rmax', 4.0)
        return {
            'kmin': kmin, 'kmax': kmax, 'dk': kmax - kmin,
            'rmin': rmin, 'rmax': rmax, 'dr': rmax - rmin,
        }

    def _estimate_nind(self, fit_result=None, transform=None, dataset=None):
        """Estimate the EXAFS number of independent points.

        Uses  Nind = max(1.0, 2 * Δk * ΔR / π).

        If the fit result already exposes ``epsilon_k`` / ``nvarys`` etc. that
        give a better Nind, prefer those, but the formula above is the reliable
        fallback.

        Returns
        -------
        float
        """
        ranges = self._get_transform_ranges(fit_result, transform, dataset)
        if ranges is None:
            return 1.0
        nind = max(1.0, 2.0 * ranges['dk'] * ranges['dr'] / np.pi)
        return nind

    def _compute_aicc(self, fit_result, transform=None, dataset=None,
                      n_varying=None):
        """Compute AIC and corrected AIC (AICc) for a fit result.

        Parameters
        ----------
        fit_result : object
            Larch feffit result.
        transform : object, optional
            TransformGroup; inferred from fit_result if not given.
        dataset : object, optional
            feffit_dataset; inferred from fit_result if not given.
        n_varying : int, optional
            Number of free parameters.  Auto-counted if not given.

        Returns
        -------
        dict  with keys 'rss', 'nind', 'k', 'aic', 'aicc', 'valid', 'reason'
        """
        result = {
            'rss': np.inf, 'nind': 1.0, 'k': 0,
            'aic': np.inf, 'aicc': np.inf,
            'valid': False, 'reason': '',
        }

        if fit_result is None:
            result['reason'] = 'fit_result is None'
            return result

        # RSS
        residual = self._get_fit_residual_array(fit_result)
        if residual is None or len(residual) == 0:
            result['reason'] = 'residual unavailable'
            return result
        rss = float(np.sum(residual ** 2))
        if not np.isfinite(rss) or rss <= 0:
            result['reason'] = f'RSS invalid ({rss})'
            return result
        result['rss'] = rss

        # Nind
        nind = self._estimate_nind(fit_result, transform, dataset)
        result['nind'] = nind

        # k = number of varying parameters
        k = n_varying if n_varying is not None else self._count_varying_parameters(fit_result)
        result['k'] = k

        if nind <= k + 1:
            result['reason'] = f'Nind ({nind:.1f}) <= k+1 ({k+1})'
            result['aicc'] = np.inf
            return result

        # AIC  = Nind * ln(RSS / Nind) + 2*k
        aic = nind * np.log(rss / nind) + 2.0 * k
        # AICc = AIC + 2*k*(k+1) / (Nind - k - 1)
        aicc = aic + (2.0 * k * (k + 1.0)) / max(1e-12, nind - k - 1.0)

        result['aic'] = aic
        result['aicc'] = aicc
        result['valid'] = True
        result['reason'] = 'OK'
        return result

    def _evaluate_model_selection_metrics(self, old_fit, new_fit,
                                          transform=None, dataset=None,
                                          old_n_varying=None, new_n_varying=None):
        """Compare two models by AICc.

        Returns
        -------
        dict  with keys 'old_aicc', 'new_aicc', 'delta_aicc',
              'old_rss', 'new_rss', 'old_rfactor', 'new_rfactor',
              'old_valid', 'new_valid', 'prefer_new'
        """
        old_m = self._compute_aicc(old_fit, transform, dataset, old_n_varying)
        new_m = self._compute_aicc(new_fit, transform, dataset, new_n_varying)

        old_rf = getattr(old_fit, 'rfactor', np.inf) if old_fit else np.inf
        new_rf = getattr(new_fit, 'rfactor', np.inf) if new_fit else np.inf

        delta = new_m['aicc'] - old_m['aicc']  # negative is better

        return {
            'old_aicc': old_m['aicc'], 'new_aicc': new_m['aicc'],
            'delta_aicc': delta,
            'old_rss': old_m['rss'], 'new_rss': new_m['rss'],
            'old_nind': old_m['nind'], 'new_nind': new_m['nind'],
            'old_k': old_m['k'], 'new_k': new_m['k'],
            'old_rfactor': old_rf, 'new_rfactor': new_rf,
            'old_valid': old_m['valid'], 'new_valid': new_m['valid'],
            'old_reason': old_m['reason'], 'new_reason': new_m['reason'],
            'prefer_new': (new_m['valid'] and old_m['valid'] and delta < 0),
        }

    def _fit_is_physically_sane(self, fit_result, paths=None, verbose=False,
                                coordination_floors=None, delr_thresholds=None):
        """Comprehensive physical sanity check for a fit result.

        Returns
        -------
        (bool, str)   (passes, reason_if_failed)
        """
        if fit_result is None:
            return False, 'fit_result is None'

        if not hasattr(fit_result, 'params'):
            return False, 'fit_result has no params'

        rfactor = getattr(fit_result, 'rfactor', None)
        if rfactor is None or not np.isfinite(rfactor):
            return False, 'R-factor is non-finite'

        params = fit_result.params
        coord_floors = coordination_floors or self.COORDINATION_FLOORS
        delr_thresh = delr_thresholds or self.DELR_SANITY_THRESHOLDS

        # --- Coordination number floors ---
        for pname, floor in coord_floors.items():
            if pname in params:
                val = params[pname].value
                if val < floor:
                    reason = f'{pname}={val:.3f} below floor {floor}'
                    if verbose:
                        print(f'    Sanity FAIL: {reason}')
                    return False, reason

        # --- Negative or near-zero sigma2 ---
        for pname in params:
            if 'sigma2' in pname.lower():
                p = params[pname]
                if p.value < 0:
                    reason = f'{pname}={p.value:.6f} is negative'
                    if verbose:
                        print(f'    Sanity FAIL: {reason}')
                    return False, reason
                if self._parameter_is_near_bound(p, frac=0.01):
                    reason = f'{pname}={p.value:.6f} pinned at bound'
                    if verbose:
                        print(f'    Sanity FAIL: {reason}')
                    return False, reason

        # --- Any parameter pinned at bounds ---
        for pname in params:
            p = params[pname]
            if not getattr(p, 'vary', False):
                continue
            if self._parameter_is_near_bound(p, frac=0.01):
                reason = f'{pname}={p.value:.6f} pinned at bound [{p.min}, {p.max}]'
                if verbose:
                    print(f'    Sanity WARN: {reason}')
                # We only hard-reject for critical parameters; others are warnings
                if any(tag in pname.lower() for tag in ['sigma2', 'n_']):
                    return False, reason

        # --- Non-finite or absurdly large uncertainties ---
        for pname in params:
            p = params[pname]
            if not getattr(p, 'vary', False):
                continue
            stderr = getattr(p, 'stderr', None)
            if stderr is not None:
                if not np.isfinite(stderr):
                    reason = f'{pname} has non-finite uncertainty'
                    if verbose:
                        print(f'    Sanity FAIL: {reason}')
                    return False, reason
                if abs(p.value) > 1e-8 and stderr > 10 * abs(p.value):
                    reason = f'{pname} uncertainty ({stderr:.4f}) >> value ({p.value:.4f})'
                    if verbose:
                        print(f'    Sanity WARN: {reason}')

        # --- ΔR sanity per path ---
        if paths is not None:
            for path in paths:
                role = self._infer_path_role(path)
                delr_val = self._evaluate_path_delr(path, fit_result)
                if delr_val is None:
                    continue
                abs_delr = abs(delr_val)

                # Find matching threshold: try exact role, then prefix (SS/MS)
                thresh = delr_thresh.get(role)
                if thresh is None:
                    prefix = role.split('_')[0] if '_' in role else role
                    if prefix == 'MS' or role.startswith('MS'):
                        thresh = delr_thresh.get('MS')
                if thresh is None:
                    continue

                if abs_delr > thresh['reject']:
                    reason = (f"path {getattr(path, 'label', '?')}: "
                              f"|ΔR|={abs_delr:.4f} > reject threshold {thresh['reject']}")
                    if verbose:
                        print(f'    Sanity FAIL: {reason}')
                    return False, reason
                elif abs_delr > thresh['warn'] and verbose:
                    print(f"    Sanity WARN: path {getattr(path, 'label', '?')}: "
                          f"|ΔR|={abs_delr:.4f} > warn threshold {thresh['warn']}")

        return True, 'OK'

    def _infer_path_role(self, path):
        """
        Return a role label like SS_I, SS_O, SS_S, MS_I, MS_O, MS_S_O, etc.
        Prefer the explicit param_type already assigned to the path.
        """
        ptype = getattr(path, 'param_type', None)
        if ptype:
            return str(ptype)

        info = self.identify_path_elements(path)
        primary = (info.get('primary') or 'UNKNOWN').upper()
        is_ms = info.get('is_ms', False)
        scatterers = tuple(sorted(s.upper() for s in info.get('all_scatterers', [])))

        if not is_ms:
            if primary == 'I':
                return 'SS_I'
            if primary == 'O':
                return 'SS_O'
            if primary == 'S':
                return 'SS_S'
            if primary == 'CS':
                return 'SS_Cs'
            return f'SS_{primary}'

        if 'S' in scatterers and 'O' in scatterers:
            return 'MS_S_O'
        if 'I' in scatterers and 'O' in scatterers:
            return 'MS_O_I'
        if 'I' in scatterers:
            return 'MS_I'
        if 'O' in scatterers:
            return 'MS_O'
        if 'S' in scatterers:
            return 'MS_S'
        return 'MS_other'

    def _candidate_param_names_from_path(self, path):
        """
        Infer likely parameter names tied to this path for sanity checks.
        """
        param_type = getattr(path, 'param_type', None)
        if not param_type:
            label = getattr(path, 'label', '').lower()
            if 'pb-i' in label or '_i' in label:
                param_type = 'I'
            elif 'pb-o' in label or '_o' in label:
                param_type = 'O'
            elif 'pb-s' in label or '_s' in label:
                param_type = 'S'

        if param_type is None:
            return [], [], []

        p = str(param_type).lower()
        delr = [f'delr_{param_type}', f'pb{p}_delr']
        sig2 = [f'sigma2_{param_type}', f'pb{p}_sig2']
        n = [f'N_{param_type}', f'n_Pb{param_type}']

        return delr, sig2, n


    def replace_large_delr_paths(
        self,
        iteration,
        feff_base_dir,
        delr_threshold=0.06,
        improvement_tolerance=0.0005,
        target_param_types=None,
        strict_scatterers=False,
        max_candidates_per_path=5,
        max_total_replacements_per_group=None,
        require_delr_improvement=True,
        verbose=True,
        plot_results=True,
        aicc_improvement_threshold=2.0,
    ):
        """
        Replace fitted paths whose |ΔR| is too large with same-kind FEFF paths.

        Key improvement:
        this function can now target only selected path roles, e.g.
        target_param_types=['SS_I']  -> replace only Pb-I paths
        target_param_types=['SS_O']  -> replace only Pb-O paths

        Parameters
        ----------
        iteration : int
            Which stored fit iteration to inspect and refit.
        feff_base_dir : str or Path
            Base FEFF directory.
        delr_threshold : float
            Only paths with |ΔR| > threshold are considered.
        improvement_tolerance : float
            Minimum R-factor decrease required to accept.
        target_param_types : list[str] or None
            Restrict replacement to these path roles. Examples:
            ['SS_I'], ['SS_O'], ['SS_I','SS_O'].
            If None, all path types are eligible.
        strict_scatterers : bool
            If True, require exact scatterer tuple match.
            For SS_I / SS_O this can stay False.
        """
        import os
        import copy
        from pathlib import Path
        from larch.xafs import feffpath

        feff_base_dir = Path(feff_base_dir)
        results = {}
        target_param_types = set(target_param_types or [])

        # --------------------------------------------------
        # Pre-scan FEFF candidates once
        # --------------------------------------------------
        candidate_paths = []
        for root, _, files in os.walk(feff_base_dir):
            for file in files:
                if not (file.lower().endswith('.dat') and 'feff' in file.lower()):
                    continue

                full_path = os.path.join(root, file)
                try:
                    fp_obj = feffpath(filename=full_path)

                    if hasattr(fp_obj, '_feffdat') and hasattr(fp_obj._feffdat, 'absorber'):
                        if str(fp_obj._feffdat.absorber).upper() != 'PB':
                            continue

                    cand = self._extract_candidate_signature(fp_obj, filename=full_path)
                    candidate_paths.append(cand)

                except Exception:
                    continue

        if verbose:
            print(f"\n{'='*70}")
            print(f"PATH REPLACEMENT FIT - Iteration {iteration}")
            print(f"ΔR threshold = {delr_threshold:.3f} Å")
            print(f"Candidate FEFF paths scanned: {len(candidate_paths)}")
            if target_param_types:
                print(f"Targeted path roles: {sorted(target_param_types)}")
            else:
                print("Targeted path roles: ALL")
            print(f"{'='*70}")

        # --------------------------------------------------
        # Loop over groups
        # --------------------------------------------------
        for proj_name, project in self.projects.items():
            for group_name, group in project.groups.items():
                group_id = f"{proj_name}.{group_name}"
                prev_fit = getattr(group, f'fit_result_iter{iteration}', None)
                prev_paths = getattr(group, f'paths_iter{iteration}', None)

                if prev_fit is None or not prev_paths:
                    continue

                if verbose:
                    print(f"\nGroup: {group_id}")
                    print(f"  Starting R-factor: {prev_fit.rfactor:.6f}")

                flagged = []
                for idx, old_path in enumerate(prev_paths):
                    role = self._infer_path_role(old_path)

                    # stage targeting
                    if target_param_types and role not in target_param_types:
                        continue

                    delr_val = self._evaluate_path_delr(old_path, prev_fit)
                    if delr_val is None or abs(delr_val) <= delr_threshold:
                        continue

                    flagged.append((idx, old_path, role, delr_val))

                if not flagged:
                    results[group_id] = {
                        'replaced': False,
                        'rfactor_before': prev_fit.rfactor,
                        'rfactor_after': prev_fit.rfactor,
                        'n_replaced': 0,
                        'status': 'No targeted large-ΔR paths',
                    }
                    continue

                flagged.sort(key=lambda x: abs(x[3]), reverse=True)
                if max_total_replacements_per_group is not None:
                    flagged = flagged[:max_total_replacements_per_group]

                best_group_fit = prev_fit
                best_group_paths = list(prev_paths)
                accepted_replacements = []

                for idx, old_path, role, delr_val in flagged:
                    old_reff = getattr(old_path, 'reff', None)
                    if old_reff is None:
                        continue

                    sig = self._path_signature(old_path)
                    target_reff = old_reff + delr_val

                    same_kind = [
                        c for c in candidate_paths
                        if self._candidate_matches_signature(c, sig, strict_scatterers=strict_scatterers)
                        and c.get('reff') is not None
                    ]

                    if not same_kind:
                        if verbose:
                            print(f"  No same-kind candidates for {getattr(old_path, 'label', 'path')}.")
                        continue

                    # keep only those closer to the fitted target than the current path is
                    current_distance = abs(old_reff - target_reff)
                    same_kind.sort(key=lambda c: abs(c['reff'] - target_reff))
                    nearby = [c for c in same_kind if abs(c['reff'] - target_reff) < current_distance - 1e-6]
                    nearby = nearby[:max_candidates_per_path]

                    if not nearby:
                        if verbose:
                            print(
                                f"  {getattr(old_path, 'label', 'path')[:40]:40s} "
                                f"flagged (ΔR={delr_val:+.3f}) but no better r_eff candidate found."
                            )
                        continue

                    # Compute AICc for the current best group fit before replacement
                    pre_repl_aicc = self._compute_aicc(best_group_fit)

                    local_best_fit = None
                    local_best_paths = None
                    local_best_meta = None

                    for cand in nearby:
                        try:
                            trial_paths = copy.deepcopy(best_group_paths)
                            replacement = feffpath(cand['filename'])
                            replacement.label = getattr(old_path, 'label', Path(cand['filename']).stem)
                            replacement = self.assign_linked_params_to_path(replacement, iteration)
                            trial_paths[idx] = replacement

                            refit_result, success, _ = self.fit_with_linked_params(
                                group, trial_paths, iteration, prev_result=best_group_fit, verbose=False
                            )

                            if not success or refit_result is None:
                                continue

                            new_delr = self._evaluate_path_delr(trial_paths[idx], refit_result)
                            if new_delr is None:
                                continue

                            improved_delr = abs(new_delr) < abs(delr_val) - 1e-4
                            if require_delr_improvement and not improved_delr:
                                continue

                            # Physical sanity check
                            sane, sane_reason = self._fit_is_physically_sane(
                                refit_result, paths=trial_paths, verbose=False)
                            if not sane:
                                if verbose:
                                    print(f"    Candidate reff={cand['reff']:.3f} sanity fail: {sane_reason}")
                                continue

                            # AICc comparison
                            cand_aicc = self._compute_aicc(refit_result)
                            if not cand_aicc['valid']:
                                continue

                            # Require AICc improvement AND (optionally) R-factor improvement
                            aicc_improved = (pre_repl_aicc['valid'] and
                                             (pre_repl_aicc['aicc'] - cand_aicc['aicc'])
                                             >= aicc_improvement_threshold)
                            rfactor_improved = (best_group_fit.rfactor - refit_result.rfactor) >= improvement_tolerance

                            if not (aicc_improved or rfactor_improved):
                                continue

                            # Score: prefer lower AICc, then smaller |ΔR|
                            score = (cand_aicc['aicc'], abs(new_delr), abs(cand['reff'] - target_reff))

                            if (local_best_fit is None) or (score < local_best_meta['score']):
                                local_best_fit = refit_result
                                local_best_paths = trial_paths
                                local_best_meta = {
                                    'score': score,
                                    'index': idx,
                                    'role': role,
                                    'old_label': getattr(old_path, 'label', 'path'),
                                    'old_reff': old_reff,
                                    'old_delr': delr_val,
                                    'target_reff': target_reff,
                                    'new_reff': cand['reff'],
                                    'new_delr': new_delr,
                                    'rfactor_before': best_group_fit.rfactor,
                                    'rfactor_after': refit_result.rfactor,
                                    'aicc_before': pre_repl_aicc['aicc'],
                                    'aicc_after': cand_aicc['aicc'],
                                    'delta_aicc': cand_aicc['aicc'] - pre_repl_aicc['aicc'],
                                    'file': cand['filename'],
                                }

                        except Exception:
                            continue

                    if local_best_meta is not None:
                        accepted_replacements.append(local_best_meta)
                        best_group_fit = local_best_fit
                        best_group_paths = local_best_paths

                        if verbose:
                            print(
                                f"  Accepted {local_best_meta['role']:>5s} replacement: "
                                f"{local_best_meta['old_reff']:.3f} + ({local_best_meta['old_delr']:+.3f}) "
                                f"-> candidate reff {local_best_meta['new_reff']:.3f}, "
                                f"new ΔR={local_best_meta['new_delr']:+.3f}, "
                                f"R {local_best_meta['rfactor_before']:.6f} -> {local_best_meta['rfactor_after']:.6f}, "
                                f"AICc {local_best_meta['aicc_before']:.2f} -> {local_best_meta['aicc_after']:.2f} "
                                f"(ΔAICc={local_best_meta['delta_aicc']:+.2f})"
                            )
                    else:
                        if verbose:
                            print(f"  No acceptable replacement found for {getattr(old_path, 'label', 'path')}")

                if accepted_replacements:
                    # Final AICc gate: verify the overall replacement is better
                    final_aicc = self._compute_aicc(best_group_fit)
                    orig_aicc = self._compute_aicc(prev_fit)
                    final_sane, final_reason = self._fit_is_physically_sane(
                        best_group_fit, paths=best_group_paths, verbose=verbose)

                    if not final_sane:
                        if verbose:
                            print(f"  REJECTED all replacements for {group_id}: "
                                  f"final sanity failed ({final_reason})")
                        results[group_id] = {
                            'replaced': False,
                            'rfactor_before': prev_fit.rfactor,
                            'rfactor_after': prev_fit.rfactor,
                            'n_replaced': 0,
                            'status': f'Sanity fail: {final_reason}',
                        }
                        continue

                    setattr(group, f'fit_result_iter{iteration}', best_group_fit)
                    setattr(group, f'paths_iter{iteration}', best_group_paths)

                    results[group_id] = {
                        'replaced': True,
                        'rfactor_before': prev_fit.rfactor,
                        'rfactor_after': best_group_fit.rfactor,
                        'aicc_before': orig_aicc.get('aicc', float('nan')),
                        'aicc_after': final_aicc.get('aicc', float('nan')),
                        'delta_aicc': final_aicc.get('aicc', float('nan')) - orig_aicc.get('aicc', float('nan')),
                        'n_replaced': len(accepted_replacements),
                        'replacements': accepted_replacements,
                        'status': 'Accepted',
                    }

                    if plot_results:
                        self._plot_fit_comparison(
                            group, group_id, iteration,
                            prev_fit, best_group_fit,
                            prev_paths, best_group_paths,
                            'replacement'
                        )
                else:
                    results[group_id] = {
                        'replaced': False,
                        'rfactor_before': prev_fit.rfactor,
                        'rfactor_after': prev_fit.rfactor,
                        'n_replaced': 0,
                        'status': 'Rejected/None',
                    }

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------
        print("\n" + "=" * 80)
        print(f"REPLACE_LARGE_DELR_PATHS SUMMARY (Iteration {iteration})")
        print(f"  Model selection: AICc (threshold={aicc_improvement_threshold})")
        print("=" * 80)
        print(f"  ΔR threshold:          {delr_threshold:.3f} Å")
        print(f"  Improvement tolerance: {improvement_tolerance:.6f}")
        print(f"  Candidate paths:       {len(candidate_paths)}")
        if target_param_types:
            print(f"  Target path roles:     {sorted(target_param_types)}")
        total_replaced = sum(1 for r in results.values() if r.get('replaced'))
        print(f"  Groups with replacements accepted: {total_replaced}/{len(results)}")
        print("-" * 80)
        print("  Results per group:")
        print(f"  {'Group':<40s}  {'R_bef':>10s}  {'R_aft':>10s}  "
              f"{'AICc_bef':>10s}  {'AICc_aft':>10s}  {'ΔAICc':>8s}  {'#r':>3s}  {'Status':>15s}")
        print("-" * 80)

        for group_id, res in results.items():
            aicc_b = res.get('aicc_before', float('nan'))
            aicc_a = res.get('aicc_after', float('nan'))
            d_aicc = res.get('delta_aicc', float('nan'))
            print(
                f"    {group_id:<35s}  "
                f"{res.get('rfactor_before', float('nan')):>10.6f}  "
                f"{res.get('rfactor_after', float('nan')):>10.6f}  "
                f"{aicc_b:>10.2f}  {aicc_a:>10.2f}  {d_aicc:>+8.2f}  "
                f"{res.get('n_replaced', 0):>3d}  "
                f"{res.get('status', 'Unknown'):>15s}"
            )

        print("=" * 80 + "\n")
        return results

    def final_shared_e0_fit(self, iteration, verbose=True, plot_results=False):
        """
        Refit every group for a given iteration using one fixed shared e0.

        The shared e0 is taken as the arithmetic mean of the fitted e0 values from
        ``fit_result_iter{iteration}`` across all groups that already have a fit.
        The refit keeps the same path list for each group but fixes e0 to that mean.
        """
        import copy
        from larch.xafs import feffit_dataset
        from larch import Group

        e0_values = []
        fit_groups = []
        for proj_name, project in self.projects.items():
            for group_name, group in project.groups.items():
                fit_result = getattr(group, f'fit_result_iter{iteration}', None)
                paths = getattr(group, f'paths_iter{iteration}', None)
                if fit_result is None or not paths:
                    continue
                if hasattr(fit_result, 'params') and 'e0' in fit_result.params:
                    e0_values.append(fit_result.params['e0'].value)
                    fit_groups.append((proj_name, group_name, group, fit_result, paths))

        if not e0_values:
            raise ValueError(f"No fitted e0 values found for iteration {iteration}.")

        shared_e0 = float(np.mean(e0_values))
        results = {'shared_e0': shared_e0, 'groups': {}}

        if verbose:
            print(f"\n{'='*70}")
            print(f"FINAL SHARED e0 FIT - Iteration {iteration}")
            print(f"Shared e0 fixed to mean value: {shared_e0:.6f} eV")
            print(f"Number of groups contributing: {len(e0_values)}")
            print(f"{'='*70}")

        for proj_name, group_name, group, prev_fit, paths in fit_groups:
            group_id = f"{proj_name}.{group_name}"
            available_elements = set()
            for path in paths:
                info = self.identify_path_elements(path)
                if info['primary']:
                    available_elements.add(info['primary'])
                available_elements.update(info['all_scatterers'])
            available_elements = list(available_elements)

            params = self.build_linked_params(
                iteration=iteration,
                prev_result=prev_fit,
                s02_value=0.78,
                fix_s02=True,
                fix_e0=True,
                available_elements=available_elements,
            )
            params.e0.value = shared_e0

            configured_paths = []
            for path in copy.deepcopy(paths):
                configured_paths.append(self.assign_linked_params_to_path(path, iteration))

            # --- FIT RANGES DEFINED HERE ---
            trans = self.get_transform_for_group(group)
            temp_data = Group(k=group.k, chi=group.chi)
            dataset = feffit_dataset(data=temp_data, pathlist=configured_paths, transform=trans)
            final_fit, success, _ = self.robust_feffit(params, [dataset], verbose=False)

            # Compute AICc before and after
            prev_aicc_info = self._compute_aicc(prev_fit)
            prev_aicc = prev_aicc_info.get('aicc', float('nan'))
            if success and final_fit is not None:
                final_aicc_info = self._compute_aicc(final_fit)
                final_aicc = final_aicc_info.get('aicc', float('nan'))
                delta_aicc = final_aicc - prev_aicc
                setattr(group, f'fit_result_iter{iteration}_shared_e0', final_fit)
                setattr(group, f'paths_iter{iteration}_shared_e0', configured_paths)
                setattr(group, f'shared_e0_iter{iteration}', shared_e0)
                results['groups'][group_id] = {
                    'success': True,
                    'rfactor_before': prev_fit.rfactor,
                    'rfactor_after': final_fit.rfactor,
                    'aicc_before': prev_aicc,
                    'aicc_after': final_aicc,
                    'delta_aicc': delta_aicc,
                }
                if verbose:
                    print(f"{group_id:45s} R: {prev_fit.rfactor:.6f} -> {final_fit.rfactor:.6f} | AICc: {prev_aicc:.2f} -> {final_aicc:.2f} (ΔAICc: {delta_aicc:+.2f})")
                if plot_results:
                    self._plot_fit_comparison(group, group_id, iteration, prev_fit, final_fit,
                                              paths, configured_paths, 'shared e0')
            else:
                results['groups'][group_id] = {
                    'success': False,
                    'rfactor_before': prev_fit.rfactor,
                    'rfactor_after': None,
                    'aicc_before': prev_aicc,
                    'aicc_after': None,
                    'delta_aicc': None,
                }
                if verbose:
                    print(f"{group_id:45s} fit failed with shared e0")

        # ===== SUMMARY: final_shared_e0_fit =====
        print("\n" + "="*80)
        print(f"FINAL_SHARED_E0_FIT SUMMARY (Iteration {iteration})")
        print("="*80)
        print(f"  Shared e0 value:  {results.get('shared_e0', float('nan')):.3f} eV")
        successful = sum(1 for g in results.get('groups', {}).values() if g.get('success'))
        total = len(results.get('groups', {}))
        print(f"  Successful fits:  {successful}/{total}")
        print("-"*80)
        print("  Results per group:")
        print(f"  {'Group':<50s}  {'R_before':>10s}  {'R_after':>10s}  {'ΔR':>10s}  {'AICc_before':>12s}  {'AICc_after':>12s}  {'ΔAICc':>10s}  {'Status':>8s}")
        print("-"*120)
        for group_id, res in results.get('groups', {}).items():
            r_before = res.get('rfactor_before', float('nan'))
            r_after = res.get('rfactor_after')
            aicc_before = res.get('aicc_before', float('nan'))
            aicc_after = res.get('aicc_after', float('nan'))
            delta_aicc = res.get('delta_aicc', float('nan'))
            if r_after is not None:
                delta = r_after - r_before
                status = "OK"
            else:
                r_after = float('nan')
                delta = float('nan')
                aicc_after = float('nan')
                delta_aicc = float('nan')
                status = "FAILED"
            print(f"    {group_id:<45s}  {r_before:>10.6f}  {r_after:>10.6f}  {delta:>+10.6f}  {aicc_before:>12.2f}  {aicc_after:>12.2f}  {delta_aicc:>+10.2f}  {status:>8s}")
        print("="*80 + "\n")

        return results
    def _plot_fit_comparison(self, group, group_id, iteration, prev_fit, new_fit,
                             prev_paths, new_paths, feff_category):
        """
        Plot comparison between previous and current iteration fits.
        
        Shows k-space and R-space comparisons with improvement metrics.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Get data arrays
        k = group.k
        chi = group.chi
        kweight = 3
        
        # --- k-space plots (top row) ---
        ax_k_prev = axes[0, 0]
        ax_k_new = axes[0, 1]
        
        # Previous iteration k-space
        if hasattr(prev_fit, 'datasets') and len(prev_fit.datasets) > 0:
            ds_prev = prev_fit.datasets[0]
            if hasattr(ds_prev.data, 'k') and hasattr(ds_prev.model, 'chi'):
                k_prev = ds_prev.data.k
                chi_data_prev = ds_prev.data.chi * k_prev**kweight
                chi_model_prev = ds_prev.model.chi * k_prev**kweight
                ax_k_prev.plot(k_prev, chi_data_prev, 'b-', lw=1.5, label='Data')
                ax_k_prev.plot(k_prev, chi_model_prev, 'r--', lw=1.5, label='Fit')
                ax_k_prev.plot(k_prev, chi_data_prev - chi_model_prev - 0.5*np.max(np.abs(chi_data_prev)), 
                              'g-', lw=1, alpha=0.7, label='Residual (offset)')
        
        ax_k_prev.set_xlabel(r'$k$ (Å$^{-1}$)')
        ax_k_prev.set_ylabel(r'$k^3\chi(k)$ (Å$^{-3}$)')
        ax_k_prev.set_title(f'Iteration {iteration-1}: R = {prev_fit.rfactor:.6f}')
        ax_k_prev.legend(loc='upper right', fontsize=9)
        ax_k_prev.grid(True, alpha=0.3)
        
        # New iteration k-space
        if hasattr(new_fit, 'datasets') and len(new_fit.datasets) > 0:
            ds_new = new_fit.datasets[0]
            if hasattr(ds_new.data, 'k') and hasattr(ds_new.model, 'chi'):
                k_new = ds_new.data.k
                chi_data_new = ds_new.data.chi * k_new**kweight
                chi_model_new = ds_new.model.chi * k_new**kweight
                ax_k_new.plot(k_new, chi_data_new, 'b-', lw=1.5, label='Data')
                ax_k_new.plot(k_new, chi_model_new, 'r--', lw=1.5, label='Fit')
                ax_k_new.plot(k_new, chi_data_new - chi_model_new - 0.5*np.max(np.abs(chi_data_new)), 
                             'g-', lw=1, alpha=0.7, label='Residual (offset)')
        
        ax_k_new.set_xlabel(r'$k$ (Å$^{-1}$)')
        ax_k_new.set_ylabel(r'$k^3\chi(k)$ (Å$^{-3}$)')
        ax_k_new.set_title(f'Iteration {iteration} (+{feff_category}): R = {new_fit.rfactor:.6f}')
        ax_k_new.legend(loc='upper right', fontsize=9)
        ax_k_new.grid(True, alpha=0.3)
        
        # --- R-space plots (bottom row) ---
        ax_r_prev = axes[1, 0]
        ax_r_new = axes[1, 1]
        
        # Previous iteration R-space
        if hasattr(prev_fit, 'datasets') and len(prev_fit.datasets) > 0:
            ds_prev = prev_fit.datasets[0]
            if hasattr(ds_prev.data, 'r') and hasattr(ds_prev.data, 'chir_mag'):
                r_prev = ds_prev.data.r
                chir_data_prev = ds_prev.data.chir_mag
                chir_model_prev = ds_prev.model.chir_mag
                ax_r_prev.plot(r_prev, chir_data_prev, 'b-', lw=1.5, label='Data')
                ax_r_prev.plot(r_prev, chir_model_prev, 'r--', lw=1.5, label='Fit')
                ax_r_prev.fill_between(r_prev, chir_data_prev, chir_model_prev, 
                                       color='green', alpha=0.2, label='Difference')
        
        ax_r_prev.set_xlabel(r'$R$ (Å)')
        ax_r_prev.set_ylabel(r'$|\chi(R)|$ (Å$^{-4}$)')
        ax_r_prev.set_title(f'Iteration {iteration-1}: {len(prev_paths)} paths')
        ax_r_prev.legend(loc='upper right', fontsize=9)
        ax_r_prev.set_xlim(0, 6)
        ax_r_prev.grid(True, alpha=0.3)
        
        # New iteration R-space
        if hasattr(new_fit, 'datasets') and len(new_fit.datasets) > 0:
            ds_new = new_fit.datasets[0]
            if hasattr(ds_new.data, 'r') and hasattr(ds_new.data, 'chir_mag'):
                r_new = ds_new.data.r
                chir_data_new = ds_new.data.chir_mag
                chir_model_new = ds_new.model.chir_mag
                ax_r_new.plot(r_new, chir_data_new, 'b-', lw=1.5, label='Data')
                ax_r_new.plot(r_new, chir_model_new, 'r--', lw=1.5, label='Fit')
                ax_r_new.fill_between(r_new, chir_data_new, chir_model_new, 
                                      color='green', alpha=0.2, label='Difference')
        
        ax_r_new.set_xlabel(r'$R$ (Å)')
        ax_r_new.set_ylabel(r'$|\chi(R)|$ (Å$^{-4}$)')
        ax_r_new.set_title(f'Iteration {iteration}: {len(new_paths)} paths')
        ax_r_new.legend(loc='upper right', fontsize=9)
        ax_r_new.set_xlim(0, 6)
        ax_r_new.grid(True, alpha=0.3)
        
        # Improvement annotation
        improvement = prev_fit.rfactor - new_fit.rfactor
        pct_improvement = (improvement / prev_fit.rfactor) * 100 if prev_fit.rfactor > 0 else 0
        
        color = 'green' if improvement > 0 else 'red'
        sign = '+' if improvement > 0 else ''
        
        fig.suptitle(f'{group_id}\nR-factor: {prev_fit.rfactor:.6f} → {new_fit.rfactor:.6f} '
                    f'({sign}{pct_improvement:.1f}%)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

    def plot_publication_figure(self, figsize=(16, 12), dpi=300, save_path=None,
                                 include_paths_panel=True, style='seaborn-v0_8-whitegrid'):
        """
        Create a publication-quality multi-panel figure of EXAFS fit results.
        
        Pulls parameters from the best/most recent iterative fit for each group.
        
        Panels:
            1. Coordination Number vs Concentration (dual-y-axis scatter with errors)
               - Pb-I in purple, Pb-O in red
            2. Representative k-space fit (data + model + residual)
            3. Representative R-space fit (magnitude with path contributions)
            4. R-factor vs Concentration (fit quality assessment)
            5. Parameter correlation / Debye-Waller factors (optional)
            
        Parameters:
            figsize (tuple): Figure size in inches (width, height)
            dpi (int): Resolution for saved figure
            save_path (str): Path to save figure (None = display only)
            include_paths_panel (bool): Include individual path contributions
            style (str): Matplotlib style to use
            
        Returns:
            fig, axes: Matplotlib figure and axes objects
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import MaxNLocator
        import matplotlib.patches as mpatches
        
        # Try to apply style, fall back gracefully
        try:
            plt.style.use(style)
        except:
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                pass  # Use default style
        
        # --- Collect data from all groups ---
        data_collection = self._collect_fit_data_for_publication()
        
        if not data_collection['groups']:
            print("No fit data found. Run iterative_fit_linked_all() first.")
            return None, None
        
        # --- Create figure with GridSpec for flexible layout ---
        fig = plt.figure(figsize=figsize, dpi=100)
        
        if include_paths_panel:
            gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], 
                                  hspace=0.35, wspace=0.25)
            ax_coord = fig.add_subplot(gs[0, 0])
            ax_rfactor = fig.add_subplot(gs[0, 1])
            ax_kspace = fig.add_subplot(gs[1, :])
            ax_rspace = fig.add_subplot(gs[2, 0])
            ax_paths = fig.add_subplot(gs[2, 1])
        else:
            gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], 
                                  hspace=0.35, wspace=0.25)
            ax_coord = fig.add_subplot(gs[0, 0])
            ax_rfactor = fig.add_subplot(gs[0, 1])
            ax_kspace = fig.add_subplot(gs[1, 0])
            ax_rspace = fig.add_subplot(gs[1, 1])
            ax_paths = None
        
        # === PANEL 1: Coordination number vs concentration (dual-axis) ===
        self._plot_coordination_vs_concentration(ax_coord, data_collection)
        
        # === PANEL 2: Bond length vs concentration ===
        self._plot_bondlength_vs_concentration(ax_rfactor, data_collection)
        
        # === PANEL 3: Representative k-space fit ===
        self._plot_representative_kspace(ax_kspace, data_collection)
        
        # === PANEL 4: Representative R-space fit ===
        self._plot_representative_rspace(ax_rspace, data_collection)
        
        # === PANEL 5: Path contributions (if included) ===
        if ax_paths is not None:
            self._plot_path_contributions(ax_paths, data_collection)
        
        # --- Final adjustments ---
        fig.suptitle('EXAFS Fit Analysis Summary', fontsize=14, fontweight='bold', y=0.98)
        
        # Add metadata annotation
        n_groups = len(data_collection['groups'])
        max_iter = max([d['iteration'] for d in data_collection['groups']])
        avg_rfactor = np.mean([d['rfactor'] for d in data_collection['groups']])
        
        metadata_text = (f"N = {n_groups} samples | "
                        f"Max iteration: {max_iter} | "
                        f"Mean R-factor: {avg_rfactor:.4f}")
        fig.text(0.5, 0.01, metadata_text, ha='center', fontsize=9, style='italic',
                color='gray')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Figure saved to: {save_path}")
        
        plt.show()

        # ===== SUMMARY: plot_publication_figure =====
        print("\n" + "="*80)
        print("PLOT_PUBLICATION_FIGURE SUMMARY")
        print("="*80)
        print(f"  Groups plotted:    {n_groups}")
        print(f"  Max iteration:     {max_iter}")
        print(f"  Mean R-factor:     {avg_rfactor:.6f}")
        print(f"  Figure saved:      {save_path if save_path else 'Not saved'}")
        print("-"*80)
        print("  Data summary per group:")
        for g in data_collection['groups']:
            print(f"    {g.get('group_id', 'N/A'):<45s}  "
                  f"conc: {g.get('concentration', float('nan')):.3f} M, "
                  f"R: {g.get('rfactor', float('nan')):.6f}, "
                  f"N_I: {g.get('N_I', float('nan')):.2f}, "
                  f"N_O: {g.get('N_O', float('nan')):.2f}")
        print("="*80 + "\n")

        return fig, (ax_coord, ax_rfactor, ax_kspace, ax_rspace, ax_paths)

    def _collect_fit_data_for_publication(self):
        """
        Collect fit parameters and data from all groups for publication plotting.
        
        Returns:
            dict: Collected data including concentrations, coordination numbers,
                  errors, R-factors, reff values from paths, and representative fit data
        """
        import numpy as np
        
        def _extract_reff_from_paths(paths):
            """
            Extract reff values for Pb-I and Pb-O paths.
            Returns (reff_I, reff_O) - the reff of the first matching path of each type.
            """
            reff_I = None
            reff_O = None
            
            for path in paths:
                # Get path label or scattering type
                label = getattr(path, 'label', '')
                scat_type = getattr(path, 'scat_type', '')
                geom = getattr(path, 'geom', None)
                
                # Try to get reff from the path
                reff = getattr(path, 'reff', None)
                if reff is None and hasattr(path, '_feffdat'):
                    reff = getattr(path._feffdat, 'reff', None)
                
                if reff is None:
                    continue
                
                # Determine path type from label, scat_type, or geometry
                is_pbi = False
                is_pbo = False
                
                # Check label first
                if label:
                    label_lower = label.lower()
                    if 'pb-i' in label_lower or 'pbi' in label_lower or 'pb_i' in label_lower:
                        is_pbi = True
                    elif 'pb-o' in label_lower or 'pbo' in label_lower or 'pb_o' in label_lower:
                        is_pbo = True
                
                # Check scat_type
                if not is_pbi and not is_pbo and scat_type:
                    scat_lower = scat_type.lower()
                    if 'pbi' in scat_lower or 'pb-i' in scat_lower:
                        is_pbi = True
                    elif 'pbo' in scat_lower or 'pb-o' in scat_lower:
                        is_pbo = True
                
                # Check geometry for scatterer element
                if not is_pbi and not is_pbo and geom:
                    for atom in geom:
                        elem = atom[0] if isinstance(atom, (list, tuple)) else ''
                        if elem == 'I' and reff_I is None:
                            is_pbi = True
                            break
                        elif elem == 'O' and reff_O is None:
                            is_pbo = True
                            break
                
                # Store the first reff found for each type
                if is_pbi and reff_I is None:
                    reff_I = reff
                elif is_pbo and reff_O is None:
                    reff_O = reff
                
                # Stop if we have both
                if reff_I is not None and reff_O is not None:
                    break
            
            return reff_I, reff_O
        
        data = {
            'groups': [],
            'concentrations': [],
            'N_I': [], 'N_I_err': [],
            'N_O': [], 'N_O_err': [],
            'sigma2_I': [], 'sigma2_I_err': [],
            'sigma2_O': [], 'sigma2_O_err': [],
            'delr_I': [], 'delr_I_err': [],
            'delr_O': [], 'delr_O_err': [],
            'reff_I': [], 'reff_O': [],  # Actual reff from FEFF paths
            'rfactors': [],
            'representative_group': None,
            'representative_fit': None,
            'representative_paths': []
        }
        
        best_rfactor = float('inf')
        
        for proj_name, project in self.projects.items():
            for group_name, group in project.groups.items():
                group_id = f"{proj_name}.{group_name}"
                
                # Find the best (most recent) fit result
                best_iter = None
                best_fit = None
                for i in range(10, 0, -1):  # Check iterations 10 down to 1
                    fit_attr = f"fit_result_iter{i}"
                    if hasattr(group, fit_attr):
                        best_fit = getattr(group, fit_attr)
                        best_iter = i
                        break
                
                if best_fit is None:
                    continue
                
                # Parse concentration from project/group name (project_name evaluated first)
                _, concentration = self._parse_formula_from_group_name(group_name, proj_name)
                
                # Extract parameters with errors
                params = best_fit.params if hasattr(best_fit, 'params') else {}
                
                # Map parameter names (handle both old and new naming conventions)
                param_values = self._extract_params_with_errors(params)
                
                if param_values is None:
                    continue
                
                # Get paths
                paths_attr = f"paths_iter{best_iter}"
                paths = getattr(group, paths_attr, [])
                
                # Extract actual reff values from the FEFF paths
                reff_I, reff_O = _extract_reff_from_paths(paths)
                # Use defaults if not found (fallback to common values)
                if reff_I is None:
                    reff_I = 3.2  # Default Pb-I distance
                    print(f"  Warning: No Pb-I path found for {group_id}, using default reff_I=3.2")
                if reff_O is None:
                    reff_O = 2.4  # Default Pb-O distance
                    print(f"  Warning: No Pb-O path found for {group_id}, using default reff_O=2.4")
                
                # Store data
                group_data = {
                    'group_id': group_id,
                    'group_name': group_name,
                    'group': group,
                    'concentration': concentration,
                    'iteration': best_iter,
                    'fit_result': best_fit,
                    'paths': paths,
                    'rfactor': best_fit.rfactor,
                    'reff_I': reff_I,
                    'reff_O': reff_O,
                    **param_values
                }
                
                data['groups'].append(group_data)
                data['concentrations'].append(concentration)
                data['N_I'].append(param_values['N_I'])
                data['N_I_err'].append(param_values['N_I_err'])
                data['N_O'].append(param_values['N_O'])
                data['N_O_err'].append(param_values['N_O_err'])
                data['sigma2_I'].append(param_values['sigma2_I'])
                data['sigma2_I_err'].append(param_values.get('sigma2_I_err', 0))
                data['sigma2_O'].append(param_values['sigma2_O'])
                data['sigma2_O_err'].append(param_values.get('sigma2_O_err', 0))
                data['delr_I'].append(param_values.get('delr_I', 0))
                data['delr_I_err'].append(param_values.get('delr_I_err', 0))
                data['delr_O'].append(param_values.get('delr_O', 0))
                data['delr_O_err'].append(param_values.get('delr_O_err', 0))
                data['reff_I'].append(reff_I)
                data['reff_O'].append(reff_O)
                data['rfactors'].append(best_fit.rfactor)
                
                # Track best fit for representative plots
                if best_fit.rfactor < best_rfactor:
                    best_rfactor = best_fit.rfactor
                    data['representative_group'] = group
                    data['representative_fit'] = best_fit
                    data['representative_paths'] = paths
                    data['representative_group_id'] = group_id
        
        # Convert to numpy arrays
        for key in ['concentrations', 'N_I', 'N_I_err', 'N_O', 'N_O_err', 
                   'sigma2_I', 'sigma2_I_err', 'sigma2_O', 'sigma2_O_err',
                   'delr_I', 'delr_I_err', 'delr_O', 'delr_O_err', 
                   'reff_I', 'reff_O', 'rfactors']:
            data[key] = np.array(data[key])
        
        return data

    def _extract_params_with_errors(self, params):
        """
        Extract parameter values and errors from fit result params.
        Handles both old naming convention (n_PbI, pbi_sig2) and new (N_I, sigma2_I).
        """
        if not params:
            return None
        
        # Parameter name mappings (old_name: new_name)
        mappings = {
            'N_I': ['N_I', 'n_PbI', 'n_I', 'N_PbI'],
            'N_O': ['N_O', 'n_PbO', 'n_O', 'N_PbO'],
            'sigma2_I': ['sigma2_I', 'pbi_sig2', 'sig2_I', 'sigma_I'],
            'sigma2_O': ['sigma2_O', 'pbo_sig2', 'sig2_O', 'sigma_O'],
            'delr_I': ['delr_I', 'pbi_delr', 'dr_I', 'deltar_I'],
            'delr_O': ['delr_O', 'pbo_delr', 'dr_O', 'deltar_O'],
            'e0': ['e0', 'del_e0', 'E0', 'dE0'],
            's02': ['s02', 'amp', 'S02', 'amplitude']
        }
        
        result = {}
        
        for standard_name, aliases in mappings.items():
            found = False
            for alias in aliases:
                if alias in params:
                    p = params[alias]
                    result[standard_name] = p.value
                    # Get error (stderr if available)
                    if hasattr(p, 'stderr') and p.stderr is not None:
                        result[f'{standard_name}_err'] = p.stderr
                    else:
                        result[f'{standard_name}_err'] = 0.0
                    found = True
                    break
            if not found:
                # Use defaults if parameter not found
                if 'N_' in standard_name:
                    result[standard_name] = 0.0
                    result[f'{standard_name}_err'] = 0.0
                elif 'sigma2' in standard_name:
                    result[standard_name] = 0.01
                    result[f'{standard_name}_err'] = 0.0
                elif 'delr' in standard_name:
                    result[standard_name] = 0.0
                    result[f'{standard_name}_err'] = 0.0
                elif standard_name == 'e0':
                    result[standard_name] = 0.0
                    result[f'{standard_name}_err'] = 0.0
                elif standard_name == 's02':
                    result[standard_name] = 0.78
                    result[f'{standard_name}_err'] = 0.0
        
        # Check we have the essential parameters
        if result.get('N_I', 0) == 0 and result.get('N_O', 0) == 0:
            return None
        
        return result

    def plot_group_transparency_figure(self, figsize=(12, 10), dpi=300, save_path=None,
                                        style='seaborn-v0_8-whitegrid', n_groups=3,
                                        alpha_range=(1.0, 0.3)):
        """
        Create a two-panel figure showing coordination number and bond length vs concentration,
        with data point transparency varying by group number within each concentration.
        
        Each concentration should have multiple groups labeled 'group 1', 'group 2', etc.
        Higher group numbers are plotted with higher transparency (lower alpha).
        
        Parameters:
            figsize (tuple): Figure size in inches (width, height)
            dpi (int): Resolution for saved figure
            save_path (str): Path to save figure (None = display only)
            style (str): Matplotlib style to use
            n_groups (int): Expected number of groups per concentration (default: 3)
            alpha_range (tuple): (max_alpha, min_alpha) for group 1 and group n_groups
            
        Returns:
            fig, axes: Matplotlib figure and axes objects
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import re
        from matplotlib.lines import Line2D
        
        # Try to apply style
        try:
            plt.style.use(style)
        except:
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                pass
        
        # --- Collect data from all groups ---
        data_collection = self._collect_fit_data_for_publication()
        
        if not data_collection['groups']:
            print("No fit data found. Run iterative_fit_linked_all() first.")
            return None, None
        
        # --- Parse group numbers from group names ---
        def extract_group_number(group_name):
            """
            Extract group number from group name.
            Matches patterns like 'group 1', 'group1', 'Group 2', 'group_3', etc.
            """
            patterns = [
                r'group\s*(\d+)',      # 'group 1', 'group1', 'group  2'
                r'group_(\d+)',         # 'group_1'
                r'Group\s*(\d+)',       # 'Group 1', 'Group2'
                r'grp\s*(\d+)',         # 'grp 1', 'grp1'
                r'g(\d+)',              # 'g1', 'g2' (at word boundary)
            ]
            group_name_lower = group_name.lower()
            for pattern in patterns:
                match = re.search(pattern, group_name, re.IGNORECASE)
                if match:
                    return int(match.group(1))
            return 1  # Default to group 1 if not found
        
        # Calculate alpha based on group number
        def get_alpha_for_group(group_num, n_grps, alpha_rng):
            """
            Calculate alpha value for a given group number.
            Group 1 gets max alpha, higher groups get progressively lower alpha.
            """
            max_alpha, min_alpha = alpha_rng
            if n_grps <= 1:
                return max_alpha
            # Linear interpolation from max_alpha (group 1) to min_alpha (group n_grps)
            alpha = max_alpha - (max_alpha - min_alpha) * (group_num - 1) / (n_grps - 1)
            return np.clip(alpha, min_alpha, max_alpha)
        
        # --- Organize data by group number ---
        for group_data in data_collection['groups']:
            group_name = group_data.get('group_name', '')
            group_num = extract_group_number(group_name)
            group_data['group_number'] = group_num
            group_data['alpha'] = get_alpha_for_group(group_num, n_groups, alpha_range)
        
        # --- Create figure with two panels ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=100)
        
        # Define colors
        color_I = '#8B008B'  # Dark magenta/purple for Pb-I
        color_O = '#DC143C'  # Crimson red for Pb-O
        
        # === PANEL 1: Coordination Number vs Concentration ===
        ax1_twin = ax1.twinx()
        
        # Sort data by concentration for trend lines (using averages per concentration)
        conc_unique = np.unique(data_collection['concentrations'])
        
        for group_data in data_collection['groups']:
            conc = group_data['concentration']
            N_I = group_data['N_I']
            N_I_err = group_data['N_I_err']
            N_O = group_data['N_O']
            N_O_err = group_data['N_O_err']
            alpha = group_data['alpha']
            group_num = group_data['group_number']
            
            # Plot Pb-I on primary axis
            ax1.errorbar(conc, N_I, yerr=N_I_err, fmt='o', markersize=10,
                        color=color_I, markeredgecolor='k', markeredgewidth=0.8,
                        capsize=4, capthick=1.5, elinewidth=1.5,
                        alpha=alpha, zorder=5)
            
            # Plot Pb-O on secondary axis
            ax1_twin.errorbar(conc, N_O, yerr=N_O_err, fmt='s', markersize=10,
                             color=color_O, markeredgecolor='k', markeredgewidth=0.8,
                             capsize=4, capthick=1.5, elinewidth=1.5,
                             alpha=alpha, zorder=5)
        
        # Add trend lines using mean values per concentration
        conc_means = []
        N_I_means = []
        N_O_means = []
        for c in conc_unique:
            mask = np.array([g['concentration'] == c for g in data_collection['groups']])
            groups_at_c = [g for g, m in zip(data_collection['groups'], mask) if m]
            if groups_at_c:
                conc_means.append(c)
                N_I_means.append(np.mean([g['N_I'] for g in groups_at_c]))
                N_O_means.append(np.mean([g['N_O'] for g in groups_at_c]))
        
        sort_idx = np.argsort(conc_means)
        conc_means = np.array(conc_means)[sort_idx]
        N_I_means = np.array(N_I_means)[sort_idx]
        N_O_means = np.array(N_O_means)[sort_idx]
        
        ax1.plot(conc_means, N_I_means, '--', color=color_I, alpha=0.5, lw=2, label='_nolegend_')
        ax1_twin.plot(conc_means, N_O_means, '--', color=color_O, alpha=0.5, lw=2, label='_nolegend_')
        
        # Axis labels and formatting
        ax1.set_xlabel('Concentration (M)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('$N_{Pb-I}$', fontsize=12, fontweight='bold', color=color_I)
        ax1.set_ylim(0, 6)
        ax1.tick_params(axis='y', labelcolor=color_I, labelsize=10)
        ax1.tick_params(axis='x', labelsize=10)
        ax1_twin.set_ylabel('$N_{Pb-O}$', fontsize=12, fontweight='bold', color=color_O)
        ax1_twin.tick_params(axis='y', labelcolor=color_O, labelsize=10)
        ax1_twin.set_ylim(0, 6)
        ax1.set_title('Coordination Number vs. Concentration', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(left=0)
        
        # Legend for panel 1 showing group transparency
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_I,
                   markeredgecolor='k', markersize=10, label='$N_{Pb-I}$', alpha=1.0),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=color_O,
                   markeredgecolor='k', markersize=10, label='$N_{Pb-O}$', alpha=1.0),
        ]
        # Add transparency legend
        for i in range(1, n_groups + 1):
            alpha_val = get_alpha_for_group(i, n_groups, alpha_range)
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markeredgecolor='k', markersize=8, alpha=alpha_val,
                       label=f'Group {i} (α={alpha_val:.2f})')
            )
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=9,
                  framealpha=0.9, edgecolor='gray')
        
        # === PANEL 2: Bond Length vs Concentration ===
        ax2_twin = ax2.twinx()
        
        for group_data in data_collection['groups']:
            conc = group_data['concentration']
            reff_I = group_data['reff_I']
            reff_O = group_data['reff_O']
            delr_I = group_data.get('delr_I', 0)
            delr_O = group_data.get('delr_O', 0)
            delr_I_err = group_data.get('delr_I_err', 0)
            delr_O_err = group_data.get('delr_O_err', 0)
            alpha = group_data['alpha']
            
            R_I = reff_I + delr_I
            R_O = reff_O + delr_O
            
            # Plot Pb-I bond length on primary axis
            ax2.errorbar(conc, R_I, yerr=delr_I_err, fmt='o', markersize=10,
                        color=color_I, markeredgecolor='k', markeredgewidth=0.8,
                        capsize=4, capthick=1.5, elinewidth=1.5,
                        alpha=alpha, zorder=5)
            
            # Plot Pb-O bond length on secondary axis
            ax2_twin.errorbar(conc, R_O, yerr=delr_O_err, fmt='s', markersize=10,
                             color=color_O, markeredgecolor='k', markeredgewidth=0.8,
                             capsize=4, capthick=1.5, elinewidth=1.5,
                             alpha=alpha, zorder=5)
        
        # Add trend lines for bond lengths
        R_I_means = []
        R_O_means = []
        for c in conc_unique:
            groups_at_c = [g for g in data_collection['groups'] if g['concentration'] == c]
            if groups_at_c:
                R_I_means.append(np.mean([g['reff_I'] + g.get('delr_I', 0) for g in groups_at_c]))
                R_O_means.append(np.mean([g['reff_O'] + g.get('delr_O', 0) for g in groups_at_c]))
        
        R_I_means = np.array(R_I_means)[sort_idx]
        R_O_means = np.array(R_O_means)[sort_idx]
        
        ax2.plot(conc_means, R_I_means, '--', color=color_I, alpha=0.5, lw=2)
        ax2_twin.plot(conc_means, R_O_means, '--', color=color_O, alpha=0.5, lw=2)
        
        # Axis labels and formatting
        ax2.set_xlabel('Concentration (M)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Pb–I Bond Length (Å)', fontsize=12, fontweight='bold', color=color_I)
        ax2.tick_params(axis='y', labelcolor=color_I, labelsize=10)
        ax2.tick_params(axis='x', labelsize=10)
        ax2_twin.set_ylabel('Pb–O Bond Length (Å)', fontsize=12, fontweight='bold', color=color_O)
        ax2_twin.tick_params(axis='y', labelcolor=color_O, labelsize=10)
        ax2.set_title('Bond Length vs. Concentration', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.25)
        ax2.set_xlim(left=0)
        ax2.set_ylim(2.0, 3.5)
        ax2_twin.set_ylim(2.0, 3.5)
        
        # --- Final adjustments ---
        fig.suptitle('EXAFS Fit Analysis: Group Variability', fontsize=14, fontweight='bold', y=0.98)
        
        # Add metadata annotation
        n_total = len(data_collection['groups'])
        n_conc = len(conc_unique)
        metadata_text = (f"N = {n_total} total samples | "
                        f"{n_conc} concentrations | "
                        f"{n_groups} groups per concentration")
        fig.text(0.5, 0.01, metadata_text, ha='center', fontsize=9, style='italic', color='gray')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Figure saved to: {save_path}")
        
        plt.show()

        # ===== SUMMARY: plot_group_transparency_figure =====
        print("\n" + "="*80)
        print("PLOT_GROUP_TRANSPARENCY_FIGURE SUMMARY")
        print("="*80)
        print(f"  Total groups:      {n_total}")
        print(f"  Concentrations:    {n_conc}")
        print(f"  Figure saved:      {save_path if save_path else 'Not saved'}")
        print("-"*80)
        print("  Per-concentration summary:")
        for c in conc_unique:
            groups_at_c = [g for g in data_collection['groups'] if g['concentration'] == c]
            avg_N_I = np.mean([g.get('N_I', 0) for g in groups_at_c])
            avg_N_O = np.mean([g.get('N_O', 0) for g in groups_at_c])
            avg_R = np.mean([g.get('rfactor', 0) for g in groups_at_c])
            print(f"    Conc {c:.3f} M: {len(groups_at_c)} groups, avg N_I={avg_N_I:.2f}, avg N_O={avg_N_O:.2f}, avg R={avg_R:.6f}")
        print("="*80 + "\n")

        return fig, (ax1, ax1_twin, ax2, ax2_twin)

    def _plot_coordination_vs_concentration(self, ax, data):
        """
        Plot coordination numbers vs concentration with dual y-axis.
        Pb-I in purple, Pb-O in red with error bars.
        """
        import numpy as np
        
        conc = data['concentrations']
        N_I = data['N_I']
        N_I_err = data['N_I_err']
        N_O = data['N_O']
        N_O_err = data['N_O_err']
        
        # Sort by concentration for connected lines
        sort_idx = np.argsort(conc)
        conc_sorted = conc[sort_idx]
        
        # Primary axis: Pb-I (purple)
        color_I = '#8B008B'  # Dark magenta/purple
        ax.errorbar(conc, N_I, yerr=N_I_err, fmt='o', markersize=10, 
                   color=color_I, markeredgecolor='k', markeredgewidth=0.8,
                   capsize=4, capthick=1.5, elinewidth=1.5, 
                   label='$N_{Pb-I}$', zorder=5)
        ax.plot(conc_sorted, N_I[sort_idx], '--', color=color_I, alpha=0.4, lw=1.5)
        
        ax.set_xlabel('Concentration (M)', fontsize=12, fontweight='bold')
        ax.set_ylabel('$N_{Pb-I}$', fontsize=12, fontweight='bold', color=color_I)
        ax.tick_params(axis='y', labelcolor=color_I, labelsize=10)
        ax.tick_params(axis='x', labelsize=10)
        ax.set_ylim(0, 6) 
        
        # Secondary axis: Pb-O (red)
        ax2 = ax.twinx()
        color_O = '#DC143C'  # Crimson red
        ax2.errorbar(conc, N_O, yerr=N_O_err, fmt='s', markersize=10,
                    color=color_O, markeredgecolor='k', markeredgewidth=0.8,
                    capsize=4, capthick=1.5, elinewidth=1.5,
                    label='$N_{Pb-O}$', zorder=5)
        ax2.plot(conc_sorted, N_O[sort_idx], '--', color=color_O, alpha=0.4, lw=1.5)
        
        ax2.set_ylabel('$N_{Pb-O}$', fontsize=12, fontweight='bold', color=color_O)
        ax2.tick_params(axis='y', labelcolor=color_O, labelsize=10)
        ax2.set_ylim(0, 6) 
        # Combined legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_I, 
                   markeredgecolor='k', markersize=10, label='$N_{Pb-I}$'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=color_O,
                   markeredgecolor='k', markersize=10, label='$N_{Pb-O}$')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                 framealpha=0.9, edgecolor='gray')
        
        ax.set_title('Coordination Number vs. Concentration', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)

    def _plot_bondlength_vs_concentration(self, ax, data):
        import numpy as np

        conc = data['concentrations']

        # Use actual reff values from FEFF paths (stored during data collection)
        reff_I = data['reff_I']  # Actual Pb-I reff from each group's paths
        reff_O = data['reff_O']  # Actual Pb-O reff from each group's paths

        # Compute bond lengths: R = reff + delr
        R_I = reff_I + data['delr_I']
        R_O = reff_O + data['delr_O']

        R_I_err = data.get('delr_I_err', np.zeros_like(R_I))
        R_O_err = data.get('delr_O_err', np.zeros_like(R_O))

        sort_idx = np.argsort(conc)
        conc_sorted = conc[sort_idx]

        # --- Pb–I (left axis) ---
        color_I = '#4B0082'  # deeper indigo
        ax.errorbar(conc, R_I, yerr=R_I_err, fmt='o',
                    color=color_I, markeredgecolor='k',
                    capsize=4, label='Pb–I', zorder=5)

        ax.plot(conc_sorted, R_I[sort_idx], '--', color=color_I, alpha=0.4)

        ax.set_xlabel('Concentration (M)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pb–I Bond Length (Å)', color=color_I, fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelcolor=color_I)
        ax.set_ylim(2.0, 3.5) 

        # --- Pb–O (right axis) ---
        ax2 = ax.twinx()
        color_O = '#B22222'

        ax2.errorbar(conc, R_O, yerr=R_O_err, fmt='s',
                     color=color_O, markeredgecolor='k',
                     capsize=4, label='Pb–O', zorder=5)

        ax2.plot(conc_sorted, R_O[sort_idx], '--', color=color_O, alpha=0.4)

        ax2.set_ylabel('Pb–O Bond Length (Å)', color=color_O, fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_O)
        ax2.set_ylim(2.0, 3.5)
        ax.set_title('Bond Length vs. Concentration', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.25)

    def _plot_representative_kspace(self, ax, data):
        """
        Plot representative k-space fit showing data, model, and residual.
        """
        import numpy as np
        
        fit = data['representative_fit']
        group_id = data.get('representative_group_id', 'Best Fit')
        
        if fit is None or not hasattr(fit, 'datasets') or len(fit.datasets) == 0:
            ax.text(0.5, 0.5, 'No fit data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        ds = fit.datasets[0]
        k = ds.data.k
        kweight = 3
        
        chi_data = ds.data.chi * k**kweight
        chi_model = ds.model.chi * k**kweight
        residual = chi_data - chi_model
        
        # Offset for residual
        offset = 1.2 * np.max(np.abs(chi_data))
        
        # Plot data
        ax.plot(k, chi_data, 'b-', lw=1.8, label='Data', zorder=3)
        # Plot fit
        ax.plot(k, chi_model, 'r--', lw=1.8, label='Fit', zorder=4)
        # Plot residual (offset)
        ax.fill_between(k, -offset + residual, -offset, color='green', alpha=0.3)
        ax.plot(k, -offset + residual, 'g-', lw=1, label='Residual (offset)', zorder=2)
        ax.axhline(-offset, color='gray', linestyle=':', lw=0.8)
        
        # Add FT window markers
        if hasattr(ds, 'transform'):
            kmin = getattr(ds.transform, 'kmin', None)
            kmax = getattr(ds.transform, 'kmax', None)
            if kmin:
                ax.axvline(kmin, color='orange', linestyle='--', lw=1.5, alpha=0.7)
            if kmax:
                ax.axvline(kmax, color='orange', linestyle='--', lw=1.5, alpha=0.7)
        
        ax.set_xlabel(r'$k$ (Å$^{-1}$)', fontsize=12, fontweight='bold')
        ax.set_ylabel(r'$k^3\chi(k)$ (Å$^{-3}$)', fontsize=12, fontweight='bold')
        ax.set_title(f'k-space: {group_id} (R = {fit.rfactor:.4f})', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, max(k))

    def _plot_representative_rspace(self, ax, data):
        """
        Plot representative R-space fit showing magnitude.
        """
        import numpy as np
        
        fit = data['representative_fit']
        group_id = data.get('representative_group_id', 'Best Fit')
        
        if fit is None or not hasattr(fit, 'datasets') or len(fit.datasets) == 0:
            ax.text(0.5, 0.5, 'No fit data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        ds = fit.datasets[0]
        r = ds.data.r
        
        # Calculate magnitudes
        chir_data_mag = np.sqrt(ds.data.chir_re**2 + ds.data.chir_im**2)
        chir_model_mag = np.sqrt(ds.model.chir_re**2 + ds.model.chir_im**2)
        
        # Plot data and fit
        ax.plot(r, chir_data_mag, 'b-', lw=1.8, label='Data', zorder=3)
        ax.plot(r, chir_model_mag, 'r--', lw=1.8, label='Fit', zorder=4)
        
        # Fill difference region
        ax.fill_between(r, chir_data_mag, chir_model_mag, 
                        color='green', alpha=0.2, label='Difference')
        
        # Add R-window markers
        if hasattr(ds, 'transform'):
            rmin = getattr(ds.transform, 'rmin', None)
            rmax = getattr(ds.transform, 'rmax', None)
            if rmin:
                ax.axvline(rmin, color='orange', linestyle='--', lw=1.5, alpha=0.7)
            if rmax:
                ax.axvline(rmax, color='orange', linestyle='--', lw=1.5, alpha=0.7)
        
        ax.set_xlabel(r'$R$ (Å)', fontsize=12, fontweight='bold')
        ax.set_ylabel(r'$|\chi(R)|$ (Å$^{-4}$)', fontsize=12, fontweight='bold')
        ax.set_title(f'R-space Magnitude: {group_id}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 6)
        ax.set_ylim(bottom=0)

    def _plot_path_contributions(self, ax, data):
        import numpy as np
        from larch.xafs import feffit_dataset, feffit_transform, path2chi

        fit = data['representative_fit']
        paths = data['representative_paths']

        if fit is None or len(fit.datasets) == 0:
            ax.text(0.5, 0.5, 'No path data available', ha='center', va='center',
                    transform=ax.transAxes)
            return

        ds = fit.datasets[0]
        r = ds.data.r

        # --- Plot total data ---
        chir_data_mag = np.sqrt(ds.data.chir_re**2 + ds.data.chir_im**2)
        ax.plot(r, chir_data_mag, 'k-', lw=2.5, label='Data', zorder=10)

        # --- Plot total fit ---
        chir_model_mag = np.sqrt(ds.model.chir_re**2 + ds.model.chir_im**2)
        ax.plot(r, chir_model_mag, 'r--', lw=2, label='Fit', zorder=9)

        colors = ['#6A0DAD', '#DC143C', '#228B22', '#FF8C00', '#1E90FF']

        # --- Loop over paths and plot vertical markers for reff ---
        for i, path in enumerate(paths[:5]):
            color = colors[i % len(colors)]
            label = getattr(path, 'label', f'Path {i+1}')
            
            # Get effective path distance
            reff = getattr(path, 'reff', None)
            if reff is None and hasattr(path, '_feffdat'):
                reff = getattr(path._feffdat, 'reff', None)
            
            if reff is not None and 1.0 <= reff <= 5.0:
                # Plot vertical marker at path position
                ax.axvline(reff, color=color, linestyle='--', lw=1.5, alpha=0.6)
                
                # Add label at top
                ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else np.max(chir_data_mag)
                ax.scatter([reff], [ymax * 0.9], marker='v', s=60, color=color,
                          edgecolors='k', linewidths=0.5, zorder=11, label=label)

        ax.set_xlim(1, 5)
        ax.set_ylim(bottom=0)

        ax.set_xlabel(r'$R$ (Å)', fontsize=12, fontweight='bold')
        ax.set_ylabel(r'$|\chi(R)|$', fontsize=12, fontweight='bold')

        ax.set_title('R-space Fit with Path Contributions', fontsize=12, fontweight='bold')

        ax.legend(fontsize=9, framealpha=0.9, loc='upper right')
        ax.grid(True, alpha=0.25)
    # def assign_params_to_paths(self, paths, iteration):
    #     """
    #     Assign parameter names to each FEFF path based on its scattering type.
    #     For coordination number fitting, sets degeneracy to 1 for all paths and assigns expressions
    #     (e.g. 'amp * n_PbI' for s02).
    #     """
    #     print(f"Assigning parameters to {len(paths)} paths for iteration {iteration}...")
    #     path_types = {'pbi': 0, 'pbo': 0, 'pbs': 0, 'other': 0}
    #     for path in paths:
    #         path.e0 = 'del_e0'
    #         path.orig_degen = getattr(path, 'degen', 1.0)
    #         path.degen = 1.0
    #         scat_type = getattr(path, 'scat_type', '').lower()
    #         if 'pbi' in scat_type:
    #             path.s02 = 'amp * n_PbI'
    #             path.sigma2 = 'pbi_sig2'
    #             path.deltar = 'pbi_delr'
    #             path_types['pbi'] += 1
    #         elif 'pbo' in scat_type:
    #             path.s02 = 'amp * n_PbO'
    #             path.sigma2 = 'pbo_sig2'
    #             path.deltar = 'pbo_delr'
    #             path_types['pbo'] += 1
    #         else:
    #             path.s02 = 'amp * n_PbI'
    #             path.sigma2 = 'pbi_sig2'
    #             path.deltar = 'pbi_delr'
    #             path_types['other'] += 1
    #     print(f"  Assigned parameters to {path_types['pbi']} Pb-I paths (s02 = 'amp * n_PbI')")
    #     print(f"  Assigned parameters to {path_types['pbo']} Pb-O paths (s02 = 'amp * n_PbO')")
    #     print(f"  Assigned parameters to {path_types['other']} other paths (s02 = 'amp * n_PbI')")
    #     return paths

    def get_transform_for_group(self, data_group):
        """
        Create and return a TransformGroup for the given data_group.
        Looks for optimized kmin/kmax values in this order:
          1. best_kmin, best_kmax (from ft_processing_batch)
          2. best_kmin_kw3, best_kmax_kw3 (legacy naming)
          3. default values (3.0, 12.0)
        """
        print(f"Creating Fourier Transform parameters for {getattr(data_group, 'group_name', 'unknown')}...")
        kw = 3
        kmin = 3.0
        kmax = 12.0
        
        # Check for best_kmin (from ft_processing_batch) first, then best_kmin_kw3
        if hasattr(data_group, 'best_kmin'):
            kmin = data_group.best_kmin
            print(f"  Found optimized kmin = {kmin:.2f} (from ft_processing_batch)")
        elif hasattr(data_group, 'best_kmin_kw3'):
            kmin = data_group.best_kmin_kw3
            print(f"  Found optimized kmin = {kmin:.2f} for kw=3")
        else:
            print(f"  Using default kmin = {kmin:.2f}")
        
        # Check for best_kmax (from ft_processing_batch) first, then best_kmax_kw3
        if hasattr(data_group, 'best_kmax'):
            kmax = data_group.best_kmax
            print(f"  Found optimized kmax = {kmax:.2f} (from ft_processing_batch)")
        elif hasattr(data_group, 'best_kmax_kw3'):
            kmax = data_group.best_kmax_kw3
            print(f"  Found optimized kmax = {kmax:.2f} for kw=3")
        else:
            print(f"  Using default kmax = {kmax:.2f}")
        try:
            from larch.xafs.feffit import TransformGroup  # ensure TransformGroup is imported
        except ImportError:
            from larch.xafs import TransformGroup

        rbkg = getattr(data_group, 'rbkg', 0.9)
        if rbkg is None:
            print("  Warning: No rbkg value found, using default 0.9")
            rbkg = 0.9
        # --- FIT RANGES DEFINED HERE ---
        # TransformGroup parameters:
        #   kmin, kmax: from group's best_kmin/best_kmax attributes or defaults (3.0, 12.0)
        #   rmin: set to rbkg value (default 0.9 Å)
        #   rmax: fixed at 4.0 Å
        #   dk=3, kweight=kw(default 3), window='hanning'
        trans = TransformGroup(kmin=kmin, kmax=kmax, kweight=kw, dk=3, window='hanning', rmin=rbkg, rmax=4.0)
        print(f"  Created transform with k-range: {kmin:.2f} to {kmax:.2f} Å⁻¹, kw={kw}")
        return trans

    # def categorize_paths(self, path_list):
    #     """
    #     Categorize FEFF paths by scattering type (e.g. Pb-I single scattering, Pb-O single scattering, etc.)
    #     """
    #     print(f"Categorizing {len(path_list)} paths by scattering type...")
    #     path_categories = {
    #         'pbi_ss': [],  # Pb-I single scattering
    #         'pbo_ss': [],  # Pb-O single scattering
    #         'pbs_ss': [],  # Pb-S single scattering
    #         'pb_ms':  [],  # Pb multiple scattering
    #         'other':  []
    #     }
    #     for path in path_list:
    #         scat_type = getattr(path, 'scat_type', '').lower()
    #         nleg = getattr(path, 'nleg', 2)
    #         if nleg == 2 and 'pbi' in scat_type:
    #             path_categories['pbi_ss'].append(path)
    #         elif nleg == 2 and 'pbo' in scat_type:
    #             path_categories['pbo_ss'].append(path)
    #         elif nleg == 2 and 'pbs' in scat_type:
    #             path_categories['pbs_ss'].append(path)
    #         elif 'pb_ms' in scat_type or (nleg > 2):
    #             path_categories['pb_ms'].append(path)
    #         else:
    #             path_categories['other'].append(path)
    #     print(f"  Found {len(path_categories['pbi_ss'])} Pb-I single scattering paths")
    #     print(f"  Found {len(path_categories['pbo_ss'])} Pb-O single scattering paths")
    #     print(f"  Found {len(path_categories['pbs_ss'])} Pb-S single scattering paths")
    #     print(f"  Found {len(path_categories['pb_ms'])} Pb multiple scattering paths")
    #     print(f"  Found {len(path_categories['other'])} other paths")
    #     return path_categories
    def visualize_fit_results(self, group, group_name, iteration, paths=None, show_prev_iteration=True):
        """
        Create standardized visualizations for EXAFS fit results.
        
        Parameters:
            group (Group): The data group containing fit results
            group_name (str): Name of the group for plot titles
            iteration (int): Current iteration number (e.g., 1, 2)
            paths (list, optional): List of FEFF paths used in the fit. If None, will try to get
                                paths from group.paths_iter{iteration}
            show_prev_iteration (bool): Whether to show previous iteration results in the plots
        
        Creates a two-panel figure showing:
            1. k-space data and fit (with optional previous iteration fit)
            2. R-space magnitude data and fit (with optional previous iteration fit)
            and path markers showing the Reff values of significant paths
        
        The method uses fit data from group attributes:
            - fit_result_iter{iteration}: The feffit result object
            - k_space_iter{iteration}: Dict with 'k', 'data', 'fit' arrays
            - r_space_iter{iteration}: Dict with 'r', 'data', 'fit' arrays
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Get the fit result
        fit_attr = f"fit_result_iter{iteration}"
        fit_result = getattr(group, fit_attr, None)
        if fit_result is None:
            print(f"Warning: No fit result found at group.{fit_attr}")
            return
        
        # Get paths if not provided
        if paths is None:
            paths_attr = f"paths_iter{iteration}"
            paths = getattr(group, paths_attr, [])
        
        # Get k-space and R-space data - either from stored arrays or recalculate
        k_space = getattr(group, f"k_space_iter{iteration}", None)
        r_space = getattr(group, f"r_space_iter{iteration}", None)
        
        if k_space is None or r_space is None:
            # Recalculate from the fit_result's datasets
            if hasattr(fit_result, 'datasets') and len(fit_result.datasets) > 0:
                dset0 = fit_result.datasets[0]
                k = dset0.data.k
                chi_data = dset0.data.chi * k**3
                chi_fit = dset0.model.chi * k**3
                r = dset0.data.r
                chir_data_mag = np.sqrt(dset0.data.chir_re**2 + dset0.data.chir_im**2)
                chir_fit_mag = np.sqrt(dset0.model.chir_re**2 + dset0.model.chir_im**2)
                k_space = {'k': k, 'data': chi_data, 'fit': chi_fit}
                r_space = {'r': r, 'data': chir_data_mag, 'fit': chir_fit_mag}
            else:
                print(f"Warning: Cannot extract fit data for {group_name}, iteration {iteration}")
                return
        
        # Create the visualization
        fig, (ax_k, ax_r) = plt.subplots(1, 2, figsize=(15, 6))
        
        # --- k-space plot ---
        ax_k.plot(k_space['k'], k_space['data'], 'b-', label='Data')
        ax_k.plot(k_space['k'], k_space['fit'], 'r--', label=f'Fit (Iter {iteration})')
        
        # Optionally include previous iteration's fit
        if show_prev_iteration and iteration > 1:
            prev_k_attr = f"k_space_iter{iteration-1}"
            if hasattr(group, prev_k_attr):
                iter_prev = getattr(group, prev_k_attr)
                ax_k.plot(iter_prev['k'], iter_prev['fit'], 'g-.', 
                        label=f'Fit (Iter {iteration-1})', alpha=0.5)
        
        # Add FT window markers if available
        if hasattr(fit_result, 'datasets') and len(fit_result.datasets) > 0:
            dset = fit_result.datasets[0]
            if hasattr(dset, 'transform'):
                kmin = getattr(dset.transform, 'kmin', None)
                kmax = getattr(dset.transform, 'kmax', None)
                if kmin is not None:
                    ax_k.axvline(kmin, color='gray', linestyle=':')
                if kmax is not None:
                    ax_k.axvline(kmax, color='gray', linestyle=':')
        
        ax_k.set_xlabel('$k$ ($\AA^{-1}$)')
        ax_k.set_ylabel('$k^3\chi(k)$ ($\AA^{-3}$)')
        ax_k.set_title(f'{group_name} - k-space (Iteration {iteration})')
        ax_k.legend()
        ax_k.grid(True, alpha=0.3)
        
        # --- R-space plot ---
        ax_r.plot(r_space['r'], r_space['data'], 'b-', label='Data')
        ax_r.plot(r_space['r'], r_space['fit'], 'r--', label=f'Fit (Iter {iteration})')
        
        # Optionally include previous iteration's fit
        if show_prev_iteration and iteration > 1:
            prev_r_attr = f"r_space_iter{iteration-1}"
            if hasattr(group, prev_r_attr):
                prev_r = getattr(group, prev_r_attr)
                ax_r.plot(prev_r['r'], prev_r['fit'], 'g-.', 
                        label=f'Fit (Iter {iteration-1})', alpha=0.5)
        
        # Add R-window markers if available
        if hasattr(fit_result, 'datasets') and len(fit_result.datasets) > 0:
            dset = fit_result.datasets[0]
            if hasattr(dset, 'transform'):
                rmin = getattr(dset.transform, 'rmin', None) 
                rmax = getattr(dset.transform, 'rmax', None)
                if rmin is not None:
                    ax_r.axvline(rmin, color='gray', linestyle=':')
                if rmax is not None:
                    ax_r.axvline(rmax, color='gray', linestyle=':')
        
        # Mark significant paths on the R-space plot
        colors = ['green', 'magenta', 'cyan', 'orange', 'purple']  # Colors for different path types
        path_labels = {}  # To avoid duplicate labels
        
        for i, path in enumerate(paths):
            if not hasattr(path, 'reff'):
                continue
                
            reff = path.reff
            path_type = 'unknown'
            color_idx = 0
            
            # Determine path type and color
            if 'Pb-I' in path.label:
                path_type = 'Pb-I'
                color_idx = 0
            elif 'Pb-O' in path.label:
                path_type = 'Pb-O' 
                color_idx = 1
            else:
                path_type = path.label[:4]  # Use first 4 chars
                color_idx = i % len(colors)
                
            color = colors[color_idx]
            
            # Only add line and label if we haven't seen this path type yet
            if path_type not in path_labels:
                ax_r.axvline(reff, color=color, linestyle='--', alpha=0.6)
                
                # Position the label at 80% of the y-axis height
                y_pos = 0.8 * ax_r.get_ylim()[1] * (0.9 - 0.1*color_idx)  # Stagger vertically
                
                ax_r.text(reff, y_pos, path.label, ha='center', rotation=90, 
                        alpha=0.8, fontsize=8, color=color,
                        bbox=dict(facecolor='white', alpha=0.5, pad=1))
                
                path_labels[path_type] = True
        
        ax_r.set_xlabel('$R$ ($\AA$)')
        ax_r.set_ylabel('$|\chi(R)|$ ($\AA^{-3}$)')
        ax_r.set_title(f'{group_name} - R-space (Iteration {iteration})')
        ax_r.legend()
        ax_r.grid(True, alpha=0.3)
        
        # Add a text box with fit summary info
        if hasattr(fit_result, 'rfactor') and hasattr(fit_result, 'chi_square') and hasattr(fit_result, 'params'):
            # Initialize info text
            fit_info = [
                f"R-factor: {fit_result.rfactor:.6f}",
                f"χ²: {fit_result.chi_square:.2f}",
            ]
            
            # Add k-range if available
            if hasattr(fit_result, 'datasets') and len(fit_result.datasets) > 0:
                if hasattr(fit_result.datasets[0], 'transform'):
                    trans = fit_result.datasets[0].transform
                    kmin = getattr(trans, 'kmin', None)
                    kmax = getattr(trans, 'kmax', None)
                    if kmin is not None and kmax is not None:
                        fit_info.append(f"k-range: {kmin:.1f}-{kmax:.1f} Å⁻¹")
            
            # Add path-specific details
            path_categories = {}
            for path in paths:
                if 'Pb-I' in path.label:
                    path_categories['Pb-I'] = path
                    if 'pbi_delr' in fit_result.params:
                        pbi_r_eff = path.reff + fit_result.params['pbi_delr'].value
                        fit_info.append(f"Pb-I Effective R: {pbi_r_eff:.3f} Å")
                    if 'n_PbI' in fit_result.params:
                        fit_info.append(f"N(Pb-I): {fit_result.params['n_PbI'].value:.2f}")
                elif 'Pb-O' in path.label:
                    path_categories['Pb-O'] = path
                    if 'pbo_delr' in fit_result.params:
                        pbo_r_eff = path.reff + fit_result.params['pbo_delr'].value
                        fit_info.append(f"Pb-O Effective R: {pbo_r_eff:.3f} Å")
                    if 'n_PbO' in fit_result.params:
                        fit_info.append(f"N(Pb-O): {fit_result.params['n_PbO'].value:.2f}")
            
            # Add R-factor info for first iteration
            if iteration == 1 and 'pbi_delr' in fit_result.params:
                delr_abs = abs(fit_result.params['pbi_delr'].value)
                fit_info.append(f"|ΔR|: {delr_abs:.4f}")
                
            # Add the text box to the R-space plot
            ax_r.text(0.05, 0.95, '\n'.join(fit_info), 
                    transform=ax_r.transAxes, fontsize=9,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        
        return fig
    def enhanced_iteration1_fit(self, feff_base_dir=r'C:/Users/kwill/.larch/feff/relaxed'):
        """
        Enhanced Iteration 1 – Evaluate and select the optimal Pb-I path for all groups.
        This method:
          - Walks the FEFF base directory to collect and analyze all FEFF .dat files.
          - Filters for relevant Pb-I paths.
          - Loops over every project and group (with required k and chi data from autobk).
          - For each group, creates a transform, then evaluates each filtered FEFF path by performing
            a test fit.
          - Selects the best path by a combined quality metric (function of ΔR and R-factor) and then
            performs a final fit.
          - Stores the fit result and k-/R-space data on the group, and displays diagnostic visualizations.
        """
        import os
        from pathlib import Path
        import copy
        import numpy as np
        import matplotlib.pyplot as plt
        
        try:
            from larch.xafs import feffpath, feffit_dataset, feffit
            from larch.fitting import param as create_param, param_group
            from larch.xafs.feffit import TransformGroup
        except Exception as e:
            print("Error importing required fitting modules:", e)
            return
        
        # --- Inner helper: analyze a FEFF path file ---
        def analyze_feff_path(file_path):
            path_info = {'is_pbi': False, 'path': file_path, 'filename': os.path.basename(file_path)}
            try:
                fp_obj = feffpath(filename=file_path)
                if hasattr(fp_obj._feffdat, 'absorber') and fp_obj._feffdat.absorber != 'Pb':
                    return path_info
                scatterers = []
                iodine_atoms = []
                for atom in fp_obj._feffdat.geom:
                    atom_symbol = atom[0]
                    if atom[2] != 0:  # not the absorber
                        scatterers.append(atom_symbol)
                        if atom_symbol == 'I':
                            iodine_atoms.append(atom)
                if 'I' in scatterers:
                    path_info['is_pbi'] = True
                    path_info['reff'] = fp_obj._feffdat.reff
                    path_info['nleg'] = fp_obj._feffdat.nleg
                    path_info['degen'] = fp_obj._feffdat.degen
                    path_info['geometry'] = fp_obj._feffdat.geom
                    path_info['n_iodine'] = scatterers.count('I')
                    path_info['feffpath'] = fp_obj
                    if fp_obj._feffdat.nleg == 2 and len(iodine_atoms) == 1:
                        path_info['path_type'] = 'ss'
                    elif fp_obj._feffdat.nleg > 2:
                        path_info['path_type'] = 'ms'
                    if path_info.get('path_type') == 'ss':
                        path_info['label'] = f"Pb-I SS: {path_info['reff']:.3f}Å"
                    else:
                        atoms_count = {}
                        for atom in scatterers:
                            atoms_count[atom] = atoms_count.get(atom, 0) + 1
                        atom_string = '-'.join([f"{cnt}{atom}" for atom, cnt in atoms_count.items()])
                        path_info['label'] = f"Pb-{atom_string} MS: {path_info['reff']:.3f}Å"
                return path_info
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                return path_info
        
        # --- Collect all available Pb-I paths ---
        feff_base_dir = Path(feff_base_dir)
        print("\nSearching for and analyzing Pb-I FEFF paths in", feff_base_dir)
        pbi_path_infos = []
        for root, dirs, files in os.walk(feff_base_dir):
            for file in files:
                if file.lower().endswith('.dat') and 'feff' in file.lower():
                    full_path = os.path.join(root, file)
                    info = analyze_feff_path(full_path)
                    if info.get('is_pbi'):
                        pbi_path_infos.append(info)
        pbi_path_infos.sort(key=lambda x: x.get('reff', 0))
        print(f"Found {len(pbi_path_infos)} Pb-I FEFF paths.")
        
        # Display basic summary of paths
        if pbi_path_infos:
            print("\nDetailed information about Pb-I paths:")
            print(f"{'Index':6} {'Type':6} {'Reff':8} {'Deg':6} {'n-leg':6} {'Label'}")
            print("-"*80)
            for i, info in enumerate(pbi_path_infos):
                ptype = info.get('path_type', 'unk')
                reff = info.get('reff', 0)
                deg = info.get('degen', 0)
                nleg = info.get('nleg', 0)
                label = info.get('label', info.get('filename', 'unknown'))
                print(f"{i+1:6d} {ptype:6s} {reff:.3f} Å {deg:6g} {nleg:6d}  {label}")
        
        # --- Pb-I PATH FILTERING LOGIC ---
        # Step 1: Filter by reff range (2.8 to 3.5 Å for Pb-I bonds)
        reff_min = 2.8
        reff_max = 3.5
        reff_filtered = [p for p in pbi_path_infos if reff_min <= p.get('reff', 0) <= reff_max]
        print(f"\nStep 1 - reff filter ({reff_min}-{reff_max} Å): {len(pbi_path_infos)} -> {len(reff_filtered)} paths")
        
        # Step 2: Remove all MS (multiple scattering) paths - keep only SS paths
        ss_only = [p for p in reff_filtered if p.get('path_type') == 'ss']
        print(f"Step 2 - Remove MS paths: {len(reff_filtered)} -> {len(ss_only)} paths")
        
        # Step 3: Apply tolerance to remove redundant paths (paths with very similar reff)
        reff_tolerance = 0.02  # Å - paths within this tolerance are considered redundant
        def remove_redundant_paths(paths, tolerance):
            """Remove paths with reff values within tolerance of each other, keeping the first."""
            if not paths:
                return []
            # Sort by reff
            sorted_paths = sorted(paths, key=lambda x: x.get('reff', 0))
            unique_paths = [sorted_paths[0]]
            for path in sorted_paths[1:]:
                last_reff = unique_paths[-1].get('reff', 0)
                current_reff = path.get('reff', 0)
                if abs(current_reff - last_reff) > tolerance:
                    unique_paths.append(path)
                else:
                    print(f"    Removing redundant path: {path.get('label', 'unknown')} (reff={current_reff:.3f} Å, "
                          f"within {tolerance} Å of {last_reff:.3f} Å)")
            return unique_paths
        
        filtered_paths = remove_redundant_paths(ss_only, reff_tolerance)
        print(f"Step 3 - Remove redundant paths (tolerance={reff_tolerance} Å): {len(ss_only)} -> {len(filtered_paths)} paths")
        
        # Fallback: if no paths remain after filtering, relax constraints
        if len(filtered_paths) == 0:
            print("\nWARNING: No Pb-I paths found after filtering. Relaxing constraints...")
            # Try with just reff filter, no MS removal
            filtered_paths = remove_redundant_paths(reff_filtered, reff_tolerance)
            if len(filtered_paths) == 0:
                # Use any Pb-I path with reff < 4.0
                fallback = [p for p in pbi_path_infos if p.get('reff', 0) < 4.0]
                if fallback:
                    filtered_paths = [fallback[0]]
                    print(f"  Using fallback path: {filtered_paths[0].get('label', 'unknown')}")
        
        print(f"\nFinal filtered list: {len(filtered_paths)} Pb-I paths for testing")
        if filtered_paths:
            print(f"{'Index':6} {'Type':6} {'Reff':8} {'Label'}")
            print("-"*60)
            for i, info in enumerate(filtered_paths):
                ptype = info.get('path_type', 'unk')
                reff = info.get('reff', 0)
                label = info.get('label', info.get('filename', 'unknown'))
                print(f"{i+1:6d} {ptype:6s} {reff:.3f} Å  {label}")

        # --- Inner helper: process a single group ---
        def process_group(group, group_name):
            print("\n" + "="*50)
            print(f"Processing group: {group_name}")
            print("="*50)
            if not (hasattr(group, 'k') and hasattr(group, 'chi')):
                print(f"Skipping {group_name} - missing required k and chi data")
                return False
            try:
                # --- FIT RANGES DEFINED HERE ---
                # get_transform_for_group() returns TransformGroup with:
                #   k-range: kmin, kmax from group's best_kmin/best_kmax or defaults (3.0, 12.0)
                #   R-range: rmin=rbkg (default 0.9), rmax=4.0 Å
                trans = self.get_transform_for_group(group)
                temp_data = Group(k=group.k, chi=group.chi)
            except Exception as e:
                print(f"Error creating transform for {group_name}: {e}")
                return False
            from larch.fitting import param as create_param, param_group
            print(f"Evaluating {len(filtered_paths)} Pb-I paths for {group_name}...")
            single_path_results = []
            for i, path_info in enumerate(filtered_paths):
                try:
                    fp_obj = path_info['feffpath']
                    reff = path_info.get('reff', 0)
                    path_label = path_info.get('label', 'unknown')
                    # test_path = feffpath(fp_obj.filename)
                    # test_path.degen = 1.0
                    # test_path.s02 = 'amp * n_PbI'
                    # test_path.e0 = 'del_e0'
                    # test_path.sigma2 = 'pbi_sig2'
                    # test_path.deltar = 'pbi_delr'
                    # test_path.label = path_label
                    # test_path = self.prepare_path(fp_obj, 'Pb-O', 'pbo')
                    # test_path.label = candidate_label
                    test_path = feffpath(fp_obj.filename)
                    test_path.degen = 1.0
                    test_path.s02 = 'amp * n_PbI'  # Use PbI parameters for iteration 1
                    test_path.e0 = 'del_e0'
                    test_path.sigma2 = 'pbi_sig2'
                    test_path.deltar = 'pbi_delr'
                    test_path.label = path_label

                    
                    
                    # param_group(
                    #     amp = create_param(0.94, vary=False, min=0.5, max=1.2),
                    #     del_e0 = create_param(0.0, vary=True, min=-15, max=15),
                    #     n_PbI = create_param(3.0, vary=True, min=0.0, max=8.0),
                    #     pbi_sig2 = create_param(0.01, vary=True, min=0.001, max=0.03),
                    # #     pbi_delr = create_param(0.0, vary=True, min=-0.2, max=0.2)
                    # # )
                    # dset = feffit_dataset(data=temp_data, pathlist=[test_path], transform=trans)
                    # fit_result = feffit(test_params, dset)

                    test_params = self.build_params_for_iteration(1)
                    from larch.xafs import feffit_dataset, feffit
                    dset = feffit_dataset(data=temp_data, pathlist=[test_path], transform=trans)
                    
                    # IMPORTANT: Wrap the dataset in a list when passing to feffit
                    fit_result = feffit(test_params, [dset]) 
                    # dset = self.create_feffit_dataset(group, [test_path], transform=trans)
                    # fit_result = feffit(test_params, dset)

                    if fit_result is None or not hasattr(fit_result, 'rfactor') or np.isnan(fit_result.rfactor):
                        print(f"  WARNING: Fit did not converge for path {path_label}")
                        continue
                    
                    delr_abs = abs(fit_result.params['pbi_delr'].value)
                    single_path_results.append({
                        'path_info': path_info,
                        'path': test_path,
                        'reff': reff,
                        'rfactor': fit_result.rfactor,
                        'n_PbI': fit_result.params['n_PbI'].value,
                        'pbi_sig2': fit_result.params['pbi_sig2'].value,
                        'pbi_delr': fit_result.params['pbi_delr'].value,
                        'chi_square': fit_result.chi_square,
                        'fit_result': fit_result,
                        'label': path_label,
                        'path_type': path_info.get('path_type', 'unk'),
                        'delr_abs': delr_abs
                    })
                    print(f"  Path {i+1}: {path_label} - |ΔR|: {delr_abs:.3f}, R-factor: {fit_result.rfactor:.6f}")
                except Exception as e:
                    print(f"  Error testing path {path_label} with {group_name}: {e}")
            if not single_path_results:
                print(f"No successful path fits for {group_name}")
                return False

            # --- AICc-based path selection ---
            # Compute AICc for each candidate and filter by physical sanity
            for spr in single_path_results:
                aicc_info = self._compute_aicc(spr['fit_result'])
                spr['aicc'] = aicc_info['aicc']
                spr['aicc_valid'] = aicc_info['valid']
                spr['nind'] = aicc_info['nind']
                # Lightweight sanity: reject negative sigma2, N_I < 1
                spr['sane'] = (spr['n_PbI'] >= 1.0 and spr['pbi_sig2'] > 0
                               and spr['delr_abs'] < 0.08)

            sane_results = [s for s in single_path_results if s['sane'] and s['aicc_valid']]
            if sane_results:
                sane_results.sort(key=lambda x: x['aicc'])
                best_path_result = sane_results[0]
                print(f"\n  Selected best path by AICc (from {len(sane_results)} sane candidates):")
            else:
                # Fallback to R-factor if no sane AICc candidates
                single_path_results.sort(key=lambda x: x['rfactor'])
                best_path_result = single_path_results[0]
                print(f"\n  WARNING: No sane AICc candidates; falling back to best R-factor:")

            print(f"  Path: {best_path_result['label']}")
            print(f"  R-factor: {best_path_result['rfactor']:.8f}")
            print(f"  AICc: {best_path_result.get('aicc', float('nan')):.2f}  "
                  f"(Nind={best_path_result.get('nind', 0):.1f})")
            print(f"  |ΔR|: {best_path_result['delr_abs']:.6f}")
            print(f"  N(Pb-I): {best_path_result['n_PbI']:.2f}")
            print(f"  Pb-I sigma2: {best_path_result['pbi_sig2']:.6f}")
            # Perform final fit with the best path
            path_info = best_path_result['path_info']
            fp_obj = path_info['feffpath']
            pbi_path = feffpath(fp_obj.filename)
            pbi_path.degen = 1.0
            pbi_path.s02 = 'amp * n_PbI'
            pbi_path.e0 = 'del_e0'
            pbi_path.sigma2 = 'pbi_sig2'
            pbi_path.deltar = 'pbi_delr'
            pbi_path.label = best_path_result['label']
            final_params = param_group(
                amp = create_param(0.78, vary=False, min=0.5, max=1.2),
                del_e0 = create_param(best_path_result['fit_result'].params['del_e0'].value, vary=True, min=-15, max=15),
                n_PbI = create_param(best_path_result['n_PbI'], vary=True, min=0.0, max=8.0),
                pbi_sig2 = create_param(best_path_result['pbi_sig2'], vary=True, min=0.001, max=0.03),
                pbi_delr = create_param(best_path_result['pbi_delr'], vary=True, min=-0.2, max=0.2)
            )
            final_dset = feffit_dataset(data=temp_data, pathlist=[pbi_path], transform=trans)
            fit_result_iter1 = feffit(final_params, final_dset)
            pbi_r_eff = pbi_path.reff + fit_result_iter1.params['pbi_delr'].value
            print(f"Final fit results:")
            print(f"  R-factor = {fit_result_iter1.rfactor:.6f}")
            print(f"  N(Pb-I) = {fit_result_iter1.params['n_PbI'].value:.2f}")
            print(f"  |ΔR| = {abs(fit_result_iter1.params['pbi_delr'].value):.6f}")
            print(f"  Effective R = {pbi_r_eff:.3f} Å")
            group.fit_result_iter1 = fit_result_iter1
            group.paths_iter1 = [pbi_path]
            dset0 = fit_result_iter1.datasets[0]
            k = dset0.data.k
            chi_data = dset0.data.chi * k**3
            chi_fit = dset0.model.chi * k**3
            r = dset0.data.r
            chir_data_mag = np.sqrt(dset0.data.chir_re**2 + dset0.data.chir_im**2)
            chir_fit_mag = np.sqrt(dset0.model.chir_re**2 + dset0.model.chir_im**2)
            group.k_space_iter1 = {'k': k, 'data': chi_data, 'fit': chi_fit}
            group.r_space_iter1 = {'r': r, 'data': chir_data_mag, 'fit': chir_fit_mag}
            create_visualizations(group, group_name, pbi_path, fit_result_iter1, dset0, k, r, chi_data, chi_fit, chir_data_mag, chir_fit_mag)
            return True

        def create_visualizations(group, group_name, pbi_path, fit_result, dset, k, r, chi_data, chi_fit, chir_data_mag, chir_fit_mag):
            """Create standard visualizations for the group fit."""
            fig, (ax_k, ax_r) = plt.subplots(1, 2, figsize=(15, 6))
            ax_k.plot(k, chi_data, 'b-', label='Data')
            ax_k.plot(k, chi_fit, 'r--', label='Fit (Pb-I)')
            ax_k.set_xlabel('$k$ ($\AA^{-1}$)')
            ax_k.set_ylabel('$k^3\chi(k)$ ($\AA^{-3}$)')
            ax_k.set_title(f'{group_name} - k-space (Iteration 1)')
            ax_k.legend()
            ax_k.axvline(dset.transform.kmin, color='gray', linestyle=':')
            ax_k.axvline(dset.transform.kmax, color='gray', linestyle=':')
            ax_r.plot(r, chir_data_mag, 'b-', label='Data')
            ax_r.plot(r, chir_fit_mag, 'r--', label='Fit (Pb-I)')
            ax_r.axvline(dset.transform.rmin, color='gray', linestyle=':')
            ax_r.axvline(dset.transform.rmax, color='gray', linestyle=':')
            reff = pbi_path.reff
            ax_r.axvline(reff, color='green', alpha=0.6, linestyle='--')
            y_pos = 0.8*ax_r.get_ylim()[1]*0.9
            ax_r.text(reff, y_pos, pbi_path.label, ha='center', rotation=90, alpha=0.8, fontsize=8,
                      color='green', bbox=dict(facecolor='white', alpha=0.5, pad=1))
            ax_r.set_xlabel('$R$ ($\AA$)')
            ax_r.set_ylabel('$|\chi(R)|$ ($\AA^{-3}$)')
            ax_r.set_title(f'{group_name} - R-space (Iteration 1)')
            ax_r.legend()
            pbi_r_eff = pbi_path.reff + fit_result.params['pbi_delr'].value
            fit_info = (
                f"Best Pb-I path: {pbi_path.label}\n"
                f"R-factor: {fit_result.rfactor:.6f}\n"
                f"χ²: {fit_result.chi_square:.2f}\n"
                f"k-range: {dset.transform.kmin:.1f}-{dset.transform.kmax:.1f} Å⁻¹\n"
                f"N(Pb-I) = {fit_result.params['n_PbI'].value:.2f} ± {fit_result.params['n_PbI'].stderr:.2f}\n"
                f"|ΔR| = {abs(fit_result.params['pbi_delr'].value):.6f}\n"
                f"Effective R = {pbi_r_eff:.3f} Å (reff = {pbi_path.reff:.3f} Å)\n"
            )
            ax_r.text(0.05, 0.95, fit_info, transform=ax_r.transAxes, fontsize=9,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            plt.tight_layout()
            plt.show()

        # --- Main loop: process each group in each project ---
        processed_count = 0
        results_summary = []
        for proj_name, project in self.projects.items():
            print(f"\nProcessing Project: {proj_name}")
            for group_name, group in project.groups.items():
                full_name = f"{proj_name}.{group_name}"
                print(f"\nProcessing Group: {full_name}")
                if not (hasattr(group, 'chi') and hasattr(group, 'k')):
                    print(f"  Skipping {full_name} - missing required k and chi data")
                    continue
                try:
                    success = process_group(group, full_name)
                    if success:
                        processed_count += 1
                        fit_result = group.fit_result_iter1
                        path_used = group.paths_iter1[0]
                        delr_abs = abs(fit_result.params['pbi_delr'].value)
                        results_summary.append({
                            'group': full_name,
                            'path': path_used.label,
                            'reff': path_used.reff,
                            'delr_abs': delr_abs,
                            'n_PbI': fit_result.params['n_PbI'].value,
                            'effective_r': path_used.reff + fit_result.params['pbi_delr'].value,
                            'rfactor': fit_result.rfactor
                        })
                except Exception as e:
                    import traceback
                    print(f"Error processing {full_name}: {e}")
                    traceback.print_exc()
        print("\n" + "="*100)
        print("ITERATION 1 SUMMARY: BEST Pb-I PATHS FOR ALL GROUPS")
        print("="*100)
        print(f"Successfully processed {processed_count} groups")
        print("-"*100)
        if results_summary:
            results_summary.sort(key=lambda x: x['group'])
            print(f"{'Group':30} {'Path':25} {'Reff':7} {'|ΔR|':7} {'N(Pb-I)':7} {'Eff. R':7} {'R-factor':10}")
            print("-"*100)
            for res in results_summary:
                print(f"{res['group']:30} {res['path']:25} {res['reff']:.3f} {res['delr_abs']:.3f} "
                      f"{res['n_PbI']:.2f} {res['effective_r']:.3f} {res['rfactor']:.8f}")
        print("\nEnhanced Iteration 1 complete for all groups!")
        print("="*70)
    def _get_candidate_paths(self, feff_base_dir, category):
        """
        Find and filter FEFF paths for a given category.
        
        Parameters:
            feff_base_dir (str or Path): Directory containing FEFF calculation files
            category (str): The category to filter for (e.g., "Pb-O")
        
        Returns:
            list: Filtered paths that match the category criteria
        """
        import os
        from pathlib import Path
        
        print(f"Searching for candidate '{category}' paths in {feff_base_dir}...")
        candidate_infos = []
        
        # Define inner helper function to analyze a path
        def analyze_single_path(file_path, category):
            path_info = {f'is_{category.lower()}': False, 'path': file_path,
                        'filename': os.path.basename(file_path)}
            try:
                from larch.xafs import feffpath
                fp_obj = feffpath(filename=file_path)
                # Check absorber is Pb
                if hasattr(fp_obj._feffdat, 'absorber') and fp_obj._feffdat.absorber != 'Pb':
                    return path_info
                    
                # Extract target element from category (e.g., "O" from "Pb-O")
                target_element = None
                if '-' in category:
                    parts = category.split('-')
                    if len(parts) >= 2:
                        target_element = parts[1]  # Second part after hyphen
                else:
                    target_element = category
                    
                if not target_element:
                    return path_info
                    
                scatterers = []
                target_atoms = []
                for atom in fp_obj._feffdat.geom:
                    atom_symbol = atom[0]
                    if atom[2] != 0:  # not the absorber
                        scatterers.append(atom_symbol)
                        if atom_symbol.lower() == target_element.lower():
                            target_atoms.append(atom)
                            
                if target_atoms:
                    path_info[f'is_{category.lower()}'] = True
                    path_info['reff'] = fp_obj._feffdat.reff
                    path_info['nleg'] = fp_obj._feffdat.nleg
                    path_info['degen'] = fp_obj._feffdat.degen
                    path_info['geometry'] = fp_obj._feffdat.geom
                    path_info[f'n_{category.lower()}'] = scatterers.count(target_element)
                    path_info['feffpath'] = fp_obj
                    if fp_obj._feffdat.nleg == 2 and len(target_atoms) == 1:
                        path_info['path_type'] = 'ss'
                    elif fp_obj._feffdat.nleg > 2:
                        path_info['path_type'] = 'ms'
                    # Create label
                    if path_info.get('path_type') == 'ss':
                        path_info['label'] = f"Pb-{target_element} SS: {path_info['reff']:.3f}Å"
                    else:
                        atoms_count = {}
                        for a in scatterers:
                            atoms_count[a] = atoms_count.get(a, 0) + 1
                        atom_string = '-'.join([f"{cnt}{atom}" for atom, cnt in atoms_count.items()])
                        path_info['label'] = f"Pb-{atom_string} MS: {path_info['reff']:.3f}Å"
                return path_info
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                return path_info
        
        # Walk directory to find FEFF files
        feff_base_dir = Path(feff_base_dir)
        for root, dirs, files in os.walk(feff_base_dir):
            for file in files:
                if file.lower().endswith('.dat') and 'feff' in file.lower():
                    full_path = os.path.join(root, file)
                    info = analyze_single_path(full_path, category)
                    if info.get(f'is_{category.lower()}'):
                        candidate_infos.append(info)
                        
        # Sort by reff and filter
        import numpy as np
        candidate_infos.sort(key=lambda x: x.get('reff', np.inf))
        print(f"Found {len(candidate_infos)} candidate paths for category '{category}'.")
        
        # Filter candidates
        filtered = [p for p in candidate_infos if p.get('reff', 0) < 3.0 and p.get('path_type')=='ss']
        if len(filtered) == 0 and candidate_infos:
            filtered = [candidate_infos[0]]
        print(f"Filtered down to {len(filtered)} candidate paths for testing.")
        
        return filtered

    def _create_test_path(self, path_info, category):
        """
        Create a FEFF path object for testing based on path information.
        
        Parameters:
            path_info (dict): Information about the path from _get_candidate_paths
            category (str): The category of the path (e.g., "Pb-O")
        
        Returns:
            feffpath object: Configured path ready for fitting
        """
        from larch.xafs import feffpath
        
        fp_obj = path_info['feffpath']
        label = path_info.get('label', 'unknown')
        
        test_path = feffpath(fp_obj.filename)
        test_path.degen = 1.0
        
        if category.lower() == 'pb-o':
            test_path.s02 = 'amp * n_PbO'
            test_path.sigma2 = 'pbo_sig2'
            test_path.deltar = 'pbo_delr'
        elif category.lower() == 'pb-s':
            test_path.s02 = 'amp * n_PbS'
            test_path.sigma2 = 'pbs_sig2'
            test_path.deltar = 'pbs_delr'
        else:
            # Default case
            param_prefix = category.lower().replace('-', '')
            test_path.s02 = f'amp * n_{param_prefix}'
            test_path.sigma2 = f'{param_prefix}_sig2'
            test_path.deltar = f'{param_prefix}_delr'
            
        test_path.e0 = 'del_e0'
        test_path.label = label
        
        return test_path

    def transfer_parameters(self, fit_result, params_dict):
        """Extract parameters from a fit result into a dictionary"""
        
        # Extract values to the dictionary
        for param_name in fit_result.params:
            param_obj = fit_result.params[param_name]
            params_dict[param_name] = param_obj.value
            if param_obj.stderr is not None:
                params_dict[f"{param_name}_stderr"] = param_obj.stderr
        
        return params_dict


    def iterative_fit(self, group_identifier, feff_category, iteration, feff_base_dir):
        """
        Perform an iterative EXAFS fit by evaluating candidate FEFF paths of a given category and adding
        the best one to the model.
        
        Parameters:
          group_identifier (str): A string "proj.group" identifying the data group.
          feff_category (str): The FEFF path category to evaluate (e.g., "Pb-O").
          iteration (int): The iteration number (e.g., 2).
          feff_base_dir (str or Path): Directory containing FEFF calculation files.
        
        Workflow:
          - Retrieve the data group from self.projects.
          - Retrieve the previous iteration fit result (e.g., fit_result_iter1 if iteration==2).
          - Extract key parameters (e.g., amp, del_e0, n_PbI, pbi_sig2, pbi_delr) from the previous fit.
          - Walk the feff_base_dir and analyze all FEFF .dat files to select candidate paths
            that match the feff_category. (For "Pb-O", paths must show oxygen in the geometry.)
          - Filter the candidate list (for instance, by a maximum reff or by single‐scattering type).
          - For each candidate, build a test FEFF path, combine it with your previously chosen Pb‑I paths,
            and perform a test fit (using feffit). Compute a quality metric (for example, a product of |ΔR| and R‑factor).
          - Choose the candidate with the lowest quality metric.
          - Perform a final fit that now (for example) allows Pb‑I parameters to vary.
          - Store the final fit result on the group (e.g. as fit_result_iter{iteration}) as well as the updated
            list of paths.
          - Generate visualizations in the same style as earlier iterations.
        """
        import os
        from pathlib import Path
        import copy
        import numpy as np
        import matplotlib.pyplot as plt

        # For convenience, split group_identifier "proj.group" into its parts.
        try:
            proj_name, gname = group_identifier.split('.', 1)
            data_group = self.projects[proj_name].groups[gname]
        except Exception as e:
            print(f"Error retrieving group from identifier '{group_identifier}': {e}")
            return False

        # Retrieve previous iteration result. For iteration 2, expect 'fit_result_iter1'
        prev_fit_attr = f"fit_result_iter{iteration-1}"
        prev_fit = getattr(data_group, prev_fit_attr, None)
        if prev_fit is None:
            print(f"No previous fit result ({prev_fit_attr}) found in group {group_identifier}.")
            return False
        else:
            print(f"Found previous fit result ({prev_fit_attr}) for group {group_identifier}.")
        
        # Extract key parameters from previous fit
        try:
            amp_value   = prev_fit.params['amp'].value
            del_e0_value = prev_fit.params['del_e0'].value
            n_PbI_value  = prev_fit.params['n_PbI'].value
            pbi_sig2_value = prev_fit.params['pbi_sig2'].value
            pbi_delr_value = prev_fit.params['pbi_delr'].value
            print("Parameters from previous fit:")
            print(f"  amp = {amp_value:.4f}")
            print(f"  del_e0 = {del_e0_value:.4f}")
            print(f"  n_PbI = {n_PbI_value:.4f}")
            print(f"  pbi_sig2 = {pbi_sig2_value:.4f}")
            print(f"  pbi_delr = {pbi_delr_value:.4f}")
        except Exception as e:
            print("Error extracting parameters from previous iteration:", e)
            return False
        
        # Define an inner helper function to analyze FEFF paths for a given category.
        def analyze_feff_path(file_path, category):
            """Analyze a FEFF .dat file and extract information according to category.
               For category 'Pb-O', the code looks for oxygen in the scattering geometry."""
            path_info = {f'is_{category.lower()}': False, 'path': file_path,
                         'filename': os.path.basename(file_path)}
            try:
                # Use feffpath from larch.xafs to read the file.
                from larch.xafs import feffpath
                fp_obj = feffpath(filename=file_path)
                # Check absorber: require it be Pb.
                if hasattr(fp_obj._feffdat, 'absorber') and fp_obj._feffdat.absorber != 'Pb':
                    return path_info
                scatterers = []
                target_atoms = []
                for atom in fp_obj._feffdat.geom:
                    atom_symbol = atom[0]
                    if atom[2] != 0:  # not the absorber
                        scatterers.append(atom_symbol)
                        if category.lower() in atom_symbol.lower():  # e.g. "O" for Pb-O
                            target_atoms.append(atom)
                if target_atoms:
                    path_info[f'is_{category.lower()}'] = True
                    path_info['reff'] = fp_obj._feffdat.reff
                    path_info['nleg'] = fp_obj._feffdat.nleg
                    path_info['degen'] = fp_obj._feffdat.degen
                    path_info['geometry'] = fp_obj._feffdat.geom
                    path_info[f'n_{category.lower()}'] = scatterers.count(category[0])
                    path_info['feffpath'] = fp_obj
                    # Decide on the path type: single scattering if nleg==2 and only one target atom.
                    if fp_obj._feffdat.nleg == 2 and len(target_atoms)==1:
                        path_info['path_type'] = 'ss'
                    elif fp_obj._feffdat.nleg > 2:
                        path_info['path_type'] = 'ms'
                    # Create label
                    if path_info.get('path_type') == 'ss':
                        path_info['label'] = f"Pb-{category} SS: {path_info['reff']:.3f}Å"
                    else:
                        # For multiple scattering, build a label from atom counts.
                        atoms_count = {}
                        for a in scatterers:
                            atoms_count[a] = atoms_count.get(a, 0) + 1
                        atom_string = '-'.join([f"{cnt}{atom}" for atom, cnt in atoms_count.items()])
                        path_info['label'] = f"Pb-{atom_string} MS: {path_info['reff']:.3f}Å"
                return path_info
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                return path_info

        # --- Collect candidate FEFF paths for the desired category ---
        feff_base_dir = Path(feff_base_dir)
        print(f"\nSearching for and analyzing FEFF paths for category '{feff_category}' in {feff_base_dir}")
        candidate_infos = []
        for root, dirs, files in os.walk(feff_base_dir):
            for file in files:
                if file.lower().endswith('.dat') and 'feff' in file.lower():
                    full_path = os.path.join(root, file)
                    info = analyze_feff_path(full_path, feff_category)
                    if info.get(f'is_{feff_category.lower()}'):
                        candidate_infos.append(info)
        candidate_infos.sort(key=lambda x: x.get('reff', np.inf))
        print(f"Found {len(candidate_infos)} candidate FEFF paths for category '{feff_category}'.")
        
        # (Optional) Display a summary of candidate paths
        for i, info in enumerate(candidate_infos):
            label = info.get('label', info.get('filename', 'unknown'))
            reff = info.get('reff', np.nan)
            print(f"{i+1:3d}: {label}, reff: {reff:.3f} Å")
        
        # --- Filter candidates as desired.
        # For example, for Pb-O you might want only single-scattering paths with reff < 3.0 Å.
        filtered = [p for p in candidate_infos if p.get('reff', 0) < 3.0 and p.get('path_type')=='ss']
        if len(filtered) == 0 and candidate_infos:
            # Fallback: use first candidate.
            filtered = [candidate_infos[0]]
        print(f"\nFiltered to {len(filtered)} candidate paths for testing.")
        
        # --- FIT RANGES DEFINED HERE ---
        # get_transform_for_group() returns TransformGroup with:
        #   k-range: kmin, kmax from group's best_kmin/best_kmax or defaults (3.0, 12.0)
        #   R-range: rmin=rbkg (default 0.9), rmax=4.0 Å
        trans = self.get_transform_for_group(data_group)
        # Create a temporary Group object (assumed from larch) that holds k and chi.
        from larch import Group
        temp_data = Group(k=data_group.k, chi=data_group.chi)
        
        # For convenience, get the Pb-I paths from the previous iteration.
        # Here we assume they are stored in data_group.paths_iter{iteration-1} (e.g., for iteration 2, from iteration 1)
        # We require at least one Pb-I path.
        pbi_paths = []
        prev_paths_attr = f"paths_iter{iteration-1}"
        if hasattr(data_group, prev_paths_attr):
            pbi_paths = getattr(data_group, prev_paths_attr)
            print(f"Using {len(pbi_paths)} Pb-I paths from previous iteration.")
        else:
            print("WARNING: No Pb-I paths found from previous iteration!")
        
        # --- Evaluate each candidate path by performing a test fit ---
        from larch.fitting import param as create_param, param_group
        test_results = []
        print("\nEvaluating candidate paths...")
        for i, info in enumerate(filtered):
            try:
                fp_obj = info['feffpath']
                candidate_label = info.get('label', 'unknown')
                reff_candidate = info.get('reff', 0)
                # Build a test path for the candidate category.
                # For a category like "Pb-O", assign parameters accordingly.
                from larch.xafs import feffpath
                test_path = feffpath(fp_obj.filename)
                test_path.degen = 1.0
                if feff_category.lower() == 'pb-o':
                    test_path.s02 = 'amp * n_PbO'
                    test_path.sigma2 = 'pbo_sig2'
                    test_path.deltar = 'pbo_delr'
                else:
                    # Use a default if needed.
                    test_path.s02 = 'amp * n_PbO'
                    test_path.sigma2 = 'pbo_sig2'
                    test_path.deltar = 'pbo_delr'
                test_path.e0 = 'del_e0'
                test_path.label = candidate_label

                # Build a test set: combine previous Pb-I paths with this candidate.
                test_paths = copy.deepcopy(pbi_paths)  # copy the existing Pb-I paths
                test_paths.append(test_path)
                # (Ensure that for each Pb-I path, parameters are set correctly.)
                for p in test_paths:
                    if 'Pb-I' in p.label:
                        p.degen = 1.0
                        p.s02 = 'amp * n_PbI'
                        p.sigma2 = 'pbi_sig2'
                        p.deltar = 'pbi_delr'
                        p.e0 = 'del_e0'
                
                # Create dataset from temp_data and the test_paths with the transform.
                from larch.xafs import feffit_dataset, feffit
                dset = feffit_dataset(data=temp_data, pathlist=test_paths, transform=trans)
                # In the test fit, we fix Pb-I parameters so that the candidate’s impact is isolated.
                def create_fixed_params():
                    return param_group(
                        amp = create_param(amp_value, vary=False, min=0.0, max=2.0),
                        del_e0 = create_param(del_e0_value, vary=False, min=-15, max=15),
                        n_PbI = create_param(n_PbI_value, vary=False, min=0.0, max=8.0),
                        pbi_sig2 = create_param(pbi_sig2_value, vary=False, min=0.0, max=0.03),
                        pbi_delr = create_param(pbi_delr_value, vary=False, min=-0.3, max=0.3),
                        # For the candidate category, allow variation (e.g., for Pb-O)
                        n_PbO = create_param(1.0, vary=True, min=1.0, max=8.0),
                        pbo_sig2 = create_param(0.01, vary=True, min=0.0, max=0.5),
                        pbo_delr = create_param(0.0, vary=True, min=-0.3, max=0.3)
                    )
                test_params = create_fixed_params()
                test_fit = feffit(test_params, dset)
                delr_abs = abs(test_fit.params['pbo_delr'].value)
                test_results.append({
                    'info': info,
                    'path': test_path,
                    'reff': reff_candidate,
                    'rfactor': test_fit.rfactor,
                    'n_PbO': test_fit.params['n_PbO'].value,
                    'pbo_sig2': test_fit.params['pbo_sig2'].value,
                    'pbo_delr': test_fit.params['pbo_delr'].value,
                    'delr_abs': delr_abs,
                    'fit_result': test_fit,
                    'label': candidate_label,
                    'path_type': info.get('path_type', 'unk')
                })
                print(f"  Candidate {i+1}: {candidate_label}  |ΔR| = {delr_abs:.3f}, R-factor = {test_fit.rfactor:.6f}")
            except Exception as e:
                print(f"  Error testing candidate {i+1} ({candidate_label}): {e}")

        if not test_results:
            print("No candidate paths yielded a successful test fit.")
            return False
        
        # Sort by R-factor (smaller is better)
        test_results.sort(key=lambda x: x['rfactor'])
        best_candidate = test_results[0]
        print("\nBest candidate:")
        print(f"  Label: {best_candidate['label']}")
        print(f"  R-factor: {best_candidate['rfactor']:.6f}")
        print(f"  |ΔR|: {best_candidate['delr_abs']:.3f}")

        # --- Final fit: now allow previous Pb-I parameters to vary ---
        print("\nPerforming final fit with best candidate and with Pb-I parameters allowed to vary...")
        # Recreate candidate path to ensure a clean start
        from larch.xafs import feffpath
        best_info = best_candidate['info']
        best_fp_obj = best_info['feffpath']
        candidate_path = feffpath(best_fp_obj.filename)
        candidate_path.degen = 1.0
        if feff_category.lower() == 'pb-o':
            candidate_path.s02 = 'amp * n_PbO'
            candidate_path.sigma2 = 'pbo_sig2'
            candidate_path.deltar = 'pbo_delr'
        else:
            candidate_path.s02 = 'amp * n_PbO'
            candidate_path.sigma2 = 'pbo_sig2'
            candidate_path.deltar = 'pbo_delr'
        candidate_path.e0 = 'del_e0'
        candidate_path.label = best_candidate['label']
        # Combine with previous Pb-I paths
        final_paths = copy.deepcopy(pbi_paths)
        final_paths.append(candidate_path)
        # Reset the Pb-I paths to allow variation now:
        def create_final_params():
            return param_group(
                amp = create_param(amp_value, vary=False, min=0.0, max=2.0),
                del_e0 = create_param(del_e0_value, vary=True, min=-15, max=15),
                n_PbI = create_param(n_PbI_value, vary=True, min=0.0, max=8.0),
                pbi_sig2 = create_param(pbi_sig2_value, vary=True, min=0.0, max=0.03),
                pbi_delr = create_param(pbi_delr_value, vary=False, min=-0.3, max=0.3),  # still fixed
                n_PbO = create_param(best_candidate['n_PbO'], vary=True, min=1.0, max=8.0),
                pbo_sig2 = create_param(best_candidate['pbo_sig2'], vary=True, min=0.0, max=0.5),
                pbo_delr = create_param(best_candidate['pbo_delr'], vary=True, min=-0.3, max=0.3)
            )
        final_params = create_final_params()
        final_dset = feffit_dataset(data=temp_data, pathlist=final_paths, transform=trans)
        from larch.xafs import feffit
        final_fit = feffit(final_params, final_dset)
        
        if final_fit is None or not hasattr(final_fit, 'rfactor') or np.isnan(final_fit.rfactor):
            print(f"  WARNING: Final fit did not converge")
            return False
        
        print("\nFinal fit results:")
        print(f"  R-factor = {final_fit.rfactor:.6f}")
        print(f"  χ² = {final_fit.chi_square:.2f}")
        # Calculate effective distances
        pbi_r_eff = np.mean([p.reff for p in pbi_paths]) + final_fit.params['pbi_delr'].value
        pbo_r_eff = candidate_path.reff + final_fit.params['pbo_delr'].value
        print(f"  Pb-I Effective R = {pbi_r_eff:.3f} Å")
        print(f"  Pb-O Effective R = {pbo_r_eff:.3f} Å (reff = {candidate_path.reff:.3f} Å)")

        # Store the final fit result and paths on the group:
        fit_attr = f"fit_result_iter{iteration}"
        paths_attr = f"paths_iter{iteration}"
        setattr(data_group, fit_attr, final_fit)
        setattr(data_group, paths_attr, final_paths)
        # Also store candidate path separately (if desired):
        data_group.__dict__['pbo_path_iter' + str(iteration)] = candidate_path

        # --- Visualization (use the same style as previous iterations) ---
        # Create a figure with k-space and R-space plots.
        fig, (ax_k, ax_r) = plt.subplots(1, 2, figsize=(15, 6))
        dset0 = final_fit.datasets[0]
        k = dset0.data.k
        chi_data = dset0.data.chi * k**3
        chi_fit = dset0.model.chi * k**3
        ax_k.plot(k, chi_data, 'b-', label='Data')
        ax_k.plot(k, chi_fit, 'r--', label=f'Fit (Iter {iteration})')
        # Optionally, plot previous iteration’s fit (if available)
        prev_k_attr = f"k_space_iter{iteration-1}"
        if hasattr(data_group, prev_k_attr):
            iter_prev = getattr(data_group, prev_k_attr)
            ax_k.plot(iter_prev['k'], iter_prev['fit'], 'g-.', label=f'Fit (Iter {iteration-1})', alpha=0.5)
        ax_k.set_xlabel('$k$ ($\AA^{-1}$)')
        ax_k.set_ylabel('$k^3\chi(k)$ ($\AA^{-3}$)')
        ax_k.set_title(f'{group_identifier} - k-space (Iteration {iteration})')
        ax_k.legend()
        ax_k.axvline(dset0.transform.kmin, color='gray', linestyle=':')
        ax_k.axvline(dset0.transform.kmax, color='gray', linestyle=':')

        r = dset0.data.r
        chir_data_mag = np.sqrt(dset0.data.chir_re**2 + dset0.data.chir_im**2)
        chir_fit_mag = np.sqrt(dset0.model.chir_re**2 + dset0.model.chir_im**2)
        ax_r.plot(r, chir_data_mag, 'b-', label='Data')
        ax_r.plot(r, chir_fit_mag, 'r--', label=f'Fit (Iter {iteration})')
        if hasattr(data_group, 'r_space_iter' + str(iteration-1)):
            prev_r = getattr(data_group, 'r_space_iter' + str(iteration-1))
            ax_r.plot(prev_r['r'], prev_r['fit'], 'g-.', label=f'Fit (Iter {iteration-1})', alpha=0.5)
        ax_r.axvline(dset0.transform.rmin, color='gray', linestyle=':')
        ax_r.axvline(dset0.transform.rmax, color='gray', linestyle=':')
        # Mark the candidate path for Pb-O in R-space.
        ax_r.axvline(candidate_path.reff, color='magenta', linestyle='--', alpha=0.6)
        ax_r.text(candidate_path.reff, 0.8*ax_r.get_ylim()[1], candidate_path.label,
                  rotation=90, ha='center', fontsize=8, color='magenta',
                  bbox=dict(facecolor='white', alpha=0.5, pad=1))
        ax_r.set_xlabel('$R$ ($\AA$)')
        ax_r.set_ylabel('$|\chi(R)|$ ($\AA^{-3}$)')
        ax_r.set_title(f'{group_identifier} - R-space (Iteration {iteration})')
        ax_r.legend()

        # (Optional) Add a text box with fit details.
        fit_info = (f"Best {feff_category} path: {candidate_path.label}\n"
                    f"R-factor: {final_fit.rfactor:.6f}\n"
                    f"χ²: {final_fit.chi_square:.2f}\n"
                    f"k-range: {dset0.transform.kmin:.1f}-{dset0.transform.kmax:.1f} Å⁻¹\n"
                    f"Pb-I Effective R: {pbi_r_eff:.3f} Å\n"
                    f"Pb-O Effective R: {pbo_r_eff:.3f} Å")
        ax_r.text(0.05, 0.95, fit_info, transform=ax_r.transAxes, fontsize=9,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        plt.tight_layout()
        plt.show()

        # Also store the k- and r-space data for future comparisons.
        data_group.__dict__['k_space_iter' + str(iteration)] = {'k': k, 'data': chi_data, 'fit': chi_fit}
        data_group.__dict__['r_space_iter' + str(iteration)] = {'r': r, 'data': chir_data_mag, 'fit': chir_fit_mag}
        
        print(f"\nIteration {iteration} complete for group {group_identifier}.")
        return True
    # def iterative_fit_all(self, feff_category, iteration, feff_base_dir, previous_params=None,
    #                       physical_sanity_mode='final'):
    #     """
    #     Perform an iterative EXAFS fit for all groups by evaluating candidate FEFF paths
    #     and adding the best ones to the fit model if they improve the R-factor.
    #     """
    #     import os
    #     from pathlib import Path
    #     import copy
    #     import numpy as np
    #     import matplotlib.pyplot as plt
        
    #     # Existing special handling code for Pb-S paths
    #     if feff_category == 'Pb-S':
    #             if 'n_PbS' not in locals() and 'n_PbS' not in globals():
    #                 global n_PbS
    #                 n_PbS = param(3.0, min=1, max=8)  # Example values
                    
    #             if 'n_PbO' in globals():
    #                 global n_PbO
    #                 n_PbO = n_PbS  # Link the parameters
    #                 print("Linked n_PbO to n_PbS based on chemical knowledge")
        
    #     # Initialize return dictionary to store parameters for next iteration
    #     return_params = {}
        
    #     # Define an inner helper to analyze a FEFF file for a given category.
    #     def analyze_feff_path(file_path, category):
    #         path_info = {f'is_{category.lower()}': False, 'path': file_path,
    #                     'filename': os.path.basename(file_path)}
    #         try:
    #             from larch.xafs import feffpath
    #             fp_obj = feffpath(filename=file_path)
    #             # Check that the absorber is Pb.
    #             if hasattr(fp_obj._feffdat, 'absorber') and fp_obj._feffdat.absorber != 'Pb':
    #                 return path_info
                
    #             # Extract target element from category (e.g., "O" from "Pb-O")
    #             target_element = None
    #             if '-' in category:
    #                 parts = category.split('-')
    #                 if len(parts) >= 2:
    #                     target_element = parts[1]  # Second part after hyphen
    #             else:
    #                 target_element = category
                    
    #             if not target_element:
    #                 return path_info
                    
    #             scatterers = []
    #             target_atoms = []
    #             for atom in fp_obj._feffdat.geom:
    #                 atom_symbol = atom[0]
    #                 if atom[2] != 0:  # not the absorber
    #                     scatterers.append(atom_symbol)
    #                     if atom_symbol.lower() == target_element.lower():
    #                         target_atoms.append(atom)
                            
    #             if target_atoms:
    #                 path_info[f'is_{category.lower()}'] = True
    #                 path_info['reff'] = fp_obj._feffdat.reff
    #                 path_info['nleg'] = fp_obj._feffdat.nleg
    #                 path_info['degen'] = fp_obj._feffdat.degen
    #                 path_info['geometry'] = fp_obj._feffdat.geom
    #                 path_info[f'n_{category.lower()}'] = scatterers.count(target_element)
    #                 path_info['feffpath'] = fp_obj
    #                 if fp_obj._feffdat.nleg == 2 and len(target_atoms) == 1:
    #                     path_info['path_type'] = 'ss'
    #                 elif fp_obj._feffdat.nleg > 2:
    #                     path_info['path_type'] = 'ms'
    #                 if path_info.get('path_type') == 'ss':
    #                     path_info['label'] = f"Pb-{target_element} SS: {path_info['reff']:.3f}Å"
    #                 else:
    #                     atoms_count = {}
    #                     for a in scatterers:
    #                         atoms_count[a] = atoms_count.get(a, 0) + 1
    #                     atom_string = '-'.join([f"{cnt}{atom}" for atom, cnt in atoms_count.items()])
    #                     path_info['label'] = f"Pb-{atom_string} MS: {path_info['reff']:.3f}Å"
    #             return path_info
    #         except Exception as e:
    #             print(f"Error analyzing {file_path}: {e}")
    #             return path_info
        
    #     # Loop through every project and group
    #     total_processed = 0
    #     summary = []
    #     feff_base_dir = Path(feff_base_dir)
    #     for proj_name, project in self.projects.items():
    #         print(f"\nProcessing project: {proj_name}")
    #         for gname, group in project.groups.items():
    #             group_id = f"{proj_name}.{gname}"
    #             print(f"\nProcessing group: {group_id}")
    #             # Check if group has required data for iterative fitting.
    #             if not (hasattr(group, 'k') and hasattr(group, 'chi_tapered')):
    #                 print(f"  Skipping {group_id} – missing k or chi_tapered data.")
    #                 continue

    #             # Retrieve previous iteration result
    #             prev_fit_attr = f"fit_result_iter{iteration-1}"
    #             prev_fit = getattr(group, prev_fit_attr, None)
    #             if prev_fit is None:
    #                 print(f"  Skipping {group_id} – no previous iteration result ({prev_fit_attr}).")
    #                 continue
    #             else:
    #                 print(f"  Found previous iteration result for {group_id}.")
                
    #             # Store the previous R-factor for comparison
    #             prev_rfactor = prev_fit.rfactor
    #             print(f"  Previous iteration R-factor: {prev_rfactor:.6f}")

    #             # Extract key parameters from previous fit or provided parameters
    #             try:
    #                 # IMPORTANT CHANGE: Always get the original amp value from iteration 1
    #                 # First check if iteration 1's result exists
    #                 fit_result_iter1 = getattr(group, "fit_result_iter1", None)
    #                 if fit_result_iter1 is not None and hasattr(fit_result_iter1, 'params'):
    #                     # Use amplitude from iteration 1
    #                     amp_value = fit_result_iter1.params['amp'].value
    #                     print(f"  Using amplitude value {amp_value:.4f} from iteration 1")
    #                 else:
    #                     # Fallback to previous iteration
    #                     amp_value = prev_fit.params['amp'].value
    #                     print(f"  Using amplitude value {amp_value:.4f} from previous iteration")
                    
    #                 # Get other parameters from previous iteration or provided parameters
    #                 if previous_params and group_id in previous_params:
    #                     group_params = previous_params[group_id]
    #                     print(f"  Using provided parameters for {group_id}")
    #                     del_e0_value = group_params.get('del_e0', prev_fit.params['del_e0'].value)
    #                     n_PbI_value = group_params.get('n_PbI', prev_fit.params['n_PbI'].value)
    #                     pbi_sig2_value = group_params.get('pbi_sig2', prev_fit.params['pbi_sig2'].value)
    #                     pbi_delr_value = group_params.get('pbi_delr', prev_fit.params['pbi_delr'].value)
    #                 else:
    #                     del_e0_value = prev_fit.params['del_e0'].value
    #                     n_PbI_value = prev_fit.params['n_PbI'].value
    #                     pbi_sig2_value = prev_fit.params['pbi_sig2'].value
    #                     pbi_delr_value = prev_fit.params['pbi_delr'].value
                    
    #                 # Initialize entry in return_params with parameters
    #                 return_params[group_id] = {
    #                     'amp': amp_value,  # Always use the consistent amplitude
    #                     'del_e0': del_e0_value,
    #                     'n_PbI': n_PbI_value,
    #                     'pbi_sig2': pbi_sig2_value,
    #                     'pbi_delr': pbi_delr_value,
    #                     'rfactor': prev_rfactor,
    #                     'paths': [],
    #                     'iteration': iteration-1
    #                 }
                    
    #                 print(f"    Parameters: amp={amp_value:.4f}, del_e0={del_e0_value:.4f}, n_PbI={n_PbI_value:.4f}")
    #             except Exception as e:
    #                 print(f"  Error extracting parameters from previous fit for {group_id}: {e}")
    #                 continue
            
    #                 # Get candidate paths
    #             candidate_infos = []
    #             for root, dirs, files in os.walk(feff_base_dir):
    #                 for file in files:
    #                     if file.lower().endswith('.dat') and 'feff' in file.lower():
    #                         full_path = os.path.join(root, file)
    #                         info = analyze_feff_path(full_path, feff_category)
    #                         if info.get(f'is_{feff_category.lower()}'):
    #                             candidate_infos.append(info)
                
    #             # Sort by reff
    #             candidate_infos.sort(key=lambda x: x.get('reff', 0))
    #             print(f"    Found {len(candidate_infos)} candidate paths for category '{feff_category}'.")
                
    #             # ADD THIS: Filter candidates to relevant paths
    #             filtered = [p for p in candidate_infos if p.get('reff', 0) < 3.0 and p.get('path_type') == 'ss']
    #             if len(filtered) == 0 and candidate_infos:
    #                 # Fallback: use first candidate if no paths match criteria
    #                 filtered = [candidate_infos[0]]
    #             print(f"    Filtered down to {len(filtered)} candidate paths for testing.")
    #              # IMPORTANT: Get the previous iteration's paths BEFORE evaluating candidates

    #             prev_paths_attr = f"paths_iter{iteration-1}"
    #             pbi_paths = getattr(group, prev_paths_attr, [])
    #             if not pbi_paths:
    #                 print(f"  Warning: No Pb-I paths from previous iteration for {group_id}, skipping.")
    #                 continue
    #             print(f"  Using {len(pbi_paths)} Pb-I paths from previous iteration.")
            
    #             # Check if we have paths to test
    #             if not filtered:
    #                 print(f"    No candidate paths available for {group_id}, skipping.")
    #                 continue
    #               # Create transform and data object for fitting
    #             trans = self.get_transform_for_group(group)
    #             from larch import Group
    #             temp_data = Group(k=group.k, chi=group.chi_tapered)  
                  
    #             # --- Now we can evaluate each candidate path ---
    #             test_results = []
    #             print("    Evaluating candidate paths...")
    #             # --- Evaluate each candidate by performing a test fit ---
    #             from larch.fitting import param as create_param, param_group
    #             test_results = []
    #             print("    Evaluating candidate paths...")
    #             for i, info in enumerate(filtered):
    #                 try:
    #                     from larch.xafs import feffpath, feffit_dataset, feffit
    #                     fp_obj = info['feffpath']
    #                     candidate_label = info.get('label', 'unknown')
    #                     reff_candidate = info.get('reff', 0)
    #                     # Set up a test path for the candidate.
    #                     test_path = feffpath(fp_obj.filename)
    #                     test_path.degen = 1.0
    #                     if feff_category.lower() == 'pb-o':
    #                         test_path.s02 = 'amp * n_PbO'
    #                         test_path.sigma2 = 'pbo_sig2'
    #                         test_path.deltar = 'pbo_delr'
    #                     else:
    #                         test_path.s02 = 'amp * n_PbO'
    #                         test_path.sigma2 = 'pbo_sig2'
    #                         test_path.deltar = 'pbo_delr'
    #                     test_path.e0 = 'del_e0'
    #                     test_path.label = candidate_label


                        
    #                     # Build test set: combine copies of the previous paths with candidate
    #                     test_paths = copy.deepcopy(pbi_paths)
    #                     test_paths.append(test_path)
                        
    #                     # New function to identify path types
    #                     ss_paths, ms_paths = self._categorize_ss_ms_paths(test_paths)
                        
    #                     # In test fit, fix Pb-I parameters.
    #                     def create_fixed_params():
    #                         params = param_group(
    #                             amp = create_param(amp_value, vary=False, min=0.0, max=2.0),
    #                             del_e0 = create_param(del_e0_value, vary=False, min=-15, max=15),
    #                             n_PbI = create_param(n_PbI_value, vary=False, min=0.0, max=8.0),
    #                             pbi_delr = create_param(pbi_delr_value, vary=False, min=-0.3, max=0.3),
    #                             n_PbO = create_param(1.0, vary=True, min=0.0, max=8.0),
    #                             pbo_delr = create_param(0.0, vary=True, min=-0.3, max=0.3)
    #                         )
                            
    #                         # Define independent sigma2 parameters for SS paths
    #                         params.pbi_sig2 = create_param(pbi_sig2_value, vary=False, min=0.0, max=0.03)
    #                         params.pbo_sig2 = create_param(0.01, vary=True, min=0.0, max=0.03)
                            
    #                         # Define constrained sigma2 parameters for MS paths
    #                         self._add_ms_path_parameters(params, ss_paths, ms_paths)
                            
    #                         return params
                        
    #                     # Apply parameters to paths
    #                     self._apply_parameters_to_paths(test_paths, ss_paths, ms_paths)
                        
    #                     dset = feffit_dataset(data=temp_data, pathlist=test_paths, transform=trans)
    #                     # In test fit fix Pb-I parameters.
    #                     def create_fixed_params():
    #                         return param_group(
    #                             amp = create_param(amp_value, vary=False, min=0.0, max=2.0),
    #                             del_e0 = create_param(del_e0_value, vary=False, min=-15, max=15),
    #                             n_PbI = create_param(n_PbI_value, vary=False, min=0.0, max=8.0),
    #                             pbi_sig2 = create_param(pbi_sig2_value, vary=False, min=0.0, max=0.03),
    #                             pbi_delr = create_param(pbi_delr_value, vary=False, min=-0.3, max=0.3),
    #                             n_PbO = create_param(1.0, vary=True, min=1.0, max=8.0),
    #                             pbo_sig2 = create_param(0.01, vary=True, min=0.0, max=0.03),
    #                             pbo_delr = create_param(0.0, vary=True, min=-0.3, max=0.3)
    #                         )
    #                     test_params = create_fixed_params()
    #                     test_fit = feffit(test_params, dset)
    #                     delr_abs = abs(test_fit.params['pbo_delr'].value)
    #                     quality_metric = delr_abs * test_fit.rfactor
    #                     test_results.append({
    #                         'info': info,
    #                         'path': test_path,
    #                         'reff': reff_candidate,
    #                         'rfactor': test_fit.rfactor,
    #                         'n_PbO': test_fit.params['n_PbO'].value,
    #                         'pbo_sig2': test_fit.params['pbo_sig2'].value,
    #                         'pbo_delr': test_fit.params['pbo_delr'].value,
    #                         'delr_abs': delr_abs,
    #                         'quality_metric': quality_metric,
    #                         'fit_result': test_fit,
    #                         'label': candidate_label,
    #                         'path_type': info.get('path_type', 'unk')
    #                     })
    #                     print(f"      Candidate {i+1}: {candidate_label}  |ΔR| = {delr_abs:.3f}, "
    #                         f"R-factor = {test_fit.rfactor:.6f}, Quality = {quality_metric:.6f}")
    #                 except Exception as e:
    #                     print(f"      Error evaluating candidate {i+1} ({candidate_label}): {e}")
    #             if not test_results:
    #                 print(f"    No candidate paths yielded a successful test fit for {group_id}.")
    #                 continue
    #             test_results.sort(key=lambda x: x['quality_metric'])
    #             best_candidate = test_results[0]
    #             print(f"    Best candidate for {group_id}: {best_candidate['label']}")
                
    #             # --- Final fit: allow Pb-I parameters to vary now ---
    #             print(f"    Performing final fit with best candidate for {group_id}...")
    #             try:
    #                 from larch.xafs import feffpath, feffit_dataset, feffit
    #                 best_info = best_candidate['info']
    #                 best_fp_obj = best_info['feffpath']
    #                 candidate_path = feffpath(best_fp_obj.filename)
    #                 candidate_path.degen = 1.0
    #                 if feff_category.lower() == 'pb-o':
    #                     candidate_path.s02 = 'amp * n_PbO'
    #                     candidate_path.sigma2 = 'pbo_sig2'
    #                     candidate_path.deltar = 'pbo_delr'
    #                 else:
    #                     candidate_path.s02 = 'amp * n_PbO'
    #                     candidate_path.sigma2 = 'pbo_sig2'
    #                     candidate_path.deltar = 'pbo_delr'
    #                 candidate_path.e0 = 'del_e0'
    #                 candidate_path.label = best_candidate['label']
    #                 final_paths = copy.deepcopy(pbi_paths)
    #                 final_paths.append(candidate_path)
                
                

                    
    #                 # Categorize SS and MS paths
    #                 ss_paths, ms_paths = self._categorize_ss_ms_paths(final_paths)
                    
    #                 def create_final_params():
    #                     params = param_group(
    #                         amp = create_param(amp_value, vary=False, min=0.0, max=2.0),
    #                         del_e0 = create_param(del_e0_value, vary=True, min=-15, max=15),
    #                         n_PbI = create_param(n_PbI_value, vary=True, min=0.0, max=8.0),
    #                         pbi_delr = create_param(pbi_delr_value, vary=False, min=-0.3, max=0.3),
    #                         n_PbO = create_param(best_candidate['n_PbO'], vary=True, min=0.0, max=8.0),
    #                         pbo_delr = create_param(best_candidate['pbo_delr'], vary=True, min=-0.3, max=0.3)
    #                     )
                        
    #                     # Define independent sigma2 parameters for SS paths
    #                     params.pbi_sig2 = create_param(pbi_sig2_value, vary=True, min=0.0, max=0.03)
    #                     params.pbo_sig2 = create_param(best_candidate['pbo_sig2'], vary=True, min=0.0, max=0.03)
                        
    #                     # Define constrained sigma2 parameters for MS paths
    #                     self._add_ms_path_parameters(params, ss_paths, ms_paths)
                        
    #                     return params
                    
    #                 # Apply parameters to paths
    #                 self._apply_parameters_to_paths(final_paths, ss_paths, ms_paths)
                    
   
    #                 final_params = create_final_params()
    #                 # final_params = self.build_params_for_iteration(iteration, prev_result=prev_fit)
    #                 final_dset = feffit_dataset(data=temp_data, pathlist=final_paths, transform=trans)
    #                 final_fit = feffit(final_params, final_dset)
                    
    #                 # Check if R-factor improved
    #                 new_rfactor = final_fit.rfactor
    #                 print(f"    Previous R-factor: {prev_rfactor:.6f}, New R-factor: {new_rfactor:.6f}")
                    
    #                 # Only accept the new path if the R-factor improved
    #                 if new_rfactor < prev_rfactor:
    #                     print(f"    R-factor improved by {prev_rfactor - new_rfactor:.6f}, incorporating new path.")
                        
    #                     # Calculate effective distances
    #                     pbi_r_eff = np.mean([p.reff for p in pbi_paths]) + final_fit.params['pbi_delr'].value
    #                     pbo_r_eff = candidate_path.reff + final_fit.params['pbo_delr'].value
    #                     print(f"    Pb-I effective R: {pbi_r_eff:.3f} Å, Pb-O effective R: {pbo_r_eff:.3f} Å")
                        
    #                     # Store final results on the group
    #                     setattr(group, f"fit_result_iter{iteration}", final_fit)
    #                     setattr(group, f"paths_iter{iteration}", final_paths)
    #                     group.__dict__['pbo_path_iter' + str(iteration)] = candidate_path
                        
    #                     # Update return_params with new fit parameters
    #                     return_params[group_id].update({
    #                         'amp': final_fit.params['amp'].value,
    #                         'del_e0': final_fit.params['del_e0'].value,
    #                         'n_PbI': final_fit.params['n_PbI'].value,
    #                         'pbi_sig2': final_fit.params['pbi_sig2'].value,
    #                         'pbi_delr': final_fit.params['pbi_delr'].value,
    #                         'n_PbO': final_fit.params['n_PbO'].value,
    #                         'pbo_sig2': final_fit.params['pbo_sig2'].value,
    #                         'pbo_delr': final_fit.params['pbo_delr'].value,
    #                         'rfactor': new_rfactor,
    #                         'iteration': iteration
    #                     })
                        
    #                     # Add new path to paths list
    #                     return_params[group_id]['paths'].append({
    #                         'label': candidate_path.label,
    #                         'reff': candidate_path.reff,
    #                         'category': feff_category
    #                     })
                        
    #                     # Standardized visualization
    #                     fig, (ax_k, ax_r) = plt.subplots(1, 2, figsize=(15, 6))
    #                     dset0 = final_fit.datasets[0]
    #                     k = dset0.data.k
    #                     chi_data = dset0.data.chi * k**3
    #                     chi_fit = dset0.model.chi * k**3
    #                     ax_k.plot(k, chi_data, 'b-', label='Data')
    #                     ax_k.plot(k, chi_fit, 'r--', label=f'Fit (Iter {iteration})')
                        
    #                     # Add previous iteration's fit if available
    #                     prev_k_attr = f"k_space_iter{iteration-1}"
    #                     if hasattr(group, prev_k_attr):
    #                         iter_prev = getattr(group, prev_k_attr)
    #                         ax_k.plot(iter_prev['k'], iter_prev['fit'], 'g-.', 
    #                                 label=f'Fit (Iter {iteration-1})', alpha=0.5)
    #                     ax_k.set_xlabel('$k$ ($\AA^{-1}$)')
    #                     ax_k.set_ylabel('$k^3\chi(k)$ ($\AA^{-3}$)')
    #                     ax_k.set_title(f'{group_id} - k-space (Iter {iteration})')
    #                     ax_k.legend()
    #                     ax_k.axvline(dset0.transform.kmin, color='gray', linestyle=':')
    #                     ax_k.axvline(dset0.transform.kmax, color='gray', linestyle=':')
                        
    #                     r = dset0.data.r
    #                     chir_data_mag = np.sqrt(dset0.data.chir_re**2 + dset0.data.chir_im**2)
    #                     chir_fit_mag = np.sqrt(dset0.model.chir_re**2 + dset0.model.chir_im**2)
    #                     ax_r.plot(r, chir_data_mag, 'b-', label='Data')
    #                     ax_r.plot(r, chir_fit_mag, 'r--', label=f'Fit (Iter {iteration})')
    #                     if hasattr(group, f"r_space_iter{iteration-1}"):
    #                         prev_r = getattr(group, f"r_space_iter{iteration-1}")
    #                         ax_r.plot(prev_r['r'], prev_r['fit'], 'g-.', 
    #                                 label=f'Fit (Iter {iteration-1})', alpha=0.5)
    #                     ax_r.axvline(dset0.transform.rmin, color='gray', linestyle=':')
    #                     ax_r.axvline(dset0.transform.rmax, color='gray', linestyle=':')
                        
    #                     # Mark candidate path in R-space
    #                     ax_r.axvline(candidate_path.reff, color='magenta', linestyle='--', alpha=0.6)
    #                     y_pos = 0.8*ax_r.get_ylim()[1]
    #                     ax_r.text(candidate_path.reff, y_pos, candidate_path.label,
    #                             rotation=90, ha='center', fontsize=8, color='magenta',
    #                             bbox=dict(facecolor='white', alpha=0.5, pad=1))
    #                     ax_r.set_xlabel('$R$ ($\AA$)')
    #                     ax_r.set_ylabel('$|\chi(R)|$ ($\AA^{-3}$)')
    #                     ax_r.set_title(f'{group_id} - R-space (Iter {iteration})')
    #                     ax_r.legend()
                        
    #                     # Add text box with fit info
    #                     fit_info = (
    #                         f"R-factor: {new_rfactor:.6f} (improved by {prev_rfactor - new_rfactor:.6f})\n"
    #                         f"χ²: {final_fit.chi_square:.2f}\n"
    #                         f"k-range: {dset0.transform.kmin:.1f}-{dset0.transform.kmax:.1f} Å⁻¹\n\n"
    #                         f"Refined parameters:\n"
    #                         f"amp = {final_fit.params['amp'].value:.3f} ± {final_fit.params['amp'].stderr or 0:.3f}\n"
    #                         f"del_e0 = {final_fit.params['del_e0'].value:.2f} ± {final_fit.params['del_e0'].stderr or 0:.2f}\n"
    #                         f"n_PbI = {final_fit.params['n_PbI'].value:.2f} ± {final_fit.params['n_PbI'].stderr or 0:.2f}\n"
    #                         f"pbi_sig2 = {final_fit.params['pbi_sig2'].value:.5f} ± {final_fit.params['pbi_sig2'].stderr or 0:.5f}\n"
    #                         f"Pb-I Eff R = {pbi_r_eff:.3f} Å\n"
    #                         f"n_PbO = {final_fit.params['n_PbO'].value:.2f} ± {final_fit.params['n_PbO'].stderr or 0:.2f}\n"
    #                         f"pbo_sig2 = {final_fit.params['pbo_sig2'].value:.5f} ± {final_fit.params['pbo_sig2'].stderr or 0:.5f}\n"
    #                         f"pbo_delr = {final_fit.params['pbo_delr'].value:.3f} ± {final_fit.params['pbo_delr'].stderr or 0:.3f}\n"
    #                         f"Pb-O Eff R = {pbo_r_eff:.3f} Å"
    #                     )
    #                     ax_r.text(0.05, 0.95, fit_info, transform=ax_r.transAxes, fontsize=8,
    #                             verticalalignment='top',
    #                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    #                     plt.tight_layout()
    #                     plt.show()
                        
    #                     # Store k- and r-space data
    #                     group.__dict__[f'k_space_iter{iteration}'] = {'k': k, 'data': chi_data, 'fit': chi_fit}
    #                     group.__dict__[f'r_space_iter{iteration}'] = {'r': r, 'data': chir_data_mag, 'fit': chir_fit_mag}
                        
    #                     print(f"    Iteration {iteration} complete for {group_id}.")
    #                     total_processed += 1
    #                     summary.append({
    #                         'group': group_id,
    #                         'best_candidate': candidate_path.label,
    #                         'reff': candidate_path.reff,
    #                         'rfactor_improvement': prev_rfactor - new_rfactor,
    #                         'new_rfactor': new_rfactor
    #                     })
    #                 else:
    #                     print(f"    R-factor did not improve ({prev_rfactor:.6f} -> {new_rfactor:.6f}), "
    #                         f"keeping previous model.")
    #                     # Keep the previous parameters in return_params, which we already set
    #                     summary.append({
    #                         'group': group_id,
    #                         'status': 'rejected',
    #                         'best_candidate': candidate_path.label,
    #                         'prev_rfactor': prev_rfactor,
    #                         'new_rfactor': new_rfactor
    #                     })
                    
    #             except Exception as e:
    #                 print(f"    Error during final fit for {group_id}: {e}")
    #                 continue
                    
    #         # Print summary table
    #     print("\n" + "="*80)
    #     print(f"ITERATION {iteration} SUMMARY")
    #     print("="*80)
    #     print(f"Processed {total_processed} groups")
    #     if summary:
    #         print("\nDetailed Parameter Summary:")
    #         print("-"*80)
    #         for group_id, params in return_params.items():
    #             print(f"\nGroup: {group_id}")
    #             print(f"  Iteration: {params['iteration']}")
    #             print(f"  R-factor: {params['rfactor']:.6f}")
    #             print("  Parameters:")
    #             for param_name, param_value in params.items():
    #                 if param_name not in ['rfactor', 'paths', 'iteration']:
    #                     print(f"    {param_name} = {param_value}")
    #             print("  Paths:")
    #             for path in params['paths']:
    #                 print(f"    {path['label']} (reff = {path['reff']:.3f} Å, category = {path['category']})")
        
        
    #     return return_params

    def _categorize_ss_ms_paths(self, paths):
        """
        Categorize paths into single-scattering (SS) and multiple-scattering (MS) paths.
        
        Parameters:
            paths (list): List of FEFF path objects
            
        Returns:
            tuple: (ss_paths, ms_paths) - dictionaries of paths by category
        """
        ss_paths = {}  # Category -> list of SS paths
        ms_paths = {}  # Category -> list of MS paths
        
        for path in paths:
            # Determine category from path label
            category = 'unknown'
            if 'Pb-I' in path.label:
                category = 'pbi'
            elif 'Pb-O' in path.label:
                category = 'pbo'
            elif 'Pb-S' in path.label:
                category = 'pbs'
            elif 'Pb-Cs' in path.label:
                category = 'pbcs'
            
            # Determine if SS or MS path
            is_ms = False
            if hasattr(path, 'nleg'):
                is_ms = path.nleg > 2
            elif 'MS' in path.label:
                is_ms = True
                
            # Add to appropriate dictionary
            if is_ms:
                if category not in ms_paths:
                    ms_paths[category] = []
                ms_paths[category].append(path)
            else:
                if category not in ss_paths:
                    ss_paths[category] = []
                ss_paths[category].append(path)
        
        return ss_paths, ms_paths

    def _add_ms_path_parameters(self, params, ss_paths, ms_paths):
        """
        Add constrained parameters for MS paths based on SS paths.
        
        Parameters:
            params (param_group): Parameter group to modify
            ss_paths (dict): Dictionary of SS paths by category
            ms_paths (dict): Dictionary of MS paths by category
        """
        from larch.fitting import param
        
        # Create MS path parameters for each category with MS paths
        for category, paths in ms_paths.items():
            if not paths:
                continue
                
            for i, ms_path in enumerate(paths):
                # Create a unique identifier for this MS path
                ms_id = f"{category}_ms{i+1}"
                
                # Base MS sigma2 on corresponding SS sigma2
                # If we have SS paths in this category, link to them
                if category in ss_paths and ss_paths[category]:
                    # If this is a Pb-X-Pb or Pb-X-X type MS path, we can link it to SS parameters
                    # For simplicity, using a formula where MS sigma2 = sum of relevant SS sigma2s
                    # This could be refined based on specific MS path types
                    ss_sig2_param = f"{category}_sig2"
                    
                    # Create a parameter expression - in a real implementation, you might
                    # analyze the MS path to determine which SS paths contribute
                    sig2_expr = f"{ss_sig2_param} * 1.5"  # Simple example: MS disorder ~1.5x SS
                    
                    # Add the parameter with the expression
                    setattr(params, f"{ms_id}_sig2", param(expr=sig2_expr, vary=False))
                else:
                    # If no SS paths available, create an independent parameter
                    setattr(params, f"{ms_id}_sig2", param(0.015, vary=True, min=0.001, max=0.04))
                    
                # Store the mapping of path to parameter name
                ms_path._sig2_param_name = f"{ms_id}_sig2"
       
    def _apply_parameters_to_paths(self, all_paths, ss_paths, ms_paths):
        """
        Apply parameter names to each path based on its category and type.
        
        Parameters:
            all_paths (list): All paths to be used
            ss_paths (dict): Dictionary of SS paths by category
            ms_paths (dict): Dictionary of MS paths by category
        """
        # Apply standard parameters to SS paths
        for category, paths in ss_paths.items():
            for path in paths:
                path.s02 = f"amp * n_{category}"
                path.e0 = "del_e0"
                path.sigma2 = f"{category}_sig2"
                path.deltar = f"{category}_delr"
                
        # Apply parameters to MS paths
        for category, paths in ms_paths.items():
            for i, path in enumerate(paths):
                # Set standard parameters
                path.s02 = f"amp * n_{category}"  # Using the same amplitude as SS
                path.e0 = "del_e0"
                
                # Set the constrained sigma2 parameter name stored earlier
                if hasattr(path, '_sig2_param_name'):
                    path.sigma2 = path._sig2_param_name
                else:
                    # Fallback if not set
                    path.sigma2 = f"{category}_ms{i+1}_sig2"
                    
                # For deltar, we could also create constrained parameters
                # but for simplicity, using separate parameters for now
                path.deltar = f"{category}_ms{i+1}_delr"

    def iterative_fit_all(self, feff_category, iteration, feff_base_dir, previous_params=None,
                          physical_sanity_mode='final'):
        """
        Perform an iterative EXAFS fit for all groups by evaluating candidate FEFF paths
        and adding the best ones to the fit model if they improve the R-factor.
        
        Parameters:
        feff_category (str): The FEFF path category to evaluate (e.g., "Pb-O").
        iteration (int): The current iteration number (e.g., 2).
        feff_base_dir (str or Path): The directory where FEFF files are stored.
        previous_params (dict, optional): Dictionary of previous iteration parameters
                                            keyed by group_id.
        
        Parameters
        ----------
        physical_sanity_mode : str, optional
            Controls when physical sanity checks are enforced. Options are:
            - 'all'   : enforce sanity during candidate ranking and on final fit
            - 'final' : use sanity only as a diagnostic during candidate ranking,
                        but enforce it on the final accepted fit
            - 'off'   : never reject in this method on physical sanity; print warnings only
            Default is 'final', which is appropriate for bottom-up model building.

        Returns:
        dict: Dictionary of fit parameters for all groups, keyed by group_id.
                Each entry contains refined parameter values, R-factors, and path info.
        """
        import os
        from pathlib import Path
        import copy
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        
        if feff_category == 'Pb-S':
                if 'n_PbS' not in locals() and 'n_PbS' not in globals():
                    global n_PbS
                    n_PbS = param(3.0, min=1, max=8)  # Example values
                    
                if 'n_PbO' in globals():
                    global n_PbO
                    n_PbO = n_PbS  # Link the parameters
                    print("Linked n_PbO to n_PbS based on chemical knowledge")
        # Initialize return dictionary to store parameters for next iteration
        return_params = {}
       
        # Define an inner helper to analyze a FEFF file for a given category.
        def analyze_feff_path(file_path, category):
            path_info = {f'is_{category.lower()}': False, 'path': file_path,
                        'filename': os.path.basename(file_path)}
            try:
                from larch.xafs import feffpath
                fp_obj = feffpath(filename=file_path)
                # Check that the absorber is Pb.
                if hasattr(fp_obj._feffdat, 'absorber') and fp_obj._feffdat.absorber != 'Pb':
                    return path_info
                
                # Extract target element from category (e.g., "O" from "Pb-O")
                target_element = None
                if '-' in category:
                    parts = category.split('-')
                    if len(parts) >= 2:
                        target_element = parts[1]  # Second part after hyphen
                else:
                    target_element = category
                    
                if not target_element:
                    return path_info
                    
                scatterers = []
                target_atoms = []
                for atom in fp_obj._feffdat.geom:
                    atom_symbol = atom[0]
                    if atom[2] != 0:  # not the absorber
                        scatterers.append(atom_symbol)
                        if atom_symbol.lower() == target_element.lower():
                            target_atoms.append(atom)
                            
                if target_atoms:
                    path_info[f'is_{category.lower()}'] = True
                    path_info['reff'] = fp_obj._feffdat.reff
                    path_info['nleg'] = fp_obj._feffdat.nleg
                    path_info['degen'] = fp_obj._feffdat.degen
                    path_info['geometry'] = fp_obj._feffdat.geom
                    path_info[f'n_{category.lower()}'] = scatterers.count(target_element)
                    path_info['feffpath'] = fp_obj
                    if fp_obj._feffdat.nleg == 2 and len(target_atoms) == 1:
                        path_info['path_type'] = 'ss'
                    elif fp_obj._feffdat.nleg > 2:
                        path_info['path_type'] = 'ms'
                    if path_info.get('path_type') == 'ss':
                        path_info['label'] = f"Pb-{target_element} SS: {path_info['reff']:.3f}Å"
                    else:
                        atoms_count = {}
                        for a in scatterers:
                            atoms_count[a] = atoms_count.get(a, 0) + 1
                        atom_string = '-'.join([f"{cnt}{atom}" for atom, cnt in atoms_count.items()])
                        path_info['label'] = f"Pb-{atom_string} MS: {path_info['reff']:.3f}Å"
                return path_info
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                return path_info

        # Loop through every project and group.
        total_processed = 0
        summary = []
        feff_base_dir = Path(feff_base_dir)
        
        for proj_name, project in self.projects.items():
            print(f"\nProcessing project: {proj_name}")
            for gname, group in project.groups.items():
                group_id = f"{proj_name}.{gname}"
                print(f"\nProcessing group: {group_id}")
                # Check if group has required data for iterative fitting.
                if not (hasattr(group, 'k') and hasattr(group, 'chi')):
                    print(f"  Skipping {group_id} – missing k or chi data.")
                    continue

                # Retrieve previous iteration result
                prev_fit_attr = f"fit_result_iter{iteration-1}"
                prev_fit = getattr(group, prev_fit_attr, None)
                if prev_fit is None:
                    print(f"  Skipping {group_id} – no previous iteration result ({prev_fit_attr}).")
                    continue
                else:
                    print(f"  Found previous iteration result for {group_id}.")
                
                # Store the previous R-factor for comparison
                prev_rfactor = prev_fit.rfactor
                print(f"  Previous iteration R-factor: {prev_rfactor:.6f}")

                # Extract key parameters from previous fit or provided parameters
                try:
                    # IMPORTANT CHANGE: Always get the original amp value from iteration 1
                    # First check if iteration 1's result exists
                    fit_result_iter1 = getattr(group, "fit_result_iter1", None)
                    if fit_result_iter1 is not None and hasattr(fit_result_iter1, 'params'):
                        # Use amplitude from iteration 1
                        amp_value = fit_result_iter1.params['amp'].value
                        print(f"  Using amplitude value {amp_value:.4f} from iteration 1")
                    else:
                        # Fallback to previous iteration
                        amp_value = prev_fit.params['amp'].value
                        print(f"  Using amplitude value {amp_value:.4f} from previous iteration")
                    
                    # Get other parameters from previous iteration or provided parameters
                    if previous_params and group_id in previous_params:
                        group_params = previous_params[group_id]
                        print(f"  Using provided parameters for {group_id}")
                        del_e0_value = group_params.get('del_e0', prev_fit.params['del_e0'].value)
                        n_PbI_value = group_params.get('n_PbI', prev_fit.params['n_PbI'].value)
                        pbi_sig2_value = group_params.get('pbi_sig2', prev_fit.params['pbi_sig2'].value)
                        pbi_delr_value = group_params.get('pbi_delr', prev_fit.params['pbi_delr'].value)
                    else:
                        del_e0_value = prev_fit.params['del_e0'].value
                        n_PbI_value = prev_fit.params['n_PbI'].value
                        pbi_sig2_value = prev_fit.params['pbi_sig2'].value
                        pbi_delr_value = prev_fit.params['pbi_delr'].value
                    
                    # Initialize entry in return_params with parameters
                    return_params[group_id] = {
                        'amp': amp_value,  # Always use the consistent amplitude
                        'del_e0': del_e0_value,
                        'n_PbI': n_PbI_value,
                        'pbi_sig2': pbi_sig2_value,
                        'pbi_delr': pbi_delr_value,
                        'rfactor': prev_rfactor,
                        'paths': [],
                        'iteration': iteration-1
                    }
                    
                    print(f"    Parameters: amp={amp_value:.4f}, del_e0={del_e0_value:.4f}, n_PbI={n_PbI_value:.4f}")
                except Exception as e:
                    print(f"  Error extracting parameters from previous fit for {group_id}: {e}")
                    continue
             

                # --- Collect candidate FEFF paths for the desired category ---
                print(f"  Searching for candidate '{feff_category}' paths in {feff_base_dir}...")
                candidate_infos = []
                for root, dirs, files in os.walk(feff_base_dir):
                    for file in files:
                        if file.lower().endswith('.dat') and 'feff' in file.lower():
                            full_path = os.path.join(root, file)
                            info = analyze_feff_path(full_path, feff_category)
                            if info.get(f'is_{feff_category.lower()}'):
                                candidate_infos.append(info)
                candidate_infos.sort(key=lambda x: x.get('reff', np.inf))
                print(f"    Found {len(candidate_infos)} candidate paths for category '{feff_category}'.")
                
                # Optionally, display a brief summary.
                for i, info in enumerate(candidate_infos[:5]):  # Show only first 5 for brevity
                    label = info.get('label', info.get('filename', 'unknown'))
                    reff = info.get('reff', np.nan)
                    print(f"      {i+1:2d}: {label}, reff = {reff:.3f} Å")
                if len(candidate_infos) > 5:
                    print(f"      ... ({len(candidate_infos)-5} more paths)")
                
                # --- FIT RANGES DEFINED HERE ---
                # get_transform_for_group() returns TransformGroup with:
                #   k-range: kmin, kmax from group's best_kmin/best_kmax or defaults (3.0, 12.0)
                #   R-range: rmin=rbkg (default 0.9), rmax=4.0 Å (extended by +0.2*iteration below)
                trans = self.get_transform_for_group(group)
                trans.rmax = trans.rmax + 0.2 * iteration
                rmin_fit = trans.rmin + 0.3
                rmax_fit = trans.rmax
                
                # Apply category-specific R-range filtering
                # For Pb-O: use fixed 2.1-2.9 Å constraint (physically meaningful range)
                # For other categories: use transform-based R-range
                if feff_category.upper() == 'PB-O':
                    pbo_reff_min = 2.1
                    pbo_reff_max = 2.9
                    filtered = [p for p in candidate_infos if pbo_reff_min <= p.get('reff', 0) <= pbo_reff_max]
                    print(f"    Using Pb-O reff constraint: {pbo_reff_min:.1f} ≤ reff ≤ {pbo_reff_max:.1f} Å")
                    print(f"    Filtered to {len(filtered)} paths within Pb-O range")
                else:
                    filtered = [p for p in candidate_infos if rmin_fit <= p.get('reff', 0) <= rmax_fit]
                    print(f"    Using R-range filter: {rmin_fit:.2f} ≤ reff ≤ {rmax_fit:.2f} Å")
                
                # For early iterations (iteration <= 2), only consider single scattering (SS) paths
                # Multiple scattering paths should be handled in later iterations or separate methods
                if iteration <= 2:
                    before_ss_filter = len(filtered)
                    filtered = [p for p in filtered if p.get('path_type', '').lower() == 'ss']
                    print(f"    Iteration {iteration}: Filtering to SS paths only (MS paths excluded)")
                    print(f"    SS filter reduced candidates from {before_ss_filter} to {len(filtered)}")
                else:
                    print(f"    Iteration {iteration}: Testing all path types (SS and MS) within fit window")
                
                # ===== OPTIMIZATION: Deduplicate and limit candidates =====
                # Parameters for optimization
                reff_tolerance = 0.02  # Å - paths within this tolerance are considered duplicates
                max_candidates = 50    # Maximum paths to test per group
                
                # Deduplicate by reff: keep only one path per reff bin
                # Prioritize SS paths over MS paths within each bin
                before_dedup = len(filtered)
                seen_reff_bins = {}
                deduplicated = []
                
                # Sort by path_type (ss first) then by reff
                filtered.sort(key=lambda x: (0 if x.get('path_type', '').lower() == 'ss' else 1, x.get('reff', np.inf)))
                
                for p in filtered:
                    reff = p.get('reff', 0)
                    # Create bin key by rounding to tolerance
                    bin_key = round(reff / reff_tolerance)
                    path_type = p.get('path_type', 'unknown').lower()
                    
                    # For each bin, prefer to keep the first path (SS preferred due to sorting)
                    if bin_key not in seen_reff_bins:
                        seen_reff_bins[bin_key] = p
                        deduplicated.append(p)
                    # If we already have an MS path but this is SS, replace it
                    elif path_type == 'ss' and seen_reff_bins[bin_key].get('path_type', '').lower() != 'ss':
                        # Replace MS with SS
                        deduplicated.remove(seen_reff_bins[bin_key])
                        seen_reff_bins[bin_key] = p
                        deduplicated.append(p)
                
                filtered = deduplicated
                print(f"    Deduplication (tolerance={reff_tolerance}Å): reduced from {before_dedup} to {len(filtered)} unique paths")
                
                # Limit maximum candidates
                if len(filtered) > max_candidates:
                    # Keep a balanced mix: prioritize SS paths, then sample MS paths evenly across reff range
                    ss_paths = [p for p in filtered if p.get('path_type', '').lower() == 'ss']
                    ms_paths = [p for p in filtered if p.get('path_type', '').lower() != 'ss']
                    
                    # Take all SS paths (up to half the limit), then sample MS paths
                    max_ss = min(len(ss_paths), max_candidates // 2)
                    max_ms = max_candidates - max_ss
                    
                    selected_ss = ss_paths[:max_ss]
                    
                    # Sample MS paths evenly across the reff range
                    if len(ms_paths) > max_ms:
                        step = len(ms_paths) / max_ms
                        selected_ms = [ms_paths[int(i * step)] for i in range(max_ms)]
                    else:
                        selected_ms = ms_paths
                    
                    filtered = selected_ss + selected_ms
                    # Re-sort by reff
                    filtered.sort(key=lambda x: x.get('reff', np.inf))
                    print(f"    Limited to {max_candidates} candidates: {len(selected_ss)} SS + {len(selected_ms)} MS paths")
                
                print(f"    Final candidate count: {len(filtered)} paths (from {len(candidate_infos)} total).")
                
                if not filtered:
                    print(f"    No candidate paths available within R-range {rmin_fit:.2f}-{rmax_fit:.2f} Å for {group_id}, skipping.")
                    continue
                from larch import Group
                temp_data = Group(k=group.k, chi=group.chi)

                # Get the Pb-I paths from the previous iteration
                prev_paths_attr = f"paths_iter{iteration-1}"
                pbi_paths = getattr(group, prev_paths_attr, [])
                if not pbi_paths:
                    print(f"    Warning: No Pb-I paths stored from previous iteration for {group_id}.")
                    continue
                print(f"    Using {len(pbi_paths)} Pb-I paths from previous iteration.")
                
                # Store path information in return_params
                
                for p in pbi_paths:
                    # Determine category based on path label
                    if 'Pb-I' in p.label:
                        category = 'Pb-I'
                    elif 'Pb-O' in p.label:
                        category = 'Pb-O'
                    elif 'Pb-S' in p.label:
                        category = 'Pb-S'
                    elif 'Pb-Cs' in p.label:
                        category = 'Pb-Cs'
                    else:
                        category = 'unknown'
                        
                    return_params[group_id]['paths'].append({
                        'label': p.label,
                        'reff': p.reff,
                        'category': category
                    })    

                # --- Evaluate each candidate by performing a test fit ---
                from larch.fitting import param as create_param, param_group
                from larch.xafs import feffpath, feffit_dataset, feffit
                
                # Helper function to get parameter names based on category (defined once outside loop)
                def get_param_names(category):
                    if category.lower() == 'pb-o':
                        return {'n': 'n_PbO', 'sig2': 'pbo_sig2', 'delr': 'pbo_delr'}
                    elif category.lower() == 'pb-c':
                        return {'n': 'n_PbC', 'sig2': 'pbc_sig2', 'delr': 'pbc_delr'}
                    elif category.lower() == 'pb-s':
                        return {'n': 'n_PbS', 'sig2': 'pbs_sig2', 'delr': 'pbs_delr'}
                    elif category.lower() == 'pb-cs':
                        return {'n': 'n_PbCs', 'sig2': 'pbcs_sig2', 'delr': 'pbcs_delr'}
                    else:
                        category_short = category.lower().replace('pb-', '').replace('-', '')
                        return {
                            'n': f'n_Pb{category_short.capitalize()}',
                            'sig2': f'pb{category_short}_sig2',
                            'delr': f'pb{category_short}_delr'
                        }
                
                # Pre-compute param_names for the category
                param_names = get_param_names(feff_category)
                
                test_results = []
                total_candidates = len(filtered)
                print(f"    Evaluating {total_candidates} candidate paths...")
                
                # Add warning for large number of paths to prevent overfitting
                if total_candidates > 10:
                    print(f"    WARNING: Testing {total_candidates} paths - monitor for potential overfitting!")
                
                evaluation_start_time = time.time()
                
                for i, info in enumerate(filtered):
                    # Progress tracking
                    if i % 5 == 0 or i == total_candidates - 1:
                        elapsed = time.time() - evaluation_start_time
                        print(f"      Progress: {i+1}/{total_candidates} ({(i+1)/total_candidates*100:.1f}%) - {elapsed:.1f}s elapsed")
                    
                    try:
                        fp_obj = info['feffpath']
                        candidate_label = info.get('label', 'unknown')
                        reff_candidate = info.get('reff', 0)
                        path_type = info.get('path_type', 'unknown')
                        
                        # Set up a test path for the candidate.
                        test_path = feffpath(fp_obj.filename)
                        test_path.degen = 1.0
                        
                        # Dynamic parameter assignment based on category
                        if feff_category.lower() == 'pb-o':
                            test_path.s02 = 'amp * n_PbO'
                            test_path.sigma2 = 'pbo_sig2'
                            test_path.deltar = 'pbo_delr'
                        elif feff_category.lower() == 'pb-c':
                            test_path.s02 = 'amp * n_PbC'
                            test_path.sigma2 = 'pbc_sig2'
                            test_path.deltar = 'pbc_delr'
                        elif feff_category.lower() == 'pb-s':
                            test_path.s02 = 'amp * n_PbS'
                            test_path.sigma2 = 'pbs_sig2'
                            test_path.deltar = 'pbs_delr'
                        elif feff_category.lower() == 'pb-cs':
                            test_path.s02 = 'amp * n_PbCs'
                            test_path.sigma2 = 'pbcs_sig2'
                            test_path.deltar = 'pbcs_delr'
                        else:
                            # Generic fallback for other categories
                            category_short = feff_category.lower().replace('pb-', '').replace('-', '')
                            test_path.s02 = f'amp * n_Pb{category_short.capitalize()}'
                            test_path.sigma2 = f'pb{category_short}_sig2'
                            test_path.deltar = f'pb{category_short}_delr'
                        
                        test_path.e0 = 'del_e0'
                        test_path.label = candidate_label

                        # Build test set: combine copies of the previous Pb-I paths with candidate.
                        test_paths = copy.deepcopy(pbi_paths)
                        test_paths.append(test_path)
                        # Ensure each Pb-I path has correct settings.
                        for p in test_paths:
                            if 'Pb-I' in p.label:
                                p.degen = 1.0
                                p.s02 = 'amp * n_PbI'
                                p.sigma2 = 'pbi_sig2'
                                p.deltar = 'pbi_delr'
                                p.e0 = 'del_e0'

                        dset = feffit_dataset(data=temp_data, pathlist=test_paths, transform=trans)
                        
                        def create_fixed_params():
                            params_dict = {
                                'amp': create_param(amp_value, vary=False, min=0.0, max=2.0),
                                'del_e0': create_param(del_e0_value, vary=False, min=-15, max=15),
                                'n_PbI': create_param(n_PbI_value, vary=False, min=0.0, max=8.0),
                                'pbi_sig2': create_param(pbi_sig2_value, vary=False, min=0.0, max=0.03),
                                'pbi_delr': create_param(pbi_delr_value, vary=False, min=-0.3, max=0.3),
                                param_names['n']: create_param(1.0, vary=True, min=1.0, max=8.0),
                                param_names['sig2']: create_param(0.01, vary=True, min=0.0, max=0.03),
                                param_names['delr']: create_param(0.0, vary=True, min=-0.3, max=0.3)
                            }
                            return param_group(**params_dict)
                        
                        test_params = create_fixed_params()
                        test_fit = feffit(test_params, dset)
                        if test_fit is None or not hasattr(test_fit, 'rfactor') or np.isnan(test_fit.rfactor):
                            print(f"      WARNING: Test fit did not converge for candidate {i+1}/{total_candidates}")
                            continue
                        delr_abs = abs(test_fit.params[param_names['delr']].value)
                        test_results.append({
                            'info': info,
                            'path': test_path,
                            'reff': reff_candidate,
                            'rfactor': test_fit.rfactor,
                            param_names['n']: test_fit.params[param_names['n']].value,
                            param_names['sig2']: test_fit.params[param_names['sig2']].value,
                            param_names['delr']: test_fit.params[param_names['delr']].value,
                            'delr_abs': delr_abs,
                            'fit_result': test_fit,
                            'label': candidate_label,
                            'path_type': info.get('path_type', 'unk')
                        })
                        print(f"      Candidate {i+1}/{total_candidates}: {candidate_label} ({path_type})  |ΔR| = {delr_abs:.3f}, "
                            f"R-factor = {test_fit.rfactor:.6f}")
                        
                        # Early termination: if we have a good candidate with significant improvement, stop
                        # Check after testing at least 20 candidates
                        min_tests_before_early_stop = 20
                        early_stop_rfactor_threshold = 0.8 * prev_rfactor  # 20% improvement threshold
                        
                        if len(test_results) >= min_tests_before_early_stop:
                            best_so_far = min(test_results, key=lambda x: x['rfactor'])
                            if best_so_far['rfactor'] < early_stop_rfactor_threshold:
                                print(f"      Early termination: Found excellent candidate with R-factor {best_so_far['rfactor']:.6f} "
                                      f"(< {early_stop_rfactor_threshold:.6f} threshold)")
                                break
                        
                        # Memory cleanup - remove references to large objects
                        del test_fit, dset, test_params
                        
                    except Exception as e:
                        print(f"      Error evaluating candidate {i+1}/{total_candidates} ({candidate_label}): {e}")
                        continue
                
                # Evaluation summary
                evaluation_time = time.time() - evaluation_start_time
                successful_evaluations = len(test_results)
                
                print(f"    Evaluation complete: {successful_evaluations}/{total_candidates} successful fits in {evaluation_time:.1f}s")
                
                if not test_results:
                    print(f"    No candidate paths yielded a successful test fit for {group_id}.")
                    continue
                    
                # Sort by R-factor and select best
                # --- AICc-based candidate selection and acceptance ---
                # Compute AICc for previous fit
                prev_aicc_info = self._compute_aicc(prev_fit)
                prev_aicc = prev_aicc_info.get('aicc', np.inf)
                print(f"    Previous model AICc: {prev_aicc:.3f}")

                # Compute AICc and sanity for each candidate
                for cand in test_results:
                    cand['aicc_info'] = self._compute_aicc(cand['fit_result'])
                    cand['aicc'] = cand['aicc_info'].get('aicc', np.inf)
                    # Always compute sanity as a diagnostic, but only enforce it
                    # during candidate selection when physical_sanity_mode == 'all'.
                    cand['sane'], cand['sanity_reason'] = self._fit_is_physically_sane(
                        cand['fit_result'], paths=[cand['path']]
                    )

                # Choose candidate pool based on sanity mode
                if physical_sanity_mode == 'all':
                    candidate_pool = [c for c in test_results if c['sane'] and np.isfinite(c['aicc'])]
                    pool_label = 'physically sane candidates with valid AICc'
                else:
                    candidate_pool = [c for c in test_results if np.isfinite(c['aicc'])]
                    pool_label = 'candidates with valid AICc'

                if not candidate_pool:
                    print(f"    No {pool_label} for {group_id}. Keeping previous model.")
                    continue

                # Sort by AICc
                candidate_pool.sort(key=lambda x: x['aicc'])
                best_candidate = candidate_pool[0]
                print(f"    Best candidate for {group_id}: {best_candidate['label']} ({best_candidate.get('path_type', 'unknown')})")
                print(f"    Candidate sanity: {best_candidate['sane']} ({best_candidate['sanity_reason'] if not best_candidate['sane'] else 'OK'})")
                print(f"    AICc range: {min(c['aicc'] for c in candidate_pool):.3f} - {max(c['aicc'] for c in candidate_pool):.3f}")

                # --- Final fit: allow Pb-I parameters to vary now ---
                print(f"    Performing final fit with best candidate for {group_id}...")
                try:
                    best_info = best_candidate['info']
                    best_fp_obj = best_info['feffpath']
                    candidate_path = feffpath(best_fp_obj.filename)
                    candidate_path.degen = 1.0

                    # Dynamic parameter assignment based on category
                    if feff_category.lower() == 'pb-o':
                        candidate_path.s02 = 'amp * n_PbO'
                        candidate_path.sigma2 = 'pbo_sig2'
                        candidate_path.deltar = 'pbo_delr'
                    elif feff_category.lower() == 'pb-c':
                        candidate_path.s02 = 'amp * n_PbC'
                        candidate_path.sigma2 = 'pbc_sig2'
                        candidate_path.deltar = 'pbc_delr'
                    elif feff_category.lower() == 'pb-s':
                        candidate_path.s02 = 'amp * n_PbS'
                        candidate_path.sigma2 = 'pbs_sig2'
                        candidate_path.deltar = 'pbs_delr'
                    elif feff_category.lower() == 'pb-cs':
                        candidate_path.s02 = 'amp * n_PbCs'
                        candidate_path.sigma2 = 'pbcs_sig2'
                        candidate_path.deltar = 'pbcs_delr'
                    else:
                        # Generic fallback for other categories
                        category_short = feff_category.lower().replace('pb-', '').replace('-', '')
                        candidate_path.s02 = f'amp * n_Pb{category_short.capitalize()}'
                        candidate_path.sigma2 = f'pb{category_short}_sig2'
                        candidate_path.deltar = f'pb{category_short}_delr'

                    candidate_path.e0 = 'del_e0'
                    candidate_path.label = best_candidate['label']
                    final_paths = copy.deepcopy(pbi_paths)
                    final_paths.append(candidate_path)

                    # Get parameter names for final fit
                    final_param_names = get_param_names(feff_category)

                    def create_final_params():
                        params_dict = {
                            'amp': create_param(amp_value, vary=False, min=0.0, max=2.0),
                            'del_e0': create_param(del_e0_value, vary=True, min=-15, max=15),
                            'n_PbI': create_param(n_PbI_value, vary=True, min=0.0, max=8.0),
                            'pbi_sig2': create_param(pbi_sig2_value, vary=True, min=0.0, max=0.03),
                            'pbi_delr': create_param(pbi_delr_value, vary=False, min=-0.3, max=0.3),
                            final_param_names['n']: create_param(best_candidate[final_param_names['n']], vary=True, min=0.0, max=8.0),
                            final_param_names['sig2']: create_param(best_candidate[final_param_names['sig2']], vary=True, min=0.0, max=0.03),
                            final_param_names['delr']: create_param(best_candidate[final_param_names['delr']], vary=True, min=-0.3, max=0.3)
                        }
                        return param_group(**params_dict)
                    final_params = create_final_params()
                    final_dset = feffit_dataset(data=temp_data, pathlist=final_paths, transform=trans)
                    final_fit = feffit(final_params, final_dset)

                    if final_fit is None or not hasattr(final_fit, 'rfactor') or np.isnan(final_fit.rfactor):
                        print(f"    WARNING: Final fit did not converge for {group_id}")
                        continue

                    # Compute AICc for final fit
                    final_aicc_info = self._compute_aicc(final_fit)
                    final_aicc = final_aicc_info.get('aicc', np.inf)
                    delta_aicc = prev_aicc - final_aicc
                    print(f"    Previous AICc: {prev_aicc:.3f}, New AICc: {final_aicc:.3f}, ΔAICc: {delta_aicc:.3f}")

                    # Physical sanity check
                    is_sane, sanity_reason = self._fit_is_physically_sane(final_fit, paths=final_paths)
                    if physical_sanity_mode in ('all', 'final'):
                        if not is_sane:
                            print(f"    Path rejected: Final fit failed physical sanity check: {sanity_reason}. Keeping previous model.")
                            continue
                    else:
                        if not is_sane:
                            print(f"    WARNING: Final fit is not physically sane: {sanity_reason}")

                    # Only accept if AICc improved by threshold
                    aicc_improvement_threshold = 2.0
                    if delta_aicc < aicc_improvement_threshold:
                        print(f"    Path rejected: ΔAICc={delta_aicc:.3f} < threshold ({aicc_improvement_threshold:.3f}). Keeping previous model.")
                        continue

                    # Calculate effective distances
                    pbi_r_eff = np.mean([p.reff for p in pbi_paths]) + final_fit.params['pbi_delr'].value
                    pbo_r_eff = candidate_path.reff + final_fit.params['pbo_delr'].value
                    print(f"    Pb-I effective R: {pbi_r_eff:.3f} Å, Pb-O effective R: {pbo_r_eff:.3f} Å")

                    # Store final results on the group
                    setattr(group, f"fit_result_iter{iteration}", final_fit)
                    setattr(group, f"paths_iter{iteration}", final_paths)
                    group.__dict__['pbo_path_iter' + str(iteration)] = candidate_path

                    # Update return_params with new fit parameters
                    return_params[group_id].update({
                        'amp': final_fit.params['amp'].value,
                        'del_e0': final_fit.params['del_e0'].value,
                        'n_PbI': final_fit.params['n_PbI'].value,
                        'pbi_sig2': final_fit.params['pbi_sig2'].value,
                        'pbi_delr': final_fit.params['pbi_delr'].value,
                        'n_PbO': final_fit.params['n_PbO'].value,
                        'pbo_sig2': final_fit.params['pbo_sig2'].value,
                        'pbo_delr': final_fit.params['pbo_delr'].value,
                        'rfactor': final_fit.rfactor,
                        'aicc': final_aicc,
                        'delta_aicc': delta_aicc,
                        'iteration': iteration
                    })

                    # Add new path to paths list
                    return_params[group_id]['paths'].append({
                        'label': candidate_path.label,
                        'reff': candidate_path.reff,
                        'category': feff_category
                    })

                    # Standardized visualization
                    fig, (ax_k, ax_r) = plt.subplots(1, 2, figsize=(15, 6))
                    dset0 = final_fit.datasets[0]
                    k = dset0.data.k
                    chi_data = dset0.data.chi * k**3
                    chi_fit = dset0.model.chi * k**3
                    ax_k.plot(k, chi_data, 'b-', label='Data')
                    ax_k.plot(k, chi_fit, 'r--', label=f'Fit (Iter {iteration})')

                    # Add previous iteration's fit if available
                    prev_k_attr = f"k_space_iter{iteration-1}"
                    if hasattr(group, prev_k_attr):
                        iter_prev = getattr(group, prev_k_attr)
                        ax_k.plot(iter_prev['k'], iter_prev['fit'], 'g-.', 
                                label=f'Fit (Iter {iteration-1})', alpha=0.5)
                    ax_k.set_xlabel('$k$ ($\AA^{-1}$)')
                    ax_k.set_ylabel('$k^3\chi(k)$ ($\AA^{-3}$)')
                    ax_k.set_title(f'{group_id} - k-space (Iter {iteration})')
                    ax_k.legend()
                    ax_k.axvline(dset0.transform.kmin, color='gray', linestyle=':')
                    ax_k.axvline(dset0.transform.kmax, color='gray', linestyle=':')

                    r = dset0.data.r
                    chir_data_mag = np.sqrt(dset0.data.chir_re**2 + dset0.data.chir_im**2)
                    chir_fit_mag = np.sqrt(dset0.model.chir_re**2 + dset0.model.chir_im**2)
                    ax_r.plot(r, chir_data_mag, 'b-', label='Data')
                    ax_r.plot(r, chir_fit_mag, 'r--', label=f'Fit (Iter {iteration})')
                    if hasattr(group, f"r_space_iter{iteration-1}"):
                        prev_r = getattr(group, f"r_space_iter{iteration-1}")
                        ax_r.plot(prev_r['r'], prev_r['fit'], 'g-.', 
                                label=f'Fit (Iter {iteration-1})', alpha=0.5)
                    ax_r.axvline(dset0.transform.rmin, color='gray', linestyle=':')
                    ax_r.axvline(dset0.transform.rmax, color='gray', linestyle=':')

                    # Mark candidate path in R-space
                    ax_r.axvline(candidate_path.reff, color='magenta', linestyle='--', alpha=0.6)
                    y_pos = 0.8*ax_r.get_ylim()[1]
                    ax_r.text(candidate_path.reff, y_pos, candidate_path.label,
                            rotation=90, ha='center', fontsize=8, color='magenta',
                            bbox=dict(facecolor='white', alpha=0.5, pad=1))
                    ax_r.set_xlabel('$R$ ($\AA$)')
                    ax_r.set_ylabel('$|\chi(R)|$ ($\AA^{-3}$)')
                    ax_r.set_title(f'{group_id} - R-space (Iter {iteration})')
                    ax_r.legend()

                    # Add text box with fit info
                    fit_info = (
                        f"AICc: {final_aicc:.3f} (ΔAICc: {delta_aicc:.3f})\n"
                        f"R-factor: {final_fit.rfactor:.6f}\n"
                        f"χ²: {final_fit.chi_square:.2f}\n"
                        f"k-range: {dset0.transform.kmin:.1f}-{dset0.transform.kmax:.1f} Å⁻¹\n\n"
                        f"Refined parameters:\n"
                        f"amp = {final_fit.params['amp'].value:.3f} ± {final_fit.params['amp'].stderr or 0:.3f}\n"
                        f"del_e0 = {final_fit.params['del_e0'].value:.2f} ± {final_fit.params['del_e0'].stderr or 0:.2f}\n"
                        f"n_PbI = {final_fit.params['n_PbI'].value:.2f} ± {final_fit.params['n_PbI'].stderr or 0:.2f}\n"
                        f"pbi_sig2 = {final_fit.params['pbi_sig2'].value:.5f} ± {final_fit.params['pbi_sig2'].stderr or 0:.5f}\n"
                        f"Pb-I Eff R = {pbi_r_eff:.3f} Å\n"
                        f"n_PbO = {final_fit.params['n_PbO'].value:.2f} ± {final_fit.params['n_PbO'].stderr or 0:.2f}\n"
                        f"pbo_sig2 = {final_fit.params['pbo_sig2'].value:.5f} ± {final_fit.params['pbo_sig2'].stderr or 0:.5f}\n"
                        f"pbo_delr = {final_fit.params['pbo_delr'].value:.3f} ± {final_fit.params['pbo_delr'].stderr or 0:.3f}\n"
                        f"Pb-O Eff R = {pbo_r_eff:.3f} Å"
                    )
                    ax_r.text(0.05, 0.95, fit_info, transform=ax_r.transAxes, fontsize=8,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                    plt.tight_layout()
                    plt.show()

                    # Store k- and r-space data
                    group.__dict__[f'k_space_iter{iteration}'] = {'k': k, 'data': chi_data, 'fit': chi_fit}
                    group.__dict__[f'r_space_iter{iteration}'] = {'r': r, 'data': chir_data_mag, 'fit': chir_fit_mag}

                    print(f"    Iteration {iteration} complete for {group_id}.")
                    total_processed += 1
                    summary.append({
                        'group': group_id,
                        'best_candidate': candidate_path.label,
                        'reff': candidate_path.reff,
                        'aicc': final_aicc,
                        'delta_aicc': delta_aicc,
                        'rfactor': final_fit.rfactor,
                        'iteration': iteration
                    })
                except Exception as e:
                    print(f"    Error during final fit for {group_id}: {e}")
                    continue

        # Print summary table
        print("\n" + "="*80)
        print(f"ITERATION {iteration} SUMMARY")
        print("="*80)
        print(f"Processed {total_processed} groups")
        if summary:
            print("\nDetailed Parameter Summary:")
            print("-"*80)
            for group_id, params in return_params.items():
                print(f"\nGroup: {group_id}")
                print(f"  Iteration: {params['iteration']}")
                print(f"  R-factor: {params['rfactor']:.6f}")
                print("  Parameters:")
                for param_name, param_value in params.items():
                    if param_name not in ['rfactor', 'paths', 'iteration']:
                        print(f"    {param_name} = {param_value}")
                print("  Paths:")
                for path in params['paths']:
                    print(f"    {path['label']} (reff = {path['reff']:.3f} Å, category = {path['category']})")
        
        return return_params
    def _create_fit_visualization(self, group, group_id, fit, dataset, iteration, pbi_paths, new_path, prev_rfactor):
        """Create standardized visualization of fit results with R-factor comparison"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, (ax_k, ax_r) = plt.subplots(1, 2, figsize=(15, 6))
        
        # K-space data
        k = dataset.data.k
        chi_data = dataset.data.chi * k**3
        chi_fit = dataset.model.chi * k**3
        
        ax_k.plot(k, chi_data, 'b-', label='Data')
        ax_k.plot(k, chi_fit, 'r--', label=f'Fit (Iter {iteration})')
        
        # Add previous iteration fit
        if hasattr(group, f"k_space_iter{iteration-1}"):
            prev_k = getattr(group, f"k_space_iter{iteration-1}")
            ax_k.plot(prev_k['k'], prev_k['fit'], 'g-.', label=f'Fit (Iter {iteration-1})', alpha=0.5)
            
        # R-space data
        r = dataset.data.r
        chir_data_mag = np.sqrt(dataset.data.chir_re**2 + dataset.data.chir_im**2)
        chir_fit_mag = np.sqrt(dataset.model.chir_re**2 + dataset.model.chir_im**2)
        
        ax_r.plot(r, chir_data_mag, 'b-', label='Data')
        ax_r.plot(r, chir_fit_mag, 'r--', label=f'Fit (Iter {iteration})')
        
        # Add previous iteration fit
        if hasattr(group, f"r_space_iter{iteration-1}"):
            prev_r = getattr(group, f"r_space_iter{iteration-1}")
            ax_r.plot(prev_r['r'], prev_r['fit'], 'g-.', label=f'Fit (Iter {iteration-1})', alpha=0.5)
        
        # Mark path positions
        for path in pbi_paths:
            ax_r.axvline(path.reff, color='green', linestyle='--', alpha=0.4)
        
        # Highlight the new path
        ax_r.axvline(new_path.reff, color='magenta', linestyle='--', alpha=0.8)
        ax_r.text(new_path.reff, 0.8*ax_r.get_ylim()[1], new_path.label,
                  rotation=90, ha='center', fontsize=8, color='magenta',
                  bbox=dict(facecolor='white', alpha=0.7, pad=1))
        
        # Add fit metrics text box
        from larch.fitting import fit_report
        fit_details = fit_report(fit, min_correl=0.5)
        
        # Extract key metrics for a simplified display
        pbi_r_eff = np.mean([p.reff for p in pbi_paths]) + fit.params['pbi_delr'].value
        pbo_r_eff = new_path.reff + fit.params['pbo_delr'].value
        
        fit_info = (
            f"R-factor: {fit.rfactor:.6f} (improved by {prev_rfactor - fit.rfactor:.6f})\n"
            f"χ²: {fit.chi_square:.2f}\n\n"
            f"Key Parameters:\n"
            f"amp = {fit.params['amp'].value:.3f} ± {fit.params['amp'].stderr or 0:.3f}\n"
            f"n_PbI = {fit.params['n_PbI'].value:.2f} ± {fit.params['n_PbI'].stderr or 0:.2f}\n" 
            f"Pb-I Eff R = {pbi_r_eff:.3f} Å\n"
            f"n_PbO = {fit.params['n_PbO'].value:.2f} ± {fit.params['n_PbO'].stderr or 0:.2f}\n"
            f"Pb-O Eff R = {pbo_r_eff:.3f} Å"
        )
        
        ax_r.text(0.02, 0.98, fit_info, transform=ax_r.transAxes, fontsize=9,
                 va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        # Finalize and save plot
        ax_k.set_xlabel('k (Å⁻¹)')
        ax_k.set_ylabel('k³χ(k)')
        ax_k.legend()
        
        ax_r.set_xlabel('R (Å)')
        ax_r.set_ylabel('|χ(R)|')
        ax_r.legend()
        
        plt.suptitle(f"Fit Results: {group_id} - Iteration {iteration}", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Store data for future comparisons
        group.__dict__[f'k_space_iter{iteration}'] = {'k': k, 'data': chi_data, 'fit': chi_fit}
        group.__dict__[f'r_space_iter{iteration}'] = {'r': r, 'data': chir_data_mag, 'fit': chir_fit_mag}
        
        plt.show()        
    
    
    def prepare_path(self, feff_path, category, params_prefix):
        """
        Set up a FEFF path with appropriate parameters
        
        Parameters:
            feff_path: The FEFF path object
            category: Category string (e.g., 'Pb-I', 'Pb-O')
            params_prefix: Prefix for parameter names (e.g., 'pbi', 'pbo')
        
        Returns:
            Configured path ready for fitting
        """
        from larch.xafs import feffpath
        
        # Create a copy to avoid modifying the original
        path = feffpath(feff_path.filename)
        path.degen = 1.0
        path.s02 = f'amp * n_{params_prefix}'
        path.sigma2 = f'{params_prefix}_sig2'
        path.deltar = f'{params_prefix}_delr'
        path.e0 = 'del_e0'
        
        # Copy original path attributes for reference
        path.orig_label = getattr(feff_path, 'label', 'unknown')
        path.category = category
        
        return path
    def create_feffit_dataset(self, group, paths, transform=None):
        """
        Create a dataset containing data, paths, and transform parameters for fitting
        
        Parameters:
            group: Data group with k and chi attributes
            paths: List of FEFF paths
            transform: Transform parameters (or None to use default)
        
        Returns:
            A Group with data, paths, and transform attributes
        """
        from larch import Group
        
        dataset = Group(data=Group(k=group.k, chi=group.chi))
        dataset.pathlist = paths
        
        if transform is None:
            transform = self.get_transform_for_group(group)
        dataset.transform = transform
        
        return dataset

    def calculate_chir(self, k, chi, transform):
        """
        Calculate R-space transform with cleaner implementation
        
        Parameters:
            k: k values array
            chi: chi values array
            transform: Transform parameters
        
        Returns:
            r: r values array
            chir: complex R-space transform
        """
        import numpy as np
        from larch.xafs.xafsft import xafsft_fast
        
        # Get window
        kwin = transform.get_k_window(k)
        
        # Apply window and kw weighting
        weighted_chi = chi * kwin * k**transform.kw
        
        # Calculate R-space transform
        r, chir = xafsft_fast(k, weighted_chi)
        
        return r, chir
    def analyze_feff_path_statistics(self, feff_base_dir, path_categories=None, path_type='ss', max_reff=6.0):
        """
        Analyze FEFF paths to compute statistical information for use as priors in MCMC fitting.
        
        Parameters:
            feff_base_dir (str or Path): Directory containing FEFF calculation files
            path_categories (list or str, optional): Category or list of categories to analyze 
                                                  (e.g., "Pb-I", ["Pb-I", "Pb-O"])
            path_type (str): Type of paths to analyze - 'ss' (single scattering), 
                            'ms' (multiple scattering), or 'all'
            max_reff (float): Maximum effective distance to include in analysis
            
        Returns:
            dict: Dictionary of statistical information by path category with distance metrics
        """
        import os
        from pathlib import Path
        import numpy as np
        from collections import defaultdict
        
        # If path_categories is a string, convert to list
        if isinstance(path_categories, str):
            path_categories = [path_categories]
        
        # Convert to Path object
        feff_base_dir = Path(feff_base_dir)
        
        print(f"Analyzing FEFF path statistics in {feff_base_dir}")
        print(f"  Path categories: {path_categories or 'All'}")
        print(f"  Path type: {path_type}")
        print(f"  Max reff: {max_reff} Å")
        
        # Storage for paths by category
        paths_by_category = defaultdict(list)
        
        # Process all requested path categories
        if path_categories:
            # Process specified categories
            for category in path_categories:
                print(f"\nAnalyzing {category} paths...")
                candidate_paths = self._get_candidate_paths(feff_base_dir, category)
                for path_info in candidate_paths:
                    # Filter by path type
                    if path_type != 'all' and path_info.get('path_type') != path_type:
                        continue
                    
                    # Filter by max reff
                    if path_info.get('reff', 0) > max_reff:
                        continue
                    
                    paths_by_category[category].append(path_info)
        else:
            # If no categories specified, collect all paths
            all_paths = []
            print("\nAnalyzing all paths (no category filter)...")
            for root, dirs, files in os.walk(feff_base_dir):
                for file in files:
                    if file.lower().endswith('.dat') and 'feff' in file.lower():
                        full_path = os.path.join(root, file)
                        path_info = self._analyze_generic_feff_path(full_path)
                        
                        # Filter by path type
                        if path_type != 'all' and path_info.get('path_type') != path_type:
                            continue
                        
                        # Filter by max reff
                        if path_info.get('reff', 0) > max_reff:
                            continue
                        
                        category = path_info.get('category', 'unknown')
                        paths_by_category[category].append(path_info)
        
        # Calculate statistics for each category
        results = {}
        
        for category, paths in paths_by_category.items():
            # Skip if no paths in this category
            if not paths:
                continue
                
            # Extract relevant data
            reff_values = [p.get('reff', 0) for p in paths]
            degen_values = [p.get('degen', 1) for p in paths]
            
            # Calculate statistics
            stats = {
                'count': len(paths),
                'reff': {
                    'min': min(reff_values),
                    'max': max(reff_values),
                    'mean': np.mean(reff_values),
                    'median': np.median(reff_values),
                    'std': np.std(reff_values),
                    'values': sorted(reff_values)
                },
                'degen': {
                    'min': min(degen_values),
                    'max': max(degen_values),
                    'mean': np.mean(degen_values),
                    'values': sorted(degen_values)
                },
                'paths': paths  # Include all path info for reference
            }
            
            # Group paths by similar distances (potential shells)
            shells = self._identify_coordination_shells(reff_values)
            stats['shells'] = shells
            
            results[category] = stats
        
        # Print summary
        print("\nPath Statistics Summary:")
        print("-" * 60)
        for category, stats in results.items():
            print(f"\nCategory: {category}")
            print(f"  Total paths: {stats['count']}")
            print(f"  Distance (Å): min={stats['reff']['min']:.3f}, max={stats['reff']['max']:.3f}, "
                  f"mean={stats['reff']['mean']:.3f}, std={stats['reff']['std']:.3f}")
            
            print("  Potential coordination shells:")
            for i, shell in enumerate(stats['shells']):
                print(f"    Shell {i+1}: {shell['min']:.3f}-{shell['max']:.3f} Å, "
                      f"center={shell['center']:.3f} Å, count={shell['count']}, std={shell['std']:.3f}")
        
        return results
    
    def _analyze_generic_feff_path(self, file_path):
        """
        Analyze a FEFF path file without assuming a specific category.
        Returns a dictionary with path information and identified category.
        """
        try:
            from larch.xafs import feffpath
            import os
            
            fp_obj = feffpath(filename=file_path)
            
            # Basic path info
            path_info = {
                'path': file_path,
                'filename': os.path.basename(file_path),
                'reff': fp_obj._feffdat.reff,
                'degen': fp_obj._feffdat.degen,
                'nleg': fp_obj._feffdat.nleg,
                'geometry': fp_obj._feffdat.geom,
                'feffpath': fp_obj
            }
            
            # Determine path type
            path_info['path_type'] = 'ss' if fp_obj._feffdat.nleg == 2 else 'ms'
            
            # Identify scatterers
            absorber = None
            scatterers = []
            
            for atom in fp_obj._feffdat.geom:
                if atom[2] == 0:  # This is the absorber
                    absorber = atom[0]
                else:
                    scatterers.append(atom[0])
            
            # Determine category based on absorber and first scatterer
            if absorber and scatterers:
                if 'I' in scatterers:
                    path_info['category'] = f"{absorber}-I"
                elif 'O' in scatterers:
                    path_info['category'] = f"{absorber}-O"
                elif 'S' in scatterers:
                    path_info['category'] = f"{absorber}-S"
                elif 'Cs' in scatterers:
                    path_info['category'] = f"{absorber}-Cs"
                else:
                    path_info['category'] = f"{absorber}-{scatterers[0]}"
            else:
                path_info['category'] = 'unknown'
            
            # Create path label
            if path_info['path_type'] == 'ss':
                path_info['label'] = f"{path_info['category']} SS: {path_info['reff']:.3f}Å"
            else:
                atoms_count = {}
                for atom in scatterers:
                    atoms_count[atom] = atoms_count.get(atom, 0) + 1
                atom_string = '-'.join([f"{cnt}{atom}" for atom, cnt in atoms_count.items()])
                path_info['label'] = f"{absorber}-{atom_string} MS: {path_info['reff']:.3f}Å"
            
            return path_info
        
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return {
                'path': file_path,
                'filename': os.path.basename(file_path),
                'category': 'unknown',
                'path_type': 'unknown',
                'error': str(e)
            }
    
    def _identify_coordination_shells(self, reff_values, tolerance=0.1):
        """
        Group similar distances into potential coordination shells.
        
        Parameters:
            reff_values (list): List of reff values
            tolerance (float): Maximum distance difference to consider part of the same shell
            
        Returns:
            list: List of dictionaries with shell information
        """
        import numpy as np
        
        if not reff_values:
            return []
        
        # Sort distances
        sorted_reff = sorted(reff_values)
        
        # Initialize shells
        shells = []
        current_shell = {'values': [sorted_reff[0]], 'min': sorted_reff[0], 'max': sorted_reff[0]}
        
        # Group distances into shells
        for reff in sorted_reff[1:]:
            # If this distance is within tolerance of the current shell's max, add to shell
            if reff - current_shell['max'] <= tolerance:
                current_shell['values'].append(reff)
                current_shell['max'] = reff
            else:
                # Finalize current shell and start a new one
                current_shell['center'] = np.mean(current_shell['values'])
                current_shell['count'] = len(current_shell['values'])
                current_shell['std'] = np.std(current_shell['values'])
                shells.append(current_shell)
                
                # Start new shell
                current_shell = {'values': [reff], 'min': reff, 'max': reff}
        
        # Add the last shell
        current_shell['center'] = np.mean(current_shell['values'])
        current_shell['count'] = len(current_shell['values'])
        current_shell['std'] = np.std(current_shell['values'])
        shells.append(current_shell)
        
        return shells