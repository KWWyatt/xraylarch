from pathlib import Path


def save_fit_data_and_parameters(fit_result, group_name, output_dir="fit_results",
                                save_data=True, save_params=True, save_report=True):
    """
    Save fit data and parameters to text files.
    
    Parameters:
    -----------
    fit_result : feffit result object
        The result from feffit() function
    group_name : str
        Name of the group/sample for file naming
    output_dir : str
        Directory to save the files (default: "fit_results")
    save_data : bool
        Whether to save the fit data (k, chi_data, chi_fit, r, chir_data, chir_fit)
    save_params : bool
        Whether to save the refined parameters
    save_report : bool
        Whether to save the feffit report
    
    Returns:
    --------
    dict : Dictionary with file paths of saved files
    """
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    saved_files = {}
    
    # Get the transform group from the fit result
    transform_group = fit_result.transform
    
    if save_data:
        # Save fit data (k-space and R-space)
        data_filename = output_path / f"{group_name}_fit_data.txt"
        
        # Get k-space data
        k = transform_group.k
        chi_data = transform_group.chi
        chi_fit = transform_group.chifit
        
        # Get R-space data
        r = transform_group.r
        chir_data = transform_group.chir
        chir_fit = transform_group.chirfit
        
        # Save to file
        with open(data_filename, 'w') as f:
            f.write(f"# EXAFS Fit Data for {group_name}\n")
            f.write("# Generated from feffit analysis\n")
            f.write("#\n")
            f.write("# k-space data (columns: k, chi_data, chi_fit)\n")
            f.write("# R-space data (columns: r, chir_data, chir_fit)\n")
            f.write("#\n")
            f.write("# k-space data:\n")
            f.write("# k(A^-1)    chi_data    chi_fit\n")
            
            for i in range(len(k)):
                f.write(f"{k[i]:10.6f} {chi_data[i]:12.6e} {chi_fit[i]:12.6e}\n")
            
            f.write("\n# R-space data:\n")
            f.write("# r(A)       chir_data   chir_fit\n")
            
            for i in range(len(r)):
                f.write(f"{r[i]:10.6f} {chir_data[i]:12.6e} {chir_fit[i]:12.6e}\n")
        
        saved_files['data'] = str(data_filename)
        print(f"Fit data saved to: {data_filename}")
    
    if save_params:
        # Save refined parameters
        params_filename = output_path / f"{group_name}_refined_parameters.txt"
        
        with open(params_filename, 'w') as f:
            f.write(f"# Refined Parameters for {group_name}\n")
            f.write("# Generated from feffit analysis\n")
            f.write("#\n")
            f.write("# Fit Quality Metrics:\n")
            f.write(f"chi_square: {fit_result.chi_square:.6f}\n")
            f.write(f"rfactor: {fit_result.rfactor:.6f}\n")
            
            if hasattr(fit_result, 'reduced_chi_square'):
                f.write(f"reduced_chi_square: {fit_result.reduced_chi_square:.6f}\n")
            elif hasattr(fit_result, 'redchi'):
                f.write(f"reduced_chi_square: {fit_result.redchi:.6f}\n")
            
            f.write("#\n")
            f.write("# Refined Parameters:\n")
            f.write("# Parameter    Value    Uncertainty    Vary\n")
            
            # Get all parameters
            for param_name in fit_result.params:
                param = fit_result.params[param_name]
                value = param.value
                stderr = getattr(param, 'stderr', 'N/A')
                vary = param.vary
                
                if stderr == 'N/A':
                    f.write(f"{param_name:12s} {value:10.6f} {stderr:>12s} {vary:>6s}\n")
                else:
                    f.write(f"{param_name:12s} {value:10.6f} {stderr:10.6f} {vary:>6s}\n")
            
            f.write(f"\n# Correlation Matrix:\n")
            f.write(f"# (if available)\n")
            
            # Try to get correlation matrix
            if hasattr(fit_result, 'correl'):
                correl = fit_result.correl
                param_names = list(fit_result.params.keys())
                
                f.write(f"#           ")
                for name in param_names:
                    f.write(f"{name:>10s} ")
                f.write(f"\n")
                
                for i, name1 in enumerate(param_names):
                    f.write(f"{name1:12s} ")
                    for j, name2 in enumerate(param_names):
                        if i < len(correl) and j < len(correl[i]):
                            f.write(f"{correl[i][j]:10.3f} ")
                        else:
                            f.write(f"{'N/A':>10s} ")
                    f.write(f"\n")
        
        saved_files['parameters'] = str(params_filename)
        print(f"Refined parameters saved to: {params_filename}")
    
    if save_report:
        # Save feffit report
        report_filename = output_path / f"{group_name}_feffit_report.txt"
        
        with open(report_filename, 'w') as f:
            f.write(f"# FEFFIT Report for {group_name}\n")
            f.write(f"# Generated from feffit analysis\n")
            f.write(f"#\n")
            
            # Get the feffit report as string
            from larch.xafs import feffit_report
            report_text = feffit_report(fit_result)
            f.write(report_text)
        
        saved_files['report'] = str(report_filename)
        print(f"FEFFIT report saved to: {report_filename}")
    
    return saved_files

def save_batch_fit_results(projects, fit_results, output_dir="batch_fit_results"):
    """
    Save fit results for multiple projects in batch mode.
    
    Parameters:
    -----------
    projects : dict
        Dictionary of projects from batch_process_athena_projects()
    fit_results : dict
        Dictionary of fit results for each project
    output_dir : str
        Directory to save the files (default: "batch_fit_results")
    
    Returns:
    --------
    dict : Dictionary with file paths for each project
    """
    
    all_saved_files = {}
    
    for group_name, fit_result in fit_results.items():
        if fit_result is not None:
            saved_files = save_fit_data_and_parameters(
                fit_result, group_name, output_dir
            )
            all_saved_files[group_name] = saved_files
    
    return all_saved_files

def save_fit_summary_table(projects, fit_results, output_dir="fit_results"):
    """
    Save a summary table of all fit results.
    
    Parameters:
    -----------
    projects : dict
        Dictionary of projects from batch_process_athena_projects()
    fit_results : dict
        Dictionary of fit results for each project
    output_dir : str
        Directory to save the summary file (default: "fit_results")
    
    Returns:
    --------
    str : Path to the summary file
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    summary_filename = output_path / "fit_summary_table.txt"
    
    with open(summary_filename, 'w') as f:
        f.write(f"# EXAFS Fit Summary Table\n")
        f.write(f"# Generated from batch feffit analysis\n")
        f.write(f"#\n")
        f.write(f"# Sample    chi_square    rfactor    reduced_chi_square    So2    E0\n")
        f.write(f"#\n")
        
        for group_name, fit_result in fit_results.items():
            if fit_result is not None:
                chi_square = fit_result.chi_square
                rfactor = fit_result.rfactor
                
                if hasattr(fit_result, 'reduced_chi_square'):
                    red_chi = fit_result.reduced_chi_square
                elif hasattr(fit_result, 'redchi'):
                    red_chi = fit_result.redchi
                else:
                    red_chi = 'N/A'
                
                # Try to get So2 and E0 from parameters
                so2 = 'N/A'
                e0 = 'N/A'
                
                if hasattr(fit_result, 'params'):
                    if 'so2' in fit_result.params:
                        so2 = f"{fit_result.params['so2'].value:.4f}"
                    if 'e0' in fit_result.params:
                        e0 = f"{fit_result.params['e0'].value:.2f}"
                
                f.write(f"{group_name:12s} {chi_square:10.6f} {rfactor:10.6f} {red_chi:>18s} {so2:>6s} {e0:>6s}\n")
    
    print(f"Fit summary table saved to: {summary_filename}")
    return str(summary_filename)
