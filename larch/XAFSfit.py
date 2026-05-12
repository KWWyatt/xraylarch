from pathlib import Path
try:
    from .XASmu2r import XASmu2r
except ImportError:
    from XASmu2r import XASmu2r


class XAFSfit(XASmu2r):
    """
    Fitting-focused class that loads previously processed XAFS/XAS checkpoint data.

    Initialize with the ``processed`` directory produced by
    ``XASmu2r.save_processed_checkpoint()`` or ``XASmu2r.save_processed_after_ft()``.
    All inherited fitting methods operate on the loaded ``self.projects`` structure.
    """

    def __init__(self, processed_dir, checkpoint_filename='processed_checkpoint.pkl', verbose=True):
        self.prj_folder = Path(processed_dir)
        self.projects = {}
        self.processed_dir = Path(processed_dir)
        self.checkpoint_filename = checkpoint_filename
        self.load_processed_data(processed_dir=self.processed_dir, checkpoint_filename=checkpoint_filename, verbose=verbose)

    def process_athena_projects(self):
        """Disabled for XAFSfit. Use load_processed_data instead."""
        raise RuntimeError('XAFSfit does not load .prj files directly. Use load_processed_data().')

    def load_processed_data(self, processed_dir=None, checkpoint_filename='processed_checkpoint.pkl', verbose=True):
        """Load processed checkpoint data created by XASmu2r.save_processed_checkpoint()."""
        processed_dir = Path(processed_dir) if processed_dir is not None else self.processed_dir
        self.processed_dir = processed_dir
        self.checkpoint_filename = checkpoint_filename
        return self.load_processed_checkpoint(processed_dir=processed_dir, filename=checkpoint_filename, verbose=verbose)

    def reload(self, verbose=True):
        """Reload the current processed checkpoint from disk."""
        return self.load_processed_data(processed_dir=self.processed_dir, checkpoint_filename=self.checkpoint_filename, verbose=verbose)

    def show_integrated_fit_paths(self, latest_only=True, include_shared_e0=True, verbose=True):
        """
        Summarize which FEFF paths are currently integrated into fitting for each group.

        Parameters
        ----------
        latest_only : bool
            If True, report only the most recent stored path set per group. If False,
            report every stored ``paths_iterN`` / ``paths_iterN_shared_e0`` set.
        include_shared_e0 : bool
            If True, include ``_shared_e0`` variants when present. When ``latest_only``
            is also True, a shared-e0 result at the same iteration is preferred because
            it typically reflects the more finalized fit state.
        verbose : bool
            If True, print a readable summary to the console.

        Returns
        -------
        dict
            Nested summary keyed by ``project.group`` with the selected path sets and
            per-path metadata.
        """
        import re

        def collect_path_sets(group):
            path_sets = []
            for attr_name in vars(group):
                match = re.fullmatch(r'paths_iter(\d+)(?:(_shared_e0))?', attr_name)
                if not match:
                    continue

                iteration = int(match.group(1))
                is_shared_e0 = bool(match.group(2))
                if is_shared_e0 and not include_shared_e0:
                    continue

                paths = getattr(group, attr_name, None)
                if not paths:
                    continue

                fit_attr = f"fit_result_iter{iteration}" + ("_shared_e0" if is_shared_e0 else "")
                fit_result = getattr(group, fit_attr, None)

                path_sets.append({
                    'attr_name': attr_name,
                    'iteration': iteration,
                    'shared_e0': is_shared_e0,
                    'fit_attr': fit_attr,
                    'fit_result': fit_result,
                    'path_count': len(paths),
                    'paths': paths,
                })

            path_sets.sort(key=lambda item: (item['iteration'], item['shared_e0']))
            return path_sets

        def select_path_sets(path_sets):
            if not latest_only or not path_sets:
                return path_sets
            return [max(path_sets, key=lambda item: (item['iteration'], item['shared_e0']))]

        summary = {
            'projects': {},
            'groups': {},
            'group_count': 0,
            'groups_with_paths': 0,
        }

        for proj_name, project in self.projects.items():
            project_summary = summary['projects'].setdefault(proj_name, {'groups': {}})
            for group_name, group in project.groups.items():
                summary['group_count'] += 1
                group_id = f"{proj_name}.{group_name}"

                path_sets = select_path_sets(collect_path_sets(group))
                group_summary = {
                    'project': proj_name,
                    'group_name': group_name,
                    'group_id': group_id,
                    'path_sets': [],
                }

                for path_set in path_sets:
                    path_details = []
                    for index, path in enumerate(path_set['paths'], start=1):
                        path_info = self.identify_path_elements(path)
                        path_details.append({
                            'index': index,
                            'label': getattr(path, 'label', f'Path {index}'),
                            'role': self._infer_path_role(path),
                            'param_type': getattr(path, 'param_type', None),
                            'reff': getattr(path, 'reff', None),
                            'primary': path_info.get('primary'),
                            'scatterers': path_info.get('all_scatterers', []),
                            'is_ms': path_info.get('is_ms', False),
                            'nleg': path_info.get('nleg'),
                        })

                    group_summary['path_sets'].append({
                        'attr_name': path_set['attr_name'],
                        'iteration': path_set['iteration'],
                        'shared_e0': path_set['shared_e0'],
                        'fit_attr': path_set['fit_attr'],
                        'rfactor': getattr(path_set['fit_result'], 'rfactor', None),
                        'path_count': path_set['path_count'],
                        'paths': path_details,
                    })

                if group_summary['path_sets']:
                    summary['groups_with_paths'] += 1

                project_summary['groups'][group_name] = group_summary
                summary['groups'][group_id] = group_summary

        if verbose:
            print("\n" + "=" * 80)
            print("INTEGRATED FIT PATHS")
            print("=" * 80)
            print(f"  Groups scanned:      {summary['group_count']}")
            print(f"  Groups with paths:   {summary['groups_with_paths']}")
            print(f"  Latest only:         {latest_only}")
            print(f"  Include shared e0:   {include_shared_e0}")
            print("-" * 80)

            for group_id, group_summary in summary['groups'].items():
                print(f"\n{group_id}")
                if not group_summary['path_sets']:
                    print("  No stored integrated fit paths found.")
                    continue

                for path_set in group_summary['path_sets']:
                    mode_label = "shared_e0" if path_set['shared_e0'] else "standard"
                    rfactor = path_set['rfactor']
                    rfactor_text = f"{rfactor:.6f}" if rfactor is not None else "n/a"
                    print(
                        f"  {path_set['attr_name']} | iter={path_set['iteration']} | "
                        f"mode={mode_label} | n_paths={path_set['path_count']} | "
                        f"R-factor={rfactor_text}"
                    )

                    for path_info in path_set['paths']:
                        scatterers = ", ".join(path_info['scatterers']) if path_info['scatterers'] else "unknown"
                        reff = path_info['reff']
                        reff_text = f"{reff:.3f} Å" if reff is not None else "n/a"
                        print(
                            f"    {path_info['index']:2d}. {path_info['label']} | "
                            f"role={path_info['role']} | reff={reff_text} | "
                            f"scatterers={scatterers}"
                        )

            print("\n" + "=" * 80)

        return summary

    def plot_fit_2x2_summary(self, figsize=(16, 12), dpi=300, save_path=None,
                             style='seaborn-v0_8-whitegrid', n_groups=3,
                             alpha_range=(1.0, 0.3)):
        """
        Create a 2x2 fit-summary figure for fitted groups.

        Top row:
          1. Coordination number vs concentration with group-dependent transparency
          2. Bond length vs concentration with group-dependent transparency

        Bottom row:
          3. sigma2 vs concentration (Pb-I and Pb-O on dual y-axes)
          4. e0 vs concentration

        Parameters
        ----------
        figsize : tuple
            Figure size in inches.
        dpi : int
            Resolution for saved figure.
        save_path : str or Path or None
            Output path for saving the figure. If None, the figure is only displayed.
        style : str
            Matplotlib style name.
        n_groups : int
            Expected number of groups per concentration for transparency scaling.
        alpha_range : tuple
            (max_alpha, min_alpha) assigned from group 1 to the highest group number.

        Returns
        -------
        fig, axes
            The matplotlib figure and axes tuple.
        """
        import re
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        try:
            plt.style.use(style)
        except Exception:
            try:
                plt.style.use('seaborn-whitegrid')
            except Exception:
                pass

        data_collection = self._collect_fit_data_for_publication()

        if not data_collection['groups']:
            print("No fit data found. Run the fitting workflow before plotting.")
            return None, None

        def extract_group_number(group_name):
            patterns = [
                r'group\s*(\d+)',
                r'group_(\d+)',
                r'Group\s*(\d+)',
                r'grp\s*(\d+)',
                r'g(\d+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, group_name, re.IGNORECASE)
                if match:
                    return int(match.group(1))
            return 1

        def get_alpha_for_group(group_num, n_grps, alpha_rng):
            max_alpha, min_alpha = alpha_rng
            if n_grps <= 1:
                return max_alpha
            alpha = max_alpha - (max_alpha - min_alpha) * (group_num - 1) / (n_grps - 1)
            return np.clip(alpha, min_alpha, max_alpha)

        def mean_by_concentration(groups, value_getter):
            conc_unique = np.unique([g['concentration'] for g in groups])
            conc_unique = np.sort(conc_unique.astype(float))
            means = []
            for conc in conc_unique:
                groups_at_c = [g for g in groups if np.isclose(g['concentration'], conc)]
                means.append(np.mean([value_getter(g) for g in groups_at_c]))
            return conc_unique, np.array(means, dtype=float)

        def set_dynamic_limits(axis, values, errors=None, pad_fraction=0.12, min_span=1e-3):
            arr = np.asarray(values, dtype=float)
            if errors is None:
                err = np.zeros_like(arr)
            else:
                err = np.asarray(errors, dtype=float)
            finite = np.isfinite(arr) & np.isfinite(err)
            if not np.any(finite):
                return
            lo = np.min(arr[finite] - err[finite])
            hi = np.max(arr[finite] + err[finite])
            span = max(hi - lo, min_span)
            pad = max(span * pad_fraction, min_span)
            axis.set_ylim(lo - pad, hi + pad)

        for group_data in data_collection['groups']:
            group_num = extract_group_number(group_data.get('group_name', ''))
            group_data['group_number'] = group_num
            group_data['alpha'] = get_alpha_for_group(group_num, n_groups, alpha_range)

        groups = data_collection['groups']
        concentrations = np.unique(data_collection['concentrations'])
        concentrations = np.sort(concentrations.astype(float))

        color_I = '#8B008B'
        color_O = '#DC143C'
        color_O1 = '#DC143C'   # short Pb-O shell (same red as legacy)
        color_O2 = '#FF8C00'   # long  Pb-O shell (orange)
        color_O_total = '#7F1D1D'  # composite total (dark crimson)
        color_e0 = '#1F2937'

        # Detect whether the split-shell Pb-O model is in use for any group.
        split_groups_present = any(
            ('N_O1' in g and 'N_O2' in g) for g in groups
        )

        fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=100)
        ax_coord = axs[0, 0]
        ax_bond = axs[0, 1]
        ax_sigma2 = axs[1, 0]
        ax_e0 = axs[1, 1]

        ax_coord_twin = ax_coord.twinx()
        ax_bond_twin = ax_bond.twinx()
        ax_sigma2_twin = ax_sigma2.twinx()

        # Top-left: coordination vs concentration
        for group_data in groups:
            conc = group_data['concentration']
            alpha = group_data['alpha']

            ax_coord.errorbar(
                conc, group_data['N_I'], yerr=group_data.get('N_I_err', 0.0),
                fmt='o', markersize=10, color=color_I, markeredgecolor='k',
                markeredgewidth=0.8, capsize=4, capthick=1.5, elinewidth=1.5,
                alpha=alpha, zorder=5
            )
            if split_groups_present and ('N_O1' in group_data) and ('N_O2' in group_data):
                # Plot N_O1 (short shell), N_O2 (long shell), and N_O_total
                ax_coord_twin.errorbar(
                    conc, group_data['N_O1'], yerr=group_data.get('N_O1_err', 0.0),
                    fmt='s', markersize=9, color=color_O1, markeredgecolor='k',
                    markeredgewidth=0.8, capsize=4, capthick=1.2, elinewidth=1.2,
                    alpha=alpha, zorder=6
                )
                ax_coord_twin.errorbar(
                    conc, group_data['N_O2'], yerr=group_data.get('N_O2_err', 0.0),
                    fmt='D', markersize=8, color=color_O2, markeredgecolor='k',
                    markeredgewidth=0.8, capsize=4, capthick=1.2, elinewidth=1.2,
                    alpha=alpha, zorder=6
                )
                ax_coord_twin.errorbar(
                    conc, group_data['N_O'], yerr=group_data.get('N_O_err', 0.0),
                    fmt='^', markersize=10, color=color_O_total, markeredgecolor='k',
                    markeredgewidth=0.8, capsize=4, capthick=1.5, elinewidth=1.5,
                    alpha=alpha, zorder=7
                )
            else:
                ax_coord_twin.errorbar(
                    conc, group_data['N_O'], yerr=group_data.get('N_O_err', 0.0),
                    fmt='s', markersize=10, color=color_O, markeredgecolor='k',
                    markeredgewidth=0.8, capsize=4, capthick=1.5, elinewidth=1.5,
                    alpha=alpha, zorder=5
                )

        conc_mean, N_I_mean = mean_by_concentration(groups, lambda g: g['N_I'])
        ax_coord.plot(conc_mean, N_I_mean, '--', color=color_I, alpha=0.5, lw=2)
        if split_groups_present:
            _, N_O1_mean = mean_by_concentration(
                groups, lambda g: g.get('N_O1', g.get('N_O', 0.0)))
            _, N_O2_mean = mean_by_concentration(
                groups, lambda g: g.get('N_O2', 0.0))
            _, N_O_total_mean = mean_by_concentration(groups, lambda g: g['N_O'])
            ax_coord_twin.plot(conc_mean, N_O1_mean, '--', color=color_O1, alpha=0.5, lw=1.5)
            ax_coord_twin.plot(conc_mean, N_O2_mean, '--', color=color_O2, alpha=0.5, lw=1.5)
            ax_coord_twin.plot(conc_mean, N_O_total_mean, '-', color=color_O_total, alpha=0.6, lw=2.2)
        else:
            _, N_O_mean = mean_by_concentration(groups, lambda g: g['N_O'])
            ax_coord_twin.plot(conc_mean, N_O_mean, '--', color=color_O, alpha=0.5, lw=2)
        ax_coord.set_xlabel('Concentration (M)', fontsize=12, fontweight='bold')
        ax_coord.set_ylabel('$N_{Pb-I}$', fontsize=12, fontweight='bold', color=color_I)
        if split_groups_present:
            ax_coord_twin.set_ylabel('$N_{Pb-O}$ (O1, O2, total)', fontsize=12,
                                     fontweight='bold', color=color_O_total)
            ax_coord_twin.tick_params(axis='y', labelcolor=color_O_total, labelsize=10)
        else:
            ax_coord_twin.set_ylabel('$N_{Pb-O}$', fontsize=12, fontweight='bold', color=color_O)
            ax_coord_twin.tick_params(axis='y', labelcolor=color_O, labelsize=10)
        ax_coord.tick_params(axis='y', labelcolor=color_I, labelsize=10)
        ax_coord.tick_params(axis='x', labelsize=10)
        ax_coord.grid(True, alpha=0.3, linestyle='--')
        ax_coord.set_xlim(left=0)
        ax_coord.set_ylim(0, 6)
        ax_coord_twin.set_ylim(0, 8 if split_groups_present else 6)
        ax_coord.set_title('Coordination Number vs. Concentration', fontsize=12, fontweight='bold')

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_I,
                   markeredgecolor='k', markersize=10, label='$N_{Pb-I}$', alpha=1.0),
        ]
        if split_groups_present:
            legend_elements += [
                Line2D([0], [0], marker='s', color='w', markerfacecolor=color_O1,
                       markeredgecolor='k', markersize=9, label='$N_{Pb-O_1}$ (short)', alpha=1.0),
                Line2D([0], [0], marker='D', color='w', markerfacecolor=color_O2,
                       markeredgecolor='k', markersize=8, label='$N_{Pb-O_2}$ (long)', alpha=1.0),
                Line2D([0], [0], marker='^', color='w', markerfacecolor=color_O_total,
                       markeredgecolor='k', markersize=10,
                       label='$N_{Pb-O}^{total}$', alpha=1.0),
            ]
        else:
            legend_elements.append(
                Line2D([0], [0], marker='s', color='w', markerfacecolor=color_O,
                       markeredgecolor='k', markersize=10, label='$N_{Pb-O}$', alpha=1.0)
            )
        for i in range(1, n_groups + 1):
            alpha_val = get_alpha_for_group(i, n_groups, alpha_range)
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       markeredgecolor='k', markersize=8, alpha=alpha_val,
                       label=f'Group {i} (alpha={alpha_val:.2f})')
            )
        ax_coord.legend(handles=legend_elements, loc='upper right', fontsize=9,
                        framealpha=0.9, edgecolor='gray')

        # Top-right: bond length vs concentration
        for group_data in groups:
            conc = group_data['concentration']
            alpha = group_data['alpha']
            bond_I = group_data['reff_I'] + group_data.get('delr_I', 0.0)
            bond_O = group_data['reff_O'] + group_data.get('delr_O', 0.0)

            ax_bond.errorbar(
                conc, bond_I, yerr=group_data.get('delr_I_err', 0.0),
                fmt='o', markersize=10, color=color_I, markeredgecolor='k',
                markeredgewidth=0.8, capsize=4, capthick=1.5, elinewidth=1.5,
                alpha=alpha, zorder=5
            )
            if split_groups_present and ('reff_O1' in group_data) and ('reff_O2' in group_data):
                bond_O1 = (group_data['reff_O1']
                           + group_data.get('delr_O1', group_data.get('delr_O', 0.0)))
                bond_O2 = (group_data['reff_O2']
                           + group_data.get('delr_O1', group_data.get('delr_O', 0.0))
                           + group_data.get('split_O', 0.0)
                           - (group_data['reff_O2'] - group_data['reff_O1']))
                # bond_O2 = reff_O2 + delr_O2_effective. The split-shell model
                # parameterises delr_O2 = (reff_O1 - reff_O2) + delr_O1 + split_O,
                # so bond_O2 simplifies to reff_O1 + delr_O1 + split_O.
                bond_O2 = (group_data['reff_O1']
                           + group_data.get('delr_O1', group_data.get('delr_O', 0.0))
                           + group_data.get('split_O', 0.0))
                ax_bond_twin.errorbar(
                    conc, bond_O1, yerr=group_data.get('delr_O1_err',
                                                       group_data.get('delr_O_err', 0.0)),
                    fmt='s', markersize=9, color=color_O1, markeredgecolor='k',
                    markeredgewidth=0.8, capsize=4, capthick=1.2, elinewidth=1.2,
                    alpha=alpha, zorder=6
                )
                ax_bond_twin.errorbar(
                    conc, bond_O2, yerr=group_data.get('split_O_err', 0.0),
                    fmt='D', markersize=8, color=color_O2, markeredgecolor='k',
                    markeredgewidth=0.8, capsize=4, capthick=1.2, elinewidth=1.2,
                    alpha=alpha, zorder=6
                )
            else:
                ax_bond_twin.errorbar(
                    conc, bond_O, yerr=group_data.get('delr_O_err', 0.0),
                    fmt='s', markersize=10, color=color_O, markeredgecolor='k',
                    markeredgewidth=0.8, capsize=4, capthick=1.5, elinewidth=1.5,
                    alpha=alpha, zorder=5
                )

        _, bond_I_mean = mean_by_concentration(groups, lambda g: g['reff_I'] + g.get('delr_I', 0.0))
        ax_bond.plot(conc_mean, bond_I_mean, '--', color=color_I, alpha=0.5, lw=2)
        if split_groups_present:
            _, bond_O1_mean = mean_by_concentration(
                groups,
                lambda g: g.get('reff_O1', g['reff_O']) + g.get('delr_O1', g.get('delr_O', 0.0))
            )
            _, bond_O2_mean = mean_by_concentration(
                groups,
                lambda g: g.get('reff_O1', g['reff_O']) + g.get('delr_O1', g.get('delr_O', 0.0)) + g.get('split_O', 0.0)
            )
            ax_bond_twin.plot(conc_mean, bond_O1_mean, '--', color=color_O1, alpha=0.5, lw=1.5)
            ax_bond_twin.plot(conc_mean, bond_O2_mean, '--', color=color_O2, alpha=0.5, lw=1.5)
        else:
            _, bond_O_mean = mean_by_concentration(groups, lambda g: g['reff_O'] + g.get('delr_O', 0.0))
            ax_bond_twin.plot(conc_mean, bond_O_mean, '--', color=color_O, alpha=0.5, lw=2)
        ax_bond.set_xlabel('Concentration (M)', fontsize=12, fontweight='bold')
        ax_bond.set_ylabel(r'Pb-I Bond Length ($\AA$)', fontsize=12, fontweight='bold', color=color_I)
        if split_groups_present:
            ax_bond_twin.set_ylabel(r'Pb-O Bond Length ($\AA$) [O$_1$, O$_2$]',
                                    fontsize=12, fontweight='bold', color=color_O_total)
            ax_bond_twin.tick_params(axis='y', labelcolor=color_O_total, labelsize=10)
        else:
            ax_bond_twin.set_ylabel(r'Pb-O Bond Length ($\AA$)', fontsize=12, fontweight='bold', color=color_O)
            ax_bond_twin.tick_params(axis='y', labelcolor=color_O, labelsize=10)
        ax_bond.tick_params(axis='y', labelcolor=color_I, labelsize=10)
        ax_bond.tick_params(axis='x', labelsize=10)
        ax_bond.grid(True, alpha=0.25)
        ax_bond.set_xlim(left=0)
        ax_bond.set_title('Bond Length vs. Concentration', fontsize=12, fontweight='bold')
        set_dynamic_limits(
            ax_bond,
            [g['reff_I'] + g.get('delr_I', 0.0) for g in groups],
            [g.get('delr_I_err', 0.0) for g in groups]
        )
        if split_groups_present:
            bond_O_all = []
            bond_O_errs = []
            for g in groups:
                base = g.get('reff_O1', g['reff_O']) + g.get('delr_O1', g.get('delr_O', 0.0))
                err1 = g.get('delr_O1_err', g.get('delr_O_err', 0.0))
                err2 = g.get('split_O_err', g.get('delr_O_err', 0.0))
                bond_O_all.append(base)
                bond_O_errs.append(err1)
                bond_O_all.append(base + g.get('split_O', 0.0))
                bond_O_errs.append(err2)
            set_dynamic_limits(ax_bond_twin, bond_O_all, bond_O_errs)
        else:
            set_dynamic_limits(
                ax_bond_twin,
                [g['reff_O'] + g.get('delr_O', 0.0) for g in groups],
                [g.get('delr_O_err', 0.0) for g in groups]
            )

        # Bottom-left: sigma2 vs concentration
        for group_data in groups:
            conc = group_data['concentration']
            alpha = group_data['alpha']

            ax_sigma2.errorbar(
                conc, group_data.get('sigma2_I', 0.0), yerr=group_data.get('sigma2_I_err', 0.0),
                fmt='o', markersize=10, color=color_I, markeredgecolor='k',
                markeredgewidth=0.8, capsize=4, capthick=1.5, elinewidth=1.5,
                alpha=alpha, zorder=5
            )
            ax_sigma2_twin.errorbar(
                conc, group_data.get('sigma2_O', 0.0), yerr=group_data.get('sigma2_O_err', 0.0),
                fmt='s', markersize=10, color=color_O, markeredgecolor='k',
                markeredgewidth=0.8, capsize=4, capthick=1.5, elinewidth=1.5,
                alpha=alpha, zorder=5
            )

        _, sigma2_I_mean = mean_by_concentration(groups, lambda g: g.get('sigma2_I', 0.0))
        _, sigma2_O_mean = mean_by_concentration(groups, lambda g: g.get('sigma2_O', 0.0))
        ax_sigma2.plot(conc_mean, sigma2_I_mean, '--', color=color_I, alpha=0.5, lw=2)
        ax_sigma2_twin.plot(conc_mean, sigma2_O_mean, '--', color=color_O, alpha=0.5, lw=2)
        ax_sigma2.set_xlabel('Concentration (M)', fontsize=12, fontweight='bold')
        ax_sigma2.set_ylabel(r'$\sigma^2_{Pb-I}$ ($\AA^2$)', fontsize=12, fontweight='bold', color=color_I)
        ax_sigma2_twin.set_ylabel(r'$\sigma^2_{Pb-O}$ ($\AA^2$)', fontsize=12, fontweight='bold', color=color_O)
        ax_sigma2.tick_params(axis='y', labelcolor=color_I, labelsize=10)
        ax_sigma2_twin.tick_params(axis='y', labelcolor=color_O, labelsize=10)
        ax_sigma2.tick_params(axis='x', labelsize=10)
        ax_sigma2.grid(True, alpha=0.25)
        ax_sigma2.set_xlim(left=0)
        ax_sigma2.set_title(r'$\sigma^2$ vs. Concentration', fontsize=12, fontweight='bold')
        if split_groups_present:
            ax_sigma2.text(
                0.98, 0.95,
                r'$\sigma^2_{Pb-O}$ fixed at 0.025 $\AA^2$ (split-shell)',
                transform=ax_sigma2.transAxes, ha='right', va='top',
                fontsize=9, color=color_O_total,
                bbox=dict(facecolor='white', edgecolor=color_O_total, alpha=0.85)
            )
        set_dynamic_limits(
            ax_sigma2,
            [g.get('sigma2_I', 0.0) for g in groups],
            [g.get('sigma2_I_err', 0.0) for g in groups]
        )
        set_dynamic_limits(
            ax_sigma2_twin,
            [g.get('sigma2_O', 0.0) for g in groups],
            [g.get('sigma2_O_err', 0.0) for g in groups]
        )

        # Bottom-right: e0 vs concentration
        for group_data in groups:
            ax_e0.errorbar(
                group_data['concentration'], group_data.get('e0', 0.0),
                yerr=group_data.get('e0_err', 0.0),
                fmt='o', markersize=10, color=color_e0, markeredgecolor='k',
                markeredgewidth=0.8, capsize=4, capthick=1.5, elinewidth=1.5,
                alpha=group_data['alpha'], zorder=5
            )

        _, e0_mean = mean_by_concentration(groups, lambda g: g.get('e0', 0.0))
        ax_e0.plot(conc_mean, e0_mean, '--', color=color_e0, alpha=0.55, lw=2)
        ax_e0.set_xlabel('Concentration (M)', fontsize=12, fontweight='bold')
        ax_e0.set_ylabel(r'$e_0$ (eV)', fontsize=12, fontweight='bold', color=color_e0)
        ax_e0.tick_params(axis='y', labelcolor=color_e0, labelsize=10)
        ax_e0.tick_params(axis='x', labelsize=10)
        ax_e0.grid(True, alpha=0.25)
        ax_e0.set_xlim(left=0)
        ax_e0.set_title(r'$e_0$ vs. Concentration', fontsize=12, fontweight='bold')
        set_dynamic_limits(
            ax_e0,
            [g.get('e0', 0.0) for g in groups],
            [g.get('e0_err', 0.0) for g in groups],
            pad_fraction=0.15,
            min_span=0.05
        )

        n_total = len(groups)
        n_conc = len(concentrations)
        fig.suptitle('EXAFS Fit Analysis: Group Variability and Fit Trends',
                     fontsize=14, fontweight='bold', y=0.98)
        metadata_text = (
            f"N = {n_total} total samples | "
            f"{n_conc} concentrations | "
            f"{n_groups} groups per concentration"
        )
        fig.text(0.5, 0.01, metadata_text, ha='center', fontsize=9,
                 style='italic', color='gray')

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Figure saved to: {save_path}")

        plt.show()

        print("\n" + "=" * 80)
        print("PLOT_FIT_2X2_SUMMARY")
        print("=" * 80)
        print(f"  Total groups:      {n_total}")
        print(f"  Concentrations:    {n_conc}")
        print(f"  Figure saved:      {save_path if save_path else 'Not saved'}")
        print("-" * 80)
        print("  Per-concentration summary:")
        for conc in concentrations:
            groups_at_c = [g for g in groups if np.isclose(g['concentration'], conc)]
            avg_N_I = np.mean([g.get('N_I', 0.0) for g in groups_at_c])
            avg_sig2_I = np.mean([g.get('sigma2_I', 0.0) for g in groups_at_c])
            avg_e0 = np.mean([g.get('e0', 0.0) for g in groups_at_c])
            split_at_c = [g for g in groups_at_c if ('N_O1' in g and 'N_O2' in g)]
            if split_at_c and len(split_at_c) == len(groups_at_c):
                avg_N_O1 = np.mean([g['N_O1'] for g in groups_at_c])
                avg_N_O2 = np.mean([g['N_O2'] for g in groups_at_c])
                avg_N_O_total = np.mean([g.get('N_O', g['N_O1'] + g['N_O2'])
                                         for g in groups_at_c])
                avg_sig2_O = np.mean([g.get('sigma2_O', 0.0) for g in groups_at_c])
                print(
                    f"    Conc {conc:.3f} M: {len(groups_at_c)} groups, "
                    f"avg N_I={avg_N_I:.2f}, "
                    f"avg N_Pb-O1={avg_N_O1:.2f}, avg N_Pb-O2={avg_N_O2:.2f}, "
                    f"avg N_Pb-O(total)={avg_N_O_total:.2f}, "
                    f"avg sigma2_I={avg_sig2_I:.5f}, "
                    f"sigma2_Pb-O={avg_sig2_O:.5f} (FIXED), "
                    f"avg e0={avg_e0:.3f}"
                )
            else:
                avg_N_O = np.mean([g.get('N_O', 0.0) for g in groups_at_c])
                avg_sig2_O = np.mean([g.get('sigma2_O', 0.0) for g in groups_at_c])
                print(
                    f"    Conc {conc:.3f} M: {len(groups_at_c)} groups, "
                    f"avg N_I={avg_N_I:.2f}, avg N_O={avg_N_O:.2f}, "
                    f"avg sigma2_I={avg_sig2_I:.5f}, avg sigma2_O={avg_sig2_O:.5f}, "
                    f"avg e0={avg_e0:.3f}"
                )
        print("=" * 80 + "\n")

        return fig, (
            ax_coord, ax_coord_twin,
            ax_bond, ax_bond_twin,
            ax_sigma2, ax_sigma2_twin,
            ax_e0,
        )
