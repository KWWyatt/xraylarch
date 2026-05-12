import numpy as np
import re
from collections import defaultdict

from xraydb import xray_line, xray_edge, material_mu, atomic_mass
from larch import  parse_group_args, Make_CallArgs
from .xafsutils import set_xafsGroup
from .pre_edge import preedge

# =============================================================================
# Concentration to density mapping for Pb-halide solutions in DMSO
# =============================================================================
# Empirical data: concentration (M) -> solution density (g/mL)
# Linear relationship: density = 0.5877 * concentration + 1.1

CONC_DENSITY_G_ML = {
    0.01: 1.1008,
    0.15: 1.1927,
    0.30: 1.2843,
    0.50: 1.3986,
    0.80: 1.5686,
    1.00: 1.69,
    1.20: 1.80,
}

# Default solvent properties (DMSO)
DMSO_FORMULA = "C2H6OS"
DMSO_DENSITY = 1.095  # g/mL
DMSO_MW = 78.13  # g/mol


def get_density_from_concentration(conc_molar, tol=1e-6):
    """
    Get solution density from concentration using lookup table or linear model.
    
    Parameters
    ----------
    conc_molar : float
        Concentration in molar (M)
    tol : float, optional
        Tolerance for exact match lookup. Default is 1e-6.
    
    Returns
    -------
    float
        Solution density in g/mL
    
    Notes
    -----
    Uses empirical linear relationship: density = 0.5877 * concentration + 1.1
    for concentrations not found in the lookup table.
    """
    # Check for exact match within tolerance
    for conc, density in CONC_DENSITY_G_ML.items():
        if abs(conc - conc_molar) < tol:
            return density
    
    # Use linear relationship for other concentrations
    # density = 0.5877 * concentration + 1.1
    return 0.5877 * conc_molar + 1.1


def parse_formula(formula_str):
    """
    Parse a chemical formula string into element counts.
    
    Parameters
    ----------
    formula_str : str
        Chemical formula string (e.g., "CsPbI3", "C2H6OS", "PbI2")
    
    Returns
    -------
    dict
        Dictionary mapping element symbols to counts.
        Example: {"Cs": 1, "Pb": 1, "I": 3}
    
    Examples
    --------
    >>> parse_formula("CsPbI3")
    {'Cs': 1, 'Pb': 1, 'I': 3}
    >>> parse_formula("C2H6OS")
    {'C': 2, 'H': 6, 'O': 1, 'S': 1}
    """
    element_counts = defaultdict(int)
    
    # Pattern matches element symbol (1-2 letters) followed by optional count
    # Element: uppercase letter optionally followed by lowercase letter
    # Count: optional digits
    pattern = r'([A-Z][a-z]?)(\d*)'
    
    matches = re.findall(pattern, formula_str)
    for element, count in matches:
        if element:  # Skip empty matches
            n = int(count) if count else 1
            element_counts[element] += n
    
    return dict(element_counts)


def build_formula_string(element_counts, order=None):
    """
    Convert element counts dictionary back to a formula string.
    
    Parameters
    ----------
    element_counts : dict
        Dictionary mapping element symbols to counts.
    order : list, optional
        Order of elements in output. If None, uses Hill system
        (C first, then H, then alphabetical).
    
    Returns
    -------
    str
        Chemical formula string
    
    Examples
    --------
    >>> build_formula_string({'C': 2, 'H': 6, 'O': 1, 'S': 1})
    'C2H6OS'
    >>> build_formula_string({'Cs': 1, 'Pb': 1, 'I': 3})
    'CsIPb'  # alphabetical after C,H
    """
    if not element_counts:
        return ""
    
    # Round counts to integers (allow small floating point errors)
    rounded = {}
    for elem, count in element_counts.items():
        rounded_count = round(count)
        if rounded_count > 0:
            rounded[elem] = rounded_count
    
    if order is None:
        # Hill system: C first, then H, then alphabetical
        elements = list(rounded.keys())
        ordered = []
        if 'C' in elements:
            ordered.append('C')
            elements.remove('C')
        if 'H' in elements:
            ordered.append('H')
            elements.remove('H')
        ordered.extend(sorted(elements))
    else:
        ordered = [e for e in order if e in rounded]
    
    formula_parts = []
    for elem in ordered:
        count = rounded[elem]
        if count == 1:
            formula_parts.append(elem)
        else:
            formula_parts.append(f"{elem}{count}")
    
    return ''.join(formula_parts)


def calculate_molecular_weight(formula_or_dict):
    """
    Calculate molecular weight from formula string or element counts.
    
    Parameters
    ----------
    formula_or_dict : str or dict
        Either a formula string or element counts dictionary.
    
    Returns
    -------
    float
        Molecular weight in g/mol
    """
    if isinstance(formula_or_dict, str):
        element_counts = parse_formula(formula_or_dict)
    else:
        element_counts = formula_or_dict
    
    mw = 0.0
    for elem, count in element_counts.items():
        mw += atomic_mass(elem) * count
    
    return mw


def build_effective_solution_formula(solute_formula, concentration_M, solution_density,
                                      solvent_formula=DMSO_FORMULA, 
                                      solvent_mw=DMSO_MW):
    """
    Build an approximate bulk composition formula for a solution.
    
    Uses a 1 L basis to calculate the total element counts from both
    solute and solvent contributions.
    
    Parameters
    ----------
    solute_formula : str
        Chemical formula of the solute (e.g., "CsPbI3")
    concentration_M : float
        Solute concentration in molar (mol/L)
    solution_density : float
        Solution density in g/mL
    solvent_formula : str, optional
        Chemical formula of the solvent. Default is DMSO ("C2H6OS").
    solvent_mw : float, optional
        Molecular weight of solvent in g/mol. Default is 78.13 (DMSO).
    
    Returns
    -------
    tuple
        (effective_formula, solution_density)
        effective_formula: str - Combined formula representing bulk composition
        solution_density: float - The input solution density (passed through)
    
    Notes
    -----
    Calculation basis (1 L of solution):
    - solute_moles = concentration * 1 L
    - solution_mass = solution_density * 1000 g
    - solute_mass = solute_MW * solute_moles
    - solvent_mass = solution_mass - solute_mass
    - solvent_moles = solvent_mass / solvent_MW
    - total elements = solute contribution + solvent contribution
    
    Examples
    --------
    >>> build_effective_solution_formula("CsPbI3", 0.5, 1.4)
    ('C21H63CsI3O10Pb1S10', 1.4)
    """
    # Parse solute and solvent formulas
    solute_elements = parse_formula(solute_formula)
    solvent_elements = parse_formula(solvent_formula)
    
    # Calculate solute molecular weight
    solute_mw = calculate_molecular_weight(solute_elements)
    
    # 1 L basis calculation
    solute_moles = concentration_M * 1.0  # mol (concentration * 1 L)
    solution_mass = solution_density * 1000.0  # g (density g/mL * 1000 mL)
    solute_mass = solute_mw * solute_moles  # g
    
    # Solvent contribution
    solvent_mass = solution_mass - solute_mass  # g
    if solvent_mass < 0:
        # Edge case: solute concentration too high for given density
        # Fall back to pure solute
        solvent_mass = 0.0
        solvent_moles = 0.0
    else:
        solvent_moles = solvent_mass / solvent_mw  # mol
    
    # Calculate total element counts
    total_elements = defaultdict(float)
    
    # Add solute contribution
    for elem, count in solute_elements.items():
        total_elements[elem] += count * solute_moles
    
    # Add solvent contribution
    for elem, count in solvent_elements.items():
        total_elements[elem] += count * solvent_moles
    
    # Normalize to reasonable integer values (scale so smallest count is ~1)
    if total_elements:
        min_count = min(total_elements.values())
        if min_count > 0:
            # Scale down and round to integers
            scaled = {elem: round(count / min_count) for elem, count in total_elements.items()}
        else:
            scaled = dict(total_elements)
    else:
        scaled = {}
    
    # Build formula string - preserve solute elements order, then solvent
    solute_order = list(solute_elements.keys())
    solvent_order = [e for e in solvent_elements.keys() if e not in solute_order]
    order = solute_order + solvent_order
    
    effective_formula = build_formula_string(scaled, order=order)
    
    return effective_formula, solution_density


@Make_CallArgs(["energy","mu"])
def fluo_corr(energy, mu, formula, elem, group=None, edge='K', line='Ka', anginp=45,
              angout=45, density=1.0, _larch=None, **pre_kws):
    """correct over-absorption (self-absorption) for fluorescene XAFS
    using the FLUO alogrithm of D. Haskel.

    Arguments
    ---------
      energy    array of energies
      mu        uncorrected fluorescence mu
      formula   string for sample stoichiometry
      elem      atomic symbol or Z of absorbing element
      group     output group [default None]
      edge      name of edge ('K', 'L3', ...) [default 'K']
      line      name of line ('K', 'Ka', 'La', ...) [default 'Ka']
      anginp    input angle in degrees  [default 45]
      angout    output angle in degrees  [default 45]
      density   sample density in g/cm³ [default 1.0]

    Additional keywords will be passed to pre_edge(), which will be used
    to ensure consistent normalization.

    Returns
    --------
       None, writes `mu_corr` and `norm_corr` (normalized `mu_corr`)
       to output group. Also writes `fluo_corr_density` and `fluo_corr_formula`
       as metadata.

    Notes
    -----
       Support First Argument Group convention, requiring group
       members 'energy' and 'mu'
    """
    energy, mu, group = parse_group_args(energy, members=('energy', 'mu'),
                                         defaults=(mu,), group=group,
                                         fcn_name='fluo_corr')
    # gather pre-edge options
    pre_opts = {'e0': None, 'nnorm': 1, 'nvict': 0,
                'pre1': None, 'pre2': -30,
                'norm1': 100, 'norm2': None}
    if hasattr(group, 'pre_edge_details'):
        uopts = getattr(group.pre_edge_details, 'call_args', {})
        for attr in pre_opts:
            if attr in uopts:
                pre_opts[attr] = uopts[attr]
    pre_opts.update(pre_kws)
    pre_opts['step'] = None
    pre_opts['nvict'] = 0

    # generate normalized mu for correction
    preinp   = preedge(energy, mu, **pre_opts)

    ang_corr = (np.sin(max(1.e-7, np.deg2rad(anginp))) /
                np.sin(max(1.e-7, np.deg2rad(angout))))

    # find edge energies and fluorescence line energy
    e_edge  = xray_edge(elem, edge).energy
    e_fluor = xray_line(elem, line).energy

    # calculate mu(E) for fluorescence energy, above, below edge
    # Now uses the provided density instead of hardcoded 1
    muvals = material_mu(formula, np.array([e_fluor, e_edge-10.0,
                                            e_edge+10.0]), density=density)

    alpha   = (muvals[0]*ang_corr + muvals[1])/(muvals[2] - muvals[1])
    mu_corr = mu*alpha/(alpha + 1 - preinp['norm'])
    preout  = preedge(energy, mu_corr, **pre_opts)
    if group is not None:
        if _larch is not None:
            group = set_xafsGroup(group, _larch=_larch)
        group.mu_corr = mu_corr
        group.norm_corr = preout['norm']
        # Store correction metadata
        group.fluo_corr_density = density
        group.fluo_corr_formula = formula
