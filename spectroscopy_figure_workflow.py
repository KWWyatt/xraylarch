from __future__ import annotations

import re
import warnings
from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from lmfit import Parameters, minimize


NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def default_uv_vis_folder() -> Path:
    candidates = [
        Path(r"C:\Users\kwill\Keenan_UCB-O365\OneDrive - UCB-O365\Data\UV-vis"),
        Path(r"C:\Users\kwill\Keenan_UCB-O365\OneDrive_UCB-O365\Data\UV-vis"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def parse_concentration(folder_name: str) -> float:
    match = re.search(r"(\d+)p(\d+)M", folder_name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse concentration from {folder_name!r}")
    return float(f"{match.group(1)}.{match.group(2)}")


def extract_value(text: str) -> float:
    match = re.search(NUM_RE, text)
    return float(match.group()) if match else np.nan


def parse_report_text(text: str) -> dict[str, float | str]:
    info: dict[str, float | str] = {}
    cleaned_lines = [line.lstrip("#").rstrip() for line in text.splitlines()]

    stats_map = {
        "r-factor": "r_factor",
        "Akaike info crit": "aic",
        "Bayesian info crit": "bic",
    }
    for raw_line in cleaned_lines:
        stripped = raw_line.strip()
        for key, out_key in stats_map.items():
            if stripped.startswith(key):
                info[out_key] = extract_value(stripped.split("=", 1)[1])

    in_variables = False
    for raw_line in cleaned_lines:
        stripped = raw_line.strip()
        if stripped == "[[Variables]]":
            in_variables = True
            continue
        if in_variables and stripped.startswith("[["):
            in_variables = False
        if not in_variables or "=" not in stripped:
            continue
        match = re.match(
            rf"^([A-Za-z0-9_*]+)\s*=\s*({NUM_RE})(?:\s*\+/-\s*({NUM_RE}))?",
            stripped,
        )
        if not match:
            continue
        key = match.group(1).replace("*", "_star_")
        info[key] = float(match.group(2))
        info[f"{key}_err"] = float(match.group(3)) if match.group(3) else np.nan

    path_sections = re.split(r"(?m)^\s*#?\s*= Path '", text)
    for section in path_sections[1:]:
        label, rest = section.split("'", 1)
        body = "\n".join(line.lstrip("#").rstrip() for line in rest.splitlines())
        if re.match(r"^Pb_I", label):
            prefix = "pb_i"
        elif re.match(r"^Pb_O(?!_)", label):
            prefix = "pb_o"
        else:
            continue
        for source_key, target_key in (("reff", f"{prefix}_reff"), ("r", f"{prefix}_r")):
            match = re.search(rf"(?m)^\s*{re.escape(source_key)}\s*=\s*({NUM_RE})", body)
            if match:
                info[target_key] = float(match.group(1))
        info[f"{prefix}_label"] = label

    if "delr_I308" in info and "pb_i_reff" in info and "pb_i_r" not in info:
        info["pb_i_r"] = float(info["pb_i_reff"]) + float(info["delr_I308"])
    if "delr_O257" in info and "pb_o_reff" in info and "pb_o_r" not in info:
        info["pb_o_r"] = float(info["pb_o_reff"]) + float(info["delr_O257"])
    return info


def is_plot_text(path: Path) -> bool:
    name = path.name.lower()
    return any(tag in name for tag in ("chikw", "chik", "chir", "rechir"))


def get_report_candidates(folder: Path) -> list[tuple[Path, dict[str, float | str]]]:
    report_paths = sorted(folder.glob("*Fit*.txt"))
    if not report_paths:
        report_paths = sorted(path for path in folder.glob("*.txt") if is_plot_text(path))

    candidates: list[tuple[Path, dict[str, float | str]]] = []
    for path in report_paths:
        parsed = parse_report_text(read_text(path))
        if parsed:
            candidates.append((path, parsed))
    return candidates


def score_report_candidate(item: tuple[Path, dict[str, float | str]]) -> tuple[float, float, str]:
    path, parsed = item
    aic = float(parsed.get("aic", np.inf))
    r_factor = float(parsed.get("r_factor", np.inf))
    return (aic, r_factor, path.name)


def choose_report(folder: Path) -> tuple[Path | None, dict[str, float | str]]:
    candidates = get_report_candidates(folder)

    if not candidates:
        return None, {}

    return min(candidates, key=score_report_candidate)


def classify_report_source(report_file: str | None) -> str:
    if not report_file:
        return "unknown"
    lowered = report_file.lower()
    if "constrainedfit" in lowered:
        return "constrained"
    if "relaxedfit" in lowered:
        return "relaxed"
    if is_plot_text(Path(report_file)):
        return "plot_text_fallback"
    return "other"


def parse_fit_variant(folder: Path, variant_tag: str) -> dict[str, float | str]:
    candidates: list[tuple[Path, dict[str, float | str]]] = []
    for path in sorted(folder.glob(f"*{variant_tag}*.txt")):
        parsed = parse_report_text(read_text(path))
        if parsed:
            candidates.append((path, parsed))
    if not candidates:
        return {}
    path, parsed = min(
        candidates,
        key=lambda item: (
            float(item[1].get("aic", np.inf)),
            float(item[1].get("r_factor", np.inf)),
            item[0].name,
        ),
    )
    return {
        "file": path.name,
        "r_factor": float(parsed.get("r_factor", np.nan)),
        "aic": float(parsed.get("aic", np.nan)),
    }


def resolve_report_metric(
    report: dict[str, float | str],
    *,
    exact_keys: tuple[str, ...] = (),
    regex_patterns: tuple[str, ...] = (),
) -> tuple[str | None, float, float]:
    for key in exact_keys:
        if key in report:
            value = float(report.get(key, np.nan))
            error = float(report.get(f"{key}_err", np.nan))
            return key, value, error

    candidate_keys = [
        key
        for key in report
        if not key.endswith("_err") and all(token not in key for token in ("reff", "_label"))
    ]
    for pattern in regex_patterns:
        for key in sorted(candidate_keys):
            if re.fullmatch(pattern, key):
                value = float(report.get(key, np.nan))
                error = float(report.get(f"{key}_err", np.nan))
                return key, value, error

    return None, np.nan, np.nan


def warn_if_missing_error(
    *,
    folder: Path,
    report_path: Path | None,
    metric_name: str,
    source_key: str | None,
    value: float,
    error: float,
) -> None:
    if np.isfinite(value) and not np.isfinite(error):
        location = folder.name if report_path is None else f"{folder.name}/{report_path.name}"
        key_text = source_key if source_key is not None else metric_name
        warnings.warn(
            f"Extracted {metric_name} from '{location}' using key '{key_text}' but no associated error was found.",
            stacklevel=2,
        )


def choose_metric_fallback(
    candidates: list[tuple[Path, dict[str, float | str]]],
    *,
    primary_path: Path | None,
    exact_keys: tuple[str, ...] = (),
    regex_patterns: tuple[str, ...] = (),
) -> tuple[Path | None, dict[str, float | str], str | None, float, float]:
    ordered_candidates = sorted(candidates, key=score_report_candidate)
    primary_item: tuple[Path, dict[str, float | str]] | None = None
    if primary_path is not None:
        for path, parsed in ordered_candidates:
            if path == primary_path:
                primary_item = (path, parsed)
                break

    if primary_item is None and ordered_candidates:
        primary_item = ordered_candidates[0]

    if primary_item is None:
        return None, {}, None, np.nan, np.nan

    selected_path, selected_report = primary_item
    key, value, error = resolve_report_metric(selected_report, exact_keys=exact_keys, regex_patterns=regex_patterns)
    if np.isfinite(error):
        return selected_path, selected_report, key, value, error

    for candidate_path, candidate_report in ordered_candidates:
        if candidate_path == selected_path:
            continue
        alt_key, alt_value, alt_error = resolve_report_metric(candidate_report, exact_keys=exact_keys, regex_patterns=regex_patterns)
        if np.isfinite(alt_error):
            return candidate_path, candidate_report, alt_key, alt_value, alt_error

    return selected_path, selected_report, key, value, error


def find_plot_file(folder: Path, kind: str = "chir_mag") -> Path | None:
    files = sorted(path for path in folder.iterdir() if path.is_file())
    for path in files:
        name = path.name.lower()
        if kind == "chir_mag" and "chir" in name and "re" not in name and ("paths" in name or "mag" in name):
            return path
        if kind == "rechir" and ("rechir" in name or "chir_re" in name):
            return path
        if kind == "chik" and ("chikw" in name or "chik" in name):
            return path
    return None


def parse_plot_data(path: Path) -> pd.DataFrame:
    text = read_text(path)
    lines = text.splitlines()
    header_candidates: list[str] = []
    for line in lines:
        if line.lstrip().startswith("#"):
            cleaned = line.lstrip("#").strip()
            if cleaned:
                header_candidates.append(cleaned)
        elif line.strip():
            break

    data_lines = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    frame = pd.read_csv(StringIO("\n".join(data_lines)), sep=r"\s+", header=None)

    header = next(
        (
            line
            for line in reversed(header_candidates)
            if "model" in line.lower() and ("expdata" in line.lower() or line.lower().startswith("x1"))
        ),
        None,
    )
    if header:
        names = re.split(r"\s{2,}", header)
        if len(names) == frame.shape[1]:
            frame.columns = names

    if "expdata" in frame.columns and "model" in frame.columns:
        return pd.DataFrame({"x": frame.iloc[:, 0], "data": frame["expdata"], "fit": frame["model"]})
    if frame.shape[1] >= 4:
        return pd.DataFrame({"x": frame.iloc[:, 0], "data": frame.iloc[:, 1], "fit": frame.iloc[:, 3]})
    if frame.shape[1] >= 3:
        return pd.DataFrame({"x": frame.iloc[:, 0], "data": frame.iloc[:, 1], "fit": frame.iloc[:, 2]})
    raise ValueError(f"Unexpected EXAFS plot format in {path}")


def collect_exafs(input_folder: Path, exafs_panel_kind: str = "chir_mag") -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    records: list[dict[str, Any]] = []
    spectra: list[pd.DataFrame] = []
    for folder in sorted(path for path in input_folder.iterdir() if path.is_dir() and path.name.lower().startswith("cspbi3_")):
        concentration = parse_concentration(folder.name)
        report_candidates = get_report_candidates(folder)
        constrained_stats = parse_fit_variant(folder, "ConstrainedFit")
        relaxed_stats = parse_fit_variant(folder, "RelaxedFit")
        if report_candidates:
            report_path, report = min(report_candidates, key=score_report_candidate)
        else:
            report_path, report = None, {}
        n_i_key, n_i_value, n_i_err = resolve_report_metric(report, exact_keys=("N_I",))
        n_o_key, n_o_value, n_o_err = resolve_report_metric(report, exact_keys=("N_O",))
        (
            pb_i_metric_path,
            pb_i_metric_report,
            delr_i_key,
            delr_i_value,
            delr_i_err,
        ) = choose_metric_fallback(
            report_candidates,
            primary_path=report_path,
            exact_keys=("delr_I308",),
            regex_patterns=(r"delr_I.*",),
        )
        delr_o_key, delr_o_value, delr_o_err = resolve_report_metric(report, exact_keys=("delr_O257",), regex_patterns=(r"delr_O.*",))
        sigma2_i_key, sigma2_i_value, sigma2_i_err = resolve_report_metric(
            report,
            exact_keys=("sigma2_I308",),
            regex_patterns=(r"sigma2_I.*",),
        )
        sigma2_o_key, sigma2_o_value, sigma2_o_err = resolve_report_metric(
            report,
            exact_keys=("sigma2_O257",),
            regex_patterns=(r"sigma2_O.*",),
        )
        e0_key, e0_value, e0_err = resolve_report_metric(report, exact_keys=("e0",))

        warn_if_missing_error(folder=folder, report_path=report_path, metric_name="n_i", source_key=n_i_key, value=n_i_value, error=n_i_err)
        warn_if_missing_error(folder=folder, report_path=report_path, metric_name="n_o", source_key=n_o_key, value=n_o_value, error=n_o_err)
        warn_if_missing_error(folder=folder, report_path=pb_i_metric_path, metric_name="delr_i", source_key=delr_i_key, value=delr_i_value, error=delr_i_err)
        warn_if_missing_error(folder=folder, report_path=report_path, metric_name="delr_o", source_key=delr_o_key, value=delr_o_value, error=delr_o_err)
        warn_if_missing_error(folder=folder, report_path=report_path, metric_name="sigma2_i", source_key=sigma2_i_key, value=sigma2_i_value, error=sigma2_i_err)
        warn_if_missing_error(folder=folder, report_path=report_path, metric_name="sigma2_o", source_key=sigma2_o_key, value=sigma2_o_value, error=sigma2_o_err)
        warn_if_missing_error(folder=folder, report_path=report_path, metric_name="e0", source_key=e0_key, value=e0_value, error=e0_err)

        pb_i_reff = pb_i_metric_report.get("pb_i_reff", report.get("pb_i_reff", np.nan))
        pb_i_r = pb_i_metric_report.get("pb_i_r", np.nan)
        if not np.isfinite(float(pb_i_r)) and np.isfinite(float(pb_i_reff)) and np.isfinite(delr_i_value):
            pb_i_r = float(pb_i_reff) + delr_i_value

        record = {
            "folder": folder.name,
            "concentration_m": concentration,
            "report_file": report_path.name if report_path else None,
            "pb_i_report_file": pb_i_metric_path.name if pb_i_metric_path else None,
            "fit_source": classify_report_source(report_path.name if report_path else None),
            "constrained_report_file": constrained_stats.get("file", None),
            "constrained_r_factor": constrained_stats.get("r_factor", np.nan),
            "constrained_aic": constrained_stats.get("aic", np.nan),
            "relaxed_report_file": relaxed_stats.get("file", None),
            "relaxed_r_factor": relaxed_stats.get("r_factor", np.nan),
            "relaxed_aic": relaxed_stats.get("aic", np.nan),
            "r_factor": report.get("r_factor", np.nan),
            "aic": report.get("aic", np.nan),
            "bic": report.get("bic", np.nan),
            "n_i": n_i_value,
            "n_i_err": n_i_err,
            "n_o": n_o_value,
            "n_o_err": n_o_err,
            "delr_i": delr_i_value,
            "delr_i_err": delr_i_err,
            "delr_o": delr_o_value,
            "delr_o_err": delr_o_err,
            "sigma2_i": sigma2_i_value,
            "sigma2_i_err": sigma2_i_err,
            "sigma2_o": sigma2_o_value,
            "sigma2_o_err": sigma2_o_err,
            "e0": e0_value,
            "e0_err": e0_err,
            "pb_i_reff": pb_i_reff,
            "pb_i_r": pb_i_r,
            "pb_o_reff": report.get("pb_o_reff", np.nan),
            "pb_o_r": report.get("pb_o_r", np.nan),
            "pb_i_r_err": delr_i_err,
            "pb_o_r_err": delr_o_err,
        }
        records.append(record)

        plot_path = find_plot_file(folder, exafs_panel_kind)
        if plot_path:
            spectrum = parse_plot_data(plot_path)
            spectrum["folder"] = folder.name
            spectrum["concentration_m"] = concentration
            spectra.append(spectrum)

    summary = pd.DataFrame(records).sort_values("concentration_m").reset_index(drop=True)
    return summary, spectra


def load_uv_vis_sets(uv_vis_folder: Path) -> list[pd.DataFrame]:
    file_paths = [
        uv_vis_folder / "OliverTraining_CsPbI.txt",
        uv_vis_folder / "OliverTraining_CsPbI2.txt",
        uv_vis_folder / "OliverTrainingThing.txt",
    ]

    datasets: list[pd.DataFrame] = []
    for path in file_paths:
        frame = pd.read_csv(path, skipinitialspace=True)
        frame["Wavelength (nm)"] = pd.to_numeric(frame["Wavelength (nm)"], errors="coerce")
        frame["Transmittance"] = pd.to_numeric(frame["Transmittance"], errors="coerce")
        frame = frame.dropna(subset=["Wavelength (nm)", "Transmittance"]).copy()
        frame["Transmittance"] = frame["Transmittance"].clip(lower=1e-9)
        frame["Absorbance"] = -np.log10(frame["Transmittance"]) + 2
        datasets.append(frame)

    solvent, one_pct, ten_pct = [datasets[index].copy() for index in (2, 0, 1)]
    one_pct["Absorbance_Corrected"] = one_pct["Absorbance"] - solvent["Absorbance"].to_numpy()
    ten_pct["Absorbance_Corrected"] = ten_pct["Absorbance"] - solvent["Absorbance"].to_numpy()

    converted: list[pd.DataFrame] = []
    for label, frame in (("1 wt% CsPbI$_3$", one_pct), ("10 wt% CsPbI$_3$", ten_pct)):
        dataset = frame.copy()
        dataset["Wavenumber (cm^-1)"] = 1e7 / dataset["Wavelength (nm)"].to_numpy()
        dataset = dataset.sort_values("Wavenumber (cm^-1)").reset_index(drop=True)
        dataset["label"] = label
        converted.append(dataset)
    return converted


def gaussian(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width**2))


def multi_gaussian(params: Parameters, x: np.ndarray, n_gaussians: int) -> np.ndarray:
    model = np.zeros_like(x)
    for index in range(1, n_gaussians + 1):
        model += gaussian(x, params[f"A{index}"].value, params[f"mu{index}"].value, params[f"sigma{index}"].value)
    return model


def residual(params: Parameters, x: np.ndarray, data: np.ndarray, n_gaussians: int) -> np.ndarray:
    return data - multi_gaussian(params, x, n_gaussians)


def build_params_1pct() -> Parameters:
    params = Parameters()
    params.add("A1", value=0.18, min=0, max=10, vary=True)
    params.add("mu1", value=34988, min=25000, max=45000, vary=True)
    params.add("sigma1", value=1800, min=200, max=2000, vary=True)
    params.add("A2", value=0.09, min=0, max=10, vary=True)
    params.add("mu2", value=31033, min=25000, max=45000, vary=False)
    params.add("sigma2", value=1186, min=200, max=2000, vary=True)
    params.add("A3", value=0.20, min=0, max=3, vary=True)
    params.add("mu3", value=38327, min=30000, max=40000, vary=True)
    params.add("sigma3", value=1400, min=200, max=2000, vary=False)
    return params


def build_params_10pct() -> Parameters:
    params = Parameters()
    params.add("A1", value=1.66, min=0, max=10, vary=True)
    params.add("mu1", value=34988, min=25000, max=45000, vary=False)
    params.add("sigma1", value=1400, min=200, max=10000, vary=True)
    params.add("A2", value=1.58, min=0, max=10, vary=True)
    params.add("mu2", value=31021, min=25000, max=45000, vary=True)
    params.add("sigma2", value=1186, min=200, max=10000, vary=False)
    params.add("A3", value=0.27, min=0, max=10, vary=True)
    params.add("mu3", value=27500, min=20000, max=35000, vary=True)
    params.add("sigma3", value=1542, min=200, max=10000, vary=False)
    return params


def run_gaussian_fit(
    x: np.ndarray,
    y: np.ndarray,
    params: Parameters,
    n_gaussians: int,
    fit_min: float,
    fit_max: float,
) -> dict[str, Any]:
    mask = (x >= fit_min) & (x <= fit_max)
    x_fit = x[mask]
    y_fit = y[mask]
    result = minimize(residual, params, args=(x_fit, y_fit, n_gaussians))
    y_pred = multi_gaussian(result.params, x_fit, n_gaussians)
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    return {
        "x": x_fit,
        "y": y_fit,
        "result": result,
        "r2": 1 - ss_res / ss_tot,
        "n_gaussians": n_gaussians,
    }


def add_wavelength_axis(ax: Any) -> Any:
    def wn_to_nm(wn: np.ndarray) -> np.ndarray:
        wn = np.asarray(wn, dtype=float)
        out = np.full_like(wn, np.inf, dtype=float)
        mask = wn != 0
        out[mask] = 1e7 / wn[mask]
        return out

    def nm_to_wn(nm: np.ndarray) -> np.ndarray:
        nm = np.asarray(nm, dtype=float)
        out = np.full_like(nm, np.inf, dtype=float)
        mask = nm != 0
        out[mask] = 1e7 / nm[mask]
        return out

    secax = ax.secondary_xaxis("top", functions=(wn_to_nm, nm_to_wn))
    secax.set_xlabel("Wavelength (nm)")
    secax.set_xticks([250, 300, 350, 400])
    secax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    secax.tick_params(axis="x", labeltop=True, top=True, pad=3, width=1.3, length=4.5)
    return secax


def format_uv_wavenumber_axis(ax: Any) -> None:
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value / 1000:.0f}"))
    ax.set_xlabel(r"Wavenumber ($\times 10^3$ cm$^{-1}$)")


def add_panel_label(ax: Any, label: str) -> None:
    formatted = f"{label.lower()})"
    ax.text(-0.18, 1.08, formatted, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def style_axis_heavier(ax: Any) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis="both", which="major", width=1.3, length=4.5)
    ax.tick_params(axis="both", which="minor", width=1.1, length=2.8)


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "mathtext.fontset": "dejavusans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.linewidth": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 1.3,
            "ytick.major.width": 1.3,
            "xtick.minor.width": 1.1,
            "ytick.minor.width": 1.1,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.minor.size": 2.8,
            "ytick.minor.size": 2.8,
        }
    )


def build_uv_fit_components(fit_result: dict[str, Any]) -> dict[str, Any]:
    x = np.asarray(fit_result["x"], dtype=float)
    y = np.asarray(fit_result["y"], dtype=float)
    x_smooth = np.linspace(float(x.min()), float(x.max()), 600)
    components: list[np.ndarray] = []
    for index in range(1, int(fit_result["n_gaussians"]) + 1):
        component = gaussian(
            x_smooth,
            fit_result["result"].params[f"A{index}"].value,
            fit_result["result"].params[f"mu{index}"].value,
            fit_result["result"].params[f"sigma{index}"].value,
        )
        components.append(component)
    return {"x": x, "y": y, "x_smooth": x_smooth, "components": components, "total": np.sum(components, axis=0)}


def set_errorbar_limits(axis: Any, values: pd.Series, errors: pd.Series, padding_fraction: float = 0.18) -> None:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(errors, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(v)
    if not np.any(finite):
        return
    safe_errors = np.where(np.isfinite(e), np.abs(e), 0.0)
    lower = np.nanmin(v[finite] - safe_errors[finite])
    upper = np.nanmax(v[finite] + safe_errors[finite])
    span = upper - lower
    margin = span * padding_fraction if span > 0 else max(abs(lower), abs(upper), 1.0) * 0.08
    axis.set_ylim(lower - margin, upper + margin)


def generate_spectroscopy_figure(
    repo_root: str | Path | None = None,
    input_folder: str | Path | None = None,
    output_folder: str | Path | None = None,
    uv_vis_folder: str | Path | None = None,
    exafs_panel_kind: str = "chir_mag",
    exafs_r_window: tuple[float, float] = (0.8, 5.2),
) -> dict[str, Any]:
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    input_folder = Path(input_folder) if input_folder is not None else repo_root / "data" / "CsPbI3_fits"
    output_folder = Path(output_folder) if output_folder is not None else input_folder / "Figures"
    uv_vis_folder = Path(uv_vis_folder) if uv_vis_folder is not None else default_uv_vis_folder()
    output_folder.mkdir(parents=True, exist_ok=True)

    apply_publication_style()

    summary, spectra = collect_exafs(input_folder, exafs_panel_kind=exafs_panel_kind)
    summary_csv_path = output_folder / "exafs_fit_summary.csv"
    summary.to_csv(summary_csv_path, index=False)

    uv_sets = load_uv_vis_sets(uv_vis_folder)
    fit_1 = run_gaussian_fit(
        uv_sets[0]["Wavenumber (cm^-1)"].to_numpy(),
        uv_sets[0]["Absorbance_Corrected"].to_numpy(),
        build_params_1pct(),
        3,
        25000,
        38750,
    )
    fit_10 = run_gaussian_fit(
        uv_sets[1]["Wavenumber (cm^-1)"].to_numpy(),
        uv_sets[1]["Absorbance_Corrected"].to_numpy(),
        build_params_10pct(),
        3,
        25000,
        35000,
    )

    fig = plt.figure(figsize=(13.6, 8.8))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], hspace=0.38, wspace=0.34)
    ax_uv1 = fig.add_subplot(grid[0, 0])
    ax_uv2 = fig.add_subplot(grid[0, 1])
    ax_exafs = fig.add_subplot(grid[0, 2])
    ax_cn = fig.add_subplot(grid[1, 0])
    ax_r = fig.add_subplot(grid[1, 1])
    ax_sig = fig.add_subplot(grid[1, 2])
    for axis in (ax_uv1, ax_uv2, ax_exafs, ax_cn, ax_r, ax_sig):
        style_axis_heavier(axis)

    uv_colors = {
        "data_1": "#D55E00",
        "data_10": "#0072B2",
        "fit": "#111111",
        "PbI+": "#E69F00",
        "PbI2": "#56B4E9",
        "PbI3-": "#009E73",
        "Pb2+": "#D7263D",
    }

    uv_plot_specs = [
        (
            ax_uv1,
            uv_sets[0],
            fit_1,
            uv_colors["data_1"],
            "1 wt% CsPbI$_3$ in DMSO",
            [(r"PbI$^+$", "PbI+"), (r"PbI$_2$", "PbI2"), (r"Pb$^{2+}$", "Pb2+")],
        ),
        (
            ax_uv2,
            uv_sets[1],
            fit_10,
            uv_colors["data_10"],
            "10 wt% CsPbI$_3$ in DMSO",
            [(r"PbI$^+$", "PbI+"), (r"PbI$_2$", "PbI2"), (r"PbI$_3^-$", "PbI3-")],
        ),
    ]

    for ax, fit_result, dataset_color, title, species in uv_plot_specs:
        x = fit_result["x"]
        y = fit_result["y"]
        x_smooth = np.linspace(x.min(), x.max(), 600)
        components: list[np.ndarray] = []
        for index in range(1, int(fit_result["n_gaussians"]) + 1):
            component = gaussian(
                x_smooth,
                fit_result["result"].params[f"A{index}"].value,
                fit_result["result"].params[f"mu{index}"].value,
                fit_result["result"].params[f"sigma{index}"].value,
            )
            components.append(component)
        total = np.sum(components, axis=0)

        ax.scatter(x, y, s=7, color=dataset_color, alpha=0.55, edgecolors="none", label="Data", zorder=3)
        for component, (label, color_key) in zip(components, species):
            ax.fill_between(x_smooth, 0, component, color=uv_colors[color_key], alpha=0.20)
            ax.plot(x_smooth, component, "--", color=uv_colors[color_key], linewidth=1.1, label=label)
        ax.plot(x_smooth, total, color=uv_colors["fit"], linewidth=1.5, label="Total fit")

        ax.set_xlim(x.min() - 500, x.max() + 500)
        ax.set_ylim(0, max(float(np.max(y)), float(np.max(total))) * 1.12)
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax.set_ylabel(r"$\Delta$Absorbance")
        ax.set_title(title, pad=8)
        ax.text(0.98, 0.96, f"$R^2$ = {fit_result['r2']:.4f}", transform=ax.transAxes, ha="right", va="top")
        add_wavelength_axis(ax)
        ax.legend(loc="upper left", frameon=False, handlelength=2.1)

    concentration_values = summary["concentration_m"].to_numpy()
    concentration_colors = plt.cm.cividis(np.linspace(0.1, 0.9, len(concentration_values)))
    conc_to_color = {conc: color for conc, color in zip(concentration_values, concentration_colors)}
    stack_offset = 1.15

    for index, frame in enumerate(sorted(spectra, key=lambda item: item["concentration_m"].iloc[0])):
        concentration = float(frame["concentration_m"].iloc[0])
        color = conc_to_color[concentration]
        subset = frame[(frame["x"] >= exafs_r_window[0]) & (frame["x"] <= exafs_r_window[1])].copy()
        scale = np.nanmax(np.abs(np.r_[subset["data"].to_numpy(), subset["fit"].to_numpy()]))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        y_data = subset["data"] / scale + index * stack_offset
        y_fit = subset["fit"] / scale + index * stack_offset

        ax_exafs.plot(subset["x"], y_data, color=color, linewidth=1.2)
        ax_exafs.plot(subset["x"], y_fit, color="black", linestyle="--", linewidth=0.95, alpha=0.9)
        ax_exafs.text(exafs_r_window[1] + 0.06, index * stack_offset, f"{concentration:.1f} M", va="center", fontsize=7.5, color=color)

    ax_exafs.set_xlim(exafs_r_window[0], exafs_r_window[1] + 0.35)
    ax_exafs.set_ylim(-0.2, max(1, len(spectra) - 1) * stack_offset + 1.0)
    ax_exafs.set_xlabel(r"R ($\AA$)")
    ax_exafs.set_ylabel(r"Normalized |$\chi(R)$| + offset")
    ax_exafs.set_title(r"EXAFS |$\chi(R)$| fits", pad=8)
    ax_exafs.set_yticks([])
    ax_exafs.text(0.02, 0.03, "solid: experiment   dashed: fit", transform=ax_exafs.transAxes, fontsize=7)

    series_colors = {"Pb-I": "#1f4e79", "Pb-O": "#c15a1b"}
    for axis in (ax_cn, ax_r, ax_sig):
        axis.set_xlabel("CsPbI$_3$ concentration (M)")
        axis.grid(False)
        axis.set_xlim(summary["concentration_m"].min() - 0.03, summary["concentration_m"].max() + 0.03)
        axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

    ax_cn.errorbar(summary["concentration_m"], summary["n_i"], yerr=summary["n_i_err"], fmt="o-", color=series_colors["Pb-I"], capsize=2, linewidth=1.2, markersize=4.2, label="Pb-I")
    ax_cn.errorbar(summary["concentration_m"], summary["n_o"], yerr=summary["n_o_err"], fmt="s-", color=series_colors["Pb-O"], capsize=2, linewidth=1.2, markersize=4.0, label="Pb-O")
    ax_cn.set_ylabel("Coordination number")
    ax_cn.set_title("EXAFS coordination numbers", pad=8)
    ax_cn.legend(frameon=False, loc="best")

    ax_r.errorbar(summary["concentration_m"], summary["pb_i_r"], yerr=summary["pb_i_r_err"], fmt="o-", color=series_colors["Pb-I"], capsize=2, linewidth=1.2, markersize=4.2, label="Pb-I")
    ax_r.errorbar(summary["concentration_m"], summary["pb_o_r"], yerr=summary["pb_o_r_err"], fmt="s-", color=series_colors["Pb-O"], capsize=2, linewidth=1.2, markersize=4.0, label="Pb-O")
    ax_r.set_ylabel(r"Effective distance, R ($\AA$)")
    ax_r.set_title("EXAFS path distances", pad=8)
    ax_r.legend(frameon=False, loc="best")

    ax_sig.errorbar(summary["concentration_m"], summary["sigma2_i"], yerr=summary["sigma2_i_err"], fmt="o-", color=series_colors["Pb-I"], capsize=2, linewidth=1.2, markersize=4.2, label="Pb-I")
    ax_sig.errorbar(summary["concentration_m"], summary["sigma2_o"], yerr=summary["sigma2_o_err"], fmt="s-", color=series_colors["Pb-O"], capsize=2, linewidth=1.2, markersize=4.0, label="Pb-O")
    ax_sig.set_ylabel(r"$\sigma^2$ ($\AA^2$)")
    ax_sig.set_title("EXAFS Debye-Waller factors", pad=8)
    ax_sig.legend(frameon=False, loc="best")

    for axis, label in zip((ax_uv1, ax_uv2, ax_exafs, ax_cn, ax_r, ax_sig), "ABCDEF"):
        add_panel_label(axis, label)

    png_path = output_folder / "spectroscopy_uvvis_exafs_summary.png"
    pdf_path = output_folder / "spectroscopy_uvvis_exafs_summary.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)

    return {
        "summary": summary,
        "spectra_count": len(spectra),
        "fit_1": fit_1,
        "fit_10": fit_10,
        "figure": fig,
        "png_path": png_path,
        "pdf_path": pdf_path,
        "summary_csv_path": summary_csv_path,
        "uv_vis_folder": uv_vis_folder,
        "input_folder": input_folder,
        "output_folder": output_folder,
    }


def generate_spectroscopy_figure_matched_uv_split_distance(
    repo_root: str | Path | None = None,
    input_folder: str | Path | None = None,
    output_folder: str | Path | None = None,
    uv_vis_folder: str | Path | None = None,
    exafs_panel_kind: str = "chir_mag",
    exafs_r_window: tuple[float, float] = (0.8, 5.2),
    output_stem: str = "spectroscopy_uvvis_exafs_summary_matched_uv_split_distance",
) -> dict[str, Any]:
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    input_folder = Path(input_folder) if input_folder is not None else repo_root / "data" / "CsPbI3_fits"
    output_folder = Path(output_folder) if output_folder is not None else input_folder / "Figures"
    uv_vis_folder = Path(uv_vis_folder) if uv_vis_folder is not None else default_uv_vis_folder()
    output_folder.mkdir(parents=True, exist_ok=True)

    apply_publication_style()

    summary, spectra = collect_exafs(input_folder, exafs_panel_kind=exafs_panel_kind)
    summary_csv_path = output_folder / f"{output_stem}_summary.csv"
    summary.to_csv(summary_csv_path, index=False)

    r_factor_columns = [
        "folder",
        "concentration_m",
        "report_file",
        "fit_source",
        "r_factor",
        "aic",
        "constrained_report_file",
        "constrained_r_factor",
        "constrained_aic",
        "relaxed_report_file",
        "relaxed_r_factor",
        "relaxed_aic",
    ]
    r_factor_table = summary[r_factor_columns].copy()
    r_factor_csv_path = output_folder / f"{output_stem}_r_factors.csv"
    r_factor_table.to_csv(r_factor_csv_path, index=False)

    uv_sets = load_uv_vis_sets(uv_vis_folder)
    fit_1 = run_gaussian_fit(
        uv_sets[0]["Wavenumber (cm^-1)"].to_numpy(),
        uv_sets[0]["Absorbance_Corrected"].to_numpy(),
        build_params_1pct(),
        3,
        25000,
        38750,
    )
    fit_10 = run_gaussian_fit(
        uv_sets[1]["Wavenumber (cm^-1)"].to_numpy(),
        uv_sets[1]["Absorbance_Corrected"].to_numpy(),
        build_params_10pct(),
        3,
        25000,
        35000,
    )

    fig = plt.figure(figsize=(14.4, 9.6))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], hspace=0.38, wspace=0.34)
    ax_uv1 = fig.add_subplot(grid[0, 0])
    ax_uv2 = fig.add_subplot(grid[0, 1])
    ax_exafs = fig.add_subplot(grid[0, 2])
    ax_cn = fig.add_subplot(grid[1, 0])
    distance_grid = grid[1, 1].subgridspec(2, 1, hspace=0.42)
    ax_r_i = fig.add_subplot(distance_grid[0, 0])
    ax_r_o = fig.add_subplot(distance_grid[1, 0])
    ax_sig = fig.add_subplot(grid[1, 2])
    for axis in (ax_uv1, ax_uv2, ax_exafs, ax_cn, ax_r_i, ax_r_o, ax_sig):
        style_axis_heavier(axis)

    uv_colors = {
        "data_1": "#D55E00",
        "data_10": "#0072B2",
        "fit": "#111111",
        "PbI+": "#E69F00",
        "PbI2": "#56B4E9",
        "PbI3-": "#009E73",
        "Pb2+": "#D7263D",
    }

    uv_plot_specs = [
        (
            ax_uv1,
            uv_sets[0],
            fit_1,
            uv_colors["data_1"],
            "1 wt% CsPbI$_3$ in DMSO",
            [(r"PbI$^+$", "PbI+"), (r"PbI$_2$", "PbI2"), (r"Pb$^{2+}$", "Pb2+")],
        ),
        (
            ax_uv2,
            uv_sets[1],
            fit_10,
            uv_colors["data_10"],
            "10 wt% CsPbI$_3$ in DMSO",
            [(r"PbI$^+$", "PbI+"), (r"PbI$_2$", "PbI2"), (r"PbI$_3^-$", "PbI3-")],
        ),
    ]

    uv_models: list[dict[str, Any]] = []
    uv_xmin = np.inf
    uv_xmax = -np.inf
    uv_ymax = 0.0
    for _, full_dataset, fit_result, _, _, _ in uv_plot_specs:
        model = build_uv_fit_components(fit_result)
        uv_models.append(model)
        full_x = pd.to_numeric(full_dataset["Wavenumber (cm^-1)"], errors="coerce").to_numpy(dtype=float)
        finite_full_x = full_x[np.isfinite(full_x)]
        uv_xmin = min(uv_xmin, float(np.min(finite_full_x)))
        uv_xmax = max(uv_xmax, float(np.max(finite_full_x)))
        uv_ymax = max(
            uv_ymax,
            float(np.max(model["y"])),
            float(np.max(model["total"])),
        )
    shared_uv_xlim = (25000.0, 38750.0)
    shared_uv_ylim = (0.0, uv_ymax * 1.12)

    for (ax, full_dataset, fit_result, dataset_color, title, species), model in zip(uv_plot_specs, uv_models):
        x_full = pd.to_numeric(full_dataset["Wavenumber (cm^-1)"], errors="coerce").to_numpy(dtype=float)
        y_full = pd.to_numeric(full_dataset["Absorbance_Corrected"], errors="coerce").to_numpy(dtype=float)
        full_mask = np.isfinite(x_full) & np.isfinite(y_full) & (x_full >= shared_uv_xlim[0]) & (x_full <= shared_uv_xlim[1])
        x = x_full[full_mask]
        y = y_full[full_mask]
        x_smooth = np.linspace(shared_uv_xlim[0], shared_uv_xlim[1], 800)
        components = model["components"]
        total = np.zeros_like(x_smooth)
        components_full: list[np.ndarray] = []
        for index in range(1, int(fit_result["n_gaussians"]) + 1):
            component = gaussian(
                x_smooth,
                fit_result["result"].params[f"A{index}"].value,
                fit_result["result"].params[f"mu{index}"].value,
                fit_result["result"].params[f"sigma{index}"].value,
            )
            components_full.append(component)
            total += component

        ax.scatter(x, y, s=7, color=dataset_color, alpha=0.55, edgecolors="none", label="Data", zorder=3)
        for component, (label, color_key) in zip(components_full, species):
            ax.fill_between(x_smooth, 0, component, color=uv_colors[color_key], alpha=0.20)
            ax.plot(x_smooth, component, "--", color=uv_colors[color_key], linewidth=1.1, label=label)
        ax.plot(x_smooth, total, color=uv_colors["fit"], linewidth=1.5, label="Total fit")

        ax.set_xlim(*shared_uv_xlim)
        ax.set_ylim(*shared_uv_ylim)
        format_uv_wavenumber_axis(ax)
        ax.set_ylabel(r"Absorbance")
        ax.set_title(title, pad=8)
        ax.text(0.98, 0.96, f"$R^2$ = {fit_result['r2']:.4f}", transform=ax.transAxes, ha="right", va="top")
        add_wavelength_axis(ax)
        ax.legend(loc="upper left", frameon=False, handlelength=2.1)

    concentration_values = summary["concentration_m"].to_numpy()
    concentration_colors = plt.cm.cividis(np.linspace(0.1, 0.9, len(concentration_values)))
    conc_to_color = {conc: color for conc, color in zip(concentration_values, concentration_colors)}
    stack_offset = 1.15

    for index, frame in enumerate(sorted(spectra, key=lambda item: item["concentration_m"].iloc[0])):
        concentration = float(frame["concentration_m"].iloc[0])
        color = conc_to_color[concentration]
        subset = frame[(frame["x"] >= exafs_r_window[0]) & (frame["x"] <= exafs_r_window[1])].copy()
        scale = np.nanmax(np.abs(np.r_[subset["data"].to_numpy(), subset["fit"].to_numpy()]))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        y_data = subset["data"] / scale + index * stack_offset
        y_fit = subset["fit"] / scale + index * stack_offset

        ax_exafs.plot(subset["x"], y_data, color=color, linewidth=1.2)
        ax_exafs.plot(subset["x"], y_fit, color="black", linestyle="--", linewidth=0.95, alpha=0.9)
        ax_exafs.text(exafs_r_window[1] + 0.06, index * stack_offset, f"{concentration:.1f} M", va="center", fontsize=7.5, color=color)

    ax_exafs.set_xlim(exafs_r_window[0], exafs_r_window[1] + 0.35)
    ax_exafs.set_ylim(-0.2, max(1, len(spectra) - 1) * stack_offset + 1.0)
    ax_exafs.set_xlabel(r"R ($\AA$)")
    ax_exafs.set_ylabel(r"Normalized |$\chi(R)$| + offset")
    ax_exafs.set_title(r"EXAFS |$\chi(R)$| solid: experiment   dashed: fit", pad=8)
    ax_exafs.set_yticks([])
    # ax_exafs.text(0.02, 0.03, "solid: experiment   dashed: fit", transform=ax_exafs.transAxes, fontsize=7)

    series_colors = {"Pb-I": "#1f4e79", "Pb-O": "#c15a1b"}
    for axis in (ax_cn, ax_r_i, ax_r_o, ax_sig):
        axis.set_xlabel("CsPbI$_3$ concentration (M)")
        axis.grid(False)
        axis.set_xlim(summary["concentration_m"].min() - 0.03, summary["concentration_m"].max() + 0.03)
        axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax_r_i.set_xlabel("")
    ax_r_i.tick_params(labelbottom=False)

    ax_cn.errorbar(summary["concentration_m"], summary["n_i"], yerr=summary["n_i_err"], fmt="o-", color=series_colors["Pb-I"], capsize=2, linewidth=1.2, markersize=4.2, label="Pb-I")
    ax_cn.errorbar(summary["concentration_m"], summary["n_o"], yerr=summary["n_o_err"], fmt="s-", color=series_colors["Pb-O"], capsize=2, linewidth=1.2, markersize=4.0, label="Pb-O")
    ax_cn.set_ylabel("Coordination number")
    ax_cn.set_title("EXAFS coordination numbers", pad=8)
    ax_cn.legend(frameon=False, loc="best")

    ax_r_i.errorbar(summary["concentration_m"], summary["pb_i_r"], yerr=summary["pb_i_r_err"], fmt="o-", color=series_colors["Pb-I"], capsize=2, linewidth=1.2, markersize=4.2)
    ax_r_i.set_ylabel(r"Pb-I distance, R ($\AA$)")
    ax_r_i.set_title("Pb-I path distance", pad=8)
    set_errorbar_limits(ax_r_i, summary["pb_i_r"], summary["pb_i_r_err"])

    ax_r_o.errorbar(summary["concentration_m"], summary["pb_o_r"], yerr=summary["pb_o_r_err"], fmt="s-", color=series_colors["Pb-O"], capsize=2, linewidth=1.2, markersize=4.0)
    ax_r_o.set_ylabel(r"Pb-O distance, R ($\AA$)")
    ax_r_o.set_title("Pb-O path distance", pad=8)
    set_errorbar_limits(ax_r_o, summary["pb_o_r"], summary["pb_o_r_err"])

    ax_sig.errorbar(summary["concentration_m"], summary["sigma2_i"], yerr=summary["sigma2_i_err"], fmt="o-", color=series_colors["Pb-I"], capsize=2, linewidth=1.2, markersize=4.2, label="Pb-I")
    ax_sig.errorbar(summary["concentration_m"], summary["sigma2_o"], yerr=summary["sigma2_o_err"], fmt="s-", color=series_colors["Pb-O"], capsize=2, linewidth=1.2, markersize=4.0, label="Pb-O")
    ax_sig.set_ylabel(r"$\sigma^2$ ($\AA^2$)")
    ax_sig.set_title("EXAFS Debye-Waller factors", pad=8)
    ax_sig.legend(frameon=False, loc="best")

    common_box_aspect = 1.0
    for axis in (ax_uv1, ax_uv2, ax_exafs, ax_cn, ax_sig):
        axis.set_box_aspect(common_box_aspect)

    for axis, label in zip((ax_uv1, ax_uv2, ax_exafs, ax_cn, ax_r_i, ax_r_o, ax_sig), "ABCDEFG"):
        add_panel_label(axis, label)

    png_path = output_folder / f"{output_stem}.png"
    pdf_path = output_folder / f"{output_stem}.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)

    return {
        "summary": summary,
        "r_factor_table": r_factor_table,
        "r_factor_csv_path": r_factor_csv_path,
        "spectra_count": len(spectra),
        "fit_1": fit_1,
        "fit_10": fit_10,
        "figure": fig,
        "png_path": png_path,
        "pdf_path": pdf_path,
        "summary_csv_path": summary_csv_path,
        "uv_vis_folder": uv_vis_folder,
        "input_folder": input_folder,
        "output_folder": output_folder,
    }
