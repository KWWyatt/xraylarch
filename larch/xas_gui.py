
# import sys
# from pathlib import Path

# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
#     QLabel, QListWidget, QTextEdit, QHBoxLayout, QCheckBox, QSpinBox,
#     QDoubleSpinBox, QGroupBox, QFormLayout, QTabWidget
# )

# from XASmu2r import XASmu2r


# class XASGUI(QWidget):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle("XAS Processing GUI")
#         self.setGeometry(100, 100, 1000, 700)

#         self.xas = None
#         self.feff_folder = None

#         main_layout = QVBoxLayout()

#         # -----------------------
#         # Tabs
#         # -----------------------
#         self.tabs = QTabWidget()

#         self.preprocess_tab = QWidget()
#         self.ft_tab = QWidget()
#         self.feff_tab = QWidget()

#         self.tabs.addTab(self.preprocess_tab, "Preprocessing")
#         self.tabs.addTab(self.ft_tab, "FT Parameters")
#         self.tabs.addTab(self.feff_tab, "FEFF Fitting")

#         main_layout.addWidget(self.tabs)

#         self.setLayout(main_layout)

#         # Build each tab
#         self.setup_preprocess_tab()
#         self.setup_ft_tab()
#         self.setup_feff_tab()

#     # =================================================
#     # PREPROCESS TAB (your original UI)
#     # =================================================
#     def setup_preprocess_tab(self):
#         layout = QVBoxLayout()

#         self.folder_label = QLabel("No folder selected")
#         self.load_button = QPushButton("Load Athena Project Folder")
#         self.load_button.clicked.connect(self.load_folder)

#         layout.addWidget(self.folder_label)
#         layout.addWidget(self.load_button)

#         self.project_list = QListWidget()
#         layout.addWidget(QLabel("Projects / Groups"))
#         layout.addWidget(self.project_list)

#         controls = QGroupBox("Processing Controls")
#         controls_layout = QFormLayout()

#         self.smooth_checkbox = QCheckBox()
#         self.smooth_checkbox.setChecked(True)

#         self.window_length = QSpinBox()
#         self.window_length.setValue(15)

#         self.polyorder = QSpinBox()
#         self.polyorder.setValue(3)

#         self.pre1 = QDoubleSpinBox()
#         self.pre1.setRange(-500, 0)
#         self.pre1.setValue(-150)

#         self.pre2 = QDoubleSpinBox()
#         self.pre2.setRange(-200, 0)
#         self.pre2.setValue(-50)

#         controls_layout.addRow("Smooth:", self.smooth_checkbox)
#         controls_layout.addRow("Window Length:", self.window_length)
#         controls_layout.addRow("Polyorder:", self.polyorder)
#         controls_layout.addRow("pre1 (eV):", self.pre1)
#         controls_layout.addRow("pre2 (eV):", self.pre2)

#         controls.setLayout(controls_layout)
#         layout.addWidget(controls)

#         btn_layout = QHBoxLayout()

#         self.process_button = QPushButton("Run Batch Processing")
#         self.process_button.clicked.connect(self.run_processing)

#         self.fluo_button = QPushButton("Apply Self-Absorption (FLUO)")
#         self.fluo_button.clicked.connect(self.run_fluo)

#         btn_layout.addWidget(self.process_button)
#         btn_layout.addWidget(self.fluo_button)

#         layout.addLayout(btn_layout)

#         self.log = QTextEdit()
#         self.log.setReadOnly(True)

#         layout.addWidget(QLabel("Log"))
#         layout.addWidget(self.log)

#         self.preprocess_tab.setLayout(layout)

#     # =================================================
#     # FT TAB
#     # =================================================
#     def setup_ft_tab(self):
#         layout = QVBoxLayout()

#         controls = QFormLayout()

#         self.kmin = QDoubleSpinBox()
#         self.kmin.setRange(0, 10)
#         self.kmin.setValue(2.0)

#         self.kmax = QDoubleSpinBox()
#         self.kmax.setRange(5, 20)
#         self.kmax.setValue(12.0)

#         self.kweight = QSpinBox()
#         self.kweight.setRange(0, 3)
#         self.kweight.setValue(2)

#         self.dk = QDoubleSpinBox()
#         self.dk.setValue(1.0)

#         controls.addRow("kmin:", self.kmin)
#         controls.addRow("kmax:", self.kmax)
#         controls.addRow("k-weight:", self.kweight)
#         controls.addRow("dk:", self.dk)

#         run_btn = QPushButton("Run FT")
#         run_btn.clicked.connect(self.run_ft)

#         layout.addLayout(controls)
#         layout.addWidget(run_btn)

#         self.ft_tab.setLayout(layout)

#     # =================================================
#     # FEFF TAB
#     # =================================================
#     def setup_feff_tab(self):
#         layout = QVBoxLayout()

#         self.feff_button = QPushButton("Load FEFF Folder")
#         self.feff_button.clicked.connect(self.load_feff)

#         self.fit_button = QPushButton("Run FEFF Fit")
#         self.fit_button.clicked.connect(self.run_feff_fit)

#         layout.addWidget(self.feff_button)
#         layout.addWidget(self.fit_button)

#         self.feff_tab.setLayout(layout)

#     # =================================================
#     # LOAD DATA
#     # =================================================
#     def load_folder(self):
#         folder = QFileDialog.getExistingDirectory(self, "Select Folder")

#         if folder:
#             self.folder_label.setText(folder)

#             try:
#                 self.xas = XASmu2r(folder)
#                 self.log.append("Loaded projects successfully")
#                 self.populate_projects()
#             except Exception as e:
#                 self.log.append(f"Error loading: {e}")

#     def populate_projects(self):
#         self.project_list.clear()

#         if not self.xas:
#             return

#         for proj_name, project in self.xas.projects.items():
#             self.project_list.addItem(f"[Project] {proj_name}")
#             for group_name in project.groups:
#                 self.project_list.addItem(f"   └─ {group_name}")

#     # =================================================
#     # PROCESSING
#     # =================================================
#     def run_processing(self):
#         if not self.xas:
#             self.log.append("No project loaded")
#             return

#         try:
#             self.log.append("Running batch processing...")

#             self.xas.batch_process_projects(
#                 smooth=self.smooth_checkbox.isChecked(),
#                 window_length=self.window_length.value(),
#                 polyorder=self.polyorder.value(),
#                 pre1=self.pre1.value(),
#                 pre2=self.pre2.value(),
#             )

#             self.log.append("Batch processing complete")

#         except Exception as e:
#             self.log.append(f"Error: {e}")

#     def run_fluo(self):
#         if not self.xas:
#             self.log.append("No project loaded")
#             return

#         try:
#             self.log.append("Applying FLUO correction...")
#             self.xas.apply_self_absorption_correction_all()
#             self.log.append("FLUO correction complete")
#         except Exception as e:
#             self.log.append(f"Error: {e}")

#     # =================================================
#     # FT
#     # =================================================
#     def run_ft(self):
#         if not self.xas:
#             self.log.append("Load data first")
#             return

#         from larch.xafs import xftf
#         import matplotlib.pyplot as plt

#         kmin = self.kmin.value()
#         kmax = self.kmax.value()
#         kweight = self.kweight.value()
#         dk = self.dk.value()

#         self.log.append(f"FT: kmin={kmin}, kmax={kmax}, kw={kweight}")

#         for proj_name, project in self.xas.projects.items():
#             for group_name, group in project.groups.items():
#                 try:
#                     xftf(group, kmin=kmin, kmax=kmax,
#                          dk=dk, kweight=kweight, kwindow='hanning')

#                     if hasattr(group, 'r'):
#                         plt.figure()
#                         plt.plot(group.r, group.chir_mag)
#                         plt.title(f"{proj_name} - {group_name}")
#                         plt.xlabel("R (Å)")
#                         plt.ylabel("|χ(R)|")
#                         plt.show()

#                 except Exception as e:
#                     self.log.append(f"{group_name} FT failed: {e}")

#     # =================================================
#     # FEFF
#     # =================================================
#     def load_feff(self):
#         folder = QFileDialog.getExistingDirectory(self, "Select FEFF Folder")
#         if folder:
#             self.feff_folder = Path(folder)
#             self.log.append(f"Loaded FEFF folder")

#     def run_feff_fit(self):
#         if not self.xas or not self.feff_folder:
#             self.log.append("Need data + FEFF folder")
#             return

#         from larch.xafs import feffpath, feffit, feffit_dataset, TransformGroup
#         from larch.fitting import param_group, guess

#         self.log.append("Running FEFF fit...")

#         for proj_name, project in self.xas.projects.items():
#             for group_name, group in project.groups.items():
#                 try:
#                     path = feffpath(str(self.feff_folder / "feff0001.dat"))

#                     params = param_group(
#                         s02=guess(1.0),
#                         delr=guess(0.0),
#                         sigma2=guess(0.003)
#                     )

#                     trans = TransformGroup(kmin=2, kmax=12, kweight=2)

#                     dset = feffit_dataset(data=group, pathlist=[path], transform=trans)

#                     feffit(params, dset)

#                     self.log.append(f"{group_name}: fit complete")

#                 except Exception as e:
#                     self.log.append(f"{group_name}: fit failed ({e})")


# # =================================================
# # RUN
# # =================================================
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = XASGUI()
#     window.show()
#     sys.exit(app.exec_())

import sys
import inspect
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QLabel, QListWidget, QTextEdit, QHBoxLayout, QCheckBox, QSpinBox,
    QDoubleSpinBox, QGroupBox, QFormLayout, QTabWidget, QLineEdit,
    QMessageBox, QComboBox
)

try:
    from .XASmu2r import XASmu2r
except ImportError:
    from XASmu2r import XASmu2r


class XASGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("XAS Processing GUI")
        self.setGeometry(100, 100, 1100, 760)

        self.xas = None
        self.feff_folder = None

        main_layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.preprocess_tab = QWidget()
        self.ft_tab = QWidget()
        self.feff_tab = QWidget()

        self.tabs.addTab(self.preprocess_tab, "Preprocessing")
        self.tabs.addTab(self.ft_tab, "FT Processing")
        self.tabs.addTab(self.feff_tab, "FEFF Fitting")

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

        self.setup_preprocess_tab()
        self.setup_ft_tab()
        self.setup_feff_tab()

    # =========================================================
    # Utility helpers
    # =========================================================
    def log_msg(self, msg: str):
        self.log.append(str(msg))
        print(msg)

    def safe_bool(self, widget):
        return bool(widget.isChecked())

    def safe_float(self, widget):
        return float(widget.value())

    def safe_int(self, widget):
        return int(widget.value())

    def call_method_if_exists(self, method_name, possible_kwargs):
        """
        Call a method on self.xas, but only pass kwargs that actually exist
        in that method's signature. This protects you from mismatches between
        GUI assumptions and the exact local version of XASmu2r.py.
        """
        if self.xas is None:
            raise RuntimeError("No XAS object loaded.")

        if not hasattr(self.xas, method_name):
            raise AttributeError(f"XASmu2r has no method named '{method_name}'")

        method = getattr(self.xas, method_name)
        sig = inspect.signature(method)

        accepted = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if name in possible_kwargs:
                accepted[name] = possible_kwargs[name]

        self.log_msg(f"Calling {method_name}({accepted})")
        return method(**accepted)

    # =========================================================
    # PREPROCESS TAB
    # =========================================================
    def setup_preprocess_tab(self):
        layout = QVBoxLayout()

        self.folder_label = QLabel("No folder selected")
        self.load_button = QPushButton("Load Athena Project Folder")
        self.load_button.clicked.connect(self.load_folder)

        layout.addWidget(self.folder_label)
        layout.addWidget(self.load_button)

        self.project_list = QListWidget()
        layout.addWidget(QLabel("Projects / Groups"))
        layout.addWidget(self.project_list)

        controls = QGroupBox("Processing Controls")
        controls_layout = QFormLayout()

        self.smooth_checkbox = QCheckBox()
        self.smooth_checkbox.setChecked(True)

        self.window_length = QSpinBox()
        self.window_length.setRange(3, 501)
        self.window_length.setSingleStep(2)
        self.window_length.setValue(15)

        self.polyorder = QSpinBox()
        self.polyorder.setRange(1, 10)
        self.polyorder.setValue(3)

        self.pre1 = QDoubleSpinBox()
        self.pre1.setRange(-1000, 0)
        self.pre1.setValue(-150)

        self.pre2 = QDoubleSpinBox()
        self.pre2.setRange(-500, 0)
        self.pre2.setValue(-50)

        self.optimize_preedge_checkbox = QCheckBox()
        self.optimize_preedge_checkbox.setChecked(True)

        controls_layout.addRow("Smooth:", self.smooth_checkbox)
        controls_layout.addRow("Window Length:", self.window_length)
        controls_layout.addRow("Polyorder:", self.polyorder)
        controls_layout.addRow("Optimize pre-edge:", self.optimize_preedge_checkbox)
        controls_layout.addRow("pre1 (eV):", self.pre1)
        controls_layout.addRow("pre2 (eV):", self.pre2)

        controls.setLayout(controls_layout)
        layout.addWidget(controls)

        btn_layout = QHBoxLayout()

        self.process_button = QPushButton("Run Batch Processing")
        self.process_button.clicked.connect(self.run_processing)

        self.fluo_button = QPushButton("Apply Self-Absorption (FLUO)")
        self.fluo_button.clicked.connect(self.run_fluo)

        btn_layout.addWidget(self.process_button)
        btn_layout.addWidget(self.fluo_button)

        layout.addLayout(btn_layout)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        layout.addWidget(QLabel("Log"))
        layout.addWidget(self.log)

        self.preprocess_tab.setLayout(layout)

    # =========================================================
    # FT TAB
    # =========================================================
    def setup_ft_tab(self):
        layout = QVBoxLayout()

        # --- Background optimizer controls ---
        bkg_group = QGroupBox("Background Optimizer")
        bkg_form = QFormLayout()

        self.ft_optimize_kmax = QCheckBox()
        self.ft_optimize_kmax.setChecked(True)

        self.ft_optimize_kmin = QCheckBox()
        self.ft_optimize_kmin.setChecked(True)

        self.ft_auto_clamp = QCheckBox()
        self.ft_auto_clamp.setChecked(True)

        self.ft_plot_results = QCheckBox()
        self.ft_plot_results.setChecked(True)

        self.ft_plot_clamp_diag = QCheckBox()
        self.ft_plot_clamp_diag.setChecked(False)

        self.ft_kmax_min = QDoubleSpinBox()
        self.ft_kmax_min.setRange(0.0, 30.0)
        self.ft_kmax_min.setValue(9.0)

        self.ft_kmax_max = QDoubleSpinBox()
        self.ft_kmax_max.setRange(0.0, 30.0)
        self.ft_kmax_max.setValue(16.0)

        self.ft_kmin_min = QDoubleSpinBox()
        self.ft_kmin_min.setRange(0.0, 10.0)
        self.ft_kmin_min.setValue(0.0)

        self.ft_kmin_max = QDoubleSpinBox()
        self.ft_kmin_max.setRange(0.0, 10.0)
        self.ft_kmin_max.setValue(1.5)

        self.ft_rbkg_min = QDoubleSpinBox()
        self.ft_rbkg_min.setRange(0.0, 5.0)
        self.ft_rbkg_min.setDecimals(3)
        self.ft_rbkg_min.setValue(0.9)

        self.ft_rbkg_max = QDoubleSpinBox()
        self.ft_rbkg_max.setRange(0.0, 5.0)
        self.ft_rbkg_max.setDecimals(3)
        self.ft_rbkg_max.setValue(1.5)

        bkg_form.addRow("Optimize kmax:", self.ft_optimize_kmax)
        bkg_form.addRow("Optimize kmin:", self.ft_optimize_kmin)
        bkg_form.addRow("Auto clamp:", self.ft_auto_clamp)
        bkg_form.addRow("Plot optimizer results:", self.ft_plot_results)
        bkg_form.addRow("Plot clamp diagnostic:", self.ft_plot_clamp_diag)
        bkg_form.addRow("kmax range min:", self.ft_kmax_min)
        bkg_form.addRow("kmax range max:", self.ft_kmax_max)
        bkg_form.addRow("kmin range min:", self.ft_kmin_min)
        bkg_form.addRow("kmin range max:", self.ft_kmin_max)
        bkg_form.addRow("rbkg range min:", self.ft_rbkg_min)
        bkg_form.addRow("rbkg range max:", self.ft_rbkg_max)

        bkg_group.setLayout(bkg_form)
        layout.addWidget(bkg_group)

        # --- FT processing controls ---
        ft_proc_group = QGroupBox("FT Batch Processing")
        ft_proc_form = QFormLayout()

        self.ft_snr_threshold = QDoubleSpinBox()
        self.ft_snr_threshold.setRange(0.0, 100.0)
        self.ft_snr_threshold.setDecimals(3)
        self.ft_snr_threshold.setValue(1.5)

        self.ft_plot_snr_diag = QCheckBox()
        self.ft_plot_snr_diag.setChecked(True)

        self.ft_verify_results = QCheckBox()
        self.ft_verify_results.setChecked(True)

        self.ft_save_outputs = QCheckBox()
        self.ft_save_outputs.setChecked(False)

        ft_proc_form.addRow("SNR threshold:", self.ft_snr_threshold)
        ft_proc_form.addRow("Plot SNR diagnostic:", self.ft_plot_snr_diag)
        ft_proc_form.addRow("Verify results:", self.ft_verify_results)
        ft_proc_form.addRow("Save processed EXAFS:", self.ft_save_outputs)

        ft_proc_group.setLayout(ft_proc_form)
        layout.addWidget(ft_proc_group)

        # --- Buttons ---
        btn_layout = QHBoxLayout()

        self.run_bkg_opt_button = QPushButton("Run Background Optimizer")
        self.run_bkg_opt_button.clicked.connect(self.run_background_optimizer)

        self.run_ft_batch_button = QPushButton("Run FT Processing Batch")
        self.run_ft_batch_button.clicked.connect(self.run_ft_processing_batch)

        btn_layout.addWidget(self.run_bkg_opt_button)
        btn_layout.addWidget(self.run_ft_batch_button)

        layout.addLayout(btn_layout)

        self.ft_tab.setLayout(layout)

    # =========================================================
    # FEFF TAB
    # =========================================================
    def setup_feff_tab(self):
        layout = QVBoxLayout()

        feff_dir_group = QGroupBox("FEFF Paths")
        feff_dir_form = QFormLayout()

        self.feff_dir_label = QLabel("No FEFF folder selected")
        self.load_feff_button = QPushButton("Load FEFF Folder")
        self.load_feff_button.clicked.connect(self.load_feff)

        feff_dir_form.addRow("FEFF folder:", self.feff_dir_label)
        feff_dir_form.addRow(self.load_feff_button)
        feff_dir_group.setLayout(feff_dir_form)
        layout.addWidget(feff_dir_group)

        fit_group = QGroupBox("FEFF Fitting Controls")
        fit_form = QFormLayout()

        self.feff_category = QLineEdit()
        self.feff_category.setText("Pb-O")

        self.feff_iteration = QSpinBox()
        self.feff_iteration.setRange(1, 20)
        self.feff_iteration.setValue(2)

        self.feff_plot_results = QCheckBox()
        self.feff_plot_results.setChecked(True)

        self.feff_max_paths = QSpinBox()
        self.feff_max_paths.setRange(1, 20)
        self.feff_max_paths.setValue(3)

        self.feff_method_combo = QComboBox()
        self.feff_method_combo.addItems([
            "enhanced_iteration1_fit",
            "iterative_fit_all",
            "iterative_fit_linked_all"
        ])

        fit_form.addRow("FEFF category:", self.feff_category)
        fit_form.addRow("Iteration:", self.feff_iteration)
        fit_form.addRow("Plot results:", self.feff_plot_results)
        fit_form.addRow("Max paths to add:", self.feff_max_paths)
        fit_form.addRow("Method:", self.feff_method_combo)

        fit_group.setLayout(fit_form)
        layout.addWidget(fit_group)

        btn_layout = QHBoxLayout()

        self.run_iter1_button = QPushButton("Run Enhanced Iteration 1 Fit")
        self.run_iter1_button.clicked.connect(self.run_enhanced_iteration1_fit)

        self.run_iter_all_button = QPushButton("Run Iterative Fit All")
        self.run_iter_all_button.clicked.connect(self.run_iterative_fit_all)

        self.run_iter_linked_button = QPushButton("Run Iterative Fit Linked All")
        self.run_iter_linked_button.clicked.connect(self.run_iterative_fit_linked_all)

        btn_layout.addWidget(self.run_iter1_button)
        btn_layout.addWidget(self.run_iter_all_button)
        btn_layout.addWidget(self.run_iter_linked_button)

        layout.addLayout(btn_layout)

        self.feff_tab.setLayout(layout)

    # =========================================================
    # Loaders
    # =========================================================
    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Athena Project Folder")
        if not folder:
            return

        self.folder_label.setText(folder)

        try:
            self.xas = XASmu2r(folder)
            self.log_msg("Loaded projects successfully.")
            self.populate_projects()
        except Exception as e:
            self.log_msg(f"Error loading project folder: {e}")
            QMessageBox.critical(self, "Load Error", str(e))

    def load_feff(self):
        folder = QFileDialog.getExistingDirectory(self, "Select FEFF Folder")
        if not folder:
            return

        self.feff_folder = Path(folder)
        self.feff_dir_label.setText(str(self.feff_folder))
        self.log_msg(f"Loaded FEFF folder: {self.feff_folder}")

    def populate_projects(self):
        self.project_list.clear()
        if not self.xas:
            return

        for proj_name, project in self.xas.projects.items():
            self.project_list.addItem(f"[Project] {proj_name}")
            for group_name in project.groups:
                self.project_list.addItem(f"   └─ {group_name}")

    # =========================================================
    # Preprocessing actions
    # =========================================================
    def run_processing(self):
        if not self.xas:
            self.log_msg("No project loaded.")
            return

        try:
            self.log_msg("Running batch processing...")
            self.xas.batch_process_projects(
                smooth=self.safe_bool(self.smooth_checkbox),
                window_length=self.safe_int(self.window_length),
                polyorder=self.safe_int(self.polyorder),
                optimize_preedge=self.safe_bool(self.optimize_preedge_checkbox),
                pre1=self.safe_float(self.pre1),
                pre2=self.safe_float(self.pre2),
            )
            self.log_msg("Batch processing complete.")
        except Exception as e:
            self.log_msg(f"Batch processing error: {e}")

    def run_fluo(self):
        if not self.xas:
            self.log_msg("No project loaded.")
            return

        try:
            self.log_msg("Applying self-absorption correction...")
            self.xas.apply_self_absorption_correction_all()
            self.log_msg("FLUO correction complete.")
        except Exception as e:
            self.log_msg(f"FLUO correction error: {e}")

    # =========================================================
    # FT actions
    # =========================================================
    def run_background_optimizer(self):
        if not self.xas:
            self.log_msg("No project loaded.")
            return

        try:
            kwargs = {
                "optimize_kmax": self.safe_bool(self.ft_optimize_kmax),
                "optimize_kmin": self.safe_bool(self.ft_optimize_kmin),
                "kmax_range": (self.safe_float(self.ft_kmax_min), self.safe_float(self.ft_kmax_max)),
                "kmin_range": (self.safe_float(self.ft_kmin_min), self.safe_float(self.ft_kmin_max)),
                "rbkg_range": (self.safe_float(self.ft_rbkg_min), self.safe_float(self.ft_rbkg_max)),
                "plot_results": self.safe_bool(self.ft_plot_results),
                "auto_clamp": self.safe_bool(self.ft_auto_clamp),
                "plot_clamp_diagnostic": self.safe_bool(self.ft_plot_clamp_diag),
            }
            self.call_method_if_exists("background_optimizer", kwargs)
            self.log_msg("Background optimization complete.")
        except Exception as e:
            self.log_msg(f"Background optimizer error: {e}")

    def run_ft_processing_batch(self):
        if not self.xas:
            self.log_msg("No project loaded.")
            return

        try:
            kwargs = {
                "snr_threshold": self.safe_float(self.ft_snr_threshold),
                "plot_snr_diagnostic": self.safe_bool(self.ft_plot_snr_diag),
                "verify_results": self.safe_bool(self.ft_verify_results),
            }

            self.call_method_if_exists("ft_processing_batch", kwargs)
            self.log_msg("FT processing batch complete.")

            if self.safe_bool(self.ft_verify_results) and hasattr(self.xas, "verify_ft_processing"):
                try:
                    self.xas.verify_ft_processing()
                    self.log_msg("FT verification complete.")
                except Exception as e:
                    self.log_msg(f"FT verification warning: {e}")

            if self.safe_bool(self.ft_save_outputs) and hasattr(self.xas, "save_processed_exafs"):
                outdir = QFileDialog.getExistingDirectory(self, "Select Output Folder for Processed EXAFS")
                if outdir:
                    try:
                        self.xas.save_processed_exafs(outdir)
                        self.log_msg(f"Saved processed EXAFS to: {outdir}")
                    except Exception as e:
                        self.log_msg(f"Save processed EXAFS error: {e}")

        except Exception as e:
            self.log_msg(f"FT processing batch error: {e}")

    # =========================================================
    # FEFF actions
    # =========================================================
    def ensure_feff_ready(self):
        if not self.xas:
            self.log_msg("No project loaded.")
            return False
        if self.feff_folder is None:
            self.log_msg("No FEFF folder selected.")
            return False
        return True

    def run_enhanced_iteration1_fit(self):
        if not self.ensure_feff_ready():
            return

        try:
            kwargs = {
                "feff_base_dir": str(self.feff_folder),
                "plot_results": self.safe_bool(self.feff_plot_results),
                "feff_category": self.feff_category.text().strip(),
                "max_paths_to_add": self.safe_int(self.feff_max_paths),
            }
            self.call_method_if_exists("enhanced_iteration1_fit", kwargs)
            self.log_msg("Enhanced iteration 1 fit complete.")
        except Exception as e:
            self.log_msg(f"Enhanced iteration 1 fit error: {e}")

    def run_iterative_fit_all(self):
        if not self.ensure_feff_ready():
            return

        try:
            kwargs = {
                "feff_category": self.feff_category.text().strip(),
                "iteration": self.safe_int(self.feff_iteration),
                "feff_base_dir": str(self.feff_folder),
                "plot_results": self.safe_bool(self.feff_plot_results),
                "max_paths_to_add": self.safe_int(self.feff_max_paths),
            }
            self.call_method_if_exists("iterative_fit_all", kwargs)
            self.log_msg("Iterative fit all complete.")
        except Exception as e:
            self.log_msg(f"Iterative fit all error: {e}")

    def run_iterative_fit_linked_all(self):
        if not self.ensure_feff_ready():
            return

        try:
            kwargs = {
                "feff_category": self.feff_category.text().strip(),
                "iteration": self.safe_int(self.feff_iteration),
                "feff_base_dir": str(self.feff_folder),
                "plot_results": self.safe_bool(self.feff_plot_results),
                "max_paths_to_add": self.safe_int(self.feff_max_paths),
            }
            self.call_method_if_exists("iterative_fit_linked_all", kwargs)
            self.log_msg("Iterative linked fit all complete.")
        except Exception as e:
            self.log_msg(f"Iterative linked fit all error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XASGUI()
    window.show()
    sys.exit(app.exec_())
