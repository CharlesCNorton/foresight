"""
foresight.py

A user-friendly line-based menu for HeinSight with 13 menu options/features, including:
 1. Per-class bounding box/label toggles: Vessel, Solid, Residue, Empty, Homo, Hetero
 2. Per-metric toggles for T (turbidity), C (color/hue), V (volume fraction)
 3. Optional color patch for measured hue
 4. Side-by-side outputs (original + annotated)
 5. Adjustable annotation font size
 6. Optional top-left summary of bounding boxes
 7. Debouncing mechanism for T/C/V to reduce flicker ("3-frame rule")
 8. Auto-open the final annotated media
 9. Advanced (T/C/V) overlay on/off
10. Thresholds for T/C/V + detection confidence
11. Dynamic import of 'heinsight'
12. No CSV logs: bounding boxes + overlays in final media only
13. Ability to pick input file / output directory for detection

Author: ChatGPT (prompted by user)
"""

import sys
import os
import traceback
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np

# ----- COLORAMA for colored text in terminal -----
try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


def ctext(text, color=None, style=None):
    """
    Utility: Return colored text if colorama is available, otherwise plain text.
    color can be Fore.RED, Fore.GREEN, etc.
    style can be Style.DIM, Style.NORMAL, Style.BRIGHT, etc.
    """
    if not COLORAMA_AVAILABLE or (not color and not style):
        return text
    prefix = ""
    if style:
        prefix += style
    if color:
        prefix += color
    suffix = Style.RESET_ALL
    return f"{prefix}{text}{suffix}"


# -------------------------------------------------------------------------
#  FIRST: Force selection of the "heinsight4.0" folder and verify structure
# -------------------------------------------------------------------------
heinsight_dir = None       # The path to ".../heinsight4.0/heinsight" subfolder
HeinSight = None           # The actual class from heinsight.py
heinsight_obj = None       # HeinSight instance
heinsight_module_available = False

def enforce_heinsight_directory():
    """
    Force the user to select the folder containing 'heinsight4.0',
    then verify we can find 'heinsight/heinsight.py'.
    """
    global heinsight_dir

    # Use tkinter to pick the "heinsight4.0" folder
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Foresight", "Select the 'heinsight4.0' folder.")
    chosen_dir = filedialog.askdirectory(title="Select HeinSight Root Folder (heinsight4.0)")
    root.destroy()

    if not chosen_dir or not os.path.isdir(chosen_dir):
        print(ctext("ERROR: No valid 'heinsight4.0' directory selected.", color=Fore.RED))
        sys.exit(1)

    # we expect a subfolder: chosen_dir/heinsight/heinsight.py
    sub_heinsight = os.path.join(chosen_dir, "heinsight")
    heinsight_py = os.path.join(sub_heinsight, "heinsight.py")
    if not os.path.isfile(heinsight_py):
        print(ctext("ERROR: 'heinsight4.0' folder missing 'heinsight/heinsight.py'.", color=Fore.RED))
        print(ctext(f"Expected path: {heinsight_py}", color=Fore.RED))
        sys.exit(1)

    # Looks good => store that subfolder in a global
    heinsight_dir = sub_heinsight
    # Insert at front of sys.path so Python sees it first
    sys.path.insert(0, heinsight_dir)
    print(ctext(f"HeinSight folder set to: {heinsight_dir}", color=Fore.GREEN))


import importlib.util

def attempt_import_heinsight():
    """
    Attempt to load 'HeinSight' from `heinsight.py`.
    """
    global heinsight_module_available, HeinSight, heinsight_dir

    heinsight_py = os.path.join(heinsight_dir, "heinsight.py")
    try:
        spec = importlib.util.spec_from_file_location("heinsight", heinsight_py)
        heinsight_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(heinsight_module)

        HeinSight = heinsight_module.HeinSight
        heinsight_module_available = True
        print(ctext("SUCCESS: HeinSight imported successfully.", color=Fore.GREEN))
    except Exception as e:
        print(ctext(f"ERROR importing HeinSight: {e}", color=Fore.RED))
        sys.exit(1)


# -------------------------------------------------------------------------
#  GLOBAL STATE FOR YOUR MENU/FEATURES
# -------------------------------------------------------------------------
thresholds = None                  # (turb, color, volume, conf)
input_file = None                  # path to user-chosen image or video
output_dir = None                  # path to user-chosen output directory

# Overlay and output behavior
advanced_overlay = True            # Whether to measure T/C/V at all
auto_open_output = False           # Whether to auto-open annotated file after processing
side_by_side = False               # Show original & annotated side-by-side
annotation_font_scale = 0.6        # Font scale for bounding-box labels
show_top_left_list = False         # If True, show a text summary of boxes at top-left
show_color_patch = False           # If True, draw a color patch for hue
debounce_enabled = False           # If True, apply T/C/V debouncing (3-frame rule)

# Per-class bounding box/label toggles
show_label_vessel = True
show_label_solid = True
show_label_residue = True
show_label_empty = True
show_label_homo = True
show_label_hetero = True

# Per-metric overlays
show_metric_t = True  # Turbidity
show_metric_c = True  # Color/Hue
show_metric_v = True  # Volume Fraction

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# For storing bounding-box strings from the last processed frame
last_frame_bboxes = []
# For debouncing T/C/V across frames in a video
debounce_map = {}

# --- ADDED FOR CPU-ONLY TOGGLE ---
cpu_only_inference = False  # Whether to force CPU-only inference


def main():
    """
    Main entry point after HeinSight is confirmed to be imported.
    We show a hierarchical menu with submenus.
    """
    print(ctext("=== Foresight Terminal Menu ===", color=Fore.CYAN, style=Style.BRIGHT))
    print("Offers a line-based menu for controlling HeinSight detection with advanced overlays, "
          "side-by-side output, annotation font adjustments, T/C/V debouncing, per-class toggles, etc.\n")

    while True:
        show_main_menu()
        choice = input(ctext("Enter your choice: ", color=Fore.YELLOW)).strip()
        if choice == "1":
            submenu_setup_and_run()
        elif choice == "2":
            submenu_overlays_and_thresholds()
        elif choice == "3":
            submenu_heinsight_management()
        elif choice == "4":
            handle_display_config()
        elif choice == "5":
            print(ctext("Goodbye.", color=Fore.CYAN))
            break
        else:
            print(ctext("Invalid choice. Please enter a number 1-5.", color=Fore.RED))


# =============================================================================
#   TOP-LEVEL MENU
# =============================================================================

def show_main_menu():
    print(ctext("\n---------------------------------------", color=Fore.CYAN))
    print(ctext("Foresight Main Menu - Choose an option:", color=Fore.GREEN))
    print(ctext("1) ", color=Fore.YELLOW) + "Setup & Run Detection")
    print(ctext("2) ", color=Fore.YELLOW) + "Overlays & Thresholds")
    print(ctext("3) ", color=Fore.YELLOW) + "HeinSight Management")
    print(ctext("4) ", color=Fore.YELLOW) + "Display Current Configuration")
    print(ctext("5) ", color=Fore.YELLOW) + "Quit")
    print(ctext("---------------------------------------", color=Fore.CYAN))


# =============================================================================
#   SUB-MENU 1: SETUP & RUN DETECTION
# =============================================================================

def submenu_setup_and_run():
    while True:
        print(ctext("\n=== Setup & Run Detection ===", color=Fore.GREEN, style=Style.BRIGHT))
        print(ctext("1) ", color=Fore.YELLOW) + f"Pick Input File (current: {input_file if input_file else 'NONE'})")
        print(ctext("2) ", color=Fore.YELLOW) + f"Pick Output Directory (current: {output_dir if output_dir else 'NONE'})")
        print(ctext("3) ", color=Fore.YELLOW) + "Set or Update Thresholds (turb, color, volume, confidence)")
        print(ctext("4) ", color=Fore.YELLOW) + f"Toggle Auto-Open Output (currently: {'ON' if auto_open_output else 'OFF'})")
        print(ctext("5) ", color=Fore.YELLOW) + "Run Detection")
        print(ctext("6) ", color=Fore.YELLOW) + "Return to Main Menu")

        choice = input(ctext("Enter your choice: ", color=Fore.YELLOW)).strip()
        if choice == "1":
            handle_pick_input_file()
        elif choice == "2":
            handle_pick_output_directory()
        elif choice == "3":
            handle_set_thresholds()
        elif choice == "4":
            handle_toggle_auto_open()
        elif choice == "5":
            handle_run_detection()
        elif choice == "6":
            break
        else:
            print(ctext("Invalid choice (1-6).", color=Fore.RED))


# =============================================================================
#   SUB-MENU 2: OVERLAYS & THRESHOLDS
# =============================================================================

def submenu_overlays_and_thresholds():
    while True:
        print(ctext("\n=== Overlays & Thresholds ===", color=Fore.GREEN, style=Style.BRIGHT))
        print(ctext("1) ", color=Fore.YELLOW) + f"Toggle Advanced Overlay (T/C/V) (currently: {'ON' if advanced_overlay else 'OFF'})")
        print(ctext("2) ", color=Fore.YELLOW) + f"Toggle Side-by-Side Output (currently: {'ON' if side_by_side else 'OFF'})")
        print(ctext("3) ", color=Fore.YELLOW) + f"Set Annotation Font Scale (currently: {annotation_font_scale})")
        print(ctext("4) ", color=Fore.YELLOW) + f"Toggle Top-Left Summary of Boxes (currently: {'ON' if show_top_left_list else 'OFF'})")
        print(ctext("5) ", color=Fore.YELLOW) + f"Toggle Color Patch for Hue (currently: {'ON' if show_color_patch else 'OFF'})")
        print(ctext("6) ", color=Fore.YELLOW) + f"Toggle Value Debouncing (3-frame) (currently: {'ON' if debounce_enabled else 'OFF'})")
        print(ctext("7) ", color=Fore.YELLOW) + "Per-Class Label Visibility Toggles (vessel, solid, residue, empty, homo, hetero)")
        print(ctext("8) ", color=Fore.YELLOW) + "T/C/V Visibility Toggles")
        print(ctext("9) ", color=Fore.YELLOW) + "Return to Main Menu")

        choice = input(ctext("Enter your choice: ", color=Fore.YELLOW)).strip()
        if choice == "1":
            handle_toggle_overlay()
        elif choice == "2":
            handle_toggle_side_by_side()
        elif choice == "3":
            handle_set_font_scale()
        elif choice == "4":
            handle_toggle_top_left_list()
        elif choice == "5":
            handle_toggle_color_patch()
        elif choice == "6":
            handle_toggle_debouncing()
        elif choice == "7":
            submenu_label_visibility()
        elif choice == "8":
            submenu_metric_visibility()
        elif choice == "9":
            break
        else:
            print(ctext("Invalid choice (1-9).", color=Fore.RED))


def handle_toggle_side_by_side():
    global side_by_side
    side_by_side = not side_by_side
    print(ctext(f"Side-by-Side is now {'ON' if side_by_side else 'OFF'}.", color=Fore.GREEN))

def handle_set_font_scale():
    global annotation_font_scale
    val_str = input(ctext(f"Enter new font scale (float), current={annotation_font_scale}: ", color=Fore.YELLOW))
    try:
        if val_str.strip():
            annotation_font_scale = float(val_str)
            print(ctext(f"Annotation font scale set to {annotation_font_scale}.", color=Fore.GREEN))
    except ValueError:
        print(ctext("Invalid float, no change made.", color=Fore.RED))

def handle_toggle_top_left_list():
    global show_top_left_list
    show_top_left_list = not show_top_left_list
    print(ctext(f"Top-Left List is now {'ON' if show_top_left_list else 'OFF'}.", color=Fore.GREEN))

def handle_toggle_color_patch():
    global show_color_patch
    show_color_patch = not show_color_patch
    print(ctext(f"Color Patch for Hue is now {'ON' if show_color_patch else 'OFF'}.", color=Fore.GREEN))

def handle_toggle_overlay():
    global advanced_overlay
    advanced_overlay = not advanced_overlay
    print(ctext(f"Advanced overlay is now {'ON' if advanced_overlay else 'OFF'}.", color=Fore.GREEN))

def handle_toggle_debouncing():
    global debounce_enabled
    debounce_enabled = not debounce_enabled
    print(ctext(f"Value Debouncing is now {'ON' if debounce_enabled else 'OFF'}.", color=Fore.GREEN))


def submenu_label_visibility():
    global show_label_vessel, show_label_solid, show_label_residue, show_label_empty
    global show_label_homo, show_label_hetero

    while True:
        print(ctext("\n=== Per-Class Label Visibility ===", color=Fore.GREEN, style=Style.BRIGHT))
        print(ctext("1) ", color=Fore.YELLOW) + f"Toggle Vessel (currently: {'ON' if show_label_vessel else 'OFF'})")
        print(ctext("2) ", color=Fore.YELLOW) + f"Toggle Solid (currently: {'ON' if show_label_solid else 'OFF'})")
        print(ctext("3) ", color=Fore.YELLOW) + f"Toggle Residue (currently: {'ON' if show_label_residue else 'OFF'})")
        print(ctext("4) ", color=Fore.YELLOW) + f"Toggle Empty (currently: {'ON' if show_label_empty else 'OFF'})")
        print(ctext("5) ", color=Fore.YELLOW) + f"Toggle Homo (currently: {'ON' if show_label_homo else 'OFF'})")
        print(ctext("6) ", color=Fore.YELLOW) + f"Toggle Hetero (currently: {'ON' if show_label_hetero else 'OFF'})")
        print(ctext("7) ", color=Fore.YELLOW) + "Return to Overlays Menu")

        choice = input(ctext("Enter your choice: ", color=Fore.YELLOW)).strip()
        if choice == "1":
            show_label_vessel = not show_label_vessel
            print(ctext(f"Vessel Label: {'ON' if show_label_vessel else 'OFF'}", color=Fore.GREEN))
        elif choice == "2":
            show_label_solid = not show_label_solid
            print(ctext(f"Solid Label: {'ON' if show_label_solid else 'OFF'}", color=Fore.GREEN))
        elif choice == "3":
            show_label_residue = not show_label_residue
            print(ctext(f"Residue Label: {'ON' if show_label_residue else 'OFF'}", color=Fore.GREEN))
        elif choice == "4":
            show_label_empty = not show_label_empty
            print(ctext(f"Empty Label: {'ON' if show_label_empty else 'OFF'}", color=Fore.GREEN))
        elif choice == "5":
            show_label_homo = not show_label_homo
            print(ctext(f"Homo Label: {'ON' if show_label_homo else 'OFF'}", color=Fore.GREEN))
        elif choice == "6":
            show_label_hetero = not show_label_hetero
            print(ctext(f"Hetero Label: {'ON' if show_label_hetero else 'OFF'}", color=Fore.GREEN))
        elif choice == "7":
            break
        else:
            print(ctext("Invalid choice (1-7).", color=Fore.RED))


def submenu_metric_visibility():
    global show_metric_t, show_metric_c, show_metric_v

    while True:
        print(ctext("\n=== T/C/V Metric Visibility ===", color=Fore.GREEN, style=Style.BRIGHT))
        print(ctext("1) ", color=Fore.YELLOW) + f"Toggle Turbidity (T) (currently: {'ON' if show_metric_t else 'OFF'})")
        print(ctext("2) ", color=Fore.YELLOW) + f"Toggle Color (C) (currently: {'ON' if show_metric_c else 'OFF'})")
        print(ctext("3) ", color=Fore.YELLOW) + f"Toggle Volume (V) (currently: {'ON' if show_metric_v else 'OFF'})")
        print(ctext("4) ", color=Fore.YELLOW) + "Return to Overlays Menu")

        choice = input(ctext("Enter your choice: ", color=Fore.YELLOW)).strip()
        if choice == "1":
            show_metric_t = not show_metric_t
            print(ctext(f"Turbidity (T) Overlay: {'ON' if show_metric_t else 'OFF'}", color=Fore.GREEN))
        elif choice == "2":
            show_metric_c = not show_metric_c
            print(ctext(f"Color (C) Overlay: {'ON' if show_metric_c else 'OFF'}", color=Fore.GREEN))
        elif choice == "3":
            show_metric_v = not show_metric_v
            print(ctext(f"Volume (V) Overlay: {'ON' if show_metric_v else 'OFF'}", color=Fore.GREEN))
        elif choice == "4":
            break
        else:
            print(ctext("Invalid choice (1-4).", color=Fore.RED))


# =============================================================================
#   SUB-MENU 3: HEINSIGHT MANAGEMENT
# =============================================================================

def submenu_heinsight_management():
    while True:
        print(ctext("\n=== HeinSight Management ===", color=Fore.GREEN, style=Style.BRIGHT))
        print(ctext("1) ", color=Fore.YELLOW) + "Pick HeinSight Directory (for dynamic import)")
        print(ctext("2) ", color=Fore.YELLOW) + "Reload YOLO Models")
        # --- ADDED FOR CPU-ONLY TOGGLE ---
        print(ctext("3) ", color=Fore.YELLOW) + f"Toggle CPU Only Inference (currently: {'ON' if cpu_only_inference else 'OFF'})")
        print(ctext("4) ", color=Fore.YELLOW) + "Return to Main Menu")

        choice = input(ctext("Enter your choice: ", color=Fore.YELLOW)).strip()
        if choice == "1":
            handle_pick_heinsight_dir()
        elif choice == "2":
            handle_reload_models()
        elif choice == "3":
            handle_toggle_cpu_only_inference()
        elif choice == "4":
            break
        else:
            print(ctext("Invalid choice (1-4).", color=Fore.RED))

def handle_pick_heinsight_dir():
    """
    Let the user pick a new 'heinsight4.0' folder at runtime (if needed).
    """
    global heinsight_dir, heinsight_module_available, HeinSight
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Foresight", "Select the 'heinsight4.0' folder.")
    chosen_dir = filedialog.askdirectory(title="Select HeinSight Root Folder (heinsight4.0)")
    root.destroy()

    if not chosen_dir or not os.path.isdir(chosen_dir):
        print(ctext("No valid directory selected for heinsight. Aborting.", color=Fore.RED))
        return

    # check subfolder
    sub_heinsight = os.path.join(chosen_dir, "heinsight")
    heinsight_py = os.path.join(sub_heinsight, "heinsight.py")
    if not os.path.isfile(heinsight_py):
        print(ctext("Folder missing 'heinsight/heinsight.py'.", color=Fore.RED))
        return

    # update
    heinsight_dir = sub_heinsight
    sys.path.insert(0, heinsight_dir)
    print(ctext(f"Added to sys.path: {heinsight_dir}", color=Fore.GREEN))

    attempt_import_heinsight()
    if heinsight_module_available:
        print(ctext("HeinSight import succeeded after adding path.", color=Fore.GREEN))
    else:
        print(ctext("HeinSight import still failed. Check folder structure.", color=Fore.RED))

def handle_reload_models():
    global heinsight_obj
    if not heinsight_module_available or HeinSight is None:
        print(ctext("ERROR: HeinSight not imported. Use 'Pick HeinSight Directory' first.", color=Fore.RED))
        return
    if heinsight_dir is None or not os.path.isdir(heinsight_dir):
        print(ctext("ERROR: No valid HeinSight directory set.", color=Fore.RED))
        return

    print(ctext("Reloading YOLO models...", color=Fore.GREEN))
    heinsight_obj = None
    loaded_ok = load_heinsight_models()
    if loaded_ok:
        print(ctext("Models reloaded successfully.", color=Fore.GREEN))
    else:
        print(ctext("Failed to reload models.", color=Fore.RED))

# --- ADDED FOR CPU-ONLY TOGGLE ---
def handle_toggle_cpu_only_inference():
    global cpu_only_inference
    cpu_only_inference = not cpu_only_inference
    print(ctext(f"CPU Only Inference is now {'ON' if cpu_only_inference else 'OFF'}.", color=Fore.GREEN))


# =============================================================================
#   MENU HANDLERS (Setup & Run)
# =============================================================================

def handle_pick_input_file():
    global input_file
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Foresight", "Select an image (jpg, png) or video (mp4, avi, etc.).")
    chosen_file = filedialog.askopenfilename(
        title="Select Image or Video",
        filetypes=[
            ("Image/Video", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.mp4 *.avi *.mkv *.mov"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()

    if chosen_file and os.path.isfile(chosen_file):
        input_file = chosen_file
        print(ctext(f"Input file set to: {input_file}", color=Fore.GREEN))
    else:
        print(ctext("No valid file selected. Input file not changed.", color=Fore.RED))

def handle_pick_output_directory():
    global output_dir
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Foresight", "Select a directory to save your annotated image/video.")
    chosen_dir = filedialog.askdirectory(title="Select Output Directory")
    root.destroy()

    if chosen_dir and os.path.isdir(chosen_dir):
        output_dir = chosen_dir
        print(ctext(f"Output directory set to: {output_dir}", color=Fore.GREEN))
    else:
        print(ctext("No valid directory selected. Output directory not changed.", color=Fore.RED))

def handle_set_thresholds():
    global thresholds
    print(ctext("\n=== Set or Update Thresholds ===", color=Fore.GREEN, style=Style.BRIGHT))
    defaults = {"turb": 50.0, "col": 20.0, "vol": 0.1, "conf": 0.4}

    def ask_float(prompt_str, default_val):
        val_str = input(ctext(f"{prompt_str} [default={default_val}]: ", color=Fore.YELLOW))
        if not val_str.strip():
            return default_val
        try:
            return float(val_str)
        except ValueError:
            print(ctext(f"Invalid number, using default: {default_val}", color=Fore.RED))
            return default_val

    t_val = ask_float("Min turbidity (0-255)", defaults["turb"])
    c_val = ask_float("Min color/Hue (0-180)", defaults["col"])
    v_val = ask_float("Min volume fraction (0-1)", defaults["vol"])
    conf_val = ask_float("Detection confidence (0-1)", defaults["conf"])

    thresholds = (t_val, c_val, v_val, conf_val)
    print(ctext(f"\nUpdated thresholds: {thresholds}\n", color=Fore.GREEN))


def handle_toggle_auto_open():
    global auto_open_output
    auto_open_output = not auto_open_output
    print(ctext(f"Auto-Open Output is now {'ON' if auto_open_output else 'OFF'}.", color=Fore.GREEN))


def handle_run_detection():
    global input_file, output_dir, heinsight_obj, thresholds
    if not heinsight_module_available or HeinSight is None:
        print(ctext("ERROR: HeinSight is not imported. Pick the directory under 'HeinSight Management'.", color=Fore.RED))
        return
    if heinsight_obj is None:
        print(ctext("ERROR: YOLO models not loaded. Use 'HeinSight Management' -> Reload YOLO Models.", color=Fore.RED))
        return
    if not input_file or not os.path.isfile(input_file):
        print(ctext("ERROR: No valid input file. Pick one under 'Setup & Run Detection'.", color=Fore.RED))
        return

    if not output_dir or not os.path.isdir(output_dir):
        fallback_dir = os.path.dirname(input_file)
        print(ctext(f"WARNING: No valid output directory. Using {fallback_dir}", color=Fore.YELLOW))
        out_dir = fallback_dir
    else:
        out_dir = output_dir

    ext = os.path.splitext(input_file)[1].lower()
    is_image = ext in IMAGE_EXTS
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    out_name = f"{base_name}_annotated.png" if is_image else f"{base_name}_annotated.mp4"
    out_path = os.path.join(out_dir, out_name)

    try:
        print(ctext(f"\nRunning detection on: {input_file}", color=Fore.GREEN))
        if is_image:
            success = process_image(input_file, out_path, heinsight_obj, thresholds)
        else:
            success = process_video(input_file, out_path, heinsight_obj, thresholds)

        if success:
            print(ctext(f"Annotated output saved to: {out_path}", color=Fore.GREEN))
            if auto_open_output:
                open_file_in_viewer(out_path)
        else:
            print(ctext("Detection finished, but output was not saved successfully.", color=Fore.RED))
    except Exception as e:
        print(ctext("ERROR during detection:", color=Fore.RED))
        traceback.print_exc()


# =============================================================================
#   LOAD HEINSIGHT MODELS
# =============================================================================

def load_heinsight_models():
    global heinsight_obj, heinsight_module_available, HeinSight, heinsight_dir
    if not heinsight_module_available or HeinSight is None:
        print(ctext("ERROR: 'heinsight' not imported, cannot load YOLO models.", color=Fore.RED))
        return False
    if not heinsight_dir or not os.path.isdir(heinsight_dir):
        print(ctext("ERROR: No valid heinsight_dir set. Cannot load YOLO models.", color=Fore.RED))
        return False

    try:
        print(ctext("Loading YOLO models from HeinSight...", color=Fore.GREEN))
        vessel_path = os.path.join(heinsight_dir, "models", "best_vessel.pt")
        content_path = os.path.join(heinsight_dir, "models", "best_content.pt")

        obj = HeinSight(vial_model_path=vessel_path, contents_model_path=content_path)
        heinsight_obj = obj
        print(ctext("HeinSight YOLO models loaded successfully.\n", color=Fore.GREEN))
        return True
    except Exception as e:
        print(ctext("Failed to load HeinSight YOLO models.", color=Fore.RED))
        traceback.print_exc()
        heinsight_obj = None
        return False


# =============================================================================
#   IMAGE / VIDEO PROCESSING
# =============================================================================

last_frame_bboxes = []
debounce_map = {}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def process_image(file_path, out_path, h_obj, thr):
    frame = cv2.imread(file_path)
    if frame is None:
        print(ctext(f"ERROR: Could not read image: {file_path}", color=Fore.RED))
        return False

    annotated = run_detection_on_frame(frame, h_obj, thr)
    if annotated is None:
        annotated = frame

    if side_by_side:
        side = combine_side_by_side(frame, annotated)
        annotated = side

    if show_top_left_list:
        annotated = draw_top_left_list(annotated, last_frame_bboxes, annotation_font_scale * 2.0)

    saved = cv2.imwrite(out_path, annotated)
    return bool(saved)


def process_video(file_path, out_path, h_obj, thr):
    global debounce_map
    debounce_map.clear()  # start fresh for this video

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(ctext(f"ERROR: Cannot open video: {file_path}", color=Fore.RED))
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_width = width * 2 if side_by_side else width
    out_height = height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 25, (out_width, out_height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        annotated = run_detection_on_frame(frame, h_obj, thr)
        if annotated is None:
            annotated = frame

        if side_by_side:
            annotated = combine_side_by_side(frame, annotated)

        if show_top_left_list:
            annotated = draw_top_left_list(annotated, last_frame_bboxes, annotation_font_scale * 2.0)

        out_vid.write(annotated)
        progress_bar(frame_idx, total_frames)

    cap.release()
    out_vid.release()
    if total_frames > 0:
        print()  # newline
    return True


def run_detection_on_frame(frame, h_obj, thr):
    global last_frame_bboxes
    last_frame_bboxes = []

    conf_thr = 0.4
    if thr is not None:
        conf_thr = thr[3]  # detection confidence

    # --- ADDED FOR CPU-ONLY TOGGLE ---
    # If cpu_only_inference is True, we pass device='cpu', otherwise 'cuda:0'
    device_for_inference = 'cpu' if cpu_only_inference else 'cuda:0'

    # Vessel detection
    v_preds = h_obj.vial_model.predict(
        frame,
        conf=conf_thr,
        verbose=False,
        device=device_for_inference  # ensure CPU if toggled
    )
    v_boxes = v_preds[0].boxes
    if len(v_boxes) == 0:
        return None

    annotated = frame.copy()
    h, w = annotated.shape[:2]
    all_boxes = []

    for vbox in v_boxes:
        x1v, y1v, x2v, y2v, confv, vessel_label = parse_box_info(vbox, h_obj.vial_model.names)

        # Check if user toggled vessel off
        if vessel_label.lower() == "vessel" and not show_label_vessel:
            pass
        else:
            x1v, x2v = sorted([max(0, min(w, x1v)), max(0, min(w, x2v))])
            y1v, y2v = sorted([max(0, min(h, y1v)), max(0, min(h, y2v))])
            v_txt = f"{vessel_label} {confv:.2f}"
            all_boxes.append((x1v, y1v, x2v, y2v, v_txt))

        # Content detection
        vessel_crop = annotated[y1v:y2v, x1v:x2v]
        if vessel_crop.size == 0:
            continue

        c_preds = h_obj.contents_model.predict(
            vessel_crop,
            conf=conf_thr,
            verbose=False,
            device=device_for_inference  # ensure CPU if toggled
        )
        c_boxes = c_preds[0].boxes

        for cbox in c_boxes:
            x1c, y1c, x2c, y2c, confc, label_c = parse_box_info(cbox, h_obj.contents_model.names)
            x1c += x1v
            x2c += x1v
            y1c += y1v
            y2c += y1v
            x1c, x2c = sorted([max(0, min(w, x1c)), max(0, min(w, x2c))])
            y1c, y2c = sorted([max(0, min(h, y1c)), max(0, min(h, y2c))])

            # skip if user toggled label off
            if not class_label_is_visible(label_c):
                continue

            c_txt = f"{label_c} {confc:.2f}"

            # measure T/C/V if advanced_overlay & it's a "homo"/"hetero"
            if advanced_overlay and is_liquid_label(label_c):
                measure_str = measure_liquid_overlay(annotated, x1c, y1c, x2c, y2c, h, thr)
                if measure_str:
                    c_txt += " | " + measure_str

            all_boxes.append((x1c, y1c, x2c, y2c, c_txt))

    # Draw bounding boxes
    font_scale = annotation_font_scale
    for (bx1, by1, bx2, by2, text_label) in all_boxes:
        color = (0, 255, 255) if "vessel" in text_label.lower() else (0, 0, 255)
        cv2.rectangle(annotated, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)
        cv2.putText(annotated, text_label, (int(bx1), max(int(by1)-5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    last_frame_bboxes = [b[4] for b in all_boxes]
    return annotated


def parse_box_info(box, class_names):
    arr = box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, arr[:4])
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])
    label = class_names[cls_id]
    return x1, y1, x2, y2, conf, label


def class_label_is_visible(label_name):
    lbl = label_name.lower()
    if "vessel" in lbl:
        return show_label_vessel
    elif "solid" in lbl:
        return show_label_solid
    elif "residue" in lbl:
        return show_label_residue
    elif "empty" in lbl:
        return show_label_empty
    elif "homo" in lbl:
        return show_label_homo
    elif "hetero" in lbl:
        return show_label_hetero
    return True


def is_liquid_label(label):
    return label.lower().startswith("homo") or label.lower().startswith("hetero")


def measure_liquid_overlay(frame, x1, y1, x2, y2, full_height, thr):
    cropped = frame[int(y1):int(y2), int(x1):int(x2)]
    if cropped.size == 0:
        return ""

    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv[:, :, 0])   # 0..180
    avg_val = np.mean(hsv[:, :, 2])   # 0..255 => turbidity
    box_height = (y2 - y1)
    vol_frac = box_height / full_height if full_height > 0 else 0

    if show_color_patch:
        draw_color_patch(frame, int(x2), int(y1), avg_hue)

    # thresholds if any
    min_turb, min_col, min_vol = 0, 0, 0
    if thr:
        min_turb, min_col, min_vol, _ = thr

    do_show_t = (advanced_overlay and show_metric_t and (avg_val >= min_turb))
    do_show_c = (advanced_overlay and show_metric_c and (avg_hue >= min_col))
    do_show_v = (advanced_overlay and show_metric_v and (vol_frac >= min_vol))

    # If no debouncing, just build text
    if not debounce_enabled:
        return build_label_str(avg_val, avg_hue, vol_frac, do_show_t, do_show_c, do_show_v)

    # Debouncing with 3-frame rule
    box_key = f"{x1}:{y1}:{x2}:{y2}"
    if box_key not in debounce_map:
        debounce_map[box_key] = {
            'T': {'last_val': 0.0, 'zero_count': 3},
            'C': {'last_val': 0.0, 'zero_count': 3},
            'V': {'last_val': 0.0, 'zero_count': 3},
        }

    entry_t = debounce_map[box_key]['T']
    entry_c = debounce_map[box_key]['C']
    entry_v = debounce_map[box_key]['V']

    # T
    if do_show_t:
        entry_t['zero_count'] = 0
        entry_t['last_val'] = avg_val
    else:
        entry_t['zero_count'] += 1
        if entry_t['zero_count'] >= 3:
            entry_t['last_val'] = 0.0

    # C
    if do_show_c:
        entry_c['zero_count'] = 0
        entry_c['last_val'] = avg_hue
    else:
        entry_c['zero_count'] += 1
        if entry_c['zero_count'] >= 3:
            entry_c['last_val'] = 0.0

    # V
    if do_show_v:
        entry_v['zero_count'] = 0
        entry_v['last_val'] = vol_frac
    else:
        entry_v['zero_count'] += 1
        if entry_v['zero_count'] >= 3:
            entry_v['last_val'] = 0.0

    t_val = entry_t['last_val']
    c_val = entry_c['last_val']
    v_val = entry_v['last_val']

    final_show_t = (t_val > 0.0)
    final_show_c = (c_val > 0.0)
    final_show_v = (v_val > 0.0)

    return build_label_str(t_val, c_val, v_val, final_show_t, final_show_c, final_show_v)


def build_label_str(turb, hue, vol, show_t, show_c, show_v):
    parts = []
    if show_t:
        parts.append(f"T={turb:.1f}")
    if show_c:
        parts.append(f"C={hue:.1f}")
    if show_v:
        parts.append(f"V={vol:.2f}")
    return ", ".join(parts)


def draw_color_patch(frame, x_left, y_top, hue):
    patch_size = 20
    color_hsv = np.uint8([[[hue, 255, 255]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
    b, g, r = int(color_bgr[0, 0, 0]), int(color_bgr[0, 0, 1]), int(color_bgr[0, 0, 2])

    x2 = x_left + patch_size
    y2 = y_top + patch_size
    cv2.rectangle(frame, (x_left, y_top), (x2, y2), (b, g, r), -1)


def combine_side_by_side(original, annotated):
    h1, w1 = original.shape[:2]
    h2, w2 = annotated.shape[:2]
    if h1 != h2:
        annotated = cv2.resize(annotated, (w2, h1))
    return np.hstack((original, annotated))


def draw_top_left_list(frame, box_lines, font_scale):
    overlay = frame.copy()
    x, y = 10, 30
    for line in box_lines:
        cv2.putText(overlay, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 3, cv2.LINE_AA)
        y += int(30 * font_scale + 10)
    return overlay


def progress_bar(current, total):
    if total <= 0:
        return
    bar_len = 50
    frac = current / total
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\rProcessing video frames: [{bar}] {current}/{total}", end="", flush=True)


# =============================================================================
#   DISPLAY CONFIG
# =============================================================================

def handle_display_config():
    global advanced_overlay, auto_open_output, side_by_side
    global annotation_font_scale, show_top_left_list, show_color_patch
    global debounce_enabled, thresholds, input_file, output_dir, heinsight_obj, heinsight_dir
    # --- ADDED FOR CPU-ONLY TOGGLE ---
    global cpu_only_inference

    print(ctext("\n=== Current Configuration ===", color=Fore.CYAN, style=Style.BRIGHT))
    if heinsight_module_available:
        print(ctext("HeinSight Import: AVAILABLE", color=Fore.GREEN))
    else:
        print(ctext("HeinSight Import: NOT AVAILABLE", color=Fore.RED))

    print(f"HeinSight Directory: {heinsight_dir if heinsight_dir else '(none)'}")

    if heinsight_obj is None:
        print(ctext("YOLO Models: Not loaded or reloading needed.", color=Fore.RED))
    else:
        print(ctext("YOLO Models: Loaded", color=Fore.GREEN))

    if thresholds is None:
        print("Thresholds: (none) => all T/C/V displayed")
    else:
        print(f"Thresholds: {thresholds}")

    print(f"Input File: {input_file if input_file else '(none)'}")
    print(f"Output Dir: {output_dir if output_dir else '(none)'}")

    print(f"Advanced Overlay: {'ON' if advanced_overlay else 'OFF'}")
    print(f"Auto-Open Output: {'ON' if auto_open_output else 'OFF'}")
    print(f"Side-by-Side Output: {'ON' if side_by_side else 'OFF'}")
    print(f"Annotation Font Scale: {annotation_font_scale}")
    print(f"Top-Left List: {'ON' if show_top_left_list else 'OFF'}")
    print(f"Color Patch for Hue: {'ON' if show_color_patch else 'OFF'}")
    print(f"Debouncing for T/C/V: {'ON' if debounce_enabled else 'OFF'}")

    # --- ADDED FOR CPU-ONLY TOGGLE ---
    print(f"CPU Only Inference: {'ON' if cpu_only_inference else 'OFF'}")

    print(ctext("\n--- Class Label Visibility ---", color=Fore.CYAN))
    print(f"Vessel: {'ON' if show_label_vessel else 'OFF'}")
    print(f"Solid: {'ON' if show_label_solid else 'OFF'}")
    print(f"Residue: {'ON' if show_label_residue else 'OFF'}")
    print(f"Empty: {'ON' if show_label_empty else 'OFF'}")
    print(f"Homo: {'ON' if show_label_homo else 'OFF'}")
    print(f"Hetero: {'ON' if show_label_hetero else 'OFF'}")

    print(ctext("--- Metric Visibility (T/C/V) ---", color=Fore.CYAN))
    print(f"Turbidity (T): {'ON' if show_metric_t else 'OFF'}")
    print(f"Color/Hue (C): {'ON' if show_metric_c else 'OFF'}")
    print(f"Volume Fraction (V): {'ON' if show_metric_v else 'OFF'}")

    print(ctext("================================", color=Fore.CYAN))


# =============================================================================
#   FILE VIEWER UTILITY
# =============================================================================

def open_file_in_viewer(filepath):
    print(ctext(f"Opening file in viewer: {filepath}", color=Fore.CYAN))
    if not os.path.isfile(filepath):
        print(ctext(f"Cannot open. File does not exist: {filepath}", color=Fore.RED))
        return

    if os.name == "nt":
        os.startfile(filepath)
    else:
        try:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.run([opener, filepath])
        except Exception as e:
            print(ctext(f"Failed to open: {e}", color=Fore.RED))


# =============================================================================
#   LAUNCH
# =============================================================================

if __name__ == "__main__":
    # 1) Force the user to pick the 'heinsight4.0' folder
    enforce_heinsight_directory()
    # 2) Attempt to import HeinSight
    attempt_import_heinsight()
    # 3) Enter main menu
    main()
