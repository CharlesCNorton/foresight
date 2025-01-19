# Foresight: A Friendly Terminal-Based Interface for HeinSight

**Foresight** is a user-friendly, line-based menu script that wraps and extends the functionality of the [HeinSight](https://gitlab.com/heingroup/heinsight4.0) computer vision system for analyzing chemical experiments. It allows you to selectively enable/disable bounding boxes for each class (vessel, solid, residue, empty, homo, hetero), toggle real-time metric overlays (turbidity, color/hue, volume fraction), apply debouncing to reduce flicker, show side-by-side comparisons, and more—all without writing additional code.

---

## Inspiration

HeinSight was developed to automatically detect phase changes in chemistry experiments (e.g., solutions transitioning from homogeneous to heterogeneous), using two YOLO-based models:
1. **Vessel Detection**  
2. **Contents (Phases) Detection**

While **HeinSight** itself offers a straightforward API and command-line usage, **Foresight** takes it a step further by providing an **interactive, menu-driven interface**. Users can configure detection thresholds, pick input files via a simple GUI dialog, optionally open the final annotated media, and control many overlay settings—making HeinSight more accessible and customizable for diverse use cases.

---

## HeinSight Installation

Before using **Foresight**, you need to install (or otherwise have access to) the **HeinSight** package and its dependencies.

### 1. Clone HeinSight (Optional if you have it installed)
```bash
git clone https://gitlab.com/heingroup/heinsight4.0.git
cd heinsight4.0
```

### 2. Install Requirements
HeinSight relies on Ultralytics YOLOv8, OpenCV, PyTorch, etc. You can install everything via:
```bash
pip install -r requirements.txt
```
This ensures PyTorch, Ultralytics YOLO, and other necessary libraries are set up.

### 3. Verify the Models
HeinSight expects two `.pt` files (e.g., `best_vessel.pt` and `best_content.pt`) for vessel and contents detection.  
Place them in a known location, for example:
```
heinsight4.0
└── heinsight
    └── models
        ├── best_vessel.pt
        └── best_content.pt
```

### 4. Check HeinSight Imports
In Python:
```python
from heinsight import HeinSight
```
If you can import `HeinSight` without errors, you’re ready to go!

---

## Using Foresight

1. **Download/Copy `foresight.py`**  
   Place it inside or alongside your HeinSight repository folder, or ensure that the `heinsight/` package is discoverable in your Python path.

2. **Run `foresight.py`**  
   Open a command prompt/terminal and type:
   ```bash
   python foresight.py
   ```
   This will launch a line-based menu with 13 options:

   - **Pick Input File** (image or video)
   - **Pick Output Directory** for annotated results
   - **Toggle** bounding boxes per class (vessel, solid, residue, etc.)
   - **Toggle** T/C/V overlays, side-by-side mode, color patches, debouncing, and more
   - **Run Detection** to generate an annotated image or video

3. **Select Your Options**  
   Foresight guides you with prompts. You can:
   - **Set thresholds** for turbidity, color, volume fraction, and detection confidence
   - **Adjust annotation fonts,** e.g., making bounding box labels larger
   - **Auto-open** the annotated file after processing
   - **Hide** bounding boxes for classes you’re not interested in
   - **Enable** (or disable) T/C/V overlays individually
   - **Use** real-time debouncing for T/C/V to reduce flicker across frames

4. **Check the Output**  
   By default, your annotated file is saved as `<input_basename>_annotated.png` (for images) or `.mp4` (for videos). If “Auto-Open” is **ON**, it will open in your default image/video viewer after detection completes.

---

## Example Walkthrough

1. **Launch the Script**  
   ```bash
   python foresight.py
   ```
2. **Menu Option 5**: Reload YOLO Models  
   - Ensures HeinSight loads `best_vessel.pt` and `best_content.pt`.
3. **Menu Option 2**: Pick Input File  
   - A Tkinter dialog lets you browse and select a `.png` or `.mp4`.
4. **Menu Option 3**: Pick Output Directory  
   - Or skip if you’re fine saving in the same folder as your input.
5. **Menu Option 12**: Label Visibility Toggles  
   - Turn off “solid” or “residue” if you don’t want those bounding boxes.
   - Toggle metric overlays (T for turbidity, C for color, V for volume).
6. **Menu Option 7**: Run Detection & Overlay  
   - The script processes frames, draws bounding boxes, and merges them if side-by-side is enabled.

---

## FAQ

1. **Why No CSV Logs?**  
   We designed Foresight as a purely visual interface—annotated frames only. If you want CSV logs for turbidity/color, use HeinSight’s built-in `heinsight.run(...)` approach instead.

2. **Performance Concerns**  
   If your video is very large or if you enable many overlays, frame processing can slow down. Consider reducing resolution or toggling fewer features.

3. **Debouncing**  
   The default “3-frame rule” helps avoid flicker in T/C/V overlays when values fluctuate near thresholds. You can disable this if you prefer raw readings.

---

## License

The code here is shared under the MIT license.

---

### Credits

- **HeinSight** research team for the YOLO-based vessel/contents detection system.
- **ChatGPT** for bridging user requirements into a flexible, interactive interface.  

Happy detecting!
```
