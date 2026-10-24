# Dataset audit outputs

Run from PowerShell with `python -B C:\MMA3001\project\dataset_audit.py`.
The script reads the original dataset without modifying it.

- `split_summary.csv`: image and label file counts, annotation totals, empty label counts, missing pairs, and invalid annotation counts per split.
- `class_distribution.csv`: annotation count per class per split.
- `issues.csv`: file and one-based line locations for detected problems; line 0 denotes a file-level issue.
- `zero_example_classes.csv`: classes absent from validation or test; headers only means none are absent.
- `class_distribution.png`: grouped bar chart of the class counts.

`total_boxes` counts nonempty label lines, including malformed annotations. `valid_boxes` counts five-field rows with valid class IDs and normalized coordinates. Per-class counts include five-field rows with valid class IDs even if their coordinates are invalid. Class IDs must be integer-valued numbers from 0 through 4. Coordinates must be finite, centers in [0,1], and widths/heights in (0,1]. Boxes whose edges extend outside the image are reported separately as warnings. Empty labels are counted, not automatically treated as errors: they may represent background images. Missing labels are reported for review, although YOLO may also treat such images as background.

Pairs are matched by relative path without the extension, within each split. This audit examines filenames and annotation text; it does not decode images, verify visual annotation correctness, or detect duplicate image contents and split leakage.
