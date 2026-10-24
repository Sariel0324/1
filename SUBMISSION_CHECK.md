# Submission preparation verification

- Main project was not an initialized Git repository. No main-project commit,
  GitHub remote or upload was created.
- Git ignore rules selected 97 files, 6,918,760 bytes at the packaging check
  (before adding this verification note). No .pt weights, dataset copies or JPG
  prediction overlays were selected.
- A temporary local repository was created under ignored `.submission_check/`,
  committed with the explicit identity `Portability Verification`, and genuinely
  cloned with Git. This is a verification snapshot, not the student's submission
  commit or a GitHub upload.
- From that clone, without a dataset present: **18 pytest tests passed in 3.47s**.
  The default dataset path resolved beside the new checkout correctly.
- Main-project tests after portability changes: **18 passed in 3.40s**, saved in
  `results/test_evaluation/pytest_portability.txt`.
- pdoc was regenerated from current source, including project_paths.py.
- Existing experimental evidence was preserved. No training, test evaluation or
  inference benchmark was rerun.
- The first local clone attempt failed because Git's shell could not run in the
  restricted sandbox. A local-only clone outside that restriction succeeded.
- This verifies checkout/import/test portability on this machine using installed
  dependencies. It does not claim testing on a second operating system or a fresh
  dependency installation. GPU execution still requires compatible hardware,
  separately supplied data and checkpoints.

Follow README.md to initialize the main repository, review and commit files,
authenticate and push to your own GitHub repository, then create a fresh clone
and zip it including `.git` for Moodle. Confirm module-specific submission size,
licensing, authorship and checkpoint requirements before uploading.
