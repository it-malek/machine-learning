# CSCI 450 — Assignment Notebooks

Cleaned and re-titled versions of the original CSCI 450 (Machine
Learning) coursework. Each notebook now opens with a header cell
explaining what techniques it covers and how to obtain its dataset,
and closes with a *“What I'd do differently now”* reflection cell.

| File | Topics | External data needed |
|---|---|---|
| `hw1_image_filters_and_retrieval.ipynb` | Roberts edge detection, image rescaling, color-histogram retrieval (RGB/HSV × Euclidean/Manhattan) | [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) under `./jpg/` |
| `hw2_smoothing_and_tiny_image_retrieval.ipynb` | Difference-of-Gaussians, median filter, "tiny image" (16×16) retrieval | Oxford 102 Flowers + the `SP.jpg` noisy image distributed with the assignment |
| `hw3_image_box_drawing.ipynb` | Generate `n` random axis-aligned boxes; manually draw rectangles via NumPy slicing | A local `testpic.jpeg` (or substitute `skimage.data.astronaut()`) |
| `hw4_hog_features_and_svm_classifier.ipynb` | HOG features, per-class average images, one-vs-rest SVM for the `airplanes` class | [Caltech 101](https://data.caltech.edu/records/mzrjq-6wc02) at `./101_ObjectCategories/` |
| `midterm_green_rectangle_detection.ipynb` | NumPy boolean masking to find a coloured region's bounding box | A local `test.jpg` (pre-baked outputs in the cells show the expected answer) |

## Why these are kept as notebooks (not refactored into `src/`)

The two main projects (`projects/heart_failure_prediction/` and
`projects/lhc_collision_classification/`) have been refactored into
proper Python packages with cross-validated benchmarks because they
ship a meaningful end-to-end pipeline. The assignments here, by
contrast, are short standalone exercises — turning them into modules
would be busywork that doesn't increase the value of the repository
as a portfolio piece. Instead, each notebook keeps its original
solution intact (so you can see the actual class submission) but
gains a markdown reflection at the end calling out the things I'd
fix today.
