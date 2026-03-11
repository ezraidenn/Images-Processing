# Image Processing Filters – Unit 2

Implementation of three fundamental image processing filters (**Gaussian**, **Sobel**, and **Median**) using three computational approaches, with performance benchmarking and analysis.

## Team

| Name | Role |
|---|---|
| Raúl Cetina | Developer |
| Daniel Gómez | Developer |
| Christopher Quiñones | Developer |

## Filters Implemented

| Filter | Purpose |
|---|---|
| **Gaussian** | Noise reduction / blurring via weighted average (3×3 kernel) |
| **Sobel** | Edge detection via gradient magnitude in X and Y directions |
| **Median** | Salt-and-pepper noise removal using neighborhood median |

## Approaches

1. **Pure Python** – No external libraries; loops over pixels manually.
2. **NumPy** – Vectorized array operations for significant speed-up.
3. **NumPy + Cython** – Compiled C extensions with typed memoryviews for maximum performance.

## Project Structure

```
Images-Processing/
├── filters/
│   ├── pure_python.py        # Pure Python implementations
│   ├── numpy_filters.py      # NumPy implementations
│   └── cython_filters.pyx    # Cython implementations
├── images/                   # Input images
├── results/                  # Generated output images and benchmarks
├── report/                   # LaTeX report and compiled PDF
├── main.py                   # Entry point
├── setup.py                  # Cython build script
├── requirements.txt          # Dependencies
└── README.md
```

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Compile Cython module

```bash
python setup.py build_ext --inplace
```

### 3. Run the pipeline

```bash
python main.py
```

This will:
- Load and convert the test image to grayscale
- Apply all 9 filter/approach combinations
- Benchmark execution times
- Save filtered images and comparison charts to `results/`
- Print a performance summary table

## Output

The `results/` directory will contain:
- Filtered images for each filter × approach
- `performance_chart.png` – Bar chart comparing execution times
- `benchmark_results.csv` – Raw timing data

## Report

The compiled PDF report is located at `report/report.pdf` and includes:
- Problem description with kernel definitions
- Implementation details for each approach
- Performance analysis with timing comparisons
- Visual results showing original vs. filtered images
- Discussion of trade-offs and insights
