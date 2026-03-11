"""
Image Processing Filters - Main Script
========================================
Loads a grayscale image, applies Gaussian, Sobel, and Median filters
using Pure Python, NumPy, and NumPy+Cython, benchmarks performance,
and generates comparison visualizations.

Authors: Raúl Cetina, Daniel Gómez, Christopher Quiñones
"""

import os
import sys
import time
import csv

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

# -- Import filter modules -------------------------------------------------
from filters import pure_python as pp
from filters import numpy_filters as npf

# Attempt to import Cython module
try:
    from filters import cython_filters as cyf
    CYTHON_AVAILABLE = True
except ImportError:
    print("[WARNING] Cython module not compiled. Run: python setup.py build_ext --inplace")
    print("          Skipping Cython benchmarks.\n")
    CYTHON_AVAILABLE = False


# -- Configuration ---------------------------------------------------------
IMAGE_PATH = os.path.join("images", "sample.png")
RESULTS_DIR = "results"
NUM_RUNS = 3  # Number of runs to average timing


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join("report", "images"), exist_ok=True)


def load_image(path):
    """Load an image and convert to grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Cannot load image: {path}")
        sys.exit(1)
    print(f"Image loaded: {path}  ({img.shape[1]}x{img.shape[0]} pixels)")
    return img


def image_to_list(img):
    """Convert a NumPy grayscale image to a 2D Python list."""
    return img.tolist()


def list_to_image(lst):
    """Convert a 2D Python list back to a NumPy uint8 array."""
    return np.array(lst, dtype=np.uint8)


def benchmark(func, *args, num_runs=NUM_RUNS):
    """
    Run a filter function multiple times and return (result, avg_time).
    """
    times = []
    result = None
    for _ in range(num_runs):
        start = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    avg_time = sum(times) / len(times)
    return result, avg_time


def run_benchmarks(gray_img):
    """
    Run all filter × approach combinations and return timing results.

    Returns
    -------
    results : dict
        {(filter_name, approach): (output_image, avg_time)}
    """
    gray_list = image_to_list(gray_img)
    results = {}

    filters_info = [
        ("Gaussian", "gaussian_filter"),
        ("Sobel",    "sobel_filter"),
        ("Median",   "median_filter"),
    ]

    for filter_name, func_name in filters_info:
        print(f"\n{'-'*50}")
        print(f"  Filter: {filter_name}")
        print(f"{'-'*50}")

        # -- Pure Python ----------------------------------------
        pp_func = getattr(pp, func_name)
        print(f"  Running Pure Python {filter_name}...", end="", flush=True)
        result_list, t = benchmark(pp_func, gray_list)
        result_img = list_to_image(result_list)
        results[(filter_name, "Pure Python")] = (result_img, t)
        print(f"  {t:.4f}s")

        # -- NumPy -----------------------------------------------
        np_func = getattr(npf, func_name)
        print(f"  Running NumPy {filter_name}...", end="", flush=True)
        result_img, t = benchmark(np_func, gray_img)
        results[(filter_name, "NumPy")] = (result_img, t)
        print(f"  {t:.4f}s")

        # -- Cython ----------------------------------------------
        if CYTHON_AVAILABLE:
            cy_func = getattr(cyf, func_name)
            print(f"  Running Cython {filter_name}...", end="", flush=True)
            result_img, t = benchmark(cy_func, gray_img)
            results[(filter_name, "NumPy + Cython")] = (result_img, t)
            print(f"  {t:.4f}s")

    return results


def save_filtered_images(results, gray_img):
    """Save all filtered images and the original to results/."""
    # Save original
    cv2.imwrite(os.path.join(RESULTS_DIR, "original.png"), gray_img)

    for (filter_name, approach), (img, _) in results.items():
        safe_approach = approach.replace(" + ", "_plus_").replace(" ", "_").lower()
        safe_filter = filter_name.lower()
        filename = f"{safe_filter}_{safe_approach}.png"
        cv2.imwrite(os.path.join(RESULTS_DIR, filename), img)
        print(f"  Saved: {filename}")


def print_timing_table(results):
    """Print a formatted timing comparison table."""
    print(f"\n{'='*60}")
    print("  PERFORMANCE COMPARISON (seconds)")
    print(f"{'='*60}")
    print(f"  {'Filter':<12} {'Approach':<18} {'Avg Time (s)':>12}  {'Speedup':>8}")
    print(f"  {'-'*12} {'-'*18} {'-'*12}  {'-'*8}")

    filters = ["Gaussian", "Sobel", "Median"]
    approaches = ["Pure Python", "NumPy", "NumPy + Cython"]

    for f in filters:
        base_time = None
        for a in approaches:
            key = (f, a)
            if key in results:
                _, t = results[key]
                if base_time is None:
                    base_time = t
                    speedup = "1.00x"
                else:
                    speedup = f"{base_time / t:.2f}x"
                print(f"  {f:<12} {a:<18} {t:>12.6f}  {speedup:>8}")
        print()

    print(f"{'='*60}")


def save_csv(results):
    """Save benchmark results to CSV."""
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filter", "Approach", "AvgTime_s"])
        for (filt, approach), (_, t) in sorted(results.items()):
            writer.writerow([filt, approach, f"{t:.6f}"])
    print(f"  Benchmark data saved to {csv_path}")


def generate_performance_chart(results):
    """Generate a grouped bar chart comparing execution times."""
    filters = ["Gaussian", "Sobel", "Median"]
    approaches = ["Pure Python", "NumPy", "NumPy + Cython"]
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(filters))
    width = 0.22

    for idx, approach in enumerate(approaches):
        times = []
        for f in filters:
            key = (f, approach)
            if key in results:
                _, t = results[key]
                times.append(t)
            else:
                times.append(0)
        bars = ax.bar(x + idx * width, times, width, label=approach,
                      color=colors[idx], edgecolor='#2C3E50', linewidth=0.5)
        # Add value labels on bars
        for bar, t in zip(bars, times):
            if t > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{t:.4f}s', ha='center', va='bottom', fontsize=7,
                        fontweight='bold')

    ax.set_xlabel('Filter', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Execution Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: Pure Python vs NumPy vs Cython',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(filters, fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "performance_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved to {chart_path}")

    # Also copy to report/images
    import shutil
    shutil.copy2(chart_path, os.path.join("report", "images", "performance_chart.png"))


def generate_comparison_figures(results, gray_img):
    """Generate side-by-side comparison figures for each filter."""
    filters = ["Gaussian", "Sobel", "Median"]
    approaches = ["Pure Python", "NumPy", "NumPy + Cython"]

    for f in filters:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f'{f} Filter - Comparison', fontsize=14, fontweight='bold', y=1.02)

        # Original
        axes[0].imshow(gray_img, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Original', fontsize=11, fontweight='bold')
        axes[0].axis('off')

        for idx, approach in enumerate(approaches):
            key = (f, approach)
            if key in results:
                img, t = results[key]
                axes[idx + 1].imshow(img, cmap='gray', vmin=0, vmax=255)
                axes[idx + 1].set_title(f'{approach}\n({t:.4f}s)',
                                         fontsize=10, fontweight='bold')
            else:
                axes[idx + 1].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                    fontsize=14, transform=axes[idx + 1].transAxes)
                axes[idx + 1].set_title(approach, fontsize=10)
            axes[idx + 1].axis('off')

        plt.tight_layout()
        safe_f = f.lower()
        fig_path = os.path.join(RESULTS_DIR, f"{safe_f}_comparison.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Comparison figure saved: {safe_f}_comparison.png")

        # Copy to report/images
        import shutil
        shutil.copy2(fig_path, os.path.join("report", "images", f"{safe_f}_comparison.png"))


def main():
    """Main entry point."""
    print("=" * 60)
    print("  IMAGE PROCESSING FILTERS - UNIT 2")
    print("  Raul Cetina | Daniel Gomez | Christopher Quinones")
    print("=" * 60)

    ensure_dirs()

    # Load image
    gray_img = load_image(IMAGE_PATH)

    # Save original for report
    cv2.imwrite(os.path.join(RESULTS_DIR, "original.png"), gray_img)
    import shutil
    shutil.copy2(os.path.join(RESULTS_DIR, "original.png"),
                 os.path.join("report", "images", "original.png"))

    # Run benchmarks
    results = run_benchmarks(gray_img)

    # Save outputs
    print(f"\n{'-'*50}")
    print("  Saving outputs...")
    print(f"{'-'*50}")

    save_filtered_images(results, gray_img)
    save_csv(results)
    generate_performance_chart(results)
    generate_comparison_figures(results, gray_img)
    print_timing_table(results)

    print("\n[OK] All done! Check the 'results/' folder for outputs.")
    print("  Run pdflatex on report/report.tex to generate the PDF report.")


if __name__ == "__main__":
    main()
