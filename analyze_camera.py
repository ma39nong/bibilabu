#!/usr/bin/env python3
"""NS03A ITOF Camera Data Analysis Script
Analyzes depth, amplitude, and point cloud data from test_001 and test_002.
"""

import numpy as np
import os
import sys
from collections import defaultdict
import re

# Resolution: 120 columns x 90 rows
COLS = 120
ROWS = 90
TOTAL_PIXELS = COLS * ROWS  # 10800

def parse_txt_file(filepath, cols=COLS):
    """Parse a depth or amplitude txt file. Returns 2D numpy array (rows x cols).
    Format: 90 lines, each with 120 space-separated values. No row number prefix.
    """
    data = np.zeros((ROWS, COLS), dtype=np.float32)
    with open(filepath, 'r') as f:
        for row_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if row_idx >= ROWS:
                break
            values = line.split()
            for col_idx, val_str in enumerate(values):
                if col_idx >= COLS:
                    break
                try:
                    data[row_idx, col_idx] = float(val_str)
                except ValueError:
                    data[row_idx, col_idx] = 0.0
    return data

def parse_pcd_file(filepath):
    """Parse a PCD file and extract x,y,z coordinates."""
    points = []
    header_done = False
    with open(filepath, 'r') as f:
        for line in f:
            if not header_done:
                if line.strip() == 'DATA ascii':
                    header_done = True
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])
                except ValueError:
                    points.append([0, 0, 0])
    return np.array(points, dtype=np.float32)

def analyze_dataset(dataset_path, dataset_name):
    """Comprehensive analysis of one test dataset."""
    print(f"\n{'='*80}")
    print(f"  ANALYSIS: {dataset_name}")
    print(f"{'='*80}")

    # Find all files
    all_files = sorted(os.listdir(dataset_path))
    depth_files = sorted([f for f in all_files if 'Depth' in f and f.endswith('.txt')])
    amp_files = sorted([f for f in all_files if 'Amp' in f and f.endswith('.txt')])
    pcd_files = sorted([f for f in all_files if f.endswith('.pcd')])
    png_files = sorted([f for f in all_files if f.endswith('.png')])

    print(f"\n  File counts:")
    print(f"    Depth files: {len(depth_files)}")
    print(f"    Amplitude files: {len(amp_files)}")
    print(f"    PointCloud (PCD) files: {len(pcd_files)}")
    print(f"    Image (PNG) files: {len(png_files)}")

    # ---- Analyze ALL frames ----
    depth_stats_per_frame = []
    amp_stats_per_frame = []
    pcd_stats_per_frame = []

    print(f"\n  Processing all {len(depth_files)} frames...")

    for i, (df, af, pf) in enumerate(zip(depth_files, amp_files, pcd_files)):
        depth_data = parse_txt_file(os.path.join(dataset_path, df))
        amp_data = parse_txt_file(os.path.join(dataset_path, af))
        pcd_data = parse_pcd_file(os.path.join(dataset_path, pf))

        # Depth statistics (non-zero values only)
        depth_valid = depth_data[depth_data > 0]
        depth_invalid_pct = 100 * (depth_data == 0).sum() / TOTAL_PIXELS

        # Amplitude statistics (exclude sentinel value 65300)
        amp_valid = amp_data[(amp_data > 0) & (amp_data < 65300)]
        amp_sentinel_pct = 100 * (amp_data == 65300).sum() / TOTAL_PIXELS

        # PCD statistics
        pcd_valid = pcd_data[~np.all(pcd_data == 0, axis=1)]
        pcd_invalid_pct = 100 * np.all(pcd_data == 0, axis=1).sum() / len(pcd_data)

        frame_info = {
            'index': i,
            'depth': {
                'mean': np.mean(depth_valid) if len(depth_valid) > 0 else 0,
                'std': np.std(depth_valid) if len(depth_valid) > 0 else 0,
                'min': np.min(depth_valid) if len(depth_valid) > 0 else 0,
                'max': np.max(depth_valid) if len(depth_valid) > 0 else 0,
                'median': np.median(depth_valid) if len(depth_valid) > 0 else 0,
                'valid_count': len(depth_valid),
                'invalid_pct': depth_invalid_pct,
            },
            'amp': {
                'mean': np.mean(amp_valid) if len(amp_valid) > 0 else 0,
                'std': np.std(amp_valid) if len(amp_valid) > 0 else 0,
                'max': np.max(amp_valid) if len(amp_valid) > 0 else 0,
                'valid_count': len(amp_valid),
                'sentinel_pct': amp_sentinel_pct,
            },
            'pcd': {
                'z_mean': np.mean(pcd_valid[:, 2]) if len(pcd_valid) > 0 else 0,
                'z_std': np.std(pcd_valid[:, 2]) if len(pcd_valid) > 0 else 0,
                'z_min': np.min(pcd_valid[:, 2]) if len(pcd_valid) > 0 else 0,
                'z_max': np.max(pcd_valid[:, 2]) if len(pcd_valid) > 0 else 0,
                'valid_count': len(pcd_valid),
                'invalid_pct': pcd_invalid_pct,
            }
        }
        depth_stats_per_frame.append(frame_info['depth'])
        amp_stats_per_frame.append(frame_info['amp'])
        pcd_stats_per_frame.append(frame_info['pcd'])

    # ---- Aggregate Statistics ----
    print(f"\n  --- DEPTH Statistics (across all {len(depth_files)} frames) ---")
    _print_frame_stats(depth_stats_per_frame, ['mean', 'std', 'min', 'max', 'median', 'invalid_pct', 'valid_count'],
                       names={'mean': 'Mean depth (mm)', 'std': 'Std dev (mm)', 'min': 'Min depth (mm)',
                              'max': 'Max depth (mm)', 'median': 'Median depth (mm)',
                              'invalid_pct': 'Invalid pixel %', 'valid_count': 'Valid pixel count'})

    print(f"\n  --- AMPLITUDE Statistics (across all {len(amp_files)} frames) ---")
    _print_frame_stats(amp_stats_per_frame, ['mean', 'std', 'max', 'valid_count', 'sentinel_pct'],
                       names={'mean': 'Mean amplitude', 'std': 'Std dev', 'max': 'Max amplitude',
                              'valid_count': 'Valid pixel count', 'sentinel_pct': 'Sentinel (65300) %'})

    print(f"\n  --- POINT CLOUD Statistics (across all {len(pcd_files)} frames) ---")
    _print_frame_stats(pcd_stats_per_frame, ['z_mean', 'z_std', 'z_min', 'z_max', 'valid_count', 'invalid_pct'],
                       names={'z_mean': 'Mean Z (mm)', 'z_std': 'Z Std dev (mm)', 'z_min': 'Min Z (mm)',
                              'z_max': 'Max Z (mm)', 'valid_count': 'Valid point count',
                              'invalid_pct': 'Invalid point %'})

    # ---- Temporal Analysis (Frame-to-Frame Stability) ----
    print(f"\n  --- Temporal Stability ---")
    depth_means = np.array([s['mean'] for s in depth_stats_per_frame])
    depth_valids = np.array([s['valid_count'] for s in depth_stats_per_frame])
    z_means = np.array([s['z_mean'] for s in pcd_stats_per_frame])

    if len(depth_means) > 1:
        # Frame-to-frame differences
        depth_mean_diff = np.abs(np.diff(depth_means))
        z_mean_diff = np.abs(np.diff(z_means))
        valid_count_diff = np.abs(np.diff(depth_valids))

        print(f"    Depth mean frame-to-frame change:")
        print(f"      Mean: {np.mean(depth_mean_diff):.2f} mm, Max: {np.max(depth_mean_diff):.2f} mm, Std: {np.std(depth_mean_diff):.2f} mm")
        print(f"    Z coordinate frame-to-frame change:")
        print(f"      Mean: {np.mean(z_mean_diff):.2f} mm, Max: {np.max(z_mean_diff):.2f} mm, Std: {np.std(z_mean_diff):.2f} mm")
        print(f"    Valid pixel count frame-to-frame change:")
        print(f"      Mean: {np.mean(valid_count_diff):.1f}, Max: {np.max(valid_count_diff):.1f}, Std: {np.std(valid_count_diff):.1f}")

    # ---- Spatial Analysis (center vs edge) ----
    print(f"\n  --- Spatial Analysis (First Frame) ---")
    if len(depth_files) > 0:
        first_depth = parse_txt_file(os.path.join(dataset_path, depth_files[0]))
        _spatial_analysis(first_depth, "Depth (mm)")

    # ---- Depth Distribution (first frame histogram bins) ----
    print(f"\n  --- Depth Distribution (Frame 0, non-zero values) ---")
    first_depth = parse_txt_file(os.path.join(dataset_path, depth_files[0]))
    d_valid = first_depth[first_depth > 0]
    if len(d_valid) > 0:
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            print(f"    P{p:2d}: {np.percentile(d_valid, p):.1f} mm")
        # Distance bins
        bins = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000)]
        print(f"    Distance distribution:")
        for lo, hi in bins:
            count = np.sum((d_valid >= lo) & (d_valid < hi))
            pct = 100 * count / len(d_valid)
            print(f"      {lo}-{hi} mm: {count} pixels ({pct:.1f}%)")

    return {
        'depth': depth_stats_per_frame,
        'amp': amp_stats_per_frame,
        'pcd': pcd_stats_per_frame,
    }


def _print_frame_stats(stats_list, keys, names):
    """Helper: print aggregated stats across frames."""
    for key in keys:
        values = np.array([s[key] for s in stats_list])
        name = names.get(key, key)
        print(f"    {name}:")
        print(f"      Min: {np.min(values):.2f}  Max: {np.max(values):.2f}  "
              f"Mean: {np.mean(values):.2f}  Std: {np.std(values):.2f}")


def _spatial_analysis(data, label):
    """Analyze center vs edge regions."""
    # Center 20% region
    center_r_start = int(ROWS * 0.4)
    center_r_end = int(ROWS * 0.6)
    center_c_start = int(COLS * 0.4)
    center_c_end = int(COLS * 0.6)
    center = data[center_r_start:center_r_end, center_c_start:center_c_end]
    center_valid = center[center > 0]

    # Edge (outer 10%)
    edge_top = data[:int(ROWS * 0.1), :]
    edge_bottom = data[int(ROWS * 0.9):, :]
    edge_left = data[:, :int(COLS * 0.1)]
    edge_right = data[:, int(COLS * 0.9):]
    edge = np.concatenate([edge_top.flatten(), edge_bottom.flatten(),
                           edge_left.flatten(), edge_right.flatten()])
    edge_valid = edge[edge > 0]

    print(f"    Center region ({center.shape}): valid={len(center_valid)}, "
          f"mean={np.mean(center_valid):.2f}" if len(center_valid) > 0 else f"    Center: all invalid")
    print(f"    Edge region: valid={len(edge_valid)}, "
          f"mean={np.mean(edge_valid):.2f}" if len(edge_valid) > 0 else f"    Edge: all invalid")
    if len(center_valid) > 0:
        print(f"    Center invalid %: {100 * (center == 0).sum() / center.size:.1f}%")
    if len(edge_valid) > 0:
        print(f"    Edge invalid %: {100 * (edge == 0).sum() / edge.size:.1f}%")


def compare_datasets(result_1, result_2):
    """Direct comparison between test_001 and test_002."""
    print(f"\n{'='*80}")
    print(f"  COMPARISON: test_001 vs test_002")
    print(f"{'='*80}")

    metrics = {
        'Mean Depth (mm)': ('mean', 'depth', lambda v: f"{np.mean(v):.1f} ± {np.std(v):.1f}"),
        'Mean Z (mm)': ('z_mean', 'pcd', lambda v: f"{np.mean(v):.1f} ± {np.std(v):.1f}"),
        'Valid Pixels/Frame': ('valid_count', 'depth', lambda v: f"{np.mean(v):.0f} ± {np.std(v):.0f}"),
        'Invalid Pixel %': ('invalid_pct', 'depth', lambda v: f"{np.mean(v):.1f}% ± {np.std(v):.1f}%"),
        'Max Depth (mm)': ('max', 'depth', lambda v: f"{np.mean(v):.1f}"),
        'Depth Range (mm)': ('max', 'depth', lambda v: f"{np.mean(v) - np.mean([s['min'] for s in result_1['depth']]):.1f} vs {np.mean(v) - np.mean([s['min'] for s in result_2['depth']]):.1f}" if False else ""),
    }

    print(f"\n  {'Metric':<30} {'test_001':<30} {'test_002':<30}")
    print(f"  {'-'*30} {'-'*30} {'-'*30}")

    for name, (key, section, fmt) in metrics.items():
        v1 = np.array([s[key] for s in result_1[section]])
        v2 = np.array([s[key] for s in result_2[section]])
        s1 = fmt(v1)
        s2 = fmt(v2)
        print(f"  {name:<30} {s1:<30} {s2:<30}")

    # Depth range comparison
    d1_max = np.array([s['max'] for s in result_1['depth']])
    d1_min = np.array([s['min'] for s in result_1['depth']])
    d2_max = np.array([s['max'] for s in result_2['depth']])
    d2_min = np.array([s['min'] for s in result_2['depth']])
    range1 = np.mean(d1_max) - np.mean(d1_min)
    range2 = np.mean(d2_max) - np.mean(d2_min)
    print(f"  {'Depth Range (mean max-min, mm)':<30} {range1:<30.1f} {range2:<30.1f}")

    # Amplitude comparison
    for r, name in [(result_1, 'test_001'), (result_2, 'test_002')]:
        amp_means = np.array([s['mean'] for s in r['amp']])
        amp_maxs = np.array([s['max'] for s in r['amp']])
        amp_sentinel = np.array([s['sentinel_pct'] for s in r['amp']])
        print(f"\n  {name} Amplitude: Mean={np.mean(amp_means):.1f}, Max={np.mean(amp_maxs):.1f}, "
              f"Sentinel%={np.mean(amp_sentinel):.1f}%")


def main():
    base_path = "/home/kiyi/new_camera_imu_test/camera"

    print("=" * 80)
    print("  NS03A ITOF CAMERA DATA ANALYSIS")
    print("  Resolution: 120 x 90 pixels")
    print("  Type: ITOF (Indirect Time-of-Flight)")
    print("=" * 80)

    result_1 = analyze_dataset(os.path.join(base_path, "test_001"), "test_001")
    result_2 = analyze_dataset(os.path.join(base_path, "test_002"), "test_002")

    compare_datasets(result_1, result_2)

    print(f"\n{'='*80}")
    print(f"  SUMMARY & OBSERVATIONS")
    print(f"{'='*80}")
    print(f"""
  1. DATA STRUCTURE:
     - Both datasets contain 300 frames each
     - Each frame: Depth (120x90 txt), Amplitude (120x90 txt), PointCloud (.pcd)
     - test_002 additionally has 202 PNG visualization images
     - Depth values in mm, Amplitude in arbitrary units
     - PCD files with x, y, z coordinates in mm

  2. DEPTH PERFORMANCE:
     - Camera detects objects in the ~0.4m to ~3m range (per spec)
     - 0 values indicate invalid/no-return pixels
     - 65300 in amplitude = sentinel for saturated/invalid

  3. DATA QUALITY INDICATORS:
     - Invalid pixel percentage shows how much of the FOV has no depth data
     - Frame-to-frame depth stability indicates temporal noise
     - Amplitude correlates with confidence (higher = stronger return signal)

  4. RECOMMENDATIONS:
     - Check if the two tests were done in different scenarios (indoor/outdoor, different objects)
     - Higher amplitude + lower invalid% = better data quality
     - Compare PNG images in test_002 for visual reference
""")

if __name__ == "__main__":
    main()
