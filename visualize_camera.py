#!/usr/bin/env python3
"""Generate TOF camera data visualizations matching the NS03A demo software format.
Depth: jet colormap with black background for invalid pixels
Amplitude: hot/grayscale colormap with sentinel masking
"""

import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as ticker

COLS = 120
ROWS = 90

def parse_txt_file(filepath):
    data = np.zeros((ROWS, COLS), dtype=np.float32)
    with open(filepath, 'r') as f:
        for row_idx, line in enumerate(f):
            line = line.strip()
            if not line or row_idx >= ROWS:
                continue
            values = line.split()
            for col_idx, val_str in enumerate(values):
                if col_idx >= COLS:
                    break
                try:
                    data[row_idx, col_idx] = float(val_str)
                except ValueError:
                    data[row_idx, col_idx] = 0.0
    return data


def create_depth_image(data, title="Depth", vmin=0, vmax=3000, cmap='jet'):
    """Create a depth heatmap matching demo software style.
    - Jet colormap (blue=close, red=far)
    - Black background for invalid (0) pixels
    - Color bar in mm
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor='black')
    fig.subplots_adjust(left=0.05, right=0.92, top=0.95, bottom=0.05)

    # Mask invalid pixels (0)
    masked = np.where(data > 0, data, np.nan)

    # Create custom colormap: NaN -> black, jet otherwise
    jet_cmap = plt.cm.jet.copy()
    jet_cmap.set_bad('black')

    im = ax.imshow(masked, cmap=jet_cmap, vmin=vmin, vmax=vmax,
                   aspect='auto', origin='upper', interpolation='nearest')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('Depth (mm)', color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Grid
    ax.set_xticks(np.arange(0, COLS, 10))
    ax.set_yticks(np.arange(0, ROWS, 10))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color='gray', alpha=0.3, linewidth=0.3)
    ax.set_title(title, color='white', fontsize=13, fontweight='bold')

    return fig


def create_amplitude_image(data, title="Amplitude", vmin=0, vmax=3000, cmap='hot'):
    """Create an amplitude heatmap matching demo software style."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor='black')
    fig.subplots_adjust(left=0.05, right=0.92, top=0.95, bottom=0.05)

    # Mask sentinel (65300) and 0
    masked = np.where((data > 0) & (data < 65300), data, np.nan)

    hot_cmap = plt.cm.hot.copy()
    hot_cmap.set_bad('black')

    im = ax.imshow(masked, cmap=hot_cmap, vmin=vmin, vmax=vmax,
                   aspect='auto', origin='upper', interpolation='nearest')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('Amplitude', color='white', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.set_xticks(np.arange(0, COLS, 10))
    ax.set_yticks(np.arange(0, ROWS, 10))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color='gray', alpha=0.3, linewidth=0.3)
    ax.set_title(title, color='white', fontsize=13, fontweight='bold')

    return fig


def create_dashboard(depth_data, amp_data, title_prefix=""):
    """Create a combined dashboard: Depth (left) + Amplitude (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor='black')
    fig.subplots_adjust(left=0.03, right=0.95, top=0.90, bottom=0.05, wspace=0.25)

    # --- Depth ---
    depth_masked = np.where(depth_data > 0, depth_data, np.nan)
    jet_cmap = plt.cm.jet.copy()
    jet_cmap.set_bad('black')

    d_max = np.nanmax(depth_masked) if not np.all(np.isnan(depth_masked)) else 3000
    im1 = ax1.imshow(depth_masked, cmap=jet_cmap, vmin=0, vmax=max(d_max, 100),
                     aspect='auto', origin='upper', interpolation='nearest')
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cbar1.set_label('Depth (mm)', color='white', fontsize=11)
    cbar1.ax.yaxis.set_tick_params(color='white')
    cbar1.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white')

    valid_pct = 100 * (depth_data > 0).sum() / (COLS * ROWS)
    d_valid = depth_data[depth_data > 0]
    info_text = f"Valid: {valid_pct:.1f}%\nMean: {np.mean(d_valid):.0f} mm" if len(d_valid) > 0 else "No valid data"
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, color='lime', fontsize=9,
             va='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ax1.set_title(f'{title_prefix} Depth', color='white', fontsize=14, fontweight='bold')
    ax1.set_xticks(np.arange(0, COLS, 15))
    ax1.set_yticks(np.arange(0, ROWS, 15))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.grid(True, color='gray', alpha=0.2, linewidth=0.3)


    # --- Amplitude ---
    amp_masked = np.where((amp_data > 0) & (amp_data < 65300), amp_data, np.nan)
    hot_cmap = plt.cm.hot.copy()
    hot_cmap.set_bad('black')

    a_max = np.nanmax(amp_masked) if not np.all(np.isnan(amp_masked)) else 3000
    im2 = ax2.imshow(amp_masked, cmap=hot_cmap, vmin=0, vmax=max(a_max, 100),
                     aspect='auto', origin='upper', interpolation='nearest')
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cbar2.set_label('Amplitude', color='white', fontsize=11)
    cbar2.ax.yaxis.set_tick_params(color='white')
    cbar2.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')

    amp_valid_pct = 100 * ((amp_data > 0) & (amp_data < 65300)).sum() / (COLS * ROWS)
    a_valid = amp_data[(amp_data > 0) & (amp_data < 65300)]
    info_text2 = f"Valid: {amp_valid_pct:.1f}%\nMean: {np.mean(a_valid):.0f}" if len(a_valid) > 0 else "No valid data"
    ax2.text(0.02, 0.98, info_text2, transform=ax2.transAxes, color='lime', fontsize=9,
             va='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ax2.set_title(f'{title_prefix} Amplitude', color='white', fontsize=14, fontweight='bold')
    ax2.set_xticks(np.arange(0, COLS, 15))
    ax2.set_yticks(np.arange(0, ROWS, 15))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.grid(True, color='gray', alpha=0.2, linewidth=0.3)

    fig.suptitle(f'NS03A ITOF Camera — {title_prefix}', color='white', fontsize=16, fontweight='bold', y=0.97)
    return fig


def create_comparison(d1, a1, d2, a2):
    """Create a 2×2 comparison: test_001 vs test_002, row=dataset, col=depth/amp."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='black')
    fig.subplots_adjust(left=0.03, right=0.95, top=0.93, bottom=0.03, wspace=0.2, hspace=0.25)

    datasets = [('test_001', d1, a1), ('test_002', d2, a2)]

    for row, (name, depth, amp) in enumerate(datasets):
        # Depth
        depth_masked = np.where(depth > 0, depth, np.nan)
        jet_cmap = plt.cm.jet.copy()
        jet_cmap.set_bad('black')
        d_max = np.nanmax(depth_masked) if not np.all(np.isnan(depth_masked)) else 3000
        im_d = axes[row, 0].imshow(depth_masked, cmap=jet_cmap, vmin=0, vmax=max(d_max, 100),
                                    aspect='auto', origin='upper', interpolation='nearest')
        cbar_d = fig.colorbar(im_d, ax=axes[row, 0], fraction=0.046, pad=0.02)
        cbar_d.set_label('mm', color='white', fontsize=10)
        cbar_d.ax.yaxis.set_tick_params(color='white', labelsize=8)
        cbar_d.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar_d.ax.axes, 'yticklabels'), color='white')

        d_valid = depth[depth > 0]
        vp = 100 * len(d_valid) / (COLS * ROWS)
        axes[row, 0].text(0.02, 0.98, f"Valid: {vp:.1f}% | Mean: {np.mean(d_valid):.0f} mm",
                          transform=axes[row, 0].transAxes, color='lime', fontsize=9,
                          va='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        axes[row, 0].set_title(f'{name} — Depth', color='white', fontsize=13, fontweight='bold')
        axes[row, 0].grid(True, color='gray', alpha=0.2, linewidth=0.3)

        # Amplitude
        amp_masked = np.where((amp > 0) & (amp < 65300), amp, np.nan)
        hot_cmap = plt.cm.hot.copy()
        hot_cmap.set_bad('black')
        a_max = np.nanmax(amp_masked) if not np.all(np.isnan(amp_masked)) else 3000
        im_a = axes[row, 1].imshow(amp_masked, cmap=hot_cmap, vmin=0, vmax=max(a_max, 100),
                                    aspect='auto', origin='upper', interpolation='nearest')
        cbar_a = fig.colorbar(im_a, ax=axes[row, 1], fraction=0.046, pad=0.02)
        cbar_a.set_label('Amp', color='white', fontsize=10)
        cbar_a.ax.yaxis.set_tick_params(color='white', labelsize=8)
        cbar_a.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar_a.ax.axes, 'yticklabels'), color='white')

        a_valid = amp[(amp > 0) & (amp < 65300)]
        avp = 100 * len(a_valid) / (COLS * ROWS)
        axes[row, 1].text(0.02, 0.98, f"Valid: {avp:.1f}% | Mean: {np.mean(a_valid):.0f}",
                          transform=axes[row, 1].transAxes, color='lime', fontsize=9,
                          va='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        axes[row, 1].set_title(f'{name} — Amplitude', color='white', fontsize=13, fontweight='bold')
        axes[row, 1].grid(True, color='gray', alpha=0.2, linewidth=0.3)

    fig.suptitle('NS03A ITOF Camera — Dataset Comparison (Frame 50)', color='white', fontsize=16, fontweight='bold')
    return fig


def main():
    base = "/home/kiyi/new_camera_imu_test/camera"
    output_dir = "/home/kiyi/new_camera_imu_test/output_images"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating TOF camera visualizations...")

    for dataset in ['test_001', 'test_002']:
        path = os.path.join(base, dataset)
        depth_files = sorted([f for f in os.listdir(path) if 'Depth' in f and f.endswith('.txt')])
        amp_files = sorted([f for f in os.listdir(path) if 'Amp' in f and f.endswith('.txt')])

        # Create subdirectory
        ds_out = os.path.join(output_dir, dataset)
        os.makedirs(ds_out, exist_ok=True)

        # Generate sample frames (0, 50, 100, 150, 200, 250, 299)
        sample_frames = [0, 50, 100, 150, 200, 250, 299]
        sample_frames = [i for i in sample_frames if i < len(depth_files)]

        print(f"\n  {dataset}: generating {len(sample_frames)} sample frames...")

        for fi in sample_frames:
            depth = parse_txt_file(os.path.join(path, depth_files[fi]))
            amp = parse_txt_file(os.path.join(path, amp_files[fi]))

            # Dashboard (depth + amp side by side)
            dash = create_dashboard(depth, amp, f"{dataset} Frame {fi}")
            dash.savefig(os.path.join(ds_out, f"{dataset}_dashboard_frame_{fi:03d}.png"),
                        dpi=150, facecolor='black')
            plt.close(dash)

            # Individual depth
            fig_d = create_depth_image(depth, f"{dataset} Depth Frame {fi}")
            fig_d.savefig(os.path.join(ds_out, f"{dataset}_depth_frame_{fi:03d}.png"),
                         dpi=150, facecolor='black')
            plt.close(fig_d)

            # Individual amplitude
            fig_a = create_amplitude_image(amp, f"{dataset} Amplitude Frame {fi}")
            fig_a.savefig(os.path.join(ds_out, f"{dataset}_amplitude_frame_{fi:03d}.png"),
                         dpi=150, facecolor='black')
            plt.close(fig_a)

        print(f"    Saved to {ds_out}/")

        # Generate temporal sequence (every 10 frames) for GIF-like overview
        seq_out = os.path.join(ds_out, "sequence")
        os.makedirs(seq_out, exist_ok=True)
        step = 10
        for fi in range(0, len(depth_files), step):
            depth = parse_txt_file(os.path.join(path, depth_files[fi]))
            fig_d = create_depth_image(depth, f"{dataset} Frame {fi}")
            fig_d.savefig(os.path.join(seq_out, f"depth_{fi:03d}.png"), dpi=100, facecolor='black')
            plt.close(fig_d)
        print(f"    Sequence frames ({len(range(0, len(depth_files), step))}): {seq_out}/")

    # ---- COMPARISON IMAGE ----
    print("\n  Generating comparison image...")
    # Use frame 50 from both datasets
    for ds in ['test_001', 'test_002']:
        path = os.path.join(base, ds)
        dfiles = sorted([f for f in os.listdir(path) if 'Depth' in f and f.endswith('.txt')])
        afiles = sorted([f for f in os.listdir(path) if 'Amp' in f and f.endswith('.txt')])

        fi = 50
        if fi < len(dfiles):
            if ds == 'test_001':
                d1 = parse_txt_file(os.path.join(path, dfiles[fi]))
                a1 = parse_txt_file(os.path.join(path, afiles[fi]))
            else:
                d2 = parse_txt_file(os.path.join(path, dfiles[fi]))
                a2 = parse_txt_file(os.path.join(path, afiles[fi]))

    comp = create_comparison(d1, a1, d2, a2)
    comp.savefig(os.path.join(output_dir, "comparison_test001_vs_test002.png"),
                 dpi=200, facecolor='black')
    plt.close(comp)
    print(f"    Comparison saved to {output_dir}/comparison_test001_vs_test002.png")

    print(f"\nDone! All images saved to {output_dir}/")
    print(f"  test_001: {output_dir}/test_001/")
    print(f"  test_002: {output_dir}/test_002/")
    print(f"  comparison: {output_dir}/comparison_test001_vs_test002.png")

if __name__ == "__main__":
    main()
