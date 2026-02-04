#!/usr/bin/env python3
"""Benchmark vmap parallelism for compute_Cell_spectrum_from_cosmo_params.

This script times how long compute_Cell_spectrum_from_cosmo_params takes to run
when vmapped over different batch sizes (log-scaled: 1, 2, 4, 8, ...).

It outputs a plot showing average time per invocation vs batch size.

Usage:
    python scripts/benchmark_vmap_parallelism.py N

where N is the maximum batch size to test (will test 1, 2, 4, 8, ... up to N).
"""

import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Enable 64-bit precision and GPU
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'gpu')

from discoeb.cmb import compute_Cell_spectrum_from_cosmo_params


# DISCO-Notebook cosmological parameters (baseline values)
DISCO_NOTEBOOK_PARAMS = {
    'Omegam': 0.3099,
    'Omegab': 0.0488911,
    'H0': 67.742,
    'n_s': 0.96822,
    'A_s': 2.1064e-09,
    'mnu': 0.06,
    'Tcmb': 2.7255,
    'YHe': 0.248,
    'Neff': 2.046,
    'Nmnu': 1,
    'w_DE_0': -0.99,
    'w_DE_a': 0.0,
    'cs2_DE': 1.0,
    'Omegak': 0.0,
    'k_p': 0.05,
}

# Parameters that can be varied (continuous parameters)
VARIABLE_PARAMS = ['Omegam', 'Omegab', 'H0', 'n_s', 'A_s', 'mnu', 'Tcmb', 'YHe', 'Neff']


def bytes_to_gib(bytes_val):
    """Convert bytes to GiB."""
    return bytes_val / (1024 ** 3)


def get_gpu_memory_limit():
    """Get the available GPU memory limit in bytes."""
    device = jax.local_devices()[0]
    # Try to get memory stats - this may not be available on all devices
    try:
        stats = device.memory_stats()
        if stats and 'bytes_limit' in stats:
            return stats['bytes_limit']
    except Exception:
        pass

    # Fallback: try to get from device client
    try:
        # For CUDA devices, we can try nvidia-smi or assume a default
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # Returns memory in MiB
            memory_mib = int(result.stdout.strip().split('\n')[0])
            return memory_mib * 1024 * 1024
    except Exception:
        pass

    # Default fallback (assume 10 GiB)
    print("Warning: Could not determine GPU memory, assuming 10 GiB")
    return 10 * 1024 ** 3


def generate_noisy_params(base_params, n_samples, key, sigma_frac=0.05/3):
    """Generate n_samples parameter sets with Gaussian noise.

    Parameters
    ----------
    base_params : dict
        Base cosmological parameters
    n_samples : int
        Number of parameter sets to generate
    key : jax.random.PRNGKey
        Random key for reproducibility
    sigma_frac : float
        Fractional standard deviation (default: 5%/3 ≈ 1.67% for 3-sigma = 5%)

    Returns
    -------
    dict
        Dictionary with arrays of shape (n_samples,) for variable params
    """
    params_batch = {}

    for param_name, base_value in base_params.items():
        if param_name in VARIABLE_PARAMS and base_value != 0:
            # Add Gaussian noise with sigma = sigma_frac * base_value
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=(n_samples,)) * sigma_frac * abs(base_value)
            params_batch[param_name] = jnp.full(n_samples, base_value) + noise
        else:
            # Keep fixed (integer params like Nmnu, or params that shouldn't vary)
            params_batch[param_name] = jnp.full(n_samples, base_value)

    return params_batch


def predict_memory_usage(fn, *args, **kwargs):
    """Predict memory usage of a JAX function without executing it.

    Parameters
    ----------
    fn : callable
        JIT-compiled JAX function
    *args, **kwargs
        Arguments to pass to the function

    Returns
    -------
    dict
        Memory analysis with keys:
        - 'temp_size': Temporary memory needed during execution (bytes)
        - 'input_size': Input argument size (bytes)
        - 'output_size': Output size (bytes)
        - 'peak_size': Estimated peak memory (bytes)
    """
    # Lower the function to HLO
    lowered = fn.lower(*args, **kwargs)

    # Compile to get memory analysis
    compiled = lowered.compile()

    # Get memory analysis
    mem_analysis = compiled.memory_analysis()

    # Extract relevant fields
    temp_size = getattr(mem_analysis, 'temp_size_in_bytes', 0) or 0
    input_size = getattr(mem_analysis, 'argument_size_in_bytes', 0) or 0
    output_size = getattr(mem_analysis, 'output_size_in_bytes', 0) or 0
    alias_size = getattr(mem_analysis, 'alias_size_in_bytes', 0) or 0

    # Peak memory is roughly: inputs + outputs + temps - aliased memory
    peak_size = input_size + output_size + temp_size - alias_size

    return {
        'temp_size': temp_size,
        'input_size': input_size,
        'output_size': output_size,
        'alias_size': alias_size,
        'peak_size': peak_size,
    }


def run_benchmark(max_batch_size, seed=42):
    """Run the vmap parallelism benchmark.

    Parameters
    ----------
    max_batch_size : int
        Maximum batch size to test (will test 1, 2, ..., max_batch_size)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    batch_sizes : list
        List of batch sizes tested (only those that fit in memory), log-scaled (1,2,4,8,...)
    total_times : list
        Total time to process the entire batch for each batch size
    memory_usage : list
        Predicted peak memory usage in GiB for each batch size
    skipped_sizes : list
        Batch sizes that were skipped due to memory constraints
    """
    print(f"\nJAX backend: {jax.default_backend()}")
    devices = jax.devices()
    print(f"Devices: {devices}")

    # Get GPU memory limit
    gpu_memory_limit = get_gpu_memory_limit()
    # Use 90% of GPU memory as safety margin
    safe_memory_limit = int(gpu_memory_limit * 0.90)
    print(f"GPU memory: {bytes_to_gib(gpu_memory_limit):.2f} GiB")
    print(f"Safe limit (90%): {bytes_to_gib(safe_memory_limit):.2f} GiB")

    # Fixed computation parameters
    ellmax = 2500
    nmodes = 128
    kmin = 1e-4
    kmax = 1.0

    # Create the vmapped version of the function
    @partial(jax.jit, static_argnames=['ellmax', 'nmodes', 'kmin', 'kmax'])
    def compute_batched(params_batch, ellmax, nmodes, kmin, kmax):
        """Compute Cell spectrum for a batch of parameter sets."""
        def single_compute(params):
            Cell, _ = compute_Cell_spectrum_from_cosmo_params(
                param_dict=params,
                ellmax=ellmax,
                nmodes=nmodes,
                kmin=kmin,
                kmax=kmax,
            )
            return Cell

        # vmap over the 0th axis of all dict values
        return jax.vmap(single_compute)(params_batch)

    # Generate all parameter sets upfront for the maximum batch size
    key = jax.random.PRNGKey(seed)
    all_params = generate_noisy_params(DISCO_NOTEBOOK_PARAMS, max_batch_size, key)

    # Generate log-scale batch sizes: 1, 2, 4, 8, 16, ... up to max_batch_size
    # Always include max_batch_size even if it's not a power of 2
    batch_sizes_to_test = []
    size = 1
    while size < max_batch_size:
        batch_sizes_to_test.append(size)
        size *= 2
    # Add max_batch_size if not already included
    if not batch_sizes_to_test or batch_sizes_to_test[-1] != max_batch_size:
        batch_sizes_to_test.append(max_batch_size)

    batch_sizes = []
    total_times = []
    memory_usage = []
    skipped_sizes = []

    print("\n" + "="*70)
    print(f"{'Batch':>6} | {'Pred. Memory':>12} | {'Status':>10} | {'Total Time':>12}")
    print(f"{'Size':>6} | {'(GiB)':>12} | {'':>10} | {'(s)':>12}")
    print("="*70)

    for batch_size in batch_sizes_to_test:
        # Extract subset of parameters
        batch_params = {k: v[:batch_size] for k, v in all_params.items()}

        # Predict memory usage before compiling/running
        try:
            mem_info = predict_memory_usage(
                compute_batched, batch_params,
                ellmax=ellmax, nmodes=nmodes, kmin=kmin, kmax=kmax
            )
            predicted_memory = mem_info['peak_size']
            predicted_memory_gib = bytes_to_gib(predicted_memory)
        except Exception as e:
            print(f"{batch_size:>6} | {'Error':>12} | {'SKIP':>10} | {'-':>12}")
            print(f"         Memory prediction failed: {e}")
            skipped_sizes.append(batch_size)
            jax.clear_caches()
            continue

        # Check if it fits in memory
        if predicted_memory > safe_memory_limit:
            print(f"{batch_size:>6} | {predicted_memory_gib:>12.2f} | {'SKIP (OOM)':>10} | {'-':>12}")
            skipped_sizes.append(batch_size)
            jax.clear_caches()
            continue

        # Memory looks OK, proceed with execution
        try:
            # Warmup / JIT compile for this batch size
            result = compute_batched(batch_params, ellmax=ellmax, nmodes=nmodes, kmin=kmin, kmax=kmax)
            result.block_until_ready()

            # Timed run (function is now JIT compiled for this batch size)
            start = time.perf_counter()
            result = compute_batched(batch_params, ellmax=ellmax, nmodes=nmodes, kmin=kmin, kmax=kmax)
            result.block_until_ready()
            end = time.perf_counter()

            total_time = end - start

            batch_sizes.append(batch_size)
            total_times.append(total_time)
            memory_usage.append(predicted_memory_gib)

            print(f"{batch_size:>6} | {predicted_memory_gib:>12.2f} | {'OK':>10} | {total_time:>12.4f}")

        except Exception as e:
            print(f"{batch_size:>6} | {predicted_memory_gib:>12.2f} | {'FAIL':>10} | {'-':>12}")
            print(f"         Execution failed: {e}")
            skipped_sizes.append(batch_size)

        # Clear caches to free GPU memory before next batch size
        del result
        jax.clear_caches()

    print("="*70)

    if skipped_sizes:
        print(f"\nSkipped batch sizes due to memory constraints: {skipped_sizes}")

    if batch_sizes:
        print(f"\nSuccessfully benchmarked batch sizes: {batch_sizes}")
        print(f"Max feasible batch size on this GPU: {max(batch_sizes)}")

    # Get GPU name for plot title
    gpu_name = "Unknown GPU"
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '-L'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # Output format: "GPU 0: NVIDIA GeForce RTX 3090 (UUID: ...)"
            line = result.stdout.strip().split('\n')[0]
            # Extract the GPU name between ": " and " ("
            if ': ' in line and ' (' in line:
                gpu_name = line.split(': ')[1].split(' (')[0]
    except Exception:
        gpu_name = str(devices[0]).split('(')[0].strip() if devices else "Unknown GPU"

    return batch_sizes, total_times, memory_usage, skipped_sizes, gpu_name


def plot_results(batch_sizes, total_times, memory_usage=None, gpu_name="GPU", output_path="scripts/vmap_parallelism_benchmark.png"):
    """Plot the benchmark results.

    Parameters
    ----------
    batch_sizes : list
        List of batch sizes tested (log-scaled)
    total_times : list
        Total time to process the entire batch for each batch size
    memory_usage : list, optional
        Memory usage in GiB for each batch size
    gpu_name : str, optional
        Name of the GPU for the plot title
    output_path : str
        Path to save the plot
    """
    if not batch_sizes:
        print("No data to plot!")
        return None

    # Compute average time per invocation
    avg_times = [t / bs for t, bs in zip(total_times, batch_sizes)]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot timing on primary y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Batch Size (N)', fontsize=12)
    ax1.set_ylabel('Average Time per Invocation (seconds)', fontsize=12, color=color1)
    ax1.plot(batch_sizes, avg_times, 'b-o', linewidth=2, markersize=6, label='Avg Time')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Add reference line for single invocation time (ideal: constant)
    ax1.axhline(y=avg_times[0], color='g', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Ideal (constant at {avg_times[0]:.3f}s)')

    # Plot memory on secondary y-axis if provided
    ax2 = None
    if memory_usage:
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Predicted Peak Memory (GiB)', fontsize=12, color=color2)
        ax2.plot(batch_sizes, memory_usage, 'r-s', linewidth=2, markersize=6, label='Memory')
        ax2.tick_params(axis='y', labelcolor=color2)
        # Put ax1 in front of ax2 so legend/annotations appear on top
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)

    ax1.set_title(f'DISCO-EB CMB Spectrum: vmap Parallelism Scaling ({gpu_name})\n'
                  '(ellmax=2500, nmodes=128, kmin=1e-4, kmax=1.0)', fontsize=14)

    # Set x-axis ticks to batch sizes
    ax1.set_xticks(batch_sizes)
    ax1.set_xticklabels([str(bs) for bs in batch_sizes])

    ax1.grid(True, alpha=0.3)

    # Combined legend (zorder ensures it's on top of all plot elements)
    lines1, labels1 = ax1.get_legend_handles_labels()
    if memory_usage:
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        legend = ax1.legend(loc='best')
    legend.set_zorder(20)

    # Add text annotation with speedup at max batch size
    if len(batch_sizes) > 1:
        max_batch = batch_sizes[-1]
        speedup = avg_times[0] / avg_times[-1]
        avg_time_max = avg_times[-1]
        ax1.annotate(f'N={max_batch}: {avg_time_max:.3f}s/inv ({speedup:.2f}x speedup)',
                     xy=(max_batch, avg_times[-1]),
                     xytext=(max_batch * 0.3, (avg_times[0] + avg_times[-1]) / 2),
                     arrowprops=dict(arrowstyle='->', color='green'),
                     fontsize=10, color='green', zorder=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark vmap parallelism for compute_Cell_spectrum_from_cosmo_params'
    )
    parser.add_argument('N', type=int, help='Maximum batch size to test (will test 1 to N)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='scripts/vmap_parallelism_benchmark.png',
                        help='Output path for the plot')

    args = parser.parse_args()

    if args.N < 1:
        parser.error("N must be at least 1")

    print("="*60)
    print("DISCO-EB vmap Parallelism Benchmark")
    print("="*60)
    print(f"\nTesting batch sizes: 1, 2, 4, 8, ... up to {args.N}")
    print(f"Random seed: {args.seed}")
    print(f"Cosmological parameters: DISCO-Notebook with ~1.67% Gaussian noise")
    print(f"Computation settings: ellmax=2500, nmodes=128, kmin=1e-4, kmax=1.0")

    batch_sizes, total_times, memory_usage, _, gpu_name = run_benchmark(args.N, seed=args.seed)

    if batch_sizes:
        plot_results(batch_sizes, total_times, memory_usage, gpu_name=gpu_name, output_path=args.output)
    else:
        print("\nNo batch sizes could be executed. Consider reducing ellmax or nmodes.")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
