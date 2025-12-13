"""
Example: GPU Selection and Management

This example demonstrates various ways to select and manage GPUs in MRLM.
"""

import torch
from mrlm.distributed import (
    get_available_gpus,
    get_gpu_info,
    select_gpus,
    setup_gpu_environment,
    print_gpu_info,
    auto_select_gpus_for_model,
    get_gpu_memory_usage,
)


def example_1_check_available_gpus():
    """Example 1: Check available GPUs."""
    print("\n" + "=" * 70)
    print("Example 1: Check Available GPUs")
    print("=" * 70)

    # Get list of GPU IDs
    available = get_available_gpus()
    print(f"\nAvailable GPU IDs: {available}")
    print(f"Number of GPUs: {len(available)}")

    if not available:
        print("No GPUs found - will use CPU")
        return

    # Get detailed GPU information
    gpu_info = get_gpu_info()
    print("\nDetailed GPU Information:")
    for info in gpu_info:
        print(f"\nGPU {info['id']}:")
        print(f"  Name: {info['name']}")
        print(f"  Memory: {info['memory_total']} MB")
        print(f"  Compute Capability: {info['compute_capability']}")
        print(f"  Multiprocessors: {info['multi_processor_count']}")


def example_2_select_specific_gpus():
    """Example 2: Select specific GPUs."""
    print("\n" + "=" * 70)
    print("Example 2: Select Specific GPUs")
    print("=" * 70)

    available = get_available_gpus()
    if not available:
        print("No GPUs available")
        return

    # Select first GPU
    gpus = select_gpus(gpu_ids=[available[0]])
    print(f"\nSelected GPU {gpus[0]}")

    # If multiple GPUs available, select first two
    if len(available) >= 2:
        gpus = select_gpus(gpu_ids=available[:2])
        print(f"Selected GPUs {gpus}")


def example_3_select_by_count():
    """Example 3: Select by number of GPUs."""
    print("\n" + "=" * 70)
    print("Example 3: Select by Count")
    print("=" * 70)

    # Select first 2 GPUs
    try:
        gpus = select_gpus(num_gpus=2)
        print(f"\nRequested 2 GPUs, got: {gpus}")
    except Exception as e:
        print(f"Could not select 2 GPUs: {e}")

    # Select all available GPUs
    gpus = select_gpus()
    print(f"All available GPUs: {gpus}")


def example_4_select_by_memory():
    """Example 4: Select GPUs by memory size."""
    print("\n" + "=" * 70)
    print("Example 4: Select by Memory Size")
    print("=" * 70)

    # Select GPUs with at least 10GB memory
    try:
        gpus = select_gpus(min_memory_mb=10000)
        print(f"\nGPUs with >=10GB memory: {gpus}")
    except ValueError as e:
        print(f"No GPUs with sufficient memory: {e}")

    # Select GPUs with at least 1GB memory (should work)
    try:
        gpus = select_gpus(min_memory_mb=1000)
        print(f"GPUs with >=1GB memory: {gpus}")
    except Exception:
        print("No GPUs found")


def example_5_auto_select_for_model():
    """Example 5: Auto-select GPUs based on model size."""
    print("\n" + "=" * 70)
    print("Example 5: Auto-select for Model Size")
    print("=" * 70)

    # For a 7B parameter model (~14GB)
    try:
        gpus = auto_select_gpus_for_model(model_size_gb=14)
        print(f"\nGPUs for 7B model (14GB): {gpus}")
    except Exception as e:
        print(f"Could not select GPUs: {e}")

    # For a 70B parameter model (~140GB)
    try:
        gpus = auto_select_gpus_for_model(model_size_gb=140)
        print(f"GPUs for 70B model (140GB): {gpus}")
        print("(Will use FSDP if needed)")
    except Exception as e:
        print(f"Could not select GPUs: {e}")


def example_6_setup_gpu_environment():
    """Example 6: Setup complete GPU environment."""
    print("\n" + "=" * 70)
    print("Example 6: Setup GPU Environment")
    print("=" * 70)

    # Setup with automatic selection
    device = setup_gpu_environment(num_gpus=1)
    print(f"\nSetup complete. Using device: {device}")

    # Check if we're on CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")


def example_7_monitor_gpu_memory():
    """Example 7: Monitor GPU memory usage."""
    print("\n" + "=" * 70)
    print("Example 7: Monitor GPU Memory")
    print("=" * 70)

    available = get_available_gpus()
    if not available:
        print("No GPUs available")
        return

    print("\nCurrent GPU Memory Usage:")
    for gpu_id in available:
        allocated, total, utilization = get_gpu_memory_usage(gpu_id)
        print(f"\nGPU {gpu_id}:")
        print(f"  Allocated: {allocated} MB")
        print(f"  Total: {total} MB")
        print(f"  Utilization: {utilization:.1f}%")

        # Warning if highly utilized
        if utilization > 80:
            print(f"  ⚠️  Warning: GPU {gpu_id} is {utilization:.1f}% full!")


def example_8_print_gpu_summary():
    """Example 8: Print comprehensive GPU summary."""
    print("\n" + "=" * 70)
    print("Example 8: GPU Summary")
    print("=" * 70)

    print("\nComprehensive GPU Information:")
    print_gpu_info()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MRLM GPU Selection Examples")
    print("=" * 70)

    # Run all examples
    example_1_check_available_gpus()
    example_2_select_specific_gpus()
    example_3_select_by_count()
    example_4_select_by_memory()
    example_5_auto_select_for_model()
    example_6_setup_gpu_environment()
    example_7_monitor_gpu_memory()
    example_8_print_gpu_summary()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
