#!/usr/bin/env python3
"""
I6: Split vs Single Bind Group Performance Demo - Python Wrapper

Python wrapper for the Rust performance benchmark that compares bind group churn
vs single bind group performance. Validates CSV output and provides easy access
to the benchmark from Python environments.
"""

import argparse
import subprocess
import sys
import os
import csv
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run split vs single bind group performance benchmark"
    )
    parser.add_argument(
        "--frames", 
        type=int, 
        default=600,
        help="Number of frames to render (default: 600)"
    )
    parser.add_argument(
        "--objects",
        type=int,
        default=1000,
        help="Number of objects to render (default: 1000)"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default="artifacts/perf/I6_bg_churn.csv",
        help="Output CSV path (default: artifacts/perf/I6_bg_churn.csv)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()

def ensure_output_dir(output_path):
    """Ensure output directory exists."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

def run_rust_benchmark(frames, objects, output_path):
    """Run the Rust benchmark via subprocess."""
    # Find the project root (directory containing Cargo.toml)
    project_root = Path(__file__).parent.parent.parent
    logger.info(f"Project root: {project_root}")
    # GPU gating: only execute if FORGE3D_CI_GPU=1; otherwise build-only and skip run
    if os.environ.get("FORGE3D_CI_GPU", "0") != "1":
        logger.info("FORGE3D_CI_GPU != 1 -> building only (skipping benchmark run)")
        try:
            result = subprocess.run([
                "cargo", "build", "--example", "split_vs_single_bg", "--release"
            ], cwd=project_root, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Build failed")
                if result.stderr:
                    logger.error(result.stderr)
                return False
            return True
        except Exception as e:
            logger.error(f"Build exception: {e}")
            return False
    
    # Build the command
    cmd = [
        "cargo", "run", "--example", "split_vs_single_bg", "--release",
        "--", "--frames", str(frames), "--objects", str(objects), "--out", str(output_path)
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run with RUST_LOG=info as specified in acceptance criteria
        env = os.environ.copy()
        env["RUST_LOG"] = "info"
        
        result = subprocess.run(
            cmd,
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info("Rust benchmark completed successfully")
            if result.stdout:
                logger.info(f"Stdout: {result.stdout}")
        else:
            logger.error(f"Rust benchmark failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"Stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Benchmark timed out after 5 minutes")
        return False
    except FileNotFoundError:
        logger.error("cargo command not found. Make sure Rust is installed.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running benchmark: {e}")
        return False
    
    return True

def validate_csv_output(output_path, expected_rows):
    """Validate that the CSV output exists and has required structure."""
    output_file = Path(output_path)
    
    if not output_file.exists():
        logger.error(f"Output CSV file does not exist: {output_path}")
        return False
    
    logger.info(f"Validating CSV file: {output_path}")
    
    try:
        with open(output_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check required columns
            required_columns = {'Configuration', 'TotalTimeMs', 'AvgFrameTimeMs'}
            actual_columns = set(reader.fieldnames or [])
            
            missing_columns = required_columns - actual_columns
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                logger.error(f"Available columns: {actual_columns}")
                return False
            
            # Count rows and validate data
            rows = list(reader)
            row_count = len(rows)
            
            logger.info(f"CSV validation results:")
            logger.info(f"  Columns: {list(reader.fieldnames)}")
            logger.info(f"  Row count: {row_count}")
            
            if row_count < 2:  # Should have at least 2 configs (split vs single)
                logger.error(f"Expected at least 2 rows (configurations), got {row_count}")
                return False
            
            # Validate that we have both configurations
            configs = {row['Configuration'] for row in rows}
            expected_configs = {'SplitBindGroups', 'SingleBindGroup'}
            
            if not expected_configs.issubset(configs):
                logger.error(f"Missing expected configurations: {expected_configs - configs}")
                logger.error(f"Found configurations: {configs}")
                return False
            
            # Validate numeric data
            for i, row in enumerate(rows):
                try:
                    float(row['TotalTimeMs'])
                    float(row['AvgFrameTimeMs'])
                except ValueError as e:
                    logger.error(f"Row {i}: Invalid numeric data - {e}")
                    return False
            
            logger.info("✅ CSV validation passed")
            
            # Print summary for user
            print("\n=== BENCHMARK RESULTS ===")
            for row in rows:
                config = row['Configuration']
                avg_time = float(row['AvgFrameTimeMs'])
                total_time = float(row['TotalTimeMs'])
                print(f"{config}:")
                print(f"  Average frame time: {avg_time:.3f} ms")
                print(f"  Total time: {total_time:.3f} ms")
            
            # Calculate improvement
            split_time = next((float(r['AvgFrameTimeMs']) for r in rows if r['Configuration'] == 'SplitBindGroups'), None)
            single_time = next((float(r['AvgFrameTimeMs']) for r in rows if r['Configuration'] == 'SingleBindGroup'), None)
            
            if split_time and single_time:
                improvement = split_time / single_time
                print(f"\nPerformance improvement: {improvement:.2f}x faster with single bind group")
            
            return True
            
    except Exception as e:
        logger.error(f"Error validating CSV: {e}")
        return False

def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("I6: Split vs Single Bind Group Performance Demo")
    logger.info(f"Configuration: {args.frames} frames, {args.objects} objects")
    logger.info(f"Output: {args.out}")
    
    # Ensure output directory exists
    ensure_output_dir(args.out)
    
    # Run the Rust benchmark
    if not run_rust_benchmark(args.frames, args.objects, args.out):
        logger.error("Benchmark execution failed")
        return 1
    
    # Validate CSV output
    if not validate_csv_output(args.out, args.frames):
        logger.error("CSV validation failed")
        return 1
    
    logger.info(f"✅ Benchmark completed successfully. Results written to: {args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())