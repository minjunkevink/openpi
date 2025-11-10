#!/usr/bin/env python3
"""
Quick test script to verify the inference setup for π 0.5 -LIBERO.
This script tests that the checkpoint can be downloaded and the policy can be created.
"""

import os
import sys

# Set GPU 0 before importing JAX
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

def test_setup():
    print("=" * 60)
    print("Testing π 0.5 -LIBERO Inference Setup")
    print("=" * 60)
    
    # Test 1: Load config
    print("\n[1/3] Loading config...")
    try:
        config = _config.get_config("pi05_libero")
        print(f"✓ Config loaded: {config.model.__class__.__name__}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False
    
    # Test 2: Download checkpoint (this will cache it)
    print("\n[2/3] Downloading checkpoint (this may take a while)...")
    try:
        checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
        print(f"✓ Checkpoint available at: {checkpoint_dir}")
    except Exception as e:
        print(f"✗ Failed to download checkpoint: {e}")
        return False
    
    # Test 3: Create policy
    print("\n[3/3] Creating policy...")
    try:
        policy = policy_config.create_trained_policy(config, checkpoint_dir)
        print(f"✓ Policy created successfully")
        print(f"  Policy metadata: {policy.metadata}")
    except Exception as e:
        print(f"✗ Failed to create policy: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Setup is complete.")
    print("=" * 60)
    print("\nYou can now run inference using:")
    print("  ./run_inference_libero_gpu0.sh")
    print("\nOr manually:")
    print("  CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --env LIBERO")
    
    return True

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)

