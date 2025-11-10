#!/usr/bin/env python3
"""
Simple test script to query the π 0.5 -LIBERO policy server.
Run this while the server is running (./run_inference_libero_gpu0.sh)
"""

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

def main():
    print("=" * 60)
    print("Testing π 0.5 -LIBERO Policy Server")
    print("=" * 60)
    
    # Initialize client
    print("\n[1/3] Connecting to policy server...")
    try:
        client = websocket_client_policy.WebsocketClientPolicy(
            host="localhost",
            port=8000
        )
        print("✓ Connected to server")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        print("\nMake sure the server is running:")
        print("  ./run_inference_libero_gpu0.sh")
        return False
    
    # Get server metadata
    print("\n[2/3] Getting server metadata...")
    try:
        metadata = client.get_server_metadata()
        print(f"✓ Server metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"✗ Failed to get metadata: {e}")
        return False
    
    # Create a test observation (LIBERO format)
    print("\n[3/3] Sending test observation and getting actions...")
    try:
        # Create random test images (replace with real images in your use case)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        wrist_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create test state: [x, y, z, axis_x, axis_y, axis_z, gripper, unused]
        state = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Prepare observation
        observation = {
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(image, 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_image, 224, 224)
            ),
            "observation/state": state,
            "prompt": "pick up the red block",
        }
        
        # Query policy
        result = client.infer(observation)
        actions = result["actions"]
        
        print(f"✓ Successfully received actions!")
        print(f"  Action chunk shape: {actions.shape}")
        print(f"  Number of actions: {len(actions)}")
        print(f"  Action dimension: {actions.shape[1]}")
        print(f"  First action: {actions[0]}")
        
        # Show timing if available
        if "server_timing" in result:
            print(f"\n  Server timing: {result['server_timing']}")
        if "policy_timing" in result:
            print(f"  Policy timing: {result['policy_timing']}")
            
    except Exception as e:
        print(f"✗ Failed to get actions: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Your policy server is working correctly.")
    print("=" * 60)
    print("\nYou can now:")
    print("  1. Use the simple client: uv run examples/simple_client/main.py --env LIBERO")
    print("  2. Write your own code using the WebsocketClientPolicy class")
    print("  3. Run full LIBERO evaluation (see examples/libero/README.md)")
    print("\nSee HOW_TO_QUERY_POLICY.md for detailed examples.")
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

