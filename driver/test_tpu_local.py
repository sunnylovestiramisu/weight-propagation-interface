import os
import sys
import time
import logging
import threading
import numpy as np

# Append consumer directory to import wpi_client
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "consumer"))
from wpi_client.client import WPIClient
from wpi_client.proto import wpi_pb2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test():
    logger.info("Starting local TPU weight propagation test...")
    
    # 1. Connect to driver
    # Force backend to TPU
    os.environ["WPI_BACKEND"] = "tpu"
    local_socket_dir = os.environ.get("WPI_SOCKET_DIR", "/run/wpi/sockets")
    client = WPIClient(socket_dir=local_socket_dir, driver_host="localhost", driver_port=50051)
    
    buffer_id = "wb-tpu-test"
    size_bytes = 10 * 1024 * 1024  # 10 MB
    
    # 2. Stage source shard (shard 0 of 2)
    logger.info("Staging source buffer (shard 0 of 2)...")
    client.stage_weight(
        buffer_id=buffer_id,
        size_bytes=size_bytes,
        claim_id="claim-source",
        shard_index=0,
        total_shards=2
    )
    
    # 3. Stage target shard (shard 1 of 2)
    logger.info("Staging target buffer (shard 1 of 2)...")
    client.stage_weight(
        buffer_id=buffer_id,
        size_bytes=size_bytes,
        claim_id="claim-target",
        shard_index=1,
        total_shards=2
    )
    
    # 4. Connect notify socket for target buffer
    logger.info("Connecting notify socket for target buffer...")
    client.connect_notify_socket(
        buffer_id=buffer_id,
        shard_index=1,
        total_shards=2
    )
    
    # 5. Map source shard and write dummy data
    logger.info("Mapping source buffer and writing test pattern...")
    src_array = client.map_memory(
        buffer_id=buffer_id,
        size_bytes=size_bytes,
        shard_index=0,
        total_shards=2
    )
    
    # Create test pattern: values from 0 to 255 repeating
    pattern = np.array([i % 256 for i in range(size_bytes)], dtype=np.uint8)
    src_array[:] = pattern
    logger.info(f"Source pattern written. First 10 bytes: {src_array[:10]}")
    
    # 6. Set up a listener thread for target READY notification
    def wait_target_ready():
        logger.info("Waiting for target READY signal in helper thread...")
        client.wait_for_ready(timeout=10.0)
        logger.info("Target READY signal received!")
        
        # Map target shard and verify data
        logger.info("Mapping target buffer and verifying test pattern...")
        target_client = WPIClient(socket_dir=local_socket_dir, driver_host="localhost", driver_port=50051)
        dst_array = target_client.map_memory(
            buffer_id=buffer_id,
            size_bytes=size_bytes,
            shard_index=1,
            total_shards=2
        )
        
        logger.info(f"Target memory mapped. First 10 bytes: {dst_array[:10]}")
        match = np.array_equal(src_array, dst_array)
        if match:
            logger.info("SUCCESS: Target memory matches source memory perfectly!")
        else:
            logger.error("FAILURE: Target memory DOES NOT match source memory!")
            # Show diff
            diff_indices = np.where(src_array != dst_array)[0]
            logger.error(f"First 5 diff indices: {diff_indices[:5]}")
            logger.error(f"Source values: {src_array[diff_indices[:5]]}")
            logger.error(f"Target values: {dst_array[diff_indices[:5]]}")
            sys.exit(1)
            
    wait_thread = threading.Thread(target=wait_target_ready)
    wait_thread.start()
    
    # Give the thread a split second to block on wait_for_ready
    time.sleep(0.5)
    
    # 7. Trigger NodePropagate in SCATTER mode to target 127.0.0.1
    # We assign source range [0, size_bytes] to target shard 1
    logger.info("Triggering NodePropagate (SCATTER mode)...")
    assignment = wpi_pb2.ShardAssignment(
        target_node_id="127.0.0.1",
        shard_index=1,
        offset_bytes=0,
        length_bytes=size_bytes,
    )
    
    client.propagate(
        buffer_id=client._effective_buffer_id(buffer_id, 0, 2),
        target_node_ids=["127.0.0.1"],
        mode=1,  # SCATTER mode
        shard_assignments=[assignment]
    )
    
    # 8. Clean up
    wait_thread.join()
    
    logger.info("Cleaning up buffers...")
    client.unstage_weight("claim-source")
    client.unstage_weight("claim-target")
    client.close()
    
    logger.info("Test run finished.")

if __name__ == "__main__":
    run_test()
