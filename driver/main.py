import os
import random
import time
import logging
import concurrent.futures
import grpc
import numpy as np
import socket
import threading
import array
import ctypes
import mmap

try:
    import cupy
    import cupy.cuda.nccl
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3

class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("location", CUmemLocation),
        ("flags", ctypes.c_int)
    ]

class AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4)
    ]

class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", AllocFlags)
    ]

class CudaAllocator:
    def __init__(self):
        self.libcuda = ctypes.CDLL("/usr/local/nvidia/lib64/libcuda.so.1")
        if self.libcuda.cuInit(0) != 0:
            raise Exception("cuInit failed")
        
        self.contexts = {}
        self.devices = {}
        # Pre-initialize context for device 0 (default)
        self.get_or_create_context(0)
        self.device = self.devices[0]
        self.ctx = self.contexts[0]

    def get_or_create_context(self, device_id: int):
        if device_id in self.contexts:
            return self.contexts[device_id]
        
        logger.info(f"Initializing CUDA Context for Absolute GPU {device_id}...")
        dev = ctypes.c_int()
        if self.libcuda.cuDeviceGet(ctypes.byref(dev), device_id) != 0:
            raise Exception(f"cuDeviceGet failed for device {device_id}")
        
        ctx = ctypes.c_void_p()
        if self.libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) != 0:
            raise Exception(f"cuCtxCreate failed for device {device_id}")
        
        self.contexts[device_id] = ctx
        self.devices[device_id] = dev
        logger.info(f"CUDA Context for GPU {device_id} initialized successfully.")
        return ctx

    def allocate_and_export(self, size_bytes: int, device_id: int = 0) -> tuple:
        logger.info(f"Allocating memory on GPU {device_id}...")
        ctx = self.get_or_create_context(device_id)
        self.libcuda.cuCtxSetCurrent(ctx)
        
        dev = self.devices[device_id]

        prop = CUmemAllocationProp()
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = dev.value
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

        granularity = ctypes.c_size_t()
        self.libcuda.cuMemGetAllocationGranularity(ctypes.byref(granularity), ctypes.byref(prop), CU_MEM_ALLOC_GRANULARITY_MINIMUM)
        
        if size_bytes % granularity.value != 0:
            size_bytes = ((size_bytes // granularity.value) + 1) * granularity.value

        handle = ctypes.c_ulonglong()
        if self.libcuda.cuMemCreate(ctypes.byref(handle), ctypes.c_size_t(size_bytes), ctypes.byref(prop), ctypes.c_ulonglong(0)) != 0:
            raise Exception("cuMemCreate failed")
            
        shareable_handle = ctypes.c_int()
        if self.libcuda.cuMemExportToShareableHandle(ctypes.byref(shareable_handle), handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, ctypes.c_ulonglong(0)) != 0:
            raise Exception("cuMemExportToShareableHandle failed")

        device_ptr = ctypes.c_ulonglong()
        if self.libcuda.cuMemAddressReserve(ctypes.byref(device_ptr), ctypes.c_size_t(size_bytes), ctypes.c_size_t(0), ctypes.c_ulonglong(0), ctypes.c_ulonglong(0)) != 0:
            raise Exception("cuMemAddressReserve failed")
            
        if self.libcuda.cuMemMap(device_ptr, ctypes.c_size_t(size_bytes), ctypes.c_size_t(0), handle, ctypes.c_ulonglong(0)) != 0:
            raise Exception("cuMemMap failed")

        desc = CUmemAccessDesc()
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = dev.value
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        if self.libcuda.cuMemSetAccess(device_ptr, ctypes.c_size_t(size_bytes), ctypes.byref(desc), ctypes.c_size_t(1)) != 0:
            raise Exception("cuMemSetAccess failed")

        return shareable_handle.value, handle.value, device_ptr.value

    def free(self, handle: int, device_ptr: int, size_bytes: int):
        if self.libcuda.cuMemUnmap(ctypes.c_ulonglong(device_ptr), ctypes.c_size_t(size_bytes)) != 0:
            raise Exception("cuMemUnmap failed")
        if self.libcuda.cuMemAddressFree(ctypes.c_ulonglong(device_ptr), ctypes.c_size_t(size_bytes)) != 0:
            raise Exception("cuMemAddressFree failed")
        if self.libcuda.cuMemRelease(ctypes.c_ulonglong(handle)) != 0:
            raise Exception("cuMemRelease failed")

# These will be generated by the Dockerfile build process
try:
    import wpi_pb2
    import wpi_pb2_grpc
except ImportError:
    # Fallback for local development/IDE
    import sys
    sys.path.append('.')
    import wpi_pb2
    import wpi_pb2_grpc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cuda_allocator = None
try:
    cuda_allocator = CudaAllocator()
    logger.info("CudaAllocator initialized successfully.")
except Exception as e:
    logger.warning(f"Could not initialize CudaAllocator (fallback to mock): {e}")

MOUNT_PATH = "/dev/wpi/weights"
SOCKET_DIR = os.environ.get("WPI_SOCKET_DIR", "/run/wpi/sockets")
FILE_SIZE_GIB = 10

ALLOCATED_BUFFERS = {}  # mapping: buffer_id -> {"device_ptr": device_ptr, "size_bytes": size_bytes, "ref_count": int}
KNOWN_CLAIMS = {}       # mapping: claim_id -> buffer_id

DRIVER_BACKEND = os.environ.get("WPI_BACKEND")
if not DRIVER_BACKEND:
    if os.path.exists("/dev/accel0"):
        DRIVER_BACKEND = "tpu"
    else:
        DRIVER_BACKEND = "cuda"
logger.info(f"WPI Driver starting in {DRIVER_BACKEND.upper()} backend mode")

def start_nccl_target_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = 50052
    server.bind(("0.0.0.0", port))
    server.listen(10)
    logger.info(f"NCCL target receiver server listening on 0.0.0.0:{port}")
    
def send_pre_update_signal(buffer_id: str):
    """Send PRE_UPDATE signal to local consumers before writing new weights.
    
    This allows consumers (e.g., vLLM/SGLang rollout workers) to flush
    caches (like the KV cache) before the shared VRAM buffer is overwritten.
    """
    if buffer_id not in ALLOCATED_BUFFERS:
        return
    info = ALLOCATED_BUFFERS[buffer_id]
    notify_sockets = info.get("notify_sockets", [])
    broken_sockets = []
    for s in notify_sockets:
        try:
            s.sendall(b"PRE_UPDATE\n")
            logger.debug(f"Sent PRE_UPDATE notification for buffer {buffer_id}")
        except Exception as e:
            logger.warning(f"Failed to send PRE_UPDATE: {e}")
            broken_sockets.append(s)
    for s in broken_sockets:
        if s in notify_sockets:
            notify_sockets.remove(s)

def handle_target_connection(conn, addr):
    try:
        logger.info(f"Accepted NCCL transfer connection from {addr}")
        
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(1024)
            if not chunk:
                break
            buf += chunk
        
        if not buf:
            conn.close()
            return
        
        parts = buf.split(b"\n", 5)
        buffer_id = parts[0].decode('utf-8')
        world_size = int(parts[1].decode('utf-8'))
        rank = int(parts[2].decode('utf-8'))
        # Parse propagation mode: 0=BROADCAST, 1=SCATTER
        mode = int(parts[3].decode('utf-8')) if len(parts) > 4 else 0
        
        # For SCATTER, parse the receive offset and length
        scatter_offset = 0
        scatter_length = 0
        if mode == 1 and len(parts) > 5:
            scatter_meta = parts[4].decode('utf-8')
            meta_parts = scatter_meta.split(',')
            scatter_offset = int(meta_parts[0])
            scatter_length = int(meta_parts[1])
        
        nccl_id_bytes = parts[-1]
        while len(nccl_id_bytes) < 128:
            nccl_id_bytes += conn.recv(128 - len(nccl_id_bytes))
            
        mode_str = "SCATTER" if mode == 1 else "BROADCAST"
        logger.info(f"Target received buffer_id: {buffer_id}, world_size: {world_size}, "
                    f"rank: {rank}, mode: {mode_str} and nccl_id.")
        
        if buffer_id not in ALLOCATED_BUFFERS:
            logger.error(f"Target buffer {buffer_id} not found in ALLOCATED_BUFFERS!")
            conn.sendall(b"ERROR\n")
            conn.close()
            return
            
        info = ALLOCATED_BUFFERS[buffer_id]
        device_ptr = info["device_ptr"]
        size_bytes = info["size_bytes"]
        
        # Send pre-update signal to local consumers on this target node
        send_pre_update_signal(buffer_id)
        
        conn.sendall(b"OK\n")
        
        if CUPY_AVAILABLE and cuda_allocator:
            logger.info(f"Target initializing NCCL comm...")
            
            # Wrap VMM device_ptr through cupy's memory management
            # so NCCL can properly handle the pointer
            mem = cupy.cuda.UnownedMemory(device_ptr, size_bytes, None)
            memptr = cupy.cuda.MemoryPointer(mem, 0)
            
            with cupy.cuda.Device(info["gpu_id"]):
                comm = cupy.cuda.nccl.NcclCommunicator(world_size, nccl_id_bytes, rank)
                
                start_time = time.time()
                
                if mode == 1 and scatter_length > 0:
                    # SCATTER mode: receive our specific shard slice
                    shard_elements = scatter_length // 2  # fp16
                    recv_ptr = memptr.ptr + scatter_offset
                    logger.info(f"Target SCATTER recv: {scatter_length} bytes at offset {scatter_offset} "
                                f"(rank {rank}/{world_size})")
                    comm.recv(recv_ptr, shard_elements, cupy.cuda.nccl.NCCL_FLOAT16, 0, 0)
                else:
                    # BROADCAST mode: receive full buffer
                    num_elements = size_bytes // 2  # float16
                    logger.info(f"Target BROADCAST recv (Rank {rank}/{world_size})")
                    comm.bcast(memptr.ptr, num_elements, cupy.cuda.nccl.NCCL_FLOAT16, 0, 0)
                
                cupy.cuda.Device(info["gpu_id"]).synchronize()
            try:
                comm.destroy()
            except AttributeError:
                pass
            end_time = time.time()
            
            duration = end_time - start_time
            recv_bytes = scatter_length if (mode == 1 and scatter_length > 0) else size_bytes
            bandwidth_gbps = (recv_bytes / (1024**3)) / duration if duration > 0 else 0
            logger.info(f"Target NCCL {mode_str} recv complete in {duration:.4f}s. Bandwidth: {bandwidth_gbps:.2f} GB/s")
            info["ready"] = True
            
            # Notify all local consumers that rely on this buffer
            notify_sockets = info.get("notify_sockets", [])
            broken_sockets = []
            for s in notify_sockets:
                try:
                    s.sendall(b"READY\n")
                    logger.info("Sent READY notification to a listening consumer.")
                except Exception as e:
                    logger.warning(f"Failed to notify a consumer socket (it may have disconnected): {e}")
                    broken_sockets.append(s)
            
            # Clean up broken sockets
            for s in broken_sockets:
                if s in notify_sockets:
                    notify_sockets.remove(s)
        else:
            logger.error("CUPY NOT AVAILABLE for target transfer!")
            
    except Exception as e:
        logger.error(f"Error in target receiver thread: {e}")
    finally:
        conn.close()

def start_nccl_target_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = 50052
    server.bind(("0.0.0.0", port))
    server.listen(10)
    logger.info(f"NCCL target receiver server listening on 0.0.0.0:{port}")
    
    while True:
        try:
            conn, addr = server.accept()
            t = threading.Thread(target=handle_target_connection, args=(conn, addr), daemon=True)
            t.start()
        except Exception as e:
            logger.error(f"Error accepting target connection: {e}")


def pass_fd_server(sock_path: str, buffer_id: str):
    """
    Runs in a background thread. Listens on a UNIX socket and uses sendmsg
    to pass the file descriptor to any connecting client.
    Now supports Dynamic Relocation for cross-GPU isolation.
    """
    if os.path.exists(sock_path):
        os.remove(sock_path)
        
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    os.chmod(sock_path, 0o777)  # Allow non-root Ray workers to connect
    server.listen(5)
    logger.info(f"FD passing server listening on {sock_path} for buffer {buffer_id}")
    
    while True:
        try:
            conn, addr = server.accept()
            logger.info(f"Client connected to {sock_path}. Reading metadata...")
            
            # 1. Read absolute GPU index from client
            data = conn.recv(1024).decode('utf-8').strip()
            target_gpu = 0
            if data.startswith("GPU="):
                try:
                    target_gpu = int(data.split("=")[1])
                    logger.info(f"Client requested absolute GPU ID: {target_gpu}")
                except ValueError:
                    logger.warning(f"Invalid GPU metadata: {data}, defaulting to index 0")

            if buffer_id not in ALLOCATED_BUFFERS:
                logger.error(f"Buffer {buffer_id} missing from tracking inside pass_fd_server!")
                conn.close()
                continue
                
            info = ALLOCATED_BUFFERS[buffer_id]
            current_gpu = info.get("gpu_id", 0)
            fd_to_send = info["fd"]

            # 2. Perform Dynamic Relocation if target_gpu != current_gpu
            if target_gpu != current_gpu:
                relocate_key = f"{buffer_id}_gpu{target_gpu}"
                if relocate_key in ALLOCATED_BUFFERS:
                    logger.info(f"Relocation: Reusing already relocated buffer {relocate_key}")
                    fd_to_send = ALLOCATED_BUFFERS[relocate_key]["fd"]
                else:
                    logger.info(f"GPU Mismatch for client! Staged on GPU {current_gpu}, Client wants GPU {target_gpu}. Relocating...")
                    try:
                        size_bytes = info["size_bytes"]
                        
                        # Allocate on target GPU
                        new_fd, new_handle, new_dptr = cuda_allocator.allocate_and_export(size_bytes, device_id=target_gpu)
                        logger.info(f"Relocation: Allocated on GPU {target_gpu}. FD: {new_fd}")

                        if CUPY_AVAILABLE:
                            # Switch context to source to read
                            src_ctx = cuda_allocator.get_or_create_context(current_gpu)
                            cuda_allocator.libcuda.cuCtxSetCurrent(src_ctx)
                            
                            src_mem = cupy.cuda.UnownedMemory(info["device_ptr"], size_bytes, None)
                            src_ptr = cupy.cuda.MemoryPointer(src_mem, 0)
                            src_arr = cupy.ndarray((size_bytes // 2,), dtype=cupy.float16, memptr=src_ptr)

                            # Switch context to target to write
                            dst_ctx = cuda_allocator.get_or_create_context(target_gpu)
                            cuda_allocator.libcuda.cuCtxSetCurrent(dst_ctx)
                            
                            dst_mem = cupy.cuda.UnownedMemory(new_dptr, size_bytes, None)
                            dst_ptr = cupy.cuda.MemoryPointer(dst_mem, 0)
                            dst_arr = cupy.ndarray((size_bytes // 2,), dtype=cupy.float16, memptr=dst_ptr)

                            logger.info("Relocation: Starting CuPy VRAM D2D copy...")
                            dst_arr[:] = src_arr[:]
                            cupy.cuda.Device(target_gpu).synchronize()
                            logger.info("Relocation: copy completed.")
                        else:
                            raise Exception("CuPy not available for D2D copy.")

                        # 4. Store in ALLOCATED_BUFFERS for future reuse and cleanup
                        ALLOCATED_BUFFERS[relocate_key] = {
                            "handle": new_handle,
                            "device_ptr": new_dptr,
                            "fd": new_fd,
                            "size_bytes": size_bytes,
                            "gpu_id": target_gpu,
                            "notify_sockets": [],
                            "ready": True,
                        }
                        fd_to_send = new_fd
                        logger.info(f"Relocation: Stored relocated buffer as {relocate_key}")
                    except Exception as e:
                        logger.error(f"Dynamic Relocation failed: {e}")
                        conn.sendall(b"ERROR_RELOCATION_FAILED\n")
                        conn.close()
                        continue

            logger.info(f"Sending FD {fd_to_send} to client.")
            msg = b"OK\n"
            fds = array.array("i", [fd_to_send])
            conn.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)])
            conn.close()
        except Exception as e:
            logger.error(f"Error passing fd: {e}")

def notify_server(sock_path: str, buffer_id: str):
    """
    Runs in a background thread. Listens on a UNIX socket and collects
    consumer connections that are waiting for VRAM update notifications.
    """
    if os.path.exists(sock_path):
        try:
            os.remove(sock_path)
        except OSError:
            pass
        
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(sock_path)
        os.chmod(sock_path, 0o777)  # Allow non-root Ray workers to connect
    except OSError as e:
        logger.error(f"Failed to bind notify server at {sock_path}: {e}")
        return
    server.listen(100)
    logger.info(f"Notify passing server listening on {sock_path}")
    
    while True:
        try:
            conn, addr = server.accept()
            logger.info(f"Consumer connected to notify socket {sock_path}.")
            if buffer_id in ALLOCATED_BUFFERS:
                info = ALLOCATED_BUFFERS[buffer_id]
                if "notify_sockets" not in info:
                    info["notify_sockets"] = []
                info["notify_sockets"].append(conn)
                if info.get("ready", False):
                    try:
                        conn.sendall(b"READY\n")
                        logger.info("Notify server: Sent READY notification to consumer immediately on connection.")
                    except Exception as e:
                        logger.warning(f"Notify server: Failed to send immediate READY: {e}")
            else:
                logger.error(f"Cannot accept notify connection: buffer {buffer_id} not active.")
                conn.close()
        except Exception as e:
            logger.error(f"Error accepting notify connection: {e}")
            break
def push_tpu_chunk(target_ip: str, buffer_id: str, src_offset: int, dst_offset: int, length: int, source_fd: int):
    """Push weight chunk from source_fd (at src_offset) to target_ip (at dst_offset) over TCP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((target_ip, 50053))
        
        # Send header: "buffer_id\ndst_offset\nlength\n\n"
        header = f"{buffer_id}\n{dst_offset}\n{length}\n\n".encode('utf-8')
        s.sendall(header)
        
        resp = s.recv(1024)
        if b"OK" not in resp:
            raise Exception(f"Target rejected transfer: {resp.decode('utf-8')}")
            
        sent = 0
        while sent < length:
            to_send = min(4 * 1024 * 1024, length - sent)
            data = os.pread(source_fd, to_send, src_offset + sent)
            if not data:
                break
            s.sendall(data)
            sent += len(data)
            
        if sent != length:
            raise Exception(f"Transfer incomplete: sent {sent}/{length} bytes")
            
        logger.info(f"TPU Push Client: pushed {sent} bytes to {target_ip} (dst_offset={dst_offset})")
    except Exception as e:
        logger.error(f"TPU Push Client failed for target {target_ip}: {e}")
        raise
    finally:
        s.close()

def handle_incoming_tpu_data(conn, addr):
    try:
        header = b""
        while b"\n\n" not in header:
            chunk = conn.recv(1024)
            if not chunk:
                break
            header += chunk
            
        if not header:
            return
            
        parts = header.split(b"\n")
        buffer_id = parts[0].decode('utf-8')
        dst_offset = int(parts[1])
        length = int(parts[2])
        
        logger.info(f"TPU Receiver: receiving {length} bytes for {buffer_id} at offset {dst_offset} from {addr}")
        
        if buffer_id not in ALLOCATED_BUFFERS:
            logger.error(f"TPU Receiver: buffer {buffer_id} not found!")
            conn.sendall(b"ERROR: Buffer not found\n")
            return
            
        info = ALLOCATED_BUFFERS[buffer_id]
        fd = info["fd"]
        
        send_pre_update_signal(buffer_id)
        
        conn.sendall(b"OK\n")
        
        received = 0
        while received < length:
            chunk = conn.recv(min(4 * 1024 * 1024, length - received))
            if not chunk:
                break
            os.pwrite(fd, chunk, dst_offset + received)
            received += len(chunk)
            
        if received != length:
            logger.error(f"TPU Receiver: incomplete transfer for {buffer_id}, got {received}/{length} bytes")
            return
            
        logger.info(f"TPU Receiver: successfully wrote {received} bytes for {buffer_id}")
        info["ready"] = True
        
        notify_sockets = info.get("notify_sockets", [])
        broken_sockets = []
        for s in notify_sockets:
            try:
                s.sendall(b"READY\n")
                logger.info("TPU Receiver: Sent READY notification to consumer.")
            except Exception as e:
                logger.warning(f"TPU Receiver: Failed to notify consumer: {e}")
                broken_sockets.append(s)
        for s in broken_sockets:
            if s in notify_sockets:
                notify_sockets.remove(s)
                
    except Exception as e:
        logger.error(f"TPU Receiver error: {e}")
    finally:
        conn.close()

def start_tpu_receiver_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = 50053
    server.bind(("0.0.0.0", port))
    server.listen(100)
    logger.info(f"WPI TPU Receiver Server listening on 0.0.0.0:50053")
    while True:
        try:
            conn, addr = server.accept()
            t = threading.Thread(target=handle_incoming_tpu_data, args=(conn, addr), daemon=True)
            t.start()
        except Exception as e:
            logger.error(f"Error accepting receiver connection: {e}")


class NodeService(wpi_pb2_grpc.NodeServiceServicer):
    def NodeStageWeight(self, request, context):
        # Determine effective buffer_id: use shard-scoped ID if sharding is active
        shard_index = getattr(request, 'shard_index', -1)
        total_shards = getattr(request, 'total_shards', 0)
        is_sharded = shard_index >= 0 and total_shards > 0

        effective_buffer_id = request.buffer_id
        if is_sharded:
            effective_buffer_id = f"{request.buffer_id}__shard_{shard_index}"

        logger.info(f"NodeStageWeight called for claim: {request.claim_id}, "
                    f"buffer: {request.buffer_id}, effective_id: {effective_buffer_id}"
                    f"{f', shard: {shard_index}/{total_shards}' if is_sharded else f' (Not sharded - raw shard_index={shard_index}, raw_total={total_shards})'}")
        
        try:
            KNOWN_CLAIMS[request.claim_id] = effective_buffer_id
            
            if effective_buffer_id in ALLOCATED_BUFFERS:
                logger.info(f"Buffer {effective_buffer_id} is already staged. Incrementing refcount.")
                ALLOCATED_BUFFERS[effective_buffer_id]["ref_count"] += 1
                return wpi_pb2.NodeStageWeightResponse()

            if not os.path.exists(SOCKET_DIR):
                os.makedirs(SOCKET_DIR, exist_ok=True)
            os.chmod(SOCKET_DIR, 0o777)  # Ensure non-root Ray workers can access sockets
                
            weight_size_bytes = request.size_bytes
            source_path = request.source_path
            
            if not weight_size_bytes or weight_size_bytes == 0:
                # Fallback to 10GB if the operator didn't specify
                logger.warning(f"No size_bytes provided by Operator for {effective_buffer_id}, falling back to 10GiB")
                weight_size_bytes = 10 * 1024 * 1024 * 1024

            if DRIVER_BACKEND == "tpu":
                shm_path = f"/dev/shm/wpi-{effective_buffer_id}"
                logger.info(f"TPU Stage: Allocating shared memory file {shm_path} ({weight_size_bytes} bytes)...")
                try:
                    os.makedirs(os.path.dirname(shm_path), exist_ok=True)
                    with open(shm_path, "wb") as f:
                        f.truncate(weight_size_bytes)
                    os.chmod(shm_path, 0o777)
                    
                    fd = os.open(shm_path, os.O_RDWR)
                    mapped_mem = mmap.mmap(fd, weight_size_bytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
                except Exception as e:
                    logger.error(f"TPU Stage failed: {e}")
                    context.abort(grpc.StatusCode.INTERNAL, f"Failed to allocate shared memory: {e}")
                    return wpi_pb2.NodeStageWeightResponse()
                
                ALLOCATED_BUFFERS[effective_buffer_id] = {
                    "fd": fd,
                    "shm_path": shm_path,
                    "mapped_mem": mapped_mem,
                    "size_bytes": weight_size_bytes,
                    "source_path": source_path,
                    "notify_sockets": [],
                    "ref_count": 1,
                    "shard_index": shard_index if is_sharded else -1,
                    "total_shards": total_shards if is_sharded else 0,
                    "parent_buffer": request.buffer_id,
                    "ready": False,
                }
                
                if source_path:
                    if os.path.exists(source_path):
                        logger.info(f"Loading real weights from {source_path} into shared memory...")
                        try:
                            from safetensors import safe_open
                            with safe_open(source_path, framework="pt", device="cpu") as f:
                                offset = 0
                                for key in f.keys():
                                    tensor = f.get_tensor(key)
                                    arr_bytes = tensor.numpy().tobytes()
                                    mapped_mem[offset:offset+len(arr_bytes)] = arr_bytes
                                    offset += len(arr_bytes)
                            logger.info(f"Successfully loaded safetensors into shared memory!")
                            ALLOCATED_BUFFERS[effective_buffer_id]["ready"] = True
                        except Exception as e:
                            logger.error(f"Failed to load safetensors: {e}")
                            context.abort(grpc.StatusCode.INTERNAL, f"Failed to load safetensors: {e}")
                            return wpi_pb2.NodeStageWeightResponse()
                    else:
                        msg = f"Source path {source_path} specified but does not exist!"
                        logger.error(msg)
                        context.abort(grpc.StatusCode.NOT_FOUND, msg)
                        return wpi_pb2.NodeStageWeightResponse()
                else:
                    logger.info("No source_path specified. Allocating empty shared memory locally.")
            else:
                if cuda_allocator:
                    gpu_id = 0
                    if is_sharded:
                        try:
                            num_gpus = cupy.cuda.runtime.getDeviceCount()
                            gpu_id = shard_index % num_gpus
                        except Exception:
                            pass
                    
                    logger.info(f"Setting CUDA context in current thread for GPU {gpu_id}...")
                    gpu_ctx = cuda_allocator.get_or_create_context(gpu_id)
                    cuda_allocator.libcuda.cuCtxSetCurrent(gpu_ctx)

                    logger.info(f"Using CUDA to allocate {weight_size_bytes} bytes for {effective_buffer_id} on GPU {gpu_id}...")
                    fd, handle, device_ptr = cuda_allocator.allocate_and_export(weight_size_bytes, gpu_id)
                    logger.info(f"CUDA alloc successful. Exported FD: {fd}, handle: {handle}, mapped device_ptr: {device_ptr}")
                    
                    ALLOCATED_BUFFERS[effective_buffer_id] = {
                        "device_ptr": device_ptr,
                        "size_bytes": weight_size_bytes,
                        "source_path": source_path,
                        "handle": handle,
                        "fd": fd,
                        "notify_sockets": [],
                        "ref_count": 1,
                        "gpu_id": gpu_id,
                        "shard_index": shard_index if is_sharded else -1,
                        "total_shards": total_shards if is_sharded else 0,
                        "parent_buffer": request.buffer_id,
                        "ready": False,
                    }
                    
                    if source_path:
                        if os.path.exists(source_path):
                            logger.info(f"Loading real weights from {source_path} into VRAM...")
                            try:
                                from safetensors import safe_open
                                import torch 
                                
                                mem = cupy.cuda.UnownedMemory(device_ptr, weight_size_bytes, None)
                                memptr = cupy.cuda.MemoryPointer(mem, 0)
                                
                                num_elements = weight_size_bytes // 2
                                device_array = cupy.ndarray((num_elements,), dtype=cupy.float16, memptr=memptr)
                                
                                with safe_open(source_path, framework="pt", device="cpu") as f:
                                    offset = 0
                                    for key in f.keys():
                                        tensor = f.get_tensor(key)
                                        elements = tensor.numel()
                                        src_ptr = tensor.data_ptr()
                                        dst_ptr = device_array.data.ptr + (offset * 2)
                                        
                                        cupy.cuda.runtime.memcpy(
                                            dst_ptr,
                                            src_ptr,
                                            elements * 2,
                                            cupy.cuda.runtime.memcpyHostToDevice
                                        )
                                        offset += elements
                                        
                                logger.info(f"Successfully loaded safetensors from {source_path} into VRAM!")
                                ALLOCATED_BUFFERS[effective_buffer_id]["ready"] = True
                            except Exception as e:
                                logger.error(f"Failed to load safetensors from {source_path}: {e}")
                                context.abort(grpc.StatusCode.INTERNAL, f"Failed to load safetensors: {e}")
                        else:
                            msg = f"Source path {source_path} specified but does not exist!"
                            logger.error(msg)
                            context.abort(grpc.StatusCode.NOT_FOUND, msg)
                            return wpi_pb2.NodeStageWeightResponse()
                    else:
                        logger.info("No source_path specified. Allocating empty VRAM buffer locally without disk load.")
                else:
                    msg = "CUDA allocator failed to initialize. Cannot allocate VRAM."
                    logger.error(msg)
                    context.abort(grpc.StatusCode.UNAVAILABLE, msg)
                    return wpi_pb2.NodeStageWeightResponse()
            
            # Start background FD-passing server using effective (shard-scoped) buffer ID
            sock_path = os.path.join(SOCKET_DIR, f"{effective_buffer_id}.sock")
            logger.info(f"Starting FD passing server for buffer {effective_buffer_id}")
            t = threading.Thread(target=pass_fd_server, args=(sock_path, effective_buffer_id), daemon=True)
            t.start()
            
            # Start background notify server
            notify_sock_path = os.path.join(SOCKET_DIR, f"{effective_buffer_id}_notify.sock")
            ALLOCATED_BUFFERS[effective_buffer_id]["notify_sock_path"] = notify_sock_path
            t_notify = threading.Thread(target=notify_server, args=(notify_sock_path, effective_buffer_id), daemon=True)
            t_notify.start()

            return wpi_pb2.NodeStageWeightResponse()
            
        except Exception as e:
            logger.error(f"Error in NodeStageWeight: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def NodeUnstageWeight(self, request, context):
        logger.info(f"NodeUnstageWeight called for claim: {request.claim_id}")
        try:
            claim_id = request.claim_id
            if claim_id not in KNOWN_CLAIMS:
                logger.warning(f"NodeUnstageWeight: claim {claim_id} not found in tracking!")
                return wpi_pb2.NodeUnstageWeightResponse()
                
            buffer_id = KNOWN_CLAIMS.pop(claim_id)
            if buffer_id in ALLOCATED_BUFFERS:
                info = ALLOCATED_BUFFERS[buffer_id]
                info["ref_count"] -= 1
                logger.info(f"Decremented ref_count for {buffer_id} to {info['ref_count']}")
                
                if info["ref_count"] <= 0:
                    logger.info(f"Ref count for {buffer_id} is 0. Freeing all related allocations...")
                    if DRIVER_BACKEND == "tpu":
                        try:
                            if "mapped_mem" in info:
                                info["mapped_mem"].close()
                            if "fd" in info:
                                os.close(info["fd"])
                            if "shm_path" in info and os.path.exists(info["shm_path"]):
                                os.remove(info["shm_path"])
                            logger.info(f"Successfully freed shared memory buffer {buffer_id}")
                        except Exception as e:
                            logger.error(f"Failed to free shared memory buffer {buffer_id}: {e}")
                        ALLOCATED_BUFFERS.pop(buffer_id, None)
                    else:
                        relocated_keys = [k for k in ALLOCATED_BUFFERS.keys() if k.startswith(f"{buffer_id}_gpu")]
                        all_to_free = [(buffer_id, info)] + [(k, ALLOCATED_BUFFERS[k]) for k in relocated_keys]
                        
                        for k, item in all_to_free:
                            if cuda_allocator and "handle" in item and "device_ptr" in item:
                                gpu_id = item.get("gpu_id", 0)
                                logger.info(f"Setting CUDA context to free {k} on GPU {gpu_id}...")
                                try:
                                    gpu_ctx = cuda_allocator.get_or_create_context(gpu_id)
                                    cuda_allocator.libcuda.cuCtxSetCurrent(gpu_ctx)
                                    cuda_allocator.free(item["handle"], item["device_ptr"], item["size_bytes"])
                                    logger.info(f"Successfully freed {k} on GPU {gpu_id}")
                                except Exception as e:
                                    logger.error(f"Failed to free allocation {k} on GPU {gpu_id}: {e}")
                                    
                                if "fd" in item:
                                    try:
                                        os.close(item["fd"])
                                    except OSError:
                                        pass
                                        
                            if k != buffer_id:
                                ALLOCATED_BUFFERS.pop(k, None)
                                
                        ALLOCATED_BUFFERS.pop(buffer_id, None)
                    logger.info(f"Fully cleared claims for buffer {buffer_id}.")
            else:
                logger.warning(f"NodeUnstageWeight: buffer {buffer_id} for claim {claim_id} not found in tracking!")
                
            return wpi_pb2.NodeUnstageWeightResponse()
            
        except Exception as e:
            logger.error(f"Error in NodeUnstageWeight: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def NodeRegisterWeight(self, request, context):
        logger.info(f"NodeRegisterWeight called for buffer: {request.buffer_id}")
        if request.buffer_id not in ALLOCATED_BUFFERS:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Buffer {request.buffer_id} not found")
        handle_str = str(ALLOCATED_BUFFERS[request.buffer_id].get("handle", "unknown-handle"))
        return wpi_pb2.NodeRegisterWeightResponse(dma_buf_handle=handle_str)

    def NodePropagate(self, request, context):
        targets_str = ", ".join(request.target_node_ids)
        propagate_mode = getattr(request, 'mode', 0)  # 0=BROADCAST, 1=SCATTER
        mode_str = "SCATTER" if propagate_mode == 1 else "BROADCAST"
        logger.info(f"NodePropagate called for buffer: {request.buffer_id} to [{targets_str}] mode={mode_str}")
        
        try:
            if request.buffer_id not in ALLOCATED_BUFFERS:
                raise Exception(f"Buffer {request.buffer_id} not found in local tracking.")
            
            info = ALLOCATED_BUFFERS[request.buffer_id]
            source_fd = info["fd"]
            size_bytes = info["size_bytes"]
            
            # Phase 1: Pre-update signal to local consumers
            send_pre_update_signal(request.buffer_id)
            
            if DRIVER_BACKEND == "tpu":
                target_ips = request.target_node_ids
                
                # We transfer chunks in parallel using ThreadPoolExecutor
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = []
                    if propagate_mode == 1 and hasattr(request, 'shard_assignments') and request.shard_assignments:
                        # SCATTER mode
                        logger.info(f"TPU SCATTER: pushing shard assignments...")
                        for assignment in request.shard_assignments:
                            target_ip = assignment.target_node_id
                            clean_ip = target_ip.split(":")[0]
                            
                            target_buffer_id = request.buffer_id
                            if "__shard_" in request.buffer_id:
                                base_id = request.buffer_id.split("__shard_")[0]
                                target_buffer_id = f"{base_id}__shard_{assignment.shard_index}"
                                
                            f = executor.submit(
                                push_tpu_chunk,
                                clean_ip,
                                target_buffer_id,
                                assignment.offset_bytes,
                                assignment.offset_bytes,
                                assignment.length_bytes,
                                source_fd
                            )
                            futures.append(f)
                    else:
                        # BROADCAST mode: push full buffer to all target IPs
                        logger.info(f"TPU BROADCAST: pushing full buffer to all targets: {target_ips}...")
                        for target_ip in target_ips:
                            clean_ip = target_ip.split(":")[0]
                            f = executor.submit(
                                push_tpu_chunk,
                                clean_ip,
                                request.buffer_id,
                                0,
                                0,
                                size_bytes,
                                source_fd
                            )
                            futures.append(f)
                            
                    # Wait for all pushes to complete
                    concurrent.futures.wait(futures)
                    for f in futures:
                        f.result() # raises exception if any push failed
                        
                logger.info(f"TPU Propagation complete for '{request.buffer_id}'")
            else:
                if not CUPY_AVAILABLE:
                    raise Exception("cupy is not available. Cannot perform NCCL transfer.")
                
                device_ptr = info["device_ptr"]
                num_elements = size_bytes // 2
                nccl_id_bytes = cupy.cuda.nccl.get_unique_id()
                
                target_ips = request.target_node_ids
                num_targets = len(target_ips)
                world_size = num_targets + 1
                
                logger.info(f"Source generated NCCL Unique ID. Connecting to {num_targets} targets (world_size={world_size})...")
                
                sockets = []
                for i, target_ip in enumerate(target_ips):
                    rank = i + 1
                    logger.info(f"Connecting to target {target_ip} to assign rank {rank}...")
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect((target_ip, 50052))
                    
                    # Build per-target scatter metadata if in SCATTER mode
                    scatter_meta = "0,0"  # offset,length placeholder
                    target_buffer_id = request.buffer_id
                    if propagate_mode == 1 and hasattr(request, 'shard_assignments') and request.shard_assignments:
                        for assignment in request.shard_assignments:
                            if assignment.target_node_id == target_ip:
                                scatter_meta = f"{assignment.offset_bytes},{assignment.length_bytes}"
                                if "__shard_" in request.buffer_id:
                                    base_id = request.buffer_id.split("__shard_")[0]
                                    target_buffer_id = f"{base_id}__shard_{assignment.shard_index}"
                                break
                    
                    # Send: buffer_id\nworld_size\nrank\nmode\nscatter_meta\nnccl_id_bytes
                    msg = f"{target_buffer_id}\n{world_size}\n{rank}\n{propagate_mode}\n{scatter_meta}\n".encode('utf-8') + nccl_id_bytes
                    s.sendall(msg)
                    
                    resp = s.recv(1024)
                    if b"OK" not in resp:
                        raise Exception(f"Target node {target_ip} rejected NCCL preparation: {resp}")
                    
                    sockets.append(s)
                    
                logger.info(f"Source (Rank 0) initializing NCCL comm...")
                
                # Wrap VMM device_ptr through cupy's memory management
                mem = cupy.cuda.UnownedMemory(device_ptr, size_bytes, None)
                memptr = cupy.cuda.MemoryPointer(mem, 0)
                
                with cupy.cuda.Device(info["gpu_id"]):
                    comm = cupy.cuda.nccl.NcclCommunicator(world_size, nccl_id_bytes, 0)
                    start_time = time.time()
                    
                    if propagate_mode == 1 and hasattr(request, 'shard_assignments') and request.shard_assignments:
                        logger.info(f"SCATTER mode: sending {len(request.shard_assignments)} shard assignments...")
                        rank_map = {}
                        for i, target_ip in enumerate(target_ips):
                            rank_map[target_ip] = i + 1
                        
                        for assignment in request.shard_assignments:
                            target_rank = rank_map.get(assignment.target_node_id)
                            if target_rank is None:
                                logger.warning(f"Shard assignment target {assignment.target_node_id} not in target list, skipping")
                                continue
                            
                            offset = assignment.offset_bytes
                            length = assignment.length_bytes
                            shard_elements = length // 2  # fp16
                            shard_ptr = memptr.ptr + offset
                            
                            logger.info(f"  Shard {assignment.shard_index}: sending {length} bytes "
                                        f"(offset={offset}) to rank {target_rank} ({assignment.target_node_id})")
                            comm.send(shard_ptr, shard_elements, cupy.cuda.nccl.NCCL_FLOAT16, target_rank, 0)
                        
                        cupy.cuda.Device(info["gpu_id"]).synchronize()
                        logger.info(f"SCATTER: All shard sends completed.")
                    else:
                        logger.info(f"BROADCAST mode: sending full buffer to all {num_targets} targets...")
                        comm.bcast(memptr.ptr, num_elements, cupy.cuda.nccl.NCCL_FLOAT16, 0, 0)
                        cupy.cuda.Device(info["gpu_id"]).synchronize()
                    
                try:
                    comm.destroy()
                except AttributeError:
                    pass
                end_time = time.time()
                
                duration = end_time - start_time
                bandwidth_gbps = (size_bytes / (1024**3)) / duration if duration > 0 else 0
                logger.info(f"Source NCCL {mode_str} complete in {duration:.4f}s. Bandwidth: {bandwidth_gbps:.2f} GB/s")
                
                for s in sockets:
                    s.close()
            
            # Send READY to local consumers on the SOURCE node.
            info["ready"] = True
            notify_sockets = info.get("notify_sockets", [])
            broken_sockets = []
            for s in notify_sockets:
                try:
                    s.sendall(b"READY\n")
                    logger.info("Source: Sent READY notification to a local consumer.")
                except Exception as e:
                    logger.warning(f"Source: Failed to notify a local consumer: {e}")
                    broken_sockets.append(s)
            for s in broken_sockets:
                if s in notify_sockets:
                    notify_sockets.remove(s)
            
            return wpi_pb2.NodePropagateResponse()
            
        except Exception as e:
            logger.error(f"Error in NodePropagate: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def NodeTranslateAndMap(self, request, context):
        logger.info(f"NodeTranslateAndMap called for claim: {request.claim_id}")
        # Not fully needed anymore since we pass fd via hostpath, but we can return the standard socket.
        sock_path = os.path.join(SOCKET_DIR, f"{request.claim_id}.sock")
        return wpi_pb2.NodeTranslateAndMapResponse(device_path=sock_path)

class IdentityService(wpi_pb2_grpc.IdentityServiceServicer):
    def GetPluginInfo(self, request, context):
        return wpi_pb2.GetPluginInfoResponse(name="wpi-driver", vendor_version="v0.1.0")

    def GetPluginCapabilities(self, request, context):
        cap = wpi_pb2.GetPluginCapabilitiesResponse.PluginCapability
        return wpi_pb2.GetPluginCapabilitiesResponse(capabilities=[
            cap(capability=cap.Capability.ON_THE_FLY_RESHAPING)
        ])

def serve():
    if DRIVER_BACKEND == "tpu":
        tpu_server_thread = threading.Thread(target=start_tpu_receiver_server, daemon=True)
        tpu_server_thread.start()
    else:
        target_server_thread = threading.Thread(target=start_nccl_target_server, daemon=True)
        target_server_thread.start()

    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    wpi_pb2_grpc.add_NodeServiceServicer_to_server(NodeService(), server)
    wpi_pb2_grpc.add_IdentityServiceServicer_to_server(IdentityService(), server)
    
    port = "50051"
    server.add_insecure_port('[::]:50051')
    grpc_sock_path = os.path.join(SOCKET_DIR, 'wpi-grpc.sock')
    server.add_insecure_port(f'unix://{grpc_sock_path}')
    server.start()
    logger.info(f"WPI Driver starting on unix://{grpc_sock_path}...")
    try:
        os.chmod(grpc_sock_path, 0o777)
        logger.info(f"Successfully changed permissions of {grpc_sock_path} to 777")
    except Exception as e:
        logger.warning(f"Failed to change permissions of {grpc_sock_path}: {e}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
