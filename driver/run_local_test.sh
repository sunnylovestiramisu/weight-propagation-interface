#!/bin/bash
set -e

# Setup local sockets directory
export WPI_SOCKET_DIR="/tmp/wpi_sockets"
rm -rf "$WPI_SOCKET_DIR"
mkdir -p "$WPI_SOCKET_DIR"

export WPI_BACKEND="tpu"

echo "Starting local WPI Driver in TPU mode..."
../wpi_env/bin/python driver/main.py > driver.log 2>&1 &
DRIVER_PID=$!

# Ensure the driver is killed on exit, even if the test fails
cleanup() {
    echo "Stopping WPI Driver (PID: $DRIVER_PID)..."
    kill $DRIVER_PID || true
    rm -rf "$WPI_SOCKET_DIR"
    echo "Cleaned up test sockets."
}
trap cleanup EXIT

echo "Waiting for WPI Driver to start..."
sleep 2

# Check if driver is running
if ! kill -0 $DRIVER_PID >/dev/null 2>&1; then
    echo "ERROR: WPI Driver failed to start. Logs:"
    cat driver.log
    exit 1
fi

echo "Running TPU local weight propagation test..."
../wpi_env/bin/python driver/test_tpu_local.py

echo "Done!"
