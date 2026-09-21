#!/bin/bash
set -e

# Navigate to module3 and run the executable with the required arguments
cd "$(dirname "$0")/../module3"

echo -e "\n[Run 1] Total Threads: 512 | Threads Per Block: 256"
./assignment.exe 512 256

echo -e "\n[Run 2] Total Threads: 1024 | Threads Per Block: 128"
./assignment.exe 1024 128

echo -e "\n[Run 3] Total Threads: 2048 | Threads Per Block: 64"
./assignment.exe 2048 64