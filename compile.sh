#!/usr/bin/env bash
cd "/home/dar9586/Unisa/MNI/GPU" || exit
find ./src -name "*.cpp" -printf "Compiling %p\n" -exec /home/dar9586/.local/bin/hipifycc {} \;
find ./src -name "*.c" -printf "Compiling %p\n" -exec /home/dar9586/.local/bin/hipifycc {} \;
find ./src -name "*.cu" -printf "Compiling %p\n" -exec /home/dar9586/.local/bin/hipifycc {} \;
find ./src -name "*.out" -not -path "./out/*" -exec mv -t ./out {} \+
find ./src -name "*.hip" -not -path "./hip/*" -exec mv -t ./hip {} \+
