#!/bin/bash
find . -name "*.cu" -exec rename -v 's/\.cu$/\.cpp/i' {} \;