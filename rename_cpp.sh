#!/bin/bash
find . -name "*.cpp" -exec rename -v 's/\.cpp$/\.cu/i' {} \;