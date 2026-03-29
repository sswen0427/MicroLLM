#!/bin/bash -ex

find include/ -iname '*.h' -print0 -o -iname '*.cpp' -print0 | xargs -0 clang-format -i --style=Google
find src/ -iname '*.h' -print0 -o -iname '*.cpp' -print0 | xargs -0 clang-format -i --style=Google
clang-format -i --style=Google main.cpp