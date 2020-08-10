#!/bin/bash
wget http://www.cs.uoregon.edu/research/paracomp/tau/tauprofile/dist/tau_latest.tar.gz
tar -xvzf tau_latest.tar.gz
cd tau-2.29
# configure TAU
./configure -papi=/usr/share/papi/ -pthread -prefix=~/lib/tau/2.29
# build
make -j install
# set our path to include the new TAU installation
export PATH=$PATH:~/lib/tau/2.29/x86_64/bin
