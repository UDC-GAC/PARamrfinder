#!/usr/bin/env bash

# Install dependencies
sudo apt-get -o Acquire::ForceIPv4=true update -qq 1>/dev/null
sudo apt-get -o Acquire::ForceIPv4=true upgrade -qq 1>/dev/null
sudo apt-get -o Acquire::ForceIPv4=true install \
    gcc-10 g++-10 cmake libomp-dev openmpi-bin libopenmpi3 libopenmpi-dev -qq 1>/dev/null

# Fix openmpi complaining about not having enough slots
OPENMPI_HOSTFILE=/etc/openmpi/openmpi-default-hostfile

if [ -z "$(cat ${OPENMPI_HOSTFILE} | grep -v -E '^#')" ]; then\
    echo \"localhost slots=32\" | sudo tee -a ${OPENMPI_HOSTFILE};\
fi

# Install HTSlib
wget https://github.com/samtools/htslib/releases/download/1.17/htslib-1.17.tar.bz2 -O htslib.tar.bz2 1>/dev/null
tar -xjvf htslib.tar.bz2 1>/dev/null
cd htslib-1.17 
./configure 1>/dev/null
make 1>/dev/null
sudo make install 1>/dev/null
