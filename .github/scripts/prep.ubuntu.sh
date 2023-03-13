#!/usr/bin/env bash

# Install dependencies
sudo apt-get update -qq 1>/dev/null
sudo apt-get upgrade -qq 1>/dev/null
sudo apt-get install -Y gcc-10 g++-10 cmake libomp-dev openmpi-bin libopenmpi3 libopenmpi-dev -qq 1>/dev/null

# Fix openmpi complaining about not having enough slots
OPENMPI_HOSTFILE=/etc/openmpi/openmpi-default-hostfile

if [ -z "$(cat ${OPENMPI_HOSTFILE} | grep -v -E '^#')" ]; then\
    echo \"localhost slots=32\" | sudo tee -a ${OPENMPI_HOSTFILE};\
fi