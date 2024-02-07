# PARamrfinder
PARamrfinder is a high efficient, parallel application to predict allele-specific DNA methylation (ASM) in mammals in the absence of SNP data. It is based on the work of Fang F, Hodges E, Molaro A, Dean MD, Hannon GJ and Smith AD, [amrfinder](http://smithlabresearch.org/software/methpipe/), and it has been fully optimized to reduce its runtime from days to less than 15 minutes.

Prerequisites
-------------

Before starting the installation, please confirm that the following software is available
in your system. Particular versions using during development are shown.

 - **GCC** v8.3.0
 - **GSL** v2.6
 - **Zlib** v1.2.11
 - **HTSlib** v1.13
 - **CMake** v3.28.1
 - **MPI** compiler with support for OpenMP:
     - **OpenMPI** v3.1.4
 - **Git** (Optional)

Different versions of the software may work but they have not been tested.

Compilation
-----------

If the requirements are met, Fiuncho can be configured and built with the following commands:

```sh
mkdir build
cd build
cmake ..
make
```

Optionally, you can test the correctness of the gererated program with the command:

```sh
make test
```
 
Execution
---------

PARamrfinder must be executed with MPI execution commands (```mpiexec``` (standard) or ```mpirun```).Therefore, the tool can be executed using the following command from the root folder of the proyect:

``` sh
mpiexec -n <numProcs> ./bin/PARamrfinder [OPTIONS] <epireads>
```

when ```numProcs``` is the number of MPI processes to execute, and ```options``` is a list of the following arguments:

 - **-o, -output**                    output file 
 - **-c, -chrom**                     genome sequence file/directory (REQUIRED)
 - **-i, -itr**                       max iterations 
 - **-w, -window**                    size of sliding window 
 - **-m, -min-cov**                   min coverage per cpg to test windows 
 - **-g, -gap**                       min allowed gap between amrs (in bp) 
 - **-C, -crit**                      critical p-value cutoff (default: 0.01) 
 - **-t, -threads**                   OpenMP threads on each process 
 - **-f, -nofdr**                     omits FDR multiple testing correction 
 - **-h, -pvals**                     adjusts p-values using Hochberg step-up 
 - **-b, -bic**                       use BIC to compare models 
 - **-v, -verbose**                   print more run info 
 - **-P, -progress**                  print progress info 
 - **-a, -assert_individual_chroms**  asserts that each chrom comes in a separate file 


License
-------

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
