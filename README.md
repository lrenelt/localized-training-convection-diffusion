This repository contains the code to reproduce the results from 'Construction of local reduced spaces for Friedrichs' systems via randomized training' by Christian Engwer, Mario Ohlberger and Lukas Renelt.

The code is based on Dune-PDELab and its dependencies which are all included as submodules. After you have cloned this repository you need to invoke ```git submodule init``` and ```git submodule update``` to fetch the submodules contents. Alternatively you can clone this repository with the option ```--recurse-submodules```.

## Compiling the C++ code
The location of the build-directory is specified in the *.opts*-file, consider changing the path beforehand. Per default, the directory is set to ```~/build_gcc_10```. Please adapt the following commands accordingly if you choose any other directory.

First, call (from the base directory)

    ./dune-common/bin/dunecontrol --opts=default.opts configure

This will initialise the build directory for all modules. Afterwards, you can build them via

    ./dune-common/bin/dunecontrol --opts=default.opts --module=dune-ultraweak all

## Setup of the python environment

The tests are implemented in python using python-bindings for the DUNE code.
You can activate the python environment via

    source ~/build_gcc_10/dune-python-env/bin/activate

Now, change to folder with the python scripts (```cd ~/build_gcc_10/dune-ultraweak/python```) and run the shell script ```pyenv.sh``` i.e.

    chmod +x ./pyenv.sh
    source ./pyenv.sh

This sets the correct ```PYTHONPATH``` environmental variable.

## Reproduction of the test results

First, change to the subfolder 'convectionDiffusion'

    cd ~/build_gcc_10/dune-ultraweak/python/convectionDiffusion

To compute the solutions in Figure 6.4 and Figure 6.6, run

    python3 computeLocalExampleSolutions.py

The resulting .vtu-files can for example be visualized with the software *paraview*.

To compute the data for Figure 6.1, Figure 6.5 and Figure 6.7, run

    python3 runAllRangefinderTests.py

The results are stored as .csv-data. For a preliminary visualization via matplotlib run

    python3 visualizeCSVdata.py <<name of the csv file>>




