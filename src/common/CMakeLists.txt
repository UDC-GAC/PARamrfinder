include_directories(../smithlab_cpp)

add_library(libmethpipe STATIC
  BetaBin.cpp
  Distro.cpp
  EmissionDistribution.cpp
  Epiread.cpp
  EpireadStats.cpp
  LevelsCounter.cpp
  MethpipeSite.cpp
  Smoothing.cpp
  ThreeStateHMM.cpp
  TwoStateHMM.cpp
  TwoStateHMM_PMD.cpp
  bsutils.cpp
  numerical_utils.cpp
  ProgramSection.cpp)
