include_directories(../common ../smithlab_cpp)

add_executable(amrfinder amrfinder.cpp) 

#Link to GSL, zlib and htslib libraries
target_link_libraries(amrfinder smithlab_cpp libmethpipe ${GSL_LIBRARIES} ${ZLIB_LIBRARIES} ${HTSlib_LIBRARIES})

add_executable(amrfinderOMP amrfinderOMP.cpp) 

#Link to OpenMP, GSL, zlib and htslib libraries
target_link_libraries(amrfinderOMP smithlab_cpp libmethpipe ${GSL_LIBRARIES} ${ZLIB_LIBRARIES} ${HTSlib_LIBRARIES} ${OpenMP_CXX_FLAGS})
target_compile_options(amrfinderOMP PUBLIC ${OpenMP_CXX_FLAGS})

add_executable(PARamrfinder PARamrfinder.cpp) 

#Link to MPI, OpenMP, GSL, zlib and htslib libraries
target_link_libraries(PARamrfinder ${MPI_CXX_LIBRARIES} smithlab_cpp libmethpipe ${GSL_LIBRARIES} ${ZLIB_LIBRARIES} ${HTSlib_LIBRARIES} ${OpenMP_CXX_FLAGS})
target_compile_options(PARamrfinder PUBLIC ${OpenMP_CXX_FLAGS})


# Start testing

add_test(NAME amrfinder_OMP_default_t1 
         COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 1 -c ${CMAKE_SOURCE_DIR}/cmake/tests/chrY.fa -o ${CMAKE_SOURCE_DIR}/cmake/tests/result/chrY.OMP1.amr ${CMAKE_SOURCE_DIR}/cmake/tests/chrY.epiread)

add_test(NAME amrfinder_OMP_default_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CMAKE_SOURCE_DIR}/cmake/tests/result/chrY.OMP1.amr ${CMAKE_SOURCE_DIR}/cmake/tests/refs/chrY.amr)