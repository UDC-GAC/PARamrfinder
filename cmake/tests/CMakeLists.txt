# Start testing

## Test OMP version

### Test default parameters

#### Single thread
add_test(NAME amrfinder_OMP_default_t1 
         COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 1 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_default_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.amr ${CTEST_BASE_DIRECTORY}/refs/chrY.amr)

#### Multiple threads
add_test(NAME amrfinder_OMP_default_t4 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 4 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)
   
add_test(NAME amrfinder_OMP_default_t4_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.amr ${CTEST_BASE_DIRECTORY}/refs/chrY.amr)



### Test -b parameter (BIC to compare the models)

#### Single thread
add_test(NAME amrfinder_OMP_b_t1 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 1 -b -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.b.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_b_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.b.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_bic.amr)

#### Multiple threads
add_test(NAME amrfinder_OMP_b_t4 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 4 -b -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.b.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_b_t4_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.b.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_bic.amr)



### Test -i parameter (max iterations to fit the models)

#### Single thread
add_test(NAME amrfinder_OMP_i_t1 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 1 -i 100 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.i.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_i_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.i.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_i100.amr)

#### Multiple threads
add_test(NAME amrfinder_OMP_i_t4 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 4 -i 100 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.i.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_i_t4_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.i.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_i100.amr)



### Test -w parameter (size of the sliding window)

#### Single thread
add_test(NAME amrfinder_OMP_w_t1 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 1 -w 15 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.w.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_w_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.w.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_w15.amr)

#### Multiple threads
add_test(NAME amrfinder_OMP_w_t4 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 4 -w 15 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.w.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_w_t4_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.w.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_w15.amr)



### Test -f parameter (omits FDR multiple testing correction)

#### Single thread
add_test(NAME amrfinder_OMP_f_t1 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 1 -f -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.f.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_f_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.f.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_f.amr)

#### Multiple threads
add_test(NAME amrfinder_OMP_f_t4 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 4 -f -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.f.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_f_t4_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.f.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_f.amr)



### Test -h parameter (adjusts p-values using Hochberg step-up)

#### Single thread
add_test(NAME amrfinder_OMP_h_t1 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 1 -h -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.h.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_h_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.h.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_h.amr)

#### Multiple threads
add_test(NAME amrfinder_OMP_h_t4 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 4 -h -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.h.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_h_t4_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.h.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_h.amr)



### Test -m parameter (min coverage per cpg to test windows)

#### Single thread
add_test(NAME amrfinder_OMP_m_t1 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 1 -m 100 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.m.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_m_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.m.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_m100.amr)

#### Multiple threads
add_test(NAME amrfinder_OMP_m_t4 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 4 -m 100 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.m.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_m_t4_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.m.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_m100.amr)



### Test -C parameter (critical p-value cutoff)

#### Single thread
add_test(NAME amrfinder_OMP_C_t1 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 1 -C 0.00001 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.C.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_C_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.C.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_C.amr)

#### Multiple threads
add_test(NAME amrfinder_OMP_C_t4 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 4 -C 0.00001 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.C.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_C_t4_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.C.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_C.amr)



### Test -g parameter (min allowed gap between amrs [in bp])

#### Single thread
add_test(NAME amrfinder_OMP_g_t1 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 1 -g 4000 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.g.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_g_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP1.g.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_g.amr)

#### Multiple threads
add_test(NAME amrfinder_OMP_g_t4 
            COMMAND ${CMAKE_SOURCE_DIR}/bin/amrfinderOMP -t 4 -g 4000 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.g.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_OMP_g_t4_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.OMP4.g.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_g.amr)






## Test MPI version

### Test default parameters

#### Single process - single thread
add_test(NAME amrfinder_MPI_default_p1_t1 
         COMMAND mpirun -np 1 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 1 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_default_p1_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.amr ${CTEST_BASE_DIRECTORY}/refs/chrY.amr)

#### Multiple processes - multiple threads
add_test(NAME amrfinder_MPI_default_p2_t2 
            COMMAND mpirun -np 2 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 2 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)
   
add_test(NAME amrfinder_MPI_default_p2_t2_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.amr ${CTEST_BASE_DIRECTORY}/refs/chrY.amr)



### Test -b parameter (BIC to compare the models)

#### Single thread
add_test(NAME amrfinder_MPI_b_p1_t1 
            COMMAND mpirun -np 1 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 1 -b -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.b.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_b_p1_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.b.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_bic.amr)

#### Multiple threads
add_test(NAME amrfinder_MPI_b_p2_t2 
            COMMAND mpirun -np 2 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 2 -b -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.b.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_b_p2_t2_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.b.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_bic.amr)



### Test -i parameter (max iterations to fit the models)

#### Single thread
add_test(NAME amrfinder_MPI_i_p1_t1 
            COMMAND mpirun -np 1 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 1 -i 100 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.i.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_i_p1_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.i.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_i100.amr)

#### Multiple threads
add_test(NAME amrfinder_MPI_i_p2_t2 
            COMMAND mpirun -np 2 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 2 -i 100 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.i.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_i_p2_t2_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.i.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_i100.amr)



### Test -w parameter (size of the sliding window)

#### Single thread
add_test(NAME amrfinder_MPI_w_p1_t1 
            COMMAND mpirun -np 1 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 1 -w 15 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.w.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_w_p1_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.w.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_w15.amr)

#### Multiple threads
add_test(NAME amrfinder_MPI_w_p2_t2 
            COMMAND mpirun -np 2 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 2 -w 15 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.w.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_w_p2_t2_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.w.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_w15.amr)



### Test -f parameter (omits FDR multiple testing correction)

#### Single thread
add_test(NAME amrfinder_MPI_f_p1_t1 
            COMMAND mpirun -np 1 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 1 -f -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.f.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_f_p1_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.f.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_f.amr)

#### Multiple threads
add_test(NAME amrfinder_MPI_f_p2_t2 
            COMMAND mpirun -np 2 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 2 -f -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.f.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_f_p2_t2_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.f.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_f.amr)



### Test -h parameter (adjusts p-values using Hochberg step-up)

#### Single thread
add_test(NAME amrfinder_MPI_h_p1_t1 
            COMMAND mpirun -np 1 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 1 -h -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.h.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_h_p1_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.h.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_h.amr)

#### Multiple threads
add_test(NAME amrfinder_MPI_h_p2_t2 
            COMMAND mpirun -np 2 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 2 -h -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.h.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_h_p2_t2_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.h.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_h.amr)



### Test -m parameter (min coverage per cpg to test windows)

#### Single thread
add_test(NAME amrfinder_MPI_m_p1_t1 
            COMMAND mpirun -np 1 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 1 -m 100 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.m.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_m_p1_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.m.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_m100.amr)

#### Multiple threads
add_test(NAME amrfinder_MPI_m_p2_t2 
            COMMAND mpirun -np 2 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 2 -m 100 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.m.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_m_p2_t2_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.m.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_m100.amr)



### Test -C parameter (critical p-value cutoff)

#### Single thread
add_test(NAME amrfinder_MPI_C_p1_t1 
            COMMAND mpirun -np 1 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 1 -C 0.00001 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.C.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_C_p1_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.C.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_C.amr)

#### Multiple threads
add_test(NAME amrfinder_MPI_C_p2_t2 
            COMMAND mpirun -np 2 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 2 -C 0.00001 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.C.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_C_p2_t2_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.C.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_C.amr)


            
### Test -g parameter (min allowed gap between amrs [in bp])

#### Single thread
add_test(NAME amrfinder_MPI_g_p1_t1 
            COMMAND mpirun -np 1 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 1 -g 4000 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.g.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_g_p1_t1_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI1.g.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_g.amr)

#### Multiple threads
add_test(NAME amrfinder_MPI_g_p2_t2 
            COMMAND mpirun -np 2 ${CMAKE_SOURCE_DIR}/bin/PARamrfinder -t 2 -g 4000 -c ${CTEST_BASE_DIRECTORY}/chrY.fa -o ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.g.amr ${CTEST_BASE_DIRECTORY}/chrY.epiread)

add_test(NAME amrfinder_MPI_g_p2_t2_compare
            COMMAND ${CMAKE_COMMAND} -E compare_files ${CTEST_BASE_DIRECTORY}/result/chrY.MPI2.g.amr ${CTEST_BASE_DIRECTORY}/refs/chrY_g.amr)
