/*    amrfinder: program for resolving epialleles in a sliding window
 *    along a chromosome.
 *
 *    Copyright (C) 2014-2017 University of Southern California and
 *                            Andrew D. Smith and Benjamin E. Decato
 *
 *    Authors: Fang Fang and Benjamin E. Decato and Andrew D. Smith
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <string>
#include <vector>
#include <set>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <cstring>


#include "GenomicRegion.hpp"
#include "OptionParser.hpp"
#include "smithlab_utils.hpp"
#include "smithlab_os.hpp"
#include "EpireadStats.hpp"
#include "GenomicRegion.hpp"

#include <sys/stat.h>
#include "ProgramSection.hpp"

// OpenMP added
#include <omp.h>

// MPI added
#include <mpi.h>

using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;
using std::unordered_map;
using std::runtime_error;


//DEBUG for time measure
//#define DEBUG 1

//#define TIMEPROFILE
#ifdef TIMEPROFILE
using profile_clock_t = std::chrono::high_resolution_clock;
using profile_duration_t = std::chrono::duration<double>;
#endif
//#define PURE_BLOCK_DISTRIBUTION

#ifndef PURE_BLOCK_DISTRIBUTION
#define COST_DISTRIBUTION
#endif

#define MPI_ROOT_PROCESS 0

#define INVALID_CHROM_NAME_ID 0

#define EPIREAD_LENGTH 512

#define MIN_BLOCKS_PER_PROC 100

#define OVERLAPPING_FRACTION 0.0025

#define BATCH_SIZE_CONSTANT 180000
#define SMALL_BATCH_SIZE 100

#ifdef DEBUG
ProgramSection program("Full Program");
#endif

struct ChromInfo
{
  size_t chrom_name_id;
  size_t file_idx;
  size_t chrom_start_offset;
  size_t lines;

  ChromInfo() : chrom_name_id(INVALID_CHROM_NAME_ID), 
                file_idx(0), 
                chrom_start_offset(0), 
                lines(0) {}

  ChromInfo(size_t c_name_id) : chrom_name_id(c_name_id),
                                file_idx(0), 
                                chrom_start_offset(0), 
                                lines(0)
      {}

  ChromInfo(size_t c_name_id, size_t f_idx, size_t start) : chrom_name_id(c_name_id),
                              file_idx(f_idx), 
                              chrom_start_offset(start), 
                              lines(0) {}

  bool
  ocupies_whole_file() const {
    return (chrom_start_offset == 0 && lines == 0);
  }
                      
};

struct WindowMetadata {
  size_t chrom_name_id;
  size_t start_byte;
  size_t end_byte;
  size_t window_start;
  size_t cost;
  

  WindowMetadata() : chrom_name_id(INVALID_CHROM_NAME_ID), start_byte(0), 
                  end_byte(0), window_start(0), cost(0) {}
  WindowMetadata(const size_t c_id, const size_t s, const size_t e, const size_t ws) : 
                  chrom_name_id(c_id), start_byte(s), end_byte(e), window_start(ws), cost(0) {}
  WindowMetadata(const size_t c_id, const size_t s, const size_t e, const size_t c, const size_t ws) : 
                  chrom_name_id(c_id), start_byte(s), end_byte(e), window_start(ws), cost(c) {}

  bool
  operator==(const WindowMetadata& wm) const {
    return ( start_byte == wm.start_byte && end_byte == wm.end_byte && 
            window_start == wm.window_start && cost == wm.cost);
  }

};

struct EpireadMetadata
{
  size_t start_byte;
  size_t end_byte;

  EpireadMetadata() : start_byte(0), end_byte(0) {}
  EpireadMetadata(const size_t s, const size_t e) : 
            start_byte(s), end_byte(e) {}
};

struct BasicAmr {
  size_t chrom_name_id;
  size_t start_cpg;
  size_t end_cpg;
  size_t reads_size;
  double score;

  BasicAmr() : 
            chrom_name_id(INVALID_CHROM_NAME_ID), start_cpg(0), 
            end_cpg(0), reads_size(0), score(0.0) {}
  BasicAmr(size_t c_id, size_t sta, size_t e,
          size_t rs, double sc) : 
            chrom_name_id(c_id), start_cpg(sta), end_cpg(e), 
            reads_size(rs), score(sc) {}
  
  bool
  operator==(const BasicAmr& rhs) const {
    return ( chrom_name_id == rhs.chrom_name_id && start_cpg == rhs.start_cpg && 
            end_cpg == rhs.end_cpg && reads_size == rhs.reads_size && score == rhs.score);
  }

};

static epiread
read_raw_epiread(const char *raw_epireads, const size_t position, const size_t raw_length);

static void
init_mpi(int &argc, char ** &argv, int &rank, int &number_of_processes, 
        MPI_Datatype *MPI_WINDOW_DATATYPE, MPI_Datatype *MPI_AMR_DATATYPE, 
        MPI_Datatype *MPI_CHINFO_DATATYPE);

static void
distribute_windows(const vector<WindowMetadata> &windows,
              const size_t total_cost, const int number_of_processes,
              int windows_per_process[], int displs[],
              int &my_number_of_windows, vector<WindowMetadata> &my_windows,
              MPI_Datatype &MPI_WINDOW_DATATYPE, const int rank);

// Return estimated total cost
static size_t 
preprocess_epireads(const bool VERBOSE, const bool PROGRESS,
                  const size_t min_obs_per_cpg, const size_t window_size,
                  std::ifstream &in, vector<WindowMetadata> &windows);

// Based on get_current_epireads(...)
static WindowMetadata
get_current_epireads_with_metadata(const vector<epiread> &epireads,
                     const vector<EpireadMetadata> &epireads_metadata,
                     const size_t max_epiread_len,
                     const size_t cpg_window, const size_t start_pos,
                     size_t &read_id, vector<epiread> &current_epireads);

// Based on process_chrom(...)
//
// INPUT  : 
// -         * epireads            ->   A vector of epireads representing one chromosome
// -         * epireads_metadata   ->   A vector of epireads representing one chromosome
//
// OUTPUT : 
// -         * windows             ->   A vector with data of the windows that will be tested 
// -         * total_cost          ->   The estimated cost of testing all the windows from current chromosome
static size_t
preprocess_chrom(const bool VERBOSE, const bool PROGRESS,
              const size_t min_obs_per_cpg, const size_t window_size,
              const string &chrom_name, const vector<epiread> &epireads,
              const vector<EpireadMetadata> &epireads_metadata, 
              vector<WindowMetadata> &windows);

//
// INPUT  : 
// -         * my_windows   ->   A vector of windows with the data to get the corresponding epireads
// -         * reads_file   ->   The path to the input file containing the epireads
//
// OUTPUT : 
// -         * my_amrs      ->   A vector of AMRs
static void
process_windows(vector<BasicAmr> &my_amrs, 
              const vector<WindowMetadata> &my_windows,
              const size_t my_number_of_windows,
              const string reads_file, 
              const size_t window_size,
              const EpireadStats epistat);

static void 
gather_basic_amrs(const vector<BasicAmr> &my_amrs,
                const int my_number_of_amrs,
                const MPI_Datatype MPI_AMR_DATATYPE,
                const int recvcounts[],
                const int displs[],
                vector<BasicAmr> &amrs);

static void
from_basic_amrs_to_genomic_regions(const vector<BasicAmr> &basic_amrs, 
                                  vector<GenomicRegion> &amrs,
                                  const unordered_map<size_t, string> &chrom_id_to_name,
                                  std::set<string> &short_chroms);

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
//////////////
//////////////  FUNCTIONS FOR PARALLEL INPUT 
//////////////  AND PARALLEL PREPROCESSING
//////////////  MAIN FUNCTION IS preprocess_epireads_parallel(...)
//////////////                              

// [Parallel] Obtains data for data distribution and parallel processing
//
// INPUT  : 
// -         * epireads_file         ->   Path to the file containing the input epiread data
// -         * MPI_WINDOW_METADATA   ->   MPI datatype to transfer WindowMetadata
// -         * rank                  ->   Identifier of the process
// -         * number_of_processes   ->   Total amount of processing units involved
//
// OUTPUT : 
// -         * main_windows          ->   A vector with data of the windows that will be tested 
// -         * total_cost            ->   The estimated cost of testing all the windows
// -         * number_of_windows     ->   The number of windows to be tested
static void
preprocess_epireads_parallel(const bool VERBOSE, const bool PROGRESS,
                    const size_t min_obs_per_cpg, const size_t window_size,
                    const string epireads_file, vector<WindowMetadata> &main_windows,
                    size_t &total_cost, MPI_Datatype &MPI_WINDOW_METADATA,
                    const int rank, const int number_of_processes, size_t &number_of_windows,
                    unordered_map<string, size_t> &chrom_name_to_id,
                    unordered_map<size_t, string> &chrom_id_to_name,
                    std::set<string> &short_chroms);

// [Parallel] Reads the contents of a file to memory
// 
// INPUT  : 
// -        * epireads_file                   ->   Path to the file to read
// -        * rank                            ->   Identifier of the process
// -        * number_of_processes             ->   Total amount of processing units involved
//
// OUTPUT : 
// -        * chunk                           ->   Pointer where contents will be allocated
// -        * process_start                   ->   Offset were the process will start to read
// -        * process_size                    ->   Size of the file ''owned'' by the process
// -        * process_size_with_overlapping   ->   Size of the file read by the process
static void 
readEpireadFile(string epireads_file, int rank, int number_of_processes, 
                char* &chunk, size_t &process_start, 
                size_t &process_size, size_t &process_size_with_overlapping);

// [Concurrent] Process raw data (chars) to epireads
//
// INPUT  :
// -        * chunk           ->   Pointer to the raw data
// -        * rank            ->   Identifier of the process
// -        * chunk_size      ->   Size of the data
// -        * process_start   ->   Offset were the process started to read the file
//
// OUTPUT :
// -        * epireads                   ->   Offset were the process will start to read
// -        * em                   ->   Offset were the process will start to read
static void
from_raw_to_epireads(char* chunk, int rank,
                      const size_t chunk_size, const size_t process_start,
                      vector<epiread> &epireads, vector<EpireadMetadata> &em);

// [Concurrrent] preprocess chrom
static size_t
preprocess_chrom_parallel(const bool VERBOSE, const bool PROGRESS,
              const size_t min_obs_per_cpg, const size_t window_size,
              const string &chrom_name, const size_t chrom_cpgs,
              const size_t chrom_start_index, const size_t chrom_end_index, 
              const size_t max_epiread_len, const vector<epiread> &epireads,
              const vector<EpireadMetadata> &epireads_metadata, 
              vector<WindowMetadata> &windows,
              const size_t current_chrom_name_index);

// [Sequential-to-concurrent] Wrapper over preprocess_chrom_parallel to preprocess all chroms
static size_t
epireads_to_windows(const bool VERBOSE, const bool PROGRESS,
                    const vector<epiread> &epireads, 
                    const vector<EpireadMetadata> &em,
                    vector<WindowMetadata> &windows,
                    const size_t min_obs_per_cpg, 
                    const size_t window_size,
                    size_t &max_epiread_len,
                    MPI_Win &index_window,
                    vector<std::pair<std::pair<string, int>, size_t>> &chrom_names_with_ids);

// [Sequential] Based on get_current_epireads_with_metadata
static WindowMetadata
get_current_epireads_with_metadata_parallel(const vector<epiread> &epireads,
                     const vector<EpireadMetadata> &epireads_metadata,
                     const size_t max_epiread_len, const size_t chrom_end_index,
                     const size_t cpg_window, const size_t start_pos,
                     size_t &read_id, vector<epiread> &current_epireads,
                     const size_t current_chrom_name_index);

// [Auxiliar inline]
static size_t 
find_last_owned_byte(size_t start, size_t chunk_size, char* chunk);

// Returns the real size of 'windows'
// So windows in the range [0-i) are okey and windows [i, size()) are trash
// That is, only the first 'i' windows are worth
// This function should not be called on the last process
// Instead windows.size() should be called
static size_t
filter_leftover_windows(const string &epireads_file,
                        vector<epiread> &epireads,
                        vector<EpireadMetadata> &epireads_metadata,
                        vector<WindowMetadata> &windows, 
                        const size_t last_owned_byte, size_t &total_cost,
                        const size_t min_obs_per_cpg,
                        const size_t window_size,
                        size_t &last_chrom_max_epiread_len,
                        const unordered_map<string, size_t> &chrom_name_to_id);

// [Parallel communication] Point-to-Point
static WindowMetadata
communicate_last_valid_window(const vector<WindowMetadata> &windows,
                              const size_t last_valid_window_index,
                              const MPI_Datatype &MPI_WINDOW_METADATA,
                              const int rank, const int number_of_processes);


// Processes leading windows until the don't overlap with previous process
// This 'might' cause the current process to have 0 valid windows
// Substract the cost of the invalid windows
static size_t
remove_leading_windows(const vector<WindowMetadata> &windows,
                        const vector<epiread> &epireads,
                        const vector<EpireadMetadata> &em,
                        const WindowMetadata &validation_window,
                        const size_t true_windows_size,
                        size_t &total_cost);

static size_t 
gather_windows(const vector<WindowMetadata> &windows,
              const int my_number_of_windows, const size_t first_window,
              const size_t total_cost, const int number_of_processes,
              size_t &full_total_cost, vector<WindowMetadata> &all_windows,
              MPI_Datatype &MPI_WINDOW_METADATA, const int rank);

static void
convert_coordinates_wrapper(const int rank, const int number_of_processes, MPI_Datatype MPI_AMR_DATATYPE,
                    const bool VERBOSE, const string chroms_dir,
                    const string fasta_suffix, vector<GenomicRegion> &amrs,
                    unordered_map<string, size_t> &chrom_name_to_id,
                    unordered_map<size_t, string> &chrom_id_to_name,
                    std::set<string> &short_chroms);

static void
convert_coordinates_parallel(const int rank, const int number_of_processes, 
                    MPI_Datatype MPI_AMR_DATATYPE, MPI_Datatype MPI_CHINFO_DATATYPE,
                    const bool VERBOSE, const string chroms_dir,
                    const string fasta_suffix, vector<GenomicRegion> &amrs,
                    unordered_map<string, size_t> &chrom_name_to_id,
                    unordered_map<size_t, string> &chrom_id_to_name,
                    std::set<string> &short_chroms);

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
//////////////
//////////////  CODE BELOW HERE IS FOR FILTERING THE AMR WINDOWS AND
//////////////  MERGING THEM TO OBTAIN FINAL AMRS
//////////////

static void
eliminate_amrs_by_cutoff(const double cutoff, vector<GenomicRegion> &amrs) {
  size_t j = 0;
  for (size_t i = 0; i < amrs.size(); ++i)
    if (amrs[i].get_score() < cutoff) {
      amrs[j] = amrs[i];
      ++j;
    }
  amrs.erase(amrs.begin() + j, amrs.end());
}


static void
eliminate_amrs_by_size(const size_t min_size, vector<GenomicRegion> &amrs) {
  size_t j = 0;
  for (size_t i = 0; i < amrs.size(); ++i)
    if (amrs[i].get_width() >= min_size) {
      amrs[j] = amrs[i];
      ++j;
    }
  amrs.erase(amrs.begin() + j, amrs.end());
}


static void
collapse_amrs(vector<GenomicRegion> &amrs) {
  size_t j = 0;
  for (size_t i = 1; i < amrs.size(); ++i)
    if (amrs[j].same_chrom(amrs[i]) &&
        // The +1 below is because intervals in terms of CpGs are
        // inclusive
        amrs[j].get_end() + 1 >= amrs[i].get_start()) {
      amrs[j].set_end(amrs[i].get_end());
      amrs[j].set_score(std::min(amrs[i].get_score(), amrs[j].get_score()));
    }
    else {
      ++j;
      amrs[j] = amrs[i];
    }
  ++j;
  amrs.erase(amrs.begin() + j, amrs.end());
}


static void
merge_amrs(const size_t gap_limit, vector<GenomicRegion> &amrs) {
  size_t j = 0;
  for (size_t i = 1; i < amrs.size(); ++i)
    // check distance between two amrs is greater than gap limit
    if (amrs[j].same_chrom(amrs[i]) &&
        amrs[j].get_end() + gap_limit>= amrs[i].get_start()) {
      amrs[j].set_end(amrs[i].get_end());
      amrs[j].set_score(std::min(amrs[i].get_score(), amrs[j].get_score()));
    }
    else {
      ++j;
      amrs[j] = amrs[i];
    }
  ++j;
  amrs.erase(amrs.begin() + j, amrs.end());
}


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////
/////   CODE FOR RANDOMIZING THE READS TO GET EXPECTED NUMBER OF
/////   IDENTIFIED AMRS
/////

// static void
// set_read_states(vector<vector<char> > &state_counts, vector<epiread> &reads) {
//   for (size_t i = 0; i < reads.size(); ++i) {
//     const size_t offset = reads[i].pos;
//     for (size_t j = 0; j < reads[i].length(); ++j) {
//       reads[i].seq[j] = state_counts[offset + j].back();
//       state_counts[offset + j].pop_back();
//     }
//   }
// }

// static void
// get_state_counts(const vector<epiread> &reads, const size_t total_cpgs,
//               vector<vector<char> > &state_counts) {

//   state_counts = vector<vector<char> >(total_cpgs);
//   for (size_t i = 0; i < reads.size(); ++i) {
//     const size_t offset = reads[i].pos;
//     for (size_t j = 0; j < reads[i].length(); ++j)
//       state_counts[offset + j].push_back(reads[i].seq[j]);
//   }
//   for (size_t i = 0; i < state_counts.size(); ++i)
//     random_shuffle(state_counts[i].begin(), state_counts[i].end());
// }

// static void
// randomize_read_states(vector<epiread> &reads) {
//   srand(time(0) + getpid());
//   const size_t total_cpgs = get_n_cpgs(reads);
//   vector<vector<char> > state_counts;
//   get_state_counts(reads, total_cpgs, state_counts);
//   set_read_states(state_counts, reads);
// }


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
//////////////
//////////////  CODE FOR CONVERTING BETWEEN CPG AND BASE PAIR
//////////////  COORDINATES BELOW HERE
//////////////

inline static bool
is_cpg(const string &s, const size_t idx) {
  return toupper(s[idx]) == 'C' && toupper(s[idx + 1]) == 'G';
}


static void
collect_cpgs(const string &s, unordered_map<size_t, size_t> &cpgs) {
  const size_t lim = s.length() - 1;
  size_t cpg_count = 0;
  for (size_t i = 0; i < lim; ++i)
    if (is_cpg(s, i)) {
      cpgs[cpg_count] = i;
      ++cpg_count;
    }
}

static void
collect_cpgs(const string &s, vector<size_t> &cpgs) {
  const size_t lim = s.length() - 1;
  
  cpgs.clear();
  
  vector<vector<size_t>> omp_cpgs;
  int threads;

  #pragma omp parallel
  {
    int my_thread_id = omp_get_thread_num();
    if (my_thread_id == 0){
      threads = omp_get_num_threads();
      omp_cpgs.resize(threads);
    }

    #pragma omp barrier
    
    #pragma omp for schedule(static)
    for (size_t i = 0; i < lim; ++i)
    if (is_cpg(s, i)) {
      omp_cpgs[my_thread_id].push_back(i);
    }
  }

  for(int i = 0; i < threads; i++){
    cpgs.insert(cpgs.end(), omp_cpgs[i].begin(), omp_cpgs[i].end());
  }
}

static void
convert_coordinates(const vector<size_t> &cpgs,
                    GenomicRegion &region)  {
  if (region.get_start() >= cpgs.size() || region.get_end() >= cpgs.size())
    throw runtime_error("could not convert:\n" + region.tostring());
  region.set_start(cpgs[region.get_start()]);
  region.set_end(cpgs[region.get_end()]);
}

static void
convert_coordinates(const unordered_map<size_t, size_t> &cpgs,
                    GenomicRegion &region)  {
  const unordered_map<size_t, size_t>::const_iterator
    start_itr(cpgs.find(region.get_start()));
  const unordered_map<size_t, size_t>::const_iterator
    end_itr(cpgs.find(region.get_end()));
  if (start_itr == cpgs.end() || end_itr == cpgs.end())
    throw runtime_error("could not convert:\n" + region.tostring());
  region.set_start(start_itr->second);
  region.set_end(end_itr->second);
}


static void
get_chrom(const bool VERBOSE, const GenomicRegion &r,
          const unordered_map<string, string>& chrom_files,
      GenomicRegion &chrom_region,  string &chrom) {
  const unordered_map<string, string>::const_iterator
                              fn(chrom_files.find(r.get_chrom()));
  if (fn == chrom_files.end())
    throw runtime_error("could not find chrom: " + r.get_chrom());
  chrom.clear();
  read_fasta_file(fn->second, r.get_chrom(), chrom);
  if (chrom.empty())
    throw runtime_error("could not find chrom: " + r.get_chrom());
  else {
    chrom_region.set_chrom(r.get_chrom());
  }
}

/*static void
convert_coordinates(const bool VERBOSE, const string chroms_dir,
                    const string fasta_suffix, vector<GenomicRegion> &amrs) {

  unordered_map<string, string> chrom_files;
  identify_and_read_chromosomes(chroms_dir, fasta_suffix, chrom_files);
  if (VERBOSE)
    cerr << "CHROMS:\t" << chrom_files.size() << endl;

  unordered_map<size_t, size_t> cpgs;
  string chrom;
  GenomicRegion chrom_region("chr0", 0, 0);
  for (size_t i = 0; i < amrs.size(); ++i) {
    if (!amrs[i].same_chrom(chrom_region)) {
      get_chrom(VERBOSE, amrs[i], chrom_files, chrom_region, chrom);
      collect_cpgs(chrom, cpgs);
      if (VERBOSE)
        cerr << "CONVERTING: " << chrom_region.get_chrom() << endl;
    }
    convert_coordinates(cpgs, amrs[i]);
  }
}*/

static void
convert_coordinates(const bool VERBOSE, const string chroms_dir,
                    const string fasta_suffix, vector<GenomicRegion> &amrs) {

  unordered_map<string, string> chrom_files;
  /************************ TIME MEASURE ***********************/
  #ifdef DEBUG
  program.subsections[5].subsections[0].subsections[0].start_section();
  #endif
  /************************ TIME MEASURE ***********************/
  identify_and_read_chromosomes(chroms_dir, fasta_suffix, chrom_files);
  /************************ TIME MEASURE ***********************/
  #ifdef DEBUG
  program.subsections[5].subsections[0].subsections[0].end_section();
  program.subsections[5].subsections[0].subsections[2].start_section();
  #endif
  /************************ TIME MEASURE ***********************/
  if (VERBOSE)
    cerr << "CHROMS:\t" << chrom_files.size() << endl;

  vector<size_t> cpgs;
  string chrom;
  GenomicRegion chrom_region("chr0", 0, 0);
  for (size_t i = 0; i < amrs.size(); ++i) {
    if (!amrs[i].same_chrom(chrom_region)) {
      /************************ TIME MEASURE ***********************/
      #ifdef DEBUG
      program.subsections[5].subsections[0].subsections[2].end_section();
      program.subsections[5].subsections[0].subsections[2].store_time();
      program.subsections[5].subsections[0].subsections[1].start_section();
      #endif
      /************************ TIME MEASURE ***********************/
      get_chrom(VERBOSE, amrs[i], chrom_files, chrom_region, chrom);
      /************************ TIME MEASURE ***********************/
      #ifdef DEBUG
      program.subsections[5].subsections[0].subsections[1].end_section();
      program.subsections[5].subsections[0].subsections[1].store_time();
      program.subsections[5].subsections[0].subsections[2].start_section();
      #endif
      /************************ TIME MEASURE ***********************/
      collect_cpgs(chrom, cpgs);
      if (VERBOSE)
        cerr << "CONVERTING: " << chrom_region.get_chrom() << endl;
    }
    convert_coordinates(cpgs, amrs[i]);
  }
  /************************ TIME MEASURE ***********************/
  #ifdef DEBUG
  program.subsections[5].subsections[0].subsections[2].end_section();
  program.subsections[5].subsections[0].subsections[2].store_time();
  #endif
  /************************ TIME MEASURE ***********************/
}

/*
static void
move_data_pipelined(const size_t number_of_windows, 
                    MPI_Datatype &MPI_WINDOW_DATATYPE, MPI_Datatype &MPI_AMR_DATATYPE,
                    const string reads_file, const size_t window_size, const EpireadStats &epistat);
*/

static void 
process_windows_dynamically(const int rank,
                            const int number_of_processes,
                            const size_t number_of_windows, 
                            size_t number_of_costly_batches,
                            const size_t BATCH_SIZE,
                            vector<WindowMetadata> &windows, 
                            vector<BasicAmr> &amrs,
                            MPI_Datatype &MPI_WINDOW_DATATYPE, 
                            MPI_Datatype &MPI_AMR_DATATYPE,
                            const string reads_file, 
                            const size_t window_size, 
                            const EpireadStats &epistat);

static void 
swap_windows(vector<WindowMetadata> &windows,
            vector<size_t> &swap_info,
            const size_t total_cost,
            const size_t total_number_of_windows,
            const size_t BATCH_SIZE);

static void 
unswap_basic_amrs(vector<BasicAmr> &basic_amrs,
              vector<size_t> &swap_info,
              const size_t BATCH_SIZE);

static size_t
get_batch_size(const size_t number_of_windows, const size_t total_cost, 
              const int number_of_threads, const int number_of_procs);

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////
/////////  CODE FOR DOING THE SLIDING WINDOW STUFF BELOW HERE
/////////


static void
clip_read(const size_t start_pos, const size_t end_pos, epiread &r) {
  if (r.pos < start_pos) {
    r.seq = r.seq.substr(start_pos - r.pos);
    r.pos = start_pos;
  }
  if (r.end() > end_pos)
    r.seq = r.seq.substr(0, end_pos - r.pos);
}


static void
get_current_epireads(const vector<epiread> &epireads,
                     const size_t max_epiread_len,
                     const size_t cpg_window, const size_t start_pos,
                     size_t &read_id, vector<epiread> &current_epireads) {
  while (read_id < epireads.size() &&
         epireads[read_id].pos + max_epiread_len <=start_pos)
    ++read_id;

  const size_t end_pos = start_pos + cpg_window;
  for (size_t i = read_id; (i < epireads.size() &&
                            epireads[i].pos < end_pos); ++i) {
    if (epireads[i].end() > start_pos) {
      current_epireads.push_back(epireads[i]);
      clip_read(start_pos, end_pos, current_epireads.back());
    }
  }
}


static size_t
total_states(const vector<epiread> &epireads) {
  size_t total = 0;
  for (size_t i = 0; i < epireads.size(); ++i)
    total += epireads[i].length();
  return total;
}


static void
add_amr(const string &chrom_name, const size_t start_cpg,
        const size_t cpg_window, const vector<epiread> &reads,
        const double score, vector<GenomicRegion> &amrs) {
  static const string name_label("AMR");
  const size_t end_cpg = start_cpg + cpg_window - 1;
  const string amr_name(name_label + toa(amrs.size()) + ":" + toa(reads.size()));
  amrs.push_back(GenomicRegion(chrom_name, start_cpg, end_cpg,
                               amr_name, score, '+'));
}

// Version of "add_amr" suitable for OpenMP
//
// Now "amrs" has a fixed size and Regions
// can be inserted based on index.
//
// Also some naming logic has being removed
// from this function and inserted in
// "rename_amrs"
static void
add_amr_openmp(const string &chrom_name, const size_t start_cpg,
        const size_t cpg_window, const vector<epiread> &reads,
        const double score, vector<GenomicRegion> &amrs, 
        const size_t amrs_base_size) {
  //static const string name_label("AMR");
  const size_t end_cpg = start_cpg + cpg_window - 1;
  const string amr_name(/*name_label + toa(amrs.size()) +*/ ":" + toa(reads.size()));
  amrs[amrs_base_size + start_cpg] = GenomicRegion(chrom_name, start_cpg, end_cpg,
                               amr_name, score, '+');
}

// Version of "add_amr" suitable for OpenMP and MPI
//
// Now "amrs" has a fixed size and Regions
// can be inserted based on index.
//
// Also some naming logic has being removed
// from this function and inserted in
// "rename_amrs"
static void
add_amr_ompi(const size_t chrom_name_id, const size_t start_cpg,
        const size_t cpg_window, const vector<epiread> &reads,
        const double score, vector<BasicAmr> &amrs, 
        const size_t amrs_index) {
  //static const string name_label("AMR");
  const size_t end_cpg = start_cpg + cpg_window - 1;
  //const string amr_name(/*name_label + toa(amrs.size()) +*/ ":" + toa(reads.size()));
  amrs[amrs_index] = BasicAmr(chrom_name_id, start_cpg, end_cpg, reads.size(), score);
}


static size_t
process_chrom(const bool VERBOSE, const bool PROGRESS,
              const size_t min_obs_per_cpg, const size_t window_size,
              const EpireadStats &epistat, const string &chrom_name,
              const vector<epiread> &epireads, vector<GenomicRegion> &amrs) {
  size_t max_epiread_len = 0;


  for (size_t i = 0; i < epireads.size(); ++i)
    max_epiread_len = std::max(max_epiread_len, epireads[i].length());


  const size_t min_obs_per_window = window_size*min_obs_per_cpg;

  const size_t chrom_cpgs = get_n_cpgs(epireads);
  if (VERBOSE)
    cerr << "PROCESSING: " << chrom_name << " "
         << "[reads: " << epireads.size() << "] "
         << "[cpgs: " << chrom_cpgs << "]" << endl;
  const size_t PROGRESS_TIMING_MODULUS = std::max(1ul, epireads.size()/1000);
  size_t windows_tested = 0;
  size_t start_idx = 0;
  const size_t lim = chrom_cpgs < window_size ? 1 : chrom_cpgs - window_size + 1;


  /************************ SOLVE OPENMP DEPENDENCY ***********************/
  // Add extra null regions to fill vector
  size_t amrs_size = amrs.size();
  amrs.resize(amrs_size + lim, GenomicRegion());


  // FOR sliding_window_start = first_window_start_pos 
  // TO last_window_start_pos 
  // DO sliding_window_start++
  // Original condition: i < lim && start_idx < epireads.size()
  #pragma omp parallel for reduction(+:windows_tested) firstprivate(start_idx) schedule(dynamic)
  for (size_t i = 0; i < lim; ++i) {

    /* Removed vervose output so there are no problems with cerr */
    /* if (PROGRESS && i % PROGRESS_TIMING_MODULUS == 0)
      cerr << '\r' << chrom_name << ' ' << percent(i, chrom_cpgs) << "%\r"; */

    vector<epiread> current_epireads;

    // Read from 'epireads' the lines in the current window
    get_current_epireads(epireads, max_epiread_len,
                         window_size, i, start_idx, current_epireads);

    // IF is seems worth testing the current window
    // THEN test it
    if (total_states(current_epireads) >= min_obs_per_window) {
      bool is_significant = false;
      const double score = epistat.test_asm(current_epireads, is_significant);
      
      // IF it was worth the test
      // THEN add the window to the list
      if (is_significant)
        add_amr_openmp(chrom_name, i, window_size, current_epireads, score, amrs, amrs_size);
        //add_amr(chrom_name, i, window_size, current_epireads, score, amrs);

      ++windows_tested;
    }

  }

  // Remove null Regions, potentially rename them
  vector<GenomicRegion>::iterator real_end = std::remove(amrs.begin() + amrs_size, amrs.end(), GenomicRegion());
  amrs.erase(real_end, amrs.end());

  if (PROGRESS)
    cerr << '\r' << chrom_name << " 100%" << endl;
  return windows_tested;
}

// Extra logic because of OpenMP
//
// Now "amrs" has a fixed final size and
// Regions can be accessed based on index.
//
// Some naming logic has being removed from
// "add_amr_openmp" function and inserted here
void rename_amrs(vector<GenomicRegion> &amrs){

  //#pragma omp parallel for schedule(static) // Esto no se puede hacer, hay un hash map compartido por debajo
  for (size_t i = 0; i < amrs.size(); i++){
    static const string name_label("AMR");
    const string amr_name(name_label + std::to_string(i) + amrs[i].get_name());
    amrs[i].set_name(amr_name);
  } 
}

/************************ TIME MEASURE ***********************/
#ifdef DEBUG
ProgramSection
define_program_structure() {
  ProgramSection p("Full Program");

  ProgramSection pre("+--- Window preprocessing");
  pre.add_subsection(ProgramSection("|    +--- Input file to epireads"));
  pre.add_subsection(ProgramSection("|    +--- Chromosome preprocessing"));

  ProgramSection post("+--- AMR Post Processing");
  ProgramSection convert("|    +--- Convert Coordinates");
  convert.add_subsection(ProgramSection("|    |    +--- Read Fasta Files #1"));
  convert.add_subsection(ProgramSection("|    |    +--- Read Fasta Files #2"));
  convert.add_subsection(ProgramSection("|    |    +--- Collect cpgs"));
  post.add_subsection(convert);

  p.add_subsection(ProgramSection("+--- Init Phase"));
  p.add_subsection(pre);
  p.add_subsection(ProgramSection("+--- Window Scatter"));
  p.add_subsection(ProgramSection("+--- Window processing"));
  p.add_subsection(ProgramSection("+--- AMR Gather"));
  p.add_subsection(post);
  p.add_subsection(ProgramSection("+--- Output Writing"));
  p.add_subsection(ProgramSection("+--- [EXTRA] add overlapping time"));

  return p;
}
#endif
/************************ TIME MEASURE ***********************/

int
main(int argc, char **argv) {
  try {

    /************************ TIME MEASURE ***********************/
    #ifdef DEBUG
    double local_time, global_time;
    program = define_program_structure();
    //Start measuring Full Program Time
    program.start_section();
    //Start measuring Init Phase Time
    program.subsections[0].start_section();
    #endif
    /************************ TIME MEASURE ***********************/

    int rank, number_of_processes;
    MPI_Datatype MPI_WINDOW_METADATA, MPI_AMR, MPI_CHROM_INFO;

    init_mpi(argc, argv, rank, number_of_processes, &MPI_WINDOW_METADATA, &MPI_AMR, &MPI_CHROM_INFO);

    static const string fasta_suffix = "fa";

    bool VERBOSE = false;
    bool PROGRESS = false;

    string outfile;
    string chroms_dir;

    size_t max_itr = 10;
    size_t window_size = 10;
    size_t gap_limit = 1000;

    double high_prob = 0.75, low_prob = 0.25;
    double min_obs_per_cpg = 4;
    double critical_value = 0.01;

    int num_threads = 1;

    // bool RANDOMIZE_READS = false;
    bool USE_BIC = false;
    bool CORRECTION = false;
    bool NOFDR=false;

    bool INDIVIDUAL_CHROMS=false;

    /****************** COMMAND LINE OPTIONS ********************/
    OptionParser opt_parse(strip_path(argv[0]),
                           "identify regions of allele-specific methylation",
                           "<epireads>");
    opt_parse.add_opt("output", 'o', "output file", false, outfile);
    opt_parse.add_opt("chrom", 'c', "genome sequence file/directory",
                      true, chroms_dir);
    opt_parse.add_opt("itr", 'i', "max iterations", false, max_itr);
    opt_parse.add_opt("window", 'w', "size of sliding window",
                      false, window_size);
    opt_parse.add_opt("min-cov", 'm', "min coverage per cpg to test windows",
                      false, min_obs_per_cpg);
    opt_parse.add_opt("gap", 'g', "min allowed gap between amrs (in bp)",
                      false, gap_limit);
    opt_parse.add_opt("crit", 'C', "critical p-value cutoff (default: 0.01)",
                      false, critical_value);
    opt_parse.add_opt("threads", 't', "OpenMP threads on each process", false, num_threads);
    // BOOLEAN FLAGS
    opt_parse.add_opt("nofdr", 'f', "omits FDR multiple testing correction",
                      false, NOFDR);
    opt_parse.add_opt("pvals", 'h', "adjusts p-values using Hochberg step-up",
                      false, CORRECTION);
    opt_parse.add_opt("bic", 'b', "use BIC to compare models", false, USE_BIC);
    opt_parse.add_opt("verbose", 'v', "print more run info", false, VERBOSE);
    opt_parse.add_opt("progress", 'P', "print progress info", false, PROGRESS);
    opt_parse.add_opt("assert_individual_chroms", 'a', "asserts that each chrom comes in a separate file",
                      false, INDIVIDUAL_CHROMS);
    vector<string> leftover_args;
    opt_parse.parse(argc, const_cast<const char**>(argv), leftover_args);
    if (argc == 1 || opt_parse.help_requested()) {
      cerr << opt_parse.help_message() << endl
           << opt_parse.about_message() << endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_SUCCESS);
      return EXIT_SUCCESS;
    }
    if (opt_parse.about_requested()) {
      cerr << opt_parse.about_message() << endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_SUCCESS);
      return EXIT_SUCCESS;
    }
    if (opt_parse.option_missing()) {
      cerr << opt_parse.option_missing_message() << endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_SUCCESS);
      return EXIT_SUCCESS;
    }
    if (leftover_args.size() != 1) {
      cerr << opt_parse.help_message() << endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_SUCCESS);
      return EXIT_SUCCESS;
    }
    const string reads_file(leftover_args.front());

    omp_set_num_threads(num_threads);

    /************************ TIME MEASURE ***********************/
    #ifdef DEBUG
    //End measuring Init Phase Time
    program.subsections[0].end_section();

    local_time = program.subsections[0].get_elapsed_time();
    MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    program.subsections[0].set_elapsed_time(global_time);
    MPI_Barrier(MPI_COMM_WORLD);

    //Start measuring Preprocessing Phase Time
    program.subsections[1].start_section();
    #endif
    /************************ TIME MEASURE ***********************/

    /****************** END COMMAND LINE OPTIONS *****************/

    if (VERBOSE)
      cerr << "AMR TESTING OPTIONS: "
           << "[test=" << (USE_BIC ? "BIC" : "LRT") << "] "
           << "[iterations=" << max_itr << "]" << endl;

    const EpireadStats epistat(low_prob, high_prob, critical_value, max_itr,
                                USE_BIC);

    /*std::ifstream in(reads_file.c_str());
    if (!in)
      throw runtime_error("cannot open input file: " + reads_file);*/

    vector<GenomicRegion> amrs;
    vector<BasicAmr> basic_amrs;
    vector<BasicAmr> my_amrs;
    size_t windows_tested = 0, total_cost = 0;
    vector<WindowMetadata> windows;
    vector<WindowMetadata> my_windows;
    int windows_per_process[number_of_processes];
    int displs[number_of_processes];
    int my_number_of_windows;
    size_t total_number_of_windows;
    vector<size_t> swap_info;
    unordered_map<string, size_t> chrom_name_to_id;
    unordered_map<size_t, string> chrom_id_to_name;
    std::set<string> short_chroms;

    preprocess_epireads_parallel(VERBOSE, PROGRESS,
                    min_obs_per_cpg, window_size,
                    reads_file, windows,
                    total_cost, MPI_WINDOW_METADATA,
                    rank, number_of_processes, 
                    total_number_of_windows,
                    chrom_name_to_id,
                    chrom_id_to_name,
                    short_chroms);   

    if(rank == MPI_ROOT_PROCESS){
      /*total_cost = preprocess_epireads(VERBOSE, PROGRESS, min_obs_per_cpg, 
                                  window_size, in, windows);*/

      windows_tested = windows.size();
      basic_amrs.resize(windows_tested);
    }

    /************************ TIME MEASURE ***********************/
    #ifdef DEBUG
    //End measuring Preprocessing Phase Time
    program.subsections[1].end_section();

    local_time = program.subsections[1].get_elapsed_time();
    MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    program.subsections[1].set_elapsed_time(global_time);
    MPI_Barrier(MPI_COMM_WORLD);

    //Start measuring Scatter Phase Time
    program.subsections[2].start_section();
    #endif
    /************************ TIME MEASURE ***********************/

    const size_t BATCH_SIZE = get_batch_size(total_number_of_windows, total_cost, 
                                              num_threads, number_of_processes);

    if(rank == MPI_ROOT_PROCESS){
      swap_windows(windows, swap_info, total_cost, total_number_of_windows, BATCH_SIZE);
    }

    /*
    distribute_windows(windows, total_cost, number_of_processes,
                        windows_per_process, displs,
                        my_number_of_windows, my_windows,
                        MPI_WINDOW_METADATA, rank);
    */

    /************************ TIME MEASURE ***********************/
    #ifdef DEBUG
    //End measuring Scatter Phase Time
    program.subsections[2].end_section();

    local_time = program.subsections[2].get_elapsed_time();
    MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    program.subsections[2].set_elapsed_time(global_time);
    MPI_Barrier(MPI_COMM_WORLD);

    //Start measuring Processing Phase Time
    program.subsections[3].start_section();
    #endif
    /************************ TIME MEASURE ***********************/

    /*
    process_windows(my_amrs, my_windows, my_number_of_windows,
              reads_file, window_size, epistat);
    */

    process_windows_dynamically(rank, number_of_processes, total_number_of_windows, swap_info.size(), 
                              BATCH_SIZE, windows, basic_amrs, MPI_WINDOW_METADATA, 
                              MPI_AMR, reads_file, window_size, epistat);    

    /*                    
    move_data_pipelined(total_number_of_windows, MPI_WINDOW_METADATA, MPI_AMR,
                        reads_file, window_size, epistat);
    */

    /************************ TIME MEASURE ***********************/
    #ifdef DEBUG
    //End measuring Processing Phase Time
    program.subsections[3].end_section();

    local_time = program.subsections[3].get_elapsed_time();
    MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    program.subsections[3].set_elapsed_time(global_time);
    MPI_Barrier(MPI_COMM_WORLD);

    double local_times_processing[number_of_processes];
    MPI_Gather(&local_time, 1, MPI_DOUBLE, local_times_processing, 1, MPI_DOUBLE, MPI_ROOT_PROCESS, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    //Start measuring Gather Phase Time
    program.subsections[4].start_section();
    #endif
    /************************ TIME MEASURE ***********************/

    if(rank == MPI_ROOT_PROCESS){
      unswap_basic_amrs(basic_amrs, swap_info, BATCH_SIZE);
    }

    /*
    gather_basic_amrs(my_amrs, my_number_of_windows, MPI_AMR,
                windows_per_process, displs, basic_amrs);
    */

    /************************ TIME MEASURE ***********************/
    #ifdef DEBUG
    //End measuring Gather Phase Time
    program.subsections[4].end_section();

    local_time = program.subsections[4].get_elapsed_time();
    MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    program.subsections[4].set_elapsed_time(global_time);
    MPI_Barrier(MPI_COMM_WORLD);

    //Start measuring Postprocessing Phase Time
    program.subsections[5].start_section();
    #endif
    /************************ TIME MEASURE ***********************/

    // Remove null Regions
    vector<BasicAmr>::iterator real_end = std::remove(basic_amrs.begin(), basic_amrs.end(), BasicAmr());
    basic_amrs.erase(real_end, basic_amrs.end());

    //amrs.resize(basic_amrs.size());
    from_basic_amrs_to_genomic_regions(basic_amrs, amrs, chrom_id_to_name, short_chroms);

    //rename_amrs(amrs); // Alreadyincluded in 'from_basic_amrs_to_genomic_regions(...)'

    //////////////////////////////////////////////////////////////////
    //////  POSTPROCESSING IDENTIFIED AMRS AND COMPUTING SUMMARY STATS
    if (VERBOSE)
      cerr << "========= POST PROCESSING =========" << endl;

    // If each chrom is in its own file
    if(INDIVIDUAL_CHROMS && rank != MPI_ROOT_PROCESS){
      convert_coordinates_wrapper(rank, number_of_processes, MPI_AMR, 
                                  VERBOSE, chroms_dir, fasta_suffix, amrs, 
                                  chrom_name_to_id, chrom_id_to_name, short_chroms);
    }

    // If there is no previos info about the chroms
    if(!INDIVIDUAL_CHROMS && rank != MPI_ROOT_PROCESS) {
      convert_coordinates_parallel(rank, number_of_processes, MPI_AMR, MPI_CHROM_INFO,
                                    VERBOSE, chroms_dir, fasta_suffix, amrs,
                                    chrom_name_to_id, chrom_id_to_name, short_chroms);
    }

    const size_t windows_accepted = amrs.size();
    if (!amrs.empty() && rank == MPI_ROOT_PROCESS) {
      // Could potentially only get the first n p-vals, but would
      // have to sort here and assume sorted for smithlab_utils, or
      // sort twice...
      vector<double> pvals;
      for ( size_t i = 0; i < amrs.size(); ++i)
          pvals.push_back(amrs[i].get_score());

      const double fdr_cutoff = (!USE_BIC && !CORRECTION) ?
         smithlab::get_fdr_cutoff(windows_tested, pvals, critical_value) : 0.0;

      if (!USE_BIC && CORRECTION) {
        smithlab::correct_pvals(windows_tested, pvals);
        for ( size_t i = 0; i < pvals.size(); ++i) {
          amrs[i].set_score(pvals[i]);
        }
      }

      collapse_amrs(amrs);
      const size_t collapsed_amrs = amrs.size();

      /************************ TIME MEASURE ***********************/
      #ifdef DEBUG
      //Start measuring Convert Coordinates Phase Time
      program.subsections[5].subsections[0].start_section();
      #endif
      /************************ TIME MEASURE ***********************/

      if(INDIVIDUAL_CHROMS)
        convert_coordinates_wrapper(rank, number_of_processes, MPI_AMR, 
                                    VERBOSE, chroms_dir, fasta_suffix, amrs,
                                    chrom_name_to_id, chrom_id_to_name, short_chroms);
      else
        convert_coordinates_parallel(rank, number_of_processes, MPI_AMR, MPI_CHROM_INFO,
                                    VERBOSE, chroms_dir, fasta_suffix, amrs,
                                    chrom_name_to_id, chrom_id_to_name, short_chroms);

      /************************ TIME MEASURE ***********************/
      #ifdef DEBUG
      //Start measuring Convert Coordinates Phase Time
      program.subsections[5].subsections[0].end_section();
      #endif
      /************************ TIME MEASURE ***********************/

      merge_amrs(gap_limit, amrs);

      const size_t merged_amrs = amrs.size();

      if (!USE_BIC)
        (CORRECTION || NOFDR)
                ? eliminate_amrs_by_cutoff(critical_value, amrs) :
                       eliminate_amrs_by_cutoff(fdr_cutoff, amrs);

      const size_t amrs_passing_fdr = amrs.size();

      eliminate_amrs_by_size(gap_limit/2, amrs);

      if (VERBOSE) {
        cerr << "WINDOWS TESTED: " << windows_tested << endl
             << "WINDOWS ACCEPTED: " << windows_accepted << endl
             << "COLLAPSED WINDOWS: " << collapsed_amrs << endl
             << "MERGED WINDOWS: " << merged_amrs << endl;
        if (!NOFDR) {
          cerr  << "FDR CUTOFF: " << fdr_cutoff << endl
                << "WINDOWS PASSING FDR: " << amrs_passing_fdr << endl;
        }
        cerr << "AMRS (WINDOWS PASSING MINIMUM SIZE): " << amrs.size() << endl;
      }

      /************************ TIME MEASURE ***********************/
      #ifdef DEBUG
      //End measuring Postprocessing Phase Time
      program.subsections[5].end_section();

      //Start measuring Output Phase Time
      program.subsections[6].start_section();
      #endif
      /************************ TIME MEASURE ***********************/

      std::ofstream of;
      if (!outfile.empty()) of.open(outfile.c_str());
      std::ostream out(outfile.empty() ? cout.rdbuf() : of.rdbuf());
      copy(amrs.begin(), amrs.end(),
      std::ostream_iterator<GenomicRegion>(out, "\n"));

      /************************ TIME MEASURE ***********************/
      #ifdef DEBUG
      //End measuring Output Phase Time
      program.subsections[6].end_section();
      #endif
      /************************ TIME MEASURE ***********************/

    }
    else {

      if (VERBOSE && rank == MPI_ROOT_PROCESS)
          cerr << "No AMRs found." << endl;
      if(rank == MPI_ROOT_PROCESS)
        if(INDIVIDUAL_CHROMS)
          convert_coordinates_wrapper(rank, number_of_processes, MPI_AMR, 
                                    VERBOSE, chroms_dir, fasta_suffix, amrs,
                                    chrom_name_to_id, chrom_id_to_name, short_chroms);
        else
          convert_coordinates_parallel(rank, number_of_processes, MPI_AMR, MPI_CHROM_INFO,
                                      VERBOSE, chroms_dir, fasta_suffix, amrs,
                                      chrom_name_to_id, chrom_id_to_name, short_chroms);
    }


    /************************ TIME MEASURE ***********************/
    #ifdef DEBUG
    //End measuring Full Program Time
    program.end_section();
    //Print times
    if (rank == MPI_ROOT_PROCESS)
      cout << program.to_string() << "\n";

    if(rank == MPI_ROOT_PROCESS)
      for (int i = 0; i < number_of_processes; i++)
        printf("COMPUTE TIME ----- PROCESS %d : %lf s\n", i, local_times_processing[i]);
    #endif
    /************************ TIME MEASURE ***********************/

    MPI_Type_free(&MPI_WINDOW_METADATA);
    MPI_Type_free(&MPI_AMR);
    MPI_Finalize();

  }
  catch (const runtime_error &e) {
    cerr << e.what() << endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    return EXIT_FAILURE;
  }
  catch (std::bad_alloc &ba) {
    cerr << "ERROR: could not allocate memory" << endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


static void
init_mpi(int &argc, char ** &argv, int &rank, int &number_of_processes, 
        MPI_Datatype *MPI_WINDOW_DATATYPE, MPI_Datatype *MPI_AMR_DATATYPE, 
        MPI_Datatype *MPI_CHINFO_DATATYPE) {

  BasicAmr amr;
  MPI_Aint d1, d2;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Get WINDOW_DATATYPE
  MPI_Type_contiguous(5, MPI_AINT, MPI_WINDOW_DATATYPE);
  MPI_Type_commit(MPI_WINDOW_DATATYPE);

  // Get MPI_CHINFO_DATATYPE
  MPI_Type_contiguous(4, MPI_AINT, MPI_CHINFO_DATATYPE);
  MPI_Type_commit(MPI_CHINFO_DATATYPE);

  // Get AMR_DATATYPE
  MPI_Get_address( &(amr.chrom_name_id), &d1);
  MPI_Get_address( &(amr.score), &d2);
  int blocklengths[2] = {4, 1};
  MPI_Aint displs[2] = {0, d2-d1};
  MPI_Datatype types[2] = {MPI_AINT, MPI_DOUBLE};

  MPI_Type_create_struct(2, blocklengths, displs, types, MPI_AMR_DATATYPE);
  MPI_Type_commit(MPI_AMR_DATATYPE);

}

static size_t 
preprocess_epireads(const bool VERBOSE, const bool PROGRESS,
                  const size_t min_obs_per_cpg, const size_t window_size,
                  std::ifstream &in, vector<WindowMetadata> &windows) {

  epiread er;
  vector<EpireadMetadata> epireads_metadata;
  vector<epiread> epireads;
  string prev_chrom, curr_chrom, tmp_states;
  size_t previous_byte = in.tellg(), current_byte;
  size_t total_cost = 0;
  
  /************************ TIME MEASURE ***********************/
  #ifdef DEBUG
  //Start measuring Input-to-Epireads Phase Time
  program.subsections[1].subsections[0].start_section();
  #endif
  /************************ TIME MEASURE ***********************/

  while (in >> er) {

    curr_chrom = er.chr;
    current_byte = in.tellg();

    if (!epireads.empty() && curr_chrom != prev_chrom) {
      /************************ TIME MEASURE ***********************/
      #ifdef DEBUG
      //Start measuring Input-to-Epireads Phase Time
      program.subsections[1].subsections[0].end_section();
      program.subsections[1].subsections[0].store_time();
      program.subsections[1].subsections[1].start_section();
      #endif
      /************************ TIME MEASURE ***********************/
      total_cost +=
          preprocess_chrom(VERBOSE, PROGRESS, min_obs_per_cpg, window_size,
                        prev_chrom, epireads, epireads_metadata, windows);

      epireads.clear();
      epireads_metadata.clear();
      /************************ TIME MEASURE ***********************/
      #ifdef DEBUG
      //Start measuring Input-to-Epireads Phase Time
      program.subsections[1].subsections[1].end_section();
      program.subsections[1].subsections[1].store_time();
      program.subsections[1].subsections[0].start_section();
      #endif
      /************************ TIME MEASURE ***********************/
    }

    epireads.push_back(er);
    epireads_metadata.push_back(EpireadMetadata(previous_byte, current_byte));

    prev_chrom = curr_chrom;
    previous_byte = current_byte;
  }

  /************************ TIME MEASURE ***********************/
  #ifdef DEBUG
  program.subsections[1].subsections[0].end_section();
  program.subsections[1].subsections[0].store_time();
  program.subsections[1].subsections[1].start_section();
  #endif
  /************************ TIME MEASURE ***********************/
  if (!epireads.empty())
    total_cost +=
        preprocess_chrom(VERBOSE, PROGRESS, min_obs_per_cpg, window_size,
                      prev_chrom, epireads, epireads_metadata, windows);
  /************************ TIME MEASURE ***********************/
  #ifdef DEBUG
  program.subsections[1].subsections[1].end_section();
  program.subsections[1].subsections[1].store_time();
  #endif
  /************************ TIME MEASURE ***********************/

  return total_cost;
}

static WindowMetadata
get_current_epireads_with_metadata(const vector<epiread> &epireads,
                     const vector<EpireadMetadata> &epireads_metadata,
                     const size_t max_epiread_len,
                     const size_t cpg_window, const size_t start_pos,
                     size_t &read_id, vector<epiread> &current_epireads) {
  while (read_id < epireads.size() &&
         epireads[read_id].pos + max_epiread_len <=start_pos)
    ++read_id;

  const size_t end_pos = start_pos + cpg_window;
  size_t i;
  for (i = read_id; (i < epireads.size() &&
                            epireads[i].pos < end_pos); ++i) {
    if (epireads[i].end() > start_pos) {
      current_epireads.push_back(epireads[i]);
      clip_read(start_pos, end_pos, current_epireads.back());
    }
  }
  return WindowMetadata(INVALID_CHROM_NAME_ID, epireads_metadata[read_id].start_byte, epireads_metadata[i-1].end_byte, start_pos);
}

static size_t
preprocess_chrom(const bool VERBOSE, const bool PROGRESS,
              const size_t min_obs_per_cpg, const size_t window_size,
              const string &chrom_name, const vector<epiread> &epireads,
              const vector<EpireadMetadata> &epireads_metadata, 
              vector<WindowMetadata> &windows) {

  size_t max_epiread_len = 0;
  const size_t min_obs_per_window = window_size*min_obs_per_cpg;
  const size_t chrom_cpgs = get_n_cpgs(epireads);
  size_t total_cost = 0;
  size_t start_idx = 0;
  const size_t lim = chrom_cpgs < window_size ? 1 : chrom_cpgs - window_size + 1;

  for (size_t i = 0; i < epireads.size(); ++i)
    max_epiread_len = std::max(max_epiread_len, epireads[i].length());

  // Print status info
  if (VERBOSE)
    cerr << "PREPROCESSING: " << chrom_name << " "
         << "[reads: " << epireads.size() << "] "
         << "[cpgs: " << chrom_cpgs << "]" << endl;

  /************************ SOLVE OPENMP DEPENDENCY ***********************/
  // Add extra null windows to fill vector
  size_t previous_windows = windows.size();
  windows.resize(previous_windows + lim, WindowMetadata());

  // MAIN PREPROCESSING LOOP
  // FOR sliding_window_start = first_window_start 
  // TO last_window_start 
  // DO sliding_window_start++
  #pragma omp parallel for reduction(+:total_cost) firstprivate(start_idx) schedule(dynamic)
  for (size_t i = 0; i < lim; ++i) {

    vector<epiread> current_epireads;

    // Read from 'epireads' the lines in the current window
    WindowMetadata current_window = get_current_epireads_with_metadata(
                          epireads, epireads_metadata, max_epiread_len,
                          window_size, i, start_idx, current_epireads);

    // IF is seems worth testing the current window
    // THEN don't test it, but remember it is worth
    current_window.cost = total_states(current_epireads);
    if (current_window.cost >= min_obs_per_window) {
      windows[previous_windows + i] = current_window;
      total_cost += current_window.cost;
    }
  }

  /************************ SOLVE OPENMP DEPENDENCY ***********************/
  vector<WindowMetadata>::iterator real_end = std::remove(windows.begin() + previous_windows, windows.end(), WindowMetadata());
  windows.erase(real_end, windows.end());

  // Print status info
  if (PROGRESS)
    cerr << '\r' << chrom_name << " 100%" << endl;
  return total_cost;
}

static void
get_target_cost_for_each_process(const size_t total_cost, 
                              const int number_of_processes,
                              size_t target_cost_of_process[]) {
  size_t base_cost_per_process = total_cost / number_of_processes;
  size_t processes_with_extra_cost = total_cost % number_of_processes;
  for (size_t i = 0; i < number_of_processes; i++) {
    if (i < processes_with_extra_cost)
      target_cost_of_process[i] = (i + 1) * (base_cost_per_process + 1);
    else
      target_cost_of_process[i] = ((i + 1) * base_cost_per_process) + processes_with_extra_cost;
  }

}

static void
get_windows_per_proc_with_displs(const int number_of_processes,
                              const size_t target_cost_of_process[],
                              const vector<WindowMetadata> &windows,
                              int windows_per_process[], int displs[]) {

  int displ = 0;
  size_t window_index = 0;
  size_t acumulated_cost = 0;

  displs[0] = 0;

  for (size_t i = 0; i < number_of_processes - 1; i++) {
    //look_for_target_cost();
    while(window_index < windows.size() && acumulated_cost < target_cost_of_process[i]){
      acumulated_cost += windows[window_index].cost;
      window_index++;
    }
    size_t extra = acumulated_cost - target_cost_of_process[i];
    size_t deficit = windows[window_index-1].cost - extra;
    window_index -=  deficit > extra ? 1 : 2;

    //update_windows_and_displs
    windows_per_process[i] = window_index - displ;
    displ = window_index;
    displs[i+1] = displ;

    // reset index for next iteration
    window_index +=  deficit > extra ? 1 : 2;

  }

  windows_per_process[number_of_processes-1] = windows.size() - displ;

}

static void
distribute_windows(const vector<WindowMetadata> &windows,
              const size_t total_cost, const int number_of_processes,
              int windows_per_process[], int displs[],
              int &my_number_of_windows, vector<WindowMetadata> &my_windows,
              MPI_Datatype &MPI_WINDOW_DATATYPE, const int rank) {

  if (rank == MPI_ROOT_PROCESS){

    #ifdef COST_DISTRIBUTION

    size_t target_cost_of_process[number_of_processes];
    get_target_cost_for_each_process(total_cost, number_of_processes, target_cost_of_process);              

    get_windows_per_proc_with_displs(number_of_processes,
                              target_cost_of_process, windows,
                              windows_per_process, displs);

    #endif
    #ifdef PURE_BLOCK_DISTRIBUTION
    size_t windows_div, windows_rem;
    windows_div = windows.size() / number_of_processes;
    windows_rem = windows.size() % number_of_processes;

    displs[0] = 0;
    for (size_t i = 0; i < number_of_processes-1; i++)
    {
      windows_per_process[i] = (i < windows_rem) ? windows_div+1 : windows_div;
      displs[i+1] = displs[i] + windows_per_process[i];
    }
    windows_per_process[number_of_processes-1] = (number_of_processes-1 < windows_rem) ? windows_div+1 : windows_div;
    
    #endif
  
  }

    // first notify the processes how many elements they have to receive
  MPI_Scatter( windows_per_process , 1 , MPI_INT , 
            &my_number_of_windows , 1 , MPI_INT , 
            MPI_ROOT_PROCESS , MPI_COMM_WORLD);

    // then each process allocates memory
  my_windows.resize(my_number_of_windows, WindowMetadata());

    // finally data is actually distributed in a block fashion
  MPI_Scatterv( &windows[0] , windows_per_process , displs , MPI_WINDOW_DATATYPE , 
                &my_windows[0] , my_number_of_windows , MPI_WINDOW_DATATYPE , 
                MPI_ROOT_PROCESS , MPI_COMM_WORLD);

}

static epiread
read_raw_epiread(const char *raw_epireads, const size_t position, const size_t raw_length) {

  string raw_epiread(&(raw_epireads[position]), raw_length);
  //std::cout << raw_epiread << endl;
  string chrom;
  size_t pos;
  string seq;
  size_t t_pos = raw_epiread.find('\t');
  size_t t2_pos = raw_epiread.find('\t', t_pos+1);
  size_t n_pos = raw_epiread.find('\n', t2_pos+1);
  chrom = raw_epiread.substr(0, t_pos);
  pos = std::stoul(raw_epiread.substr(t_pos+1, t2_pos - t_pos - 1));
  seq = raw_epiread.substr(t2_pos+1, n_pos - t2_pos - 1);

  //std::cout << chrom << endl;
  //std::cout << pos << endl;
  //std::cout << seq << endl;

  return epiread(chrom, pos, seq);

}

static void
get_epireads_from_window(const WindowMetadata &window,
                        const size_t window_size,
                        const size_t initial_position, 
                        const char * raw_epireads,
                        vector<epiread> &current_epireads) {

  size_t end_pos = window.window_start + window_size;

  size_t window_start = window.start_byte - initial_position;
  size_t window_end = window.end_byte - initial_position;

  size_t position = window_start;
  size_t next_position;
  char* n_pos;
  epiread er;

  while(position < window_end){
    next_position = (size_t) ( strchr(raw_epireads+position, '\n') - (raw_epireads+position) );
    er = read_raw_epiread(raw_epireads, position, next_position);
    if (er.end() > window.window_start){
      clip_read(window.window_start, end_pos, er);
      current_epireads.push_back(er);
    }
    position += next_position+1;
  }

}

static void
process_windows(vector<BasicAmr> &my_amrs, 
              const vector<WindowMetadata> &my_windows,
              const size_t my_number_of_windows,
              const string reads_file, 
              const size_t window_size,
              const EpireadStats epistat) {

  if(my_windows.empty()){
    return;
  }

  // Prepare the space for the amrs identified              
  my_amrs.resize(my_windows.size(), BasicAmr());

  size_t initial_position = my_windows[0].start_byte;
  size_t last_position = my_windows[my_number_of_windows-1].end_byte;
  char * raw_epireads = (char*) malloc(sizeof(char) * (last_position - initial_position + 1)); // extra 1 space for \0

  // Create file stream
  std::ifstream in(reads_file.c_str());
  if (!in)
    throw runtime_error("cannot open input file: " + reads_file);


  // Set position with .seekg()
  in.seekg(initial_position);

  // Read important bytes with .get()
  in.get(raw_epireads, last_position - initial_position + 1, '\0');


  // Threads dinamically process windows
  #pragma omp parallel for schedule(dynamic)
  for(size_t i = 0; i < my_windows.size(); i++){
    WindowMetadata window = my_windows[i];
    vector<epiread> current_epireads;

    get_epireads_from_window(window, window_size, 
                          initial_position, raw_epireads, 
                          current_epireads);

    bool is_significant = false;
    const double score = epistat.test_asm(current_epireads, is_significant);
    if (is_significant)
      add_amr_ompi(window.chrom_name_id, window.window_start, window_size, current_epireads, score, my_amrs, i);
  }

  free(raw_epireads);
}

static void 
gather_basic_amrs(const vector<BasicAmr> &my_amrs,
                const int my_number_of_amrs,
                const MPI_Datatype MPI_AMR_DATATYPE,
                const int recvcounts[],
                const int displs[],
                vector<BasicAmr> &amrs) {

  MPI_Gatherv(&my_amrs[0], my_number_of_amrs, MPI_AMR_DATATYPE, 
            &amrs[0], recvcounts, displs, MPI_AMR_DATATYPE, 
            MPI_ROOT_PROCESS, MPI_COMM_WORLD);
}

static size_t
get_id_from_chrom(const unordered_map<string, size_t> &chrom_name_to_id,
                  const string chrom_name){

  auto it = chrom_name_to_id.find(chrom_name);

  if (it == chrom_name_to_id.end())
    throw runtime_error("could not find chrom ID from name:\n" + chrom_name);

  return it->second;

}

static string
get_chrom_from_id(const unordered_map<size_t, string> &chrom_id_to_name,
                  const size_t chrom_id){

  auto it = chrom_id_to_name.find(chrom_id);

  if (it == chrom_id_to_name.end())
    throw runtime_error("could not find chrom from ID:\n" + std::to_string(chrom_id));

  return it->second;

}

static bool
is_short_chrom(string chrom_name, std::set<string> &short_chroms){
  return !(short_chroms.find(chrom_name) == short_chroms.end());
}

static void
from_basic_amrs_to_genomic_regions(const vector<BasicAmr> &basic_amrs, 
                                  vector<GenomicRegion> &amrs,
                                  const unordered_map<size_t, string> &chrom_id_to_name,
                                  std::set<string> &short_chroms) {

  if(basic_amrs.size() == 0)
    return;

  static const string name_label("AMR");

  size_t current_chrom_id = basic_amrs[0].chrom_name_id;
  string current_chrom = get_chrom_from_id(chrom_id_to_name, current_chrom_id);
  // Previous version --- as reference
  /*
  //#pragma omp parallel for schedule(static) // [DON'T DO] There is a shared structure underneath
  for (size_t i = 0; i < basic_amrs.size(); i++)
  {
    if(current_chrom_id != basic_amrs[i].chrom_name_id){
      current_chrom_id = basic_amrs[i].chrom_name_id;
      current_chrom = get_chrom_from_id(chrom_id_to_name, current_chrom_id);
    }
    const string name( name_label + toa(i) + ":" + toa(basic_amrs[i].reads_size) );
    amrs[i] = GenomicRegion(current_chrom, basic_amrs[i].start_cpg, basic_amrs[i].end_cpg, name, basic_amrs[i].score, '+');
  }*/

  size_t i = 0;
  size_t amr_num = 0;

  // Add checking for small chrom on first chrom 
  while(is_short_chrom(current_chrom, short_chroms)){ 
          // Add first window
          const string name( name_label + toa(amr_num) + ":" + toa(basic_amrs[i].reads_size) );
          amr_num++;
          amrs.push_back(GenomicRegion(current_chrom, basic_amrs[i].start_cpg, basic_amrs[i].end_cpg, name, basic_amrs[i].score, '+'));
          i++;

          // Ignore all windows of this chrom
          while(i < basic_amrs.size() && basic_amrs[i].chrom_name_id == current_chrom_id){
            i++;
          }

          // If i reached end of buffer, break
          if (i>= basic_amrs.size())
            break;

          // Do the set up for the next chrom
          current_chrom_id = basic_amrs[i].chrom_name_id;
          current_chrom = get_chrom_from_id(chrom_id_to_name, current_chrom_id);
      }

  while(i < basic_amrs.size()){
    if(current_chrom_id != basic_amrs[i].chrom_name_id){
      current_chrom_id = basic_amrs[i].chrom_name_id;
      current_chrom = get_chrom_from_id(chrom_id_to_name, current_chrom_id);
      // If chrom is small we only keep the first window and we move to the next one
      while(is_short_chrom(current_chrom, short_chroms)){
          // Add first window
          const string name( name_label + toa(amr_num) + ":" + toa(basic_amrs[i].reads_size) );
          amr_num++;
          amrs.push_back(GenomicRegion(current_chrom, basic_amrs[i].start_cpg, basic_amrs[i].end_cpg, name, basic_amrs[i].score, '+'));
          i++;

          // Ignore all windows of this chrom
          while(i < basic_amrs.size() && basic_amrs[i].chrom_name_id == current_chrom_id){
            i++;
          }

          // If i reached end of buffer, break
          if (i>= basic_amrs.size())
            break;

          // Do the set up for the next chrom
          current_chrom_id = basic_amrs[i].chrom_name_id;
          current_chrom = get_chrom_from_id(chrom_id_to_name, current_chrom_id);
      }

      // If i reached end of buffer, break
      if (i>= basic_amrs.size())
        break;
    }
    const string name( name_label + toa(amr_num) + ":" + toa(basic_amrs[i].reads_size) );
    amr_num++;
    amrs.push_back(GenomicRegion(current_chrom, basic_amrs[i].start_cpg, basic_amrs[i].end_cpg, name, basic_amrs[i].score, '+'));
    i++;
  }


  
}

static void
create_chrom_name_to_id_equivalences(vector<std::pair<std::pair<string, int>, size_t>> &chrom_names_with_ids, 
                                    size_t chroms,
                                    unordered_map<string, size_t> &chrom_name_to_id,
                                    unordered_map<size_t, string> &chrom_id_to_name,
                                    std::set<string> &short_chroms){

  // Get my local max_length
  size_t local_max_length = 0;
  size_t global_max_length;
  for (auto &chrom_name_info : chrom_names_with_ids){
    local_max_length = std::max(local_max_length, chrom_name_info.first.first.length());
  }

  // Reduce local max_length to ROOT
  MPI_Allreduce(&local_max_length, &global_max_length, 1, MPI_AINT, MPI_MAX, MPI_COMM_WORLD);

  // Bcast number of chroms
  MPI_Bcast(&chroms, 1, MPI_AINT, MPI_ROOT_PROCESS, MPI_COMM_WORLD);

  // ROOT allocs memory
  vector<char> chrom_names((global_max_length+1) * chroms, '\0');
  vector<int> is_short_chrom(chroms, 0);

  // Create window to write string data
  MPI_Win chroms_window;
  MPI_Win_create(&chrom_names[0], (global_max_length+1) * chroms * sizeof(char), 
                (global_max_length+1) * sizeof(char), MPI_INFO_NULL, 
                MPI_COMM_WORLD, &chroms_window);

  MPI_Win_fence(MPI_MODE_NOPRECEDE, chroms_window);
  // Write data on ROOT
  for (auto &chrom_name_info : chrom_names_with_ids){
    MPI_Put(chrom_name_info.first.first.c_str(), chrom_name_info.first.first.length(), MPI_CHAR, 
            MPI_ROOT_PROCESS, chrom_name_info.second-1, 
            chrom_name_info.first.first.length(), MPI_CHAR, chroms_window);
  }
  MPI_Win_fence(0, chroms_window);

  MPI_Win_free(&chroms_window);

  // Create window to write boolean data
  MPI_Win short_chroms_window;
  MPI_Win_create(&is_short_chrom[0], chroms * sizeof(int), 
                sizeof(int), MPI_INFO_NULL, 
                MPI_COMM_WORLD, &short_chroms_window);

  MPI_Win_fence(MPI_MODE_NOPRECEDE, short_chroms_window);
  // Write data on ROOT
  for (auto &chrom_name_info : chrom_names_with_ids){
    MPI_Put(&chrom_name_info.first.second, 1, MPI_INT, 
            MPI_ROOT_PROCESS, chrom_name_info.second-1, 
            1, MPI_INT, short_chroms_window);
  }
  MPI_Win_fence(0, short_chroms_window);

  MPI_Win_free(&short_chroms_window);

  // Share data to all
  MPI_Bcast(&chrom_names[0], (global_max_length+1) * chroms, 
            MPI_CHAR, MPI_ROOT_PROCESS, MPI_COMM_WORLD);
  MPI_Bcast(&is_short_chrom[0], chroms, 
            MPI_INT, MPI_ROOT_PROCESS, MPI_COMM_WORLD);

  // Populate structures
  short_chroms.clear();
  for (size_t i = 0; i < chroms; i++){
    char *current_chrom = &chrom_names[i * (global_max_length+1)];
    string chrom_name(current_chrom);
    short_chroms.insert(chrom_name);
  }
  for (size_t i = 0; i < chroms; i++){
    char *current_chrom = &chrom_names[i * (global_max_length+1)];
    string chrom_name(current_chrom);
    size_t chrom_id = i+1;
    chrom_name_to_id.insert({chrom_name, chrom_id});
    chrom_id_to_name.insert({chrom_id, chrom_name});
    // If the chrom proves to be long enough
    if(is_short_chrom[i] == 0)
      short_chroms.erase(chrom_name);
  }
  
}

static void
preprocess_epireads_parallel(const bool VERBOSE, const bool PROGRESS,
                    const size_t min_obs_per_cpg, const size_t window_size,
                    const string epireads_file, vector<WindowMetadata> &main_windows,
                    size_t &total_cost, MPI_Datatype &MPI_WINDOW_METADATA,
                    const int rank, const int number_of_processes,
                    size_t &number_of_windows,
                    unordered_map<string, size_t> &chrom_name_to_id,
                    unordered_map<size_t, string> &chrom_id_to_name,
                    std::set<string> &short_chroms) {
  char* chunk;
  size_t process_start, process_size, process_size_with_overlapping, last_owned_byte;
  size_t local_cost, last_chrom_max_epiread_len;
  size_t real_windows_size, first_window = 0;
  vector<epiread> epireads;
  vector<EpireadMetadata> epireads_metadata;
  vector<WindowMetadata> my_windows;
  WindowMetadata last_valid_window;
  vector<std::pair<std::pair<string, int>, size_t>> chrom_names_with_ids;


  // Create window for shared index
  size_t chrom_name_index = 1;
  MPI_Win index_window;
  MPI_Win_create(&chrom_name_index, 1 * sizeof(size_t), sizeof(size_t), MPI_INFO_NULL, MPI_COMM_WORLD, &index_window);


  readEpireadFile(epireads_file, rank, number_of_processes, chunk, 
                  process_start, process_size, process_size_with_overlapping);

  from_raw_to_epireads(chunk, rank, process_size_with_overlapping, 
                        process_start, epireads, epireads_metadata);

  local_cost = epireads_to_windows(VERBOSE, PROGRESS, epireads, 
                                    epireads_metadata, my_windows, 
                                    min_obs_per_cpg, window_size, 
                                    last_chrom_max_epiread_len,
                                    index_window, chrom_names_with_ids);

  // All procs read the file and figure out the windows
  // Its time to create the chr_name : str <-> size_t equivalence
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Win_free(&index_window);

  create_chrom_name_to_id_equivalences(chrom_names_with_ids, chrom_name_index-1,
                                        chrom_name_to_id, chrom_id_to_name, short_chroms);


  // Si no soy el último proceso
  // Entonces busca las ventanas que sobran
  if(rank < number_of_processes-1){
    last_owned_byte = find_last_owned_byte(process_start, process_size, chunk);
    real_windows_size = filter_leftover_windows(epireads_file, epireads, epireads_metadata, 
                            my_windows, last_owned_byte, local_cost, min_obs_per_cpg, 
                            window_size, last_chrom_max_epiread_len, chrom_name_to_id);
  
  // Si soy el último proceso, 
  // Entonces no comparto las ventanas finales, por lo que todas son validas 
  } else {
    real_windows_size = my_windows.size();
  }

  // Raw data won't be used again
  free(chunk);

  last_valid_window = communicate_last_valid_window(my_windows, real_windows_size-1, 
                                MPI_WINDOW_METADATA, rank, number_of_processes);

  // Si no soy el primer proceso, limpia las falsas ventanas al inicio
  if (rank /* != 0 */){
    first_window = remove_leading_windows(my_windows, epireads, 
                                          epireads_metadata, last_valid_window, 
                                          real_windows_size, local_cost);                              

    // Remove fake leading windows from the real size
    real_windows_size -= first_window;
  }

  number_of_windows = gather_windows(my_windows, real_windows_size, first_window, local_cost, 
                                  number_of_processes, total_cost, 
                                  main_windows, MPI_WINDOW_METADATA, rank);

}

static void 
readEpireadFile(string epireads_file, int rank, int number_of_processes, 
                char* &chunk, size_t &process_start, 
                size_t &process_size, size_t &process_size_with_overlapping) {

  MPI_File input_epireads_file;
  MPI_Offset filesize;

  const size_t ONE_GB = 1 * 1024 * 1024 * 1024;

  // Abre el archivo
  if (MPI_File_open( MPI_COMM_WORLD, epireads_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &input_epireads_file))
    throw runtime_error("could not open file: " + epireads_file);

  // Consigue el tamaño del archivo
  MPI_File_get_size(input_epireads_file, &filesize);

  const size_t overlapping_size = std::max((size_t) EPIREAD_LENGTH, (size_t) (filesize * OVERLAPPING_FRACTION));

  // Dividelo en bloques
  size_t size_div, size_rem;
  size_div = filesize / number_of_processes;
  size_rem = filesize % number_of_processes;

  // Calculate the start of the block and its non-overlapping size
  if (rank < size_rem){
    process_start = rank * (size_div + 1); 
    process_size = (size_div + 1);
  } else {
    process_start = (rank * size_div) + size_rem;
    process_size = size_div;
  }

  // Add overlapping if EOF is not found
  // Else add overlapping until EOF is found
  if ( (process_start + process_size + overlapping_size) <= filesize )
    process_size_with_overlapping = process_size + overlapping_size;
  else
    process_size_with_overlapping = filesize - process_start;

  // Allocate memory
  chunk = (char*) malloc( sizeof(char) * (process_size_with_overlapping + 1) );

  // Everyone reads its block
  //MPI_File_read_at_all(input_epireads_file, process_start, chunk, process_size_with_overlapping, MPI_CHAR, MPI_STATUS_IGNORE);

  size_t read_offset = process_start;
  char *buffer_slice = chunk;
  while(process_size_with_overlapping > ONE_GB){
    // Read one GB
    MPI_File_read_at(input_epireads_file, read_offset, buffer_slice, ONE_GB, MPI_CHAR, MPI_STATUS_IGNORE);

    // Update the pointer
    process_size_with_overlapping -= ONE_GB;
    buffer_slice = buffer_slice + ONE_GB;
  }
  MPI_File_read_at(input_epireads_file, read_offset, buffer_slice, ONE_GB, MPI_CHAR, MPI_STATUS_IGNORE);

  // Add '\0' at the end, for security
  chunk[process_size_with_overlapping] = '\0';

  // At this point there is no more need to read the file
  // All the data has been moved to memory
  MPI_File_close(&input_epireads_file);

}

static void
from_raw_to_epireads(char* chunk, int rank,
                      const size_t chunk_size, const size_t process_start,
                      vector<epiread> &epireads, vector<EpireadMetadata> &em) {
                      
  vector<vector<epiread>> omp_epireads;
  vector<vector<EpireadMetadata>> omp_em;

  #pragma omp parallel
  {
    int num_threads = omp_get_num_threads();
    int thread_num = omp_get_thread_num();

    if (thread_num == 0){
      omp_epireads.resize(num_threads);
      omp_em.resize(num_threads);
    }
    
    #pragma omp barrier

    // Divide process chunk in sub-chunks, one per thread 
    // Last char of thread 'i' must overlap with first char of thread 'i+1'
    size_t thread_start, thread_end;
    size_t epiread_length;

    size_t size_div, size_rem;
    size_div = chunk_size / num_threads;
    size_rem = chunk_size % num_threads;

    if(thread_num < size_rem)
      thread_start = thread_num * (size_div + 1);
    else
      thread_start = (thread_num * size_div) + size_rem;

    if(thread_num+1 < size_rem)
      thread_end = (thread_num+1) * (size_div + 1);
    else
      thread_end = ((thread_num+1) * size_div) + size_rem;

    // Advance to the next newline so no line is processed twice
    // Unless I'm the first thread of the first process
    size_t position_in_chunk = thread_start;
    epiread er;
    if(rank /* != 0 */ || thread_num /* != 0 */){

      // Find a END_OF_LINE character
      while(chunk[position_in_chunk] != '\n')
        position_in_chunk++;
      
      // Move once again to the start of the next line
      position_in_chunk++;
    }
    if(thread_num == (num_threads-1)){
      while (chunk[thread_end]!='\n')
        thread_end--;
    }
    // Process epireads until the end of the sub-chunk is reached
    while(position_in_chunk <= thread_end){
      epiread_length = (size_t) ( strchr(chunk+position_in_chunk, '\n') - (chunk+position_in_chunk) );
      er = read_raw_epiread(chunk, position_in_chunk, epiread_length);
      omp_epireads[thread_num].push_back(er);
      omp_em[thread_num].push_back(EpireadMetadata(process_start+position_in_chunk, process_start+position_in_chunk+epiread_length+1));
      position_in_chunk += epiread_length+1;
    }

  }

  // Join all epireads together
  for (size_t i = 0; i < omp_epireads.size(); i++){
    epireads.insert(epireads.end(), omp_epireads[i].begin(), omp_epireads[i].end());
    em.insert(em.end(), omp_em[i].begin(), omp_em[i].end());
  }  

}

static WindowMetadata
get_current_epireads_with_metadata_parallel(const vector<epiread> &epireads,
                     const vector<EpireadMetadata> &epireads_metadata,
                     const size_t max_epiread_len, const size_t chrom_end_index,
                     const size_t cpg_window, const size_t start_pos,
                     size_t &read_id, vector<epiread> &current_epireads,
                     const size_t current_chrom_name_index) {
  while (read_id < chrom_end_index &&
         epireads[read_id].pos + max_epiread_len <=start_pos)
    ++read_id;

  const size_t end_pos = start_pos + cpg_window;
  size_t i;
  for (i = read_id; (i < chrom_end_index &&
                            epireads[i].pos < end_pos); ++i) {
    if (epireads[i].end() > start_pos) {
      current_epireads.push_back(epireads[i]);
      clip_read(start_pos, end_pos, current_epireads.back());
    }
  }
  return WindowMetadata(current_chrom_name_index, epireads_metadata[read_id].start_byte, epireads_metadata[i-1].end_byte, start_pos);
}

static size_t
preprocess_chrom_parallel(const bool VERBOSE, const bool PROGRESS,
              const size_t min_obs_per_cpg, const size_t window_size,
              const string &chrom_name, const size_t chrom_cpgs,
              const size_t chrom_start_index, const size_t chrom_end_index, 
              const size_t max_epiread_len, const vector<epiread> &epireads,
              const vector<EpireadMetadata> &epireads_metadata, 
              vector<WindowMetadata> &windows,
              const size_t current_chrom_name_index) {

  const size_t min_obs_per_window = window_size*min_obs_per_cpg;
  size_t total_cost = 0;
  size_t start_idx = chrom_start_index;
  const size_t lim = chrom_cpgs < window_size ? 1 : chrom_cpgs - window_size + 1;

  // Print status info
  if (VERBOSE)
    cerr << "PREPROCESSING: " << chrom_name << " "
         << "[reads: " << epireads.size() << "] "
         << "[cpgs: " << chrom_cpgs << "]" << endl;

  /************************ SOLVE OPENMP DEPENDENCY ***********************/
  // Add extra null windows to fill vector
  size_t previous_windows = windows.size();
  windows.resize(previous_windows + lim, WindowMetadata());

  // MAIN PREPROCESSING LOOP
  // FOR sliding_window_start = first_window_start 
  // TO last_window_start 
  // DO sliding_window_start++
  #pragma omp parallel for reduction(+:total_cost) firstprivate(start_idx) schedule(dynamic)
  for (size_t i = 0; i < lim; ++i) {

    vector<epiread> current_epireads;

    // Read from 'epireads' the lines in the current window
    WindowMetadata current_window = get_current_epireads_with_metadata_parallel(
                          epireads, epireads_metadata, max_epiread_len, chrom_end_index,
                          window_size, i, start_idx, current_epireads, current_chrom_name_index);

    // IF is seems worth testing the current window
    // THEN don't test it, but remember it is worth
    current_window.cost = total_states(current_epireads);
    if (current_window.cost >= min_obs_per_window) {
      windows[previous_windows + i] = current_window;
      total_cost += current_window.cost;
    }
  }

  /************************ SOLVE OPENMP DEPENDENCY ***********************/
  vector<WindowMetadata>::iterator real_end = std::remove(windows.begin() + previous_windows, windows.end(), WindowMetadata());
  windows.erase(real_end, windows.end());

  // Print status info
  if (PROGRESS)
    cerr << '\r' << chrom_name << " 100%" << endl;
  return total_cost;
}

static size_t
get_new_index(const size_t increment, MPI_Win &index_window){

  size_t new_index;
  MPI_Win_lock(MPI_LOCK_SHARED, MPI_ROOT_PROCESS, 0, index_window);
  MPI_Fetch_and_op(&increment, &new_index, MPI_AINT, 
                  MPI_ROOT_PROCESS, 0, MPI_SUM, index_window);
  MPI_Win_unlock(MPI_ROOT_PROCESS, index_window);

  return new_index;

}

static size_t
epireads_to_windows(const bool VERBOSE, const bool PROGRESS,
                    const vector<epiread> &epireads, 
                    const vector<EpireadMetadata> &em,
                    vector<WindowMetadata> &windows,
                    const size_t min_obs_per_cpg, 
                    const size_t window_size,
                    size_t &max_epiread_len,
                    MPI_Win &index_window,
                    vector<std::pair<std::pair<string, int>, size_t>> &chrom_names_with_ids){

  if(epireads.size() == 0)
    return 0; 

  const size_t INDEX_INCREMENT = 1;
  string current_chrom = epireads[0].chr;
  size_t chrom_start_index = 0, chrom_end_index;
  size_t chrom_cpgs = 0;
  size_t total_cost = 0;
  max_epiread_len = 0;

  // Get index for the current chrom
  size_t current_chrom_name_index = get_new_index(INDEX_INCREMENT, index_window);

  for (size_t i = 0; i < epireads.size(); i++){
    if (epireads[i].chr != current_chrom){
      // Work with current chrom
      chrom_end_index = i;
      total_cost += preprocess_chrom_parallel(VERBOSE, PROGRESS, 
                        min_obs_per_cpg, window_size, current_chrom, 
                        chrom_cpgs,chrom_start_index, chrom_end_index, 
                        max_epiread_len, epireads, em, windows,
                        current_chrom_name_index);
      
      // Store the equivalence
      chrom_names_with_ids.push_back(std::pair<std::pair<string, int>, size_t>(
                                      std::pair<string, int>(current_chrom, chrom_cpgs < window_size ? 1 : 0), 
                                      current_chrom_name_index)
                                    );
      
      // Prepare for the next chrom
      current_chrom = epireads[i].chr;
      chrom_start_index = i;
      max_epiread_len = 0;
      chrom_cpgs = 0;
      current_chrom_name_index = get_new_index(INDEX_INCREMENT, index_window);
    }

    chrom_cpgs = std::max(chrom_cpgs, epireads[i].end());
    max_epiread_len = std::max(max_epiread_len, epireads[i].length());

  }

  // Iterate once again with the leftover chrom at the end
  chrom_end_index = epireads.size();
  total_cost += preprocess_chrom_parallel(VERBOSE, PROGRESS, 
                        min_obs_per_cpg, window_size, current_chrom, 
                        chrom_cpgs,chrom_start_index, chrom_end_index, 
                        max_epiread_len, epireads, em, windows,
                        current_chrom_name_index);
  // Store the equivalence
  chrom_names_with_ids.push_back(std::pair<std::pair<string, int>, size_t>(
                                  std::pair<string, int>(current_chrom, chrom_cpgs < window_size ? 1 : 0), 
                                  current_chrom_name_index)
                                );

  return total_cost;
  
}

static size_t 
find_last_owned_byte(size_t start, size_t chunk_size, char* chunk) {
  size_t index = chunk_size;
  // Jump one line
  while (chunk[index] != '\n')
    index++;
  index++;

  // Go to the end of the new line
  while (chunk[index] != '\n')
    index++;
  
  return index+start;
}

// Based on '''preprocess_chrom_parallel''', but with specific range on the main loop
static size_t
preprocess_chrom_parallel_on_range(const bool VERBOSE, const bool PROGRESS,
              const size_t min_obs_per_cpg, const size_t window_size,
              const string &chrom_name, 
              const size_t window_range_start, 
              const size_t window_range_end,
              const size_t chrom_start_index, const size_t chrom_end_index, 
              const size_t max_epiread_len, const vector<epiread> &epireads,
              const vector<EpireadMetadata> &epireads_metadata, 
              vector<WindowMetadata> &windows,
              const size_t current_chrom_name_index){
 
  const size_t min_obs_per_window = window_size*min_obs_per_cpg;
  size_t total_cost = 0;
  size_t start_idx = chrom_start_index;
  const size_t range = window_range_end - window_range_start;

  // Print status info
  if (VERBOSE)
    cerr << "PREPROCESSING: " << chrom_name << " "
         << "[reads: " << epireads.size() << "] "
         << "[cpgs: " << range << "]" << endl;

  /************************ SOLVE OPENMP DEPENDENCY ***********************/
  // Add extra null windows to fill vector
  size_t previous_windows = windows.size();
  windows.resize(previous_windows + range, WindowMetadata());

  // MAIN PREPROCESSING LOOP
  // FOR sliding_window_start = first_window_start 
  // TO last_window_start 
  // DO sliding_window_start++
  #pragma omp parallel for reduction(+:total_cost) firstprivate(start_idx) schedule(dynamic)
  for (size_t i = window_range_start; i < window_range_end; ++i) {

    vector<epiread> current_epireads;

    // Read from 'epireads' the lines in the current window
    WindowMetadata current_window = get_current_epireads_with_metadata_parallel(
                          epireads, epireads_metadata, max_epiread_len, chrom_end_index,
                          window_size, i, start_idx, current_epireads, current_chrom_name_index);

    // IF is seems worth testing the current window
    // THEN don't test it, but remember it is worth
    current_window.cost = total_states(current_epireads);
    if (current_window.cost >= min_obs_per_window) {
      windows[previous_windows + i] = current_window;
      total_cost += current_window.cost;
    }
  }

  /************************ SOLVE OPENMP DEPENDENCY ***********************/
  vector<WindowMetadata>::iterator real_end = std::remove(windows.begin() + previous_windows, windows.end(), WindowMetadata());
  windows.erase(real_end, windows.end());

  // Print status info
  if (PROGRESS)
    cerr << '\r' << chrom_name << " 100%" << endl;
  return total_cost;
}

static size_t
find_epiread_index_from_offset(const size_t start_offset,
                              const size_t initial_index, 
                              const vector<EpireadMetadata> &epireads){

  size_t index = initial_index;
  while(epireads[index].start_byte > start_offset)
    index--;

  return index;

}

static void
add_overlapping_to_avoid_shared_on_end(const string &epireads_file,
                                      vector<epiread> &epireads,
                                      vector<EpireadMetadata> &epireads_metadata,
                                      vector<WindowMetadata> &windows,
                                      const size_t min_obs_per_cpg,
                                      const size_t window_size,
                                      const size_t shared_epiread_index, 
                                      size_t &last_chrom_max_epiread_len,
                                      size_t &total_cost,
                                      unordered_map<string, size_t> chrom_name_to_id){

  epiread new_read;
  const string current_chrom(epireads[shared_epiread_index].chr);
  size_t lowest_new_pos = SIZE_MAX, first_window_to_update_start;
  size_t initial_epireads_size = epireads.size();

  size_t shared_epiread_pos = epireads[shared_epiread_index].pos;
  size_t last_important_window_start_pos = 
            shared_epiread_pos + last_chrom_max_epiread_len;
  size_t last_important_window_end_pos = 
            last_important_window_start_pos + window_size;

  size_t start_to_read_offset = epireads_metadata[epireads.size()-1].end_byte;
  size_t current_pos, previous_pos;

  // Open file 
  std::ifstream in(epireads_file.c_str(), std::ios::binary);
  if (!in)
    throw runtime_error("cannot open input file " + epireads_file);
  
  // Set up the right initial position
  in.seekg(start_to_read_offset);
  previous_pos = in.tellg();

  // While reads are important for the new windows
  while(in >> new_read && new_read.chr == current_chrom && new_read.pos < last_important_window_end_pos){
    
    
    // Add the new read with metadata to the vectors
    current_pos= in.tellg();
    epireads.push_back(new_read);
    epireads_metadata.push_back(EpireadMetadata(previous_pos, current_pos));
    previous_pos = current_pos;

    // Update lower new position it needed
    lowest_new_pos = std::min(lowest_new_pos, new_read.pos);
    
    // If this new read is the largest of the chrom
    // we update max_epiread_len, and so,
    // window start and end pos.
    if(new_read.length() > last_chrom_max_epiread_len){
      last_chrom_max_epiread_len = new_read.length();
      last_important_window_start_pos = 
                shared_epiread_pos + last_chrom_max_epiread_len;
      last_important_window_end_pos = 
                last_important_window_start_pos + last_chrom_max_epiread_len;
    }

  }

  // Figure out from which window we will have to start updating
  first_window_to_update_start = lowest_new_pos < window_size ? 
                                  0 : lowest_new_pos - (window_size - 1);

  // Clean invalid windows
  // First condition makes sure the position of the window is invalidated
  // The second condition makes sure the window remains in the same chrom
  for (size_t i = windows.size() - 1; 
        windows[i].window_start > first_window_to_update_start && 
        windows[i].end_byte >= epireads_metadata[shared_epiread_index].start_byte; 
        i--) {
    total_cost -= windows[i].cost;
    windows.pop_back();
  }
  
  WindowMetadata last_window = windows[windows.size() - 1];

  if(last_window.window_start == first_window_to_update_start){
    windows.pop_back();
    total_cost -= last_window.cost;
  }

  size_t chrom_start_index = find_epiread_index_from_offset(last_window.start_byte, 
                                                              initial_epireads_size-1, 
                                                              epireads_metadata);
  
  // Get the id of the chrom to preprocess
  auto it = chrom_name_to_id.find(current_chrom);
  size_t chrom_name_id = it == chrom_name_to_id.end() ? INVALID_CHROM_NAME_ID : it->second;

  // Preprocess invalid windows once again
  total_cost += preprocess_chrom_parallel_on_range(false, false,
                            min_obs_per_cpg, window_size,
                            current_chrom, 
                            first_window_to_update_start, 
                            last_important_window_start_pos + 1,
                            chrom_start_index, epireads.size(),
                            last_chrom_max_epiread_len, epireads,
                            epireads_metadata, windows, chrom_name_id);

}

static size_t
filter_leftover_windows(const string &epireads_file,
                        vector<epiread> &epireads,
                        vector<EpireadMetadata> &epireads_metadata,
                        vector<WindowMetadata> &windows, 
                        const size_t last_owned_byte, size_t &total_cost,
                        const size_t min_obs_per_cpg,
                        const size_t window_size,
                        size_t &last_chrom_max_epiread_len,
                        const unordered_map<string, size_t> &chrom_name_to_id){

  // If last window keeps using the shared epiread, more overlapping is required
  if(windows[windows.size()-1].start_byte <= last_owned_byte){
    size_t shared_epiread_index = find_epiread_index_from_offset(last_owned_byte, 
                                                        epireads_metadata.size()-1, 
                                                        epireads_metadata);
    add_overlapping_to_avoid_shared_on_end(epireads_file, epireads, epireads_metadata, windows,
                                min_obs_per_cpg, window_size, shared_epiread_index, 
                                last_chrom_max_epiread_len, total_cost, chrom_name_to_id);
    return windows.size();
  }

  size_t i = windows.size()-1;
  
  // After this 'i' points to the first invalid window
  while (i > 0 && windows[i-1].start_byte > last_owned_byte) {
    total_cost -= windows[i].cost;
    i--;
  }
  total_cost -= windows[i].cost;
  
  // If last valid window uses the last epiread more overlapping is required
  if (windows[i-1].end_byte >= epireads_metadata[epireads_metadata.size()-1].end_byte){
    size_t shared_epiread_index = find_epiread_index_from_offset(last_owned_byte, 
                                                        epireads_metadata.size()-1, 
                                                        epireads_metadata);
    add_overlapping_to_avoid_shared_on_end(epireads_file, epireads, epireads_metadata, windows,
                                min_obs_per_cpg, window_size, shared_epiread_index, 
                                last_chrom_max_epiread_len, total_cost, chrom_name_to_id);
    return windows.size();
  }

  return i;

}

static WindowMetadata
communicate_last_valid_window(const vector<WindowMetadata> &windows,
                              const size_t last_valid_window_index,
                              const MPI_Datatype &MPI_WINDOW_METADATA,
                              const int rank, const int number_of_processes){

  MPI_Request request;
  WindowMetadata validation_window;

  // If i'm not the last process i will send the window to the next process
  if(rank < number_of_processes-1)
    MPI_Isend( &(windows[last_valid_window_index]), 1, MPI_WINDOW_METADATA, 
                rank+1, 1, MPI_COMM_WORLD, &request);

  // If i'm not the first process, receive the window from the previous process
  if(rank /* != 0 */)
    MPI_Recv( &validation_window, 1, MPI_WINDOW_METADATA, 
              rank-1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // Wait for the async-send to be completed
  if (rank < number_of_processes-1)
    MPI_Wait( &request , MPI_STATUS_IGNORE);

  return validation_window;
}

static bool
leading_window_is_not_valid(const size_t validation_window_start,
                            const WindowMetadata &candidate_window,
                            const vector<epiread> &epireads,
                            const vector<EpireadMetadata> &em,
                            const string candidate_chrom,
                            size_t &epireads_index) {

  size_t target_start = candidate_window.start_byte;

  // Look for the epiread where the window starts
  while(em[epireads_index].start_byte < target_start)
    epireads_index++;

  // If the epiread belongs to a diferent chrom, then window is valid
  if(epireads[epireads_index].chr != candidate_chrom){
    return false;
  }

  // If it belongs to the same chrom and the position is bigger than the validation position
  // Then the window is valid
  return (candidate_window.window_start <= validation_window_start);

}

static size_t
remove_leading_windows(const vector<WindowMetadata> &windows,
                        const vector<epiread> &epireads,
                        const vector<EpireadMetadata> &em,
                        const WindowMetadata &validation_window,
                        const size_t true_windows_size,
                        size_t &total_cost) {

  size_t window_index = 0, epireads_index = 0;
  size_t validation_window_start = validation_window.window_start;
  string current_chrom = epireads[0].chr;

  // Si la ventana NO contiene el epiread compartido (el primero de este proceso)
  // Entonces todas las ventanas son validas
  if(validation_window.end_byte <= em[0].start_byte){
    return window_index;
  }

  // Si SI contiene el epiread compartido
  // Entonces las ventanas no son validas mientras pertenezcan al mismo chrom y tengan una 'window_start' menor o igual al de la validation window
  while(window_index < true_windows_size && leading_window_is_not_valid(validation_window_start, windows[window_index], epireads, em, current_chrom, epireads_index)){
    total_cost -= windows[window_index].cost;
    window_index++;
  }

  return window_index;

}

static size_t 
gather_windows(const vector<WindowMetadata> &windows,
              const int my_number_of_windows, const size_t first_window,
              const size_t total_cost, const int number_of_processes,
              size_t &full_total_cost, vector<WindowMetadata> &all_windows,
              MPI_Datatype &MPI_WINDOW_METADATA, const int rank){

  int num_windows_per_process[number_of_processes];
  int displs[number_of_processes];
  int number_of_windows = 0;

  size_t num_of_windows;

  // Tell ROOT PROCESS how many windows will it receive
  MPI_Gather( &my_number_of_windows, 1, MPI_INT, 
              num_windows_per_process, 1, MPI_INT, 
              MPI_ROOT_PROCESS, MPI_COMM_WORLD);

  for (size_t i = 0; i < number_of_processes; i++)
  {
    number_of_windows += num_windows_per_process[i];
  }

  // Reduce the total cost on the root process
  MPI_Allreduce( &total_cost, &full_total_cost, 1, MPI_AINT, MPI_SUM, 
               MPI_COMM_WORLD);


  
  if(rank == MPI_ROOT_PROCESS) {
    // Allocate the buffer
    all_windows.resize(number_of_windows);

    num_of_windows = number_of_windows;

    // Calculate displacements
    int displ = 0;
    for (int i = 0; i < number_of_processes; i++)
    {
      displs[i] = displ;
      displ += num_windows_per_process[i];
    }
  }
  
  // Gather windows
  MPI_Gatherv( &(windows[first_window]) , my_number_of_windows, MPI_WINDOW_METADATA, 
                &(all_windows[0]), num_windows_per_process, displs, 
                MPI_WINDOW_METADATA, MPI_ROOT_PROCESS, MPI_COMM_WORLD);

  //B_cast number of windows
  MPI_Bcast(&num_of_windows, 1, MPI_AINT, MPI_ROOT_PROCESS, MPI_COMM_WORLD);

  return num_of_windows;

}

static bool
distribute_basic_amrs(const vector<BasicAmr> &amrs, const int number_of_processes,
              int amrs_per_process[], int displs[],
              int &my_number_of_amrs, vector<BasicAmr> &my_amrs,
              MPI_Datatype &MPI_AMR_DATATYPE, const int rank){

  size_t number_of_amrs = amrs.size();

  MPI_Bcast( &number_of_amrs , 1 , MPI_AINT , MPI_ROOT_PROCESS , MPI_COMM_WORLD);
  
  if(number_of_amrs == 0)
    return false;

  size_t amrs_div, amrs_rem;
  amrs_div = number_of_amrs / number_of_processes;
  amrs_rem = number_of_amrs % number_of_processes;

  displs[0] = 0;
  for (size_t i = 0; i < number_of_processes-1; i++){
    amrs_per_process[i] = (i < amrs_rem) ? amrs_div+1 : amrs_div;
    displs[i+1] = displs[i] + amrs_per_process[i];
  }
  amrs_per_process[number_of_processes-1] = (number_of_processes-1 < amrs_rem) ? amrs_div+1 : amrs_div;
  
  my_number_of_amrs = amrs_per_process[rank];

  // then each process allocates memory
  my_amrs.resize(my_number_of_amrs, BasicAmr());

  // finally data is actually distributed in a block fashion
  MPI_Scatterv( &amrs[0] , amrs_per_process , displs , MPI_AMR_DATATYPE , 
                &my_amrs[0] , my_number_of_amrs , MPI_AMR_DATATYPE , 
                MPI_ROOT_PROCESS , MPI_COMM_WORLD);

  return true;
}

static void
from_genomic_regions_to_basic_amrs_light(vector<GenomicRegion> &amrs, 
                                      vector<BasicAmr> &basic_amrs,
                                      unordered_map<string, size_t> &chrom_name_to_id){

  basic_amrs.resize(amrs.size(), BasicAmr());

  if (amrs.empty())
    return;

  size_t current_chrom_id = get_id_from_chrom(chrom_name_to_id, amrs[0].get_chrom());

  // Peel first iteration
  basic_amrs[0].chrom_name_id = current_chrom_id;
  basic_amrs[0].start_cpg = amrs[0].get_start();
  basic_amrs[0].end_cpg = amrs[0].get_end();

  for (size_t i = 1; i < amrs.size(); i++){
    if(!amrs[i-1].same_chrom(amrs[i]))
      current_chrom_id = get_id_from_chrom(chrom_name_to_id, amrs[i].get_chrom());

    basic_amrs[i].chrom_name_id = current_chrom_id;
    basic_amrs[i].start_cpg = amrs[i].get_start();
    basic_amrs[i].end_cpg = amrs[i].get_end();
  }
  
}

static void
from_basic_amrs_to_genomic_regions_light(vector<GenomicRegion> &amrs, vector<BasicAmr> &basic_amrs){
  for (size_t i = 0; i < amrs.size(); i++){
    amrs[i].set_start(basic_amrs[i].start_cpg);
    amrs[i].set_end(basic_amrs[i].end_cpg);
  }
}

static void
convert_coordinates(const vector<size_t> &cpgs,
                    BasicAmr &region)  {
  if (region.start_cpg >= cpgs.size() || region.end_cpg >= cpgs.size())
    throw runtime_error("could not convert:\n" + std::to_string(region.chrom_name_id) + " on cpgs " + std::to_string(region.start_cpg) + " to " + std::to_string(region.end_cpg));
  region.start_cpg = cpgs[region.start_cpg];
  region.end_cpg = cpgs[region.end_cpg];
}

static void
convert_coordinates(const vector<size_t> &cpgs,
                    BasicAmr &region,
                    const unordered_map<size_t, string> &chrom_id_to_name)  {
  if (region.start_cpg >= cpgs.size() || region.end_cpg >= cpgs.size())
    throw runtime_error("could not convert:\n" + get_chrom_from_id(chrom_id_to_name, region.chrom_name_id) + " on cpgs " + std::to_string(region.start_cpg) + " to " + std::to_string(region.end_cpg));
  region.start_cpg = cpgs[region.start_cpg];
  region.end_cpg = cpgs[region.end_cpg];
}

static bool
file_exists(const string file_path){
  struct stat f_info;
  // stat(...) return 0 on success
  return ( stat(file_path.c_str(), &f_info) == 0 );
}

static void
get_chrom_assuming(const string chroms_dir, const string fasta_suffix,
          const string chrom_name,  string &chrom) {

  string path = path_join(chroms_dir, chrom_name + "." + fasta_suffix);
  if(!file_exists(path))
    throw runtime_error("could not find chrom: " + chrom_name);
  chrom.clear();
  read_fasta_file(path, chrom_name, chrom);
  if (chrom.empty())
    throw runtime_error("could not find chrom: " + chrom_name);

}

static void
convert_coordinates_assuming(const bool VERBOSE, const string chroms_dir,
                    const string fasta_suffix, vector<BasicAmr> &amrs,
                    unordered_map<size_t, string> &chrom_id_to_name,
                    std::set<string> &short_chroms) {

  vector<size_t> cpgs;
  string chrom;
  size_t chrom_name_id = INVALID_CHROM_NAME_ID;
  string chrom_name;
  /************************ TIME MEASURE ***********************/
  #ifdef DEBUG
  program.subsections[5].subsections[0].subsections[2].start_section();
  #endif
  /************************ TIME MEASURE ***********************/
  for (size_t i = 0; i < amrs.size(); ++i) {
    if (amrs[i].chrom_name_id != chrom_name_id) {
      /************************ TIME MEASURE ***********************/
      #ifdef DEBUG
      program.subsections[5].subsections[0].subsections[2].end_section();
      program.subsections[5].subsections[0].subsections[2].store_time();
      program.subsections[5].subsections[0].subsections[1].start_section();
      #endif
      /************************ TIME MEASURE ***********************/
      chrom_name_id = amrs[i].chrom_name_id;
      chrom_name = get_chrom_from_id(chrom_id_to_name, chrom_name_id);
      get_chrom_assuming(chroms_dir, fasta_suffix, chrom_name, chrom);
      /************************ TIME MEASURE ***********************/
      #ifdef DEBUG
      program.subsections[5].subsections[0].subsections[1].end_section();
      program.subsections[5].subsections[0].subsections[1].store_time();
      program.subsections[5].subsections[0].subsections[2].start_section();
      #endif
      /************************ TIME MEASURE ***********************/
      collect_cpgs(chrom, cpgs);
      if (is_short_chrom(get_chrom_from_id(chrom_id_to_name, chrom_name_id), short_chroms))
        amrs[i].end_cpg = cpgs.size() - 1;
      if (VERBOSE)
        cerr << "CONVERTING: " << chrom_name << endl;
    }
    convert_coordinates(cpgs, amrs[i], chrom_id_to_name);
  }
  /************************ TIME MEASURE ***********************/
  #ifdef DEBUG
  program.subsections[5].subsections[0].subsections[2].end_section();
  program.subsections[5].subsections[0].subsections[2].store_time();
  #endif
  /************************ TIME MEASURE ***********************/
}

static void
convert_coordinates_wrapper(const int rank, const int number_of_processes, MPI_Datatype MPI_AMR_DATATYPE,
                    const bool VERBOSE, const string chroms_dir,
                    const string fasta_suffix, vector<GenomicRegion> &amrs,
                    unordered_map<string, size_t> &chrom_name_to_id,
                    unordered_map<size_t, string> &chrom_id_to_name, std::set<string> &short_chroms){

  vector<BasicAmr> amrs_to_communicate;
  vector<BasicAmr> my_amrs;

  int amrs_per_process[number_of_processes];
  int displs[number_of_processes];
  int my_number_of_amrs;

  if (rank == MPI_ROOT_PROCESS)
    from_genomic_regions_to_basic_amrs_light(amrs, amrs_to_communicate, chrom_name_to_id);

  if (!distribute_basic_amrs(amrs_to_communicate, number_of_processes, amrs_per_process, displs, my_number_of_amrs, my_amrs, MPI_AMR_DATATYPE, rank))
    return;

  convert_coordinates_assuming(VERBOSE, chroms_dir, fasta_suffix, my_amrs, chrom_id_to_name, short_chroms);

  gather_basic_amrs(my_amrs, my_number_of_amrs, MPI_AMR_DATATYPE, amrs_per_process, displs, amrs_to_communicate);

  if (rank == MPI_ROOT_PROCESS)
    from_basic_amrs_to_genomic_regions_light(amrs, amrs_to_communicate);

}

static size_t
identify_chrom(std::ifstream &in, string &chrom_name, size_t &lines, 
                char *buffer, const size_t INPUT_BUFFER_SIZE){

  buffer[0] = '\0';

  size_t bytes_read;

  lines = 0;

  while (!in.eof() && buffer[0] != '>'){

    in.getline(buffer, INPUT_BUFFER_SIZE);
    bytes_read = in.gcount();
    if (bytes_read == static_cast<int>(INPUT_BUFFER_SIZE))
      throw runtime_error("Line in " + chrom_name + "\nexceeds max length: "
                          + toa(INPUT_BUFFER_SIZE));

    // correct for dos carriage returns before newlines
    if (buffer[strlen(buffer) - 1] == '\r')
      buffer[strlen(buffer) - 1] = '\0';

    // Count the new line
    lines++;
    
    in.peek();
  }

  // If loop ended because a new chrom was found
  // Then the last line does not count
  if (!in.eof()){
    lines--;
    string newname(&(buffer[1]));
    chrom_name.swap(newname);
  }

  return bytes_read;
  
}

void
read_fasta_file_parallel(const int rank, 
                        const int number_of_processes,
                        const string &chrom_file,
                        const size_t chrom_file_index,
                        vector<ChromInfo> &chroms_to_files,
                        unordered_map<string, size_t> &chrom_name_to_id){

  string line;
  size_t previous_position, current_position;
  size_t process_start, process_size, process_size_with_overlapping, process_end;
  const size_t overlapping_size = 1;
  bool already_started = false;
  ChromInfo chrom;
  size_t lines;
  string chrom_name("chr0");

  static const size_t INPUT_BUFFER_SIZE = 1000000;

  char buffer[INPUT_BUFFER_SIZE + 1];

  // Open the file
  std::ifstream in(chrom_file.c_str(), std::ios::binary);
  if (!in)
    throw runtime_error("cannot open input file " + chrom_file);

  // Get size of the file
  in.seekg(0, std::ios::end);
  long filesize = in.tellg();
  
  // Divide the file in blocks
  size_t size_div, size_rem;
  size_div = filesize / number_of_processes;
  size_rem = filesize % number_of_processes; 

  // Calculate the start of the block and its non-overlapping size
  if (rank < size_rem){
    process_start = rank * (size_div + 1); 
    process_size = (size_div + 1);
  } else {
    process_start = (rank * size_div) + size_rem;
    process_size = size_div;
  }

  // Process 'i' should share its first byte with process 'i-1' 
  // So add overlapping of 1 byte on process i-1
  if ( (process_start + process_size + overlapping_size) <= filesize )
    process_size_with_overlapping = process_size + overlapping_size;
  else
    process_size_with_overlapping = filesize - process_start;

  process_end = process_start + process_size_with_overlapping;

  // Go to process initial position
  in.seekg(process_start);
  
  // If not the first process, 
  // avoid the half line you spawned in
  // and jump to a fresh new line
  if (rank /* != 0*/){
    in.getline(buffer, INPUT_BUFFER_SIZE);
    if (in.gcount() == static_cast<int>(INPUT_BUFFER_SIZE))
      throw runtime_error("Line in " + chrom_file + "\nexceeds max length: "
                          + toa(INPUT_BUFFER_SIZE));
    in.peek();
  }

  // Skip to the start of the next chrom
  size_t bytes_read = identify_chrom(in, chrom_name, lines, buffer, INPUT_BUFFER_SIZE);
    
  current_position = in.tellg();

  // After getting in place
  // identify new chroms on your block
  while (!in.eof() && current_position - bytes_read <= process_end){

    // Create your information structure
    try{
      chrom = ChromInfo(get_id_from_chrom(chrom_name_to_id, chrom_name), chrom_file_index, current_position);
    } catch (const runtime_error &e){
      chrom = ChromInfo(INVALID_CHROM_NAME_ID, chrom_file_index, current_position);
    }
    
    // Process chrom till end
    bytes_read = identify_chrom(in, chrom_name, lines, buffer, INPUT_BUFFER_SIZE);

    // Complete information
    chrom.lines = lines;

    chroms_to_files.push_back(chrom);

    // Update position
    current_position = in.tellg();
  }
  
}

void
identify_and_read_chromosomes_parallel(const int rank, 
                              const int number_of_processes,
                              const string &chrom_file,
                              const string fasta_suffix,
                              vector<string> &the_files,
                              vector<ChromInfo> &chroms_to_files,
                              unordered_map<string, size_t> &chrom_name_to_id){

  if (isdir(chrom_file.c_str())) {
    read_dir(chrom_file, fasta_suffix, the_files);
  }
  else
    the_files.push_back(chrom_file);

  // If all the chroms are stored in the same file
  if(the_files.size() == 1){
    read_fasta_file_parallel(rank, number_of_processes, 
                      the_files[0], 0, chroms_to_files, chrom_name_to_id);
  }

  // If there are several files storing chroms
  // Distribute them among the processes 
  // in a round rovin style
  if(the_files.size() > 1){
    for (size_t i = rank; i < the_files.size(); i+=number_of_processes)
    {
      vector<string> names, sequences;
      read_fasta_file(the_files[i], names, sequences);
      for (auto &name : names){
        ChromInfo chrom;
        try{
          chrom = ChromInfo(get_id_from_chrom(chrom_name_to_id, name), i, 0);
        } catch (const runtime_error &e){
          chrom = ChromInfo(INVALID_CHROM_NAME_ID, i, 0);
        }
        chroms_to_files.push_back(chrom);
      }
    }
    
  }
}

void 
get_displs_from_counts(vector<int> &chrom_counts, vector<int> &displs) {

  // Get space
  displs.resize(chrom_counts.size());

  // Calculate
  displs[0] = 0;
  for (size_t i = 1; i < chrom_counts.size(); i++)
    displs[i] = displs[i-1] + chrom_counts[i-1];
  
}

void
build_chroms_to_files_hashmap(vector<ChromInfo> &all_chroms_to_files, 
                            unordered_map<size_t, ChromInfo> &chrom_map){

  for (auto &ch_info : all_chroms_to_files){
    chrom_map[ch_info.chrom_name_id] = ch_info;
  }
}

void
share_chrom_info(const int number_of_processes, 
                vector<ChromInfo> &chroms_to_files, 
                unordered_map<size_t, ChromInfo> &chrom_map,
                MPI_Datatype MPI_CHINFO_DATATYPE){
  
  const int my_chroms = chroms_to_files.size();
  vector<int> chrom_counts(number_of_processes);
  int total_size = 0;

  // All processes share how many chroms they have discovered
  MPI_Allgather(&my_chroms, 1, MPI_INT, 
                &(chrom_counts[0]), 1, MPI_INT, 
                MPI_COMM_WORLD);

  // They all make space for them
  for (auto &count : chrom_counts){
    total_size += count;
  }

  vector<ChromInfo> all_chroms_to_files(total_size);

  vector<int> displs;
  get_displs_from_counts(chrom_counts, displs);

  // Share information all-to-all
  MPI_Allgatherv(&(chroms_to_files[0]), my_chroms, MPI_CHINFO_DATATYPE, 
                &(all_chroms_to_files[0]), &(chrom_counts[0]), &(displs[0]), 
                MPI_CHINFO_DATATYPE, MPI_COMM_WORLD);

  // Build a map for easy access
  build_chroms_to_files_hashmap(all_chroms_to_files, chrom_map);
}

static void 
get_chrom_with_info(const bool VERBOSE, const size_t chrom_name_id, 
                vector<string> &chrom_files,
                const unordered_map<size_t, ChromInfo> &chrom_map, 
                string &chrom,
                unordered_map<size_t, string> &chrom_id_to_name){
  const unordered_map<size_t, ChromInfo>::const_iterator 
                                    fn(chrom_map.find(chrom_name_id));
  if(fn == chrom_map.end())
    throw runtime_error("could not find chrom: " + get_chrom_from_id(chrom_id_to_name, chrom_name_id) + " on map\n");
  string chrom_name = get_chrom_from_id(chrom_id_to_name, fn->first);
  chrom.clear();
  if(fn->second.ocupies_whole_file())
    read_fasta_file(chrom_files[fn->second.file_idx], chrom_name, chrom);
  else
    read_fasta_file_from_offsets(chrom_files[fn->second.file_idx], 
                                  fn->second.chrom_start_offset,
                                  fn->second.lines,
                                  chrom_name, chrom);

  if (chrom.empty())
    throw runtime_error("could not find chrom: " + chrom_name + "on file" + chrom_files[fn->second.file_idx]);

}

static void
convert_coordinates_with_info(const bool VERBOSE,
                              vector<string> &chrom_files,
                              unordered_map<size_t, ChromInfo> &chrom_map, 
                              vector<BasicAmr> &amrs,
                              unordered_map<size_t, string> &chrom_id_to_name,
                              std::set<string> &short_chroms) {

  vector<size_t> cpgs;
  string chrom;
  size_t chrom_name_id = INVALID_CHROM_NAME_ID;
  /************************ TIME MEASURE ***********************/
  #ifdef DEBUG
  program.subsections[5].subsections[0].subsections[2].start_section();
  #endif
  /************************ TIME MEASURE ***********************/
  for (size_t i = 0; i < amrs.size(); ++i) {
    if (amrs[i].chrom_name_id != chrom_name_id) {
      /************************ TIME MEASURE ***********************/
      #ifdef DEBUG
      program.subsections[5].subsections[0].subsections[2].end_section();
      program.subsections[5].subsections[0].subsections[2].store_time();
      program.subsections[5].subsections[0].subsections[1].start_section();
      #endif
      /************************ TIME MEASURE ***********************/
      chrom_name_id = amrs[i].chrom_name_id;
      get_chrom_with_info(VERBOSE, chrom_name_id, chrom_files, chrom_map, chrom, chrom_id_to_name);
      /************************ TIME MEASURE ***********************/
      #ifdef DEBUG
      program.subsections[5].subsections[0].subsections[1].end_section();
      program.subsections[5].subsections[0].subsections[1].store_time();
      program.subsections[5].subsections[0].subsections[2].start_section();
      #endif
      /************************ TIME MEASURE ***********************/
      collect_cpgs(chrom, cpgs);
      if (is_short_chrom(get_chrom_from_id(chrom_id_to_name, chrom_name_id), short_chroms))
        amrs[i].end_cpg = cpgs.size() - 1;
      if (VERBOSE)
        cerr << "CONVERTING: " << get_chrom_from_id(chrom_id_to_name, chrom_name_id) << endl;
    }
    convert_coordinates(cpgs, amrs[i], chrom_id_to_name);
  }
  /************************ TIME MEASURE ***********************/
  #ifdef DEBUG
  program.subsections[5].subsections[0].subsections[2].end_section();
  program.subsections[5].subsections[0].subsections[2].store_time();
  #endif
  /************************ TIME MEASURE ***********************/
}

static void
convert_coordinates_parallel(const int rank, const int number_of_processes, 
                    MPI_Datatype MPI_AMR_DATATYPE, MPI_Datatype MPI_CHINFO_DATATYPE,
                    const bool VERBOSE, const string chroms_dir,
                    const string fasta_suffix, vector<GenomicRegion> &amrs,
                    unordered_map<string, size_t> &chrom_name_to_id,
                    unordered_map<size_t, string> &chrom_id_to_name, std::set<string> &short_chroms){

  vector<string> chrom_files;
  vector<ChromInfo> chroms_to_files;  
  vector<BasicAmr> amrs_to_communicate;
  vector<BasicAmr> my_amrs;

  int amrs_per_process[number_of_processes];
  int displs[number_of_processes];
  int my_number_of_amrs;

  /************************ TIME MEASURE ***********************/
  #ifdef DEBUG
  program.subsections[5].subsections[0].subsections[0].start_section();
  #endif
  /************************ TIME MEASURE ***********************/
  
  identify_and_read_chromosomes_parallel(rank, number_of_processes, chroms_dir, 
                                    fasta_suffix, chrom_files, chroms_to_files,
                                    chrom_name_to_id);

  if (VERBOSE && rank == MPI_ROOT_PROCESS)
    cerr << "CHROMS:\t" << chrom_files.size() << endl;

  // Create a structure to hold chrom data
  unordered_map<size_t, ChromInfo> chrom_map;
  share_chrom_info(number_of_processes, chroms_to_files, chrom_map, MPI_CHINFO_DATATYPE);

  /************************ TIME MEASURE ***********************/
  #ifdef DEBUG
  program.subsections[5].subsections[0].subsections[0].end_section();
  #endif
  /************************ TIME MEASURE ***********************/

  if (rank == MPI_ROOT_PROCESS)
    from_genomic_regions_to_basic_amrs_light(amrs, amrs_to_communicate, chrom_name_to_id);

  if (!distribute_basic_amrs(amrs_to_communicate, number_of_processes, amrs_per_process, displs, my_number_of_amrs, my_amrs, MPI_AMR_DATATYPE, rank))
    return;

  convert_coordinates_with_info(VERBOSE, chrom_files, chrom_map, my_amrs, chrom_id_to_name, short_chroms);

  gather_basic_amrs(my_amrs, my_number_of_amrs, MPI_AMR_DATATYPE, amrs_per_process, displs, amrs_to_communicate);

  if (rank == MPI_ROOT_PROCESS)
    from_basic_amrs_to_genomic_regions_light(amrs, amrs_to_communicate);
}

static void
process_last_incomplete_batch(const int rank,
                              const int number_of_processes,
                              const size_t number_of_windows,
                              const size_t BATCH_SIZE,
                              MPI_Win &windows_metadata_window,
                              MPI_Win &amrs_window,
                              MPI_Datatype &MPI_WINDOW_DATATYPE,
                              MPI_Datatype &MPI_AMR_DATATYPE,
                              const string reads_file,
                              const size_t window_size,
                              const EpireadStats &epistat){
  
  // Calculate the number of windows in the batch
  size_t number_of_windows_on_last_batch = number_of_windows % BATCH_SIZE;
  size_t last_batch_initial_index = number_of_windows - number_of_windows_on_last_batch;

  // Calculate how many should I get and from where
  size_t n_div, n_resto, local_start, local_size;
  n_div = number_of_windows_on_last_batch / number_of_processes;
  n_resto = number_of_windows_on_last_batch % number_of_processes;
  if (rank < n_resto){
    local_start = rank * (n_div + 1); 
    local_size = n_div + 1;
  } else {
    local_start = (rank * n_div) + n_resto;
    local_size = n_div;
  }

  // If I get no windows then get out
  if (local_size == 0)
    return;

  // Resize buffers
  vector<WindowMetadata> window_buffer(local_size);
  vector<BasicAmr> amr_buffer;

  // Get windows
  MPI_Win_lock(MPI_LOCK_SHARED, MPI_ROOT_PROCESS, 
              0, windows_metadata_window);
  MPI_Get(&window_buffer[0], local_size, MPI_WINDOW_DATATYPE, 
        MPI_ROOT_PROCESS, last_batch_initial_index + local_start, local_size, 
        MPI_WINDOW_DATATYPE, windows_metadata_window);
  MPI_Win_unlock(MPI_ROOT_PROCESS, windows_metadata_window);

  // Process windows
  process_windows(amr_buffer, window_buffer, local_size, reads_file, window_size, epistat);

  // Write Results
  MPI_Win_lock(MPI_LOCK_SHARED, MPI_ROOT_PROCESS, 0, amrs_window);
  MPI_Put(&amr_buffer[0], local_size, MPI_AMR_DATATYPE, 
          MPI_ROOT_PROCESS, last_batch_initial_index + local_start, 
          local_size, MPI_AMR_DATATYPE, amrs_window);
  MPI_Win_unlock(MPI_ROOT_PROCESS, amrs_window);

}

static void
get_batches_dynamically(const int rank,
                        MPI_Win &index_window,
                        MPI_Win &windows_metadata_window,
                        MPI_Win &amrs_window,
                        const size_t BATCH_SIZE,
                        const size_t NUMBER_OF_WINDOWS,
                        MPI_Datatype &MPI_WINDOW_DATATYPE,
                        MPI_Datatype &MPI_AMR_DATATYPE,
                        const string reads_file,
                        const size_t window_size,
                        const EpireadStats &epistat){

  const size_t POSITION_LIMIT = NUMBER_OF_WINDOWS - BATCH_SIZE;
  size_t elements_of_batch = BATCH_SIZE;
  size_t current_position;

  vector<WindowMetadata> window_buffer(BATCH_SIZE);
  vector<BasicAmr> amr_buffer;

  #ifdef TIMEPROFILE
  auto p1 = profile_clock_t::now();
  #endif

  // Get initial position
  MPI_Win_lock(MPI_LOCK_SHARED, MPI_ROOT_PROCESS, 0, index_window);
  MPI_Fetch_and_op(&BATCH_SIZE, &current_position, MPI_AINT, 
                  MPI_ROOT_PROCESS, 0, MPI_SUM, index_window);
  MPI_Win_unlock(MPI_ROOT_PROCESS, index_window);

  // Process windows, one batch at a time
  while(current_position < NUMBER_OF_WINDOWS){

    // If it is the last batch, 
    // and there are no BATCH_SIZE elements to process
    if(current_position > POSITION_LIMIT){
      // Resize to the actual amount of elements to process
      elements_of_batch = NUMBER_OF_WINDOWS - current_position;
      window_buffer.resize(elements_of_batch);
    }

    // Get batch of elements
    MPI_Win_lock(MPI_LOCK_SHARED, MPI_ROOT_PROCESS, 
                0, windows_metadata_window);
    MPI_Get(&window_buffer[0], elements_of_batch, MPI_WINDOW_DATATYPE, 
          MPI_ROOT_PROCESS, current_position, elements_of_batch, 
          MPI_WINDOW_DATATYPE, windows_metadata_window);
    MPI_Win_unlock(MPI_ROOT_PROCESS, windows_metadata_window);

    // Clean previous results
    amr_buffer.clear();
    
    // Compute batch
    process_windows(amr_buffer, window_buffer, elements_of_batch, reads_file, window_size, epistat);

    // Write batch of results
    MPI_Win_lock(MPI_LOCK_SHARED, MPI_ROOT_PROCESS, 0, amrs_window);
    MPI_Put(&amr_buffer[0], elements_of_batch, MPI_AMR_DATATYPE, 
            MPI_ROOT_PROCESS, current_position, 
            elements_of_batch, MPI_AMR_DATATYPE, amrs_window);
    MPI_Win_unlock(MPI_ROOT_PROCESS, amrs_window);

    #ifdef TIMEPROFILE
    auto p2 = profile_clock_t::now();

    auto dur_in_secs = profile_duration_t(p2-p1);
    std::cout << "Rank: " << rank << " ||| Position: " << current_position << " ||| Time: " << dur_in_secs.count() << std::endl;
    p1 = profile_clock_t::now();
    #endif

    // Get next position
    MPI_Win_lock(MPI_LOCK_SHARED, MPI_ROOT_PROCESS, 
                0, index_window);
    MPI_Fetch_and_op(&BATCH_SIZE, &current_position, MPI_AINT, 
                    MPI_ROOT_PROCESS, 0, MPI_SUM, index_window);
    MPI_Win_unlock(MPI_ROOT_PROCESS, index_window);
  }
}

static void 
process_windows_dynamically(const int rank,
                            const int number_of_processes,
                            const size_t number_of_windows, 
                            size_t number_of_costly_batches,
                            const size_t BATCH_SIZE,
                            vector<WindowMetadata> &windows, 
                            vector<BasicAmr> &amrs,
                            MPI_Datatype &MPI_WINDOW_DATATYPE, 
                            MPI_Datatype &MPI_AMR_DATATYPE,
                            const string reads_file, 
                            const size_t window_size, 
                            const EpireadStats &epistat){

  size_t COSTLY_BATCH_SIZE = SMALL_BATCH_SIZE;
  size_t windows_limit = number_of_windows - (number_of_windows % BATCH_SIZE);


  size_t costly_index = 0;
  MPI_Win normal_index_window, costly_index_window,
          windows_metadata_window, amrs_window;

  // Get the number of costly batches in all the processes
  MPI_Bcast(&number_of_costly_batches, 1, MPI_AINT, 
            MPI_ROOT_PROCESS, MPI_COMM_WORLD);

  const size_t COSTLY_WINDOWS = number_of_costly_batches * BATCH_SIZE;
  size_t normal_index = COSTLY_WINDOWS;

  // Create window for position counter
  MPI_Win_create(&normal_index, 1 * sizeof(size_t), sizeof(size_t), 
                MPI_INFO_NULL, MPI_COMM_WORLD, &normal_index_window);

  // Create window for small batch position counter
  MPI_Win_create(&costly_index, 1 * sizeof(size_t), sizeof(size_t), 
                MPI_INFO_NULL, MPI_COMM_WORLD, &costly_index_window);

  // Create window for vector of windows
  MPI_Win_create(&windows[0], windows.size() * sizeof(WindowMetadata), sizeof(WindowMetadata), 
                MPI_INFO_NULL, MPI_COMM_WORLD, &windows_metadata_window);

  // Create window for vector of amrs
  MPI_Win_create(&amrs[0], amrs.size() * sizeof(BasicAmr), sizeof(BasicAmr),
                MPI_INFO_NULL, MPI_COMM_WORLD, &amrs_window);

  // First, process last incomplete batch 
  process_last_incomplete_batch(rank, number_of_processes, number_of_windows,
                                BATCH_SIZE, windows_metadata_window, amrs_window,
                                MPI_WINDOW_DATATYPE, MPI_AMR_DATATYPE, 
                                reads_file, window_size, epistat);

  // Process costly windows first
  get_batches_dynamically(rank, costly_index_window, windows_metadata_window,
                          amrs_window, COSTLY_BATCH_SIZE, COSTLY_WINDOWS,
                          MPI_WINDOW_DATATYPE, MPI_AMR_DATATYPE, 
                          reads_file, window_size, epistat);

  // Process remaining normal windows
  get_batches_dynamically(rank, normal_index_window, windows_metadata_window,
                          amrs_window, BATCH_SIZE, windows_limit,
                          MPI_WINDOW_DATATYPE, MPI_AMR_DATATYPE, 
                          reads_file, window_size, epistat);

  // Free windows
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Win_free(&normal_index_window);
  MPI_Win_free(&costly_index_window);
  MPI_Win_free(&windows_metadata_window);
  MPI_Win_free(&amrs_window);

  return;
}

/*
static void
move_data_pipelined(const size_t number_of_windows, 
                    MPI_Datatype &MPI_WINDOW_DATATYPE, MPI_Datatype &MPI_AMR_DATATYPE,
                    const string reads_file, const size_t window_size, const EpireadStats &epistat){

  static const size_t BATCH_SIZE = 1000;
  static const size_t INDEX_LIMIT = number_of_windows - BATCH_SIZE;
  size_t index = 0;
  size_t previous_position, current_position, next_position;
  size_t elements_of_batch;

  size_t current_buffer_index = 0;

  vector<WindowMetadata> window_buffers[2];
  window_buffers[0].resize(BATCH_SIZE);
  window_buffers[1].resize(BATCH_SIZE);

  vector<BasicAmr> amr_buffers[2];
  amr_buffers[0].resize(BATCH_SIZE);
  amr_buffers[1].resize(BATCH_SIZE);

  MPI_Win index_window, windows_metadata_window, amrs_window;

  // Create window for position counter
  MPI_Win_create(&index, 1 * sizeof(size_t), sizeof(size_t), 
                MPI_INFO_NULL, MPI_COMM_WORLD, &index_window);

  // Create window for vector of windows
  MPI_Win_create(NULL, number_of_windows * sizeof(WindowMetadata), sizeof(WindowMetadata), 
                MPI_INFO_NULL, MPI_COMM_WORLD, &windows_metadata_window);

  // Create window for vector of amrs
  MPI_Win_create(NULL, number_of_windows * sizeof(BasicAmr), sizeof(BasicAmr),
                MPI_INFO_NULL, MPI_COMM_WORLD, &amrs_window);

  // Get initial position
  MPI_Win_lock(MPI_LOCK_SHARED, MPI_ROOT_PROCESS, 0, index_window);
  MPI_Fetch_and_op(&BATCH_SIZE, &current_position, MPI_AINT, MPI_ROOT_PROCESS, 0, MPI_SUM, index_window);
  MPI_Win_unlock(MPI_ROOT_PROCESS, index_window);

  // If position out of index
  if (current_position >= number_of_windows){
    // Free windows
    MPI_Win_free(&index_window);
    MPI_Win_free(&windows_metadata_window);
    MPI_Win_free(&amrs_window);

    return;
  }

  // Get position of next batch
  // Get batch from current position
  elements_of_batch = current_position < INDEX_LIMIT ? BATCH_SIZE : number_of_windows - current_position;
  MPI_Win_lock(MPI_LOCK_SHARED, MPI_ROOT_PROCESS, 0, index_window);
  MPI_Win_lock(MPI_LOCK_SHARED, MPI_ROOT_PROCESS, 0, windows_metadata_window);
  MPI_Fetch_and_op(&BATCH_SIZE, &next_position, MPI_AINT, MPI_ROOT_PROCESS, 0, MPI_SUM, index_window);
  MPI_Get(&(window_buffers[current_buffer_index][0]), elements_of_batch, MPI_WINDOW_DATATYPE, 
          MPI_ROOT_PROCESS, current_position, 
          elements_of_batch, MPI_WINDOW_DATATYPE, windows_metadata_window);
  MPI_Win_unlock(MPI_ROOT_PROCESS, index_window);
  MPI_Win_unlock(MPI_ROOT_PROCESS, windows_metadata_window);
  

  MPI_Win_lock(MPI_LOCK_SHARED, MPI_ROOT_PROCESS, 0, amrs_window);


  while(next_position < number_of_windows){

      previous_position = current_position;
      current_position = next_position;

      // Lock read windows
      MPI_Win_lock(MPI_LOCK_SHARED, MPI_ROOT_PROCESS, 0, index_window);
      MPI_Win_lock(MPI_LOCK_SHARED, MPI_ROOT_PROCESS, 0, windows_metadata_window);

      // Read new index 
      MPI_Fetch_and_op(&BATCH_SIZE, &next_position, MPI_AINT, MPI_ROOT_PROCESS, 0, MPI_SUM, index_window);

      // Read new batch from current
      elements_of_batch = current_position < INDEX_LIMIT ? BATCH_SIZE : number_of_windows - current_position;
      MPI_Get(&(window_buffers[current_buffer_index^1][0]), elements_of_batch, MPI_WINDOW_DATATYPE, 
              MPI_ROOT_PROCESS, current_position, 
              elements_of_batch, MPI_WINDOW_DATATYPE, windows_metadata_window);

      // process prev batch
      process_windows(amr_buffers[current_buffer_index], window_buffers[current_buffer_index], BATCH_SIZE, reads_file, window_size, epistat);

      MPI_Win_unlock(MPI_ROOT_PROCESS, index_window);
      MPI_Win_unlock(MPI_ROOT_PROCESS, windows_metadata_window);

      // write prev batch results
      MPI_Win_unlock(MPI_ROOT_PROCESS, amrs_window);
      MPI_Win_lock(MPI_LOCK_SHARED, MPI_ROOT_PROCESS, 0, amrs_window);
      MPI_Put(&(amr_buffers[current_buffer_index][0]), BATCH_SIZE, MPI_AMR_DATATYPE, 
              MPI_ROOT_PROCESS, previous_position, 
              BATCH_SIZE, MPI_AMR_DATATYPE, amrs_window);

      // Change current buffer for next iteration
      current_buffer_index ^= 1;

  }

  MPI_Win_unlock(MPI_ROOT_PROCESS, amrs_window);

  // Process current
  window_buffers[current_buffer_index].resize(elements_of_batch);
  process_windows(amr_buffers[current_buffer_index], window_buffers[current_buffer_index], elements_of_batch, reads_file, window_size, epistat);

  // Write current
  MPI_Win_lock(MPI_LOCK_SHARED, MPI_ROOT_PROCESS, 0, amrs_window);
  MPI_Put(&(amr_buffers[current_buffer_index][0]), BATCH_SIZE, MPI_AMR_DATATYPE, 
          MPI_ROOT_PROCESS, current_position, 
          elements_of_batch, MPI_AMR_DATATYPE, amrs_window);
  MPI_Win_unlock(MPI_ROOT_PROCESS, amrs_window);

  // Free windows
  MPI_Win_free(&index_window);
  MPI_Win_free(&windows_metadata_window);
  MPI_Win_free(&amrs_window);

  return;
}
*/

static void 
swap_windows(vector<WindowMetadata> &windows,
            vector<size_t> &swap_info,
            const size_t total_cost,
            const size_t total_number_of_windows,
            const size_t BATCH_SIZE){

    size_t index_to_move = 0;
    const size_t AVG_COST = (double) total_cost / (double) total_number_of_windows * BATCH_SIZE;
    const size_t COST_LIMIT = AVG_COST * 5;
    const size_t WINDOW_LIMIT = windows.size() < BATCH_SIZE ? 0 : windows.size() - BATCH_SIZE;

    // For each batch to compute
    // But the last uncomplete batch
    for (size_t i = 0; i <= WINDOW_LIMIT; i+=BATCH_SIZE){
      size_t limit = i+BATCH_SIZE;
      size_t current_cost = 0;
      // Compute its cost
      for (size_t j = i; j < limit; j++){
        current_cost += windows[j].cost;
      }
      // If the batch is an outlier
      if (current_cost > COST_LIMIT){
        // Move the batch
        for(size_t ind = 0; ind < BATCH_SIZE; ind++){
          std::swap(windows[i+ind], windows[index_to_move + ind]);
        }
        swap_info.push_back(i);
        index_to_move += BATCH_SIZE;
      }
    }
}

static void 
unswap_basic_amrs(vector<BasicAmr> &basic_amrs,
              vector<size_t> &swap_info,
              const size_t BATCH_SIZE){

  for(size_t i = swap_info.size(); i > 0; i--){
      size_t swap_index_1 = swap_info[i-1];
      size_t swap_index_2 = (i-1) * BATCH_SIZE;
      for (size_t j = 0; j < BATCH_SIZE; j++){
        std::swap(basic_amrs[swap_index_1+j], basic_amrs[swap_index_2 + j]); 
      }
      
  }

}

static size_t
get_batch_size(const size_t number_of_windows, const size_t total_cost, 
              const int number_of_threads, const int number_of_procs){

  // Get ideal batch size
  const size_t mean_cost = total_cost / number_of_windows;
  #ifdef DEBUG
  printf("Coste medio es : %lu\n", mean_cost);
  #endif

  size_t ideal_batch_size = BATCH_SIZE_CONSTANT * number_of_threads / mean_cost;
  #ifdef DEBUG
  printf("El batch ideal es : %lu\n", ideal_batch_size);
  #endif

  // Round to make it multiple of SMALL_BATCH
  ideal_batch_size +=  SMALL_BATCH_SIZE - (ideal_batch_size % SMALL_BATCH_SIZE);
  #ifdef DEBUG
  printf("Tras el redondeo, el batch ideal es : %lu\n", ideal_batch_size);
  #endif

  // Grant a minimun amount of blocks to each process
  // To get a minimun decent granularity
  if ( (number_of_windows / ideal_batch_size) < (MIN_BLOCKS_PER_PROC * number_of_procs) ){
    
    ideal_batch_size = number_of_windows / (MIN_BLOCKS_PER_PROC * number_of_procs);

    // Make sure it is a multiple of SMALL_BATCH_SIZE
    size_t mod_to_small_batch = ideal_batch_size % SMALL_BATCH_SIZE;
    ideal_batch_size = mod_to_small_batch == ideal_batch_size ? SMALL_BATCH_SIZE : ideal_batch_size - mod_to_small_batch;
    #ifdef DEBUG
    printf("No se cumplen las restricciones, así que el batch ideal se ha cambiado a : %lu\n", ideal_batch_size);
    #endif
  }

  return ideal_batch_size;

}
