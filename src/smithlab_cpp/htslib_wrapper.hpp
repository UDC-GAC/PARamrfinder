/*
 * Part of SMITHLAB_CPP software
 *
 * Copyright (C) 2019 Meng Zhou and Andrew Smith
 *
 * Authors: Meng Zhou and Andrew Smith
 *
 * This is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 */

#ifndef HTSLIB_WRAPPER_HPP
#define HTSLIB_WRAPPER_HPP

#include "smithlab_utils.hpp"
#include "sam_record.hpp"

#include <string>
#include <vector>
#include <fstream>

extern "C" {
#include <htslib/sam.h>
#include <htslib/hts.h>
}

extern "C" {char check_htslib_wrapper();}

class SAMReader {
public:
  SAMReader(const std::string &filename);
  ~SAMReader();

  operator bool() const {return good;}

  bool get_sam_record(sam_rec &sr);

  std::string get_header() const;

private:

  // data
  std::string filename;
  bool good;

  htsFile* hts;
  bam_hdr_t *hdr;
  bam1_t *b;
};

SAMReader &
operator>>(SAMReader& sam_stream, sam_rec &samr);

#endif
