/*    Copyright (C) 2014 University of Southern California and
 *                       Andrew D. Smith and Fang Fang and Benjamin Decato
 *
 *    Authors: Fang Fang and Benjamin Decato and Andrew D. Smith
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

#include "EpireadStats.hpp"

#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <numeric>
#include <limits>
#include <iostream>
#include <unordered_map>

#include <gsl/gsl_sf.h>
#include <gsl/gsl_cdf.h>

using std::string;
using std::vector;
using std::isfinite;

static const double EPIREAD_STATS_TOLERANCE = 1e-20;
static const double PSEUDOCOUNT = 1e-10;

void cache_vector_logs(const vector<double> &a, vector<double> &log_a_meth, vector<double> &log_a_unmeth){
  for (size_t i = 0; i < a.size(); ++i){
    log_a_meth[i] = log(a[i]);
    assert(isfinite(log_a_meth[i]));
    log_a_unmeth[i] = log(1 - a[i]);
    assert(isfinite(log_a_unmeth[i]));
  }
}

inline bool
is_meth(const epiread &r, const size_t pos) {return (r.seq[pos] == 'C');}

inline bool
un_meth(const epiread &r, const size_t pos) {return (r.seq[pos] == 'T');}

double
log_likelihood(const epiread &r, const vector<double> &a) {
  double ll = 0.0;
  for (size_t i = 0; i < r.seq.length(); ++i)
    if (is_meth(r, i) || un_meth(r, i)) {
      const double val = (is_meth(r, i) ? a[r.pos + i] : (1.0 - a[r.pos + i]));
      assert(isfinite(log(val)));
      ll += log(val);
    }
  return ll;
}

// Caching function for first iteration, 
// when all the elements in the vector are the same
double
log_likelihood_cached(const epiread &r, const double log_a_meth, const double inverse_log_a){
  double ll = 0.0;
  for (size_t i = 0; i < r.seq.length(); ++i)
    if (is_meth(r, i) || un_meth(r, i)) {
      const double val = (is_meth(r, i) ? log_a_meth : inverse_log_a);
      ll += val;
    }
  return ll;
}

// Caching function for standard iteration, 
// when elements in the vector are different
double
log_likelihood_cached(const epiread &r, const std::vector<double> &log_a_meth, const std::vector<double> &log_a_unmeth){
  double ll = 0.0;
  for (size_t i = 0; i < r.seq.length(); ++i)
    if (is_meth(r, i) || un_meth(r, i)) {
      const double val = (is_meth(r, i) ? log_a_meth[r.pos + i] : log_a_unmeth[r.pos + i]);
      ll += val;
    }
  return ll;
}


double
log_likelihood(const epiread &r, const double mixing, 
		const vector<double> &a1, const vector<double> &a2) {
  return log(mixing*exp(log_likelihood(r, a1)) + 
	     (1.0 - mixing)*exp(log_likelihood(r, a2)));
}

// Caching function 
double
log_likelihood_cached(const epiread &r, const double mixing,
 	  const std::vector<double> &log_a1_meth, const std::vector<double> &log_a1_unmeth,
    const std::vector<double> &log_a2_meth, const std::vector<double> &log_a2_unmeth) {
  return log(mixing*exp(log_likelihood_cached(r, log_a1_meth, log_a1_unmeth)) + 
	     (1.0 - mixing)*exp(log_likelihood_cached(r, log_a2_meth, log_a2_unmeth)));
}

double
log_likelihood(const vector<epiread> &reads, const double mixing,
		const vector<double> &a1, const vector<double> &a2) {
  double ll = 0.0;
  for (size_t i = 0; i < reads.size(); ++i)
    ll += log_likelihood(reads[i], mixing, a1, a2);
  return ll;
}

// Caching function
double
log_likelihood_caching(const vector<epiread> &reads, const double mixing,
		const vector<double> &a1, const vector<double> &a2) {
  double ll = 0.0;
  vector<double> log_a1_meth(a1.size());
  vector<double> log_a1_unmeth(a1.size());
  cache_vector_logs(a1, log_a1_meth, log_a1_unmeth);
  vector<double> log_a2_meth(a2.size());
  vector<double> log_a2_unmeth(a2.size());
  cache_vector_logs(a2, log_a2_meth, log_a2_unmeth);
  for (size_t i = 0; i < reads.size(); ++i)
    ll += log_likelihood_cached(reads[i], mixing, log_a1_meth, log_a1_unmeth, log_a2_meth, log_a2_unmeth);
  return ll;
}


static double
expectation_step(const vector<epiread> &reads, const double mixing,
		 const vector<double> &a1, const vector<double> &a2, 
		 vector<double> &indicators) {
  const double log_mixing1 = log(mixing);
  const double log_mixing2 = log(1.0 - mixing);
  assert(isfinite(log_mixing1) && isfinite(log_mixing2));
  
  double score = 0;
  for (size_t i = 0; i < reads.size(); ++i) {
    const double ll1 = log_mixing1 + log_likelihood(reads[i], a1);
    const double ll2 = log_mixing2 + log_likelihood(reads[i], a2);
    assert(isfinite(ll1) && isfinite(ll2));
    const double log_denom = log(exp(ll1) + exp(ll2));
    score += log_denom;
    indicators[i] = exp(ll1 - log_denom);
    assert(isfinite(log_denom) && isfinite(indicators[i]));
  }
  return score;
}


static double
expectation_step_caching(const vector<epiread> &reads, const double mixing,
		 const vector<double> &a1, const vector<double> &a2, 
		 vector<double> &indicators) {
  const double log_mixing1 = log(mixing);
  const double log_mixing2 = log(1.0 - mixing);
  assert(isfinite(log_mixing1) && isfinite(log_mixing2));
  
  vector<double> log_a1_meth(a1.size());
  vector<double> log_a1_unmeth(a1.size());
  cache_vector_logs(a1, log_a1_meth, log_a1_unmeth);
  vector<double> log_a2_meth(a2.size());
  vector<double> log_a2_unmeth(a2.size());
  cache_vector_logs(a2, log_a2_meth, log_a2_unmeth);
  double score = 0;
  for (size_t i = 0; i < reads.size(); ++i) {
    const double ll1 = log_mixing1 + log_likelihood_cached(reads[i], log_a1_meth, log_a1_unmeth);
    const double ll2 = log_mixing2 + log_likelihood_cached(reads[i], log_a2_meth, log_a2_unmeth);
    assert(isfinite(ll1) && isfinite(ll2));
    const double log_denom = log(exp(ll1) + exp(ll2));
    score += log_denom;
    indicators[i] = exp(ll1 - log_denom);
    assert(isfinite(log_denom) && isfinite(indicators[i]));
  }
  return score;
}


void
fit_epiallele(double pseudo, const vector<epiread> &reads, 
	      const vector<double> &indicators, vector<double> &a) {
  const size_t n_cpgs = a.size();
  vector<double> meth(n_cpgs, 0.0), total(n_cpgs, 0.0);
  for (size_t i = 0; i < reads.size(); ++i) {
    const size_t start = reads[i].pos;
    const double weight = indicators[i];
    for (size_t j = 0; j < reads[i].seq.length(); ++j)
      if (is_meth(reads[i], j) || un_meth(reads[i], j)) {
	meth[start + j] += weight*(is_meth(reads[i], j));
	total[start + j] += weight;
      }
  }
  for (size_t i = 0; i < n_cpgs; ++i)
    a[i] = (meth[i] + pseudo)/(total[i] + 2*pseudo);
}


static void
maximization_step(const vector<epiread> &reads, const vector<double> &indicators,
		  vector<double> &a1, vector<double> &a2) {
  
  vector<double> inverted_indicators(indicators);
  for (size_t i = 0; i < inverted_indicators.size(); ++i)
    inverted_indicators[i] = 1.0 - inverted_indicators[i];
  
  // Fit the regular model parameters. Since the two epialleles'
  // likelihoods are summed, we need to make sure the pseudocount
  // is proportional to the pseudocount used in the single allele model.
  fit_epiallele(0.5*PSEUDOCOUNT, reads, indicators, a1);
  fit_epiallele(0.5*PSEUDOCOUNT, reads, inverted_indicators, a2);
}


static void
rescale_indicators(const double mixing, vector<double> &indic) {
  const double n_reads = indic.size();
  const double total = accumulate(indic.begin(), indic.end(), 0.0);
  const double ratio = total/n_reads;
  
  if (mixing < ratio)
    for (size_t i = 0; i < indic.size(); ++i)
      indic[i] *= (mixing/ratio);

  else {
    const double adjustment = mixing/(1.0 - ratio);
    for (size_t i = 0; i < indic.size(); ++i)
      indic[i] = 1.0 - (1.0 - indic[i])*adjustment;
  }
}


static double
expectation_maximization(const size_t max_itr, const vector<epiread> &reads, 
			 const double &mixing, vector<double> &indicators, 
			 vector<double> &a1, vector<double> &a2) {

  double prev_score = -std::numeric_limits<double>::max();
  for (size_t i = 0; i < max_itr; ++i) {
    
    const double score = expectation_step(reads, mixing, a1, a2, indicators);
    rescale_indicators(mixing, indicators);
    maximization_step(reads, indicators, a1, a2);
    
    if ((prev_score - score)/prev_score < EPIREAD_STATS_TOLERANCE)
      break;
    prev_score = score;
  }
  return prev_score;
}

static double
expectation_maximization_caching(const size_t max_itr, const vector<epiread> &reads, 
			 const double &mixing, vector<double> &indicators, 
			 vector<double> &a1, vector<double> &a2) {

  double prev_score = -std::numeric_limits<double>::max();
  for (size_t i = 0; i < max_itr; ++i) {
    
    const double score = expectation_step_caching(reads, mixing, a1, a2, indicators);
    rescale_indicators(mixing, indicators);
    maximization_step(reads, indicators, a1, a2);
    
    if ((prev_score - score)/prev_score < EPIREAD_STATS_TOLERANCE)
      break;
    prev_score = score;
  }
  return prev_score;
}

double
resolve_epialleles(const size_t max_itr, const vector<epiread> &reads, 
		   const double &mixing, vector<double> &indicators, 
		   vector<double> &a1, vector<double> &a2) {
  
  indicators.clear();
  indicators.resize(reads.size(), 0.0);
  for (size_t i = 0; i < reads.size(); ++i) {
    const double l1 = log_likelihood(reads[i], a1);
    const double l2 = log_likelihood(reads[i], a2);
    indicators[i] = exp(l1 - log(exp(l1) + exp(l2)));
  }
  
  return expectation_maximization(max_itr, reads, mixing, 
				  indicators, a1, a2);
}

double
resolve_epialleles_caching(const size_t max_itr, const vector<epiread> &reads, 
		   const double &mixing, vector<double> &indicators, 
		   vector<double> &a1, vector<double> &a2) {
  
  indicators.clear();
  indicators.resize(reads.size(), 0.0);

  double log_a1_meth = log(a1[0]);
  assert(isfinite(log_a1_meth));
  double log_a1_unmeth = log(1 - a1[0]);
  assert(isfinite(log_a1_unmeth));
  double log_a2_meth = log(a2[0]);
  assert(isfinite(log_a2_meth));
  double log_a2_unmeth = log(1 - a2[0]);
  assert(isfinite(log_a2_unmeth));
  for (size_t i = 0; i < reads.size(); ++i) {
    const double l1 = log_likelihood_cached(reads[i], log_a1_meth, log_a1_unmeth);
    const double l2 = log_likelihood_cached(reads[i], log_a2_meth, log_a2_unmeth);
    indicators[i] = exp(l1 - log(exp(l1) + exp(l2)));
  }
  
  return expectation_maximization_caching(max_itr, reads, mixing, 
				  indicators, a1, a2);
}

double
fit_single_epiallele(const vector<epiread> &reads, vector<double> &a) {
  assert(reads.size() > 0);
  vector<double> indicators(reads.size(), 1.0);
  fit_epiallele(PSEUDOCOUNT, reads, indicators, a);
  
  double score = 0.0;
  for (size_t i = 0; i < reads.size(); ++i) {
    score += log_likelihood(reads[i], a);
    assert(isfinite(score));
  }  
  return score;
}


double
fit_single_epiallele_caching(const std::vector<epiread> &reads, std::vector<double> &a){
  assert(reads.size() > 0);
  vector<double> indicators(reads.size(), 1.0);
  fit_epiallele(PSEUDOCOUNT, reads, indicators, a);
  
  vector<double> log_a1_meth(a.size());
  vector<double> log_a1_unmeth(a.size());
  cache_vector_logs(a, log_a1_meth, log_a1_unmeth);
  double score = 0.0;
  for (size_t i = 0; i < reads.size(); ++i) {
    score += log_likelihood_cached(reads[i], log_a1_meth, log_a1_unmeth);
    assert(isfinite(score));
  }  
  return score;
}


void
compute_model_likelihoods( double &single_score, double &pair_score,
       const size_t &max_itr, const double &low_prob, const double &high_prob,
       const size_t &n_cpgs, vector<epiread> &reads ) {

  static const double mixing = 0.5;

  // try a single epi-allele and compute its log likelihood
  vector<double> a0(n_cpgs, 0.5);
  single_score = fit_single_epiallele(reads, a0);
  
  // initialize the pair epi-alleles and indicators, and do the actual
  // computation to infer alleles, compute its log likelihood
  vector<double> a1(n_cpgs, low_prob), a2(n_cpgs, high_prob), indicators;
  resolve_epialleles(max_itr, reads, mixing, indicators, a1, a2);
  pair_score = log_likelihood(reads, mixing, a1, a2);

}

void
compute_model_likelihoods_caching( double &single_score, double &pair_score,
       const size_t &max_itr, const double &low_prob, const double &high_prob,
       const size_t &n_cpgs, vector<epiread> &reads ) {

  static const double mixing = 0.5;

  // try a single epi-allele and compute its log likelihood
  vector<double> a0(n_cpgs, 0.5);
  single_score = fit_single_epiallele_caching(reads, a0);
  
  // initialize the pair epi-alleles and indicators, and do the actual
  // computation to infer alleles, compute its log likelihood
  vector<double> a1(n_cpgs, low_prob), a2(n_cpgs, high_prob), indicators;
  resolve_epialleles_caching(max_itr, reads, mixing, indicators, a1, a2);
  pair_score = log_likelihood_caching(reads, mixing, a1, a2);

}


double
test_asm_lrt(const size_t max_itr, const double low_prob, const double high_prob, 
	     vector<epiread> reads) {
  double single_score = std::numeric_limits<double>::min();
  double pair_score = std::numeric_limits<double>::min();
  adjust_read_offsets(reads);
  const size_t n_cpgs = get_n_cpgs(reads);

  /*compute_model_likelihoods( single_score, pair_score, max_itr, low_prob,
         high_prob, n_cpgs, reads );*/
  compute_model_likelihoods_caching( single_score, pair_score, max_itr, low_prob,
         high_prob, n_cpgs, reads );

  // degrees of freedom = 2*n_cpgs for two-allele model 
  // minus n_cpgs for one-allele model
  const size_t df = n_cpgs;
  
  const double llr_stat = -2*(single_score - pair_score);
  const double p_value = 1.0 - gsl_cdf_chisq_P(llr_stat, df);
  return p_value;
}


double
test_asm_bic(const size_t max_itr, const double low_prob, const double high_prob,
	     vector<epiread> reads) {

  double single_score = std::numeric_limits<double>::min();
  double pair_score = std::numeric_limits<double>::min();
  adjust_read_offsets(reads);
  const size_t n_cpgs = get_n_cpgs(reads);

  /*compute_model_likelihoods( single_score, pair_score, max_itr, low_prob,
         high_prob, n_cpgs, reads );*/
  compute_model_likelihoods_caching( single_score, pair_score, max_itr, low_prob,
         high_prob, n_cpgs, reads );

  // compute bic scores and compare
  const double bic_single = n_cpgs*log(reads.size()) - 2*single_score;
  const double bic_pair = 2*n_cpgs*log(reads.size()) - 2*pair_score;
  return bic_pair - bic_single;
}