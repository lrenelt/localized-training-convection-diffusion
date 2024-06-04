// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ULTRAWEAK_TEST_PROBLEM_FUNCTIONS_HH
#define DUNE_ULTRAWEAK_TEST_PROBLEM_FUNCTIONS_HH

#include <cmath>

#include <dune/pdelab.hh>

namespace BoundaryProfiles {
  using RF = double;

  RF C2bump(const RF& x) {
    return (x>=0.25)*(x<=0.75)*(256*pow(x,4) - 512*pow(x,3) + 352*pow(x,2) - 96*x + 9);
  }

  RF sineBump(const RF& x, const unsigned int n=3) {
    return pow(sin(n*M_PI*x), 2);
  }

  RF L2bump(const RF& x, const unsigned int n=3) {
    using std::abs;
    const RF tol = 1e-10;
    return abs(n*x - int(n*x) - 0.5) <= 0.25 + tol;
  }
}

#endif // DUNE_ULTRAWEAK_TEST_PROBLEM_FUNCTIONS_HH
