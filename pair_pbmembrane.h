/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(pbmembrane,Pairpbmembrane)

#else

#ifndef LMP_PAIR_pbmembrane_H
#define LMP_PAIR_pbmembrane_H

#include "pair.h"

namespace LAMMPS_NS {

class Pairpbmembrane : public Pair {
 public:
  Pairpbmembrane(LAMMPS *lmp);
  virtual ~Pairpbmembrane();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  virtual void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  
  
 protected:

  double cut_global;
            
  double **epsilon,**sigma,**cut,**zeta,**mu,**beta;  // model parameter values for atom-type pairs

  class AtomVecEllipsoid *avec;

  void allocate();
  double pbmembrane_analytic(const int i, const int j, double a1[3][3],
                           double a2[3][3], double *r12,
                           const double rsq, double *fforce, double *ttor,
                           double *rtor);


};

}
#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair pbmembrane requires atom style ellipsoid

Self-explanatory.

*/
