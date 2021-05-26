/* ----------------------------------------------------------------------
    This is the

    ██╗     ██╗ ██████╗  ██████╗  ██████╗ ██╗  ██╗████████╗███████╗
    ██║     ██║██╔════╝ ██╔════╝ ██╔════╝ ██║  ██║╚══██╔══╝██╔════╝
    ██║     ██║██║  ███╗██║  ███╗██║  ███╗███████║   ██║   ███████╗
    ██║     ██║██║   ██║██║   ██║██║   ██║██╔══██║   ██║   ╚════██║
    ███████╗██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║   ██║   ███████║
    ╚══════╝╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝®

    DEM simulation engine, released by
    DCS Computing Gmbh, Linz, Austria
    http://www.dcs-computing.com, office@dcs-computing.com

    LIGGGHTS® is part of CFDEM®project:
    http://www.liggghts.com | http://www.cfdem.com

    Core developer and main author:
    Christoph Kloss, christoph.kloss@dcs-computing.com

    LIGGGHTS® is open-source, distributed under the terms of the GNU Public
    License, version 2 or later. It is distributed in the hope that it will
    be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
    received a copy of the GNU General Public License along with LIGGGHTS®.
    If not, see http://www.gnu.org/licenses . See also top-level README
    and LICENSE files.

    LIGGGHTS® and CFDEM® are registered trade marks of DCS Computing GmbH,
    the producer of the LIGGGHTS® software and the CFDEM®coupling software
    See http://www.cfdem.com/terms-trademark-policy for details.

-------------------------------------------------------------------------
    Contributing author and copyright for this file:
    This file is from LAMMPS
    LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
    http://lammps.sandia.gov, Sandia National Laboratories
    Steve Plimpton, sjplimp@sandia.gov

    Copyright (2003) Sandia Corporation.  Under the terms of Contract
    DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
    certain rights in this software.  This software is distributed under
    the GNU General Public License.
------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "fix_nonviscous.h"
#include "atom.h"
#include "update.h"
#include "respa.h"
#include "error.h"
#include "force.h"
#include "fix_multisphere.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNonViscous::FixNonViscous(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  multisphere_flag = false; //temporarily set to false
  
  if (narg != 4)
    error->all(FLERR,"Illegal fix nonviscous command");
 
 
 if (modify->n_fixes_style("multisphere") > 0) multisphere_flag = true;
  	

  alpha_one = force->numeric(FLERR,arg[3]); // force member function that throws an error if the second input is not numeric (int, double, or float). arg[3] = parameter from input file
  alpha = new double[atom->ntypes+1];				// note that alpha starts at index 1 in the for loop below
  for (int i = 1; i <= atom->ntypes; i++) alpha[i] = alpha_one;
}

/* ---------------------------------------------------------------------- */

FixNonViscous::~FixNonViscous()
{
  delete [] alpha;
}

/* ---------------------------------------------------------------------- */

int FixNonViscous::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNonViscous::init()
{
  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixNonViscous::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
    post_force_respa(vflag,nlevels_respa-1,0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
  }

  // error checks on coarsegraining
  if(force->cg_active())
    error->cg(FLERR,this->style);
}

/* ---------------------------------------------------------------------- */

void FixNonViscous::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixNonViscous::post_force(int vflag)
{
  // apply damping force to atoms in group
  // direction is opposed to velocity vector
  // magnitude depends on atom type

  if (!multisphere_flag){  
    double **v = atom->v;
    double **f = atom->f;
    int *mask = atom->mask;
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double damp_coeff;

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        damp_coeff = alpha[type[i]];
        f[i][0] -= damp_coeff*fabs(f[i][0])*sign(v[i][0]);
        f[i][1] -= damp_coeff*fabs(f[i][1])*sign(v[i][1]);
        f[i][2] -= damp_coeff*fabs(f[i][2])*sign(v[i][2]);
      }
    }
}
/* ---------------------------------------------------------------------- */

double FixNonViscous::sign( double val )
{
  // Return the sign (+/- 1.0) of the input value
  return ( val > 0.0 ) ? 1.0 : ( (val < 0.0 ) ? -1.0 : 0.0 );
}

/* ---------------------------------------------------------------------- */

void FixNonViscous::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixNonViscous::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixNonViscous::get_damp(double &damp)
{
    damp = alpha_one;
}
