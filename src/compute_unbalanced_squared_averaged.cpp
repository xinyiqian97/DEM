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

/*
	This fix sums the squared unbalanced forces on each particle for the 
	purpose of calculating  an index of unbalanced forces in the input
	script.
*/
#include <mpi.h>
#include "compute_unbalanced_squared_averaged.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "domain.h"
#include "group.h"
#include "error.h"
#include "modify.h" 
#include "fix_multisphere.h" 
#include <math.h>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeUnbalancedSquaredAveraged::ComputeUnbalancedSquaredAveraged(LAMMPS *lmp, int &iarg, int narg, char **arg) :
  Compute(lmp, iarg, narg, arg),
  multisphere_(*(new MultisphereParallel(lmp)))
{
  if (narg != 3) error->all(FLERR,"Illegal compute unbalanced_squared command");

  scalar_flag = 1;
  extscalar = 1;
  fix_ms_ = NULL; 
}

/* ---------------------------------------------------------------------- */

ComputeUnbalancedSquaredAveraged::~ComputeUnbalancedSquaredAveraged()
{
}

/* ---------------------------------------------------------------------- */

void ComputeUnbalancedSquaredAveraged::init()
{
  fix_ms_ =  static_cast<FixMultisphere*>(modify->find_fix_style("multisphere",0)); 
}

/* ---------------------------------------------------------------------- */

double ComputeUnbalancedSquaredAveraged::compute_scalar()
{
  invoked_scalar = update->ntimestep;
  
  double unbalancedforcesquared = 0.0;
  
  if(!fix_ms_){
    double **f = atom->f;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    int nglobal = atom->natoms;

    for (int i = 0; i < nlocal; i++){
      if (mask[i] & groupbit){
  	    unbalancedforcesquared += f[i][0]*f[i][0]+f[i][1]*f[i][1]+f[i][2]*f[i][2];
  	  }
    }
    unbalancedforcesquared /= nglobal;
  }else{
  //Note: multisphere currently only supported for group all. Cannot distinguish different multisphere groups
    double **fcm = fix_ms_->data().fcm_.begin();
    int nbody = fix_ms_->data().n_body();
    int totalbody = fix_ms_->data().n_body_all();
    
    for (int j = 0; j < nbody; j++){
  	    unbalancedforcesquared += fcm[j][0]*fcm[j][0]+fcm[j][1]*fcm[j][1]+fcm[j][2]*fcm[j][2];
    }
    unbalancedforcesquared /= totalbody;
  }
  MPI_Allreduce(&unbalancedforcesquared,&scalar,1,MPI_DOUBLE,MPI_SUM,world);

  return scalar;
}	
