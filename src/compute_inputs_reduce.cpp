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

#include <mpi.h>
#include "compute_inputs_reduce.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "domain.h"
#include "group.h"
#include "error.h"
#include "modify.h" 
#include <math.h>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeInputsReduce::ComputeInputsReduce(LAMMPS *lmp, int &iarg, int narg, char **arg) :
  Compute(lmp, iarg, narg, arg)
{
  if (narg < 5) error->all(FLERR,"Illegal compute inputs/reduce command");

  scalar_flag = 1;
  minflag = 0;
  maxflag = 0;
  if (strcmp(arg[3],"min") == 0) minflag = 1;
  if (strcmp(arg[3],"max") == 0) maxflag = 1;
  nvalues = force->inumeric(FLERR,arg[4]);
  inputs = new double[nvalues];
  for (int i = 0; i<nvalues; i++){
  	inputs[i] = force->numeric(FLERR,arg[i+5]);
  }
  
}

/* ---------------------------------------------------------------------- */

ComputeInputsReduce::~ComputeInputsReduce()
{
    delete [] inputs;
}

/* ---------------------------------------------------------------------- */

void ComputeInputsReduce::init()
{
}

/* ---------------------------------------------------------------------- */

double ComputeInputsReduce::compute_scalar()
{
  invoked_scalar = update->ntimestep;
  double output = inputs[0];
  if (minflag==1){
  	for (int i = 1; i<nvalues; i++){
  	    if (output > inputs[i]) output = inputs[i];
  	}
  	MPI_Allreduce(&output,&scalar,1,MPI_DOUBLE,MPI_MIN,world);
  }
  if (maxflag == 1){
    for (int i = 1; i<nvalues; i++){
  	    if (output < inputs[i]) output = inputs[i];
  	}
  	MPI_Allreduce(&output,&scalar,1,MPI_DOUBLE,MPI_MAX,world);
  }

  return scalar;
}
