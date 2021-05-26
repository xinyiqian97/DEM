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

    Christoph Kloss (DCS Computing GmbH, Linz)
    Christoph Kloss (JKU Linz)
    Richard Berger (JKU Linz)

    Copyright 2012-     DCS Computing GmbH, Linz
    Copyright 2009-2012 JKU Linz
------------------------------------------------------------------------- */

#ifdef TANGENTIAL_MODEL
TANGENTIAL_MODEL(TANGENTIAL_HISTORY,history,2)
#else
#ifndef TANGENTIAL_MODEL_HISTORY_H_
#define TANGENTIAL_MODEL_HISTORY_H_
#include "contact_models.h"
#include "tangential_model_base.h"
#include <math.h>
#include "update.h"
#include "global_properties.h"
#include "atom.h"

namespace LIGGGHTS {
namespace ContactModels
{
  template<>
  class TangentialModel<TANGENTIAL_HISTORY> : public TangentialModelBase
  {
    double ** coeffFrict;
    int history_offset;

  public:
    TangentialModel(LAMMPS * lmp, IContactHistorySetup * hsetup,class ContactModelBase *c) :
      TangentialModelBase(lmp, hsetup, c),
      coeffFrict(NULL),
      heating(false),
      heating_track(false),
      cmb(c)
    {
      history_offset = hsetup->add_history_value("shearx", "1");
      hsetup->add_history_value("sheary", "1");
      hsetup->add_history_value("shearz", "1");

    }

    inline void postSettings(IContactHistorySetup * hsetup, ContactModelBase *cmb)
    {}

    inline void registerSettings(Settings& settings)
    {
        settings.registerOnOff("heating_tangential_history",heating,false);
        settings.registerOnOff("heating_tracking",heating_track,false);
        //TODO error->one(FLERR,"TODO here also check if right surface model used");
    }

    inline void connectToProperties(PropertyRegistry & registry)
    {
      registry.registerProperty("coeffFrict", &MODEL_PARAMS::createCoeffFrict);
      registry.connect("coeffFrict", coeffFrict,"tangential_model history");
    }

    inline void surfacesIntersect(const SurfacesIntersectData & sidata, ForceData & i_forces, ForceData & j_forces)
    {
      // normal forces = Hookian contact + normal velocity damping
      const double enx = sidata.en[0];
      const double eny = sidata.en[1];
      const double enz = sidata.en[2];

      // shear history effects
      if(sidata.contact_flags) *sidata.contact_flags |= CONTACT_TANGENTIAL_MODEL;
      double * const shear = &sidata.contact_history[history_offset];

      const bool update_history = sidata.computeflag && sidata.shearupdate;
      if (update_history) {
        const double dt = update->dt;
        shear[0] += sidata.vtr1 * dt;
        shear[1] += sidata.vtr2 * dt;
        shear[2] += sidata.vtr3 * dt;

        // rotate shear displacements

        double rsht = shear[0]*enx + shear[1]*eny + shear[2]*enz;
        shear[0] -= rsht * enx;
        shear[1] -= rsht * eny;
        shear[2] -= rsht * enz;
      }

      const double shrmag = sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);
      const double kt = sidata.kt;
      const double xmu = coeffFrict[sidata.itype][sidata.jtype];

      // tangential forces = shear + tangential velocity damping
      double Ft1 = -(kt * shear[0]);
      double Ft2 = -(kt * shear[1]);
      double Ft3 = -(kt * shear[2]);

      // rescale frictional displacements and forces if needed
      const double Ft_shear = kt * shrmag; // sqrt(Ft1 * Ft1 + Ft2 * Ft2 + Ft3 * Ft3);
      const double Ft_friction = xmu * fabs(sidata.Fn);

      // energy loss from sliding or damping
      if (Ft_shear > Ft_friction) {
        if (shrmag != 0.0) {
          const double ratio = Ft_friction / Ft_shear;
          
          if(heating)
          {
            sidata.P_diss += (vectorMag3DSquared(shear)*kt - ratio*ratio*vectorMag3DSquared(shear)*kt) / (update->dt); 
            if(heating_track && sidata.is_wall) cmb->tally_pw((vectorMag3DSquared(shear)*kt - ratio*ratio*vectorMag3DSquared(shear)*kt) / (update->dt),sidata.i,sidata.jtype,2);
            if(heating_track && !sidata.is_wall) cmb->tally_pp((vectorMag3DSquared(shear)*kt - ratio*ratio*vectorMag3DSquared(shear)*kt) / (update->dt),sidata.i,sidata.j,2);
          }
          Ft1 *= ratio;
          Ft2 *= ratio;
          Ft3 *= ratio;
          
          if (update_history)
          {
              shear[0] = -Ft1/kt;
              shear[1] = -Ft2/kt;
              shear[2] = -Ft3/kt;
          }
        }
        else Ft1 = Ft2 = Ft3 = 0.0;
      }
      else
      {
        const double gammat = sidata.gammat;
        Ft1 -= (gammat*sidata.vtr1);
        Ft2 -= (gammat*sidata.vtr2);
        Ft3 -= (gammat*sidata.vtr3);
        if(heating)
        {
            sidata.P_diss += gammat*(sidata.vtr1*sidata.vtr1+sidata.vtr2*sidata.vtr2+sidata.vtr3*sidata.vtr3); 
            if(heating_track && sidata.is_wall) cmb->tally_pw(gammat*(sidata.vtr1*sidata.vtr1+sidata.vtr2*sidata.vtr2+sidata.vtr3*sidata.vtr3),sidata.i,sidata.jtype,1);
            if(heating_track && !sidata.is_wall) cmb->tally_pp(gammat*(sidata.vtr1*sidata.vtr1+sidata.vtr2*sidata.vtr2+sidata.vtr3*sidata.vtr3),sidata.i,sidata.j,1);
        }
      }

      // forces & torques
      const double tor1 = eny * Ft3 - enz * Ft2;
      const double tor2 = enz * Ft1 - enx * Ft3;
      const double tor3 = enx * Ft2 - eny * Ft1;

      #ifdef NONSPHERICAL_ACTIVE_FLAG
          double torque_i[3];
          if(sidata.is_non_spherical) {
            double xci[3];
            double Ft_i[3] = { Ft1,  Ft2,  Ft3 };
            vectorSubtract3D(sidata.contact_point, atom->x[sidata.i], xci);
            vectorCross3D(xci, Ft_i, torque_i);
          } else {
            torque_i[0] = -sidata.cri * tor1;
            torque_i[1] = -sidata.cri * tor2;
            torque_i[2] = -sidata.cri * tor3;
          }
      #endif
      // return resulting forces
      if(sidata.is_wall) {
        const double area_ratio = sidata.area_ratio;
        i_forces.delta_F[0] += Ft1 * area_ratio;
        i_forces.delta_F[1] += Ft2 * area_ratio;
        i_forces.delta_F[2] += Ft3 * area_ratio;
        #ifdef NONSPHERICAL_ACTIVE_FLAG
                i_forces.delta_torque[0] += torque_i[0] * area_ratio;
                i_forces.delta_torque[1] += torque_i[1] * area_ratio;
                i_forces.delta_torque[2] += torque_i[2] * area_ratio;
        #else
                i_forces.delta_torque[0] += -sidata.cri * tor1 * area_ratio;
                i_forces.delta_torque[1] += -sidata.cri * tor2 * area_ratio;
                i_forces.delta_torque[2] += -sidata.cri * tor3 * area_ratio;
        #endif
      } else {
        i_forces.delta_F[0] += Ft1;
        i_forces.delta_F[1] += Ft2;
        i_forces.delta_F[2] += Ft3;
        j_forces.delta_F[0] += -Ft1;
        j_forces.delta_F[1] += -Ft2;
        j_forces.delta_F[2] += -Ft3;
        #ifdef NONSPHERICAL_ACTIVE_FLAG
                double torque_j[3];
                if(sidata.is_non_spherical) {
                  double xcj[3];
                  vectorSubtract3D(sidata.contact_point, atom->x[sidata.j], xcj);
                  double Ft_j[3] = { -Ft1,  -Ft2,  -Ft3 };
                  vectorCross3D(xcj, Ft_j, torque_j);
                } else {
                  torque_j[0] = -sidata.crj * tor1;
                  torque_j[1] = -sidata.crj * tor2;
                  torque_j[2] = -sidata.crj * tor3;
                }
                i_forces.delta_torque[0] += torque_i[0];
                i_forces.delta_torque[1] += torque_i[1];
                i_forces.delta_torque[2] += torque_i[2];

                j_forces.delta_torque[0] += torque_j[0];
                j_forces.delta_torque[1] += torque_j[1];
                j_forces.delta_torque[2] += torque_j[2];
        #else
                i_forces.delta_torque[0] += -sidata.cri * tor1;
                i_forces.delta_torque[1] += -sidata.cri * tor2;
                i_forces.delta_torque[2] += -sidata.cri * tor3;

                j_forces.delta_torque[0] += -sidata.crj * tor1;
                j_forces.delta_torque[1] += -sidata.crj * tor2;
                j_forces.delta_torque[2] += -sidata.crj * tor3;
        #endif
      }
    }

    inline void surfacesClose(SurfacesCloseData & scdata, ForceData&, ForceData&)
    {
      // unset non-touching neighbors
      // TODO even if shearupdate == false?
      if(scdata.contact_flags) *scdata.contact_flags &= ~CONTACT_TANGENTIAL_MODEL;
      if(!scdata.contact_history)
        return; //DO NOT access contact_history if not available
      double * const shear = &scdata.contact_history[history_offset];
      shear[0] = 0.0;
      shear[1] = 0.0;
      shear[2] = 0.0;
    }

    inline void beginPass(SurfacesIntersectData&, ForceData&, ForceData&){}
    inline void endPass(SurfacesIntersectData&, ForceData&, ForceData&){}

   protected:
    bool heating;
    bool heating_track;
    class ContactModelBase *cmb;
  };
}
}
#endif // TANGENTIAL_MODEL_HISTORY_H_
#endif
