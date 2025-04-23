#ifndef BCFILL_H
#define BCFILL_H

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

struct AmrCoreFill
{
    AMREX_GPU_DEVICE
    void operator() (const amrex::IntVect& iv, amrex::Array4<amrex::Real> const& data,
                     const int dcomp, const int numcomp,
                     amrex::GeometryData const& /*geom*/, const amrex::Real /*time*/,
                     const amrex::BCRec* bcr, const int bcomp,
                     const int /*orig_comp*/) const
        {
            // do something for external Dirichlet (BCType::ext_dir)

            for (int n = 0; n < numcomp; ++n)
            {
                const int comp = dcomp + n;
                const int bc_lo_x = bcr[n + bcomp].lo(0);
                const int bc_hi_x = bcr[n + bcomp].hi(0);
                const int bc_lo_y = bcr[n + bcomp].lo(1);
                const int bc_hi_y = bcr[n + bcomp].hi(1);
                const int bc_lo_z = bcr[n + bcomp].lo(2);
                const int bc_hi_z = bcr[n + bcomp].hi(2);

                // Set to zero if any direction is ext_dir
                if (bc_lo_x == amrex::BCType::ext_dir || bc_hi_x == amrex::BCType::ext_dir ||
                    bc_lo_y == amrex::BCType::ext_dir || bc_hi_y == amrex::BCType::ext_dir ||
                    bc_lo_z == amrex::BCType::ext_dir || bc_hi_z == amrex::BCType::ext_dir)
                {
                    data(iv, comp) = 0.0;
                }
            }
        }


};

#endif
