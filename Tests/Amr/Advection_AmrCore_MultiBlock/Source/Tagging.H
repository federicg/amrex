#ifndef TAGGING_H
#define TAGGING_H

#include <AMReX_Array4.H>

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
state_error (int i, int j, int k,
             amrex::Array4<char> const& tag,
             amrex::Array4<amrex::Real const> const& state,
             amrex::Real phierr, char tagval, amrex::IntVect valid_start, amrex::IntVect valid_end)
{
    auto current_index = amrex::IntVect(AMREX_D_DECL(std::min(std::max(i, valid_start[0]), valid_end[0]), 
                                                     std::min(std::max(j, valid_start[1]), valid_end[1]), 
                                                     std::min(std::max(k, valid_start[2]), valid_end[2])));

    if (state(i, j, k) > phierr) {
        tag(current_index) = tagval;
    }
}

#endif
