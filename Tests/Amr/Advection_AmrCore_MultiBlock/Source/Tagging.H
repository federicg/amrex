#ifndef TAGGING_H
#define TAGGING_H

#include <AMReX_Array4.H>

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
state_error (int i, int j, int k,
             amrex::Array4<char> const& tag,
             amrex::Array4<amrex::Real const> const& state,
             amrex::Array4<amrex::Real const> const& taggerfab,
             amrex::Real phierr, char tagval, amrex::IntVect valid_start, amrex::IntVect valid_end, bool is_first)
{
    amrex::IntVect current_index = amrex::IntVect(AMREX_D_DECL(i, j, k));


    bool is_taggerfab = false;
    if (!is_first)
    {
        if (i==valid_end[0]) // right boundary
        {
            if (taggerfab(i+1,j,k)==1) // take ghost, hope it exists! check it if it does not work
            {
                is_taggerfab = true;
            }
        }
        if (i==valid_start[0]) // left boundary
        {

            if (taggerfab(i-1,j,k)==1) // take ghost, hope it exists! check it if it does not work
            {
                //std::cout << current_index << " " << std::endl;
                is_taggerfab = true; 
            }
        }
        if (j==valid_end[1]) // upper boundary
        {
            if (taggerfab(i,j+1,k)==1) // take ghost, hope it exists! check it if it does not work
            {
                is_taggerfab = true;
            }
        }
        if (j==valid_start[1]) // bottom boundary
        {
            if (taggerfab(i,j-1,k)==1) // take ghost, hope it exists! check it if it does not work
            {
                is_taggerfab = true;
            }
        }
    }

    if (state(current_index) > phierr || is_taggerfab) {
        tag(current_index) = tagval;
    }

//#if (AMREX_SPACEDIM == 2)
//#elif (AMREX_SPACEDIM == 3)
//#endif
}

#endif
