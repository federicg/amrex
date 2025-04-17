#ifndef PROB_H_
#define PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
initdata(amrex::Box const& bx, amrex::Array4<amrex::Real> const& phi,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx)
{
    using namespace amrex;

    // here takes the coordinates from RealBox
    const auto lo = lbound(bx); 
    const auto hi = ubound(bx);

    constexpr double R = 0.15;

    Real center[] = {-1, 0}; //{1, 0};



    for         (int k = lo.z; k <= hi.z; ++k) {
        for     (int j = lo.y; j <= hi.y; ++j) {
            Real y = prob_lo[1] + (0.5+j)*dx[1];
                for (int i = lo.x; i <= hi.x; ++i) {
                    Real x = prob_lo[0] + (0.5+i)*dx[0];
		
		    phi(i,j,k) = 0;
                    Real r2 = std::sqrt((x-center[0]) * (x-center[0]) + (y-center[1]) * (y-center[1]));
                    // Gaussian function with smooth transition
                    Real sigma = R / 3.0; // Adjust smoothness
                    phi(i,j,k) = r2 < R ? 1.0 : 0.0; 
                    //phi(i, j, k) = std::exp(-r2*r2 / (2.0 * sigma * sigma));
                    

                    //phi(i,j,k) = x < 0 ? 1.0 : 0.0; 

                    //Real r2 = (std::pow(x-0.5, 2) + std::pow((y-0.75),2)) / 0.01;
                    //phi(i,j,k) = 1.0 + std::exp(-r2);
                }
        }
    }
}

#endif
