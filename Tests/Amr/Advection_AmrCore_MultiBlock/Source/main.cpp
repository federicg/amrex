
#include <iostream>

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>

#include <AmrCoreAdv.H>

using namespace amrex;

// define the length of the edge
static constexpr Real cube_face_length = 1.;

enum idirs { ix, iy };

// canonical basis vectors
static constexpr IntVect e_x = IntVect::TheDimensionVector(ix);
static constexpr IntVect e_y = IntVect::TheDimensionVector(iy);


int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    {
        // timer for profiling
        BL_PROFILE("main()"); 

        // wallclock time
        const auto strt_total = amrex::second();

        // if Box domain(IntVect{}, IntVect{AMREX_D_DECL(7, 7, 8)}); means that the valid indices are from 0 to 7 at level 0 and at level 1 are from 0 to 15
        Box domain(IntVect{}, IntVect{AMREX_D_DECL(63, 63, 8)});

        RealBox real_box1{{AMREX_D_DECL(-cube_face_length*.5,                  -cube_face_length*.5,                  0.0)}, {AMREX_D_DECL(+cube_face_length*.5,                     +cube_face_length*.5,                  1.0)}};
        RealBox real_box2{{AMREX_D_DECL(+cube_face_length*.5,                  -cube_face_length*.5,                  0.0)}, {AMREX_D_DECL(+cube_face_length*.5+cube_face_length,    +cube_face_length*.5,                  1.0)}};
        
        Array<int, AMREX_SPACEDIM> is_periodic1{AMREX_D_DECL(0, 1, 1)};
        Geometry geom1{domain, real_box1, CoordSys::cartesian, is_periodic1};

        Array<int, AMREX_SPACEDIM> is_periodic2{AMREX_D_DECL(0, 1, 1)};
        Geometry geom2{domain, real_box2, CoordSys::cartesian, is_periodic2};

        AmrInfo amr_info{}; 
        amr_info.max_level = 1; // maximum level number allowed -- number of levels = max_level + 1
        amr_info.blocking_factor.assign(amr_info.max_level+1, IntVect{AMREX_D_DECL( 8,  8,  8)}); // along, x, y, z
        amr_info.max_grid_size  .assign(amr_info.max_level+1, IntVect{AMREX_D_DECL(16, 16, 16)}); // along, x, y, z
        amr_info.ref_ratio      .assign(amr_info.max_level+1, IntVect{AMREX_D_DECL( 2,  2,  1)}); // controlla qui se non raffina come pensi
        amr_info.verbose = 1;


        // constructor - reads in parameters from inputs file
        //             - sizes multilevel arrays and data structures
        AmrCoreAdv amr_core_adv_1(geom1, 1, amr_info); // write the block number, then the constructor should be able to get from the input fil multiple coordinate corresponding to vrious Blocks
        AmrCoreAdv amr_core_adv_2(geom2, 2, amr_info);


        // set the pointers to the cores
        amr_core_adv_1.setOtherCore(&amr_core_adv_2, &amr_core_adv_2, &amr_core_adv_2, &amr_core_adv_2);
        amr_core_adv_2.setOtherCore(&amr_core_adv_1, &amr_core_adv_1, &amr_core_adv_1, &amr_core_adv_1);


        // put here the communication part, 0 is the right boundary of the receiver, 1 is the left boundary of the receiver, 2 is the upper boundary of the receiver, 3, is the lower boundary of the receiver
        int num_ghost = 3;
        {
            auto current_ref_ratio = IntVect(AMREX_D_DECL(1, 1, 1));
            for (int lev = 0; lev <= amr_info.max_level; ++lev)
            {
                Box domain_ref = amrex::refine(domain, current_ref_ratio);

                // ------------------------------------------------------------------------- // 1
                {   // Fill right boundary of core_1 with left mirror data of core_2
                    NonLocalBC::MultiBlockIndexMapping dtos{};
                    dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
                    dtos.offset = (domain_ref.bigEnd(ix) + 1) * e_x;
                    dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
                    Box right_boundary_to_fill_in_x = grow(shift(Box{(domain_ref.bigEnd(ix)-(num_ghost-1)) * e_x, domain_ref.bigEnd()}, num_ghost*e_x), num_ghost*e_y);

                    amr_core_adv_1.multi_block_boundaries[lev].push_back({dtos, right_boundary_to_fill_in_x, 0});
                } { // Fill left boundary of core_2 with right mirror data of core_1
                    NonLocalBC::MultiBlockIndexMapping dtos{};
                    dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
                    dtos.offset = - (domain_ref.bigEnd(ix) + 1) * e_x;
                    dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
                    Box left_boundary_to_fill_in_x = grow(shift(Box{domain_ref.smallEnd(), domain_ref.bigEnd() - (domain_ref.bigEnd(ix)-(num_ghost-1)) * e_x}, -num_ghost*e_x), num_ghost*e_y);  

                    amr_core_adv_2.multi_block_boundaries[lev].push_back({dtos, left_boundary_to_fill_in_x, 1});
                } 

                // ------------------------------------------------------------------------- // 2
                {   // Fill right boundary of core_2 with left mirror data of core_1
                    NonLocalBC::MultiBlockIndexMapping dtos{};
                    dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
                    dtos.offset = (domain_ref.bigEnd(ix) + 1) * e_x;
                    dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
                    Box right_boundary_to_fill_in_x = grow(shift(Box{(domain_ref.bigEnd(ix)-(num_ghost-1)) * e_x, domain_ref.bigEnd()}, num_ghost*e_x), num_ghost*e_y);

                    amr_core_adv_2.multi_block_boundaries[lev].push_back({dtos, right_boundary_to_fill_in_x, 0});
                } { // Fill left boundary of core_1 with right mirror data of core_2
                    NonLocalBC::MultiBlockIndexMapping dtos{};
                    dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
                    dtos.offset = - (domain_ref.bigEnd(ix) + 1) * e_x;
                    dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
                    Box left_boundary_to_fill_in_x = grow(shift(Box{domain_ref.smallEnd(), domain_ref.bigEnd() - (domain_ref.bigEnd(ix)-(num_ghost-1)) * e_x}, -num_ghost*e_x), num_ghost*e_y);

                    amr_core_adv_1.multi_block_boundaries[lev].push_back({dtos, left_boundary_to_fill_in_x, 1});
                } 

                current_ref_ratio *= amr_info.ref_ratio[lev];
            }
        }

        // now move the block boundaries,
        amr_core_adv_1.MoveMultiBlocks();
        amr_core_adv_2.MoveMultiBlocks();

        // initialize AMR data, and writes the initial condition 
        amr_core_adv_1.InitData();
        amr_core_adv_2.InitData();

        // call function to create new multifab from the stored pointers,
        amr_core_adv_1.create_ghost_multifabs(num_ghost); // set the number of ghosts
        amr_core_adv_2.create_ghost_multifabs(num_ghost); // set the number of ghosts

        // advance solution to final time
        Real cur_time = amr_core_adv_1.t_new[0];
        int last_plot_file_step = 0;

        for (int step = amr_core_adv_1.istep[0]; step < amr_core_adv_1.max_step && cur_time < amr_core_adv_1.stop_time; ++step)
        {
            amrex::Print() << "\nCoarse STEP " << step+1 << " starts ..." << '\n';

            amr_core_adv_1.ComputeDt(); // be careful here about the sync of dt, for velocity equal to a number everywhere there are no issues right now
            amr_core_adv_2.ComputeDt();

            if (!amr_core_adv_1.do_subcycle) { // sync the time steps if not using the subcycling
                const auto min_dt = std::min(amr_core_adv_1.dt[0], amr_core_adv_2.dt[0]);

                for (int lev = 0; lev <= amr_core_adv_1.getFinestLevel(); ++lev) {
                    amr_core_adv_1.dt[lev] = min_dt;
                }

                for (int lev = 0; lev <= amr_core_adv_2.getFinestLevel(); ++lev) {
                    amr_core_adv_2.dt[lev] = min_dt;
                }
            }

            if (amr_core_adv_1.dt[0] != amr_core_adv_2.dt[0]) {
                throw std::runtime_error("You have to fix the syncronization between times in various cores");
            }


            int lev = 0;
            int iteration = 1;
            if (amr_core_adv_1.do_subcycle) {

                // apply the numerical scheme
                amr_core_adv_1.timeStepWithSubcycling(lev, cur_time, iteration);
                amr_core_adv_2.timeStepWithSubcycling(lev, cur_time, iteration);
            } else {

                // perform the regridding on each core
                amr_core_adv_1.perform_regrid(cur_time);
                amr_core_adv_2.perform_regrid(cur_time);

                // call function to create new multifab from the stored pointers,
                amr_core_adv_1.create_ghost_multifabs(num_ghost); // set the number of ghosts
                amr_core_adv_2.create_ghost_multifabs(num_ghost); // set the number of ghosts

                // apply the numerical scheme
                amr_core_adv_1.timeStepNoSubcycling(cur_time, iteration);
                amr_core_adv_2.timeStepNoSubcycling(cur_time, iteration);
            }

            cur_time += amr_core_adv_1.dt[0];

            // sum phi to check conservation
            Real sum_phi = amr_core_adv_1.phi_new[0].sum() + amr_core_adv_2.phi_new[0].sum();

            amrex::Print() << "Coarse STEP " << step+1 << " ends." << " TIME = " << cur_time
                        << " DT core 1 = " << amr_core_adv_1.dt[0] << " DT core 2 = " << amr_core_adv_2.dt[0] << " Sum(Phi) = " << sum_phi << '\n';

            // sync up time, core 1
            for (lev = 0; lev < amr_core_adv_1.getFinestLevel(); ++lev) {
                amr_core_adv_1.t_new[lev] = cur_time;
            }
            // sync up time, core 2
            for (lev = 0; lev < amr_core_adv_2.getFinestLevel(); ++lev) {
                amr_core_adv_2.t_new[lev] = cur_time;
            }

            //if (amr_core_adv_1.plot_int > 0 && (step+1) % amr_core_adv_1.plot_int == 0) 
            {
                last_plot_file_step = step+1;
                amr_core_adv_1.WritePlotFile();
                amr_core_adv_2.WritePlotFile();
            }

            //std::cout << " hereee " << std::endl;
            //exit(1);
            

            if (amr_core_adv_1.chk_int > 0 && (step+1) % amr_core_adv_1.chk_int == 0) {
                amr_core_adv_1.WriteCheckpointFile();
                amr_core_adv_2.WriteCheckpointFile();
            }

#ifdef AMREX_MEM_PROFILING
            {
                std::ostringstream ss;
                ss << "[STEP " << step+1 << "]";
                MemProfiler::report(ss.str());
            }
#endif

            if (cur_time >= amr_core_adv_1.stop_time - 1.e-6*amr_core_adv_1.dt[0]) { break; }
        }

        if (amr_core_adv_1.plot_int > 0 && amr_core_adv_1.istep[0] > last_plot_file_step) {
            amr_core_adv_1.WritePlotFile();
            amr_core_adv_2.WritePlotFile();
        }

        // wallclock time
        auto end_total = amrex::second() - strt_total;

        if (amr_core_adv_1.Verbose()) {
            // print wallclock time
            ParallelDescriptor::ReduceRealMax(end_total ,ParallelDescriptor::IOProcessorNumber());
            amrex::Print() << "\nTotal Time: " << end_total << '\n';
        }
    }

    amrex::Finalize();
}
