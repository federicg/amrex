
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


template <typename T>
void
build_ghost_communicators(T& amr_core_adv_bound_1, 
		          T& amr_core_adv_bound_2, 
			  T& amr_core_adv_bound_3, 
			  T& amr_core_adv_bound_4, 
			  T& amr_core_adv_bound_5, const Box& domain_ref, const int num_ghost)
{

    // ------------------------------------------------------------------------- // 1
    {   // Fill right boundary of core_1 with left mirror data of core_2
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = (domain_ref.bigEnd(ix) + 1) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box right_boundary_to_fill_in_x = grow(shift(Box{(domain_ref.bigEnd(ix)-(num_ghost-1)) * e_x, domain_ref.bigEnd()}, num_ghost*e_x), num_ghost*e_y);

        amr_core_adv_bound_1.push_back({dtos, right_boundary_to_fill_in_x, 0});
    } { // Fill left boundary of core_2 with right mirror data of core_1
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = - (domain_ref.bigEnd(ix) + 1) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box left_boundary_to_fill_in_x = grow(shift(Box{domain_ref.smallEnd(), domain_ref.bigEnd() - (domain_ref.bigEnd(ix)-(num_ghost-1)) * e_x}, -num_ghost*e_x), num_ghost*e_y);  

        amr_core_adv_bound_2.push_back({dtos, left_boundary_to_fill_in_x, 1});
    } 


    // ------------------------------------------------------------------------- // 2
    {   // Fill right boundary of core_3 with left mirror data of core_1
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = (domain_ref.bigEnd(ix) + 1) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box right_boundary_to_fill_in_x = grow(shift(Box{(domain_ref.bigEnd(ix)-(num_ghost-1)) * e_x, domain_ref.bigEnd()}, num_ghost*e_x), num_ghost*e_y);

        amr_core_adv_bound_3.push_back({dtos, right_boundary_to_fill_in_x, 0});
    } { // Fill left boundary of core_1 with right mirror data of core_3
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = - (domain_ref.bigEnd(ix) + 1) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box left_boundary_to_fill_in_x = grow(shift(Box{domain_ref.smallEnd(), domain_ref.bigEnd() - (domain_ref.bigEnd(ix)-(num_ghost-1)) * e_x}, -num_ghost*e_x), num_ghost*e_y);

        amr_core_adv_bound_1.push_back({dtos, left_boundary_to_fill_in_x, 1});
    } 


    // ------------------------------------------------------------------------- // 3
    {   // Fill right boundary of core_4 with upper mirror data of core_2
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(1, 0, 2)};  // Swap x and y if needed
        dtos.offset = 1 * e_y;
        Box right_boundary_to_fill_in_x = grow(shift(Box{(domain_ref.bigEnd(ix)-(num_ghost-1)) * e_x, domain_ref.bigEnd()}, num_ghost*e_x), num_ghost*e_y);
        amr_core_adv_bound_4.push_back({dtos, right_boundary_to_fill_in_x, 0});
    } { // Fill upper boundary of core_2 with right mirror data of core_4
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(1, 0, 2)};
        dtos.offset = 1 * e_x;
        Box upper_boundary_to_fill_in_y = grow(shift(Box{(domain_ref.bigEnd(iy)-(num_ghost-1)) * e_y, domain_ref.bigEnd()}, num_ghost*e_y), num_ghost*e_x);
        amr_core_adv_bound_2.push_back({dtos, upper_boundary_to_fill_in_y, 2});
    }

    // ------------------------------------------------------------------------- // 4
    {   // Fill left boundary of core_4 with upper mirror data of core_3
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(1, 0, 2)};  // Swap x and y if needed
        dtos.offset = -(domain_ref.bigEnd(iy) + 1) * e_y + domain_ref.bigEnd(ix) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(-1, 1, 1)};
        Box left_boundary_to_fill_in_x = grow(shift(Box{domain_ref.smallEnd(), domain_ref.bigEnd() - (domain_ref.bigEnd(ix)-(num_ghost-1)) * e_x}, -num_ghost*e_x), num_ghost*e_y); 
        amr_core_adv_bound_4.push_back({dtos, left_boundary_to_fill_in_x, 1});
    } { // Fill upper boundary of core_3 with left mirror data of core_4
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(1, 0, 2)};
        dtos.offset = (domain_ref.bigEnd(ix) + 1) * e_x + domain_ref.bigEnd(iy) * e_y;
        dtos.sign = IntVect{AMREX_D_DECL(1, -1, 1)};
        Box upper_boundary_to_fill_in_y = grow(shift(Box{(domain_ref.bigEnd(iy)-(num_ghost-1)) * e_y, domain_ref.bigEnd()}, num_ghost*e_y), num_ghost*e_x);
        amr_core_adv_bound_3.push_back({dtos, upper_boundary_to_fill_in_y, 2});
    }

    // ------------------------------------------------------------------------- // 5
    {   // Fill lower boundary of core_4 with upper mirror data of core_1
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = - (domain_ref.bigEnd(iy) + 1) * e_y;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box lower_boundary_to_fill_in_y = grow(shift(Box{domain_ref.smallEnd(), domain_ref.bigEnd() - (domain_ref.bigEnd(iy)-(num_ghost-1)) * e_y}, -num_ghost*e_y), num_ghost*e_x);
        amr_core_adv_bound_4.push_back({dtos, lower_boundary_to_fill_in_y, 3});
    } { // Fill upper boundary of core_1 with lower mirror data of core_4
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = (domain_ref.bigEnd(iy) + 1) * e_y;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box upper_boundary_to_fill_in_y = grow(shift(Box{(domain_ref.bigEnd(iy)-(num_ghost-1)) * e_y, domain_ref.bigEnd()}, num_ghost*e_y), num_ghost*e_x);
        amr_core_adv_bound_1.push_back({dtos, upper_boundary_to_fill_in_y, 2});
    } 


    // ------------------------------------------------------------------------- // 6
    {   // Fill lower boundary of core_1 with upper mirror data of core_5
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = - (domain_ref.bigEnd(iy) + 1) * e_y;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box lower_boundary_to_fill_in_y = grow(shift(Box{domain_ref.smallEnd(), domain_ref.bigEnd() - (domain_ref.bigEnd(iy)-(num_ghost-1)) * e_y}, -num_ghost*e_y), num_ghost*e_x);
        amr_core_adv_bound_1.push_back({dtos, lower_boundary_to_fill_in_y, 3});
    } { // Fill upper boundary of core_5 with lower mirror data of core_1
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = (domain_ref.bigEnd(iy) + 1) * e_y;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box upper_boundary_to_fill_in_y = grow(shift(Box{(domain_ref.bigEnd(iy)-(num_ghost-1)) * e_y, domain_ref.bigEnd()}, num_ghost*e_y), num_ghost*e_x);
        amr_core_adv_bound_5.push_back({dtos, upper_boundary_to_fill_in_y, 2});
    } 

    // ------------------------------------------------------------------------- // 7
    {   // Fill right boundary of core_5 with lower mirror data of core_2
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(1, 0, 2)};
        dtos.offset = (domain_ref.bigEnd(iy) + 1) * e_y + domain_ref.bigEnd(ix) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(-1, 1, 1)};
        Box right_boundary_to_fill_in_x = grow(shift(Box{(domain_ref.bigEnd(ix)-(num_ghost-1)) * e_x, domain_ref.bigEnd()}, num_ghost*e_x), num_ghost*e_y);
        amr_core_adv_bound_5.push_back({dtos, right_boundary_to_fill_in_x, 0});
    } { // Fill lower boundary of core_2 with right mirror data of core_5
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(1, 0, 2)};
        dtos.offset = domain_ref.bigEnd(iy) * e_y - (domain_ref.bigEnd(ix) + 1) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(1, -1, 1)};
        Box lower_boundary_to_fill_in_y = grow(shift(Box{domain_ref.smallEnd(), domain_ref.bigEnd() - (domain_ref.bigEnd(iy)-(num_ghost-1)) * e_y}, -num_ghost*e_y), num_ghost*e_x);
        amr_core_adv_bound_2.push_back({dtos, lower_boundary_to_fill_in_y, 3});
    } 

    // ------------------------------------------------------------------------- // 8
    {   // Fill left boundary of core_5 with lower mirror data of core_3
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(1, 0, 2)};
        dtos.offset = -1 * e_y;
        Box left_boundary_to_fill_in_x = grow(shift(Box{domain_ref.smallEnd(), domain_ref.bigEnd() - (domain_ref.bigEnd(ix)-(num_ghost-1)) * e_x}, -num_ghost*e_x), num_ghost*e_y);
        amr_core_adv_bound_5.push_back({dtos, left_boundary_to_fill_in_x, 1});
    } { // Fill lower boundary of core_3 with left mirror data of core_5
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(1, 0, 2)};
        dtos.offset = -1 * e_x;
        Box lower_boundary_to_fill_in_y = grow(shift(Box{domain_ref.smallEnd(), domain_ref.bigEnd() - (domain_ref.bigEnd(iy)-(num_ghost-1)) * e_y}, -num_ghost*e_y), num_ghost*e_x);
        amr_core_adv_bound_3.push_back({dtos, lower_boundary_to_fill_in_y, 3});
    } 



}


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

        RealBox real_box1{{AMREX_D_DECL(-cube_face_length*.5,                  -cube_face_length*.5,                  0.0)}, 
			  {AMREX_D_DECL(+cube_face_length*.5,                  +cube_face_length*.5,                  1.0)}};

        RealBox real_box2{{AMREX_D_DECL(-cube_face_length*.5+cube_face_length, -cube_face_length*.5,                  0.0)}, 
			  {AMREX_D_DECL(+cube_face_length*.5+cube_face_length, +cube_face_length*.5,                  1.0)}};

        RealBox real_box3{{AMREX_D_DECL(-cube_face_length*.5-cube_face_length, -cube_face_length*.5,                  0.0)}, 
			  {AMREX_D_DECL(+cube_face_length*.5-cube_face_length, +cube_face_length*.5,                  1.0)}};

        RealBox real_box4{{AMREX_D_DECL(-cube_face_length*.5,    	       +cube_face_length-cube_face_length*.5, 0.0)}, 
		          {AMREX_D_DECL(+cube_face_length*.5,                  +cube_face_length+cube_face_length*.5, 1.0)}};

        RealBox real_box5{{AMREX_D_DECL(-cube_face_length*.5,                  -cube_face_length-cube_face_length*.5, 0.0)}, 
		          {AMREX_D_DECL(+cube_face_length*.5,    	       -cube_face_length+cube_face_length*.5, 1.0)}};

	Array<int, AMREX_SPACEDIM> is_periodic1{AMREX_D_DECL(0, 0, 0)};
        Geometry geom1{domain, real_box1, CoordSys::cartesian, is_periodic1};

        Array<int, AMREX_SPACEDIM> is_periodic2{AMREX_D_DECL(0, 0, 0)};
        Geometry geom2{domain, real_box2, CoordSys::cartesian, is_periodic2};

        Array<int, AMREX_SPACEDIM> is_periodic3{AMREX_D_DECL(0, 0, 0)};
        Geometry geom3{domain, real_box3, CoordSys::cartesian, is_periodic3};
        
        Array<int, AMREX_SPACEDIM> is_periodic4{AMREX_D_DECL(0, 0, 0)};
        Geometry geom4{domain, real_box4, CoordSys::cartesian, is_periodic4};

        Array<int, AMREX_SPACEDIM> is_periodic5{AMREX_D_DECL(0, 0, 0)};
        Geometry geom5{domain, real_box5, CoordSys::cartesian, is_periodic5};

	AmrInfo amr_info{}; 
        amr_info.max_level = 1; // maximum level number allowed -- number of levels = max_level + 1
        amr_info.blocking_factor.assign(amr_info.max_level+1, IntVect{AMREX_D_DECL(2, 2, 2)}); // along, x, y, z
        //amr_info.max_grid_size  .assign(amr_info.max_level+1, IntVect{AMREX_D_DECL(4, 4, 4)}); // along, x, y, z
        amr_info.ref_ratio      .assign(amr_info.max_level+1, IntVect{AMREX_D_DECL(2, 2, 1)}); // controlla qui se non raffina come pensi
        amr_info.verbose = 1;
        

        //amr_info.refine_grid_layout = 1;
        amr_info.n_error_buf.assign(amr_info.max_level+1, IntVect(AMREX_D_DECL(0,0,0)));


        // constructor - reads in parameters from inputs file
        //             - sizes multilevel arrays and data structures
        AmrCoreAdv amr_core_adv_1(geom1, 1, amr_info); // write the block number, then the constructor should be able to get from the input fil multiple coordinate corresponding to various Blocks
        AmrCoreAdv amr_core_adv_2(geom2, 2, amr_info);
        AmrCoreAdv amr_core_adv_3(geom3, 3, amr_info);
        AmrCoreAdv amr_core_adv_4(geom4, 4, amr_info);
        AmrCoreAdv amr_core_adv_5(geom5, 5, amr_info);

        // set the pointers to the cores
        amr_core_adv_1.setOtherCore(&amr_core_adv_2, &amr_core_adv_3, &amr_core_adv_4, &amr_core_adv_5);
        amr_core_adv_2.setOtherCore(&amr_core_adv_1, &amr_core_adv_1, &amr_core_adv_4, &amr_core_adv_5);
        amr_core_adv_3.setOtherCore(&amr_core_adv_1, &amr_core_adv_1, &amr_core_adv_4, &amr_core_adv_5);
        amr_core_adv_4.setOtherCore(&amr_core_adv_2, &amr_core_adv_3, &amr_core_adv_1, &amr_core_adv_1);
        amr_core_adv_5.setOtherCore(&amr_core_adv_2, &amr_core_adv_3, &amr_core_adv_1, &amr_core_adv_1);

        // put here the communication part, 0 is the right boundary of the receiver, 1 is the left boundary of the receiver, 2 is the upper boundary of the receiver, 3, is the lower boundary of the receiver
        int num_ghost = 3;
        {
	    Box domain_ref = domain;
            // communication part for the solution of the PDE
            //auto current_ref_ratio = IntVect(AMREX_D_DECL(1, 1, 1));
            for (int lev = 0; lev <= amr_info.max_level; ++lev)
            {
                if (lev>0) domain_ref = amrex::refine(domain_ref, amr_info.ref_ratio[lev]);

                // build the communciation
                build_ghost_communicators(amr_core_adv_1.multi_block_boundaries[lev], 
					  amr_core_adv_2.multi_block_boundaries[lev], 
					  amr_core_adv_3.multi_block_boundaries[lev], 
					  amr_core_adv_4.multi_block_boundaries[lev], 
					  amr_core_adv_5.multi_block_boundaries[lev], domain_ref, num_ghost);
                build_ghost_communicators(amr_core_adv_1.multi_block_boundariesMarkers[lev], 
					  amr_core_adv_2.multi_block_boundariesMarkers[lev], 
					  amr_core_adv_3.multi_block_boundariesMarkers[lev], 
					  amr_core_adv_4.multi_block_boundariesMarkers[lev],
					  amr_core_adv_5.multi_block_boundariesMarkers[lev], domain_ref, 1 );
 	    }           
        }

        // now move the block boundaries,
        amr_core_adv_1.MoveMultiBlocks();
        amr_core_adv_2.MoveMultiBlocks();
        amr_core_adv_3.MoveMultiBlocks();
        amr_core_adv_4.MoveMultiBlocks();
        amr_core_adv_5.MoveMultiBlocks();

        // initialize AMR data, and writes the initial condition 
        amr_core_adv_1.InitData();
        amr_core_adv_2.InitData();
        amr_core_adv_3.InitData();
        amr_core_adv_4.InitData();
        amr_core_adv_5.InitData();

        int max_finest_cores = std::max({amr_core_adv_1.getFinestLevel(), 
					 amr_core_adv_2.getFinestLevel(), 
					 amr_core_adv_3.getFinestLevel(),
					 amr_core_adv_4.getFinestLevel(),
					 amr_core_adv_5.getFinestLevel()}); 
        for (int ii_re=0; ii_re<max_finest_cores; ii_re++)
        { 
            // check here the presence of cells to be refined at the interface between various cores
            amr_core_adv_1.check_finer();
            amr_core_adv_2.check_finer();
            amr_core_adv_3.check_finer();
            amr_core_adv_4.check_finer();
            amr_core_adv_5.check_finer();

            // perform now the communciation phase
            amr_core_adv_1.check_finer_communication();
            amr_core_adv_2.check_finer_communication();
            amr_core_adv_3.check_finer_communication();
            amr_core_adv_4.check_finer_communication();
            amr_core_adv_5.check_finer_communication();

            // perform the regridding on each core
            amr_core_adv_1.perform_regrid(amr_core_adv_1.getTnewLev(0));
            amr_core_adv_2.perform_regrid(amr_core_adv_2.getTnewLev(0));
            amr_core_adv_3.perform_regrid(amr_core_adv_3.getTnewLev(0));
            amr_core_adv_4.perform_regrid(amr_core_adv_4.getTnewLev(0));
            amr_core_adv_5.perform_regrid(amr_core_adv_5.getTnewLev(0));
        }

        if (amr_core_adv_1.restart_chkfile.empty()) {
            if (amr_core_adv_1.getChk_int() > 0) {
                amr_core_adv_1.WriteCheckpointFile();
                amr_core_adv_2.WriteCheckpointFile();
                amr_core_adv_3.WriteCheckpointFile();
                amr_core_adv_4.WriteCheckpointFile();
                amr_core_adv_5.WriteCheckpointFile();
            }
        }
        if (amr_core_adv_1.getPlot_int() > 0) {
            amr_core_adv_1.WritePlotFile();
            amr_core_adv_2.WritePlotFile();
            amr_core_adv_3.WritePlotFile();
            amr_core_adv_4.WritePlotFile();
            amr_core_adv_5.WritePlotFile();
        }

        // call function to create new multifab from the stored pointers,
        amr_core_adv_1.create_ghost_multifabs(num_ghost); // set the number of ghosts
        amr_core_adv_2.create_ghost_multifabs(num_ghost); // set the number of ghosts
        amr_core_adv_3.create_ghost_multifabs(num_ghost); // set the number of ghosts
        amr_core_adv_4.create_ghost_multifabs(num_ghost); // set the number of ghosts
        amr_core_adv_5.create_ghost_multifabs(num_ghost); // set the number of ghosts

        Real sum_phi = amr_core_adv_1.computeSumLevel0() + 
		       amr_core_adv_2.computeSumLevel0() +
		       amr_core_adv_3.computeSumLevel0() +
		       amr_core_adv_4.computeSumLevel0() +
		       amr_core_adv_5.computeSumLevel0();

	//std::cout << sum_phi << std::endl;


        // advance solution to final time
        Real cur_time = amr_core_adv_1.getTnewLev(0);
        int last_plot_file_step = 0;

        for (int step = amr_core_adv_1.istep[0]; step < amr_core_adv_1.max_step && cur_time < amr_core_adv_1.stop_time; ++step)
        {
            amrex::Print() << "\nCoarse STEP " << step+1 << " starts ..." << '\n';

            amr_core_adv_1.ComputeDt(); // be careful here about the sync of dt, for velocity equal to a number everywhere there are no issues right now
            amr_core_adv_2.ComputeDt();
	    amr_core_adv_3.ComputeDt();
	    amr_core_adv_4.ComputeDt();
    	    amr_core_adv_5.ComputeDt();


            const auto min_dt = std::min({amr_core_adv_1.getLevel0Dt(), 
			    		  amr_core_adv_2.getLevel0Dt(), 
			    		  amr_core_adv_3.getLevel0Dt(),
					  amr_core_adv_4.getLevel0Dt(),
					  amr_core_adv_5.getLevel0Dt()}); // level 0 dt

            amr_core_adv_1.setLevel0Dt(min_dt);
            amr_core_adv_2.setLevel0Dt(min_dt);
            amr_core_adv_3.setLevel0Dt(min_dt);
            amr_core_adv_4.setLevel0Dt(min_dt);
            amr_core_adv_5.setLevel0Dt(min_dt);

            int lev = 0;
            int iteration = 1;
            if (amr_core_adv_1.do_subcycle) {

                // sync the time step of all cores for the given level
                amr_core_adv_1.setDtWithSubcycling();
                amr_core_adv_2.setDtWithSubcycling();
                amr_core_adv_3.setDtWithSubcycling();
                amr_core_adv_4.setDtWithSubcycling();
                amr_core_adv_5.setDtWithSubcycling();

	//	amr_core_adv_1.timeStepWithSubcycling_original(lev, cur_time, iteration);
	//	amr_core_adv_2.timeStepWithSubcycling_original(lev, cur_time, iteration);


		// here it is not working in case of the presence of particles
		std::vector<int> start_ii;
		start_ii.assign(amr_info.max_level+1, 1);
		max_finest_cores = std::max({amr_core_adv_1.getFinestLevel(), 
					     amr_core_adv_2.getFinestLevel(),
					     amr_core_adv_3.getFinestLevel(),
					     amr_core_adv_4.getFinestLevel(),
					     amr_core_adv_5.getFinestLevel()});

		for (int lev=0; lev<=max_finest_cores; lev++)
                {   
		    const auto & number_sub_cycl_lev = amr_core_adv_1.getNsubsteps(lev);
		    for (int ii = start_ii[lev]; ii <= number_sub_cycl_lev; ++ii) // the refratio is the same for all the cores
		    {     
                        // reset the tagger
                        amr_core_adv_1.reset_level_tagger(); // maybe here could start from lev, but check it first!
                        amr_core_adv_2.reset_level_tagger();
                        amr_core_adv_3.reset_level_tagger();
                        amr_core_adv_4.reset_level_tagger();
                        amr_core_adv_5.reset_level_tagger();


                        // perform the regridding on each core first, 
			// this operation could modify the max_finest_cores and so the loop upper bound!
                        amr_core_adv_1.perform_regridWithSubcycling(lev, false);
                        amr_core_adv_2.perform_regridWithSubcycling(lev, false);
                        amr_core_adv_3.perform_regridWithSubcycling(lev, false);
                        amr_core_adv_4.perform_regridWithSubcycling(lev, false);
                        amr_core_adv_5.perform_regridWithSubcycling(lev, false);

                        max_finest_cores = std::max({amr_core_adv_1.getFinestLevel(), 
						     amr_core_adv_2.getFinestLevel(),
						     amr_core_adv_3.getFinestLevel(),
						     amr_core_adv_4.getFinestLevel(),
						     amr_core_adv_5.getFinestLevel()});

			for (int iii=0; iii<max_finest_cores; iii++)
			{
			    // now perform the checking of interface compatibility and fix it in case is needed
                            // check here the presence of cells to be refined at the interface between various cores
                            amr_core_adv_1.check_finer();
                            amr_core_adv_2.check_finer();
                            amr_core_adv_3.check_finer();
                            amr_core_adv_4.check_finer();
                            amr_core_adv_5.check_finer();


                            // perform now the communication phase
                            amr_core_adv_1.check_finer_communication();
                            amr_core_adv_2.check_finer_communication();
                            amr_core_adv_3.check_finer_communication();
                            amr_core_adv_4.check_finer_communication();
                            amr_core_adv_5.check_finer_communication();
			    
                            // perform the regridding on each core
			    amr_core_adv_1.perform_regridWithSubcycling(lev, (iii==(max_finest_cores-1) ? true : false));
                            amr_core_adv_2.perform_regridWithSubcycling(lev, (iii==(max_finest_cores-1) ? true : false));
                            amr_core_adv_3.perform_regridWithSubcycling(lev, (iii==(max_finest_cores-1) ? true : false));
                            amr_core_adv_4.perform_regridWithSubcycling(lev, (iii==(max_finest_cores-1) ? true : false));
                            amr_core_adv_5.perform_regridWithSubcycling(lev, (iii==(max_finest_cores-1) ? true : false));
			}
			// call function to create new multifab from the stored pointers,
                	amr_core_adv_1.create_ghost_multifabs(num_ghost); // set the number of ghosts
                	amr_core_adv_2.create_ghost_multifabs(num_ghost); // set the number of ghosts
                	amr_core_adv_3.create_ghost_multifabs(num_ghost); // set the number of ghosts
                	amr_core_adv_4.create_ghost_multifabs(num_ghost); // set the number of ghosts
                	amr_core_adv_5.create_ghost_multifabs(num_ghost); // set the number of ghosts


                        // apply the numerical scheme, advance only if it deserves 
                        if (lev <= amr_core_adv_1.getFinestLevel()) amr_core_adv_1.timeStepWithSubcycling(lev, ii);
                        if (lev <= amr_core_adv_2.getFinestLevel()) amr_core_adv_2.timeStepWithSubcycling(lev, ii);
                        if (lev <= amr_core_adv_3.getFinestLevel()) amr_core_adv_3.timeStepWithSubcycling(lev, ii);
                        if (lev <= amr_core_adv_4.getFinestLevel()) amr_core_adv_4.timeStepWithSubcycling(lev, ii);
                        if (lev <= amr_core_adv_5.getFinestLevel()) amr_core_adv_5.timeStepWithSubcycling(lev, ii);

		
			start_ii[lev] = 1+(ii%number_sub_cycl_lev);
	  	        if (lev < max_finest_cores   ) break; 
			if (ii != number_sub_cycl_lev) continue; //check to arrive at the last iterate at the current lev
			 
			amrex::Print() << "Do reflux " << '\n';

			for (int jj=lev-1; jj>=0; jj--)
			{
			    if (start_ii[jj+1]==number_sub_cycl_lev) { lev = jj; break; }
				std::cout << jj << " " << start_ii[jj] << " " << amr_core_adv_1.getFinestLevel() << std::endl;
			    if (jj<amr_core_adv_1.getFinestLevel()) amr_core_adv_1.perform_reflux_across_lev(jj);
			    if (jj<amr_core_adv_2.getFinestLevel()) amr_core_adv_2.perform_reflux_across_lev(jj);   
			    if (jj<amr_core_adv_3.getFinestLevel()) amr_core_adv_3.perform_reflux_across_lev(jj);   
			    if (jj<amr_core_adv_4.getFinestLevel()) amr_core_adv_4.perform_reflux_across_lev(jj);   
			    if (jj<amr_core_adv_5.getFinestLevel()) amr_core_adv_5.perform_reflux_across_lev(jj);   
			}
			amrex::Print() << "Finish reflux " << '\n';
		    }
                } 

            } else {

                // sync all the levels of the cores at the same time step, 
                amr_core_adv_1.setDtNoSubcycling();
                amr_core_adv_2.setDtNoSubcycling();
                amr_core_adv_3.setDtNoSubcycling();
                amr_core_adv_4.setDtNoSubcycling();
                amr_core_adv_5.setDtNoSubcycling();


                // reset the tagger
                amr_core_adv_1.reset_level_tagger();
                amr_core_adv_2.reset_level_tagger();
                amr_core_adv_3.reset_level_tagger();
                amr_core_adv_4.reset_level_tagger();
                amr_core_adv_5.reset_level_tagger();


                // perform the regridding on each core first
                amr_core_adv_1.perform_regrid(cur_time);
                amr_core_adv_2.perform_regrid(cur_time);
                amr_core_adv_3.perform_regrid(cur_time);
                amr_core_adv_4.perform_regrid(cur_time);
                amr_core_adv_5.perform_regrid(cur_time);


                max_finest_cores = std::max({amr_core_adv_1.getFinestLevel(), 
				             amr_core_adv_2.getFinestLevel(),
				             amr_core_adv_3.getFinestLevel(),
				             amr_core_adv_4.getFinestLevel(),
				             amr_core_adv_5.getFinestLevel()});

                for (int ii=0; ii<max_finest_cores; ii++)
                { 
                    // check here the presence of cells to be refined at the interface between various cores
                    amr_core_adv_1.check_finer();
                    amr_core_adv_2.check_finer();
                    amr_core_adv_3.check_finer();
                    amr_core_adv_4.check_finer();
                    amr_core_adv_5.check_finer();


                    // perform now the communciation phase
                    amr_core_adv_1.check_finer_communication();
                    amr_core_adv_2.check_finer_communication();
                    amr_core_adv_3.check_finer_communication();
                    amr_core_adv_4.check_finer_communication();
                    amr_core_adv_5.check_finer_communication();


                    // perform the regridding on each core
                    amr_core_adv_1.perform_regrid(cur_time);
                    amr_core_adv_2.perform_regrid(cur_time);
                    amr_core_adv_3.perform_regrid(cur_time);
                    amr_core_adv_4.perform_regrid(cur_time);
                    amr_core_adv_5.perform_regrid(cur_time);
                }

                // call function to create new multifab from the stored pointers,
                amr_core_adv_1.create_ghost_multifabs(num_ghost); // set the number of ghosts
                amr_core_adv_2.create_ghost_multifabs(num_ghost); // set the number of ghosts
                amr_core_adv_3.create_ghost_multifabs(num_ghost); // set the number of ghosts
                amr_core_adv_4.create_ghost_multifabs(num_ghost); // set the number of ghosts
                amr_core_adv_5.create_ghost_multifabs(num_ghost); // set the number of ghosts


                //std::cout << "second " << (amr_core_adv_1.computeSumLevel0() + amr_core_adv_2.computeSumLevel0()) << std::endl;

                // apply the numerical scheme
                amr_core_adv_1.timeStepNoSubcycling(cur_time, iteration);
                amr_core_adv_2.timeStepNoSubcycling(cur_time, iteration);
                amr_core_adv_3.timeStepNoSubcycling(cur_time, iteration);
                amr_core_adv_4.timeStepNoSubcycling(cur_time, iteration);
                amr_core_adv_5.timeStepNoSubcycling(cur_time, iteration);
            }
            cur_time += amr_core_adv_1.getLevel0Dt();
 
            // sum phi to check conservation
            Real sum_phi = amr_core_adv_1.computeSumLevel0() + 
		           amr_core_adv_2.computeSumLevel0() +
		           amr_core_adv_3.computeSumLevel0() +
		           amr_core_adv_4.computeSumLevel0() +
		           amr_core_adv_5.computeSumLevel0();

            amrex::Print() << "Coarse STEP " << step+1 << " ends." << " TIME = " << cur_time
                        << " DT core 1 = " << amr_core_adv_1.getLevel0Dt() 
			<< " DT core 2 = " << amr_core_adv_2.getLevel0Dt() 
			<< " DT core 3 = " << amr_core_adv_3.getLevel0Dt() 
			<< " DT core 4 = " << amr_core_adv_4.getLevel0Dt() 
			<< " DT core 5 = " << amr_core_adv_5.getLevel0Dt() 
			<< " Sum(Phi) = " << sum_phi << '\n';

            // sync up time for  various cores 
            amr_core_adv_1.setTnewAllLev(cur_time);
            amr_core_adv_2.setTnewAllLev(cur_time);
            amr_core_adv_3.setTnewAllLev(cur_time);
            amr_core_adv_4.setTnewAllLev(cur_time);
            amr_core_adv_5.setTnewAllLev(cur_time);


            if (amr_core_adv_1.getPlot_int() > 0 && (step+1) % amr_core_adv_1.getPlot_int() == 0) 
            {
                last_plot_file_step = step+1;
                amr_core_adv_1.WritePlotFile();
                amr_core_adv_2.WritePlotFile();
                amr_core_adv_3.WritePlotFile();
                amr_core_adv_4.WritePlotFile();
                amr_core_adv_5.WritePlotFile();
            }
            
            if (amr_core_adv_1.getChk_int() > 0 && (step+1) % amr_core_adv_1.getChk_int() == 0) {
                amr_core_adv_1.WriteCheckpointFile();
                amr_core_adv_2.WriteCheckpointFile();
                amr_core_adv_3.WriteCheckpointFile();
                amr_core_adv_4.WriteCheckpointFile();
                amr_core_adv_5.WriteCheckpointFile();
            }

#ifdef AMREX_MEM_PROFILING
            {
                std::ostringstream ss;
                ss << "[STEP " << step+1 << "]";
                MemProfiler::report(ss.str());
            }
#endif

            if (cur_time >= amr_core_adv_1.stop_time - 1.e-6*amr_core_adv_1.getLevel0Dt()) { break; }
        }

        if (amr_core_adv_1.getPlot_int() > 0 && amr_core_adv_1.istep[0] > last_plot_file_step) {
            amr_core_adv_1.WritePlotFile();
            amr_core_adv_2.WritePlotFile();
            amr_core_adv_3.WritePlotFile();
            amr_core_adv_4.WritePlotFile();
            amr_core_adv_5.WritePlotFile();
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
