
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>

#ifdef AMREX_MEM_PROFILING
#include <AMReX_MemProfiler.H>
#endif

#include <AmrCoreAdv.H>
#include <Kernels.H>

using namespace amrex;

// constructor - reads in parameters from inputs file
//             - sizes multilevel arrays and data structures
//             - initializes BCRec boundary condition object
AmrCoreAdv::AmrCoreAdv (Geometry const& level_0_geom, int index_core,
                     AmrInfo const& amr_info = AmrInfo()) : AmrCore(level_0_geom, amr_info), index_core(index_core), other_core{nullptr, nullptr, nullptr, nullptr}
{
    ReadParameters();

    // Geometry on all levels has been defined already.

    // No valid BoxArray and DistributionMapping have been defined.
    // But the arrays for them have been resized.

    int nlevs_max = max_level + 1;

    istep.resize(nlevs_max, 0);
    nsubsteps.resize(nlevs_max, 1);
    if (do_subcycle) {
        for (int lev = 1; lev <= max_level; ++lev) {
            nsubsteps[lev] = MaxRefRatio(lev-1);
        }
    }

    t_new.resize(nlevs_max, 0.0);
    t_old.resize(nlevs_max, -1.e100);
    dt.resize(nlevs_max, 1.e100);

    phi_new.resize(nlevs_max);
    phi_old.resize(nlevs_max);

    level_tagger.resize(nlevs_max); 

    // 
    array_vec_mf_g[0].resize(nlevs_max);
    array_vec_mf_g[1].resize(nlevs_max);
    array_vec_mf_g[2].resize(nlevs_max);
    array_vec_mf_g[3].resize(nlevs_max);

    //
    multi_block_boundaries.resize(nlevs_max);
    FillBoundary_ghost    .resize(nlevs_max);

    //
    multi_block_boundariesMarkers.resize(nlevs_max);
    FillBoundaryMarkers_ghost    .resize(nlevs_max);

    //
    last_regrid_step.resize(nlevs_max, 0);


    facevel.resize(nlevs_max);

    // periodic boundaries
    //int bc_lo[] = {BCType::int_dir, BCType::int_dir, BCType::int_dir};
    //int bc_hi[] = {BCType::int_dir, BCType::int_dir, BCType::int_dir};

    int bc_lo[] = {BCType::ext_dir, BCType::ext_dir, BCType::ext_dir};
    int bc_hi[] = {BCType::ext_dir, BCType::ext_dir, BCType::ext_dir};


    // walls (Neumann)
//    int bc_lo[] = {amrex::BCType::foextrap, amrex::BCType::foextrap, amrex::BCType::foextrap};
//    int bc_hi[] = {amrex::BCType::foextrap, amrex::BCType::foextrap, amrex::BCType::foextrap};


    //std::cout << level_0_geom.isPeriodic(0) << " " << BCType::int_dir << " " << BCType::foextrap << std::endl;
    //exit(1);

    bcs.resize(1);     // Setup 1-component
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // lo-side BCs
        if (bc_lo[idim] == BCType::int_dir  ||  // periodic uses "internal Dirichlet"
            bc_lo[idim] == BCType::foextrap ||  // first-order extrapolation
            bc_lo[idim] == BCType::ext_dir ) {  // external Dirichlet
            bcs[0].setLo(idim, bc_lo[idim]);
        }
        else {
            amrex::Abort("Invalid bc_lo");
        }

        // hi-side BCSs
        if (bc_hi[idim] == BCType::int_dir  ||  // periodic uses "internal Dirichlet"
            bc_hi[idim] == BCType::foextrap ||  // first-order extrapolation
            bc_hi[idim] == BCType::ext_dir ) {  // external Dirichlet
            bcs[0].setHi(idim, bc_hi[idim]);
        }
        else {
            amrex::Abort("Invalid bc_hi");
        }
    }

    // stores fluxes at coarse-fine interface for synchronization
    // this will be sized "nlevs_max+1"
    // NOTE: the flux register associated with flux_reg[lev] is associated
    // with the lev/lev-1 interface (and has grid spacing associated with lev-1)
    // therefore flux_reg[0] is never actually used in the reflux operation
    flux_reg.resize(nlevs_max+1);

    // fillpatcher[lev] is for filling data on level lev using the data on
    // lev-1 and lev.
    fillpatcher.resize(nlevs_max+1);
}

AmrCoreAdv::~AmrCoreAdv () = default;

// advance solution to final time
void
AmrCoreAdv::Evolve ()
{
    Real cur_time = t_new[0];
    int last_plot_file_step = 0;

    int levmean = max_level;
    MultiFab mfmean;
    int test_fillpatchnlevels = 0;
    {
        ParmParse pp;
        pp.query("test_fillpatchnlevels", test_fillpatchnlevels);
    }
    if (test_fillpatchnlevels) {
        Box bxmean = geom[levmean].Domain();
        IntVect shrink = geom[levmean].Domain().length() / 4;
        bxmean.grow(-shrink);
        BoxArray bamean(bxmean);
        bamean.maxSize(32);
        mfmean.define(bamean,DistributionMapping{bamean},1,0);
        mfmean.setVal(0.0);
    }

    for (int step = istep[0]; step < max_step && cur_time < stop_time; ++step)
    {
        amrex::Print() << "\nCoarse STEP " << step+1 << " starts ..." << '\n';

        ComputeDt();

        int lev = 0;
        int iteration = 1;
        if (do_subcycle) {
            timeStepWithSubcycling_original(lev, cur_time, iteration);
        } else {
            timeStepNoSubcycling(cur_time, iteration);
        }

        cur_time += dt[0];

        // sum phi to check conservation
        Real sum_phi = phi_new[0].sum();

        amrex::Print() << "Coarse STEP " << step+1 << " ends." << " TIME = " << cur_time
                       << " DT = " << dt[0] << " Sum(Phi) = " << sum_phi << '\n';

        // sync up time
        for (lev = 0; lev <= finest_level; ++lev) {
            t_new[lev] = cur_time;
        }

        if (plot_int > 0 && (step+1) % plot_int == 0) {
            last_plot_file_step = step+1;
            WritePlotFile();
        }

        if (chk_int > 0 && (step+1) % chk_int == 0) {
            WriteCheckpointFile();
        }

#ifdef AMREX_MEM_PROFILING
        {
            std::ostringstream ss;
            ss << "[STEP " << step+1 << "]";
            MemProfiler::report(ss.str());
        }
#endif

        if (cur_time >= stop_time - 1.e-6*dt[0]) { break; }

        if (test_fillpatchnlevels)
        {
            MultiFab mftmp(mfmean.boxArray(), mfmean.DistributionMap(), 1, 0);

            CpuBndryFuncFab bndry_func(nullptr);
            Vector<PhysBCFunct<CpuBndryFuncFab>> physbcs;
            for (int ilev = 0; ilev <= max_level; ++ilev) {
                physbcs.emplace_back(geom[ilev],bcs,bndry_func);
            }
            Vector<Vector<MultiFab*>> smf(finest_level+1);
            Vector<Vector<Real>> st(finest_level+1);
            for (int ilev = 0; ilev <= finest_level; ++ilev) {
                smf[ilev].push_back(&phi_new[ilev]);
                st[ilev].push_back(0.0);
            }
            FillPatchNLevels(mftmp, levmean, IntVect(0), 0.0, smf, st, 0, 0, 1, geom,
                             physbcs, 0, refRatio(), &cell_cons_interp, bcs, 0);
            MultiFab::Add(mfmean, mftmp, 0, 0, 1, 0);
        }
    }

    if (test_fillpatchnlevels) {
        if (mfmean.is_finite()) {
            amrex::Print() << "\namrex::FillPatchNLevels test passed\n\n";
        } else {
            amrex::Abort("amrex::FillPatchNLevels test failed");
        }
    }

    if (plot_int > 0 && istep[0] > last_plot_file_step) {
        WritePlotFile();
    }
}

// initializes multilevel data
void
AmrCoreAdv::InitData ()
{
    if (restart_chkfile.empty()) {
        // start simulation from the beginning
        const Real time = 0.0;
        InitFromScratch(time);
        AverageDown();

#ifdef AMREX_PARTICLES
        if (do_tracers) {
            init_particles();
        }
#endif

        //if (chk_int > 0) {
        //    WriteCheckpointFile();
        //}
    }
    else {
        // restart from a checkpoint
        ReadCheckpointFile();
    }
    //if (plot_int > 0) {
    //    WritePlotFile();
    //}

    is_first = false;
}

// Make a new level using provided BoxArray and DistributionMapping and
// fill with interpolated coarse level data.
// overrides the pure virtual function in AmrCore
void
AmrCoreAdv::MakeNewLevelFromCoarse (int lev, Real time, const BoxArray& ba,
                                    const DistributionMapping& dm)
{
    const int ncomp = phi_new[lev-1].nComp();
    const int ng = phi_new[lev-1].nGrow();

    phi_new[lev].define(ba, dm, ncomp, ng);
    phi_old[lev].define(ba, dm, ncomp, ng);

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    // This clears the old MultiFab and allocates the new one
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
    {
        facevel[lev][idim] = MultiFab(amrex::convert(ba,IntVect::TheDimensionVector(idim)), dm, 1, nghost);
    }

    if (lev > 0 && do_reflux) {
        flux_reg[lev] = std::make_unique<FluxRegister>(ba, dm, refRatio(lev-1), lev, ncomp);
    }

    FillCoarsePatch(lev, time, phi_new[lev], 0, ncomp);
}

// Remake an existing level using provided BoxArray and DistributionMapping and
// fill with existing fine and coarse data.
// overrides the pure virtual function in AmrCore
void
AmrCoreAdv::RemakeLevel (int lev, Real time, const BoxArray& ba,
                         const DistributionMapping& dm)
{
    const int ncomp = phi_new[lev].nComp();
    const int ng    = phi_new[lev].nGrow();

    MultiFab new_state(ba, dm, ncomp, ng);
    MultiFab old_state(ba, dm, ncomp, ng);

    // Must use fillpatch_function
    FillPatch(lev, time, new_state, 0, ncomp, FillPatchType::fillpatch_function);

    std::swap(new_state, phi_new[lev]);
    std::swap(old_state, phi_old[lev]);

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    // This clears the old MultiFab and allocates the new one
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
    {
        facevel[lev][idim] = MultiFab(amrex::convert(ba,IntVect::TheDimensionVector(idim)), dm, 1, nghost);
    }

    if (lev > 0 && do_reflux) {
        flux_reg[lev] = std::make_unique<FluxRegister>(ba, dm, refRatio(lev-1), lev, ncomp);
    }
}

// Delete level data
// overrides the pure virtual function in AmrCore
void
AmrCoreAdv::ClearLevel (int lev)
{
    phi_new[lev].clear();
    phi_old[lev].clear();
    flux_reg[lev].reset(nullptr);
    fillpatcher[lev].reset(nullptr);
    //fillpatcher_ghost_boundary[lev].reset(nullptr);
}

// Make a new level from scratch using provided BoxArray and DistributionMapping.
// Only used during initialization.
// overrides the pure virtual function in AmrCore
void AmrCoreAdv::MakeNewLevelFromScratch (int lev, Real time, const BoxArray& ba,
                                          const DistributionMapping& dm)
{
    const int ncomp = 1;
    const int ng = 0;

    phi_new[lev].define(ba, dm, ncomp, ng);
    phi_old[lev].define(ba, dm, ncomp, ng);

    t_new[lev] = time;
    t_old[lev] = time - 1.e200;

    // This clears the old MultiFab and allocates the new one
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
    {
        facevel[lev][idim] = MultiFab(amrex::convert(ba,IntVect::TheDimensionVector(idim)), dm, 1, nghost);
    }

    if (lev > 0 && do_reflux) {
        flux_reg[lev] = std::make_unique<FluxRegister>(ba, dm, refRatio(lev-1), lev, ncomp);
    }

    MultiFab& state = phi_new[lev];

    const auto problo = Geom(lev).ProbLoArray();
    const auto dx     = Geom(lev).CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        //Array4<Real> fab_1 = state.array(mfi, NUM_COMPONENTS-2); // define here the other multifab array to set the initial condition
        Array4<Real> fab = state.array(mfi, NUM_COMPONENTS-1);
        const Box& box = mfi.tilebox();

        amrex::launch(box,
        [=] AMREX_GPU_DEVICE (Box const& tbx)
        {
            initdata(tbx, fab, problo, dx);
        });
    }
}

// tag all cells for refinement
// overrides the pure virtual function in AmrCore
void
AmrCoreAdv::ErrorEst (int lev, TagBoxArray& tags, Real /*time*/, int /*ngrow*/)
{
    static bool first = true;
    static Vector<Real> phierr;

    // only do this during the first call to ErrorEst
    if (first)
    {
        first = false;
        // read in an array of "phierr", which is the tagging threshold
        // in this example, we tag values of "phi" which are greater than phierr
        // for that particular level
        // in subroutine state_error, you could use more elaborate tagging, such
        // as more advanced logical expressions, or gradients, etc.
        ParmParse pp("adv");
        int n = pp.countval("phierr");
        if (n > 0) {
            pp.getarr("phierr", phierr, 0, n);
        }
    }

    if (lev >= phierr.size()) { return; }

//    const int clearval = TagBox::CLEAR;
    const int tagval = TagBox::SET;


    const auto& state      = phi_new     [lev]; 
          auto& tagger_lev = level_tagger[lev];

    if (is_first)
    {
        tagger_lev.define(phi_new[lev].boxArray(), phi_new[lev].DistributionMap(), phi_new[lev].nComp(), 1); // just consider 1 ghost cell!
        tagger_lev.setVal(0); // the tag 0 can be read as level 0
        tagger_lev.FillBoundary();
    }


    const auto problo = Geom(lev).ProbLoArray();
    const auto dx     = Geom(lev).CellSizeArray();

    const auto valid_start = geom[lev].Domain().smallEnd();
    const auto valid_end   = geom[lev].Domain().bigEnd();

    //std::cout << lev << " " << problo[0] << " " << problo[1] << std::endl;

#ifdef AMREX_USE_OMP
#pragma omp parallel if(Gpu::notInLaunchRegion())
#endif
    {

        for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box bx = mfi.tilebox();
            const Box bx_g = mfi.growntilebox();
            const auto statefab = state.const_array(mfi);
            const auto tagfab  = tags.array(mfi);
            Real phierror = phierr[lev];

            //std::cout << bx_g << " " << bx << std::endl;

            const auto taggerfab = tagger_lev.const_array(mfi);


            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                //const auto lo = lbound(bx);
                //const auto hi = ubound(bx);

                //std::cout << lev << " " << geom[0].Domain().bigEnd() << " " << bx.bigEnd() << " " << bx.smallEnd() << " " << valid_bx.bigEnd() << " " << valid_bx.smallEnd() << std::endl;

                Real x = problo[0] + (0.5+i)*dx[0];
                Real y = problo[1] + (0.5+j)*dx[1];

                //if (x>.45&&x<.5) tagfab(i,j,k) = tagval;
                //if (x>.45&&x<.55&&y>-.25&&y<.2) tagfab(i,j,k) = tagval;

                //if (x>0.25&&x<0.75&&y>0) tagfab(i,j,k) = tagval;
                //if (lev==0 && x>-.5) tagfab(i,j,k) = tagval; 
                //if (lev==1 && x>0) tagfab(i,j,k) = tagval; 
                //if (x>0.1) tagfab(i,j,k) = fine_tagval;  

                state_error(i, j, k, tagfab, statefab, taggerfab, phierror, tagval, valid_start, valid_end, is_first);
            });
        }
    }
}

// read in some parameters from inputs file
void
AmrCoreAdv::ReadParameters ()
{
    {
        ParmParse pp; // Traditionally, max_step and stop_time do not have prefix.
        pp.query("max_step", max_step);
        pp.query("stop_time", stop_time);
    }

    {
        ParmParse pp("amr"); // Traditionally, these have prefix, amr.

        pp.query("regrid_int", regrid_int);
        pp.query("plot_file", plot_file);
        pp.query("plot_int", plot_int);
        pp.query("chk_file", chk_file);
        pp.query("chk_int", chk_int);
        pp.query("restart",restart_chkfile);
    }

    {
        ParmParse pp("adv");

        pp.query("cfl", cfl);
        pp.query("do_reflux", do_reflux);
        pp.query("do_subcycle", do_subcycle);
    }

#ifdef AMREX_PARTICLES
    {
        ParmParse pp("amr");
        pp.query("do_tracers", do_tracers);
    }
#endif
}

// set covered coarse cells to be the average of overlying fine cells
void
AmrCoreAdv::AverageDown ()
{
    for (int lev = finest_level-1; lev >= 0; --lev)
    {
        amrex::average_down(phi_new[lev+1], phi_new[lev],
                            geom[lev+1], geom[lev],
                            0, phi_new[lev].nComp(), refRatio(lev));
    }
}

// more flexible version of AverageDown() that lets you average down across multiple levels
void
AmrCoreAdv::AverageDownTo (int crse_lev)
{
    amrex::average_down(phi_new[crse_lev+1], phi_new[crse_lev],
                        geom[crse_lev+1], geom[crse_lev],
                        0, phi_new[crse_lev].nComp(), refRatio(crse_lev));
}


// move
void
AmrCoreAdv::MoveMultiBlocks() {
    for (int lev=0; lev<=max_level; ++lev) {
        FillBoundary_ghost       [lev] = FillBoundaryFn{std::move(multi_block_boundaries       [lev])};
        FillBoundaryMarkers_ghost[lev] = FillBoundaryFn{std::move(multi_block_boundariesMarkers[lev])};
    }
}


// set the pointer to another core
void 
AmrCoreAdv::setOtherCore(const AmrCoreAdv* other_0, const AmrCoreAdv* other_1, const AmrCoreAdv* other_2, const AmrCoreAdv* other_3) {
    other_core[0] = other_0; // communication right
    other_core[1] = other_1; // communication left
    other_core[2] = other_2; // communication up
    other_core[3] = other_3; // communication down
}



void 
AmrCoreAdv::create_ghost_multifabs(int ng, int lev) {

     // consider in the future to put the .define outside the loop in time as the definition should be always the same, right?
    array_vec_mf_g[0][lev].define(other_core[0]->phi_new[lev].boxArray(), other_core[0]->phi_new[lev].DistributionMap(), other_core[0]->phi_new[lev].nComp(), ng);
    array_vec_mf_g[0][lev].setVal(0);
    array_vec_mf_g[0][lev].ParallelCopy(other_core[0]->phi_new[lev]);
    array_vec_mf_g[0][lev].FillBoundary();

    array_vec_mf_g[1][lev].define(other_core[1]->phi_new[lev].boxArray(), other_core[1]->phi_new[lev].DistributionMap(), other_core[1]->phi_new[lev].nComp(), ng);
    array_vec_mf_g[1][lev].setVal(0);
    array_vec_mf_g[1][lev].ParallelCopy(other_core[1]->phi_new[lev]);
    array_vec_mf_g[1][lev].FillBoundary();

    array_vec_mf_g[2][lev].define(other_core[2]->phi_new[lev].boxArray(), other_core[2]->phi_new[lev].DistributionMap(), other_core[2]->phi_new[lev].nComp(), ng);
    array_vec_mf_g[2][lev].setVal(0);
    array_vec_mf_g[2][lev].ParallelCopy(other_core[2]->phi_new[lev]);
    array_vec_mf_g[2][lev].FillBoundary();

    array_vec_mf_g[3][lev].define(other_core[3]->phi_new[lev].boxArray(), other_core[3]->phi_new[lev].DistributionMap(), other_core[3]->phi_new[lev].nComp(), ng);
    array_vec_mf_g[3][lev].setVal(0);
    array_vec_mf_g[3][lev].ParallelCopy(other_core[3]->phi_new[lev]);
    array_vec_mf_g[3][lev].FillBoundary();
}


//
void 
AmrCoreAdv::create_ghost_multifabs(int ng) {

    for (int lev = 0; lev <= finest_level; ++lev) {
        // consider in the future to put the .define outside the loop in time as the definition should be always the same, right?
        array_vec_mf_g[0][lev].define(other_core[0]->phi_new[lev].boxArray(), other_core[0]->phi_new[lev].DistributionMap(), other_core[0]->phi_new[lev].nComp(), ng);
        array_vec_mf_g[0][lev].setVal(0);
        array_vec_mf_g[0][lev].ParallelCopy(other_core[0]->phi_new[lev]);
        array_vec_mf_g[0][lev].FillBoundary();

        array_vec_mf_g[1][lev].define(other_core[1]->phi_new[lev].boxArray(), other_core[1]->phi_new[lev].DistributionMap(), other_core[1]->phi_new[lev].nComp(), ng);
        array_vec_mf_g[1][lev].setVal(0);
        array_vec_mf_g[1][lev].ParallelCopy(other_core[1]->phi_new[lev]);
        array_vec_mf_g[1][lev].FillBoundary();

        array_vec_mf_g[2][lev].define(other_core[2]->phi_new[lev].boxArray(), other_core[2]->phi_new[lev].DistributionMap(), other_core[2]->phi_new[lev].nComp(), ng);
        array_vec_mf_g[2][lev].setVal(0);
        array_vec_mf_g[2][lev].ParallelCopy(other_core[2]->phi_new[lev]);
        array_vec_mf_g[2][lev].FillBoundary();

        array_vec_mf_g[3][lev].define(other_core[3]->phi_new[lev].boxArray(), other_core[3]->phi_new[lev].DistributionMap(), other_core[3]->phi_new[lev].nComp(), ng);
        array_vec_mf_g[3][lev].setVal(0);
        array_vec_mf_g[3][lev].ParallelCopy(other_core[3]->phi_new[lev]);
        array_vec_mf_g[3][lev].FillBoundary();
    }
}



//
AMREX_NODISCARD CommHandler 
AmrCoreAdv::OnesidedMultiBlockBoundaryFn::FillBoundary_nowait(MultiFab& mf, std::array<amrex::Vector<amrex::MultiFab>, 4>& array_vec_mf_g_in, int lev) {
    if (!cmd || cached_dest_bd_key != mf.getBDKey() || cached_src_bd_key != array_vec_mf_g_in[i_boundary][lev].getBDKey()) {
        cmd = std::make_unique<MultiBlockCommMetaData>(mf, boundary_to_fill, array_vec_mf_g_in[i_boundary][lev], mf.nGrowVect(), dtos);
        cached_dest_bd_key = mf.getBDKey();
        cached_src_bd_key = array_vec_mf_g_in[i_boundary][lev].getBDKey();
    }

    return ParallelCopy_nowait(amrex::NonLocalBC::no_local_copy, mf, array_vec_mf_g_in[i_boundary][lev], *cmd, packing);
}

AMREX_NODISCARD CommHandler 
AmrCoreAdv::OnesidedMultiBlockBoundaryFn::FillBoundary_nowait(MultiFab& mf, const std::array<const AmrCoreAdv*, 4>& neigh_cores, int lev) {
    if (!cmd || cached_dest_bd_key != mf.getBDKey() || cached_src_bd_key != neigh_cores[i_boundary]->level_tagger[lev].getBDKey()) {
        cmd = std::make_unique<MultiBlockCommMetaData>(mf, boundary_to_fill, neigh_cores[i_boundary]->level_tagger[lev], mf.nGrowVect(), dtos);
        cached_dest_bd_key = mf.getBDKey();
        cached_src_bd_key = neigh_cores[i_boundary]->level_tagger[lev].getBDKey();
    }

    return ParallelCopy_nowait(amrex::NonLocalBC::no_local_copy, mf, neigh_cores[i_boundary]->level_tagger[lev], *cmd, packing);
}

void 
AmrCoreAdv::OnesidedMultiBlockBoundaryFn::FillBoundary_do_local_copy(MultiFab& mf, std::array<amrex::Vector<amrex::MultiFab>, 4>& array_vec_mf_g_in, int lev) const {
    AMREX_ASSERT(cmd && cached_dest_bd_key == mf.getBDKey() && cached_src_bd_key == array_vec_mf_g_in[i_boundary][lev].getBDKey());
    if (cmd->m_LocTags && !cmd->m_LocTags->empty()) {
        LocalCopy(packing, mf, array_vec_mf_g_in[i_boundary][lev], *cmd->m_LocTags);
    }
}

void 
AmrCoreAdv::OnesidedMultiBlockBoundaryFn::FillBoundary_do_local_copy(MultiFab& mf, const std::array<const AmrCoreAdv*, 4>& neigh_cores, int lev) const {
    AMREX_ASSERT(cmd && cached_dest_bd_key == mf.getBDKey() && cached_src_bd_key == neigh_cores[i_boundary]->level_tagger[lev].getBDKey());
    if (cmd->m_LocTags && !cmd->m_LocTags->empty()) {
        LocalCopy(packing, mf, neigh_cores[i_boundary]->level_tagger[lev], *cmd->m_LocTags);
    }
}

void 
AmrCoreAdv::OnesidedMultiBlockBoundaryFn::FillBoundary_finish(MultiFab& mf, CommHandler handler) const {
    ParallelCopy_finish(mf, std::move(handler), *cmd, packing); // NOLINT(performance-move-const-arg)
}


void 
AmrCoreAdv::FillBoundaryFn::operator()(MultiFab& mf, std::array<amrex::Vector<amrex::MultiFab>, 4>& array_vec_mf_g_in, int lev) {

    std::vector<CommHandler> comms;
    const std::size_t n_boundaries = boundaries.size();
    comms.reserve(n_boundaries);
    for (auto& boundary : boundaries) {
        comms.emplace_back(boundary.FillBoundary_nowait(mf, array_vec_mf_g_in, lev));
    }
    for (auto& boundary : boundaries) {
        boundary.FillBoundary_do_local_copy(mf, array_vec_mf_g_in, lev);
    }
    for (std::size_t i = 0; i < n_boundaries; ++i) {
        boundaries[i].FillBoundary_finish(mf, std::move(comms[i])); // NOLINT(performance-move-const-arg)
    }
}

void 
AmrCoreAdv::FillBoundaryFn::operator()(MultiFab& mf, const std::array<const AmrCoreAdv*, 4>& neigh_cores, int lev) {

    std::vector<CommHandler> comms;
    const std::size_t n_boundaries = boundaries.size();
    comms.reserve(n_boundaries);
    for (auto& boundary : boundaries) {
        comms.emplace_back(boundary.FillBoundary_nowait(mf, neigh_cores, lev));
    }
    for (auto& boundary : boundaries) {
        boundary.FillBoundary_do_local_copy(mf, neigh_cores, lev);
    }
    for (std::size_t i = 0; i < n_boundaries; ++i) {
        boundaries[i].FillBoundary_finish(mf, std::move(comms[i])); // NOLINT(performance-move-const-arg)
    }
}

void dummy_cpu_fill_extdir (Box const& bx, Array4<Real> const& dest,
                                const int dcomp, const int numcomp,
                                GeometryData const& geom, const Real /*time*/,
                                const BCRec* bcr, const int bcomp,
                                const int /*orig_comp*/)
{
    // do something for external Dirichlet (BCType::ext_dir or BCType::ext_dir_cc) if there are
    const auto dom_lo = amrex::lbound(geom.Domain());
    const auto dom_hi = amrex::ubound(geom.Domain());

    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);

#if (AMREX_SPACEDIM == 3)
    for (int k = lo.z; k <= hi.z; ++k)
#endif
#if (AMREX_SPACEDIM >= 2)
    for (int j = lo.y; j <= hi.y; ++j)
#endif
    for (int i = lo.x; i <= hi.x; ++i)
    {
        IntVect iv(AMREX_D_DECL(i, j, k));
        for (int n = 0; n < numcomp; ++n)
        {
            const int comp = dcomp + n;
            const auto bc = bcr[bcomp + n];

            bool is_ext_dir = false;
#if (AMREX_SPACEDIM >= 1)
            if ((i < dom_lo.x && (bc.lo(0) == BCType::ext_dir || bc.lo(0) == BCType::ext_dir_cc)) ||
                (i > dom_hi.x && (bc.hi(0) == BCType::ext_dir || bc.hi(0) == BCType::ext_dir_cc)))
                is_ext_dir = true;
#endif
#if (AMREX_SPACEDIM >= 2)
            if ((j < dom_lo.y && (bc.lo(1) == BCType::ext_dir || bc.lo(1) == BCType::ext_dir_cc)) ||
                (j > dom_hi.y && (bc.hi(1) == BCType::ext_dir || bc.hi(1) == BCType::ext_dir_cc)))
                is_ext_dir = true;
#endif
#if (AMREX_SPACEDIM == 3)
            if ((k < dom_lo.z && (bc.lo(2) == BCType::ext_dir || bc.lo(2) == BCType::ext_dir_cc)) ||
                (k > dom_hi.z && (bc.hi(2) == BCType::ext_dir || bc.hi(2) == BCType::ext_dir_cc)))
                is_ext_dir = true;
#endif

            if (is_ext_dir)
            {
                dest(iv, comp) = 0.0;
            }
        }
    }
}


// compute a new multifab by coping in phi from valid region and filling ghost cells
// works for single level and 2-level cases (fill fine grid ghost by interpolating from coarse)
void
AmrCoreAdv::FillPatch (int lev, Real time, MultiFab& mf, int icomp, int ncomp,
                       FillPatchType fptype)
{

    // Modify here for the boundary conditions
    if (lev == 0)
    {
        Vector<MultiFab*> smf;
        Vector<Real> stime;
        GetData(0, time, smf, stime);

        if(Gpu::inLaunchRegion())
        {
            GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
            PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > physbc(geom[lev],bcs,gpu_bndry_func);  // here are used the BCs
            amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp, geom[lev], physbc, 0);
        }
        else
        {
            CpuBndryFuncFab bndry_func(dummy_cpu_fill_extdir);  // Without EXT_DIR, we can pass a nullptr.
            PhysBCFunct<CpuBndryFuncFab> physbc(geom[lev],bcs,bndry_func);  // here are used the BCs
            amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp, geom[lev], physbc, 0);
        }
    }
    else
    {
        Vector<MultiFab*> cmf, fmf;
        Vector<Real> ctime, ftime;
        GetData(lev-1, time, cmf, ctime);
        GetData(lev  , time, fmf, ftime);

        Interpolater* mapper = &cell_cons_interp; // maybe would need to chage the interpolator in the future, look at AMReX_Interpolater.H/.cpp

        if (fptype == FillPatchType::fillpatch_class) {
            if (fillpatcher[lev] == nullptr) {
                fillpatcher[lev] = std::make_unique<FillPatcher<MultiFab>>
                    (boxArray(lev  ), DistributionMap(lev  ), Geom(lev  ),
                     boxArray(lev-1), DistributionMap(lev-1), Geom(lev-1),
                     mf.nGrowVect(), mf.nComp(), mapper);
            }
        }

        if(Gpu::inLaunchRegion())
        {
            GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
            PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > cphysbc(geom[lev-1],bcs,gpu_bndry_func);
            PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > fphysbc(geom[lev],bcs,gpu_bndry_func);

            if (fptype == FillPatchType::fillpatch_class) {
                fillpatcher[lev]->fill(mf, mf.nGrowVect(), time,
                                       cmf, ctime, fmf, ftime, 0, icomp, ncomp,
                                       cphysbc, 0, fphysbc, 0, bcs, 0);
            } else {
                amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,
                                          0, icomp, ncomp, geom[lev-1], geom[lev],
                                          cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                          mapper, bcs, 0);

            }
        }
        else
        {
            CpuBndryFuncFab bndry_func(dummy_cpu_fill_extdir);  // Without EXT_DIR, we can pass a nullptr.
            PhysBCFunct<CpuBndryFuncFab> cphysbc(geom[lev-1],bcs,bndry_func); // coarse
            PhysBCFunct<CpuBndryFuncFab> fphysbc(geom[lev],bcs,bndry_func);   // fine

            if (fptype == FillPatchType::fillpatch_class) {
                fillpatcher[lev]->fill(mf, mf.nGrowVect(), time,
                                       cmf, ctime, fmf, ftime, 0, icomp, ncomp,
                                       cphysbc, 0, fphysbc, 0, bcs, 0);
            } else {
                amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime,
                                          0, icomp, ncomp, geom[lev-1], geom[lev],
                                          cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                          mapper, bcs, 0);
            }
        }
    }
    //std::cout << mf.nGrow() << std::endl;
    // fill now the ghost cells of mf
    if (mf.nGrow()==array_vec_mf_g[0][lev].nGrow()) // this check is mandatory because also the regrid function calls this function
    {
        FillBoundary_ghost[lev](mf, array_vec_mf_g, lev);
    }
}

// fill an entire multifab by interpolating from the coarser level
// this comes into play when a new level of refinement appears
void
AmrCoreAdv::FillCoarsePatch (int lev, Real time, MultiFab& mf, int icomp, int ncomp)
{
    // Modify here for the BCs
    BL_ASSERT(lev > 0);

    Vector<MultiFab*> cmf;
    Vector<Real> ctime;
    GetData(lev-1, time, cmf, ctime);
    Interpolater* mapper = &cell_cons_interp;

    if (cmf.size() != 1) {
        amrex::Abort("FillCoarsePatch: how did this happen?");
    }

    if(Gpu::inLaunchRegion())
    {
        GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
        PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > cphysbc(geom[lev-1],bcs,gpu_bndry_func);
        PhysBCFunct<GpuBndryFuncFab<AmrCoreFill> > fphysbc(geom[lev],bcs,gpu_bndry_func);

        amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev-1], geom[lev],
                                     cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                     mapper, bcs, 0);
    }
    else
    {
        CpuBndryFuncFab bndry_func(nullptr);  // Without EXT_DIR, we can pass a nullptr.
        PhysBCFunct<CpuBndryFuncFab> cphysbc(geom[lev-1],bcs,bndry_func);
        PhysBCFunct<CpuBndryFuncFab> fphysbc(geom[lev],bcs,bndry_func);

        amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp, geom[lev-1], geom[lev],
                                     cphysbc, 0, fphysbc, 0, refRatio(lev-1),
                                     mapper, bcs, 0);
    }
    //exit(1);
    // fill now the ghost cells of mf
    if (mf.nGrow()==array_vec_mf_g[0][lev].nGrow())
    {
        FillBoundary_ghost[lev](mf, array_vec_mf_g, lev);
    }
}

void
AmrCoreAdv::GetData (int lev, Real time, Vector<MultiFab*>& data, Vector<Real>& datatime)
{
    data.clear();
    datatime.clear();

    if (amrex::almostEqual(time, t_new[lev], 5))
    {
        data.push_back(&phi_new[lev]);
        datatime.push_back(t_new[lev]);
    }
    else if (amrex::almostEqual(time, t_old[lev], 5))
    {
        data.push_back(&phi_old[lev]);
        datatime.push_back(t_old[lev]);
    }
    else
    {
        data.push_back(&phi_old[lev]);
        data.push_back(&phi_new[lev]);
        datatime.push_back(t_old[lev]);
        datatime.push_back(t_new[lev]);
    }
}

void 
AmrCoreAdv::perform_regrid(Real time)
{
    if (max_level > 0 && regrid_int > 0)  // We may need to regrid
    {
        if (istep[0] % regrid_int == 0)
        {
            regrid(0, time);
#ifdef AMREX_PARTICLES
            if (do_tracers)
            {
                    TracerPC->Redistribute();
            }
#endif
        }
    }
}

void
AmrCoreAdv::reset_level_tagger()
{
    for (int lev=0; lev<=finest_level; lev++) 
    {
        // reset here first 
	level_tagger[lev].define(phi_new[lev].boxArray(), phi_new[lev].DistributionMap(), phi_new[lev].nComp(), 1);
        level_tagger[lev].setVal(0); // the tag 0 can be read as level 
	level_tagger[lev].FillBoundary();
    } 
}



void
AmrCoreAdv::check_finer(int lev)
{
    if ((lev+1)==finest_level)
    {
        // the finest level is alredy initialized to 0 because there isn't anything behind it!
        level_tagger[finest_level].define(phi_new[finest_level].boxArray(), phi_new[finest_level].DistributionMap(), phi_new[finest_level].nComp(), 1); // just consider 1 ghost cell!
        level_tagger[finest_level].setVal(0); // the tag 0 can be read as level 0
        level_tagger[finest_level].FillBoundary();
    }

    IntVect ones_vect = IntVect(AMREX_D_DECL(1, 1, 1));
    {
              auto& current_tag_lev = level_tagger[lev];
        const auto& current_sol_lev = phi_new     [lev];

        // reset here first 
        current_tag_lev.define(current_sol_lev.boxArray(), current_sol_lev.DistributionMap(), current_sol_lev.nComp(), 1); // just consider 1 ghost cell!
        current_tag_lev.setVal(0); // the tag 0 can be read as level 0
        current_tag_lev.FillBoundary();

        for (MFIter mfi(current_tag_lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.validbox();
            const auto& fab = current_tag_lev.array(mfi);  // Access MultiFab data
            const BoxArray& finer_grids = boxArray(lev+1);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Compute the equivalent index in the finer level
                const auto current_index = IntVect(AMREX_D_DECL(i, j, k));
                auto fine_idx = current_index*refRatio(lev); // we take the left index of the tree structure at fine level

		//std::cout << refRatio(lev) << std::endl;

                bool covered_by_finer = false;
                for (int nb = 0; nb < finer_grids.size(); ++nb)
                {
                    if (finer_grids[nb].contains(fine_idx))
                    {
                        covered_by_finer = true;
                        break;
                    }
                }

                if (covered_by_finer)
                {
                    fab(current_index) = 1;  // Mark cells that have a finer level behind
                }
            });
            
        }
    }
}



void
AmrCoreAdv::check_finer()
{
    // the finest level is alredy initialized to 0 because there isn't anything behind it!
    level_tagger[finest_level].define(phi_new[finest_level].boxArray(), phi_new[finest_level].DistributionMap(), phi_new[finest_level].nComp(), 1); // just consider 1 ghost cell!
    level_tagger[finest_level].setVal(0); // the tag 0 can be read as level 0
    level_tagger[finest_level].FillBoundary();

    IntVect ones_vect = IntVect(AMREX_D_DECL(1, 1, 1));
    for (int lev=0; lev<finest_level; lev++) 
    {
              auto& current_tag_lev = level_tagger[lev];
        const auto& current_sol_lev = phi_new     [lev];

        // reset here first 
        current_tag_lev.define(current_sol_lev.boxArray(), current_sol_lev.DistributionMap(), current_sol_lev.nComp(), 1); // just consider 1 ghost cell!
        current_tag_lev.setVal(0); // the tag 0 can be read as level 0
        current_tag_lev.FillBoundary();

        for (MFIter mfi(current_tag_lev, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.validbox();
            const auto& fab = current_tag_lev.array(mfi);  // Access MultiFab data
            const BoxArray& finer_grids = boxArray(lev+1);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Compute the equivalent index in the finer level
                const auto current_index = IntVect(AMREX_D_DECL(i, j, k));
                auto fine_idx = current_index*refRatio(lev); // we take the left index of the tree structure at fine level

		//std::cout << refRatio(lev) << std::endl;
                
                for (int nb = 0; nb < finer_grids.size(); ++nb)
                {
                    if (finer_grids[nb].contains(fine_idx))
                    {	
                        fab(current_index) = 1;  // Mark cells that have a finer level behind
			break;
                    }
                }
            });
            
        }
    }

}


void
AmrCoreAdv::check_finer_communication(int lev)
{
    FillBoundaryMarkers_ghost[lev](level_tagger[lev], other_core, lev);
}

void
AmrCoreAdv::check_finer_communication()
{
    for (int lev=0; lev<=finest_level; lev++)
    {
        FillBoundaryMarkers_ghost[lev](level_tagger[lev], other_core, lev);
    }
}

void
AmrCoreAdv::perform_regridWithSubcycling(int lev, bool is_last_ref)
{
    const auto time = t_new[lev];
    if (regrid_int > 0)  // We may need to regrid
    {

        // help keep track of whether a level was already regridded
        // from a coarser level call to regrid
        //static amrex::Vector<int> last_regrid_step(max_level+1, 0);

        // regrid changes level "lev+1" so we don't regrid on max_level
        // also make sure we don't regrid fine levels again if
        // it was taken care of during a coarser regrid

	//std::cout << "aaaaaaa" << std::endl;
	//std::cout << lev << " " << istep[lev] << " " << last_regrid_step[lev] << std::endl;
        if (lev < max_level && istep[lev] > last_regrid_step[lev])
        {
            
	    //std::cout << lev << " " << istep[lev] << " " << last_regrid_step[lev] << std::endl;
            if (istep[lev] % regrid_int == 0)
            {
                // regrid could add newly refined levels (if finest_level < max_level)
                // so we save the previous finest level index
                int old_finest = finest_level;
                regrid(lev, time); // the mesh adaption is carried out just here

                // mark that we have regridded this level already
		if (is_last_ref)
		{
                    for (int k = lev; k <= finest_level; ++k) {
                        last_regrid_step[k] = istep[k];
                    }
		}

                // if there are newly created levels, set the time step
                for (int k = old_finest+1; k <= finest_level; ++k) {
                    dt[k] = dt[k-1] / MaxRefRatio(k-1);
                }
/*
#ifdef AMREX_PARTICLES
                if (do_tracers) {
                    TracerPC->Redistribute(lev);
                }
#endif */
            }
        }
    }
}

// Advance a level by dt
// (includes a recursive call for finer levels)
void
AmrCoreAdv::timeStepWithSubcycling_original (int lev, Real time, int iteration)
{
    
    //std::cout << index_core << " " << regrid_int << std::endl;
    if (regrid_int > 0)  // We may need to regrid
    {

        // help keep track of whether a level was already regridded
        // from a coarser level call to regrid
        //static amrex::Vector<int> last_regrid_step(max_level+1, 0);

        // regrid changes level "lev+1" so we don't regrid on max_level
        // also make sure we don't regrid fine levels again if
        // it was taken care of during a coarser regrid

        if (lev < max_level && istep[lev] > last_regrid_step[lev])
        {
            if (istep[lev] % regrid_int == 0)
            { 
                // regrid could add newly refined levels (if finest_level < max_level)
                // so we save the previous finest level index
                int old_finest = finest_level;
                regrid(lev, time); // the mesh adaption is carried out just here

                // mark that we have regridded this level already
                for (int k = lev; k <= finest_level; ++k) {
                    last_regrid_step[k] = istep[k];
                }

                // if there are newly created levels, set the time step
                for (int k = old_finest+1; k <= finest_level; ++k) {
                    dt[k] = dt[k-1] / MaxRefRatio(k-1);
                }
/*
#ifdef AMREX_PARTICLES
                if (do_tracers) {
                    TracerPC->Redistribute(lev);
                }
#endif */
            }
        }
    }

    if (Verbose()) {
        amrex::Print() << "[Core " << index_core << " level " << lev << " step " << istep[lev]+1 << "] ";
        amrex::Print() << "ADVANCE with time = " << t_new[lev]
                       << " dt = " << dt[lev] << " iteration " << iteration << " time " << time << '\n';
    }

    // Advance a single level for a single time step, and update flux registers
    t_old[lev] = t_new[lev];
    t_new[lev] += dt[lev];

    Real t_nph = t_old[lev] + 0.5*dt[lev];

    // here is the core part of the numerical scheme
    DefineVelocityAtLevel(lev, t_nph);
    AdvancePhiAtLevel(lev, time, dt[lev], iteration, nsubsteps[lev]);
/*
#ifdef AMREX_PARTICLES
    if (do_tracers) {
        TracerPC->AdvectWithUmac(facevel[lev].data(),lev,dt[lev]);
    }
#endif
*/
    ++istep[lev];

    if (Verbose()) 
    {
        amrex::Print() << "[Core " << index_core << " level " << lev << " step " << istep[lev] << "] ";
        amrex::Print() << "Advanced " << CountCells(lev) << " cells " << "iteration " << iteration << " time " << time << '\n';
    }

    if (lev < finest_level)
    {
        // recursive call for next-finer level
        for (int i = 1; i <= nsubsteps[lev+1]; ++i)
        {
            timeStepWithSubcycling_original(lev+1, time+(i-1)*dt[lev+1], i);
        }
std::cout << index_core << " " << lev << std::endl;
     
	if (do_reflux)
        {
            // update lev based on coarse-fine flux mismatch
            flux_reg[lev+1]->Reflux(phi_new[lev], 1.0, 0, 0, phi_new[lev].nComp(), geom[lev]);
        } 

        AverageDownTo(lev); // average lev+1 down to lev

        fillpatcher[lev+1].reset(); // Because the data on lev have changed.
   
    }
//std::cout << index_core << " " << lev << std::endl;
/*
#ifdef AMREX_PARTICLES
    if (do_tracers) {
        int redistribute_ngrow = 0;
        if ((iteration < nsubsteps[lev]) || (lev == 0)){
            if (lev == 0){
                redistribute_ngrow = 0;
            } else {
                redistribute_ngrow = iteration;
            }
            TracerPC->Redistribute(lev, TracerPC->finestLevel(), redistribute_ngrow);
        }
    }
#endif
*/
}

void
AmrCoreAdv::perform_reflux_across_lev(int lev)
{
    if (do_reflux && flux_reg[lev+1])
    {
        // update lev based on coarse-fine flux mismatch
        flux_reg[lev+1]->Reflux(phi_new[lev], 1.0, 0, 0, phi_new[lev].nComp(), geom[lev]);
    }
    AverageDownTo(lev); // average lev+1 down to lev
    fillpatcher[lev+1].reset(); // Because the data on lev have changed.
}


// Advance a level by dt
// (includes a recursive call for finer levels)
void
AmrCoreAdv::timeStepWithSubcycling (int lev, int iteration)
{
/*
    //std::cout << index_core << " " << regrid_int << std::endl;
    if (regrid_int > 0)  // We may need to regrid
    {

        // help keep track of whether a level was already regridded
        // from a coarser level call to regrid
        //static amrex::Vector<int> last_regrid_step(max_level+1, 0);

        // regrid changes level "lev+1" so we don't regrid on max_level
        // also make sure we don't regrid fine levels again if
        // it was taken care of during a coarser regrid

        //std::cout << index_core << " ///////////////////////////////////////  " << lev << " " << max_level << " " <<  istep[lev] << " " << last_regrid_step[lev] << " " << last_regrid_step.size() << std::endl;
        if (lev < max_level && istep[lev] > last_regrid_step[lev])
        {
            if (istep[lev] % regrid_int == 0)
            { 
                // regrid could add newly refine levels (if finest_level < max_level)
                // so we save the previous finest level index
                int old_finest = finest_level;
                regrid(lev, time); // the mesh adaption is carried on just here

                // mark that we have regridded this level already
                for (int k = lev; k <= finest_level; ++k) {
                    last_regrid_step[k] = istep[k];
                }

                // if there are newly created levels, set the time step
                for (int k = old_finest+1; k <= finest_level; ++k) {
                    dt[k] = dt[k-1] / MaxRefRatio(k-1);
                }

#ifdef AMREX_PARTICLES
                if (do_tracers) {
                    TracerPC->Redistribute(lev);
                }
#endif
            }
        }
    }
*/
    const auto time = t_new[lev];

    if (Verbose()) {
        amrex::Print() << "[Core " << index_core << " level " << lev << " step " << istep[lev]+1 << "] ";
        amrex::Print() << "ADVANCE with time = " << t_new[lev]
                       << " dt = " << dt[lev] << " iteration " << iteration << " time " << time << '\n';
    }

    // Advance a single level for a single time step, and update flux registers

    t_old[lev] = t_new[lev];
    t_new[lev] += dt[lev];

    Real t_nph = t_old[lev] + 0.5*dt[lev];

    // here is the core part of the numerical scheme
    DefineVelocityAtLevel(lev, t_nph);
    AdvancePhiAtLevel(lev, time, dt[lev], iteration, nsubsteps[lev]);
/*
#ifdef AMREX_PARTICLES
    if (do_tracers) {
        TracerPC->AdvectWithUmac(facevel[lev].data(),lev,dt[lev]);
    }
#endif
*/
    ++istep[lev];

    if (Verbose())
    {
        amrex::Print() << "[Core " << index_core << " level " << lev << " step " << istep[lev] << "] ";
        amrex::Print() << "Advanced " << CountCells(lev) << " cells" << " iteration " << iteration << " time " << time << '\n';
    }
/*
    if (lev < finest_level)
    {
        // recursive call for next-finer level
        for (int i = 1; i <= nsubsteps[lev+1]; ++i)
        {
            timeStepWithSubcycling(lev+1, time+(i-1)*dt[lev+1], i);
        }

        if (do_reflux)
        {
            // update lev based on coarse-fine flux mismatch
            flux_reg[lev+1]->Reflux(phi_new[lev], 1.0, 0, 0, phi_new[lev].nComp(), geom[lev]);
        }

        AverageDownTo(lev); // average lev+1 down to lev

        fillpatcher[lev+1].reset(); // Because the data on lev have changed.
    }


#ifdef AMREX_PARTICLES
    if (do_tracers) {
        int redistribute_ngrow = 0;
        if ((iteration < nsubsteps[lev]) || (lev == 0)){
            if (lev == 0){
                redistribute_ngrow = 0;
            } else {
                redistribute_ngrow = iteration;
            }
            TracerPC->Redistribute(lev, TracerPC->finestLevel(), redistribute_ngrow);
        }
    }
#endif
*/
}

void
AmrCoreAdv::particles_tracer(int lev, int iteration)
{
#ifdef AMREX_PARTICLES
    if (do_tracers) {
        int redistribute_ngrow = 0;
        if ((iteration < nsubsteps[lev]) || (lev == 0)){
            if (lev == 0){
                redistribute_ngrow = 0;
            } else {
                redistribute_ngrow = iteration;
            }
            TracerPC->Redistribute(lev, TracerPC->finestLevel(), redistribute_ngrow);
        }
    }
#endif
}

// Advance all the levels with the same dt
void
AmrCoreAdv::timeStepNoSubcycling (Real time, int iteration)
{

    if (Verbose()) {
        for (int lev = 0; lev <= finest_level; lev++)
        {
           amrex::Print() << "[Level " << lev << " step " << istep[lev]+1 << "] ";
           amrex::Print() << "ADVANCE with time = " << t_new[lev]
                          << " dt = " << dt[0] << '\n';
        }
    }

    DefineVelocityAllLevels(time+0.5_rt*dt[0]);
    AdvancePhiAllLevels (time, dt[0], iteration);

#ifdef AMREX_PARTICLES
    if (do_tracers) {
        for (int lev = 0; lev <= finest_level; lev++)
        {
            TracerPC->AdvectWithUmac(facevel[lev].data(),lev,dt[0]);
        }
        TracerPC->Redistribute();
    }
#endif

    // Make sure the coarser levels are consistent with the finer levels
    AverageDown ();

    for (auto& fp : fillpatcher) {
        fp.reset(); // Because the data have changed.
    }

    for (int lev = 0; lev <= max_level; lev++) { //(int lev = 0; lev <= finest_level; lev++) {
        ++istep[lev];
    }

    if (Verbose())
    {
        for (int lev = 0; lev <= finest_level; lev++)
        {
            amrex::Print() << "[Level " << lev << " step " << istep[lev] << "] ";
            amrex::Print() << "Advanced " << CountCells(lev) << " cells" << '\n';
        }
    }
}

// Getter functions
const int&
AmrCoreAdv::getFinestLevel() const {
    return finest_level;
}
const int& 
AmrCoreAdv::getNsubsteps(int lev) const {
    return nsubsteps[lev];
}
const Real&
AmrCoreAdv::getLevel0Dt() const {
    return dt[0];
}
const Real& 
AmrCoreAdv::getLevelLevDt(int lev) const {
    return dt[lev];
}
const Real& 
AmrCoreAdv::getTnewLev(int lev) const {
    return t_new[lev];
}
const int& 
AmrCoreAdv::getChk_int() const {
    return chk_int;
}
const int& 
AmrCoreAdv::getPlot_int() const {
    return plot_int;
}

// Setter functions 
void 
AmrCoreAdv::setDtWithSubcycling() {
    for (int lev = 1; lev <= finest_level; ++lev) {
        dt[lev] = dt[lev-1]/nsubsteps[lev];
    }
}
void 
AmrCoreAdv::setDtNoSubcycling() {
    for (int lev = 1; lev <= finest_level; ++lev) {
        dt[lev] = dt[lev-1];
    }
}
void
AmrCoreAdv::setLevel0Dt(Real min_dt) {
    dt[0] = min_dt;
}
void
AmrCoreAdv::setTnewAllLev (Real cur_time) {
    for (int lev = 0; lev < finest_level; ++lev) {
        t_new[lev] = cur_time;
    }
}

//
Real
AmrCoreAdv::computeSumLevel0() {
    return phi_new[0].sum();
}

// a wrapper for EstTimeStep
void
AmrCoreAdv::ComputeDt ()
{
    Vector<Real> dt_tmp(finest_level+1);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        dt_tmp[lev] = EstTimeStep(lev, t_new[lev]);
    }
    ParallelDescriptor::ReduceRealMin(dt_tmp.data(), int(dt_tmp.size()));

    constexpr Real change_max = 1.1;
    Real dt_0 = dt_tmp[0];
    int n_factor = 1;

    for (int lev = 0; lev <= finest_level; ++lev) {
        dt_tmp[lev] = std::min(dt_tmp[lev], change_max*dt[lev]);
        n_factor *= nsubsteps[lev];
        dt_0 = std::min(dt_0, n_factor*dt_tmp[lev]);
    }

    // Limit dt's by the value of stop_time.
    const Real eps = 1.e-3*dt_0;

    if (t_new[0] + dt_0 > stop_time - eps) {
        dt_0 = stop_time - t_new[0];
    }

    dt[0] = dt_0;
    setDtWithSubcycling();
}

// compute dt from CFL considerations
Real
AmrCoreAdv::EstTimeStep (int lev, Real time)
{
    BL_PROFILE("AmrCoreAdv::EstTimeStep()");

    Real dt_est = std::numeric_limits<Real>::max();

    const Real* dx  =  geom[lev].CellSize();

    if (time == Real(0.0)) {
       DefineVelocityAtLevel(lev,time);
    } else {
       Real t_nph_predicted = time + 0.5 * dt[lev];
       DefineVelocityAtLevel(lev,t_nph_predicted);
    }

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        Real est = facevel[lev][idim].norminf(0,0,true);
        dt_est = amrex::min(dt_est, dx[idim]/est);
    }

    dt_est *= cfl;

    return dt_est;
}

// get plotfile name
std::string
AmrCoreAdv::PlotFileName (int lev) const
{
    auto plt_out = core_prefix + std::to_string(index_core) + "/" + plot_file;
    return amrex::Concatenate(plt_out, lev, 5);
} 

// put together an array of multifabs for writing
Vector<const MultiFab*>
AmrCoreAdv::PlotFileMF () const
{
    Vector<const MultiFab*> r;
    for (int i = 0; i <= finest_level; ++i) {
        r.push_back(&phi_new[i]);

        //std::cout << refRatio(i) << std::endl; 

        //long num_cells = 0;
        //for (amrex::MFIter mfi(phi_new[i]); mfi.isValid(); ++mfi) {
        //    num_cells += mfi.validbox().numPts();  // Get the number of cells in each grid
        //}
        //amrex::Print() << "Total number of cells at level " << i << ": " << num_cells << "\n";
    }
    return r;
}

// set plotfile variable names
Vector<std::string>
AmrCoreAdv::PlotFileVarNames ()
{
    return {"phi"};
}

// write plotfile to disk
void
AmrCoreAdv::WritePlotFile () const
{
    const std::string& plotfilename = PlotFileName(istep[0]);
    const auto& mf = PlotFileMF();
    const auto& varnames = PlotFileVarNames();

    amrex::Print() << "Writing plotfile " << plotfilename << "\n";

    amrex::WriteMultiLevelPlotfile(plotfilename, finest_level+1, mf, varnames,
                                   Geom(), t_new[0], istep, refRatio());

#ifdef AMREX_PARTICLES
        if (do_tracers) {
            TracerPC->WritePlotFile(plotfilename, "particles");
        }
#endif
}

void
AmrCoreAdv::WriteCheckpointFile () const
{

    // chk00010            write a checkpoint file with this root directory
    // chk00010/Header     this contains information you need to save (e.g., finest_level, t_new, etc.) and also
    //                     the BoxArrays at each level
    // chk00010/Level_0/
    // chk00010/Level_1/
    // etc.                these subdirectories will hold the MultiFab data at each level of refinement

    // checkpoint file name, e.g., chk00010
    const std::string& checkpointname = amrex::Concatenate(chk_file,istep[0]);

    amrex::Print() << "Writing checkpoint " << checkpointname << "\n";

    const int nlevels = finest_level+1;

    // ---- prebuild a hierarchy of directories
    // ---- dirName is built first.  if dirName exists, it is renamed.  then build
    // ---- dirName/subDirPrefix_0 .. dirName/subDirPrefix_nlevels-1
    // ---- if callBarrier is true, call ParallelDescriptor::Barrier()
    // ---- after all directories are built
    // ---- ParallelDescriptor::IOProcessor() creates the directories
    amrex::PreBuildDirectorHierarchy(checkpointname, "Level_", nlevels, true);

    // write Header file
   if (ParallelDescriptor::IOProcessor()) {

       std::string HeaderFileName(checkpointname + "/Header");
       VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);
       std::ofstream HeaderFile;
       HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
       HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out   |
                                               std::ofstream::trunc |
                                               std::ofstream::binary);
       if( ! HeaderFile.good()) {
           amrex::FileOpenFailed(HeaderFileName);
       }

       HeaderFile.precision(17);

       // write out title line
       HeaderFile << "Checkpoint file for AmrCoreAdv\n";

       // write out finest_level
       HeaderFile << finest_level << "\n";

       // write out array of istep
       for (auto s : istep) {
           HeaderFile << s << " ";
       }
       HeaderFile << "\n";

       // write out array of dt
       for (auto dti : dt) {
           HeaderFile << dti << " ";
       }
       HeaderFile << "\n";

       // write out array of t_new
       for (auto t_new_i : t_new) {
           HeaderFile << t_new_i << " ";
       }
       HeaderFile << "\n";

       // write the BoxArray at each level
       for (int lev = 0; lev <= finest_level; ++lev) {
           boxArray(lev).writeOn(HeaderFile);
           HeaderFile << '\n';
       }
   }

   // write the MultiFab data to, e.g., chk00010/Level_0/
   for (int lev = 0; lev <= finest_level; ++lev) {
       VisMF::Write(phi_new[lev],
                    amrex::MultiFabFileFullPrefix(lev, checkpointname, "Level_", "phi"));
   }

#ifdef AMREX_PARTICLES
            if (do_tracers) {
                TracerPC->Checkpoint(checkpointname, "particles", true);
            }
#endif

}

#ifdef AMREX_PARTICLES
void
AmrCoreAdv::init_particles ()
{
  if (do_tracers)
    {
      BL_ASSERT(TracerPC == nullptr);

      TracerPC = std::make_unique<AmrTracerParticleContainer>(this);

      AmrTracerParticleContainer::ParticleInitData pdata = {{AMREX_D_DECL(0.0, 0.0, 0.0)},{},{},{}};

      TracerPC->SetVerbose(0);
      TracerPC->InitOnePerCell(0.5, 0.5, 0.5, pdata);
      TracerPC->Redistribute();
    }
}
#endif


namespace {
// utility to skip to next line in Header
void GotoNextLine (std::istream& is)
{
    constexpr std::streamsize bl_ignore_max { 100000 };
    is.ignore(bl_ignore_max, '\n');
}
}

void
AmrCoreAdv::ReadCheckpointFile ()
{

    amrex::Print() << "Restart from checkpoint " << restart_chkfile << "\n";

    // Header
    std::string File(restart_chkfile + "/Header");

    VisMF::IO_Buffer io_buffer(VisMF::GetIOBufferSize());

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream is(fileCharPtrString, std::istringstream::in);

    std::string line, word;

    // read in title line
    std::getline(is, line);

    // read in finest_level
    is >> finest_level;
    GotoNextLine(is);

    // read in array of istep
    std::getline(is, line);
    {
        std::istringstream lis(line);
        int i = 0;
        while (lis >> word) {
            istep[i++] = std::stoi(word);
        }
    }

    // read in array of dt
    std::getline(is, line);
    {
        std::istringstream lis(line);
        int i = 0;
        while (lis >> word) {
            dt[i++] = std::stod(word);
        }
    }

    // read in array of t_new
    std::getline(is, line);
    {
        std::istringstream lis(line);
        int i = 0;
        while (lis >> word) {
            t_new[i++] = std::stod(word);
        }
    }

    for (int lev = 0; lev <= finest_level; ++lev) {

        // read in level 'lev' BoxArray from Header
        BoxArray ba;
        ba.readFrom(is);
        GotoNextLine(is);

        // create a distribution mapping
        DistributionMapping dm { ba, ParallelDescriptor::NProcs() };

        // set BoxArray grids and DistributionMapping dmap in AMReX_AmrMesh.H class
        SetBoxArray(lev, ba);
        SetDistributionMap(lev, dm);

        // build MultiFab and FluxRegister data
        int ncomp = 1;
        int ng = 0;
        phi_old[lev].define(grids[lev], dmap[lev], ncomp, ng);
        phi_new[lev].define(grids[lev], dmap[lev], ncomp, ng);

        if (lev > 0 && do_reflux) {
            flux_reg[lev] = std::make_unique<FluxRegister>(grids[lev], dmap[lev], refRatio(lev-1), lev, ncomp);
        }

        // build face velocity MultiFabs
        for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
        {
            facevel[lev][idim] = MultiFab(amrex::convert(ba,IntVect::TheDimensionVector(idim)), dm, 1, 1);
        }
    }

    // read in the MultiFab data
    for (int lev = 0; lev <= finest_level; ++lev) {
        VisMF::Read(phi_new[lev],
                    amrex::MultiFabFileFullPrefix(lev, restart_chkfile, "Level_", "phi"));
    }

#ifdef AMREX_PARTICLES
    if (do_tracers) {
        BL_ASSERT(TracerPC == nullptr);
        TracerPC = std::make_unique<AmrTracerParticleContainer>(this);
        TracerPC->Restart(this->restart_chkfile, "particles");
    }
#endif


}
