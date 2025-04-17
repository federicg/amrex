#ifndef AmrCoreAdv_H_
#define AmrCoreAdv_H_

#include <AMReX_Config.H>
#ifdef AMREX_PARTICLES
#   include <AMReX_AmrParticles.H>
#   include <AMReX_Particles.H>
#endif

#include <AMReX_AmrCore.H>
#include <AMReX_BCRec.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_FillPatcher.H>

#ifdef AMREX_USE_OMP
#   include <omp.h>
#endif

#include <limits>
#include <memory>
#include <string>
#include <optional>

class AmrCoreAdv
    : public amrex::AmrCore
{
public:

    ////////////////
    // public member functions

    // constructor - reads in parameters from inputs file
    //             - sizes multilevel arrays and data structures
    AmrCoreAdv ();
    AmrCoreAdv (amrex::Geometry const& level_0_geom, int index_core, amrex::AmrInfo const& amr_info);
    ~AmrCoreAdv () override;

    AmrCoreAdv (AmrCoreAdv&& rhs) noexcept = default;
    AmrCoreAdv& operator= (AmrCoreAdv&& rhs) noexcept = default;

    AmrCoreAdv (const AmrCoreAdv& rhs) = delete;
    AmrCoreAdv& operator= (const AmrCoreAdv& rhs) = delete;

    // advance solution to final time
    void Evolve ();

    // initializes multilevel data
    void InitData ();

    // Make a new level using provided BoxArray and DistributionMapping and
    // fill with interpolated coarse level data.
    // overrides the pure virtual function in AmrCore
    void MakeNewLevelFromCoarse (int lev, amrex::Real time, const amrex::BoxArray& ba,
                                 const amrex::DistributionMapping& dm) override;

    // Remake an existing level using provided BoxArray and DistributionMapping and
    // fill with existing fine and coarse data.
    // overrides the pure virtual function in AmrCore
    void RemakeLevel (int lev, amrex::Real time, const amrex::BoxArray& ba,
                      const amrex::DistributionMapping& dm) override;

    // Delete level data
    // overrides the pure virtual function in AmrCore
    void ClearLevel (int lev) override;

    // Make a new level from scratch using provided BoxArray and DistributionMapping.
    // Only used during initialization.
    // overrides the pure virtual function in AmrCore
    void MakeNewLevelFromScratch (int lev, amrex::Real time, const amrex::BoxArray& ba,
                                  const amrex::DistributionMapping& dm) override;

    // tag all cells for refinement
    // overrides the pure virtual function in AmrCore
    void ErrorEst (int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override;

    // Advance phi at a single level for a single time step, update flux registers
    void AdvancePhiAtLevel (int lev, amrex::Real time, amrex::Real dt_lev, int iteration, int ncycle);

    // Advance phi at all levels for a single time step
    void AdvancePhiAllLevels (amrex::Real time, amrex::Real dt_lev, int iteration);

    // Define the advection velocity as the curl of a scalar field
    void DefineVelocityAtLevel (int lev, amrex::Real time);

    void DefineVelocityAllLevels (amrex::Real time);

    // compute dt from CFL considerations
    amrex::Real EstTimeStep (int lev, amrex::Real time);

#ifdef AMREX_PARTICLES
    amrex::AmrTracerParticleContainer* theTracerPC () { return TracerPC.get(); }
#endif

    amrex::Vector<int> istep;      // which step?

    // do we subcycle in time?
    int do_subcycle = 1;

    // write plotfile to disk
    void WritePlotFile () const;

    // write checkpoint file to disk
    void WriteCheckpointFile () const;

    // a wrapper for EstTimeStep
    void ComputeDt ();

    // maximum number of steps and stop time
    amrex::Real stop_time = std::numeric_limits<amrex::Real>::max();
    int max_step = std::numeric_limits<int>::max();

    // original substepping
    void timeStepWithSubcycling_original (int lev, amrex::Real time, int iteration);

    // Advance a level by dt - includes a recursive call for finer levels
    void timeStepWithSubcycling (int lev, int iteration);

    // Advance all levels by the same dt
    void timeStepNoSubcycling (amrex::Real time, int iteration);

    // Regrid routine
    void perform_regrid(amrex::Real);

    // Regrid routine for subcycling
    void perform_regridWithSubcycling(int,bool);

    // Reflux here across multiple levels
    void perform_reflux_across_lev(int);

    void check_finer();
    void check_finer(int);

    std::array<amrex::Vector<amrex::MultiFab>, 4> array_vec_mf_g;

    std::array<const AmrCoreAdv*, 4> other_core;  // Pointer to another AmrCoreAdv

    void setOtherCore(const AmrCoreAdv*, const AmrCoreAdv*, const AmrCoreAdv*, const AmrCoreAdv*);

    void create_ghost_multifabs(int);
    void create_ghost_multifabs(int, int);

    struct OnesidedMultiBlockBoundaryFn {
      amrex::NonLocalBC::MultiBlockIndexMapping dtos;
      amrex::Box boundary_to_fill;
      int i_boundary;
      std::unique_ptr<amrex::NonLocalBC::MultiBlockCommMetaData> cmd{};
      amrex::FabArrayBase::BDKey cached_dest_bd_key{};
      amrex::FabArrayBase::BDKey cached_src_bd_key{};
      amrex::NonLocalBC::ApplyDtosAndProjectionOnReciever<amrex::NonLocalBC::MultiBlockIndexMapping,
                                                          amrex::NonLocalBC::MapComponents<amrex::NonLocalBC::Identity, amrex::NonLocalBC::SwapComponents<1, 2>>>
          packing{amrex::NonLocalBC::PackComponents{0, 0, NUM_COMPONENTS}, dtos};

      AMREX_NODISCARD amrex::NonLocalBC::CommHandler FillBoundary_nowait(amrex::MultiFab&, std::array<amrex::Vector<amrex::MultiFab>, 4>&, int);
      AMREX_NODISCARD amrex::NonLocalBC::CommHandler FillBoundary_nowait(amrex::MultiFab&, const std::array<const AmrCoreAdv*, 4>&, int);

      void FillBoundary_do_local_copy(amrex::MultiFab&, std::array<amrex::Vector<amrex::MultiFab>, 4>&, int) const;
      void FillBoundary_do_local_copy(amrex::MultiFab&, const std::array<const AmrCoreAdv*, 4>&, int) const;
      void FillBoundary_finish(amrex::MultiFab&, amrex::NonLocalBC::CommHandler) const;
    };

    struct FillBoundaryFn {
      std::vector<OnesidedMultiBlockBoundaryFn> boundaries;
      void operator()(amrex::MultiFab&, std::array<amrex::Vector<amrex::MultiFab>, 4>&, int);
      void operator()(amrex::MultiFab&, const std::array<const AmrCoreAdv*, 4>&, int);
    };

    amrex::Vector<std::vector<OnesidedMultiBlockBoundaryFn>> multi_block_boundaries;
    amrex::Vector<FillBoundaryFn> FillBoundary_ghost;

    amrex::Vector<std::vector<OnesidedMultiBlockBoundaryFn>> multi_block_boundariesMarkers;
    amrex::Vector<FillBoundaryFn> FillBoundaryMarkers_ghost;

    void MoveMultiBlocks();

    void check_finer_communication();
    void check_finer_communication(int);
    void reset_level_tagger();

    // if >= 0 we restart from a checkpoint
    std::string restart_chkfile;

    void particles_tracer(int, int);

    // Getter functions 
    const int& getFinestLevel() const;
    const int& getNsubsteps(int) const;
    const amrex::Real& getLevel0Dt() const;
    const amrex::Real& getLevelLevDt(int) const;
    const amrex::Real& getTnewLev(int) const;
    const int& getChk_int() const;
    const int& getPlot_int() const;

    // Setter functions
    void setDtWithSubcycling();
    void setDtNoSubcycling();
    void setLevel0Dt(amrex::Real);
    void setTnewAllLev(amrex::Real);

    //
    amrex::Real computeSumLevel0();

    int index_core;

private:

    // plotfile prefix and frequency
    std::string plot_file {"plt"};
    int plot_int = -1;

    // checkpoint prefix and frequency
    std::string chk_file {"chk"};
    int chk_int = -1;

    bool is_first = true;

    amrex::Vector<amrex::MultiFab> level_tagger;

    static constexpr int NUM_COMPONENTS = 1; // number of conservative variables 

    //int index_core;

    amrex::Vector<int> last_regrid_step;

    ////////////////
    // private member functions

    // read in some parameters from inputs file
    void ReadParameters(); 

    // set covered coarse cells to be the average of overlying fine cells
    void AverageDown ();

    // more flexible version of AverageDown() that lets you average down across multiple levels
    void AverageDownTo (int crse_lev);

    enum class FillPatchType { fillpatch_class, fillpatch_function };

    // compute a new multifab by coping in phi from valid region and filling ghost cells
    // works for single level and 2-level cases (fill fine grid ghost by interpolating from coarse)
    void FillPatch (int lev, amrex::Real time, amrex::MultiFab& mf, int icomp,
                    int ncomp, FillPatchType fptype);

    // fill an entire multifab by interpolating from the coarser level
    // this comes into play when a new level of refinement appears
    void FillCoarsePatch (int lev, amrex::Real time, amrex::MultiFab& mf, int icomp, int ncomp);

    // Pack pointers to phi_old and/or phi_new and associated times.
    void GetData (int lev, amrex::Real time, amrex::Vector<amrex::MultiFab*>& data,
                  amrex::Vector<amrex::Real>& datatime);

    // get plotfile name
    [[nodiscard]] std::string PlotFileName (int lev) const;

    // put together an array of multifabs for writing
    [[nodiscard]] amrex::Vector<const amrex::MultiFab*> PlotFileMF () const;

    // set plotfile variables names
    [[nodiscard]] static amrex::Vector<std::string> PlotFileVarNames ();

    // read checkpoint file from disk
    void ReadCheckpointFile ();

    ////////////////
    // private data members

    //amrex::Vector<int> istep;      // which step?
    amrex::Vector<int> nsubsteps;  // how many substeps on each level?

    // keep track of old time, new time, and time step at each level
    amrex::Vector<amrex::Real> t_new;
    amrex::Vector<amrex::Real> t_old;
    amrex::Vector<amrex::Real> dt;

    // array of multifabs to store the solution at each level of refinement
    // after advancing a level we use "swap".
    amrex::Vector<amrex::MultiFab> phi_new;
    amrex::Vector<amrex::MultiFab> phi_old;

    // this is essentially a 2*DIM integer array storing the physical boundary
    // condition types at the lo/hi walls in each direction
    amrex::Vector<amrex::BCRec> bcs;  // 1-component

    // stores fluxes at coarse-fine interface for synchronization
    // this will be sized "nlevs_max+1"
    // NOTE: the flux register associated with flux_reg[lev] is associated
    // with the lev/lev-1 interface (and has grid spacing associated with lev-1)
    // therefore flux_reg[0] and flux_reg[nlevs_max] are never actually
    // used in the reflux operation
    amrex::Vector<std::unique_ptr<amrex::FluxRegister> > flux_reg;

    // This is for fillpatch during timestepping, but not for regridding.
    amrex::Vector<std::unique_ptr<amrex::FillPatcher<amrex::MultiFab>>> fillpatcher;

    // Velocity on all faces at all levels
    amrex::Vector< amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> > facevel;

    ////////////////
    // runtime parameters

    // advective cfl number - dt = cfl*dx/umax
    amrex::Real cfl = 0.7;

    // how often each level regrids the higher levels of refinement
    // (after a level advances that many time steps)
    int regrid_int = 2;

    // hyperbolic refluxing as part of multilevel synchronization
    int do_reflux = 1;

    // folder and core prefix
    std::string core_prefix {"sol/core"};

    //number of ghost cells on facevel
    int nghost = 2;

#ifdef AMREX_PARTICLES
    void init_particles ();
    int       do_tracers = 0;
    std::unique_ptr<amrex::AmrTracerParticleContainer> TracerPC;
#endif
};

#endif
