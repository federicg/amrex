
#include "AMReX_NonLocalBC.H"

#include "AMReX.H"
#include "AMReX_AmrCore.H"
#include "AMReX_MultiFab.H"

#include "AMReX_PlotFileUtil.H"

void MyMain();

int main(int argc, char** argv) {
#ifdef AMREX_USE_MPI
    MPI_Init(&argc, &argv);
#else
    amrex::ignore_unused(argc,argv);
#endif
    // Let me throw exceptions for triggering my debugger
    amrex::Initialize(MPI_COMM_WORLD, std::cout, std::cerr, [](const char* msg) { throw std::runtime_error(msg); });
    MyMain();
    amrex::Finalize();
#ifdef AMREX_USE_MPI
    MPI_Finalize();
#endif
}

using namespace amrex;

enum idirs { ix, iy };
enum num_components { three_components = 3 };

static constexpr IntVect e_x = IntVect::TheDimensionVector(ix);
static constexpr IntVect e_y = IntVect::TheDimensionVector(iy);

// define here the cube face length
static constexpr Real cube_face_length = 1.;
class AdvectionAmrCore : public AmrCore {

  public:
    AdvectionAmrCore(Array<Real, AMREX_SPACEDIM> vel, Geometry const& level_0_geom,
                     AmrInfo const& amr_info = AmrInfo())
        : AmrCore(level_0_geom, amr_info), velocity{vel} {
        AmrCore::InitFromScratch(0.0);
        InitData();  // CUDA does not allow extended lambdas in ctors.
    }

    void InitData() {
        const auto problo = Geom(0).ProbLoArray(); // get the physical coorinates of the physical domain
        const auto dx = Geom(0).CellSizeArray();
        const auto dy = Geom(0).CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(mass,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            Array4<Real> m  = mass.array(mfi   );
            Array4<Real> vx = mass.array(mfi, 1);
            Array4<Real> vy = mass.array(mfi, 2);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                Real shift[] = {0*cube_face_length, 1.2*cube_face_length};
                Real x[] = {problo[0] + (0.5+i)*dx[0],
                            problo[1] + (0.5+j)*dy[1]};
                const double r2 = std::sqrt((x[0]-shift[0]) * (x[0]-shift[0]) + (x[1]-shift[1]) * (x[1]-shift[1]));
                constexpr double R = 0.1;
                m(i, j, k) = r2 < R ? 1.0 : 0.0;
                vx(i, j, k) = r2 < R ? 1.0 : 0.0;
                vy(i, j, k) = r2 < R ? 0.0 : 0.0;
            });
        }
    }

    void AdvanceInTime(double dt) {
        // Do first order accurate godunov splitting
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            DoOperatorSplitStep(dt, static_cast<Direction>(d));
        }
    }

    void DoOperatorSplitStep(double dt, Direction dir) {
        // Perform first order accurate upwinding with velocity 1 in the stored direction.
        const double dx = Geom(0).CellSize(0);
        const double dy = Geom(0).CellSize(1);
        int dir_index = static_cast<int>(dir);
        const double a_dt_over_dx = dt / dx * velocity[dir_index];
        const double a_dt_over_dy = dt / dy * velocity[dir_index];
        if (dir == Direction::x) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(mass); mfi.isValid(); ++mfi) {
                Array4<Real> m = mass.array(mfi);
                Array4<Real> next = mass_next.array(mfi);
                ParallelFor(mfi.growntilebox(e_y), int(three_components),
                            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                                next(i, j, k, n) = m(i, j, k, n) - (a_dt_over_dx>0. ? a_dt_over_dx * (m(i, j, k, n) - m(i - 1, j, k, n)) : a_dt_over_dx * (m(i + 1, j, k, n) - m(i, j, k, n)) );
                            });
            }
            std::swap(mass, mass_next);
        }
        else if (dir == Direction::y) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(mass); mfi.isValid(); ++mfi) {
                Array4<Real> m = mass.array(mfi);
                Array4<Real> next = mass_next.array(mfi);
                ParallelFor(mfi.growntilebox(ix), int(three_components),
                            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                                next(i, j, k, n) = m(i, j, k, n) - (a_dt_over_dy>0. ? a_dt_over_dy * (m(i, j, k, n) - m(i, j - 1, k, n)) : a_dt_over_dy * (m(i, j + 1, k, n) - m(i, j, k, n)));
                            });
            }
            std::swap(mass, mass_next);
        }
    }

    MultiFab mass{};
    MultiFab mass_next{};
    Array<Real, AMREX_SPACEDIM> velocity;

  private:
    void ErrorEst(int /*level*/, ::amrex::TagBoxArray& /*tags*/, Real /*time_point*/,
                  int /* ngrow */) override {
        throw std::runtime_error("For simplicity, this example supports only one level.");
    }

    void
    MakeNewLevelFromScratch(int level, Real, const ::amrex::BoxArray& box_array,
                            const ::amrex::DistributionMapping& distribution_mapping) override {
        if (level > 0) {
            throw std::runtime_error("For simplicity, this example supports only one level.");
        }
        const IntVect ngrow{AMREX_D_DECL(1, 1, 0)};
        mass.define(box_array, distribution_mapping, three_components, ngrow);
        mass_next.define(box_array, distribution_mapping, three_components, ngrow);
    }

    void MakeNewLevelFromCoarse(int /*level*/, Real /*time_point*/, const ::amrex::BoxArray&,
                                const ::amrex::DistributionMapping&) override {
        throw std::runtime_error("For simplicity, this example supports only one level.");
    }

    void RemakeLevel(int /*level*/, Real /*time_point*/, const ::amrex::BoxArray&,
                     const ::amrex::DistributionMapping&) override {
        throw std::runtime_error("For simplicity, this example supports only one level.");
    }

    void ClearLevel(int level) override {
        if (level > 0) {
            throw std::runtime_error("For simplicity, this example supports only one level.");
        }
        mass.clear();
    }
};

using namespace NonLocalBC;
struct OnesidedMultiBlockBoundaryFn {
  AdvectionAmrCore* dest;
  const AdvectionAmrCore* src;
  MultiBlockIndexMapping dtos;
  Box boundary_to_fill;
  std::unique_ptr<MultiBlockCommMetaData> cmd{};
  FabArrayBase::BDKey cached_dest_bd_key{};
  FabArrayBase::BDKey cached_src_bd_key{};
  ApplyDtosAndProjectionOnReciever<MultiBlockIndexMapping,
                                   MapComponents<Identity, SwapComponents<1, 2>>>
      packing{PackComponents{0, 0, three_components}, dtos};

  AMREX_NODISCARD CommHandler FillBoundary_nowait() {
    if (!cmd || cached_dest_bd_key != dest->mass.getBDKey() || cached_src_bd_key != src->mass.getBDKey()) {
        cmd = std::make_unique<MultiBlockCommMetaData>(dest->mass, boundary_to_fill, src->mass, dest->mass.nGrowVect(), dtos);
        cached_dest_bd_key = dest->mass.getBDKey();
        cached_src_bd_key = src->mass.getBDKey();
    }

    return ParallelCopy_nowait(no_local_copy, dest->mass, src->mass, *cmd, packing);
  }

  void FillBoundary_do_local_copy() const {
    AMREX_ASSERT(cmd && cached_dest_bd_key == dest->mass.getBDKey() && cached_src_bd_key == src->mass.getBDKey());
    if (cmd->m_LocTags && !cmd->m_LocTags->empty()) {
        LocalCopy(packing, dest->mass, src->mass, *cmd->m_LocTags);
    }
  }

  void FillBoundary_finish(CommHandler handler) const {
    ParallelCopy_finish(dest->mass, std::move(handler), *cmd, packing); // NOLINT(performance-move-const-arg)
  }
};

struct FillBoundaryFn {
    std::vector<OnesidedMultiBlockBoundaryFn> boundaries;

    void operator()(AdvectionAmrCore& core_1, AdvectionAmrCore& core_2, AdvectionAmrCore& core_3, AdvectionAmrCore& core_4, AdvectionAmrCore& core_5, AdvectionAmrCore& core_6) {
        enum { coarsest_level = 0 };
        core_1.mass.FillBoundary(core_1.Geom(coarsest_level).periodicity());
        core_2.mass.FillBoundary(core_2.Geom(coarsest_level).periodicity());
        core_3.mass.FillBoundary(core_3.Geom(coarsest_level).periodicity());
        core_4.mass.FillBoundary(core_4.Geom(coarsest_level).periodicity());
        core_5.mass.FillBoundary(core_5.Geom(coarsest_level).periodicity());
        core_6.mass.FillBoundary(core_6.Geom(coarsest_level).periodicity());
        std::vector<CommHandler> comms;
        comms.reserve(boundaries.size());
        for (auto& boundary : boundaries) {
            comms.emplace_back(boundary.FillBoundary_nowait());
        }
        for (auto& boundary : boundaries) {
            boundary.FillBoundary_do_local_copy();
        }
        const std::size_t n_boundaries = boundaries.size();
        for (std::size_t i = 0; i < n_boundaries; ++i) {
            boundaries[i].FillBoundary_finish(std::move(comms[i])); // NOLINT(performance-move-const-arg)
        }
        AMREX_ASSERT(!core_1.mass.contains_nan());
    }
};

void WritePlotfiles(const AdvectionAmrCore& core_1, const AdvectionAmrCore& core_2, const AdvectionAmrCore& core_3, const AdvectionAmrCore& core_4, const AdvectionAmrCore& core_5, const AdvectionAmrCore& core_6, Real time_point, int step)
{
    static const Vector<std::string> varnames{"Mass", "Vector_X", "Vector_Y"};
    int nlevels = 1;
    std::array<char, 256> c1_pbuffer{};
    std::array<char, 256> c2_pbuffer{};
    std::array<char, 256> c3_pbuffer{};
    std::array<char, 256> c4_pbuffer{};
    std::array<char, 256> c5_pbuffer{};
    std::array<char, 256> c6_pbuffer{};
    snprintf(c1_pbuffer.data(), c1_pbuffer.size(), "MultiBlock/core_1/plt%04d", step);
    snprintf(c2_pbuffer.data(), c2_pbuffer.size(), "MultiBlock/core_2/plt%04d", step);
    snprintf(c3_pbuffer.data(), c3_pbuffer.size(), "MultiBlock/core_3/plt%04d", step);
    snprintf(c4_pbuffer.data(), c4_pbuffer.size(), "MultiBlock/core_4/plt%04d", step);
    snprintf(c5_pbuffer.data(), c5_pbuffer.size(), "MultiBlock/core_5/plt%04d", step);
    snprintf(c6_pbuffer.data(), c6_pbuffer.size(), "MultiBlock/core_6/plt%04d", step);

    {
        Vector<const MultiFab*> mf{&core_1.mass};
        Vector<Geometry> geoms{core_1.Geom(0)};
        Vector<int> level_steps{step};
        Vector<IntVect> ref_ratio{};
        std::string plotfilename{c1_pbuffer.data()};
        WriteMultiLevelPlotfile(plotfilename, nlevels, mf, varnames, geoms, time_point, level_steps, ref_ratio);
    }
    {
        Vector<const MultiFab*> mf{&core_2.mass};
        Vector<Geometry> geoms{core_2.Geom(0)};
        Vector<int> level_steps{step};
        Vector<IntVect> ref_ratio{};
        std::string plotfilename{c2_pbuffer.data()};
        WriteMultiLevelPlotfile(plotfilename, nlevels, mf, varnames, geoms, time_point, level_steps, ref_ratio);
    }
    {
        Vector<const MultiFab*> mf{&core_3.mass};
        Vector<Geometry> geoms{core_3.Geom(0)};
        Vector<int> level_steps{step};
        Vector<IntVect> ref_ratio{};
        std::string plotfilename{c3_pbuffer.data()};
        WriteMultiLevelPlotfile(plotfilename, nlevels, mf, varnames, geoms, time_point, level_steps, ref_ratio);
    }
        {
        Vector<const MultiFab*> mf{&core_4.mass};
        Vector<Geometry> geoms{core_4.Geom(0)};
        Vector<int> level_steps{step};
        Vector<IntVect> ref_ratio{};
        std::string plotfilename{c4_pbuffer.data()};
        WriteMultiLevelPlotfile(plotfilename, nlevels, mf, varnames, geoms, time_point, level_steps, ref_ratio);
    }
    {
        Vector<const MultiFab*> mf{&core_5.mass};
        Vector<Geometry> geoms{core_5.Geom(0)};
        Vector<int> level_steps{step};
        Vector<IntVect> ref_ratio{};
        std::string plotfilename{c5_pbuffer.data()};
        WriteMultiLevelPlotfile(plotfilename, nlevels, mf, varnames, geoms, time_point, level_steps, ref_ratio);
    }
    {
        Vector<const MultiFab*> mf{&core_6.mass};
        Vector<Geometry> geoms{core_6.Geom(0)};
        Vector<int> level_steps{step};
        Vector<IntVect> ref_ratio{};
        std::string plotfilename{c6_pbuffer.data()};
        WriteMultiLevelPlotfile(plotfilename, nlevels, mf, varnames, geoms, time_point, level_steps, ref_ratio);
    }
}

void MyMain() {
    Box domain(IntVect{}, IntVect{AMREX_D_DECL(63, 63, 0)});
    RealBox real_box1{{AMREX_D_DECL(-cube_face_length*.5,                  -cube_face_length*.5,                  0.0)}, {AMREX_D_DECL(+cube_face_length*.5,                     +cube_face_length*.5,                  1.0)}};
    RealBox real_box2{{AMREX_D_DECL(+cube_face_length*.5,                  -cube_face_length*.5,                  0.0)}, {AMREX_D_DECL(+cube_face_length*.5+cube_face_length,    +cube_face_length*.5,                  1.0)}};
    RealBox real_box3{{AMREX_D_DECL(+cube_face_length*.5+cube_face_length, -cube_face_length*.5,                  0.0)}, {AMREX_D_DECL(+cube_face_length*.5+cube_face_length*2., +cube_face_length*.5,                  1.0)}};
    RealBox real_box4{{AMREX_D_DECL(-cube_face_length*.5-cube_face_length, -cube_face_length*.5,                  0.0)}, {AMREX_D_DECL(-cube_face_length*.5,                     +cube_face_length*.5,                  1.0)}};
    RealBox real_box5{{AMREX_D_DECL(-cube_face_length*.5,                  -cube_face_length*.5-cube_face_length, 0.0)}, {AMREX_D_DECL(+cube_face_length*.5,                     -cube_face_length*.5,                  1.0)}};
    RealBox real_box6{{AMREX_D_DECL(-cube_face_length*.5,                  +cube_face_length*.5,                  0.0)}, {AMREX_D_DECL(+cube_face_length*.5,                     +cube_face_length*.5+cube_face_length, 1.0)}};

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

    Array<int, AMREX_SPACEDIM> is_periodic6{AMREX_D_DECL(0, 0, 0)};
    Geometry geom6{domain, real_box6, CoordSys::cartesian, is_periodic6};

    AmrInfo amr_info{};
#if AMREX_SPACEDIM > 2 // Only for 3D
    amr_info.blocking_factor[0][2] = 1;
#endif

    Array<Real, AMREX_SPACEDIM> velocity_1{}, velocity_2{}, velocity_3{}, velocity_4{}, velocity_5{}, velocity_6{};
    velocity_1[0] = 0.0;
    velocity_1[1] = 1.0; 
#if AMREX_SPACEDIM > 2 // Only for 3D
    velocity_1[2] = 0.0;  
#endif

    velocity_2[0] = 0.0;
    velocity_2[1] = -1.0; 
#if AMREX_SPACEDIM > 2 // Only for 3D
    velocity_2[2] = 0.0;  
#endif

    velocity_3[0] = 0.0;
    velocity_3[1] = -1.0; 
#if AMREX_SPACEDIM > 2 // Only for 3D
    velocity_3[2] = 0.0;  
#endif

    velocity_4[0] = 0.0;
    velocity_4[1] = -1.0; 
#if AMREX_SPACEDIM > 2 // Only for 3D
    velocity_4[2] = 0.0;  
#endif

    velocity_5[0] = 1.0;
    velocity_5[1] = 0.0; 
#if AMREX_SPACEDIM > 2 // Only for 3D
    velocity_5[2] = 0.0;  
#endif

    velocity_6[0] = -1.0;
    velocity_6[1] = 0.0; 
#if AMREX_SPACEDIM > 2 // Only for 3D
    velocity_6[2] = 0.0;  
#endif

    AdvectionAmrCore core_1(velocity_1, geom1, amr_info);
    AdvectionAmrCore core_2(velocity_2, geom2, amr_info);
    AdvectionAmrCore core_3(velocity_3, geom3, amr_info);
    AdvectionAmrCore core_4(velocity_4, geom4, amr_info);
    AdvectionAmrCore core_5(velocity_5, geom5, amr_info);
    AdvectionAmrCore core_6(velocity_6, geom6, amr_info);

    
    std::vector<OnesidedMultiBlockBoundaryFn> multi_block_boundaries{};
    // ------------------------------------------------------------------------- // 1
    {   // Fill right boundary of core_1 with left mirror data of core_2
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = (domain.bigEnd(ix) + 1) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box right_boundary_to_fill_in_x = grow(shift(Box{domain.bigEnd(ix) * e_x, domain.bigEnd()}, e_x), e_y);
        multi_block_boundaries.push_back({&core_1, &core_2, dtos, right_boundary_to_fill_in_x});
    } { // Fill left boundary of core_2 with right mirror data of core_1
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = - (domain.bigEnd(ix) + 1) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box left_boundary_to_fill_in_x = grow(shift(Box{domain.smallEnd(), domain.bigEnd() - domain.bigEnd(ix) * e_x}, -e_x), e_y); 
        multi_block_boundaries.push_back({&core_2, &core_1, dtos, left_boundary_to_fill_in_x});
    } 
    // ------------------------------------------------------------------------- // 2
    {   // Fill right boundary of core_2 with left mirror data of core_3
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = (domain.bigEnd(ix) + 1) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box right_boundary_to_fill_in_x = grow(shift(Box{domain.bigEnd(ix) * e_x, domain.bigEnd()}, e_x), e_y);
        multi_block_boundaries.push_back({&core_2, &core_3, dtos, right_boundary_to_fill_in_x});
    } { // Fill left boundary of core_3 with right mirror data of core_2
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = - (domain.bigEnd(ix) + 1) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box left_boundary_to_fill_in_x = grow(shift(Box{domain.smallEnd(), domain.bigEnd() - domain.bigEnd(ix) * e_x}, -e_x), e_y); 
        multi_block_boundaries.push_back({&core_3, &core_2, dtos, left_boundary_to_fill_in_x});
    } 
    // ------------------------------------------------------------------------- // 3
    {   // Fill right boundary of core_3 with left mirror data of core_4
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = (domain.bigEnd(ix) + 1) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box right_boundary_to_fill_in_x = grow(shift(Box{domain.bigEnd(ix) * e_x, domain.bigEnd()}, e_x), e_y);
        multi_block_boundaries.push_back({&core_3, &core_4, dtos, right_boundary_to_fill_in_x});
    } { // Fill left boundary of core_4 with right mirror data of core_3
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = - (domain.bigEnd(ix) + 1) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box left_boundary_to_fill_in_x = grow(shift(Box{domain.smallEnd(), domain.bigEnd() - domain.bigEnd(ix) * e_x}, -e_x), e_y); 
        multi_block_boundaries.push_back({&core_4, &core_3, dtos, left_boundary_to_fill_in_x});
    } 
    // ------------------------------------------------------------------------- // 4
    {   // Fill left boundary of core_1 with right mirror data of core_4
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = - (domain.bigEnd(ix) + 1) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box left_boundary_to_fill_in_x = grow(shift(Box{domain.smallEnd(), domain.bigEnd() - domain.bigEnd(ix) * e_x}, -e_x), e_y); 
        multi_block_boundaries.push_back({&core_1, &core_4, dtos, left_boundary_to_fill_in_x});
    } { // Fill right boundary of core_4 with left mirror data of core_1
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = (domain.bigEnd(ix) + 1) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box right_boundary_to_fill_in_x = grow(shift(Box{domain.bigEnd(ix) * e_x, domain.bigEnd()}, e_x), e_y); 
        multi_block_boundaries.push_back({&core_4, &core_1, dtos, right_boundary_to_fill_in_x});
    } 
    // ------------------------------------------------------------------------- // 5
    {   // Fill lower boundary of core_1 with upper mirror data of core_5
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = - (domain.bigEnd(iy) + 1) * e_y;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box lower_boundary_to_fill_in_y = grow(shift(Box{domain.smallEnd(), domain.bigEnd() - domain.bigEnd(iy) * e_y}, -e_y), e_x);
        multi_block_boundaries.push_back({&core_1, &core_5, dtos, lower_boundary_to_fill_in_y});
    } { // Fill upper boundary of core_5 with lower mirror data of core_1
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = (domain.bigEnd(iy) + 1) * e_y;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box upper_boundary_to_fill_in_y = grow(shift(Box{domain.bigEnd(iy) * e_y, domain.bigEnd()}, e_y), e_x);
        multi_block_boundaries.push_back({&core_5, &core_1, dtos, upper_boundary_to_fill_in_y});
    } 
    // ------------------------------------------------------------------------- // 6
    {   // Fill lower boundary of core_6 with upper mirror data of core_1
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = - (domain.bigEnd(iy) + 1) * e_y;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box lower_boundary_to_fill_in_y = grow(shift(Box{domain.smallEnd(), domain.bigEnd() - domain.bigEnd(iy) * e_y}, -e_y), e_x);
        multi_block_boundaries.push_back({&core_6, &core_1, dtos, lower_boundary_to_fill_in_y});
    } { // Fill upper boundary of core_1 with lower mirror data of core_6
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(0, 1, 2)};
        dtos.offset = (domain.bigEnd(iy) + 1) * e_y;
        dtos.sign = IntVect{AMREX_D_DECL(1, 1, 1)};
        Box upper_boundary_to_fill_in_y = grow(shift(Box{domain.bigEnd(iy) * e_y, domain.bigEnd()}, e_y), e_x);
        multi_block_boundaries.push_back({&core_1, &core_6, dtos, upper_boundary_to_fill_in_y});
    } 
    // ------------------------------------------------------------------------- // 7 
    {   // Fill upper boundary of core_6 with upper mirror data of core_3
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.sign = IntVect{AMREX_D_DECL(-1, 1, 1)};
        dtos.offset = 1 * e_y + domain.bigEnd(ix) * e_x;
        Box upper_boundary_to_fill_in_y = grow(shift(Box{domain.bigEnd(iy) * e_y, domain.bigEnd()}, e_y), e_x);
        multi_block_boundaries.push_back({&core_6, &core_3, dtos, upper_boundary_to_fill_in_y});
    } { // Fill upper boundary of core_3 with upper mirror data of core_6
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.sign = IntVect{AMREX_D_DECL(-1, 1, 1)};
        dtos.offset = 1 * e_y + domain.bigEnd(ix) * e_x;
        Box upper_boundary_to_fill_in_y = grow(shift(Box{domain.bigEnd(iy) * e_y, domain.bigEnd()}, e_y), e_x);
        multi_block_boundaries.push_back({&core_3, &core_6, dtos, upper_boundary_to_fill_in_y});
    } 
    // ------------------------------------------------------------------------- // 8 
    {   // Fill lower boundary of core_5 with lower mirror data of core_3
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.sign = IntVect{AMREX_D_DECL(-1, 1, 1)};
        dtos.offset = -1 * e_y + domain.bigEnd(ix) * e_x;
        Box lower_boundary_to_fill_in_y = grow(shift(Box{domain.smallEnd(), domain.bigEnd() - domain.bigEnd(iy) * e_y}, -e_y), e_x);
        multi_block_boundaries.push_back({&core_5, &core_3, dtos, lower_boundary_to_fill_in_y});
    } { // Fill lower boundary of core_3 with lower mirror data of core_5
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.sign = IntVect{AMREX_D_DECL(-1, 1, 1)};
        dtos.offset = -1 * e_y + domain.bigEnd(ix) * e_x;
        Box lower_boundary_to_fill_in_y = grow(shift(Box{domain.smallEnd(), domain.bigEnd() - domain.bigEnd(iy) * e_y}, -e_y), e_x);
        multi_block_boundaries.push_back({&core_3, &core_5, dtos, lower_boundary_to_fill_in_y});
    } 
    // ------------------------------------------------------------------------- // 9 
    {   // Fill left boundary of core_5 with lower mirror data of core_4
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(1, 0, 2)};
        dtos.offset = -1 * e_y;
        Box left_boundary_to_fill_in_x = grow(shift(Box{domain.smallEnd(), domain.bigEnd() - domain.bigEnd(ix) * e_x}, -e_x), e_y);
        multi_block_boundaries.push_back({&core_5, &core_4, dtos, left_boundary_to_fill_in_x});
    } { // Fill lower boundary of core_4 with left mirror data of core_5
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(1, 0, 2)};
        dtos.offset = -1 * e_x;
        Box lower_boundary_to_fill_in_y = grow(shift(Box{domain.smallEnd(), domain.bigEnd() - domain.bigEnd(iy) * e_y}, -e_y), e_x);
        multi_block_boundaries.push_back({&core_4, &core_5, dtos, lower_boundary_to_fill_in_y});
    } 
    // ------------------------------------------------------------------------- // 10 
    {   // Fill right boundary of core_5 with lower mirror data of core_2
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(1, 0, 2)};
        dtos.offset = (domain.bigEnd(iy) + 1) * e_y + domain.bigEnd(ix) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(-1, 1, 1)};
        Box right_boundary_to_fill_in_x = grow(shift(Box{domain.bigEnd(ix) * e_x, domain.bigEnd()}, e_x), e_y);
        multi_block_boundaries.push_back({&core_5, &core_2, dtos, right_boundary_to_fill_in_x});
    } { // Fill lower boundary of core_2 with right mirror data of core_5
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(1, 0, 2)};
        dtos.offset = domain.bigEnd(iy) * e_y - (domain.bigEnd(ix) + 1) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(1, -1, 1)};
        Box lower_boundary_to_fill_in_y = grow(shift(Box{domain.smallEnd(), domain.bigEnd() - domain.bigEnd(iy) * e_y}, -e_y), e_x);
        multi_block_boundaries.push_back({&core_2, &core_5, dtos, lower_boundary_to_fill_in_y});
    } 
    // ------------------------------------------------------------------------- // 11 
    {   // Fill right boundary of core_6 with upper mirror data of core_2
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(1, 0, 2)};  // Swap x and y if needed
        dtos.offset = 1 * e_y;  
        Box right_boundary_to_fill_in_x = grow(shift(Box{domain.bigEnd(ix) * e_x, domain.bigEnd()}, e_x), e_y);
        multi_block_boundaries.push_back({&core_6, &core_2, dtos, right_boundary_to_fill_in_x});
    } { // Fill upper boundary of core_2 with right mirror data of core_6
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(1, 0, 2)};
        dtos.offset = 1 * e_x;  
        Box upper_boundary_to_fill_in_y = grow(shift(Box{domain.bigEnd(iy) * e_y, domain.bigEnd()}, e_y), e_x);
        multi_block_boundaries.push_back({&core_2, &core_6, dtos, upper_boundary_to_fill_in_y});
    }
    // ------------------------------------------------------------------------- // 12 
    {   // Fill left boundary of core_6 with upper mirror data of core_4
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(1, 0, 2)};  // Swap x and y if needed
        dtos.offset = -(domain.bigEnd(iy) + 1) * e_y + domain.bigEnd(ix) * e_x;
        dtos.sign = IntVect{AMREX_D_DECL(-1, 1, 1)};
        Box left_boundary_to_fill_in_x = grow(shift(Box{domain.smallEnd(), domain.bigEnd() - domain.bigEnd(ix) * e_x}, -e_x), e_y); 
        multi_block_boundaries.push_back({&core_6, &core_4, dtos, left_boundary_to_fill_in_x});
    } { // Fill upper boundary of core_4 with left mirror data of core_6
        NonLocalBC::MultiBlockIndexMapping dtos{};
        dtos.permutation = IntVect{AMREX_D_DECL(1, 0, 2)};
        dtos.offset = (domain.bigEnd(ix) + 1) * e_x + domain.bigEnd(iy) * e_y;
        dtos.sign = IntVect{AMREX_D_DECL(1, -1, 1)};
        Box upper_boundary_to_fill_in_y = grow(shift(Box{domain.bigEnd(iy) * e_y, domain.bigEnd()}, e_y), e_x);
        multi_block_boundaries.push_back({&core_4, &core_6, dtos, upper_boundary_to_fill_in_y});
    }



    FillBoundaryFn FillBoundary{std::move(multi_block_boundaries)};

    int step = 0;
    const Real min_dx1_dy2 = std::min(geom1.CellSize(0), geom2.CellSize(1)); // here we can leave unaltered 
    const Real cfl = 1.0;
    const Real dt = cfl * min_dx1_dy2;
    const Real final_time = 4.0;
    Real time_point = 0.0;

    WritePlotfiles(core_1, core_2, core_3, core_4, core_5, core_6, time_point, step);
    while (time_point < final_time) {
        FillBoundary(core_1, core_2, core_3, core_4, core_5, core_6);

        core_1.AdvanceInTime(dt);
        core_2.AdvanceInTime(dt);
        core_3.AdvanceInTime(dt);
        core_4.AdvanceInTime(dt);
        core_5.AdvanceInTime(dt);
        core_6.AdvanceInTime(dt);

        time_point += dt;
        step += 1;

        amrex::Print() << "Step #" << step << ", Time Point = " << time_point << '\n';

        WritePlotfiles(core_1, core_2, core_3, core_4, core_5, core_6, time_point, step);
    }
}
