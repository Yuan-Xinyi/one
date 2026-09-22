// Straight-line drawing under the implicit-force constraint as a constrained
// motion-planning query, solved by OMPL. Derived from ompl_onestroke.cpp; the
// additions are (a) the cone axis is the task's own n_target, (b) a state is
// valid only if the effective end-effector stiffness along the pen,
// k_eff = 1 / (c_nn + mu c_nt) with C = J K_q^-1 J^T, stays below the cap
// (the constant-force encoding needs a compliant posture), and (c) a motion
// is valid only if ln k_eff changes slowly enough along the arc that the
// depth tracking can follow (|d ln k / ds| <= rate_max).
//
// Build:
//   g++ -O2 -std=c++17 ompl_force_line.cpp -o ompl_force_line \
//       -I/usr/local/include/ompl-1.6 -I/usr/include/eigen3 \
//       -L/usr/local/lib -lompl -Wl,-rpath,/usr/local/lib
// (original one-stroke header follows)
// One-stroke drawing as a constrained motion-planning query, solved by OMPL.
//
// The state is (q, s): the seven joints and how far along the figure the pen
// has got. The equality constraint p(q) = path(s) says the pen tip is on the
// figure, so the reachable set is a manifold of co-dimension three in the
// eight-dimensional ambient space, and drawing the figure in one stroke is a
// path on that manifold from s = 0 to s = L.
//
// This is the formulation the earlier attempts could not handle. A reactive
// law walks the manifold locally and can walk into a dead end; an interior
// point NLP needs an initial guess that already lies on one connected sheet of
// it, which independent IK does not provide. A sampling-based planner needs
// neither: it scatters samples on the manifold, connects what it can, and is
// probabilistically complete.
//
// Two things make the query honest rather than decorative:
//   * motions must ADVANCE (ds > 0) and respect the per-joint velocity box
//     over the arc they cover, so the planner cannot "draw" by reconfiguring
//     in place -- at a constant feed that is lifting the pen;
//   * the kinematics and the collision model are the same ones the rest of the
//     study uses, checked against them to float round-off before any planning.
//
// Build:
//   g++ -O2 -std=c++17 ompl_onestroke.cpp -o ompl_onestroke \
//       -I/usr/local/include/ompl-1.6 -I/usr/include/eigen3 \
//       -L/usr/local/lib -lompl -Wl,-rpath,/usr/local/lib
#include <ompl/base/Constraint.h>
#include <ompl/base/ConstrainedSpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/constraint/ProjectedStateSpace.h>
#include <ompl/base/spaces/constraint/AtlasStateSpace.h>
#include <ompl/base/spaces/constraint/TangentBundleStateSpace.h>
#include <ompl/base/goals/GoalRegion.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/geometric/planners/kpiece/KPIECE1.h>
#include <ompl/geometric/planners/est/EST.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/prm/PRM.h>
#include <ompl/util/Console.h>
#include <ompl/util/RandomNumbers.h>

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

namespace ob = ompl::base;
namespace og = ompl::geometric;

// ------------------------------------------------------------------ model
static const double TCP_OFFSET = 0.1034 + 0.10;
static const Eigen::Vector3d FLANGE(0.0, 0.0, 0.107 + TCP_OFFSET);
static const double LMT_LO[7] = {-2.7437, -1.7837, -2.9007, -3.0421,
                                 -2.8065, 0.5445, -3.0159};
static const double LMT_UP[7] = {2.7437, 1.7837, 2.9007, -0.1518,
                                 2.8065, 4.5169, 3.0159};
static const double QD[7] = {2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61};

struct Model
{
    Eigen::Matrix4d ZT[7];
    // collision spheres
    std::vector<Eigen::Vector3d> ctr;
    std::vector<double> rad;
    std::vector<int> lnk;
    std::vector<int> pi, pj;
    std::vector<double> rsum;

    Model()
    {
        const double a[7] = {0.0, -M_PI / 2, M_PI / 2, M_PI / 2,
                             -M_PI / 2, M_PI / 2, M_PI / 2};
        const double t[7][3] = {{0, 0, 0.333}, {0, 0, 0}, {0, -0.316, 0},
                                {0.0825, 0, 0}, {-0.0825, 0.384, 0},
                                {0, 0, 0}, {0.088, 0, 0}};
        for (int i = 0; i < 7; ++i)
        {
            ZT[i].setIdentity();
            const double c = std::cos(a[i]), s = std::sin(a[i]);
            ZT[i](1, 1) = c; ZT[i](1, 2) = -s;
            ZT[i](2, 1) = s; ZT[i](2, 2) = c;
            ZT[i](0, 3) = t[i][0]; ZT[i](1, 3) = t[i][1]; ZT[i](2, 3) = t[i][2];
        }
    }

    void loadSpheres(const std::string &f)
    {
        std::ifstream in(f);
        int n, m;
        in >> n;
        ctr.resize(n); rad.resize(n); lnk.resize(n);
        for (int k = 0; k < n; ++k)
            in >> ctr[k].x() >> ctr[k].y() >> ctr[k].z() >> rad[k] >> lnk[k];
        in >> m;
        pi.resize(m); pj.resize(m); rsum.resize(m);
        for (int k = 0; k < m; ++k)
        {
            in >> pi[k] >> pj[k];
            rsum[k] = rad[pi[k]] + rad[pj[k]];
        }
    }

    void links(const double *q, Eigen::Matrix4d *out) const
    {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        out[0] = T;
        for (int i = 0; i < 7; ++i)
        {
            Eigen::Matrix4d Tj = T * ZT[i];
            Eigen::Matrix4d Rz = Eigen::Matrix4d::Identity();
            const double c = std::cos(q[i]), s = std::sin(q[i]);
            Rz(0, 0) = c; Rz(0, 1) = -s; Rz(1, 0) = s; Rz(1, 1) = c;
            T = Tj * Rz;
            out[i + 1] = T;
        }
    }

    // tool point, tool z axis, and the position Jacobian, in one pass
    void tcp(const double *q, Eigen::Vector3d &p, Eigen::Vector3d &z,
             Eigen::Matrix<double, 3, 7> *J) const
    {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        Eigen::Vector3d ax[7], og_[7];
        for (int i = 0; i < 7; ++i)
        {
            Eigen::Matrix4d Tj = T * ZT[i];
            ax[i] = Tj.block<3, 1>(0, 2);
            og_[i] = Tj.block<3, 1>(0, 3);
            Eigen::Matrix4d Rz = Eigen::Matrix4d::Identity();
            const double c = std::cos(q[i]), s = std::sin(q[i]);
            Rz(0, 0) = c; Rz(0, 1) = -s; Rz(1, 0) = s; Rz(1, 1) = c;
            T = Tj * Rz;
        }
        p = T.block<3, 3>(0, 0) * FLANGE + T.block<3, 1>(0, 3);
        z = T.block<3, 1>(0, 2);
        if (J)
            for (int i = 0; i < 7; ++i)
                J->col(i) = ax[i].cross(p - og_[i]);
    }

    double collMargin(const double *q) const
    {
        Eigen::Matrix4d L[8];
        links(q, L);
        std::vector<Eigen::Vector3d> pos(ctr.size());
        for (size_t k = 0; k < ctr.size(); ++k)
            pos[k] = L[lnk[k]].block<3, 3>(0, 0) * ctr[k]
                   + L[lnk[k]].block<3, 1>(0, 3);
        double m = 1e9;
        for (size_t k = 0; k < pi.size(); ++k)
            m = std::min(m, (pos[pi[k]] - pos[pj[k]]).norm() - rsum[k]);
        return m;
    }
};

// ------------------------------------------------------------------ figure
static Eigen::Vector3d N_TARGET(0.0, 0.0, -1.0);   // cone axis of the task
struct Figure
{
    std::vector<Eigen::Vector3d> p, t;   // samples and unit tangents
    double step = 0.0, L = 0.0;

    void load(const std::string &f)
    {
        std::ifstream in(f);
        int n;
        in >> n >> step;
        in >> N_TARGET.x() >> N_TARGET.y() >> N_TARGET.z();
        p.resize(n); t.resize(n);
        for (int k = 0; k < n; ++k)
            in >> p[k].x() >> p[k].y() >> p[k].z()
               >> t[k].x() >> t[k].y() >> t[k].z();
        L = step * (n - 1);
    }
    // linear interpolation, so the constraint is C0 and its Jacobian is the
    // tangent; the sampling is 2 mm, far below any feature of these figures
    void at(double s, Eigen::Vector3d &pp, Eigen::Vector3d &tt) const
    {
        double u = std::max(0.0, std::min(L, s)) / step;
        int i = std::min((int)p.size() - 2, std::max(0, (int)std::floor(u)));
        double a = u - i;
        pp = (1 - a) * p[i] + a * p[i + 1];
        tt = t[i];
    }
};

// Bank of admissible (q, s) states along the figure, used by the informed
// sampler. Built outside with the same cone-constrained inverse kinematics
// every method's start pool and the pointwise bound already use; the sampler
// perturbs a random bank state and lets the projection pull it back onto the
// path manifold. Without it, uniform ambient sampling almost never lands
// inside a 5-degree tilt cone and the planner starves for reasons that have
// nothing to do with planning.
static std::vector<std::array<double, 8>> BANK;

static Model MODEL;
static Figure FIG;
static double COS_LIM = 0.0;
static double V_FEED = 0.2;
static const double KQ[7] = {600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0};
static double MU = 0.0, KN_MAX = 1e12, RATE_MAX = 1e12;

// effective stiffness along the pen axis z with the path tangent t
static double kEff(const double *q, const Eigen::Vector3d &t)
{
    Eigen::Vector3d p, z;
    Eigen::Matrix<double, 3, 7> J;
    MODEL.tcp(q, p, z, &J);
    double cnn = 0.0, cnt = 0.0;
    for (int i = 0; i < 7; ++i)
    {
        const double a = z.dot(J.col(i)), b = t.dot(J.col(i));
        cnn += a * a / KQ[i];
        cnt += a * b / KQ[i];
    }
    const double den = std::max(cnn + MU * cnt, 0.2 * cnn);
    return 1.0 / std::max(den, 1e-12);
}

// ---------------------------------------------------------- bank sampler
class BankSampler : public ob::StateSampler
{
public:
    explicit BankSampler(const ob::StateSpace *sp) : ob::StateSampler(sp) {}

    void sampleUniform(ob::State *st) override
    {
        auto *x = st->as<ob::RealVectorStateSpace::StateType>();
        const auto &b = BANK[rng_.uniformInt(0, (int)BANK.size() - 1)];
        for (int i = 0; i < 7; ++i)
            x->values[i] = b[i] + rng_.gaussian(0.0, 0.10);
        x->values[7] = b[7] + rng_.gaussian(0.0, 0.02);
    }
    void sampleUniformNear(ob::State *st, const ob::State *near,
                           double dist) override
    {
        auto *x = st->as<ob::RealVectorStateSpace::StateType>();
        const auto *n = near->as<ob::RealVectorStateSpace::StateType>();
        for (int i = 0; i < 8; ++i)
            x->values[i] = n->values[i] + rng_.uniformReal(-dist, dist);
    }
    void sampleGaussian(ob::State *st, const ob::State *mean,
                        double stdDev) override
    {
        auto *x = st->as<ob::RealVectorStateSpace::StateType>();
        const auto *m = mean->as<ob::RealVectorStateSpace::StateType>();
        for (int i = 0; i < 8; ++i)
            x->values[i] = m->values[i] + rng_.gaussian(0.0, stdDev);
    }
private:
    ompl::RNG rng_;
};

// ------------------------------------------------------------- constraint
class OnPath : public ob::Constraint
{
public:
    // The tolerance IS the tolerance tube. Every other controller may wander
    // 20 mm off the figure anywhere along it and uses that freedom on the
    // straights, not only at corners; pinning the planner to 0.1 mm made it
    // solve a 200x tighter problem. 10 mm here plus the fillet's 6.4 mm
    // worst-case deviation keeps everything inside the same 20 mm tube the
    // others are graded against.
    explicit OnPath(double tol) : ob::Constraint(8, 3, tol) {}

    void function(const Eigen::Ref<const Eigen::VectorXd> &x,
                  Eigen::Ref<Eigen::VectorXd> out) const override
    {
        Eigen::Vector3d p, z, pp, tt;
        MODEL.tcp(x.data(), p, z, nullptr);
        FIG.at(x[7], pp, tt);
        out = p - pp;
    }

    void jacobian(const Eigen::Ref<const Eigen::VectorXd> &x,
                  Eigen::Ref<Eigen::MatrixXd> out) const override
    {
        Eigen::Vector3d p, z, pp, tt;
        Eigen::Matrix<double, 3, 7> J;
        MODEL.tcp(x.data(), p, z, &J);
        FIG.at(x[7], pp, tt);
        out.setZero();
        out.block<3, 7>(0, 0) = J;
        out.block<3, 1>(0, 7) = -tt;
    }
};

// --------------------------------------------------------------- validity
static bool isValid(const ob::State *st)
{
    const auto *x = st->as<ob::ConstrainedStateSpace::StateType>();
    double q[7];
    for (int i = 0; i < 7; ++i)
    {
        q[i] = (*x)[i];
        if (q[i] < LMT_LO[i] || q[i] > LMT_UP[i]) return false;
    }
    if ((*x)[7] < -0.01 || (*x)[7] > FIG.L + 1e-9) return false;
    Eigen::Vector3d p, z, pp, tt;
    MODEL.tcp(q, p, z, nullptr);
    if (z.dot(N_TARGET) < COS_LIM) return false;     // cone about the task axis
    FIG.at((*x)[7], pp, tt);
    if (kEff(q, tt) > KN_MAX) return false;           // implicit-force cap
    return MODEL.collMargin(q) >= 0.0;
}

// ------------------------------------------------------- motion validity
// A motion is admissible only if it advances along the figure and stays inside
// the per-joint velocity box for the arc it covers. Without this the planner
// would happily reconfigure the arm at a standstill, which at a constant feed
// is not drawing -- it is lifting the pen.
class FeedMotionValidator : public ob::MotionValidator
{
public:
    explicit FeedMotionValidator(const ob::SpaceInformationPtr &si)
      : ob::MotionValidator(si),
        css_(si->getStateSpace()->as<ob::ConstrainedStateSpace>()) {}

    bool checkMotion(const ob::State *s1, const ob::State *s2) const override
    {
        std::vector<ob::State *> geo;
        if (!css_->discreteGeodesic(s1, s2, true, &geo)) { cleanup(geo); return false; }
        bool ok = checkChain(geo);
        cleanup(geo);
        return ok;
    }

    bool checkMotion(const ob::State *s1, const ob::State *s2,
                     std::pair<ob::State *, double> &lastValid) const override
    {
        std::vector<ob::State *> geo;
        bool full = css_->discreteGeodesic(s1, s2, true, &geo);
        bool ok = checkChain(geo);
        if (!ok || !full)
        {
            if (lastValid.first != nullptr && !geo.empty())
                css_->copyState(lastValid.first, geo.front());
            lastValid.second = 0.0;
        }
        cleanup(geo);
        return ok && full;
    }

private:
    bool checkChain(const std::vector<ob::State *> &g) const
    {
        if (g.size() < 2) return false;
        for (size_t k = 0; k + 1 < g.size(); ++k)
        {
            const auto *a = g[k]->as<ob::ConstrainedStateSpace::StateType>();
            const auto *b = g[k + 1]->as<ob::ConstrainedStateSpace::StateType>();
            const double ds = (*b)[7] - (*a)[7];
            if (ds <= 0.0) return false;                  // must advance
            const double dt = ds / V_FEED;
            for (int i = 0; i < 7; ++i)
                if (std::fabs((*b)[i] - (*a)[i]) > QD[i] * dt)
                    return false;                          // velocity box
            if (!si_->isValid(g[k + 1])) return false;
            {
                // the commanded depth d = f / k_eff must be trackable:
                // bound the log-stiffness slope along the arc
                double qa[7], qb[7];
                for (int i = 0; i < 7; ++i) { qa[i] = (*a)[i]; qb[i] = (*b)[i]; }
                Eigen::Vector3d pa, ta, pb, tb;
                FIG.at((*a)[7], pa, ta); FIG.at((*b)[7], pb, tb);
                const double dl = std::fabs(std::log(kEff(qb, tb)) - std::log(kEff(qa, ta)));
                if (dl > RATE_MAX * ds) return false;
            }
        }
        return si_->isValid(g.front());
    }

    void cleanup(std::vector<ob::State *> &g) const
    {
        for (auto *s : g) css_->freeState(s);
        g.clear();
    }

    ob::ConstrainedStateSpace *css_;
};

// ------------------------------------------------------------------- goal
class ReachEnd : public ob::GoalRegion
{
public:
    explicit ReachEnd(const ob::SpaceInformationPtr &si) : ob::GoalRegion(si)
    { setThreshold(1e-3); }

    double distanceGoal(const ob::State *st) const override
    {
        const auto *x = st->as<ob::ConstrainedStateSpace::StateType>();
        return std::max(0.0, FIG.L - (*x)[7]);
    }
};

// ------------------------------------------------------------------- main
int main(int argc, char **argv)
{
    if (argc < 6)
    {
        std::cerr << "usage: ompl_force_line spheres.txt figure.txt q0.txt cone_deg time_s "
                     "[out] [planner] [space] [delta] [ctol] [bank|-] [mu] [kn_max] [rate_max]\n";
        return 2;
    }
    ompl::msg::setLogLevel(ompl::msg::LOG_ERROR);
    MODEL.loadSpheres(argv[1]);
    FIG.load(argv[2]);
    COS_LIM = std::cos(std::atof(argv[4]) * M_PI / 180.0);
    const double budget = std::atof(argv[5]);
    const std::string out = (argc > 6) ? argv[6] : "solution.txt";
    const std::string pl = (argc > 7) ? argv[7] : "rrt";

    // The framework's selection stage picks its start from the full IK
    // candidate pool; handing the planner three arbitrary pool entries in
    // separate runs was not the same offer. All candidates go in as start
    // states of ONE query -- the planner's own equivalent of a selection
    // stage -- and the budget is shared across them by the planner itself.
    std::vector<std::array<double, 7>> q0s;
    {
        std::ifstream in(argv[3]);
        std::array<double, 7> q;
        while (in >> q[0] >> q[1] >> q[2] >> q[3] >> q[4] >> q[5] >> q[6])
            q0s.push_back(q);
    }
    if (q0s.empty()) { std::cerr << "no start states\n"; return 2; }

    auto rv(std::make_shared<ob::RealVectorStateSpace>(8));
    ob::RealVectorBounds b(8);
    for (int i = 0; i < 7; ++i) { b.setLow(i, LMT_LO[i]); b.setHigh(i, LMT_UP[i]); }
    b.setLow(7, -0.01); b.setHigh(7, FIG.L);
    rv->setBounds(b);

    // OMPL's own constrained-planning demos sweep the three state spaces and
    // several planners rather than fixing one; both are selectable here so the
    // baseline can be run the way the library intends.
    const std::string spc = (argc > 8) ? argv[8] : "projected";
    const double delta = (argc > 9) ? std::atof(argv[9]) : 0.05;
    const double ctol = (argc > 10) ? std::atof(argv[10]) : 0.01;
    auto con(std::make_shared<OnPath>(ctol));
    std::shared_ptr<ob::ConstrainedStateSpace> css;
    std::shared_ptr<ob::ConstrainedSpaceInformation> si;
    if (spc == "atlas")
    {
        auto a(std::make_shared<ob::AtlasStateSpace>(rv, con));
        css = a; si = std::make_shared<ob::ConstrainedSpaceInformation>(a);
    }
    else if (spc == "tangent")
    {
        auto t(std::make_shared<ob::TangentBundleStateSpace>(rv, con));
        css = t; si = std::make_shared<ob::TangentBundleSpaceInformation>(t);
    }
    else
    {
        auto pr(std::make_shared<ob::ProjectedStateSpace>(rv, con));
        css = pr; si = std::make_shared<ob::ConstrainedSpaceInformation>(pr);
    }
    css->setDelta(delta);
    css->setLambda(2.0);
    if (argc > 11 && std::string(argv[11]) != "-")
    {
        std::ifstream in(argv[11]);
        std::array<double, 8> row;
        while (in >> row[0] >> row[1] >> row[2] >> row[3] >> row[4]
                  >> row[5] >> row[6] >> row[7])
            BANK.push_back(row);
        if (!BANK.empty())
            rv->setStateSamplerAllocator(
                [](const ob::StateSpace *sp) -> ob::StateSamplerPtr
                { return std::make_shared<BankSampler>(sp); });
        std::cout << "BANK " << BANK.size() << "\n";
    }
    if (argc > 12) MU = std::atof(argv[12]);
    if (argc > 13) KN_MAX = std::atof(argv[13]);
    if (argc > 14) RATE_MAX = std::atof(argv[14]);
    std::cout << "FORCE mu " << MU << " kn_max " << KN_MAX << " rate_max " << RATE_MAX << "\n";
    si->setStateValidityChecker(isValid);
    si->setMotionValidator(std::make_shared<FeedMotionValidator>(si));
    si->setup();

    og::SimpleSetup ss(si);
    int n_valid = 0;
    for (const auto &q0 : q0s)
    {
        ob::ScopedState<> start(css);
        for (int i = 0; i < 7; ++i) start[i] = q0[i];
        start[7] = 0.0;
        if (!con->isSatisfied(start.get()))
            con->project(start.get());
        if (!si->isValid(start.get()))
            continue;
        if (spc == "atlas" || spc == "tangent")
            css->as<ob::AtlasStateSpace>()->anchorChart(start.get());
        ss.addStartState(start);
        ++n_valid;
    }
    std::cout << "STARTS " << n_valid << " of " << q0s.size() << "\n";
    if (n_valid == 0) { std::cout << "SOLVED 0\n"; return 0; }
    ss.setGoal(std::make_shared<ReachEnd>(si));

    // Only unidirectional planners are admissible: the motion validator is
    // asymmetric (the pen may not go backwards), which is exactly what
    // RRTConnect and PRM assume away. They are offered anyway so the
    // restriction can be demonstrated rather than asserted.
    ob::PlannerPtr planner;
    if (pl == "kpiece")          planner = std::make_shared<og::KPIECE1>(si);
    else if (pl == "est")        planner = std::make_shared<og::EST>(si);
    else if (pl == "rrtconnect") planner = std::make_shared<og::RRTConnect>(si);
    else if (pl == "prm")        planner = std::make_shared<og::PRM>(si);
    else                         planner = std::make_shared<og::RRT>(si);
    ss.setPlanner(planner);

    ob::PlannerStatus stat = ss.solve(budget);
    const bool exact = (stat == ob::PlannerStatus::EXACT_SOLUTION);
    std::cout << "STATUS " << stat.asString() << "\n";
    if (!ss.haveSolutionPath()) { std::cout << "SOLVED 0\n"; return 0; }

    auto path = ss.getSolutionPath();
    path.interpolate();
    std::ofstream o(out);
    o.precision(12);
    o << path.getStateCount() << "\n";
    for (std::size_t k = 0; k < path.getStateCount(); ++k)
    {
        const auto *x = path.getState(k)
                            ->as<ob::ConstrainedStateSpace::StateType>();
        for (int i = 0; i < 8; ++i) o << (*x)[i] << (i == 7 ? '\n' : ' ');
    }
    const auto *last = path.getState(path.getStateCount() - 1)
                           ->as<ob::ConstrainedStateSpace::StateType>();
    std::cout << "CONFIG " << spc << " " << pl << " delta " << delta << "\n";
    std::cout << "SOLVED " << (exact ? 1 : 0)
              << " states " << path.getStateCount()
              << " s_end " << (*last)[7] << " of " << FIG.L << "\n";
    return 0;
}
