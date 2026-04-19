use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

use nalgebra::{
    DVectorView, Isometry3, Matrix6xX, Translation3, UnitQuaternion, UnitVector3, Vector3,
};
use nlopt::{Algorithm, Nlopt, SuccessState, Target};
use ordered_float::OrderedFloat;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::{
    prelude::{IntoParallelIterator, ParallelIterator},
    ThreadPool, ThreadPoolBuilder,
};

mod config;
pub mod kinematics;
pub mod math;
pub mod objective;

pub use config::*;
use kinematics::*;
use objective::*;

pub struct Robot {
    chain: KinematicChain,
    thread_pool: ThreadPool,
}

impl Robot {
    pub fn new(chain: KinematicChain) -> Self {
        Self {
            chain,
            thread_pool: ThreadPoolBuilder::default().build().unwrap(),
        }
    }

    pub fn from_urdf(urdf: &urdf_rs::Robot, base_link: &str, ee_link: &str) -> Self {
        let chain = KinematicChain::from_urdf(urdf, base_link, ee_link);
        Robot::new(chain)
    }

    pub fn from_urdf_file(path: impl AsRef<Path>, base_link: &str, ee_link: &str) -> Self {
        let urdf = urdf_rs::read_file(path).expect("error parsing URDF file!");
        Robot::from_urdf(&urdf, base_link, ee_link)
    }

    pub fn from_urdf_str(urdf: &str, base_link: &str, ee_link: &str) -> Self {
        let urdf = urdf_rs::read_from_string(urdf).expect("error parsing URDF file!");
        Robot::from_urdf(&urdf, base_link, ee_link)
    }
}

impl Robot {
    pub fn set_parallelism(&mut self, n: usize) {
        // Don't rebuild the thread pool (an expensive operation) unless this
        // value has actually changed.
        if n != self.thread_pool.current_num_threads() {
            self.thread_pool = ThreadPoolBuilder::new().num_threads(n).build().unwrap()
        }
    }

    pub fn num_positions(&self) -> usize {
        self.chain.num_positions()
    }

    pub fn joint_limits(&self) -> (Vec<f64>, Vec<f64>) {
        self.chain
            .joints
            .iter()
            .flat_map(|j| j.limits.clone())
            .unzip()
    }

    pub fn random_configuration(&self, rng: &mut impl rand::Rng) -> Vec<f64> {
        let (lb, ub) = self.joint_limits();
        (0..self.num_positions())
            .map(|i| rng.gen_range(lb[i]..=ub[i]))
            .collect()
    }

    pub fn joint_jacobian(&self, fk: &ForwardKinematics) -> Matrix6xX<f64> {
        self.chain.joint_jacobian(fk)
    }

    pub fn fk(&self, q: &[f64], ee_offset: &Isometry3<f64>) -> ForwardKinematics {
        self.chain.forward_kinematics(q, ee_offset)
    }

    pub fn ik(
        &self,
        config: &SolverConfig,
        tfm_target: &Isometry3<f64>,
        x0: Vec<f64>,
        ee_offset: &Isometry3<f64>,
    ) -> Option<(Vec<f64>, f64)> {
        // Complain if the provided seed is out side the joint limits. The
        // solver may be able to handle this, but it seems likely that there is
        // a bug in the user's program if this occurs and we should notify them.
        let (lb, ub) = self.joint_limits();
        if x0.iter().enumerate().any(|(i, q)| *q < lb[i] || *q > ub[i]) {
            panic!("seed joint position outside of joint limits")
        }

        // Compute the time at which the user-specified timeout will expire. We
        // will ensure that no solve threads continue iterating beyond this
        // time. If no max time is specified then run until the retry count is
        // exhausted.
        let start_time = Instant::now();
        let is_timed_out = || {
            let elapsed_time = Instant::now().duration_since(start_time).as_secs_f64();
            config.max_time > 0.0 && elapsed_time > config.max_time
        };

        // In SolutionMode::Speed, when one thread finds a solution which
        // satisfies the tolerances, it will immediately tell all of the other
        // threads to exit.
        let should_exit = Arc::new(AtomicBool::new(false));

        // If a maximum number of restarts is specified then we limit to that.
        // Otherwise, limit to a huge number.
        let max_restarts = if config.max_restarts > 0 {
            config.max_restarts
        } else {
            u64::MAX
        };

        // HEURISTIC: Establish an artifical stopping criteria if the
        // optimization seems to have stalled. This case will not be considered
        // a successful result, but it will free up this thread to restart from
        // another seed.
        let tol_df = if config.tol_df > 0.0 {
            config.tol_df
        } else {
            // Stop when the gradient is a few orders of magnitude smaller than
            // the requested precision. When this happens, the optimization is
            // most likely stuck in a local minimum.
            //
            // We can't make this too big w.r.t. tol_f or we will exit in cases
            // that are still on track to converge to a valid solution.
            1e-3 * config.tol_f
        };

        // Build a parallel stream of solutions from which we can choose how to
        // draw a final result.
        self.thread_pool.install(|| {
            let solution_stream = (0..max_restarts)
                .into_par_iter()
                .panic_fuse()
                .map(|i| {
                    let mut optimizer = Nlopt::new(
                        Algorithm::Slsqp,
                        self.num_positions(),
                        |x: &[f64], grad: Option<&mut [f64]>, fk: &mut ForwardKinematics| {
                            // Early-exit if we are out of time, or if another
                            // thread has already found a satisfactory solution.
                            if is_timed_out() || should_exit.load(Ordering::Relaxed) {
                                return None;
                            }

                            // Share kinematic results between the gradient and
                            // objective function evaluations, when possible.
                            self.chain.forward_kinematics_mut(x, ee_offset, fk);

                            // Compute the gradient only if it was requested by
                            // the optimizer.
                            if let Some(g) = grad {
                                objective_grad(
                                    self,
                                    tfm_target,
                                    fk,
                                    g,
                                    Vector3::from_row_slice(&config.linear_weight),
                                    Vector3::from_row_slice(&config.angular_weight),
                                );
                            }

                            // Always compute the objective value.
                            Some(objective(
                                self,
                                tfm_target,
                                fk,
                                Vector3::from_row_slice(&config.linear_weight),
                                Vector3::from_row_slice(&config.angular_weight),
                            ))
                        },
                        Target::Minimize,
                        // Cache the forward kinematic container within each
                        // thread to avoid re-allocating memory each objective
                        // iteration.
                        ForwardKinematics::default(),
                    );

                    optimizer.set_stopval(config.tol_f).unwrap();
                    optimizer.set_ftol_abs(tol_df).unwrap();
                    optimizer.set_xtol_abs1(config.tol_dx).unwrap();
                    optimizer.set_lower_bounds(lb.as_slice()).unwrap();
                    optimizer.set_upper_bounds(ub.as_slice()).unwrap();

                    // LBFGS memory size is chosen empirically. This value seems
                    // to give the best performance for ~6 DoF robot arms.
                    const LBFGS_STORAGE_SIZE: usize = 10;
                    optimizer
                        .set_vector_storage(Some(LBFGS_STORAGE_SIZE))
                        .unwrap();

                    // Fix a global RNG seed, which is used to compute sub-seeds
                    // for each thread.
                    const RNG_SEED: u64 = 42;
                    let mut rng = ChaCha8Rng::seed_from_u64(RNG_SEED);
                    rng.set_stream(i);

                    // The first attempt gets the initial seed provided by the
                    // caller.  All other attempts start at some random point.
                    let mut x = if i == 0 {
                        x0.clone()
                    } else {
                        self.random_configuration(&mut rng)
                    };

                    if let Ok((r, c)) = optimizer.optimize(&mut x) {
                        // Make sure that we exited for the right reasons. For
                        // example, NLopt considers a timeout to be a success
                        // but we treat it as a failure.
                        let success = (config.tol_f >= 0.
                            && matches!(r, SuccessState::StopValReached))
                            || (config.tol_df >= 0. && matches!(r, SuccessState::FtolReached))
                            || (config.tol_dx >= 0. && matches!(r, SuccessState::XtolReached));

                        if success {
                            if config.solution_mode == SolutionMode::Speed {
                                should_exit.store(true, Ordering::Relaxed);
                            }

                            return Some((x, c));
                        }
                    }

                    None
                })
                // Stop issuing new solve requests once we run out of time.
                .take_any_while(|_| !is_timed_out())
                // Ignore failed solve attemps.
                .flatten();

            match config.solution_mode {
                SolutionMode::Quality => {
                    // Continue solving until the timeout is reached and take
                    // the best of all solutions. In this case, the cost of a
                    // given solution is computed as its distance from the seed.
                    solution_stream.min_by_key(|(x, _)| {
                        let x = DVectorView::from_slice(x, x.len());
                        let x0 = DVectorView::from_slice(&x0, x0.len());

                        OrderedFloat(x.metric_distance(&x0))
                    })
                }
                SolutionMode::Speed => {
                    // Take the first solution which satisfies the tolerance.
                    solution_stream.find_any(|_| true)
                }
            }
        })
    }

    /// Project a joint configuration onto the angle-between-two-vectors
    /// cone using Gauss-Newton iteration on the scalar constraint
    /// `f(q) = angle(q) - max_angle`.
    ///
    /// Unlike `apply_angle_between_two_vectors_constraint` (which runs a
    /// full IK solve to find any joint config satisfying the inequality),
    /// this projector is *continuous in the seed state*. It is the
    /// appropriate projector to use with OMPL's
    /// `ProjectedStateSpace::discreteGeodesic` where consecutive
    /// projected samples must stay close for `checkMotion` to accept
    /// an edge.
    ///
    /// Behaviour:
    /// - If the seed already satisfies the constraint (angle <= max_angle
    ///   + tol), it is returned unchanged.
    /// - Otherwise Newton steps along the analytical gradient of the
    ///   tip-axis angle w.r.t. joints, clamped to joint limits at each
    ///   step. Returns `None` if the iteration does not converge or the
    ///   gradient becomes singular.
    pub fn apply_angle_between_two_vectors_constraint_newton(
        &self,
        source_vector_tip_frame: UnitVector3<f64>,
        target_vector: UnitVector3<f64>,
        max_angle: f64,
        ee_transform: Isometry3<f64>,
        mut q: Vec<f64>,
        max_iters: usize,
        tol: f64,
    ) -> Option<Vec<f64>> {
        const SIN_ANGLE_MIN: f64 = 1e-6;
        const MIN_GRAD_NORM_SQ: f64 = 1e-12;
        const FD_EPS: f64 = 1e-5;

        let (lb, ub) = self.joint_limits();
        let n = q.len();
        for i in 0..n {
            q[i] = q[i].clamp(lb[i], ub[i]);
        }

        // Compute angle(q) and the world-frame tip axis.
        let eval = |q: &[f64]| -> (f64, Vector3<f64>) {
            let fk = self.chain.forward_kinematics(q, &ee_transform);
            let a = fk.ee_tfm().transform_vector(&source_vector_tip_frame);
            let dot = a.dot(&target_vector).clamp(-1.0, 1.0);
            (dot.acos(), a)
        };

        for _iter in 0..max_iters {
            let (angle, a) = eval(&q);
            let f = angle - max_angle;
            if f <= tol {
                return Some(q);
            }

            // Analytical gradient:
            //   d(angle)/dq = -(1 / sin(angle)) * J_w^T (a x b)
            // where J_w is the world-frame angular Jacobian at the tip.
            // The joint_jacobian() is expressed in the local ee frame,
            // so rotate the world-frame (a x b) vector into the ee
            // frame before dotting with the angular rows.
            let sin_a = angle.sin();
            let mut grad = vec![0.0; n];

            if sin_a.abs() < SIN_ANGLE_MIN {
                // Numerically unstable (tip parallel/antiparallel with
                // target). Fall back to one-sided finite differences.
                for i in 0..n {
                    let mut qp = q.clone();
                    qp[i] = (qp[i] + FD_EPS).clamp(lb[i], ub[i]);
                    let (angle_p, _) = eval(&qp);
                    grad[i] = (angle_p - angle) / FD_EPS;
                }
            } else {
                let fk = self.chain.forward_kinematics(&q, &ee_transform);
                let jac_local = self.chain.joint_jacobian(&fk);
                let r_we = fk.ee_tfm().rotation;
                let cross_world: Vector3<f64> = a.cross(&target_vector);
                let cross_local: Vector3<f64> =
                    r_we.inverse_transform_vector(&cross_world);
                let inv_sin = 1.0 / sin_a;
                for i in 0..n {
                    // angular rows are indices 3..6 in the 6xN jacobian.
                    let ax = jac_local[(3, i)];
                    let ay = jac_local[(4, i)];
                    let az = jac_local[(5, i)];
                    let d_cos = ax * cross_local[0]
                        + ay * cross_local[1]
                        + az * cross_local[2];
                    // d(angle)/dq_i = -(1/sin(angle)) * d(cos)/dq_i
                    grad[i] = -inv_sin * d_cos;
                }
            }

            let grad_norm_sq: f64 = grad.iter().map(|x| x * x).sum();
            if grad_norm_sq < MIN_GRAD_NORM_SQ {
                // Singular direction; Newton cannot make progress.
                return None;
            }
            let step = f / grad_norm_sq;
            for i in 0..n {
                q[i] = (q[i] - step * grad[i]).clamp(lb[i], ub[i]);
            }
        }

        // Final check after max_iters.
        let (angle, _) = eval(&q);
        if (angle - max_angle) <= tol {
            Some(q)
        } else {
            None
        }
    }

    pub fn apply_angle_between_two_vectors_constraint(
        &self,
        source_vector_tip_frame: UnitVector3<f64>,
        target_vector: UnitVector3<f64>,
        max_angle: f64,
        ee_transform: Isometry3<f64>,
        mut seed_joint_angles: Vec<f64>,
        config: &SolverConfig,
    ) -> Option<Vec<f64>> {
        // apply joint limits to seed joint angles
        let joint_limits = self.joint_limits();
        for index in 0..seed_joint_angles.len() {
            let joint_angle = seed_joint_angles[index];
            if joint_angle < joint_limits.0[index] {
                seed_joint_angles[index] = joint_limits.0[index]
            } else if joint_angle > joint_limits.1[index] {
                seed_joint_angles[index] = joint_limits.1[index]
            }
        }
        // calculate axis and angle of rotation
        let robot_pose: Isometry3<f64> = self.fk(&seed_joint_angles, &ee_transform).ee_tfm();
        let source_vector = robot_pose.transform_vector(&source_vector_tip_frame);
        let axis_of_rotation: Vector3<f64> = source_vector.cross(&target_vector);
        let angle_between_vectors: f64 = source_vector.dot(&target_vector).clamp(-1.0, 1.0).acos();
        if angle_between_vectors < max_angle {
            return Some(seed_joint_angles);
        };
        // Project the source vector into the valid cone
        const RNG_SEED: u64 = 42;
        let mut rng = ChaCha8Rng::seed_from_u64(RNG_SEED);
        let angle_of_rotation =
            rng.gen_range(angle_between_vectors - max_angle..angle_between_vectors);
        let rotation_onto_cone: Isometry3<f64> = Isometry3::from_parts(
            Translation3::new(0., 0., 0.),
            UnitQuaternion::from_axis_angle(
                &UnitVector3::new_normalize(axis_of_rotation),
                angle_of_rotation,
            ),
        );
        let target_pose = rotation_onto_cone * robot_pose;
        // run ik to find joint angles that match the projected pose
        let ik_solution = self.ik(&config, &target_pose, seed_joint_angles, &ee_transform);
        match ik_solution {
            Some(ik_solution) => return Some(ik_solution.0),
            None => return None,
        };
    }
}
