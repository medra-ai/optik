use divan::{black_box, Divan};
use nalgebra::Isometry3;

use optik::*;

const BENCH_MODEL_STR: &str = include_str!("../tests/data/ur3e.urdf");

fn load_benchmark_model() -> Robot {
    Robot::from_urdf_str(BENCH_MODEL_STR, "ur_base_link", "ur_ee_link")
}

#[divan::bench]
fn jacobian(bencher: divan::Bencher) {
    let robot = load_benchmark_model();
    let q = robot.random_configuration(&mut rand::thread_rng());

    bencher.bench(|| {
        robot.jacobian_local(black_box(&q));
    });
}

mod gradient {
    use super::*;

    #[divan::bench]
    fn analytical(bencher: divan::Bencher) {
        let robot = load_benchmark_model();

        let q = robot.random_configuration(&mut rand::thread_rng());
        let tfm_target = Isometry3::identity();
        let args = ObjectiveArgs {
            robot: robot.clone(),
            config: SolverConfig {
                gradient_mode: GradientMode::Analytical,
                ..Default::default()
            },
            tfm_target,
        };

        bencher.with_inputs(|| vec![0.0; q.len()]).bench_refs(|g| {
            objective_grad(black_box(&q), g, &args);
        });
    }

    #[divan::bench]
    fn numerical(bencher: divan::Bencher) {
        let robot = load_benchmark_model();

        let q = robot.random_configuration(&mut rand::thread_rng());
        let tfm_target = Isometry3::identity();
        let args = ObjectiveArgs {
            robot: robot.clone(),
            config: SolverConfig {
                gradient_mode: GradientMode::Numerical,
                ..Default::default()
            },
            tfm_target,
        };

        bencher.with_inputs(|| vec![0.0; q.len()]).bench_refs(|g| {
            objective_grad(black_box(&q), g, &args);
        });
    }
}

#[divan::bench]
fn objective_fn(bencher: divan::Bencher) {
    let robot = load_benchmark_model();

    let q = robot.random_configuration(&mut rand::thread_rng());
    let tfm_target = Isometry3::identity();
    let args = ObjectiveArgs {
        robot: robot.clone(),
        config: SolverConfig::default(),
        tfm_target,
    };

    bencher.bench(|| objective(black_box(&q), &args));
}

#[divan::bench]
fn solve_ik(bencher: divan::Bencher) {
    let robot = load_benchmark_model();
    let config = SolverConfig::default();

    let x0 = vec![0.1, 0.2, 0.0, 0.3, -0.2, -1.1];
    let tfm_target = robot.fk(&[-0.1, -0.2, 0.0, -0.3, 0.2, 1.1]);

    bencher.bench(|| robot.ik(&config, black_box(&tfm_target), x0.clone()));
}

fn main() {
    optik::set_parallelism(1);

    Divan::from_args().sample_count(1000).main();
}
