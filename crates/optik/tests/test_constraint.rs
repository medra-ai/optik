use approx::assert_abs_diff_eq;
use nalgebra::{Isometry3, UnitVector3, Vector3};
use optik::{Robot, SolverConfig};
use rand::{rngs::StdRng, Rng, SeedableRng};

const TEST_MODEL_STR: &str = include_str!("data/ur3e.urdf");

const AXIS_IN_TIP: Vector3<f64> = Vector3::new(0.0, 0.0, 1.0);
const AXIS_TARGET: Vector3<f64> = Vector3::new(0.0, 0.0, -1.0);

fn robot() -> Robot {
    Robot::from_urdf_str(TEST_MODEL_STR, "ur_base_link", "ur_ee_link")
}

/// Angle between the constrained tool axis and the target direction.
fn cone_angle(robot: &Robot, q: &[f64]) -> f64 {
    let pose = robot.fk(q, &Isometry3::identity()).ee_tfm();
    pose.transform_vector(&AXIS_IN_TIP)
        .dot(&AXIS_TARGET)
        .clamp(-1.0, 1.0)
        .acos()
}

fn tip_position(robot: &Robot, q: &[f64]) -> Vector3<f64> {
    robot
        .fk(q, &Isometry3::identity())
        .ee_tfm()
        .translation
        .vector
}

/// Random configurations, split by whether they already satisfy the constraint.
fn sample(robot: &Robot, max_angle: f64, want: usize, compliant: bool) -> Vec<Vec<f64>> {
    let (lb, ub) = robot.joint_limits();
    let mut rng = StdRng::seed_from_u64(7);
    let mut out = Vec::new();
    for _ in 0..200_000 {
        if out.len() == want {
            break;
        }
        let q: Vec<f64> = (0..lb.len()).map(|i| rng.gen_range(lb[i]..ub[i])).collect();
        if (cone_angle(robot, &q) <= max_angle) == compliant {
            out.push(q);
        }
    }
    out
}

fn project(robot: &Robot, max_angle: f64, q: &[f64]) -> Option<Vec<f64>> {
    robot.apply_angle_between_two_vectors_constraint(
        UnitVector3::new_normalize(AXIS_IN_TIP),
        UnitVector3::new_normalize(AXIS_TARGET),
        max_angle,
        Isometry3::identity(),
        q.to_vec(),
        &SolverConfig::default(),
    )
}

/// A seed already inside the cone is returned untouched.
#[test]
fn test_compliant_seed_is_returned_unchanged() {
    let robot = robot();
    let max_angle = 0.5;
    let seeds = sample(&robot, max_angle, 20, true);
    assert!(!seeds.is_empty(), "no compliant seeds sampled");

    for q in &seeds {
        let projected = project(&robot, max_angle, q).expect("compliant seed must project");
        assert_eq!(&projected, q);
    }
}

/// Whatever the projection returns satisfies the constraint it was given.
#[test]
fn test_projection_satisfies_the_constraint() {
    let robot = robot();
    for max_angle in [0.1, 0.25, 0.5, 1.0] {
        let seeds = sample(&robot, max_angle, 25, false);
        assert!(!seeds.is_empty(), "no violating seeds sampled");

        for q in &seeds {
            if let Some(projected) = project(&robot, max_angle, q) {
                assert!(
                    cone_angle(&robot, &projected) <= max_angle + 1e-6,
                    "projected configuration is outside the cone: {} > {}",
                    cone_angle(&robot, &projected),
                    max_angle
                );
            }
        }
    }
}

/// Regression guard for the pivot: the constraint is on the tool's *direction*,
/// so correcting it must not move the tool. Rotating about the world origin
/// instead of the tip satisfies the constraint but displaces the tip by
/// hundreds of millimetres, which is a pose the caller never asked to reach.
#[test]
fn test_projection_preserves_tip_position() {
    let robot = robot();
    let max_angle = 0.25;
    let seeds = sample(&robot, max_angle, 25, false);
    assert!(!seeds.is_empty(), "no violating seeds sampled");

    let mut checked = 0;
    for q in &seeds {
        if let Some(projected) = project(&robot, max_angle, q) {
            let before = tip_position(&robot, q);
            let after = tip_position(&robot, &projected);
            assert_abs_diff_eq!((after - before).norm(), 0.0, epsilon = 1e-3);
            checked += 1;
        }
    }
    assert!(checked > 0, "no projection succeeded, nothing was asserted");
}
