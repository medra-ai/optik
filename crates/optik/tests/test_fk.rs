use approx::assert_abs_diff_eq;
use nalgebra::Isometry3;
use optik::Robot;

const TEST_URDF_STR: &str = include_str!("data/ur3e.urdf");
const TEST_MJCF_STR: &str = include_str!("data/ur3e.mjcf");

macro_rules! fetch_data {
    ($path: literal) => {
        serde_json::from_str(include_str!($path)).unwrap()
    };
}

fn create_robot_urdf() -> Robot {
    Robot::from_urdf_str(TEST_URDF_STR, "ur_base_link", "ur_ee_link")
}

fn create_robot_mjcf() -> Robot {
    Robot::from_mjcf_str(TEST_MJCF_STR, "ur_base_link", "ur_ee_link")
}

fn test_fk_impl(create_robot: fn() -> Robot) {
    let inputs: Vec<Vec<f64>> = fetch_data!("data/test_fk_inputs.json");
    let outputs: Vec<Isometry3<f64>> = fetch_data!("data/test_fk_outputs.json");

    let robot = create_robot();
    for (input, output) in inputs.into_iter().zip(outputs) {
        assert_abs_diff_eq!(
            robot.fk(&input, &Isometry3::identity()).ee_tfm(),
            output,
            epsilon = 1e-6
        );
    }
}

#[test]
fn test_fk_urdf() {
    test_fk_impl(create_robot_urdf);
}

#[test]
fn test_fk_mjcf() {
    test_fk_impl(create_robot_mjcf);
}
