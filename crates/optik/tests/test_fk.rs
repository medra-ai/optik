use approx::assert_abs_diff_eq;
use nalgebra::{Isometry3, Translation3, UnitQuaternion};
use optik::Robot;

const TEST_MODEL_STR: &str = include_str!("data/ur3e.urdf");

macro_rules! fetch_data {
    ($path: literal) => {
        serde_json::from_str(include_str!($path)).unwrap()
    };
}

#[test]
fn test_fk() {
    let inputs: Vec<Vec<f64>> = fetch_data!("data/test_fk_inputs.json");
    let outputs: Vec<Isometry3<f64>> = fetch_data!("data/test_fk_outputs.json");

    let robot = Robot::from_urdf_str(TEST_MODEL_STR, "ur_base_link", "ur_ee_link");
    for (input, output) in inputs.into_iter().zip(outputs) {
        assert_abs_diff_eq!(
            robot.fk(&input, &Isometry3::identity()).ee_tfm(),
            output,
            epsilon = 1e-6
        );
    }
}

#[test]
fn fixed_root_precedes_articulated_origin() {
    let robot = Robot::from_urdf_str(include_str!("data/fixed_origin_order.urdf"), "world", "tip");
    let mount = Isometry3::from_parts(
        Translation3::identity(),
        UnitQuaternion::from_euler_angles(std::f64::consts::FRAC_PI_2, 0.0, 0.0),
    );
    let origin = Isometry3::from_parts(
        Translation3::identity(),
        UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
    );

    for angle in [-0.7, 0.0, 0.4] {
        let motion = Isometry3::from_parts(
            Translation3::identity(),
            UnitQuaternion::from_euler_angles(0.0, 0.0, angle),
        );
        let expected = mount * origin * motion * Isometry3::translation(0.1, 0.0, 0.0);

        assert_abs_diff_eq!(
            robot.fk(&[angle], &Isometry3::identity()).ee_tfm(),
            expected,
            epsilon = 1e-12
        );
    }
}

#[test]
fn consecutive_fixed_joints_precede_nonidentity_articulated_origin() {
    let robot = Robot::from_urdf_str(
        r#"<robot name="fixed_middle">
          <link name="base"/><link name="a"/><link name="b"/><link name="c"/><link name="tip"/>
          <joint name="j1" type="revolute">
            <parent link="base"/><child link="a"/><axis xyz="0 0 1"/>
            <limit lower="-3" upper="3" effort="1" velocity="1"/>
          </joint>
          <joint name="turn" type="fixed">
            <parent link="a"/><child link="b"/><origin rpy="1.5707963267948966 0 0"/>
          </joint>
          <joint name="shift" type="fixed">
            <parent link="b"/><child link="c"/><origin xyz="0 0.2 0"/>
          </joint>
          <joint name="j2" type="revolute">
            <parent link="c"/><child link="tip"/><axis xyz="0 0 1"/>
            <origin xyz="0 0.1 0" rpy="0 0 1.5707963267948966"/>
            <limit lower="-3" upper="3" effort="1" velocity="1"/>
          </joint>
        </robot>"#,
        "base",
        "tip",
    );
    let half_pi = std::f64::consts::FRAC_PI_2;
    let expected = Isometry3::from_parts(
        Translation3::new(0.0, 0.0, 0.3),
        UnitQuaternion::from_euler_angles(half_pi, 0.0, half_pi),
    );

    let actual = robot
        .fk(&[half_pi, -half_pi], &Isometry3::identity())
        .ee_tfm();

    assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
}

#[test]
fn consecutive_terminal_fixed_joints_preserve_order() {
    let robot = Robot::from_urdf_str(
        r#"<robot name="fixed_tip">
          <link name="base"/><link name="moving"/><link name="turned"/><link name="tip"/>
          <joint name="j1" type="revolute">
            <parent link="base"/><child link="moving"/><axis xyz="0 0 1"/>
            <limit lower="-3" upper="3" effort="1" velocity="1"/>
          </joint>
          <joint name="turn" type="fixed">
            <parent link="moving"/><child link="turned"/>
            <origin rpy="0 0 1.5707963267948966"/>
          </joint>
          <joint name="shift" type="fixed">
            <parent link="turned"/><child link="tip"/><origin xyz="0.1 0 0"/>
          </joint>
        </robot>"#,
        "base",
        "tip",
    );
    let turn = UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2);
    let expected = Isometry3::from_parts(Translation3::new(0.0, 0.1, 0.0), turn);

    let actual = robot.fk(&[0.0], &Isometry3::identity()).ee_tfm();

    assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
}
