use nalgebra::Isometry3;
use optik::Robot;
use serde_json::json;

fn main() {
    let robot = Robot::from_urdf_str(
        include_str!("../tests/data/fixed_origin_order.urdf"),
        "world",
        "tip",
    );
    let samples: Vec<_> = (0..=180)
        .map(|index| {
            let angle_rad = -std::f64::consts::PI + index as f64 * std::f64::consts::PI / 90.0;
            let position_mm = robot
                .fk(&[angle_rad], &Isometry3::identity())
                .ee_tfm()
                .translation
                .vector
                * 1000.0;
            // Rx(90) * Rz(90) * Rz(q) * Tx(100 mm), evaluated independently.
            json!({
                "angle_rad": angle_rad,
                "actual_mm": [position_mm.x, position_mm.y, position_mm.z],
                "expected_mm": [-100.0 * angle_rad.sin(), 0.0, 100.0 * angle_rad.cos()],
            })
        })
        .collect();
    println!(
        "{}",
        json!({"optik_version": env!("CARGO_PKG_VERSION"), "samples": samples})
    );
}
