use approx::assert_relative_eq;
use nalgebra::Isometry3;
use optik::kinematics::KinematicChain;

#[test]
fn test_urdf_mjcf_parity() {
    // Create a simple URDF model
    let urdf = r#"
        <?xml version="1.0"?>
        <robot name="test_robot">
            <link name="base_link"/>
            <link name="link1"/>
            <link name="link2"/>
            <link name="ee_link"/>
            
            <joint name="joint1" type="revolute">
                <parent link="base_link"/>
                <child link="link1"/>
                <axis xyz="0 0 1"/>
                <origin xyz="1 0 0" rpy="0 0 0"/>
                <limit lower="-3.14" upper="3.14" velocity="1.0"/>
            </joint>
            
            <joint name="joint2" type="revolute">
                <parent link="link1"/>
                <child link="link2"/>
                <axis xyz="0 1 0"/>
                <origin xyz="1 0 0" rpy="0 0 0"/>
                <limit lower="-3.14" upper="3.14" velocity="1.0"/>
            </joint>
            
            <joint name="joint3" type="fixed">
                <parent link="link2"/>
                <child link="ee_link"/>
                <origin xyz="0.5 0 0" rpy="0 0 0"/>
            </joint>
        </robot>
    "#;

    // Create equivalent MJCF model
    let mjcf = r#"
        <mujoco>
            <worldbody>
                <body name="base_link">
                    <body name="link1" pos="1 0 0">
                        <joint name="joint1" type="hinge" axis="0 0 1" limited="true" range="-3.14 3.14"/>
                        <body name="link2" pos="1 0 0">
                            <joint name="joint2" type="hinge" axis="0 1 0" limited="true" range="-3.14 3.14"/>
                            <body name="ee_link" pos="0.5 0 0">
                                <joint name="joint3" type="fixed"/>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>
        </mujoco>
    "#;

    // Parse both models
    let urdf_robot = urdf_rs::read_from_string(urdf).unwrap();
    let mjcf_model = mjcf::from_str(mjcf).unwrap();

    let urdf_chain = KinematicChain::from_urdf(&urdf_robot, "base_link", "ee_link");
    let mjcf_chain = KinematicChain::from_mjcf(&mjcf_model, "base_link", "ee_link");

    // Verify chain properties
    assert_eq!(urdf_chain.num_positions(), mjcf_chain.num_positions());
    assert_eq!(urdf_chain.joints.len(), mjcf_chain.joints.len());

    // Verify joint properties
    for (urdf_joint, mjcf_joint) in urdf_chain.joints.iter().zip(mjcf_chain.joints.iter()) {
        assert_eq!(urdf_joint.name, mjcf_joint.name);
        assert_eq!(urdf_joint.typ, mjcf_joint.typ);
        assert_eq!(urdf_joint.limits, mjcf_joint.limits);
        assert_relative_eq!(urdf_joint.origin, mjcf_joint.origin);
    }

    // Test forward kinematics
    let q = vec![0.5, 0.3];
    let ee_offset = Isometry3::identity();
    
    let urdf_fk = urdf_chain.forward_kinematics(&q, &ee_offset);
    let mjcf_fk = mjcf_chain.forward_kinematics(&q, &ee_offset);
    
    assert_relative_eq!(urdf_fk.ee_tfm(), mjcf_fk.ee_tfm());
}

#[test]
fn test_urdf_mjcf_parity_with_prismatic() {
    // Create a URDF model with prismatic joints
    let urdf = r#"
        <?xml version="1.0"?>
        <robot name="test_robot">
            <link name="base_link"/>
            <link name="link1"/>
            <link name="link2"/>
            <link name="ee_link"/>
            
            <joint name="joint1" type="prismatic">
                <parent link="base_link"/>
                <child link="link1"/>
                <axis xyz="1 0 0"/>
                <origin xyz="0 0 0" rpy="0 0 0"/>
                <limit lower="-1" upper="1" velocity="1.0"/>
            </joint>
            
            <joint name="joint2" type="prismatic">
                <parent link="link1"/>
                <child link="link2"/>
                <axis xyz="0 1 0"/>
                <origin xyz="0 0 0" rpy="0 0 0"/>
                <limit lower="-1" upper="1" velocity="1.0"/>
            </joint>
            
            <joint name="joint3" type="fixed">
                <parent link="link2"/>
                <child link="ee_link"/>
                <origin xyz="0.5 0 0" rpy="0 0 0"/>
            </joint>
        </robot>
    "#;

    // Create equivalent MJCF model
    let mjcf = r#"
        <mujoco>
            <worldbody>
                <body name="base_link">
                    <body name="link1">
                        <joint name="joint1" type="slide" axis="1 0 0" limited="true" range="-1 1"/>
                        <body name="link2">
                            <joint name="joint2" type="slide" axis="0 1 0" limited="true" range="-1 1"/>
                            <body name="ee_link" pos="0.5 0 0">
                                <joint name="joint3" type="fixed"/>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>
        </mujoco>
    "#;

    // Parse both models
    let urdf_robot = urdf_rs::read_from_string(urdf).unwrap();
    let mjcf_model = mjcf::from_str(mjcf).unwrap();

    let urdf_chain = KinematicChain::from_urdf(&urdf_robot, "base_link", "ee_link");
    let mjcf_chain = KinematicChain::from_mjcf(&mjcf_model, "base_link", "ee_link");

    // Verify chain properties
    assert_eq!(urdf_chain.num_positions(), mjcf_chain.num_positions());
    assert_eq!(urdf_chain.joints.len(), mjcf_chain.joints.len());

    // Verify joint properties
    for (urdf_joint, mjcf_joint) in urdf_chain.joints.iter().zip(mjcf_chain.joints.iter()) {
        assert_eq!(urdf_joint.name, mjcf_joint.name);
        assert_eq!(urdf_joint.typ, mjcf_joint.typ);
        assert_eq!(urdf_joint.limits, mjcf_joint.limits);
        assert_relative_eq!(urdf_joint.origin, mjcf_joint.origin);
    }

    // Test forward kinematics
    let q = vec![0.5, 0.3];
    let ee_offset = Isometry3::identity();
    
    let urdf_fk = urdf_chain.forward_kinematics(&q, &ee_offset);
    let mjcf_fk = mjcf_chain.forward_kinematics(&q, &ee_offset);
    
    assert_relative_eq!(urdf_fk.ee_tfm(), mjcf_fk.ee_tfm());
} 