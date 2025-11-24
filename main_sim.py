import pybullet as p
import pybullet_data
import numpy as np
from typing import List, Tuple
from src.se3_class import se3_class
from src.util import load_tools, process_tools
from scipy.spatial.transform import Rotation as R


def setup_bullet(use_gui: bool, timestep: float = 1.0 / 240.0):

    cid = p.connect(p.GUI if use_gui else p.DIRECT)
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(timestep)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf")
    # Enable real-time simulation and physics
    p.setRealTimeSimulation(1)
    
    # Enable default constraint solver
    p.setPhysicsEngineParameter(numSolverIterations=50)
    
    # Enable collision detection
    p.setPhysicsEngineParameter(enableFileCaching=0,
                               contactERP=0.9,
                               contactBreakingThreshold=0.01)
    return cid, plane


def load_panda(base_pos=(0, 0, 0), base_orn=(0, 0, 0, 1)) -> int:

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE
    robot = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=base_pos,
        baseOrientation=base_orn,
        useFixedBase=True,
        flags=flags,
    )
    
    # Enable collision detection between all links
    for i in range(p.getNumJoints(robot)):
        p.setCollisionFilterPair(robot, robot, i, i, 1)
    
    return robot


def get_joint_info(robot: int) -> Tuple[List[int], List[float], List[float]]:

    arm_joint_indices = []
    q_min = []
    q_max = []
    n = p.getNumJoints(robot)
    for j in range(n):
        ji = p.getJointInfo(robot, j)
        jtype = ji[2]
        jname = ji[1].decode()
        if jtype == p.JOINT_REVOLUTE and jname.startswith("panda_joint") and len(arm_joint_indices) < 7:
            arm_joint_indices.append(j)
            q_min.append(ji[8])
            q_max.append(ji[9])
    if len(arm_joint_indices) != 7:
        raise RuntimeError("Failed to find 7 Panda arm joints.")
    return arm_joint_indices, q_min, q_max


def find_ee_link(robot: int) -> int:

    ee = None
    n = p.getNumJoints(robot)
    for j in range(n):
        ji = p.getJointInfo(robot, j)
        lname = ji[12].decode()
        if lname == "panda_hand":
            ee = j
            break
    if ee is None:
        ee = n - 1  # fallback to last link
    return ee


def get_dof_order(robot: int) -> Tuple[List[int], dict]:

    dof_joint_indices = []
    n = p.getNumJoints(robot)
    for j in range(n):
        jtype = p.getJointInfo(robot, j)[2]
        if jtype != p.JOINT_FIXED:
            dof_joint_indices.append(j)
    # Map from joint index -> position in DoF vector
    dof_index_of_joint = {j: i for i, j in enumerate(dof_joint_indices)}
    return dof_joint_indices, dof_index_of_joint


def calculate_jacobian(robot: int, ee_link: int, q_dof: List[float]) -> np.ndarray:

    # PyBullet expects vectors of size numDoF (non-fixed joints)
    zero = [0.0] * len(q_dof)
    jac_t, jac_r = p.calculateJacobian(robot, ee_link, [0, 0, 0], q_dof, zero, zero)
    Jv = np.array(jac_t)
    Jw = np.array(jac_r)
    J = np.vstack([Jv, Jw])  # 6 x nJoints
    return J


def set_arm_positions(robot: int, arm_joint_indices: List[int], q_arm: np.ndarray, kp=1.0, kd=0.1):

    # Reduced gains for smoother motion
    p.setJointMotorControlArray(
        robot,
        arm_joint_indices,
        controlMode=p.POSITION_CONTROL,
        targetPositions=q_arm,
        positionGains=[kp] * len(arm_joint_indices),
        velocityGains=[kd] * len(arm_joint_indices),
        forces=[50.0] * len(arm_joint_indices),  # Reduced forces to allow natural dynamics
    )


def reset_arm(robot: int, arm_joint_indices: List[int], q_arm: np.ndarray):

    for idx, q in zip(arm_joint_indices, q_arm):
        p.resetJointState(robot, idx, float(q))



def compute_ik(robot, ee_link, arm_joint_indices, dof_joint_indices, p_target, q_target, current_q=None):
    # Set IK parameters
    max_iters = 100
    residual_threshold = 1e-4
    
    # Use current joint angles as initial guess if provided
    if current_q is None:
        current_q = [0.0] * len(dof_joint_indices)
    else:
        current_q.extend([0.0, 0.0])

    ik_sol = p.calculateInverseKinematics(
        bodyUniqueId=robot,
        endEffectorLinkIndex=ee_link,
        targetPosition=p_target.tolist(),
        targetOrientation=tuple(q_target.as_quat().tolist()),
        currentPositions=current_q,
        maxNumIterations=max_iters,
        residualThreshold=residual_threshold
    )

    q = [ik_sol[d] for d in [dof_joint_indices.index(j) for j in arm_joint_indices]]
    
    # Validate IK solution
    # reset_arm(robot, arm_joint_indices, q)
    # ee_state = p.getLinkState(robot, ee_link, computeForwardKinematics=True)
    # actual_pos = np.array(ee_state[4])
    # actual_orn = np.array(ee_state[5])
    
    # pos_error = np.linalg.norm(actual_pos - p_target)
    # orn_error = np.linalg.norm(actual_orn - q_target.as_quat())
    
    # if pos_error > 0.01 or orn_error > 0.1:
    #     print(f"Warning: Large IK error - pos: {pos_error:.4f}, orn: {orn_error:.4f}")
        
    return q


def draw_waypoints(points: np.ndarray, point_color=(0.9, 0.3, 0.1), line_color=(1, 0, 0), point_size: float = 5.0, line_width: float = 2.0, lifeTime: float = 0.0):
    """Draw waypoints as debug points plus connecting lines in PyBullet.
    - points: (N, 3) array-like in world coordinates
    - point_color/line_color: RGB in [0,1]
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    p.addUserDebugPoints(pointPositions=pts.tolist(), pointColorsRGB=[point_color] * len(pts), pointSize=point_size, lifeTime=lifeTime)



def main():
    
    """Initialize PyBullet"""
    cid, plane = setup_bullet(use_gui=True, timestep=1/240)
    

    """Load Panda and helper functions"""
    robot = load_panda()
    arm_joint_indices, qmin, qmax = get_joint_info(robot)
    ee_link = find_ee_link(robot)
    dof_joint_indices, dof_index_of_joint = get_dof_order(robot)
    arm_dof_cols = [dof_index_of_joint[j] for j in arm_joint_indices]


    """Learn and load DS"""
    p_raw, q_raw, t_raw, dt= load_tools.load_demo_dataset()
    T = t_raw[0][-1] - t_raw[0][0]
    p_in, q_in, t_in             = process_tools.pre_process(p_raw, q_raw, t_raw, opt= "savgol")
    p_out, q_out                 = process_tools.compute_output(p_in, q_in, t_in)
    p_init, q_init, p_att, q_att = process_tools.extract_state(p_in, q_in)
    p_in, q_in, p_out, q_out     = process_tools.rollout_list(p_in, q_in, p_out, q_out)
    se3_obj = se3_class(p_in, q_in, p_out, q_out, p_att, q_att, dt, K_init=4)
    se3_obj.begin()
    p_test_list = []
    for p_0, q_0 in zip(p_init, q_init):
        q_0 = R.from_quat(q_0.as_quat())
        p_test, q_test, gamma_pos, gamma_ori, v_test, w_test = se3_obj.sim(p_0, q_0, p_att, q_att, step_size=dt, duration=T)
        p_test_list.append(p_test)
    draw_waypoints(np.asarray(p_test_list[0]), point_color=(1, 0, 0), line_color=(1, 0, 0), point_size=5.0, line_width=2.0, lifeTime=0)


    """Move Panda to initial pose"""
    p_0 = p_init[0]
    q_0 = q_init[0]
    q_home = compute_ik(robot, ee_link, arm_joint_indices, dof_joint_indices, p_0, q_0)
    reset_arm(robot, arm_joint_indices, q_home)


    """Run the DS in simulation"""
    p_test = [p_0.reshape(1, -1)]
    q_test = [q_0]
    tol = 0.03
    i = 0
    current_q = None  # Initialize outside loop
    while np.linalg.norm((q_test[-1] * q_att.inv()).as_rotvec()) >= tol or np.linalg.norm((p_test[-1] - p_att)) >= tol:
        
        p_in  = p_test[-1]
        q_in  = q_test[-1]

        p_next, q_next, _, _, _, _ = se3_obj._step(p_in, q_in, se3_obj.dt)

        p_test.append(p_next)
        q_test.append(q_next)
    
        q = compute_ik(robot, ee_link, arm_joint_indices, dof_joint_indices, 
                      p_next[0], q_next, current_q=current_q)
        set_arm_positions(robot, arm_joint_indices, q)
        current_q = q  
        i += 1

        p.stepSimulation()

    p.disconnect()


if __name__ == "__main__":
    main()