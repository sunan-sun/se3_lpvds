import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from src.se3_class import se3_class
from src.util import plot_tools, load_tools, process_tools



'''Load data'''
p_raw, q_raw, t_raw, dt = load_tools.load_npy()
# p_raw, q_raw, t_raw = load_tools.load_clfd_dataset(task_id=0, num_traj=9, sub_sample=1)
# p_raw, q_raw, t_raw = load_tools.load_demo_dataset()


'''Process data'''
p_in, q_in, t_in             = process_tools.pre_process(p_raw, q_raw, t_raw, opt= "savgol")
p_out, q_out                 = process_tools.compute_output(p_in, q_in, t_in)
p_init, q_init, p_att, q_att = process_tools.extract_state(p_in, q_in)
p_in, q_in, p_out, q_out     = process_tools.rollout_list(p_in, q_in, p_out, q_out)



'''Run lpvds'''
se3_obj = se3_class(p_in, q_in, p_out, q_out, p_att, q_att, dt, K_init=4)
se3_obj.begin()



'''Evaluate results'''
p_init = p_init[0]
q_init = R.from_quat(-q_init[0].as_quat())
p_test, q_test, gamma_pos, gamma_ori, v_test, w_test = se3_obj.sim(p_init, q_init, step_size=0.05)



'''Plot results'''
# plot_tools.plot_p_out(p_in, p_out, se3_obj.pos_ds)
# plot_tools.plot_q_out(q_in, q_out, se3_obj.ori_ds)

# plot_tools.plot_gmm_pos(p_in, se3_obj.pos_ds.damm)
# plot_tools.plot_gmm_ori(p_in, se3_obj.ori_ds.gmm)

plot_tools.plot_result(p_in, p_test, q_test)

plot_tools.plot_gamma(gamma_pos.T, title="pos")
plot_tools.plot_gamma(gamma_ori.T, title="ori")

plt.show()
