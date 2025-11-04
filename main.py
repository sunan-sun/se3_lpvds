import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from src.se3_class import se3_class
from src.util import load_tools, process_tools



'''Load data'''
T = 5
# p_raw, q_raw, t_raw, dt = load_tools.load_npy(duration=T)
p_raw, q_raw, t_raw, dt = load_tools.load_clfd_dataset(task_id=1, num_traj=2, sub_sample=1, duration=T)
# p_raw, q_raw, t_raw, dt= load_tools.load_demo_dataset()
# T = t_raw[0][-1] - t_raw[0][0]

'''Process data'''
p_in, q_in, t_in             = process_tools.pre_process(p_raw, q_raw, t_raw, opt= "savgol")
p_out, q_out                 = process_tools.compute_output(p_in, q_in, t_in)
p_init, q_init, p_att, q_att = process_tools.extract_state(p_in, q_in)
p_in, q_in, p_out, q_out     = process_tools.rollout_list(p_in, q_in, p_out, q_out)



'''Run lpvds'''
se3_obj = se3_class(p_in, q_in, p_out, q_out, p_att, q_att, dt, K_init=4)
se3_obj.begin()



'''Evaluate results'''
p_test_list = []
q_test_list = []
for p_0, q_0 in zip(p_init, q_init):
    q_0 = R.from_quat(q_0.as_quat())
    p_test, q_test, gamma_pos, gamma_ori, v_test, w_test = se3_obj.sim(p_0, q_0, p_att, q_att, step_size=dt, duration=T)
    p_test_list.append(p_test)
        


'''Plot results'''
from src.lpvds.src.damm.src.util.plot_tools import plot_gmm
plot_gmm(p_in, se3_obj.pos_ds.damm.z, se3_obj.pos_ds.damm)

from src.lpvds.src.util.plot_tools import plot_ds, plot_gamma
plot_ds(p_in, p_test_list, se3_obj.pos_ds)
plot_gamma(gamma_pos, title="pos")
plot_gamma(gamma_ori, title="ori")

from src.util import plot_tools
plot_tools.plot_result(p_in, p_test, q_test)

plt.show()
