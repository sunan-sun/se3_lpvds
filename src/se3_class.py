import os, sys, json
import numpy as np
from scipy.spatial.transform import Rotation as R

from .util import quat_tools, plot_tools
from .lpvds.src.lpvds_class import lpvds_class
from .quaternion_ds.src.quat_class import quat_class




class se3_class:
    def __init__(self, p_in:np.ndarray, q_in:list, p_out:np.ndarray, q_out:list, p_att:np.ndarray, q_att:R, dt:float, K_init:int) -> None:
        """
        Parameters:
        ----------
            p_in (np.ndarray):      [M, N] NumPy array of POSITION INPUT

            q_in (list):            M-length List of Rotation objects for ORIENTATION INPUT

            p_out (np.ndarray):     [M, N] NumPy array of POSITION OUTPUT

            q_out (list):           M-length List of Rotation objects for ORIENTATION OUTPUT

            p_att (np.ndarray):     [1, N] NumPy array of POSITION ATTRACTOR

            q_att (Rotation):       Single Rotation object for ORIENTATION ATTRACTOR
            
            dt (float):             TIME DIFFERENCE in differentiating ORIENTATION

            K_init (int):           Number of Gaussian Components

            M:                      Observation size

            N:                      Observation dimenstion (assuming 3D)
        """

        # store parameters
        self.p_in  = p_in
        self.q_in  = q_in

        self.p_out = p_out
        self.q_out = q_out

        self.p_att = p_att
        self.q_att = q_att

        self.dt = dt
        self.K_init = K_init
        self.M = len(q_in)


        # simulation parameters
        self.tol = 10E-3
        self.max_iter = 5000


        # define output path
        file_path           = os.path.dirname(os.path.realpath(__file__))  
        self.output_path    = os.path.dirname(file_path)


        # initialize lpvds class
        self.pos_ds = lpvds_class(p_in, p_out, p_att)
        self.ori_ds = quat_class(q_in, q_out, q_att, dt, K_init)



    def _cluster(self):
        self.pos_ds._cluster()
        self.ori_ds._cluster()



    def _optimize(self):
        self.pos_ds._optimize()
        self.ori_ds._optimize()



    def _logOut(self):
        self.ori_ds._logOut(self.output_path)
        self.pos_ds._logOut(self.output_path)



    def begin(self):
        self._cluster()
        self._optimize()
        self._logOut()



    def sim(self, p_init, q_init, step_size, duration):
        p_test = [p_init.reshape(1, -1)]
        q_test = [q_init]

        gamma_pos_list = []
        gamma_ori_list = []

        v_test = []
        w_test = []

        i = 0
        max_iter = int(duration / step_size) * 1.5
        while i < max_iter:
            
            p_in  = p_test[i]
            q_in  = q_test[i]

            p_next, q_next, gamma_pos, gamma_ori, v, w = self._step(p_in, q_in, step_size)

            p_test.append(p_next)        
            q_test.append(q_next)        
            gamma_pos_list.append(gamma_pos[:, 0])
            gamma_ori_list.append(gamma_ori[:, 0])
            v_test.append(v)
            w_test.append(w)

            i += 1
        if np.linalg.norm((q_test[-1] * self.q_att.inv()).as_rotvec()) <= self.tol and np.linalg.norm((p_test[-1] - self.p_att)) <= self.tol:
            print("Converged within max iteration")
        else:
            print("Did not converge within max iteration")


        return np.vstack(p_test), q_test, np.array(gamma_pos_list), np.array(gamma_ori_list), v_test, w_test
        


    def _step(self, p_in, q_in, step_size):
        """ Integrate forward by one time step """
        p_in = p_in.reshape(1, -1)
        
        p_next, gamma_pos, v = self.pos_ds._step(p_in, step_size)
        q_next, gamma_ori, w = self.ori_ds._step(q_in, step_size)

        return p_next, q_next, gamma_pos, gamma_ori, v, w
