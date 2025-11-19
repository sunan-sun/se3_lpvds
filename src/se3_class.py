import os, sys, json, joblib
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
        # self.K_init = K_init
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



    def logOut(self, write_json=False, output_path=None):
        ds_linear_dict  = self.pos_ds._logOut(write_json, self.output_path)
        ds_angular_dict = self.ori_ds._logOut(write_json, self.output_path)

        return ds_linear_dict, ds_angular_dict


    def begin(self):
        self._cluster()
        self._optimize()
        # self._logOut()



    def sim(self, p_init, q_init, p_att, q_att, step_size, duration):
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

            p_next, q_next, gamma_pos, gamma_ori, v, w = self.step(p_in, q_in, step_size)

            p_test.append(p_next)        
            q_test.append(q_next)        
            gamma_pos_list.append(gamma_pos[:, 0])
            gamma_ori_list.append(gamma_ori[:, 0])
            v_test.append(v)
            w_test.append(w)

            i += 1
        if np.linalg.norm((q_test[-1] * q_att.inv()).as_rotvec()) <= self.tol and np.linalg.norm((p_test[-1] - p_att)) <= self.tol:
            print("Converged within max iteration")
        else:
            print("Did not converge within max iteration")
            print("Pos norm: ", np.linalg.norm((p_test[-1] - p_att)))
            print("Ori norm: ", np.linalg.norm((q_test[-1] * q_att.inv()).as_rotvec()))


        return np.vstack(p_test), q_test, np.array(gamma_pos_list), np.array(gamma_ori_list), v_test, w_test
        


    def step(self, p_in, q_in, step_size):
        """ Integrate forward by one time step """
        p_in = p_in.reshape(1, -1)
        
        p_next, gamma_pos, v = self.pos_ds._step(p_in, step_size)
        q_next, gamma_ori, w = self.ori_ds._step(q_in, step_size)

        return p_next, q_next, gamma_pos, gamma_ori, v, w


    def save(self, path):
        """
        Save the entire class instance to a file using joblib.
        Example: self.save("models/se3_model.pkl")
        """
        # ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[INFO] Model saved successfully to {path}")

    @classmethod
    def load(cls, path):
        """
        Load a saved class instance from a file.
        Example: se3 = se3_class.load("models/se3_model.pkl")
        """
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")
        print(f"[INFO] Model loaded successfully from {path}")
        return obj


    def __getstate__(self):
        """Prepare the object state for pickling.

        Now includes the processed dynamical system objects so that clustering
        and optimization don't need to be re-run on deserialization.
        """
        state = self.__dict__.copy()
        
        # Check if the dynamical systems have been processed
        pos_ds_processed = hasattr(self.pos_ds, 'gamma') and hasattr(self.pos_ds, 'A')
        ori_ds_processed = hasattr(self.ori_ds, 'gamma') and hasattr(self.ori_ds, 'A_ori')
        
        # Store processing state flags
        state['_pos_ds_processed'] = pos_ds_processed
        state['_ori_ds_processed'] = ori_ds_processed
        
        # If processed, extract and store the key processed attributes
        if pos_ds_processed:
            state['_pos_ds_state'] = {
                'gamma': getattr(self.pos_ds, 'gamma', None),
                'assignment_arr': getattr(self.pos_ds, 'assignment_arr', None),
                'K': getattr(self.pos_ds, 'K', None),
                'A': getattr(self.pos_ds, 'A', None),
                'damm': getattr(self.pos_ds, 'damm', None),
                'ds_opt': getattr(self.pos_ds, 'ds_opt', None)
            }
        
        if ori_ds_processed:
            state['_ori_ds_state'] = {
                'gamma': getattr(self.ori_ds, 'gamma', None),
                'K': getattr(self.ori_ds, 'K', None),
                'gmm': getattr(self.ori_ds, 'gmm', None),
                'A_ori': getattr(self.ori_ds, 'A_ori', None)
            }
        
        # Still remove the original objects as they may contain unpicklable elements
        state.pop("pos_ds", None)
        state.pop("ori_ds", None)
        
        return state

    def __setstate__(self, state):
        """Restore the object from the pickled state.

        If the dynamical systems were already processed before pickling, we restore
        them with their processed state to avoid re-running expensive clustering
        and optimization operations.
        """
        # Extract processing flags and states before updating dict
        pos_ds_processed = state.pop('_pos_ds_processed', False)
        ori_ds_processed = state.pop('_ori_ds_processed', False)
        pos_ds_state = state.pop('_pos_ds_state', None)
        ori_ds_state = state.pop('_ori_ds_state', None)
        
        # Update the object state
        self.__dict__.update(state)
        
        # Re-create the LPVDS (position) and quaternion DS (orientation)
        self.pos_ds = lpvds_class(self.p_in, self.p_out, self.p_att)
        self.ori_ds = quat_class(self.q_in, self.q_out, self.q_att, self.dt, self.K_init)
        
        # If the objects were processed before pickling, restore their processed state
        if pos_ds_processed and pos_ds_state:
            try:
                # Restore the processed attributes directly
                for attr_name, attr_value in pos_ds_state.items():
                    if attr_value is not None:
                        setattr(self.pos_ds, attr_name, attr_value)
            except Exception as e:
                print(f"Warning: Failed to restore pos_ds processed state: {e}")
                # Fallback to re-processing
                try:
                    self.pos_ds._cluster()
                    self.pos_ds._optimize()
                except Exception:
                    pass
        else:
            # Run clustering and optimization if not previously processed
            try:
                self.pos_ds._cluster()
                self.pos_ds._optimize()
            except Exception:
                pass
        
        if ori_ds_processed and ori_ds_state:
            try:
                # Restore the processed attributes directly
                for attr_name, attr_value in ori_ds_state.items():
                    if attr_value is not None:
                        setattr(self.ori_ds, attr_name, attr_value)
            except Exception as e:
                print(f"Warning: Failed to restore ori_ds processed state: {e}")
                # Fallback to re-processing
                try:
                    self.ori_ds._cluster()
                    self.ori_ds._optimize()
                except Exception:
                    pass
        else:
            # Run clustering and optimization if not previously processed
            try:
                self.ori_ds._cluster()
                self.ori_ds._optimize()
            except Exception:
                pass

    def is_processed(self):
        """Check if both dynamical systems have been clustered and optimized."""
        pos_processed = hasattr(self.pos_ds, 'gamma') and hasattr(self.pos_ds, 'A')
        ori_processed = hasattr(self.ori_ds, 'gamma') and hasattr(self.ori_ds, 'A_ori')
        return pos_processed and ori_processed