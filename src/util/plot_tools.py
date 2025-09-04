import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial.transform import Rotation as R
# from .quat_tools import *
import random


# font = {'family' : 'Times New Roman',
#          'size'   : 18
#          }
# mpl.rc('font', **font)






def plot_vel(v_test, w_test):
    """Plots the linear and angular velocities over time.

    Args:
        v_test: List or array of linear velocities (M x 3).
        w_test: List or array of angular velocities (M x 3).
    """
    v_test = np.vstack(v_test)
    M, N = v_test.shape

    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    for k in range(3):
        axs[k].scatter(np.arange(M), v_test[:, k], s=5, color=colors[k])
        # axs[k].set_ylim([0, 1])
    
    axs[0].set_title("Linear Velocity")

    w_test = np.vstack(w_test)
    M, N = w_test.shape
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    for k in range(3):
        axs[k].scatter(np.arange(M), w_test[:, k], s=5, color=colors[k])
        # axs[k].set_ylim([0, 1])
    axs[0].set_title("Angular Velocity")



def plot_gamma(gamma_arr, **argv):
    """Plots the gamma values (activation weights) over time for each component.

    Args:
        gamma_arr: Array of gamma values (M x K), where M is the number of time steps
                   and K is the number of components.
        **argv: Additional keyword arguments. Can include "title" for the plot title.
    """

    K, M = gamma_arr.shape
    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
        "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
    
    if K == 1:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.scatter(np.arange(M), gamma_arr[0, :], s=5, color=colors[0])
        ax.set_ylim([0, 1])
        if "title" in argv:
            ax.set_title(argv["title"])
        else:
            ax.set_title(r"$\gamma(\cdot)$ over Time")
    else:
        fig, axs = plt.subplots(K, 1, figsize=(12, 8))

        for k in range(K):
            axs[k].scatter(np.arange(M), gamma_arr[k, :], s=5, color=colors[k])
            axs[k].set_ylim([0, 1])
        if "title" in argv:
            axs[0].set_title(argv["title"])
        else:
            axs[0].set_title(r"$\gamma(\cdot)$ over Time")




def plot_result(p_train, p_test, q_test):
    """Plots the 3D trajectory reproduction results.

    Displays the demonstrated trajectory, the reproduced trajectory,
    initial/target points, and orientation frames along the reproduced path.

    Args:
        p_train: Demonstrated trajectory positions (N x 3).
        p_test: Reproduced trajectory positions (M x 3).
        q_test: Reproduced trajectory orientations as SciPy Rotation objects (list or array of length M).
    """

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')


    ax.plot(p_train[:, 0], p_train[:, 1], p_train[:, 2], 'o', color='gray', alpha=0.2, markersize=1.5, label="Demonstration")
    ax.plot(p_test[:, 0], p_test[:, 1], p_test[:, 2],  color='k', label = "Reproduction")

    ax.scatter(p_test[0, 0], p_test[0, 1], p_test[0, 2], 'o', facecolors='none', edgecolors='magenta',linewidth=2,  s=100, label="Initial")
    ax.scatter(p_test[-1, 0], p_test[-1, 1], p_test[-1, 2], marker=(8, 2, 0), color='k',  s=100, label="Target")


    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    x_min, x_max = ax.get_xlim()
    scale = (x_max - x_min)/4
    
    for i in np.linspace(0, len(q_test), num=30, endpoint=False, dtype=int):

        r = q_test[i]
        loc = p_test[i, :]
        for j, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                            colors)):
            line = np.zeros((2, 3))
            line[1, j] = scale
            line_rot = r.apply(line)
            line_plot = line_rot + loc
            ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c, linewidth=1)


    ax.axis('equal')
    ax.legend(ncol=4, loc="upper center")

    ax.set_xlabel(r'$\xi_1$', labelpad=20)
    ax.set_ylabel(r'$\xi_2$', labelpad=20)
    ax.set_zlabel(r'$\xi_3$', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))





def plot_gmm(p_in, gmm):
    """Plots the GMM clustering results in 3D.

    Displays the input data points colored by their assigned cluster,
    and shows the mean orientation of each Gaussian component.

    Args:
        p_in: Input position data (N x 3).
        gmm: Fitted GMM object, expected to have attributes 'assignment_arr' (N,),
             'K' (number of components), and 'gaussian_list' (list of dicts,
             where each dict contains ["mu"][1] as the mean orientation).
    """

    label = gmm.assignment_arr
    K     = gmm.K

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    color_mapping = np.take(colors, label)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(p_in[:, 0], p_in[:, 1], p_in[:, 2], 'o', color=color_mapping[:], s=1, alpha=0.4, label="Demonstration")

    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB

    x_min, x_max = ax.get_xlim()
    scale = (x_max - x_min)/4
    for k in range(K):
        label_k =np.where(label == k)[0]

        p_in_k = p_in[label_k, :]
        loc = np.mean(p_in_k, axis=0)

        r = gmm.gaussian_list[k]["mu"]
        for j, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                            colors)):
            line = np.zeros((2, 3))
            line[1, j] = scale
            line_rot = r.apply(line)
            line_plot = line_rot + loc
            ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c, linewidth=1)


    ax.axis('equal')


    ax.set_xlabel(r'$\xi_1$', labelpad=20)
    ax.set_ylabel(r'$\xi_2$', labelpad=20)
    ax.set_zlabel(r'$\xi_3$', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))






def _plot_ellipsoid(ax, mean, cov, color, alpha=0.3, confidence=2.0):
    """Plot a 3D ellipsoid representing a Gaussian component.
    
    Args:
        ax: 3D matplotlib axis
        mean: Mean vector (3,)
        cov: Covariance matrix (3x3)
        color: Color for the ellipsoid
        alpha: Transparency
        confidence: Confidence level (standard deviations)
    """
    # Eigenvalue decomposition
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Sort by eigenvalue
    order = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]
    
    # Radii of the ellipsoid (scale by confidence level)
    radii = confidence * np.sqrt(eigenvals)
    
    # Generate a unit sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Scale by radii
    ellipsoid = np.stack([x_sphere * radii[0], 
                         y_sphere * radii[1], 
                         z_sphere * radii[2]], axis=-1)
    
    # Rotate by eigenvectors and translate by mean
    ellipsoid_rotated = np.dot(ellipsoid, eigenvecs.T) + mean
    
    # Plot the ellipsoid surface
    ax.plot_surface(ellipsoid_rotated[:, :, 0], 
                   ellipsoid_rotated[:, :, 1], 
                   ellipsoid_rotated[:, :, 2], 
                   color=color, alpha=alpha, linewidth=0.1)


def plot_gmm_pos(x, gmm, ax=None):
    """Plot clustered position data with Gaussian ellipsoids.

    Args:
        x (list[list[list|np.ndarray]]): Nested list where x[i][j] is a 3-D point.
        gmm: Object with attributes assignment_arr, K, gaussian_list.
    """

    # Flatten trajectories into a single point array and compute per-point alpha
    pts = []
    alphas = []
    for traj in x:
        n = len(traj)
        for idx, p in enumerate(traj):
            pts.append(p)
            alphas.append(0.2 + 0.8 * (idx / max(n - 1, 1)))
    p_in = np.asarray(pts)

    label = gmm.assignment_arr
    K     = gmm.K

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
        "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    # Map each point's label to a base color then add alpha
    import matplotlib.colors as mcolors
    rgba_colors = [mcolors.to_rgba(colors[lbl], alpha=alphas[i]) for i, lbl in enumerate(label)]

    # Create axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(projection='3d')

    ax.scatter(
        p_in[:, 0], p_in[:, 1], p_in[:, 2], 'o',
        color=rgba_colors, s=5, label="Demonstration", depthshade=False
    )

    # Draw a polyline for each trajectory to connect its points with increasing thickness
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    for traj in x:
        traj_np = np.asarray(traj)
        if traj_np.ndim == 2 and traj_np.shape[1] == 3 and traj_np.shape[0] > 1:
            # Build line segments between consecutive points
            segments = np.stack([traj_np[:-1], traj_np[1:]], axis=1)  # (n-1, 2, 3)
            num_segs = segments.shape[0]
            if num_segs <= 0:
                continue
            # Line width ramps from thin to thick along the trajectory
            min_w, max_w = 0.6, 2.5
            widths = np.linspace(min_w, max_w, num_segs)
            # Alpha ramps from 0.2 (start) to 1.0 (end) along the trajectory
            import matplotlib.colors as mcolors
            base_color = "#444444"
            seg_alphas = np.linspace(0.2, 1.0, num_segs)
            seg_colors = [mcolors.to_rgba(base_color, a) for a in seg_alphas]
            lc = Line3DCollection(segments, colors=seg_colors, linewidths=widths)
            ax.add_collection3d(lc)

    # Plot Gaussian ellipsoids
    for k in range(K):
        mean = gmm.gaussian_list[k]["mu"]
        cov = gmm.gaussian_list[k]["sigma"]
        
        # Plot ellipsoid for this Gaussian component
        _plot_ellipsoid(ax, mean, cov, colors[k], alpha=0.1)

    # --------------------------------------------------
    # Automatically determine axis limits to encompass
    # both data points and ellipsoids
    # --------------------------------------------------
    xyz_min = p_in.min(axis=0).copy()
    xyz_max = p_in.max(axis=0).copy()

    # Include ellipsoid extents
    for k in range(K):
        cov = gmm.gaussian_list[k]["sigma"]
        mean = gmm.gaussian_list[k]["mu"]
        eigenvals, _ = np.linalg.eigh(cov)
        radii = 2.0 * np.sqrt(eigenvals)  # confidence=2.0 as in _plot_ellipsoid
        xyz_min = np.minimum(xyz_min, mean - radii)
        xyz_max = np.maximum(xyz_max, mean + radii)

    padding = 0.05 * (xyz_max - xyz_min)
    ax.set_xlim(xyz_min[0] - padding[0], xyz_max[0] + padding[0])
    ax.set_ylim(xyz_min[1] - padding[1], xyz_max[1] + padding[1])
    ax.set_zlim(xyz_min[2] - padding[2], xyz_max[2] + padding[2])

    def _set_axes_equal(ax):
        """Set 3D plot axes to equal scale."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        span = max(x_limits[1]-x_limits[0], y_limits[1]-y_limits[0], z_limits[1]-z_limits[0])
        x_center = np.mean(x_limits)
        y_center = np.mean(y_limits)
        z_center = np.mean(z_limits)
        ax.set_xlim3d(x_center - span/2, x_center + span/2)
        ax.set_ylim3d(y_center - span/2, y_center + span/2)
        ax.set_zlim3d(z_center - span/2, z_center + span/2)
    _set_axes_equal(ax)

    ax.set_xlabel(r'$\xi_1$', labelpad=20)
    ax.set_ylabel(r'$\xi_2$', labelpad=20)
    ax.set_zlabel(r'$\xi_3$', labelpad=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


def plot_p_out(p_in, p_out, pos_obj):
    """compare estimated p_out given p_in with groud truth"""


    p_out_est     = np.zeros((p_out.shape)) # (M, N)

    gamma = pos_obj.damm.logProb(p_in)

    for k in range(pos_obj.K):
        p_out_est  +=  (np.tile(gamma[k, :], (3, 1)) * (pos_obj.A[k] @ (p_in - pos_obj.x_att.reshape(1, -1)).T)).T


    M, N = p_out.shape

    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    label = pos_obj.assignment_arr

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    color_mapping = np.take(colors, label)

    label_list =["v_x", "v_y", "v_z"]
    for k in range(3):
        axs[k].scatter(np.arange(M), p_out[:, k], s=5, color=color_mapping, label=label_list[k])
        axs[k].scatter(np.arange(M), p_out_est[:, k], s=5, color=color_mapping, alpha=0.2)




def plot_q_out(q_in, q_out, ori_obj):
    """compare estimated p_out given p_in with groud truth"""

    q_out = list_to_arr(q_out)

    q_out_est     = np.zeros((q_out.shape)) # (M, N)

    gamma = ori_obj.gmm.logProb(q_in)

    q_diff  = riem_log(ori_obj.q_att, q_in)
    
    q_out_att     = np.zeros((q_out.shape)) # (M, N)
    for k in range(ori_obj.K):
        q_out_att  +=  (np.tile(gamma[k, :], (4, 1)) * (ori_obj.A_ori[k] @ q_diff.T)).T


    for i in range(q_out_est.shape[0]):
        q_out_body = parallel_transport(ori_obj.q_att, q_in[i], q_out_att[i, :].T)
        q_out_q    = riem_exp(q_in[i], q_out_body) 
        # q_out_i      = R.from_quat(q_out_q.reshape(4,))
        q_out_est[i, :] = q_out_q




    M, N = q_out_est.shape

    fig, axs = plt.subplots(4, 1, figsize=(12, 8))

    label = ori_obj.gmm.assignment_arr

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]

    color_mapping = np.take(colors, label)

    label_list =["q_x", "q_y", "q_z", "q_w"]
    for k in range(4):
        axs[k].scatter(np.arange(M), q_out[:, k], s=5, color=color_mapping, label=label_list[k])
        axs[k].scatter(np.arange(M), q_out_est[:, k], s=5, color=color_mapping, alpha=0.2)

    
