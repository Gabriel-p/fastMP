
import warnings
import numpy as np
from astropy.stats import RipleysKEstimator
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from scipy import spatial
import matplotlib.pyplot as plt
# from .GUMM import GUMMProbs


"""
TODO:

1. Estimate 'plx_rad' from plx error data

"""


class fastMP:
    """
    """

    def __init__(self,
                 N_resample=100,
                 fix_N_clust=False,
                 coord_dens_perc=0, #0.05,
                 PM_rad=10, #5,
                 PM_cent_rad=100, #0.5,
                 plx_rad=100, #0.3,
                 pc_rad=10000, #50,
                 N_break=5,
                 xy_c=None,
                 vpd_c=None,
                 plx_c=None,
                 fixed_centers=False,
                 centers_ex=None):
        self.N_resample = N_resample
        self.fix_N_clust = fix_N_clust
        self.coord_dens_perc = coord_dens_perc
        self.PM_rad = PM_rad
        self.PM_cent_rad = PM_cent_rad
        self.plx_rad = plx_rad
        self.pc_rad = pc_rad
        self.N_break = N_break
        self.xy_c = xy_c
        self.vpd_c = vpd_c
        self.plx_c = plx_c
        self.fixed_centers = fixed_centers
        self.centers_ex = centers_ex

    def fit(self, X):
        """
        """

        # Prepare dictionary of parameters for extra clusters in frame (if any)
        self.prep_extra_cl_dict()

        # Remove nans
        msk_accpt, X = self.reject_nans(X)

        # Unpack input data
        if self.N_resample >= 10:
            lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx, Gmag, BPRP = X
        else:
            raise ValueError("A minimum of 10 resamples is required")
            # lon, lat, pmRA, pmDE, plx, Gmag, BPRP = X
            # # Dummy variables
            # e_pmRA, e_pmDE, e_plx = [np.array([]) for _ in range(3)]

        # Remove the most obvious field stars to speed up the process
        msk_accpt, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c, e_pmRA,\
            e_pmDE, e_plx, Gmag, BPRP = self.first_filter(
                msk_accpt, lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx, Gmag, BPRP)

        # cx, cy = .5 * (max(lon) + min(lon)), .5 * (max(lat) + min(lat))
        # d_pc = 1000 / plx
        # l_rad, b_rad = np.deg2rad(lon - cx), np.deg2rad(lat - cy)
        # x = d_pc * np.sin(l_rad) * np.cos(b_rad)
        # y = d_pc * np.sin(b_rad)
        # p_pmRA, p_pmDE = d_pc * pmRA, d_pc * pmDE

        # import astropy.coordinates as coord
        # import astropy.units as u
        # from astropy.coordinates import SkyCoord
        # from astropy.coordinates import Galactic
        # lon_lat = SkyCoord(l=lon * u.degree, b=lat * u.degree, frame='galactic')
        # c1 = coord.SkyCoord(
        #     ra=lon_lat.fk5.ra, dec=lon_lat.fk5.dec,
        #     distance=(plx*u.mas).to(u.pc, u.parallax()),
        #     pm_ra_cosdec=pmRA*u.mas/u.yr,
        #     pm_dec=pmDE*u.mas/u.yr)
        # breakpoint()

        # Prepare Ripley's K data
        self.init_ripley(lon, lat)

        # Estimate the number of members
        if self.fix_N_clust is False:
            N_survived, st_idx = self.estimate_nmembs(
                lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c)
        else:
            N_survived, st_idx = int(self.fix_N_clust), None

        N_runs, idx_selected, centers = 0, [], [[], [], []]
        for _ in range(self.N_resample + 1):

            # print(xy_c, vpd_c, plx_c)

            centers[0].append(xy_c)
            centers[1].append(vpd_c)
            centers[2].append(plx_c)

            # Sample data
            s_pmRA, s_pmDE, s_plx = self.data_sample(
                pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx)

            # if st_idx is not None and len(st_idx) > 5:
            #     data_5d = self.get_dims_norm(
            #         lon, lat, s_pmRA, s_pmDE, s_plx, xy_c, vpd_c, plx_c,
            #         st_idx)
            #     cents_5d = np.array([[0., 0., 0., 0., 0.]])
            # else:
            #     data_5d = np.array([lon, lat, s_pmRA, s_pmDE, s_plx]).T
            #     cents_5d = np.array([xy_c + vpd_c + [plx_c]])

            # Indexes of the sorted 5D distances 5D to the estimated center
            cents_5d = np.array([xy_c + vpd_c + [plx_c]])
            data_5d = np.array([lon, lat, s_pmRA, s_pmDE, s_plx]).T
            d_idxs = self.get_Nd_dists(cents_5d, data_5d)

            # Star selection
            st_idx = d_idxs[:N_survived]

            # Filter outlier field stars
            msk = self.filter_xy_pms_plx(
                lon[st_idx], lat[st_idx], pmRA[st_idx], pmDE[st_idx],
                plx[st_idx], xy_c, vpd_c, plx_c)
            st_idx = st_idx[msk]

            # Filer extra clusters in frame (if any)
            st_idx = self.filter_cls_in_frame(
                lon[st_idx], lat[st_idx], pmRA[st_idx], pmDE[st_idx],
                plx[st_idx], xy_c, vpd_c, plx_c, st_idx)

            # Re-estimate centers using the selected stars
            if len(st_idx) > 5:
                xy_c, vpd_c, plx_c = self.get_5D_center(
                    lon[st_idx], lat[st_idx], pmRA[st_idx], pmDE[st_idx],
                    plx[st_idx])

                idx_selected += list(st_idx)
                N_runs += 1

        # Check to see if any center moved around
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c_xy, c_vpd, c_plx = [np.array(_) for _ in centers]
            zs_xy = ((c_xy - c_xy.mean()) / c_xy.std()).mean(0)
            zs_vpd = ((c_vpd - c_vpd.mean()) / c_vpd.std()).mean(0)
            zs_plx = ((c_plx - c_plx.mean()) / c_plx.std()).mean()
            if zs_xy.any() > 2:
                warnings.warn("Center in xy varied by Z_score>2", UserWarning)
            if zs_vpd.any() > 2:
                warnings.warn("Center in PMs varied by Z_score>2", UserWarning)
            if zs_plx.any() > 2:
                warnings.warn("Center in Plx varied by Z_score>2", UserWarning)

        probs_final = self.assign_probs(msk_accpt, idx_selected, N_runs)

        if abs(N_survived - (probs_final > .5).sum()) / N_survived > .25:
            warnings.warn(
                "The estimated number of members differ\nmore than 25% with "
                + "stars with P>0.5", UserWarning)

        return probs_final, N_survived

    def prep_extra_cl_dict(self):
        """
        The parameter 'centers_ex' must be a list of dictionaries, one
        dictionary for each extra cluster in the frame. The dictionaries
        must have at most three keys, 'xy', 'pms', 'plx', each with a list
        containing the center value(s) in those dimensions. Example:

        centers_ex = [{'xy': [105.39, 0.9], 'plx': [1.3]}]

        centers_ex = [
            {'xy': [105.39, 0.9], 'pms': [3.5, -0.7]},
            {'plx': [0.82], 'pms': [3.5, -0.7]}
        ]
        """
        extra_cls_dict = {'run_flag': False}

        if self.centers_ex is None:
            self.extra_cls_dict = extra_cls_dict
            return

        extra_cls_dict['run_flag'] = True

        # Set the distribution of dimensions
        self.dims_msk = {
            'xy': np.array([0, 1]), 'pms': np.array([2, 3]),
            'plx': np.array([4]), 'xy_pms': np.array([0, 1, 2, 3]),
            'xy_plx': np.array([0, 1, 4]), 'pms_plx': np.array([2, 3, 4]),
            'xy_pms_plx': np.array([0, 1, 2, 3, 4])}

        # Read centers of the extra clusters in frame
        dims = ['xy', 'pms', 'plx']
        dim_keys, cents = [], []
        for extra_cl in self.centers_ex:
            dims_ex = extra_cl.keys()
            dims_id, centers = '', []
            for d in dims:
                if d in dims_ex:
                    center = extra_cl[d]
                    if center is not None:
                        dims_id += '_' + d
                        centers += center
            # Store dimensions ids and centers for each extra cluster
            dim_keys.append(dims_id[1:])
            cents.append(centers)

        extra_cls_dict['dim_keys'] = dim_keys
        extra_cls_dict['centers'] = cents

        self.extra_cls_dict = extra_cls_dict

    def reject_nans(self, data):
        """
        Remove nans
        """
        msk_all = []
        # Process each dimension separately
        for arr in data:
            msk = ~np.isnan(arr)
            msk_all.append(msk.data)
        # Combine into a single mask
        msk_accpt = np.logical_and.reduce(msk_all)

        return msk_accpt, data.T[msk_accpt].T

    def first_filter(
        self, msk_accpt, lon, lat, pmRA, pmDE, plx, e_pmRA,
        e_pmDE, e_plx, Gmag, BPRP
    ):
        """
        """
        # Estimate initial center
        xy_c, vpd_c, plx_c = self.get_5D_center(lon, lat, pmRA, pmDE, plx)

        # Remove obvious field stars
        msk = self.filter_xy_pms_plx(
            lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c, True)
        # Update arrays
        lon, lat, pmRA, pmDE, plx = lon[msk], lat[msk], pmRA[msk], pmDE[msk],\
            plx[msk]
        if self.N_resample > 0:
            e_pmRA, e_pmDE, e_plx = e_pmRA[msk], e_pmDE[msk], e_plx[msk]
            Gmag, BPRP = Gmag[msk], BPRP[msk]

        # Update mask of accepted elements
        N_accpt = len(msk_accpt)
        # Indexes accepted by both masks
        idxs = np.arange(0, N_accpt)[msk_accpt][msk]
        msk_accpt = np.full(N_accpt, False)
        msk_accpt[idxs] = True

        return msk_accpt, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c,\
            e_pmRA, e_pmDE, e_plx, Gmag, BPRP

    def get_5D_center(self, lon, lat, pmRA, pmDE, plx, N_cent=200):
        """
        Estimate the 5-dimensional center of the cluster.

        Possible scenarios:

        A. vpd_c=None
          1. Estimate vpd_c
        B. vpd_c!=None
          1. Use manual vpd_c

        AB1. xy_c=None, plx_c=None
          2. Obtain indexes of closest distances to vpd_c center
        AB2. xy_c=None, plx_c!=None
          2. Obtain indexes of closest distances to vpd_c+plx_c center
        AB3. xy_c!=None, plx_c=None
          2. Obtain indexes of closest distances to vpd_c+xy_c center
        AB4. xy_c!=None, plx_c!=None
          2. Obtain indexes of closest distances to vpd_c+plx_c+xy_c center

        3. Estimate 5D center with KDE

        N_cent: how many stars are used to estimate the center.

        """
        # (Re)estimate VPD center
        vpd_c = self.get_pms_center(pmRA, pmDE)

        # Distances to center
        # AB1
        if self.xy_c is None and self.plx_c is None:
            dist = (pmRA - vpd_c[0])**2 + (pmDE - vpd_c[1])**2
        # AB2
        elif self.xy_c is None and self.plx_c is not None:
            dist = (pmRA - vpd_c[0])**2 + (pmDE - vpd_c[1])**2\
                + (plx - self.plx_c)**2
        # AB3
        elif self.xy_c is not None and self.plx_c is None:
            dist = (pmRA - vpd_c[0])**2 + (pmDE - vpd_c[1])**2\
                + (lon - self.xy_c[0])**2 + (lat - self.xy_c[1])**2
        # AB4
        elif self.xy_c is not None and self.plx_c is not None:
            dist = (pmRA - vpd_c[0])**2 + (pmDE - vpd_c[1])**2\
                + (lon - self.xy_c[0])**2 + (lat - self.xy_c[1])**2\
                + (plx - self.plx_c)**2

        # Closest stars to the selected center
        idx = dist.argsort()[:N_cent]  # HARDCODED

        # Estimate 5D center with KDE
        d1, d2, d3, d4, d5 = lon[idx], lat[idx], pmRA[idx], pmDE[idx], plx[idx]
        d1_5 = np.vstack([d1, d2, d3, d4, d5])
        try:
            # Define Gaussian KDE
            kde = gaussian_kde(d1_5)
            # Evaluate in the selected closest stars to the center
            density = kde(d1_5)
            # Extract new centers as those associated to the maximum density
            x_c, y_c, pmra_c, pmde_c, plx_c = d1_5[:, density.argmax()]
        except:
            warnings.warn("Could not estimate the KDE 5D center", UserWarning)
            x_c, y_c, pmra_c, pmde_c, plx_c = self.xy_c[0], self.xy_c[1],\
                vpd_c[0], vpd_c[1], self.plx_c
            breakpoint()
        vpd_c = (pmra_c, pmde_c)

        if self.fixed_centers is True:
            if self.xy_c is not None:
                [x_c, y_c] = self.xy_c
            if self.vpd_c is not None:
                vpd_c = self.vpd_c
            if self.plx_c is not None:
                plx_c = self.plx_c

        return [x_c, y_c], list(vpd_c), plx_c

    def get_pms_center(self, pmRA, pmDE, N_bins=50, zoom_f=4, N_zoom=10):
        """
        """
        vpd_mc = self.vpd_c

        vpd = np.array([pmRA, pmDE]).T
        # Center in PMs space
        cx, cxm = None, None
        for _ in range(N_zoom):
            N_stars = vpd.shape[0]
            if N_stars < 5:  # HARDCODED
                break

            # Find center coordinates as max density
            x, y = vpd.T
            H, edgx, edgy = np.histogram2d(x, y, bins=N_bins)
            flat_idx = H.argmax()
            cbx, cby = np.unravel_index(flat_idx, H.shape)
            cx = (edgx[cbx + 1] + edgx[cbx]) / 2.
            cy = (edgy[cby + 1] + edgy[cby]) / 2.

            # If a manual center was set, use it
            if vpd_mc is not None:
                # Store the auto center for later
                cxm, cym = cx, cy
                # Use the manual to zoom in
                cx, cy = vpd_mc

            # Zoom in
            rx, ry = edgx[1] - edgx[0], edgy[1] - edgy[0]
            msk = (x < (cx + zoom_f * rx))\
                & (x > (cx - zoom_f * rx))\
                & (y < (cy + zoom_f * ry))\
                & (y > (cy - zoom_f * ry))
            vpd = vpd[msk]

        # If a manual center was set
        if vpd_mc is not None:
            # If a better center value could be estimated
            if cxm is not None:
                cx, cy = cxm, cym
            else:
                cx, cy = vpd_mc
                warnings.warn("Could not estimate a better PMs center value",
                              UserWarning)
        else:
            if cx is None:
                cx, cy = vpd_mc
                raise Exception("Could not estimate the PMs center value")

        return (cx, cy)

    def filter_xy_pms_plx(
        self, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c,
        ff_flag=False, NN=5
    ):
        """
        Identify obvious field stars
        """

        # Don't use on the first filter. Dont use for too few or too many stars
        if ff_flag is False and len(lon) > 10 and len(lon) < 1000: # HARDCODED
            # Find the distance to the XY center where the density of stars
            # drops below a fixed threshold. Use this as a value to determine
            # the maximum XY radius.
            xys = np.array([lon, lat]).T
            dist_xy = cdist(xys, np.array([xy_c])).T[0]

            # Find nearest neighbors.
            tree = spatial.cKDTree(xys)
            inx = tree.query(xys, k=NN) # HARDCODED
            # Max distance to the NN neighbors.
            NN_dist = inx[0].max(1)
            # Convert to densities
            dens = 1. / NN_dist
            # Normalize
            dens /= dens.max()
            msk_dens = dens <= self.coord_dens_perc
            if any(msk_dens):
                idx = np.argmax(msk_dens)
            else:
                idx = np.argmax(dist_xy)
            msk_xy = (dist_xy < dist_xy[idx])

            # # if msk_xy.sum() < 5:
            # if not any(msk_dens):
            #     print("error")

            #     print(idx, dist_xy[idx])
            #     ax = plt.subplot(121)
            #     plt.scatter(lon, lat)
            #     circle = plt.Circle(
            #         (np.median(lon), np.median(lat)), dist_xy[idx], color='r',
            #         fill=False)
            #     ax.add_patch(circle)
            #     plt.subplot(122)
            #     plt.scatter(dist_xy, dens)
            #     plt.axhline(0.05, c='r')
            #     plt.axvline(dist_xy[idx], c='r')
            #     plt.show()
            #     breakpoint()

        else:
            msk_xy = np.full(len(pmRA), True)

        # Hard PMs limit
        pms = np.array([pmRA, pmDE]).T
        dist_pm = cdist(pms, np.array([vpd_c])).T[0]
        pmRad = self.PM_rad  #max(self.PM_rad, self.PM_cent_rad * abs(vpd_c[0]))

        # Hard Plx limit
        d_pc = 1000 / plx_c
        plx_min = 1000 / (d_pc + self.pc_rad)
        plx_max = max(.05, 1000 / (d_pc - self.pc_rad))  # HARDCODED
        plxRad_min = max(self.plx_rad, plx_c - plx_min)
        plxRad_max = max(self.plx_rad, plx_max - plx_c)

        msk = (dist_pm < pmRad) & (plx - plx_c < plxRad_max)\
            & (plx_c - plx < plxRad_min) & msk_xy

        return msk

    def init_ripley(self, lon, lat):
        """
        https://rdrr.io/cran/spatstat/man/Kest.html
        "For a rectangular window it is prudent to restrict the r values to a
        maximum of 1/4 of the smaller side length of the rectangle
        (Ripley, 1977, 1988; Diggle, 1983)"
        """
        xmin, xmax = lon.min(), lon.max()
        ymin, ymax = lat.min(), lat.max()
        area = (xmax - xmin) * (ymax - ymin)
        Kest = RipleysKEstimator(
            area=area, x_max=xmax, y_max=ymax, x_min=xmin, y_min=ymin)

        # Ripley's rule of thumb
        thumb = 0.25 * min((xmax - xmin), (ymax - ymin))
        # Large sample rule
        rho = len(lon) / area
        large = np.sqrt(1000 / (np.pi * rho))

        lmin = min(thumb, large)
        rads = np.linspace(lmin * 0.01, lmin, 100)  # HARDCODED

        C_thresh_N = 1.68 * np.sqrt(area)  # HARDCODED

        self.rads, self.Kest, self.C_thresh_N = rads, Kest, C_thresh_N

    def estimate_nmembs(
        self, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c,
        N_clust=50, N_clust_min=10, prob_cut=.5
    ):
        """
        Estimate the number of cluster members
        """
        cents_3d = np.array([list(vpd_c) + [plx_c]])
        data_3d = np.array([pmRA, pmDE, plx]).T
        d_pm_plx_idxs = self.get_Nd_dists(cents_3d, data_3d)

        xy = np.array([lon, lat]).T
        N_stars = xy.shape[0]

        def core(N_clust_surv):
            """
            """
            C_thresh = self.C_thresh_N / N_clust_surv

            idx_survived = []
            N_break_count, step_old = 0, 0
            for step in np.arange(N_clust_surv, N_stars, N_clust_surv):

                msk = d_pm_plx_idxs[step_old:step]
                C_s = self.rkfunc(xy[msk], self.rads, self.Kest)
                if not np.isnan(C_s):
                    # Cluster survived
                    if C_s >= C_thresh:
                        idx_survived += list(msk)
                    else:
                        # Break condition
                        N_break_count += 1
                if N_break_count > self.N_break:
                    break
                step_old = step

            return idx_survived

        # Select those clusters where the stars are different enough from a
        # random distribution
        idx_survived = core(N_clust)
        if not idx_survived:
            # If the default clustering number did not work, try a few
            # different values
            for N_clust_surv in (65, 75, 85, 95, 105): # HARDCODED
                idx_survived = core(N_clust_surv)
                # Break out when (if) any value selected stars
                if len(idx_survived) > 0:
                    break

        if len(idx_survived) < N_clust_min:
            warnings.warn(
                f"The estimated number of cluster members is <{N_clust_min}",
                UserWarning)
            # print(10, 10)
            return N_clust_min, None

        # Filter extra clusters in frame (if any)
        msk = np.array(idx_survived)
        idx_survived = self.filter_cls_in_frame(
            lon[msk], lat[msk], pmRA[msk], pmDE[msk], plx[msk], xy_c, vpd_c,
            plx_c, msk)

        # # Filter by GUMM in PMs
        # gumm_p = GUMMProbs(np.array([pmRA[idx_survived], pmDE[idx_survived]]))
        # # Create the percentiles vs probabilities array.
        # percent = np.arange(1, 99, 1)
        # perc_probs = np.array([percent, np.percentile(gumm_p, percent)]).T
        # # Find 'elbow' where the probabilities start climbing from ~0.
        # prob_cut0 = self.rotate(perc_probs)
        # prob_cut = min(0.5, prob_cut0)
        # # N_survived = (gumm_p > prob_cut).sum()
        # idx_survived = idx_survived[gumm_p > prob_cut]

        # IDs of stars that survived Ripley's filter but belong to extra
        # clusters in the frame
        ex_cls_ids = np.array(list(
            set(msk).symmetric_difference(set(idx_survived))))
        # Filter by KDE
        kde_probs = self.kde_probs(lon, lat, idx_survived, ex_cls_ids)
        N_survived = len(idx_survived)
        if kde_probs is not None:
            msk = kde_probs > prob_cut
            N_survived = min(N_survived, msk.sum())
            idx_survived = idx_survived[msk]

        if N_survived < N_clust_min:
            warnings.warn(
                f"The estimated number of cluster members is <{N_clust_min}",
                UserWarning)
            return N_clust_min, None

        print(f"Estimated number of members: {N_survived}")

        return N_survived, idx_survived

    # def rotate(self, data):
    #     """
    #     Rotate a 2d vector.

    #     (Very) Stripped down version of the great 'kneebow' package, by Georg
    #     Unterholzner. Source:

    #     https://github.com/georg-un/kneebow

    #     data   : 2d numpy array. The data that should be rotated.
    #     return : probability corresponding to the elbow.
    #     """
    #     # The angle of rotation in radians.
    #     theta = np.arctan2(
    #         data[-1, 1] - data[0, 1], data[-1, 0] - data[0, 0])

    #     # Rotation matrix
    #     co, si = np.cos(theta), np.sin(theta)
    #     rot_matrix = np.array(((co, -si), (si, co)))

    #     # Rotate data vector
    #     rot_data = data.dot(rot_matrix)

    #     # Find elbow index
    #     elbow_idx = np.where(rot_data == rot_data.min())[0][0]

    #     # Adding a small percentage to the selected probability improves the
    #     # results by making the cut slightly stricter.
    #     prob_cut = data[elbow_idx][1] + 0.05

    #     return prob_cut

    def get_dims_norm(
        self, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c, msk
    ):
        """
        """
        dims_norm = []
        dim_centers = (xy_c[0], xy_c[1], vpd_c[0], vpd_c[1], plx_c)
        for i, dim in enumerate((lon, lat, pmRA, pmDE, plx)):
            dims_norm.append(2 * np.percentile(
                abs(dim[msk] - dim_centers[i]), 50))

        data = np.array([lon, lat, pmRA, pmDE, plx]).T
        cents_5d = np.array(xy_c + vpd_c + [plx_c])
        data_mvd = data - cents_5d
        data_norm = data_mvd / np.array(dims_norm)

        return data_norm

    def get_Nd_dists(self, cents, data, dists_flag=False):
        """
        Obtain indexes and distances of stars to the PMs+Plx center
        """
        # Distances to center
        dist_Nd = cdist(data, cents).T[0]
        if dists_flag:
            return dist_Nd

        # Indexes that sort the distances
        d_idxs = dist_Nd.argsort()

        return d_idxs

    def filter_cls_in_frame(
        self, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c, idx_survived
    ):
        """
        """
        if self.extra_cls_dict['run_flag'] is False:
            return idx_survived

        data = np.array([lon, lat, pmRA, pmDE, plx])
        cents = np.array([xy_c + vpd_c + [plx_c]])

        # Find the distances to the center for all the combinations of data
        # dimensions in the extra clusters in frame
        dists = {}
        for dims in list(set(self.extra_cls_dict['dim_keys'])):
            msk = self.dims_msk[dims]
            dists[dims] = self.get_Nd_dists(
                cents[:, msk], data[msk, :].T, True)

        #
        msk_d = np.full(len(idx_survived), True)
        for i, cent_ex in enumerate(self.extra_cls_dict['centers']):
            dims_ex = self.extra_cls_dict['dim_keys'][i]
            msk = self.dims_msk[dims_ex]
            dists_ex = self.get_Nd_dists(
                np.array([cent_ex]), data[msk, :].T, True)

            # If the distance to the selected cluster center is smaller than
            # the distance to this extra cluster's center, keep the star
            msk_d = msk_d & (dists[dims_ex] <= dists_ex)

        # Never return less than 10 stars
        if msk_d.sum() < 10:
            return idx_survived
        return idx_survived[msk_d]

    def rkfunc(self, xy, rads, Kest):
        """
        Test how similar this cluster's (x, y) distribution is compared
        to a uniform random distribution using Ripley's K.
        https://stats.stackexchange.com/a/122816/10416

        Parameters
        ----------
        xy : TYPE
            Description
        rads : TYPE
            Description
        Kest : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        # Avoid large memory consumption if the data array is too big
        # if xy.shape[0] > 5000:
        #     mode = "none"
        # else:
        #     mode = 'translation'

        # Hide RunTimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            L_t = Kest.Lfunction(xy, rads, mode='translation')

        # Catch all-nans. Avoid 'RuntimeWarning: All-NaN slice encountered'
        if np.isnan(L_t).all():
            C_s = np.nan
        else:
            C_s = np.nanmax(abs(L_t - rads))

        return C_s

    def kde_probs(
        self, lon, lat, idx_survived, ex_cls_ids, N_min=25, Nst_max=5000
    ):
        """
        Assign probabilities to all stars after generating the KDEs for field
        and member stars. The cluster member probability is obtained applying
        the formula for two mutually exclusive and exhaustive hypotheses.
        """
        if len(idx_survived) < N_min:
            return None

        all_idxs = set(np.arange(len(lon)))
        field_idxs = np.array(list(
            all_idxs.symmetric_difference(idx_survived)))
        if len(ex_cls_ids) > 0:
            field_idxs = np.array(list(
                set(field_idxs).symmetric_difference(ex_cls_ids)))

        # Combine coordinates with the rest of the features.
        all_data = np.array([lon, lat]).T

        # Split into the two populations.
        field_stars = all_data[field_idxs]
        membs_stars = all_data[idx_survived]

        # To improve the performance, cap the number of stars using a random
        # selection of 'Nf_max' elements.
        if field_stars.shape[0] > Nst_max:
            idxs = np.arange(field_stars.shape[0])
            np.random.shuffle(idxs)
            field_stars = field_stars[idxs[:Nst_max]]

        if len(field_stars) < N_min:
            return None

        # Evaluate all stars in both KDEs
        try:
            kd_field = gaussian_kde(field_stars.T)
            kd_memb = gaussian_kde(membs_stars.T)

            L_memb = kd_memb.evaluate(membs_stars.T)
            L_field = kd_field.evaluate(membs_stars.T)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Probabilities for mutually exclusive and exhaustive
                # hypotheses
                cl_probs = 1. / (1. + (L_field / L_memb))

        except (np.linalg.LinAlgError, ValueError):
            # warnings.warn("Could not perform KDE probabilities estimation")
            cl_probs = np.ones(len(lon))

        return cl_probs

    def data_sample(self, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx):
        """
        Gaussian random sample
        """
        # if self.N_resample == 0:
        #     return pmRA, pmDE, plx

        data_3 = np.array([pmRA, pmDE, plx])
        grs = np.random.normal(0., 1., data_3.shape[1])
        data_err = np.array([e_pmRA, e_pmDE, e_plx])
        return data_3 + grs * data_err

    def assign_probs(self, msk_accpt, idx_selected, N_runs):
        """
        """
        # Initial -1 probabilities for *all* stars
        probs_final = np.zeros(len(msk_accpt)) - 1

        # Number of processed stars (ie: not rejected as nans)
        N_stars = msk_accpt.sum()
        # Initial zero probabilities for the processed stars
        probs_all = np.zeros(N_stars)

        if idx_selected:
            # Estimate probabilities as averages of counts
            values, counts = np.unique(idx_selected, return_counts=True)
            probs = counts / N_runs
            # Store probabilities for processed stars
            probs_all[values] = probs

        # Assign the estimated probabilities to the processed stars
        probs_final[msk_accpt] = probs_all

        return probs_final
