
import warnings
import numpy as np
from astropy.stats import RipleysKEstimator
from scipy import spatial
from scipy.stats import gaussian_kde


class fastMP:
    """
    """

    def __init__(self,
                 xy_c=None,
                 vpd_c=None,
                 plx_c=None,
                 fix_N_clust=False,
                 fixed_centers=False,
                 centers_ex=None,
                 N_resample=200,
                 N_clust_max=5000):
        self.xy_c = xy_c
        self.vpd_c = vpd_c
        self.plx_c = plx_c
        self.fix_N_clust = fix_N_clust
        self.fixed_centers = fixed_centers
        self.centers_ex = centers_ex
        self.N_resample = int(N_resample)
        self.N_clust_max = int(N_clust_max)

    def fit(self, X):
        """
        """
        # HARDCODED <-- TODO
        N_min_st_cent = 5
        N_min_resample = 10
        warnings.formatwarning = self.warning_on_one_line

        if self.N_resample < N_min_resample:
            raise ValueError("A minimum of 10 resample runs is required")

        # Prepare dictionary of parameters for extra clusters in frame (if any)
        self.prep_extra_cl_dict()

        # Remove nans
        N_all = X.shape[1]
        idx_clean, X = self.reject_nans(X)

        # Unpack input data
        lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx = X

        # Estimate initial center
        xy_c, vpd_c, plx_c = self.get_5D_center(lon, lat, pmRA, pmDE, plx)

        # Remove the most obvious field stars to speed up the process
        idx_clean, lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx =\
            self.first_filter(
                idx_clean, vpd_c, plx_c, lon, lat, pmRA, pmDE, plx,
                e_pmRA, e_pmDE, e_plx)

        # Prepare Ripley's K data
        self.init_ripley(lon, lat)

        # Estimate the number of members
        if type(self.fix_N_clust) in (int, float):
            N_survived, st_idx = int(self.fix_N_clust), None
        else:
            N_survived, st_idx = self.estimate_nmembs(
                lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c)

        N_runs, idx_selected, N_05_old, prob_old, break_check = 0, [], 0, 1, 0
        for _ in range(self.N_resample + 1):

            # Sample data
            s_pmRA, s_pmDE, s_plx = self.data_sample(
                pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx)

            # Data normalization
            data_5d, cents_5d = self.get_dims_norm(
                lon, lat, s_pmRA, s_pmDE, s_plx, xy_c, vpd_c, plx_c, st_idx)

            # Indexes of the sorted 5D distances 5D to the estimated center
            d_idxs = self.get_Nd_dists(cents_5d, data_5d)

            # Star selection
            st_idx = d_idxs[:N_survived]

            # Filter extra clusters in frame (if any)
            st_idx = self.filter_cls_in_frame(
                lon[st_idx], lat[st_idx], pmRA[st_idx], pmDE[st_idx],
                plx[st_idx], xy_c, vpd_c, plx_c, st_idx)

            # Re-estimate centers using the selected stars
            if len(st_idx) > N_min_st_cent:
                xy_c, vpd_c, plx_c = self.get_5D_center(
                    lon[st_idx], lat[st_idx], pmRA[st_idx], pmDE[st_idx],
                    plx[st_idx])

                idx_selected += list(st_idx)
                N_runs += 1

            # Convergence check
            if idx_selected:
                counts = np.unique(idx_selected, return_counts=True)[1]
                probs = counts/N_runs
                msk = probs > 0.5
                prob_mean = np.mean(probs[msk])
                delta_probs = abs(prob_mean - prob_old)
                N_05 = msk.sum()
                if N_05 == N_05_old or delta_probs < .001:  # HARDCODED
                    break_check += 1
                else:
                    # Reset
                    break_check = 0
                prob_old, N_05_old = prob_mean, N_05
                if break_check > break_count:
                    print(f"Convergence reached at {N_runs} runs. Breaking")
                    break

        # Assign final probabilities
        probs_final = self.assign_probs(N_all, idx_clean, idx_selected, N_runs)

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
            {'xy': [0.82, -4.5], 'pms': [3.5, -0.7], 'plx': [3.5]}
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

        # Indexes that survived
        idx_clean = np.arange(data.shape[1])[msk_accpt]

        return idx_clean, data.T[msk_accpt].T

    def get_5D_center(self, lon, lat, pmRA, pmDE, plx, N_cent=500):
        """
        Estimate the 5-dimensional center of the cluster. Steps:

        1. Estimate the center in PMs (the value can be given as input)
        2. Using this center with any other given center, obtain the
           'N_cent' stars closest to the combined center.
        3. Estimate the 5-dimensional final center using KDE

        N_cent: how many stars are used to estimate the KDE center.

        """
        # Skip process if all centers are fixed
        if self.fixed_centers is True and self.xy_c is not None\
                and self.vpd_c is not None and self.plx_c is not None:
            x_c, y_c = self.xy_c
            pmra_c, pmde_c = self.vpd_c
            plx_c = self.plx_c
            return [x_c, y_c], [pmra_c, pmde_c], plx_c

        # Re-write if this parameter is given
        if type(self.fix_N_clust) in (int, float):
            N_cent = self.fix_N_clust

        # Get filtered stars close to given xy+Plx centers (if any) to use
        # in the PMs center estimation
        pmRA_i, pmDE_i = self.filter_pms_stars(
            lon, lat, pmRA, pmDE, plx, N_cent)

        # (Re)estimate VPD center
        vpd_c = self.get_pms_center(pmRA_i, pmDE_i)

        # Get N_cent stars closest to vpd_c and given xy and/or plx centers
        lon_i, lat_i, pmRA_i, pmDE_i, plx_i = self.get_stars_close_center(
            lon, lat, pmRA, pmDE, plx, vpd_c, N_cent)

        # kNN center
        x_c, y_c, pmra_c, pmde_c, plx_c = self.get_kNN_center(np.array([
            lon_i, lat_i, pmRA_i, pmDE_i, plx_i]).T)

        # Re-write values if necessary
        if self.fixed_centers is True:
            if self.xy_c is not None:
                x_c, y_c = self.xy_c
            if self.vpd_c is not None:
                pmra_c, pmde_c = self.vpd_c
            if self.plx_c is not None:
                plx_c = self.plx_c

        return [x_c, y_c], [pmra_c, pmde_c], plx_c

    def filter_pms_stars(self, lon, lat, pmRA, pmDE, plx, N_cent):
        """
        """
        # Distances to xy_c+plx_c centers, if any was given
        if self.xy_c is None and self.plx_c is None:
            return pmRA, pmDE

        if self.xy_c is None and self.plx_c is not None:
            cent = np.array([[self.plx_c]])
            data = np.array([plx]).T
        elif self.xy_c is not None and self.plx_c is None:
            cent = np.array([self.xy_c])
            data = np.array([lon, lat]).T
        elif self.xy_c is not None and self.plx_c is not None:
            cent = np.array([list(self.xy_c) + [self.plx_c]])
            data = np.array([lon, lat, plx]).T

        # Closest stars to the selected center
        idx = self.get_Nd_dists(cent, data)[:N_cent]
        pmRA_i, pmDE_i = pmRA[idx], pmDE[idx]

        return pmRA_i, pmDE_i

    def get_pms_center(
        self, pmRA, pmDE, N_bins=50, zoom_f=4, N_zoom=10, N_min_st=5
    ):
        """
        """
        vpd_mc = self.vpd_c

        vpd = np.array([pmRA, pmDE]).T

        # Center in PMs space
        cx, cxm = None, None
        for _ in range(N_zoom):
            N_stars = vpd.shape[0]
            if N_stars < N_min_st:
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

        return [cx, cy]

    def get_stars_close_center(self, lon, lat, pmRA, pmDE, plx, vpd_c, N_cent):
        """
        Distances to centers using the vpd_c and other available data
        """
        if self.xy_c is None and self.plx_c is None:
            cent = np.array([vpd_c])
            data = np.array([pmRA, pmDE]).T
        elif self.xy_c is None and self.plx_c is not None:
            cent = np.array([vpd_c + [self.plx_c]])
            data = np.array([pmRA, pmDE, plx]).T
        elif self.xy_c is not None and self.plx_c is None:
            cent = np.array([list(self.xy_c) + vpd_c])
            data = np.array([lon, lat, pmRA, pmDE]).T
        elif self.xy_c is not None and self.plx_c is not None:
            cent = np.array([list(self.xy_c) + vpd_c + [self.plx_c]])
            data = np.array([lon, lat, pmRA, pmDE, plx]).T

        # Closest stars to the selected center
        idx = self.get_Nd_dists(cent, data)[:N_cent]

        return lon[idx], lat[idx], pmRA[idx], pmDE[idx], plx[idx]

    def get_kNN_center(self, data, Ndens=25, Nstars=10):
        """
        Estimate 5D center with kNN. Better results are obtained not using
        the parallax data
        """
        data_noplx = data[:, :4]

        tree = spatial.cKDTree(data_noplx)
        inx = tree.query(data_noplx, k=Ndens + 1)
        NN_dist = inx[0].max(1)
        # Convert to densities
        dens = 1. / NN_dist
        # Sort by largest density
        idxs = np.argsort(-dens)

        # Use the star with the largest density
        # cent = np.array([data[idxs[0]]])[0]
        # # Use median of stars with largest densities
        cent = np.median(data[idxs[:Nstars]], 0)

        x_c, y_c, pmra_c, pmde_c, plx_c = cent
        return x_c, y_c, pmra_c, pmde_c, plx_c

    def first_filter(
        self, idx_all, vpd_c, plx_c, lon, lat, pmRA, pmDE, plx, e_pmRA,
        e_pmDE, e_plx, plx_cut=0.5, v_kms_max=5, pm_max=3, N_clmax=5,
    ):
        """
        """
        # Remove obvious field stars
        pms = np.array([pmRA, pmDE]).T
        pm_rad = spatial.distance.cdist(pms, np.array([vpd_c])).T[0]
        msk1 = (plx > plx_cut) & (pm_rad / (abs(plx) + 0.0001) < v_kms_max)
        msk2 = (plx <= plx_cut) & (pm_rad < pm_max)
        # Stars to keep
        msk = msk1 | msk2

        # Update arrays
        lon, lat, pmRA, pmDE, plx = lon[msk], lat[msk], pmRA[msk],\
            pmDE[msk], plx[msk]
        e_pmRA, e_pmDE, e_plx = e_pmRA[msk], e_pmDE[msk], e_plx[msk]
        # Update indexes of surviving elements
        idx_all = idx_all[msk]

        if msk.sum() < self.N_clust_max * N_clmax:
            return idx_all, lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx

        # Sorted indexes of distances to pms+plx center
        cents_3d = np.array([list(vpd_c) + [plx_c]])
        data_3d = np.array([pmRA, pmDE, plx]).T
        d_pm_plx_idxs = self.get_Nd_dists(cents_3d, data_3d)
        # Indexes of stars to keep and reject based on their distance
        idx_acpt = d_pm_plx_idxs[:int(self.N_clust_max * N_clmax)]

        # Update arrays
        lon, lat, pmRA, pmDE, plx = lon[idx_acpt], lat[idx_acpt],\
            pmRA[idx_acpt], pmDE[idx_acpt], plx[idx_acpt]
        e_pmRA, e_pmDE, e_plx = e_pmRA[idx_acpt], e_pmDE[idx_acpt],\
            e_plx[idx_acpt]
        # Update indexes of surviving elements
        idx_all = idx_all[idx_acpt]

        return idx_all, lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx

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
        N_clust_min=25, prob_cut=.5
    ):
        """
        Estimate the number of cluster members
        """

        idx_survived = self.ripley_survive(
            lon, lat, pmRA, pmDE, plx, vpd_c, plx_c)

        if len(idx_survived) < N_clust_min:
            warnings.warn(
                f"The estimated number of cluster members is <{N_clust_min}",
                UserWarning)
            return N_clust_min, None

        # Filter extra clusters in frame (if any)
        msk = np.array(idx_survived)
        idx_survived = self.filter_cls_in_frame(
            lon[msk], lat[msk], pmRA[msk], pmDE[msk], plx[msk], xy_c, vpd_c,
            plx_c, msk)

        # Filter by KDE
        kde_probs = self.kde_probs(lon, lat, idx_survived, msk)
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

        if N_survived > self.N_clust_max:
            warnings.warn(
                "The estimated number of cluster members is "
                + f">{self.N_clust_max}", UserWarning)
            # Select the maximum number of stars from those closest to the
            # center
            cents = np.array([list(xy_c) + list(vpd_c) + [plx_c]])
            data = np.array([
                lon[idx_survived], lat[idx_survived], pmRA[idx_survived],
                pmDE[idx_survived], plx[idx_survived]]).T
            d_idxs = self.get_Nd_dists(cents, data)
            idx_survived = idx_survived[d_idxs][:self.N_clust_max]
            return self.N_clust_max, None

        return N_survived, idx_survived

    def ripley_survive(
        self, lon, lat, pmRA, pmDE, plx, vpd_c, plx_c, N_clust=50, N_extra=5,
        N_step=10, N_break=5
    ):
        """
        Process data to identify the indexes of stars that survive the
        Ripley's K filter
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
                if N_break_count > N_break:
                    break
                step_old = step
            return idx_survived

        # Select those clusters where the stars are different enough from a
        # random distribution
        idx_survived = core(N_clust)
        if not idx_survived:
            # If the default clustering number did not work, try a few
            # more values with an increasing number of cluster stars
            for _ in range(N_extra):
                N_clust_surv = int(N_clust + (_ + 1) * N_step)
                idx_survived = core(N_clust_surv)
                # Break out when (if) any value selected stars
                if len(idx_survived) > 0:
                    break

        return idx_survived

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
        self, lon, lat, idx_survived, RK_msk, N_min=25, Nst_max=5000
    ):
        """
        Assign probabilities to all stars after generating the KDEs for field
        and member stars. The cluster member probability is obtained applying
        the formula for two mutually exclusive and exhaustive hypotheses.
        """
        if len(idx_survived) < N_min:
            return None

        all_idxs = set(np.arange(len(lon)))
        # Indexes of stars not in idx_survived
        field_idxs = np.array(list(
            all_idxs.symmetric_difference(idx_survived)))

        # IDs of stars that survived Ripley's filter but belong to extra
        # clusters in the frame
        ex_cls_ids = np.array(list(
            set(RK_msk).symmetric_difference(set(idx_survived))))

        # Remove stars that belong to other clusters from the field set
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
        data_3 = np.array([pmRA, pmDE, plx])
        grs = np.random.normal(0., 1., data_3.shape[1])
        data_err = np.array([e_pmRA, e_pmDE, e_plx])
        return data_3 + grs * data_err

    def get_dims_norm(
        self, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c, msk, Nmin=10,
    ):
        """
        Normalize dimensions using twice the median of the selected probable
        members.
        """
        data_5d = np.array([lon, lat, pmRA, pmDE, plx]).T
        cents_5d = np.array([xy_c + vpd_c + [plx_c]])

        if msk is None or len(msk) < Nmin:
            return data_5d, cents_5d

        data_mvd = data_5d - cents_5d
        dims_norm = 2 * np.median(abs(data_mvd[msk]), 0)
        data_norm = data_mvd / dims_norm

        cents_5d = np.array([[0., 0., 0., 0., 0.]])

        return data_norm, cents_5d

    def get_Nd_dists(self, cents, data, dists_flag=False):
        """
        Obtain indexes and distances of stars to the given center
        """
        # Distances to center
        dist_Nd = spatial.distance.cdist(data, cents).T[0]
        if dists_flag:
            return dist_Nd

        # Indexes that sort the distances
        d_idxs = dist_Nd.argsort()

        return d_idxs

    def filter_cls_in_frame(
        self, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c, idx_survived,
        Nmin=10,
    ):
        """
        Filter extra clusters in frame (if any)
        """
        if self.extra_cls_dict['run_flag'] is False:
            return idx_survived

        # Data normalization
        msk = np.full(len(lon), True)
        data, cents = self.get_dims_norm(
            lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c, msk)
        data = data.T

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

        # Never return less than Nmin stars
        if msk_d.sum() < Nmin:
            return idx_survived
        return idx_survived[msk_d]

    def assign_probs(self, N_all, idx_clean, idx_selected, N_runs):
        """
        Assign final probabilities for all stars

        N_all: all stars in input frame
        idx_clean: indexes of stars that survived the removal of 'nans' and
        of stars that are clear field stars
        """
        # Initial -1 probabilities for *all* stars
        probs_final = np.zeros(N_all) - 1

        # Number of processed stars (ie: not rejected as nans)
        N_stars = len(idx_clean)
        # Initial zero probabilities for the processed stars
        probs_all = np.zeros(N_stars)

        if idx_selected:
            # Estimate probabilities as averages of counts
            values, counts = np.unique(idx_selected, return_counts=True)
            probs = counts / N_runs
            # Store probabilities for processed stars
            probs_all[values] = probs
        else:
            warnings.warn("No stars were identified as possible members")

        # Assign the estimated probabilities to the processed stars
        probs_final[idx_clean] = probs_all

        return probs_final

    def warning_on_one_line(
        self, message, category, filename, lineno, file=None, line=None
    ):
        return "{}: {}\n".format(category.__name__, message)
