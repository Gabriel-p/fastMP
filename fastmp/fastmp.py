import warnings
import numpy as np
from astropy.stats import RipleysKEstimator
from scipy import spatial
from scipy.stats import gaussian_kde


class fastMP:
    """ """

    def __init__(
        self,
        xy_c=None,
        vpd_c=None,
        plx_c=None,
        fix_N_clust=None,
        fixed_centers=False,
        centers_ex=None,
        N_resample=1000,
        N_min_resample=25,
        N_clust_min=25,
        N_clust_max=5000,
    ):
        self.xy_c = xy_c
        self.vpd_c = vpd_c
        self.plx_c = plx_c
        self.fix_N_clust = fix_N_clust
        self.fixed_centers = fixed_centers
        self.centers_ex = centers_ex
        self.N_resample = int(N_resample)
        self.N_min_resample = int(N_min_resample)
        self.N_clust_min = int(N_clust_min)
        self.N_clust_max = int(N_clust_max)

    def fit(self, X):
        """ """
        warnings.formatwarning = self.warning_on_one_line

        if self.fix_N_clust is not None and self.fix_N_clust is not int:
            raise ValueError(
                "Parameter 'fix_N_clust' must be either 'None' or an integer"
            )

        if self.N_resample <= 2 * self.N_min_resample:  # HARDCODED
            warnings.warn(f"The number of resamples ({self.N_resample}) is small")

        # HARDCODED
        break_count = max(self.N_min_resample, int(self.N_resample * 0.05))

        # Prepare dictionary of parameters for extra clusters in frame (if any)
        self.prep_extra_cl_dict()

        # Remove 'nan' values
        N_all = X.shape[1]
        idx_clean, X_no_nan = self.reject_nans(X)

        # Unpack input data
        lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx = X_no_nan
        # lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx, Gmag = X_no_nan
        # e_pmRA, e_pmDE, e_plx = .005*e_pmRA, .005*e_pmDE, .005*e_plx

        # Estimate initial center
        xy_c, vpd_c, plx_c = self.get_5D_center(lon, lat, pmRA, pmDE, plx)
        cents_init = [xy_c, vpd_c, plx_c]

        # Remove the most obvious field stars to speed up the process
        idx_clean, lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx = self.first_filter(
            idx_clean, vpd_c, plx_c, lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx,
        )
        # idx_clean, lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx, Gmag = self.first_filter(
        #     idx_clean, vpd_c, plx_c, lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx, Gmag
        # )

        # Prepare Ripley's K data
        self.init_ripley(lon, lat)

        # Initiate here as None, will be estimated after the members estimation
        # using a selection of probable members
        self.dims_norm = None

        # Estimate the number of members
        # if type(self.fix_N_clust) in (int, float, np.int64):
        if self.fix_N_clust is None:
            N_survived, st_idx = self.estimate_nmembs(
                lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c
            )
        else:
            N_survived = int(self.fix_N_clust)
            st_idx = None

            cents_3d = np.array([list(vpd_c) + [plx_c]])
            data_3d = np.array([pmRA, pmDE, plx]).T
            # Ordered indexes according to smallest distances to 'cents_3d'
            d_pm_plx_idxs = self.get_Nd_dists(cents_3d, data_3d)
            st_idx = d_pm_plx_idxs[:N_survived]

        # # sort the data:
        # mag_sorted = np.sort(Gmag[st_idx])
        # x = 1. * np.arange(len(st_idx)) / (len(st_idx) - 1)
        # cdf = (x, mag_sorted)

        # Define here 'dims_norm' value used for data normalization
        data_5d = np.array([lon, lat, pmRA, pmDE, plx]).T
        cents_5d = np.array([xy_c + vpd_c + [plx_c]])
        data_mvd = data_5d - cents_5d
        if st_idx is not None:
            self.dims_norm = 2 * np.nanmedian(abs(data_mvd[st_idx]), 0)
        else:
            self.dims_norm = 2 * np.nanmedian(abs(data_mvd), 0)

        # from sklearn import preprocessing
        # if st_idx is not None:
        #     scaler = preprocessing.StandardScaler().fit(data_5d[st_idx])
        # else:
        #     scaler = preprocessing.StandardScaler().fit(data_5d)
        # self.dims_norm = scaler.scale_

        idx_selected = []
        N_runs, N_05_old, prob_old, break_check = 0, 0, 1, 0
        for _ in range(self.N_resample + 1):
            # Sample data
            s_pmRA, s_pmDE, s_plx = self.data_sample(
                pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx
            )

            # Data normalization
            data_5d, cents_5d = self.get_dims_norm(
                lon, lat, s_pmRA, s_pmDE, s_plx, xy_c, vpd_c, plx_c, st_idx
            )

            # Indexes of the sorted 5D distances to the estimated center
            d_idxs = self.get_Nd_dists(cents_5d, data_5d)

            # Star selection
            st_idx = d_idxs[:N_survived]

            # Filter extra clusters in frame (if any)
            st_idx = self.filter_cls_in_frame(
                lon[st_idx],
                lat[st_idx],
                pmRA[st_idx],
                pmDE[st_idx],
                plx[st_idx],
                xy_c,
                vpd_c,
                plx_c,
                st_idx,
            )

            # Re-estimate centers using the selected stars
            if len(st_idx) > self.N_clust_min:
                xy_c, vpd_c, plx_c = self.get_5D_center(
                    lon[st_idx], lat[st_idx], pmRA[st_idx], pmDE[st_idx], plx[st_idx]
                )

                idx_selected += list(st_idx)
                N_runs += 1

            # Convergence check
            if idx_selected:
                counts = np.unique(idx_selected, return_counts=True)[1]
                probs = counts / N_runs
                msk = probs > 0.5
                N_05 = msk.sum()
                if N_05 > 2:
                    prob_mean = np.mean(probs[msk])
                    delta_probs = abs(prob_mean - prob_old)
                    if N_05 == N_05_old or delta_probs < 0.001:  # HARDCODED
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
        # Change '0' probabilities using linear relation
        probs_final = self.probs_0(X, cents_init, probs_final)

        return probs_final, N_survived
        # return probs_final, N_survived, cdf

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
        extra_cls_dict = {"run_flag": False}

        if self.centers_ex is None:
            self.extra_cls_dict = extra_cls_dict
            return

        extra_cls_dict["run_flag"] = True

        # Set the distribution of dimensions
        self.dims_msk = {
            "xy": np.array([0, 1]),
            "pms": np.array([2, 3]),
            "plx": np.array([4]),
            "xy_pms": np.array([0, 1, 2, 3]),
            "xy_plx": np.array([0, 1, 4]),
            "pms_plx": np.array([2, 3, 4]),
            "xy_pms_plx": np.array([0, 1, 2, 3, 4]),
        }

        # Read centers of the extra clusters in frame
        dims = ["xy", "pms", "plx"]
        dim_keys, cents = [], []
        for extra_cl in self.centers_ex:
            dims_ex = extra_cl.keys()
            dims_id, centers = "", []
            for d in dims:
                if d in dims_ex:
                    center = extra_cl[d]
                    if center is not None:
                        dims_id += "_" + d
                        centers += center
            # Store dimensions ids and centers for each extra cluster
            dim_keys.append(dims_id[1:])
            cents.append(centers)

        extra_cls_dict["dim_keys"] = dim_keys
        extra_cls_dict["centers"] = cents

        self.extra_cls_dict = extra_cls_dict

    def reject_nans(self, data):
        """
        Remove nans
        """
        msk_all = []
        # Process each dimension separately
        for arr in data:
            # Identify non-nan data
            msk = ~np.isnan(arr)
            # Keep this non-nan data
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
        if (
            self.fixed_centers is True
            and self.xy_c is not None
            and self.vpd_c is not None
            and self.plx_c is not None
        ):
            x_c, y_c = self.xy_c
            pmra_c, pmde_c = self.vpd_c
            plx_c = self.plx_c
            return [x_c, y_c], [pmra_c, pmde_c], plx_c

        # Re-write if this parameter is given
        # if type(self.fix_N_clust) in (int, float):
        if self.fix_N_clust is not None:
            N_cent = self.fix_N_clust

        # Get filtered stars close to given xy+Plx centers (if any) to use
        # in the PMs center estimation
        pmRA_i, pmDE_i = self.filter_pms_stars(lon, lat, pmRA, pmDE, plx, N_cent)

        # (Re)estimate VPD center
        vpd_c = self.get_pms_center(pmRA_i, pmDE_i)

        # Get N_cent stars closest to vpd_c and given xy and/or plx centers
        lon_i, lat_i, pmRA_i, pmDE_i, plx_i = self.get_stars_close_center(
            lon, lat, pmRA, pmDE, plx, vpd_c, N_cent
        )

        # kNN center
        x_c, y_c, pmra_c, pmde_c, plx_c = self.get_kNN_center(
            np.array([lon_i, lat_i, pmRA_i, pmDE_i, plx_i]).T
        )

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
        """ """
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

    def get_pms_center(self, pmRA, pmDE, N_bins=50, zoom_f=4, N_zoom=10):
        """ """
        vpd_mc = self.vpd_c

        vpd = np.array([pmRA, pmDE]).T

        # Center in PMs space
        cx, cxm = None, None
        for _ in range(N_zoom):
            N_stars = vpd.shape[0]
            if N_stars < self.N_clust_min:
                break

            # Find center coordinates as max density
            x, y = vpd.T
            H, edgx, edgy = np.histogram2d(x, y, bins=N_bins)
            flat_idx = H.argmax()
            cbx, cby = np.unravel_index(flat_idx, H.shape)
            cx = (edgx[cbx + 1] + edgx[cbx]) / 2.0
            cy = (edgy[cby + 1] + edgy[cby]) / 2.0

            # If a manual center was set, use it
            if vpd_mc is not None:
                # Store the auto center for later
                cxm, cym = cx, cy
                # Use the manual to zoom in
                cx, cy = vpd_mc

            # Zoom in
            rx, ry = edgx[1] - edgx[0], edgy[1] - edgy[0]
            msk = (
                (x < (cx + zoom_f * rx))
                & (x > (cx - zoom_f * rx))
                & (y < (cy + zoom_f * ry))
                & (y > (cy - zoom_f * ry))
            )
            vpd = vpd[msk]

        # If a manual center was set
        if vpd_mc is not None:
            # If a better center value could be estimated
            if cxm is not None:
                cx, cy = cxm, cym
            else:
                cx, cy = vpd_mc
                warnings.warn("Could not estimate a better PMs center value")
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

    def get_kNN_center(self, data):
        """
        Estimate 5D center with kNN. Better results are obtained not using
        the parallax data
        """
        data_noplx = data[:, :4]

        tree = spatial.cKDTree(data_noplx)
        inx = tree.query(data_noplx, k=self.N_clust_min + 1)
        NN_dist = inx[0].max(1)
        # Convert to densities
        dens = 1.0 / NN_dist
        # Sort by largest density
        idxs = np.argsort(-dens)

        # Use the star with the largest density
        # cent = np.array([data[idxs[0]]])[0]
        # # Use median of stars with largest densities
        cent = np.median(data[idxs[: self.N_clust_min]], 0)

        x_c, y_c, pmra_c, pmde_c, plx_c = cent
        return x_c, y_c, pmra_c, pmde_c, plx_c

    def first_filter(
        self,
        idx_all,
        vpd_c,
        plx_c,
        lon,
        lat,
        pmRA,
        pmDE,
        plx,
        e_pmRA,
        e_pmDE,
        e_plx,
        # Gmag,
        plx_cut=0.5,
        v_kms_max=5,
        pm_max=3,
        N_times=5,
    ):
        """
        plx_cut: Parallax value that determines the cut in the filter rule
        between using 'v_kms_max' or 'pm_max'
        v_kms_max:
        pm_max:
        N_times: number of times used to multiply 'N_clust_max' to determine
        how many stars to keep
        """
        # Remove obvious field stars
        pms = np.array([pmRA, pmDE]).T
        pm_rad = spatial.distance.cdist(pms, np.array([vpd_c])).T[0]
        msk1 = (plx > plx_cut) & (pm_rad / (abs(plx) + 0.0001) < v_kms_max)
        msk2 = (plx <= plx_cut) & (pm_rad < pm_max)
        # Stars to keep
        msk = msk1 | msk2

        # Update arrays
        lon, lat, pmRA, pmDE, plx = lon[msk], lat[msk], pmRA[msk], pmDE[msk], plx[msk]
        # Gmag = Gmag[msk]
        e_pmRA, e_pmDE, e_plx = e_pmRA[msk], e_pmDE[msk], e_plx[msk]
        # Update indexes of surviving elements
        idx_all = idx_all[msk]

        if msk.sum() < self.N_clust_max * N_times:
            return idx_all, lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx
            # return idx_all, lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx, Gmag

        # Sorted indexes of distances to pms+plx center
        cents_3d = np.array([list(vpd_c) + [plx_c]])
        data_3d = np.array([pmRA, pmDE, plx]).T
        d_pm_plx_idxs = self.get_Nd_dists(cents_3d, data_3d)
        # Indexes of stars to keep and reject based on their distance
        idx_acpt = d_pm_plx_idxs[: int(self.N_clust_max * N_times)]

        # Update arrays
        lon, lat, pmRA, pmDE, plx = (
            lon[idx_acpt],
            lat[idx_acpt],
            pmRA[idx_acpt],
            pmDE[idx_acpt],
            plx[idx_acpt],
        )
        # Gmag = Gmag[idx_acpt]
        e_pmRA, e_pmDE, e_plx = e_pmRA[idx_acpt], e_pmDE[idx_acpt], e_plx[idx_acpt]
        # Update indexes of surviving elements
        idx_all = idx_all[idx_acpt]

        return idx_all, lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx
        # return idx_all, lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx, Gmag

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
            area=area, x_max=xmax, y_max=ymax, x_min=xmin, y_min=ymin
        )

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
        self,
        lon,
        lat,
        pmRA,
        pmDE,
        plx,
        xy_c,
        vpd_c,
        plx_c,
        kde_prob_cut=0.25,
        N_clust=50,
        N_extra=5,
        N_step=10,
        N_break=5,
    ):
        """
        Estimate the number of cluster members
        """
        idx_survived = self.ripley_survive(
            lon,
            lat,
            pmRA,
            pmDE,
            plx,
            xy_c,
            vpd_c,
            plx_c,
            N_clust,
            N_extra,
            N_step,
            N_break,
        )
        N_survived = len(idx_survived)

        if N_survived < self.N_clust_min:
            warnings.warn(
                "The estimated number of cluster members is " + f"<{self.N_clust_min}"
            )
            return self.N_clust_min, None

        # Filter extra clusters in frame (if any)
        msk = np.array(idx_survived)
        idx_survived = self.filter_cls_in_frame(
            lon[msk], lat[msk], pmRA[msk], pmDE[msk], plx[msk], xy_c, vpd_c, plx_c, msk
        )

        # # Filter by (lon, lat) KDE
        # kde_probs = self.kde_probs(lon, lat, idx_survived, msk)
        # if kde_probs is not None:
        #     kde_prob_cut = np.percentile(kde_probs, 25)
        #     msk = kde_probs > kde_prob_cut
        #     idx_survived = idx_survived[msk]

        N_survived = len(idx_survived)

        if N_survived < self.N_clust_min:
            warnings.warn(
                "The estimated number of cluster members is " + f"<{self.N_clust_min}"
            )
            return self.N_clust_min, None

        if N_survived > self.N_clust_max:
            warnings.warn(
                "The estimated number of cluster members is " + f">{self.N_clust_max}"
            )
            # Select the maximum number of stars from those closest to the
            # center
            data_norm, cents_norm = self.get_dims_norm(
                lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c, idx_survived
            )
            d_idxs = self.get_Nd_dists(cents_norm, data_norm[idx_survived])
            idx_survived = idx_survived[d_idxs][: self.N_clust_max]
            return self.N_clust_max, idx_survived

        return N_survived, idx_survived

    # def HDBSCAN_survive(self, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c):
    #     """
    #     """
    #     import hdbscan

    #     data_5d = np.array([lon, lat, pmRA, pmDE, plx]).T
    #     cents_5d = np.array([xy_c + vpd_c + [plx_c]])
    #     data_mvd = data_5d - cents_5d
    #     dims_norm = 2 * np.nanmedian(abs(data_mvd), 0)
    #     data_norm = data_mvd / dims_norm

    #     clusterer = hdbscan.HDBSCAN(min_cluster_size=50) # min_samples=10
    #     cluster_labels = clusterer.fit_predict(data_norm)
    #     if cluster_labels.max() == -1:
    #         clusterer = hdbscan.HDBSCAN(min_cluster_size=25)
    #         cluster_labels = clusterer.fit_predict(data_norm)

    #     if cluster_labels.max() == -1:
    #         idx_survived_HDB = []
    #     else:
    #         d_old = np.inf
    #         for lbl in range(0, cluster_labels.max() + 1):
    #             msk = cluster_labels == lbl
    #             d_cl = abs(np.median(data_norm[msk]))

    #             if d_cl < d_old:
    #                 idx = lbl
    #                 d_old = d_cl
    #         msk = cluster_labels == idx
    #         idx_survived_HDB = np.arange(0, len(lon))[msk]

    #     return idx_survived_HDB

    def ripley_survive(
        self,
        lon,
        lat,
        pmRA,
        pmDE,
        plx,
        xy_c,
        vpd_c,
        plx_c,
        N_clust,
        N_extra,
        N_step,
        N_break,
    ):
        """
        Process data to identify the indexes of stars that survive the
        Ripley's K filter
        """
        cents_3d = np.array([list(vpd_c) + [plx_c]])
        data_3d = np.array([pmRA, pmDE, plx]).T
        # Ordered indexes according to smallest distances to 'cents_3d'
        d_pm_plx_idxs = self.get_Nd_dists(cents_3d, data_3d)
        xy = np.array([lon, lat]).T
        N_stars = xy.shape[0]

        # Identify stars associated to extra clusters in frame (if any)
        msk = np.arange(0, len(lon))
        idx_survived_init = self.filter_cls_in_frame(
            lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c, msk
        )

        def core(N_clust_surv, idx_survived_init):
            """
            This is the core function that estimates the number of members
            based on Ripley's K-function.
            """
            # Define K-function threshold
            C_thresh = self.C_thresh_N / N_clust_surv
            idx_survived = []
            N_break_count, step_old = 0, 0
            for step in np.arange(N_clust_surv, N_stars, N_clust_surv):
                # Ring of stars around the VPD+Plx centers
                msk_ring = d_pm_plx_idxs[step_old:step]
                # Obtain their Ripely K estimator
                C_s = self.rkfunc(xy[msk_ring], self.rads, self.Kest)

                # Remove stars associated to other clusters
                msk_extra_cls = np.isin(msk_ring, idx_survived_init)
                msk = msk_ring[msk_extra_cls]
                # If only a small percentage survived, discard
                if len(msk) < int(N_clust_surv * 0.25):  # HARDCODED
                    C_s = 0

                if not np.isnan(C_s):
                    # This group of stars survived
                    if C_s >= C_thresh:
                        idx_survived += list(msk)
                    else:
                        # Increase break condition
                        N_break_count += 1
                if N_break_count > N_break:
                    break
                step_old = step
            return idx_survived

        # Select those clusters where the stars are different enough from a
        # random distribution
        idx_survived = core(N_clust, idx_survived_init)
        if not idx_survived:
            # If the default clustering number did not work, try a few
            # more values with an increasing number of cluster stars
            for _ in range(N_extra):
                N_clust_surv = int(N_clust + (_ + 1) * N_step)
                idx_survived = core(N_clust_surv, idx_survived_init)
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
            L_t = Kest.Lfunction(xy, rads, mode="translation")

        # Catch all-nans. Avoid 'RuntimeWarning: All-NaN slice encountered'
        if np.isnan(L_t).all():
            C_s = np.nan
        else:
            C_s = np.nanmax(abs(L_t - rads))

        return C_s

    def kde_probs(self, lon, lat, idx_survived, RK_msk):
        """
        Assign probabilities to all stars after generating the KDEs for field
        and member stars. The cluster member probability is obtained applying
        the formula for two mutually exclusive and exhaustive hypotheses.
        """
        if len(idx_survived) < self.N_clust_min:
            return None

        all_idxs = set(np.arange(len(lon)))
        # Indexes of stars not in idx_survived
        field_idxs = np.array(list(all_idxs.symmetric_difference(idx_survived)))

        # IDs of stars that survived Ripley's filter but belong to extra
        # clusters in the frame
        ex_cls_ids = np.array(list(set(RK_msk).symmetric_difference(set(idx_survived))))

        # Remove stars that belong to other clusters from the field set
        if len(ex_cls_ids) > 0:
            field_idxs = np.array(
                list(set(field_idxs).symmetric_difference(ex_cls_ids))
            )

        # Combine coordinates with the rest of the features.
        all_data = np.array([lon, lat]).T
        # Split into the two populations.
        field_stars = all_data[field_idxs]
        membs_stars = all_data[idx_survived]

        # To improve the performance, cap the number of stars using a
        # selection of 'N_clust_max' elements.
        N_field = field_stars.shape[0]
        if N_field > self.N_clust_max:
            idxs = np.arange(N_field)
            # USe step to avoid randomness here
            step = max(1, int(round(N_field / self.N_clust_max)))
            idxs = idxs[::step]
            field_stars = field_stars[idxs[: self.N_clust_max]]

        if len(field_stars) < self.N_clust_min:
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
                cl_probs = 1.0 / (1.0 + (L_field / L_memb))

        except (np.linalg.LinAlgError, ValueError):
            # warnings.warn("Could not perform KDE probabilities estimation")
            cl_probs = np.ones(len(lon))

        return cl_probs

    def data_sample(self, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx):
        """
        Gaussian random sample
        """
        data_3 = np.array([pmRA, pmDE, plx])
        grs = np.random.normal(0.0, 1.0, data_3.shape[1])
        data_err = np.array([e_pmRA, e_pmDE, e_plx])
        return data_3 + grs * data_err

    def get_dims_norm(self, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c, msk):
        """
        Normalize dimensions using twice the median of the selected probable
        members.
        """
        data_5d = np.array([lon, lat, pmRA, pmDE, plx]).T
        cents_5d = np.array([xy_c + vpd_c + [plx_c]])

        if msk is None or len(msk) < self.N_clust_min:
            return data_5d, cents_5d

        data_mvd = data_5d - cents_5d

        if self.dims_norm is None:
            dims_norm = 2 * np.nanmedian(abs(data_mvd[msk]), 0)
            data_norm = data_mvd / dims_norm
            # from sklearn.preprocessing import RobustScaler
            # data_norm = RobustScaler().fit(data_5d).transform(data_5d)
        else:
            data_norm = data_mvd / self.dims_norm

        cents_norm = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])

        return data_norm, cents_norm

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
        self, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c, idx_survived
    ):
        """
        Filter extra clusters in frame (if any)
        """
        # If there are no extra clusters to remove, skip
        if self.extra_cls_dict["run_flag"] is False:
            return idx_survived

        # Add centers to the end of these lists to normalize their values too
        dims_plus_cents = [list(_) for _ in (lon, lat, pmRA, pmDE, plx)]
        for cent in self.extra_cls_dict["centers"]:
            # For each dimension, some won't be present
            for i in range(5):
                try:
                    dims_plus_cents[i].append(cent[i])
                except IndexError:
                    dims_plus_cents[i].append(np.nan)
        lon2, lat2, pmRA2, pmDE2, plx2 = dims_plus_cents

        # Data normalization
        msk = np.full(len(lon2), True)
        data, cents = self.get_dims_norm(
            lon2, lat2, pmRA2, pmDE2, plx2, xy_c, vpd_c, plx_c, msk
        )
        data = data.T

        # Extract normalized centers for extra clusters
        new_cents = []
        for i in range(len(self.extra_cls_dict["centers"]), 0, -1):
            new_cents.append([_ for _ in data[:, -i] if not np.isnan(_)])
        # Remove center values from the 'data' array
        data = data[:, : -len(new_cents)]

        # Find the distances to the center for all the combinations of data
        # dimensions in the extra clusters in frame
        dists = {}
        for dims in list(set(self.extra_cls_dict["dim_keys"])):
            msk = self.dims_msk[dims]
            dists[dims] = self.get_Nd_dists(cents[:, msk], data[msk, :].T, True)

        # Identify stars closer to the cluster's center than the centers of
        # extra clusters
        msk_d = np.full(len(idx_survived), True)
        for i, cent_ex in enumerate(new_cents):
            dims_ex = self.extra_cls_dict["dim_keys"][i]
            msk = self.dims_msk[dims_ex]
            # Distance to this extra cluster's center
            dists_ex = self.get_Nd_dists(np.array([cent_ex]), data[msk, :].T, True)

            # If the distance to the selected cluster center is smaller than
            # the distance to this extra cluster's center, keep the star
            msk_d = msk_d & (dists[dims_ex] <= dists_ex)

        # Never return less than N_clust_min stars
        if msk_d.sum() < self.N_clust_min:
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

    def probs_0(self, X, cents_init, probs_final, p_min=0.1):
        """
        To all stars with prob=0 assign a probability from 0 to p_min
        that follows a linear relation associated to their 5D distance to the
        initial defined center
        """
        # Stars with '0' probabilities
        msk_0 = probs_final == 0.0
        # If no stars with prob=0, nothing to do
        if msk_0.sum() == 0:
            return probs_final

        # Original full data
        lon, lat, pmRA, pmDE, plx = X[:5]
        xy_c, vpd_c, plx_c = cents_init

        # Data normalization for all the stars
        msk = np.full((len(lon)), True)
        data_5d, cents_5d = self.get_dims_norm(
            lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c, msk
        )
        # 5D distances to the estimated center
        dists = self.get_Nd_dists(cents_5d, data_5d, True)

        # Select 'p_min' as the minimum probability between (0., 0.5)
        msk_0_5 = (probs_final > 0.0) & (probs_final < 0.5)
        if msk_0_5.sum() > 1:
            p_min = probs_final[msk_0_5].min()

        # Linear relation for: (0, d_max), (p_min, d_min)
        d_min, d_max = dists[msk_0].min(), dists[msk_0].max()
        m, h = (d_min - d_max) / p_min, d_max
        probs_0 = (dists[msk_0] - h) / m

        # Assign new probabilities to 'msk_0' stars
        probs_final[msk_0] = probs_0

        return probs_final

    def warning_on_one_line(
        self, message, category, filename, lineno, file=None, line=None
    ):
        return "{}: {}\n".format(category.__name__, message)


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_parquet("/home/gabriel/Github/UCC/Q2N/datafiles/teutsch5.parquet")
    # Generate input data array for fastMP
    X = np.array(
        [
            data["GLON"].values,
            data["GLAT"].values,
            data["pmRA"].values,
            data["pmDE"].values,
            data["Plx"].values,
            data["e_pmRA"].values,
            data["e_pmDE"].values,
            data["e_Plx"].values,
        ]
    )
    probs_final, N_survived = fastMP().fit(X)
    breakpoint()
