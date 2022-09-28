
import warnings
import numpy as np
from astropy.stats import RipleysKEstimator
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde


class fastMP:

    def __init__(self,
                 N_resample=100,
                 N_membs_min=25,
                 N_membs_max=50,
                 hardPMRad=3,
                 hardPlxRad=0.2,
                 hardPcRad=15,
                 N_std_d=50,
                 xy_c=None,
                 vpd_c=None,
                 plx_c=None,
                 fixed_centers=False):
        self.N_resample = N_resample
        self.N_membs_min = N_membs_min
        self.N_membs_max = N_membs_max
        self.hardPMRad = hardPMRad
        self.hardPlxRad = hardPlxRad
        self.hardPcRad = hardPcRad
        self.N_std_d = N_std_d
        self.xy_c = xy_c
        self.vpd_c = vpd_c
        self.plx_c = plx_c
        self.fixed_centers = fixed_centers

    def fit(self, X):
        """
        """
        # Remove nans
        msk_accpt, X = self.nanRjct(X)

        # Unpack input data
        if self.N_resample > 0:
            lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx = X
        else:
            lon, lat, pmRA, pmDE, plx = X
            # Dummy variables
            e_pmRA, e_pmDE, e_plx = [np.array([]) for _ in range(3)]

        # Remove the most obvious field stars to speed up the process
        msk_accpt, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c, e_pmRA,\
            e_pmDE, e_plx = self.firstFilter(
                msk_accpt, lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx)

        # Prepare Ripley's K data
        self.rkparams(lon, lat)

        # Estimate the number of members
        N_survived = self.NmembsEstimate(
            lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c)

        xy_c_all, vpd_c_all, plx_c_all = [], [], []
        N_runs, idx_selected = 0, []
        for _ in range(self.N_resample + 1):

            # Sample data (if requested)
            s_pmRA, s_pmDE, s_plx = self.data_sample(
                pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx)

            # Indexes of the sorted 5D distances 5D to the estimated center
            d_idxs = self.get_5d_dists(
                xy_c, vpd_c, plx_c, lon, lat, s_pmRA, s_pmDE, s_plx)

            # Star selection
            st_idx = d_idxs[:N_survived]

            # Filter outlier field stars
            msk = self.PMPlxFilter(
                pmRA[st_idx], pmDE[st_idx], plx[st_idx], vpd_c, plx_c)
            st_idx = st_idx[msk]

            # Re-estimate centers using the selected stars
            xy_c, vpd_c, plx_c = self.get_5D_center(
                lon[st_idx], lat[st_idx], pmRA[st_idx], pmDE[st_idx],
                plx[st_idx])
            #
            # xy_c_all.append(xy_c)
            # vpd_c_all.append(vpd_c)
            # plx_c_all.append(plx_c)
            # xy_c = list(np.median(xy_c_all, 0))
            # vpd_c = list(np.median(vpd_c_all, 0))
            # plx_c = np.median(plx_c_all)

            idx_selected += list(st_idx)
            N_runs += 1

        probs_final = self.assignProbs(msk_accpt, idx_selected, N_runs)

        return probs_final, N_survived

    def nanRjct(self, data):
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

    def firstFilter(
            self, msk_accpt, lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx):
        """
        """
        # Estimate initial center
        xy_c, vpd_c, plx_c = self.get_5D_center(lon, lat, pmRA, pmDE, plx)

        # Remove obvious field stars
        msk = self.PMPlxFilter(pmRA, pmDE, plx, vpd_c, plx_c)
        # Update arrays
        lon, lat, pmRA, pmDE, plx = lon[msk], lat[msk], pmRA[msk], pmDE[msk],\
            plx[msk]
        if self.N_resample > 0:
            e_pmRA, e_pmDE, e_plx = e_pmRA[msk], e_pmDE[msk], e_plx[msk]

        # Update mask of accepted elements
        N_accpt = len(msk_accpt)
        # Indexes accepted by both masks
        idxs = np.arange(0, N_accpt)[msk_accpt][msk]
        msk_accpt = np.full(N_accpt, False)
        msk_accpt[idxs] = True

        return msk_accpt, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c,\
            e_pmRA, e_pmDE, e_plx

    def get_5D_center(self, lon, lat, pmRA, pmDE, plx, M=4):
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

        M: multiplier factor that determines how many N_membs are used to
        estimate the xy+Plx center.

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
        idx = dist.argsort()[:M * self.N_membs_min]  # HARDCODED

        # Estimate 5D center with KDE
        d1, d2, d3, d4, d5 = lon[idx], lat[idx], pmRA[idx], pmDE[idx], plx[idx]
        d1_5 = np.vstack([d1, d2, d3, d4, d5])
        # Define Gaussian KDE
        kde = gaussian_kde(d1_5)
        # Evaluate in the selected closest stars to the center
        density = kde(d1_5)
        # Extract new centers as those associated to the maximum density
        x_c, y_c, pmra_c, pmde_c, plx_c = d1_5[:, density.argmax()]
        vpd_c = (pmra_c, pmde_c)

        # # Estimate 3D center with KDE
        # d1, d2, d5 = lon[idx], lat[idx], plx[idx]
        # d1_5 = np.vstack([d1, d2, d5])
        # # Define Gaussian KDE
        # kde = gaussian_kde(d1_5)
        # # Evaluate in the selected closest stars to the center
        # density = kde(d1_5)
        # # Extract new centers as those associated to the maximum density
        # x_c, y_c, plx_c = d1_5[:, density.argmax()]

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

        if vpd_mc is not None:
            cx, cy = cxm, cym

        return (cx, cy)

    def PMPlxFilter(self, pmRA, pmDE, plx, vpd_c, plx_c):
        """
        Identify obvious field stars
        """
        pms = np.array([pmRA, pmDE]).T
        dist_pm = cdist(pms, np.array([vpd_c])).T[0]

        # Hard PMs limit
        pmRad = max(self.hardPMRad, .5 * abs(vpd_c[0]))  # HARDCODED

        # Hard Plx limit
        d_pc = 1000 / plx_c
        plx_min = 1000 / (d_pc + self.hardPcRad)
        plx_max = max(.05, 1000 / (d_pc - self.hardPcRad))  # HARDCODED
        plxRad_min = max(self.hardPlxRad, plx_c - plx_min)
        plxRad_max = max(self.hardPlxRad, plx_max - plx_c)

        msk = (dist_pm < pmRad) & (plx - plx_c < plxRad_max)\
            & (plx_c - plx < plxRad_min)

        return msk

    def rkparams(self, lon, lat):
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

    def NmembsEstimate(self, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c):
        """
        Estimate the number of cluster members
        """
        d_pm_plx_idxs, d_pm_plx_sorted, d_xy_sorted = self.get_3d_dists(
            lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c)

        xy = np.array([lon, lat]).T
        # N_membs_d = []
        # for N_clust in range(self.N_membs_min, self.N_membs_max):
        #     # Most probable members given their distances
        #     N_survived, d_avrg = self.get_stars(
        #         xy, d_pm_plx_idxs, d_pm_plx_sorted, d_xy_sorted, N_clust)
        #     N_membs_d.append([N_survived, d_avrg])
        # N_membs_d = np.array(N_membs_d).T

        # # The final value is obtained as the one associated to the median
        # # distance where the break happens, also called 'threshold'
        # thresh_med = np.median(N_membs_d[1])
        # idx = np.argmin(abs(N_membs_d[1] - thresh_med))
        # N_survived = int(N_membs_d[0][idx])

        N_survived = self.get_stars(
            xy, d_pm_plx_idxs, d_pm_plx_sorted, d_xy_sorted, self.N_membs_min)
        N_survived = max(N_survived, 10)

        # N_membs_d = []
        # for N_clust in range(self.N_membs_min, self.N_membs_max):
        #     # Most probable members given their distances
        #     N_survived = self.get_stars(
        #         xy, d_pm_plx_idxs, d_pm_plx_sorted, d_xy_sorted, N_clust)
        #     N_membs_d.append(N_survived)
        # N_survived = int(np.median(N_membs_d))

        return N_survived

    def data_sample(self, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx):
        """
        Gaussian random sample
        """
        if self.N_resample == 0:
            return pmRA, pmDE, plx

        data_3 = np.array([pmRA, pmDE, plx])
        grs = np.random.normal(0., 1., data_3.shape[1])
        data_err = np.array([e_pmRA, e_pmDE, e_plx])
        return data_3 + grs * data_err

    def get_3d_dists(self, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c):
        """
        Obtain indexes and distances of stars to the PMs+Plx center
        """
        all_c = np.array([list(vpd_c) + [plx_c]])
        # Distances to this center
        PMsPlx_data = np.array([pmRA, pmDE, plx]).T
        dist_3d = cdist(PMsPlx_data, all_c).T[0]
        # Indexes that sort the distances
        d_pm_plx_idxs = dist_3d.argsort()
        # Sorted distances
        d_pm_plx_sorted = dist_3d[d_pm_plx_idxs]

        # Distances to this center
        dist_xy = cdist(np.array([lon, lat]).T, np.array([xy_c])).T[0]
        d_xy_sorted = dist_xy[d_pm_plx_idxs]

        return d_pm_plx_idxs, d_pm_plx_sorted, d_xy_sorted

    def get_5d_dists(
            self, xy_c, vpd_c, plx_c, lon, lat, s_pmRA, s_pmDE, s_plx):
        """
        Obtain the distances of all stars to the center and sort by the
        smallest value.
        """
        all_data = np.array([lon, lat, s_pmRA, s_pmDE, s_plx]).T
        all_c = np.array([xy_c + vpd_c + [plx_c]])
        all_dist = cdist(all_data, all_c).T[0]
        d_idxs = all_dist.argsort()

        return d_idxs

    def get_stars(self, xy, d_idxs, d_sorted, d_xy_sorted, N_clust):
        """
        """
        N_stars = xy.shape[0]

        # Select those clusters where the stars are different enough from a
        # random distribution
        N_survived, step_old = 0, 0

        C_thresh = self.C_thresh_N / N_clust

        # temp_accpt, temp_rjct = [], []

        # last_dists = [100000]
        # ld_avrg, ld_std, d_avrg = 0, 0, -np.inf
        N_break = 0

        for step in np.arange(N_clust, N_stars, N_clust):

            msk = d_idxs[step_old:step]

            # d_step_range = d_sorted[step_old:step]
            # d_step_range_xy = d_xy_sorted[step_old:step]

            C_s = self.rkfunc(xy[msk], self.rads, self.Kest)
            if not np.isnan(C_s):
                # Cluster survived
                if C_s >= C_thresh:
                    N_survived += N_clust

                    # # Break condition
                    # d_avrg = np.median(d_step_range)
                    # ld_avrg = np.median(last_dists)
                    # ld_std = np.std(last_dists)
                    # last_dists = 1. * d_step_range

                    # if step_old != 0:
                    #     temp_accpt.append([step, C_s, np.std(d_step_range_xy)])
                else:
                    # temp_rjct.append([step, C_s, np.std(d_step_range_xy)])
                    N_break += 1

            # Break condition
            # if d_avrg > ld_avrg + self.N_std_d * ld_std:
            #     break
            if N_break > 10:
                break

            step_old = step

        # import matplotlib.pyplot as plt
        # temp_accpt = np.array(temp_accpt).T
        # temp_rjct = np.array(temp_rjct).T
        # plt.subplot(121)
        # plt.title(f"{N_clust}, {N_survived}")
        # plt.scatter(temp_accpt[0], temp_accpt[1], c='g', alpha=.7)
        # plt.scatter(temp_rjct[0], temp_rjct[1], c='r', alpha=.7)
        # plt.subplot(122)
        # plt.scatter(temp_accpt[0][:-1], temp_accpt[2][:-1], c='g', alpha=.7)
        # plt.scatter(temp_rjct[0][:-1], temp_rjct[2][:-1], c='r', alpha=.7)
        # plt.show()

        return N_survived

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

    def assignProbs(self, msk_accpt, idx_selected, N_runs):
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
