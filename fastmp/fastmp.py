
import warnings
import numpy as np
from astropy.stats import RipleysKEstimator
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt


class fastMP:

    def __init__(self,
                 N_membs_min=25,
                 N_membs_max=50,
                 # maxPMrad='auto',
                 # maxPlxrad='auto',
                 N_resample=0,
                 N_std_d=5,
                 N_break=20,
                 pmsplxSTDDEV=5,
                 N_bins=50,
                 zoom_f=4,
                 N_zoom=10,
                 vpd_c=None):
        self.N_membs_min = N_membs_min
        self.N_membs_max = N_membs_max
        # self.maxPMrad = maxPMrad
        # self.maxPlxrad = maxPlxrad
        self.N_resample = N_resample
        self.N_std_d = N_std_d
        self.N_break = N_break
        self.pmsplxSTDDEV = pmsplxSTDDEV
        self.N_bins = N_bins
        self.zoom_f = zoom_f
        self.N_zoom = N_zoom
        self.vpd_c = vpd_c

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
            e_pmRA, e_pmDE, e_plx = [np.array([]) for _ in range(3)]

        # Estimate center
        xy_c, vpd_c, plx_c = self.centXYPMPlx(lon, lat, pmRA, pmDE, plx)
        print(xy_c, vpd_c, plx_c)

        # Remove obvious field stars
        msk = self.PMPlxFilter(pmRA, pmDE, plx, vpd_c, plx_c, True)
        lon, lat, pmRA, pmDE, plx = lon[msk], lat[msk], pmRA[msk], pmDE[msk],\
            plx[msk]
        # Update mask of accepted elements
        i1 = np.arange(0, len(msk_accpt))
        i2 = i1[msk_accpt]
        i3 = i2[msk]
        msk_accpt = np.array([False for _ in np.arange(0, len(msk_accpt))])
        msk_accpt[i3] = True

        # Prepare Ripley's K data
        self.rkparams(lon, lat)

        # Pack data
        data_3 = np.array([pmRA, pmDE, plx])
        data_err = np.array([e_pmRA, e_pmDE, e_plx])
        lon_lat = np.array([lon, lat]).T

        N_runs, idx_selected, N_membs = 0, [], []
        for _ in range(self.N_resample + 1):
            # Sample data (if requested)
            s_pmRA, s_pmDE, s_Plx = self.dataSample(data_3, data_err)

            for N_clust in range(self.N_membs_min, self.N_membs_max):

                # Obtain indexes and distances of stars to the center
                d_idxs, d_pm_plx_idxs, d_pm_plx_sorted = self.getDists(
                    lon_lat, s_pmRA, s_pmDE, s_Plx, xy_c, vpd_c, plx_c)

                # Most probable members given their indexed distances
                st_idx = self.getStars(
                    lon, lat, d_pm_plx_idxs, d_pm_plx_sorted, N_clust)
                if not st_idx:
                    continue

                # plt.subplot(221)
                # plt.scatter(lon[st_idx], lat[st_idx], alpha=.5, color='r')
                # plt.subplot(222)
                # # plt.hist(s_pmRA[st_idx], alpha=.5, color='r')
                # plt.scatter(s_pmRA[st_idx], s_pmDE[st_idx], alpha=.5, color='r')
                # plt.subplot(223)
                # # plt.hist(s_pmDE[st_idx], alpha=.5, color='r')
                # plt.scatter(s_pmRA[st_idx], s_Plx[st_idx], alpha=.5, color='r')
                # plt.subplot(224)
                # plt.hist(s_Plx[st_idx], alpha=.5, color='r')

                # Store from 'd_idxs' a number of stars given by 'st_idx'
                st_idx = list(d_idxs)[:len(st_idx)]
                if not st_idx:
                    continue

                # Filter field stars
                msk = self.PMPlxFilter(
                    s_pmRA[st_idx], s_pmDE[st_idx], s_Plx[st_idx], vpd_c,
                    plx_c)
                st_idx = list(np.array(st_idx)[msk])

                # plt.subplot(221)
                # plt.scatter(lon[st_idx], lat[st_idx], alpha=.5, color='b')
                # plt.subplot(222)
                # # plt.hist(s_pmRA[st_idx], alpha=.5, color='b')
                # plt.scatter(s_pmRA[st_idx], s_pmDE[st_idx], alpha=.5, color='b')
                # plt.subplot(223)
                # # plt.hist(s_pmDE[st_idx], alpha=.5, color='b')
                # plt.scatter(s_pmRA[st_idx], s_Plx[st_idx], alpha=.5, color='b')
                # plt.subplot(224)
                # plt.hist(s_Plx[st_idx], alpha=.5, color='b')
                # plt.show()

                # Re-estimate centers using the selected stars
                xy_c, vpd_c, plx_c = self.centXYPMPlx(
                    lon[st_idx], lat[st_idx], s_pmRA[st_idx],
                    s_pmDE[st_idx], s_Plx[st_idx])

                idx_selected += st_idx
                N_runs += 1
                N_membs.append(len(st_idx))

        if N_membs:
            print(np.mean(N_membs), np.std(N_membs))

        probs_final = self.assignProbs(msk_accpt, idx_selected, N_runs)

        return probs_final

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

    def centXYPMPlx(self, lon, lat, pmRA, pmDE, Plx):
        """
        Estimate the center of the cluster.

        This is the most important function in the algorithm If this function
        fails, everything else will fail too.

        M: multiplier factor that determines how many N_membs are used to
        estimate the parallax center.

        """
        vpd_mc = self.vpd_c

        vpd = np.array([pmRA, pmDE]).T
        # Center in PMs space
        for _ in range(self.N_zoom):
            x, y = vpd.T
            N_stars = len(x)

            if N_stars < 5:
                break

            # Find center coordinates as max density
            H, edgx, edgy = np.histogram2d(x, y, bins=self.N_bins)
            flat_idx = H.argmax()
            cbx, cby = np.unravel_index(flat_idx, H.shape)
            cx = (edgx[cbx + 1] + edgx[cbx]) / 2.
            cy = (edgy[cby + 1] + edgy[cby]) / 2.

            # If a manual center was set, use it
            if vpd_mc is not None:
                # Store auto center
                cxm, cym = cx, cy
                # Reset to manual
                cx, cy = vpd_mc

            # Zoom in
            rx, ry = edgx[1] - edgx[0], edgy[1] - edgy[0]
            msk = (x < (cx + self.zoom_f * rx))\
                & (x > (cx - self.zoom_f * rx))\
                & (y < (cy + self.zoom_f * ry))\
                & (y > (cy - self.zoom_f * ry))
            vpd = vpd[msk]

        if vpd_mc is not None:
            cx, cy = cxm, cym
        vpd_c = (cx, cy)

        # Distance to VPD center
        dist = (pmRA - vpd_c[0])**2 + (pmDE - vpd_c[1])**2
        # Closest stars to PMs center
        idx = dist.argsort()[:4 * self.N_membs_min] # HARDCODED

        # XY and Plx centers
        x, y, z = lon[idx], lat[idx], Plx[idx]
        xyz = np.vstack([x, y, z])
        kde = gaussian_kde(xyz)
        density = kde(xyz)
        c1, c2, plx_c = xyz[:, density.argmax()]
        xy_c = [c1, c2]

        return xy_c, vpd_c, plx_c

    def PMPlxFilter(self, pmRA, pmDE, plx, vpd_c, plx_c, firstCall=False):
        """
        Identify obvious field stars
        """
        pms = np.array([pmRA, pmDE]).T
        dist_pm = cdist(pms, np.array([vpd_c])).T[0]
        dist_plx = abs(plx_c - plx)

        pmRad = max(3, .5 * abs(vpd_c[0])) # HARDCODED

        # plxRad = max(.5, .5 * plx_c)
        d_pc = 1000 / plx_c
        plx_1, plx_2 = 1000 / (d_pc + 250), max(.05, 1000 / (d_pc - 250)) # HARDCODED
        plx_r = max(plx_c - plx_1, plx_2 - plx_c)
        plxRad = max(.2, plx_r) # HARDCODED

        # msk = (dist_pm < pmRad) & (dist_plx < plxRad)

    #     return msk

    # def PMPlxFilterSTDDEV(self, pmRA, pmDE, plx, vpd_c, plx_c):
        """
        Identify obvious field stars
        """
        # pms = np.array([pmRA, pmDE]).T
        # dist_pm = cdist(pms, np.array([vpd_c])).T[0]
        # dist_plx = abs(plx_c - plx)

        if firstCall is False:
            pmRad_std = self.pmsplxSTDDEV * np.std(pms, 0).mean()
            plxRad_std = self.pmsplxSTDDEV * np.std(plx)

            pmRad = min(pmRad, pmRad_std)
            plxRad = min(plxRad, plxRad_std)

        #
        msk = (dist_pm < pmRad) & (dist_plx < plxRad)

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
        lmin = min((xmax - xmin), (ymax - ymin))
        rads = np.linspace(lmin * 0.01, lmin * .25, 10)  # HARDCODED

        area = (xmax - xmin) * (ymax - ymin)
        C_thresh_N = 1.68 * np.sqrt(area)  # HARDCODED

        self.rads, self.Kest, self.C_thresh_N = rads, Kest, C_thresh_N

    def dataSample(self, data_3, data_err):
        """
        Gaussian random sample
        """
        if self.N_resample == 0:
            return data_3
        grs = np.random.normal(0., 1., data_3.shape[1])
        return data_3 + grs * data_err

    def getDists(
            self, lon_lat, s_pmRA, s_pmDE, s_Plx, xy_c, vpd_c, plx_c):
        """
        Obtain the distances of all stars to the center and sort by the
        smallest value.
        """
        dist_xy = cdist(lon_lat, np.array([xy_c])).T[0]
        dist_pm = cdist(np.array([s_pmRA, s_pmDE]).T, np.array([vpd_c])).T[0]
        dist_plx = abs(plx_c - s_Plx)

        # The first argsort() returns the indexes ordered by smallest distance,
        # the second argsort() returns those indexes ordered from min to max.
        # The resulting arrays contain the position for each element in the
        # input arrays according to their minimum distances in each dimension
        dxy_idxs = dist_xy.argsort().argsort()
        dpm_idxs = dist_pm.argsort().argsort()
        dplx_idxs = dist_plx.argsort().argsort()
        # Sum the positions for each element for all the dimensions
        idx_sum = dxy_idxs + dpm_idxs + dplx_idxs
        # idx_sum = dpm_idxs + dplx_idxs
        # idx_sum = dxy_idxs + dpm_idxs
        # idx_sum = dxy_idxs + dplx_idxs
        # Sort
        d_idxs = idx_sum.argsort()

        # Sort distances using only PMs+Plx
        dist_sum = dist_pm + dist_plx
        d_pm_plx_idxs = dist_sum.argsort()
        d_pm_plx_sorted = dist_sum[d_pm_plx_idxs]

        # lon, lat = lon_lat.T
        # def plot(N):
        #     msk = d_idxs[:N]
        #     print(np.std(s_pmRA[msk]), np.std(s_pmDE[msk]), np.std(s_Plx[msk]))
        #     plt.subplot(221)
        #     plt.title(N)
        #     plt.scatter(lon, lat, alpha=.25, color='grey')
        #     plt.scatter(lon[msk], lat[msk], marker='x', color='r')
        #     plt.subplot(222)
        #     plt.scatter(s_pmRA, s_pmDE, alpha=.25, color='grey')
        #     plt.scatter(s_pmRA[msk], s_pmDE[msk], marker='x', color='r')
        #     plt.subplot(223)
        #     # plt.scatter(s_Plx, s_pmRA, alpha=.25, color='grey')
        #     # plt.scatter(s_Plx[msk], s_pmRA[msk], marker='x', color='r')
        #     plt.hist(s_Plx, alpha=.5, color='grey', density=True)
        #     plt.hist(s_Plx[msk], alpha=.5, color='b', density=True)
        #     # plt.subplot(224)
        #     # plt.scatter(s_Plx, s_pmDE, alpha=.25, color='grey')
        #     # plt.scatter(s_Plx[msk], s_pmDE[msk], marker='x', color='r')
        #     plt.show()
        # plot(50)
        # breakpoint()

        return d_idxs, d_pm_plx_idxs, d_pm_plx_sorted

    def getStars(self, lon, lat, d_idxs, d_sorted, N_clust):
        """
        Parameters
        ----------
        Kest : TYPE
            Description
        rads : TYPE
            Description
        lon : TYPE
            Description
        lat : TYPE
            Description
        d_idxs : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        N_stars = len(lon)

        # Select those clusters where the stars are different enough from a
        # random distribution
        idxs_survived = []

        C_thresh = self.C_thresh_N / N_clust

        last_dists = [100000]
        N_break_count, step_old, ld_avrg, ld_std, d_avrg = 0, 0, 0, 0, -np.inf
        for step in np.arange(N_clust, N_stars, N_clust):
            msk = d_idxs[step_old:step]
            xy = np.array([lon[msk], lat[msk]]).T

            C_s = self.rkfunc(xy, self.rads, self.Kest)
            if not np.isnan(C_s):
                # Cluster survived
                if C_s >= C_thresh:
                    idxs_survived += list(msk)

                    # Break condition 1
                    d_avrg = np.median(d_sorted[step_old:step])
                    # d_std = np.std(dist_sorted[step_old:step])
                    ld_avrg = np.median(last_dists)
                    ld_std = np.std(last_dists)
                    last_dists = 1. * d_sorted[step_old:step]

                    # Reset break condition 2
                    # N_break_count = 0
                else:
                    # Break condition 2
                    N_break_count += 1

            if N_break_count == self.N_break:
            # if N_break_count * N_clust > len(idxs_survived):
                break
            if d_avrg > ld_avrg + self.N_std_d * ld_std:
                break

            step_old = step

        return idxs_survived

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
        if xy.shape[0] > 5000:
            mode = "none"
        else:
            mode = 'translation'

        # Hide RunTimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            L_t = Kest.Lfunction(xy, rads, mode=mode)

        # Catch all-nans
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
