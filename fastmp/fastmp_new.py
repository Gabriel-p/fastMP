
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde


class fastMP:

    def __init__(self,
                 N_resample=100,
                 hardPMRad=3,
                 hardPlxRad=0.3,
                 pmsplxSTDDEV=5,
                 N_bins=50,
                 zoom_f=4,
                 N_zoom=10,
                 vpd_c=None):
        self.N_resample = N_resample
        self.hardPMRad = hardPMRad
        self.hardPlxRad = hardPlxRad
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

        # Remove obvious field stars
        msk = self.PMPlxFilter(pmRA, pmDE, plx, vpd_c, plx_c, True)
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

        # Pack data
        data_3 = np.array([pmRA, pmDE, plx])
        data_err = np.array([e_pmRA, e_pmDE, e_plx])
        lon_lat = np.array([lon, lat]).T

        # Obtain initial distances of stars to the center (as sum of indexes)
        d_sum_init = self.getDists(
            lon_lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c)
        xy_c0 = 1. * np.array(xy_c)
        vpd_c0 = 1. * np.array(vpd_c)
        plx_c0 = 1. * plx_c

        x_lims = np.percentile(lon, (10, 90))
        y_lims = np.percentile(lat, (10, 90))
        pmRA_lims = np.percentile(pmRA, (10, 90))
        pmDE_lims = np.percentile(pmDE, (10, 90))
        plx_lims = np.percentile(plx, (10, 90))

        # N_grid = 3
        # x_c = np.linspace(*x_lims, N_grid)
        # y_c = np.linspace(*y_lims, N_grid)
        # pmRA_c = np.linspace(*pmRA_lims, N_grid)
        # pmDE_c = np.linspace(*pmDE_lims, N_grid)
        # plx_c = np.linspace(*plx_lims, N_grid)
        # breakpoint()

        all_idxs = np.arange(0, len(lon))
        idx_selected, N_runs = [], 0
        for _ in range(self.N_resample + 1):
            # Sample data (if requested)
            s_pmRA, s_pmDE, s_Plx = self.dataSample(data_3, data_err)

            # New centers
            x_c = np.random.uniform(*x_lims)
            y_c = np.random.uniform(*y_lims)
            pmRA_c = np.random.uniform(*pmRA_lims)
            pmDE_c = np.random.uniform(*pmDE_lims)
            plx_c = np.random.uniform(*plx_lims)
            xy_c = [x_c, y_c]
            vpd_c = (pmRA_c, pmDE_c)
            # print(xy_c, vpd_c, plx_c)

            # Obtain indexes and distances of stars to the center
            d_sum = self.getDists(
                lon_lat, s_pmRA, s_pmDE, s_Plx, xy_c, vpd_c, plx_c)

            # if idx_sum < idx_sum_init it means that this star is not a
            # probable member
            msk = d_sum_init < d_sum

            # msk1 = dxy_idxs_init < dxy_idxs
            # msk2 = dpm_idxs_init < dpm_idxs
            # msk3 = dplx_idxs_init < dplx_idxs
            # msk = msk1 & msk2 & msk3

            # import matplotlib.pyplot as plt
            # # msk = idx_sum_init < idx_sum_init.min() + 1000
            # plt.subplot(221)
            # plt.scatter(lon[~msk], lat[~msk], c='grey', alpha=.3)
            # plt.scatter(lon[msk], lat[msk], c=d_sum_init[msk], alpha=.5)
            # plt.scatter(xy_c0[0], xy_c0[1], c='r', marker='x', s=50)
            # plt.scatter(xy_c[0], xy_c[1], c='k', marker='x', s=50)
            # plt.colorbar()
            # plt.subplot(222)
            # plt.scatter(s_pmRA[~msk], s_pmDE[~msk], c='grey', alpha=.3)
            # plt.scatter(s_pmRA[msk], s_pmDE[msk], c=d_sum_init[msk], alpha=.5)
            # plt.scatter(vpd_c0[0], vpd_c0[1], c='r', marker='x', s=50)
            # plt.scatter(pmRA_c, pmDE_c, c='k', marker='x', s=50)
            # plt.subplot(223)
            # plt.hist(s_Plx[~msk], color='grey', alpha=.5)
            # plt.hist(s_Plx[msk], color='grey', alpha=.5)
            # plt.axvline(plx_c0, c='r')
            # plt.axvline(plx_c, c='k')
            # plt.show()

            # # Re-estimate centers using the selected stars
            # xy_c, vpd_c, plx_c = self.centXYPMPlx(
            #     lon[st_idx], lat[st_idx], s_pmRA[st_idx],
            #     s_pmDE[st_idx], s_Plx[st_idx])

            N_runs += 1
            idx_selected += list(all_idxs[msk])

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
        idx = dist.argsort()[:100] # HARDCODED

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

        # Hard PMs limit
        pmRad = max(self.hardPMRad, .5 * abs(vpd_c[0]))  # HARDCODED

        # # Hard Plx limit
        # d_pc = 1000 / plx_c
        # plx_min = 1000 / (d_pc + self.hardPcRad)
        # plx_max = max(.05, 1000 / (d_pc - self.hardPcRad))  # HARDCODED
        # plx_r = max(plx_c - plx_min, plx_max - plx_c)
        # plxRad = max(self.hardPlxRad, plx_r)

        plxRad = max(self.hardPlxRad, .5 * plx_c)  # HARDCODED

        if firstCall is False:
            pmRad_std = self.pmsplxSTDDEV * np.std(pms, 0).mean()
            pmRad = min(pmRad, pmRad_std)

            plxRad_std = self.pmsplxSTDDEV * np.std(plx)
            # plxRad = max(self.hardPlxRad, min(plxRad, plxRad_std))
            plxRad = max(self.hardPlxRad, plxRad_std)

        msk = (dist_pm < pmRad) & (dist_plx < plxRad)

        return msk

    def dataSample(self, data_3, data_err):
        """
        Gaussian random sample
        """
        return data_3
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
        lon, lat = lon_lat.T
        all_data = np.array([lon, lat, s_pmRA, s_pmDE, s_Plx]).T
        all_c = np.array([xy_c + list(vpd_c) + [plx_c]])




        dist_xy = cdist(lon_lat, np.array([xy_c])).T[0]
        dist_pm = cdist(np.array([s_pmRA, s_pmDE]).T, np.array([vpd_c])).T[0]
        dist_plx = abs(plx_c - s_Plx)

        all_dist = cdist(all_data, all_c).T[0]
        d1_idxs = all_dist.argsort()
        msk1 = d1_idxs[:700]

        dist_xy = cdist(lon_lat, np.array([xy_c])).T[0]
        dist_pm = cdist(np.array([s_pmRA, s_pmDE]).T, np.array([vpd_c])).T[0]
        dist_plx = abs(plx_c - s_Plx)
        dxy_idxs = dist_xy.argsort().argsort()
        dpm_idxs = dist_pm.argsort().argsort()
        dplx_idxs = dist_plx.argsort().argsort()
        idx_sum = dxy_idxs + dpm_idxs + dplx_idxs
        d2_idxs = idx_sum.argsort()
        msk2 = d2_idxs[:700]


        import matplotlib.pyplot as plt
        plt.subplot(221)
        plt.scatter(lon[msk1], lat[msk1], c='b', alpha=.3)
        plt.scatter(lon[msk2], lat[msk2], c='r', alpha=.5)
        plt.colorbar()
        plt.subplot(222)
        plt.scatter(s_pmRA[msk1], s_pmDE[msk1], c='b', alpha=.3)
        plt.scatter(s_pmRA[msk2], s_pmDE[msk2], c='r', alpha=.5)
        plt.subplot(223)
        plt.hist(s_Plx[msk1], color='b', alpha=.5)
        plt.hist(s_Plx[msk2], color='r', alpha=.5)
        plt.show()

        return cdist(all_data, all_c).T[0]


        dist_xy = cdist(lon_lat, np.array([xy_c])).T[0]
        dist_pm = cdist(np.array([s_pmRA, s_pmDE]).T, np.array([vpd_c])).T[0]
        dist_plx = abs(plx_c - s_Plx)

        # return dist_xy, dist_pm, dist_plx

        # The first argsort() returns the indexes ordered by smallest distance,
        # the second argsort() returns those indexes ordered from min to max.
        # The resulting arrays contain the position for each element in the
        # input arrays according to their minimum distances in each dimension
        dxy_idxs = dist_xy.argsort().argsort()
        dpm_idxs = dist_pm.argsort().argsort()
        dplx_idxs = dist_plx.argsort().argsort()
        # Sum the positions for each element for all the dimensions
        idx_sum = dxy_idxs + dpm_idxs + dplx_idxs
        # idx_sum = np.sqrt(dxy_idxs**2 + dpm_idxs**2 + dplx_idxs**2)
        # Sort
        d_idxs = idx_sum.argsort()


        lon, lat = lon_lat.T
        all_data = np.array([lon, lat, s_pmRA, s_pmDE, s_Plx]).T
        all_c = np.array([xy_c + list(vpd_c) + [plx_c]])
        dist_all = cdist(all_data, all_c).T[0]
        d_idxs2 = dist_all.argsort()

        msk1 = d_idxs[:500]
        msk2 = d_idxs2[:500]

        return dxy_idxs, dpm_idxs, dplx_idxs

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
