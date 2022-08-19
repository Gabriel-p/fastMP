
import warnings
import numpy as np
from astropy.stats import RipleysKEstimator, histogram
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import matplotlib.pyplot as plt

class fastMP:
    """
    Attributes
    ----------
    large_dist_perc : TYPE
        Description
    n_clusters : TYPE
        Description
    N_groups_large_dist : TYPE
        Description
    N_loop : TYPE
        Description
    N_membs : TYPE
        Description
    N_resample : TYPE
        Description
    N_std : TYPE
        Description
    N_zoom : TYPE
        Description
    read_PM_center : TYPE
        Description
    vpd_c : TYPE
        Description
    zoom_f : TYPE
        Description
    """

    def __init__(self,
                 N_membs=25,
                 N_groups_large_dist=100,
                 large_dist_perc=0.75,
                 N_resample=25,
                 N_loop=2,
                 N_std=3,
                 vpd_c=None):
        self.N_groups_large_dist = N_groups_large_dist
        self.large_dist_perc = large_dist_perc
        self.N_resample = N_resample
        self.N_membs = N_membs
        self.N_loop = N_loop
        self.N_std = N_std
        self.vpd_c = vpd_c

    def fit(self, X):
        """
        """
        # Remove outliers and nans
        msk_accpt, X = self.outlRjct(X)

        # Prepare input data
        lon, lat, pmRA, e_pmRA, pmDE, e_pmDE, plx, e_plx = X

        # xy_data = np.array([lon, lat]).T
        # _xrange, _yrange = np.ptp(xy_data, 0)
        # xy = MinMaxScaler().fit(xy_data).transform(xy_data)
        # lon, lat = xy.T

        rads, Kest, C_thresh_N = self.rkparams(lon, lat)
        data_3 = np.array([pmRA, pmDE, plx])
        data_err = np.array([e_pmRA, e_pmDE, e_plx])

        # Estimate VPD center
        vpd_c = self.centVPD(np.array([pmRA, pmDE]).T)
        # Estimate Plx center
        plx_c = self.centPlx(pmRA, pmDE, plx, vpd_c)

        idx_selected = []
        for _ in range(self.N_resample):
            # Sample data
            s_pmRA, s_pmDE, s_Plx = self.dataSample(data_3, data_err)

            # # Estimate VPD center
            # vpd_c = self.centVPD(np.array([s_pmRA, s_pmDE]).T)
            # # Estimate Plx center
            # plx_c = self.centPlx(s_pmRA, s_pmDE, s_Plx, vpd_c)

            # Obtain indexes of stars closest to the VPD+Plx center
            cent_all = np.array([vpd_c[0], vpd_c[1], plx_c])
            dist_idxs, dist_sorted = self.getDists(s_pmRA, s_pmDE, s_Plx, cent_all)

            # fig, ax = plt.subplots()
            # plt.scatter(mag[dist_idxs], dist_sorted)
            # # _, xedges = histogram(dist_sorted, bins='knuth')
            # # _, yedges = histogram(mag, bins='knuth')
            # # ax.set_xticks(xedges, minor=False)
            # # ax.set_yticks(yedges, minor=False)
            # # ax.xaxis.grid(True, which='major')
            # # ax.yaxis.grid(True, which='major')
            # plt.show()
            # breakpoint()

            # Most probable members given their coordinates distribution
            st_idx = self.getStars(_, rads, Kest, C_thresh_N, lon, lat, dist_idxs, dist_sorted)

            # # Remove outliers
            # st_idx = self.sigmaClip(s_pmRA, s_pmDE, s_Plx, st_idx)

            # st_idx = self.KDEprobs(lon, lat, s_pmRA, s_pmDE, s_Plx, st_idx)

            # msk = np.array(list(set(set(np.arange(0, len(lon))) - set(st_idx))))
            # photom = np.array([mag, col]).T
            # field = photom[msk]
            # cluster = photom[st_idx]
            # from scipy import spatial
            # # Find NN_dd nearest neighbors.
            # tree = spatial.cKDTree(cluster)
            # inx_c = tree.query(cluster, k=50)[0]
            # c_dist = np.median(inx_c, 1)
            # msk_all = []
            # for _ in range(100):
            #     i_field = np.random.choice(
            #         np.arange(0, len(field)), len(cluster), replace=False)
            #     s_field = field[i_field]
            #     inx_f = tree.query(s_field, k=50)[0]
            #     # Mean distance to the NN_dd neighbors.
            #     f_dist = np.median(inx_f, 1)
            #     msk = c_dist <= f_dist
            #     msk_all.append(msk)
            #     print(msk.sum())
            # msk_all = np.array(msk_all)
            # msk = msk_all.mean(0) > .1
            # # plt.scatter(s_field.T[1], s_field.T[0], c='grey', alpha=.5)
            # # plt.scatter(cluster.T[1][msk], cluster.T[0][msk], c='g', alpha=.5)
            # # plt.scatter(cluster.T[1][~msk], cluster.T[0][~msk], c='r', alpha=.5)
            # # plt.gca().invert_yaxis()
            # # plt.show()
            # # breakpoint()
            # st_idx = list(np.array(st_idx)[msk])

            idx_selected += st_idx

        probs_final = self.assignProbs(msk_accpt, idx_selected)

        return probs_final

    def outlRjct(self, data, nstd=50):
        """
        Remove outliers and nans
        """
        msk_all = []
        # Process each dimension separately
        for arr in data:
            med, std = np.nanmedian(arr), np.nanstd(arr)
            dmin, dmax = med - nstd * std, med + nstd * std
            msk = (arr > dmin) & (arr < dmax) & ~np.isnan(arr)
            msk_all.append(msk.data)
        # Combine into a single mask
        msk_accpt = np.logical_and.reduce(msk_all)

        return msk_accpt, data.T[msk_accpt].T

    def rkparams(self, lon, lat):
        """
        """
        xmin, xmax = lon.min(), lon.max()
        ymin, ymax = lat.min(), lat.max()
        area = (xmax - xmin) * (ymax - ymin)
        Kest = RipleysKEstimator(
            area=area, x_max=xmax, y_max=ymax, x_min=xmin, y_min=ymin)
        # https://rdrr.io/cran/spatstat/man/Kest.html
        # "For a rectangular window it is prudent to restrict the r values to a
        # maximum of 1/4 of the smaller side length of the rectangle
        # (Ripley, 1977, 1988; Diggle, 1983)"
        lmin = min((xmax - xmin), (ymax - ymin))
        rads = np.linspace(lmin * 0.01, lmin * .25, 10)

        area = (xmax - xmin) * (ymax - ymin)
        C_thresh_N = 1.68 * np.sqrt(area)

        return rads, Kest, C_thresh_N

    def dataSample(self, data_3, data_err):
        """
        Gaussian random sample
        """
        grs = np.random.normal(0., 1., data_3.shape[1]) * 0
        return data_3 + grs * data_err

    def centVPD(self, vpd, N_bins=50, zoom_f=4, N_zoom=10):
        """
        Estimate the center of the cluster in the proper motions space.

        This is the most important function in the algorithm If this function
        fails, everything else will fail too.

        Parameters
        ----------
        vpd : TYPE
            Description
        N_dim : int, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        vpd_mc = self.vpd_c

        for _ in range(N_zoom):
            x, y = vpd.T
            N_stars = len(x)

            if N_stars < 5:
                break

            # Find center coordinates as max density
            H, edgx, edgy = np.histogram2d(x, y, bins=N_bins)
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
            msk = (x < (cx + zoom_f * rx))\
                & (x > (cx - zoom_f * rx))\
                & (y < (cy + zoom_f * ry))\
                & (y > (cy - zoom_f * ry))
            vpd = vpd[msk]

        if vpd_mc is not None:
            cx, cy = cxm, cym

        return (cx, cy)

    def centPlx(self, pmRA, pmDE, Plx, vpd_c, M=2):
        """
        Estimate the center of the cluster in parallax.

        M: multiplier factor that determines how many N_membs are used to
        estimate the parallax center.

        Parameters
        ----------
        pmRA : TYPE
            Description
        pmDE : TYPE
            Description
        Plx : TYPE
            Description
        vpd_c : TYPE
            Description
        M : int, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        # Distance to VPD center
        dist = (pmRA - vpd_c[0])**2 + (pmDE - vpd_c[1])**2
        # Closest stars to center
        idx = dist.argsort()[: M * self.N_membs]  # M IS HARDCODED
        # Center in Plx
        plx_c = np.median(Plx[idx])

        return plx_c

    def getDists(self, pmRA, pmDE, Plx, cent, norm=True):
        """
        Obtain the distances of all stars to the center and sort by the
        smallest value.
        """
        # Normalize
        if norm:
            data = np.concatenate((np.array(
                [pmRA, pmDE, Plx]), cent.reshape(3, -1)), 1).T

            data0 = StandardScaler().fit(data).transform(data)

            data -= data.mean()
            data /= data.std()

            if not np.allclose(data0, data):
                print("problem with scaler")
                breakpoint()

            cent = data[-1]
            pmRA, pmDE, Plx = data[:-1].T

        # 3D distance to the estimated (VPD + Plx) center
        cent = cent.reshape(1, 3)
        # Distance of all the stars to the center
        data = np.array([pmRA, pmDE, Plx]).T
        dist = cdist(data, cent).T[0]
        # Sort by smallest distance
        d_idxs = dist.argsort()
        dist_sorted = dist[d_idxs]

        return d_idxs, dist_sorted

    def getStars(self, i, rads, Kest, C_thresh_N, lon, lat, d_idxs, dist_sorted):
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

        N_clust = self.N_membs + i

        N_stars = len(lon)
        C_thresh = C_thresh_N / N_clust
        # Select those clusters where the stars are different enough from a
        # random distribution
        N_break, step_old = 0, 0
        idxs_survived, last_dists = [], [100000]
        ld_avrg, ld_std, d_avrg = 0, 0, -np.inf
        for step in np.arange(N_clust, N_stars, N_clust):
            msk = d_idxs[step_old:step]
            xy = np.array([lon[msk], lat[msk]]).T

            C_s = self.rkfunc(xy, rads, Kest)
            if not np.isnan(C_s):
                # Cluster survived
                if C_s >= C_thresh:

                    d_avrg = np.median(dist_sorted[step_old:step])
                    ld_avrg, ld_std = np.median(last_dists), np.std(last_dists)
                    last_dists = 1. * dist_sorted[step_old:step]

                    N_break = 0  # Reset
                    idxs_survived += list(msk)
                else:
                    N_break += 1
            # if N_break > 100:
            #     print("N break")
            #     break
            if d_avrg > ld_avrg + 5 * ld_std:
                break
            step_old = step

        return idxs_survived

        # # return idxs_survived
        # if not idxs_survived:
        #     return []

        # if len(idxs_survived) <= self.N_membs:
        #     return idxs_survived

        # # cxy = np.median([[lon[idxs_survived], lat[idxs_survived]]], 2)
        # data = np.array([lon[idxs_survived], lat[idxs_survived]]).T
        # cxy = np.median(data, 0).reshape(1, 2)
        # dist_xy = cdist(data, cxy).T[0]

        # # from sklearn.neighbors import LocalOutlierFactor
        # from sklearn.ensemble import IsolationForest

        # data = np.array([dist_xy, dist_sorted[idxs_survived]])
        # # y_pred = LocalOutlierFactor().fit_predict(data.T)
        # y_pred = IsolationForest().fit_predict(data.T)
        # msk = y_pred > 0

        # # plt.scatter(dist_xy, dist_sorted[idxs_survived])
        # # plt.scatter(dist_xy[msk], dist_sorted[idxs_survived][msk], c='r')
        # # plt.scatter(dist_xy[msk2], dist_sorted[idxs_survived][msk2], marker='x', c='k')
        # # plt.show()

        # return list(np.array(idxs_survived)[msk])

        # rad = np.median(dist) + 2 * np.std(dist)
        # msk = dist < rad
        # return list(np.array(idxs_survived)[msk])

        # N_stars = len(idxs_survived)
        # idxs_survived_ran = []
        # for _ in range(100):
        #     np.random.shuffle(idxs_survived)
        #     step_old = 0
        #     for step in np.arange(self.N_membs, N_stars, self.N_membs):
        #         msk = idxs_survived[step_old:step]
        #         xy = np.array([lon[msk], lat[msk]]).T
        #         C_s = self.rkfunc(xy, rads, Kest)
        #         if not np.isnan(C_s):
        #             if C_s >= C_thresh:
        #                 idxs_survived_ran += list(msk)
        #         step_old = step

        # values, counts = np.unique(idxs_survived_ran, return_counts=True)
        # probs = counts / 100
        # msk = probs > .5

        # return list(values[msk])

        # N_stars = len(lon)
        # # Estimate average C_s of large distance stars
        # C_S_field = []
        # #
        # N_low1 = N_stars - self.N_groups_large_dist * self.N_membs
        # N_low2 = int(N_stars * self.large_dist_perc)
        # N_low = max(N_low1, N_low2)
        # if N_low > N_stars - self.N_membs:
        #     N_low = N_stars - self.N_membs - 1
        # step_old = -1
        # for step in np.arange(N_stars - self.N_membs, N_low, -self.N_membs):
        #     msk = d_idxs[step:step_old]
        #     xy = np.array([lon[msk], lat[msk]]).T
        #     C_s = self.rkfunc(xy, rads, Kest)
        #     if not np.isnan(C_s):
        #         C_S_field.append(C_s)
        #     step_old = step
        # # This value is associated to the field stars' distribution. When it
        # # is achieved, the block below breaks out, having achieved the value
        # # associated to the field distribution. The smaller this value,
        # # the more stars will be included in the members selection returned.
        # if C_S_field:
        #     C_thresh = np.median(C_S_field) + np.std(C_S_field)
        # else:
        #     C_thresh = np.inf
        # # Find Ripley's K value where the stars differ from the estimated
        # # field value
        # step_old = 0
        # for step in np.arange(self.N_membs, N_stars, self.N_membs):
        #     msk = d_idxs[step_old:step]
        #     xy = np.array([lon[msk], lat[msk]]).T
        #     C_s = self.rkfunc(xy, rads, Kest)
        #     if not np.isnan(C_s):
        #         if C_s < C_thresh:
        #             break
        #     step_old = step
        # return list(d_idxs[:step])

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

    # def sigmaClip(self, s_pmRA, s_pmDE, s_Plx, st_idx):
    #     """
    #     Remove outliers in the VPD+Plx space.
    #     """
    #     xyz = np.array([s_pmRA[st_idx], s_pmDE[st_idx], s_Plx[st_idx]]).T
    #     idxs = np.array(st_idx)
    #     for _ in range(self.N_loop):
    #         xy_d = cdist(xyz, xyz.mean(0).reshape(-1, 3)).T[0]
    #         xy_std = xyz.std(0).mean()
    #         msk_s3 = xy_d < self.N_std * xy_std
    #         xyz, idxs = xyz[msk_s3], idxs[msk_s3]

    #     return list(idxs)

    # def KDEprobs(self, lon, lat, s_pmRA, s_pmDE, s_Plx, st_idx, Nst_max=5000):
    #     """
    #     Assign probabilities to all stars after generating the KDEs for field and
    #     member stars. The Cluster probability is obtained applying the formula for
    #     two mutually exclusive and exhaustive hypotheses.
    #     """
    #     from scipy.stats import gaussian_kde

    #     if not st_idx:
    #         return st_idx

    #     # Combine coordinates with the rest of the features.
    #     all_data = np.array([lon, lat, s_pmRA, s_pmDE, s_Plx]).T
    #     # Split into the two populations.
    #     membs_stars = all_data[st_idx]
    #     msk = np.array(list(set(set(np.arange(0, len(lon))) - set(st_idx))))
    #     field_stars = all_data[msk]

    #     # To improve the performance, cap the number of stars using a random
    #     # selection of 'Nf_max' elements.
    #     if field_stars.shape[0] > Nst_max:
    #         idxs = np.arange(field_stars.shape[0])
    #         np.random.shuffle(idxs)
    #         field_stars = field_stars[idxs[:Nst_max]]

    #     # Evaluate all stars in both KDEs
    #     # try:
    #     kd_field = gaussian_kde(field_stars.T)
    #     kd_memb = gaussian_kde(membs_stars.T)

    #     L_memb = kd_memb.evaluate(all_data.T) + 1e-6
    #     L_field = kd_field.evaluate(all_data.T) + 1e-6

    #     # with warnings.catch_warnings():
    #     #     warnings.simplefilter("ignore")
    #     # Probabilities for mutually exclusive and exhaustive
    #     # hypotheses
    #     cl_probs = 1. / (1. + (L_field / L_memb))

    #     # except (np.linalg.LinAlgError, ValueError):
    #     #     pass

    #     msk = cl_probs[st_idx] > 0.5
    #     st_idx = list(np.array(st_idx)[msk])

    #     return st_idx

    def assignProbs(self, msk_accpt, idx_selected):
        """
        """
        N_stars = msk_accpt.sum()
        # Assign probabilities as averages of counts
        values, counts = np.unique(idx_selected, return_counts=True)
        probs = counts / self.N_resample
        probs_all = np.zeros(N_stars)
        probs_all[values] = probs

        # Mark nans with '-1'
        probs_final = np.zeros(len(msk_accpt)) - 1
        probs_final[msk_accpt] = probs_all

        return probs_final
