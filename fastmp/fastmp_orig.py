
import warnings
import numpy as np
from astropy.stats import RipleysKEstimator
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


class fastMP:

    def __init__(self,
                 N_membs_min=25,
                 N_membs_max=50,
                 N_std_d=5,
                 N_break=50,
                 N_resample=0,
                 outl_nstd=5,
                 norm_data=False,
                 vpd_c=None):
        self.N_membs_min = N_membs_min
        self.N_membs_max = N_membs_max
        self.N_std_d = N_std_d
        self.N_break = N_break
        self.N_resample = N_resample
        self.outl_nstd = outl_nstd
        self.norm_data = norm_data
        self.vpd_c = vpd_c

    def fit(self, X):
        """
        """
        # Remove outliers and nans
        msk_accpt, X = self.outlRjct(X)

        # Unpack input data
        if self.N_resample > 0:
            lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx = X
        else:
            lon, lat, pmRA, pmDE, plx = X
            e_pmRA, e_pmDE, e_plx = [np.array([]) for _ in range(3)]

        # Prepare Ripley's K data
        rads, Kest, C_thresh_N = self.rkparams(lon, lat)

        # Pack PMs+Plx data
        data_3 = np.array([pmRA, pmDE, plx])
        data_err = np.array([e_pmRA, e_pmDE, e_plx])

        # Estimate VPD center
        vpd_c = self.centVPD(np.array([pmRA, pmDE]).T)
        # Estimate Plx center
        plx_c = self.centPlx(pmRA, pmDE, plx, vpd_c)
        print(vpd_c, plx_c)

        # cents_old = [[vpd_c[0], vpd_c[1], plx_c]]
        nruns = 0

        idx_selected = []
        vpd_c_all, plx_c_all = [], []
        # for _ in range(self.N_resample + 1):
        for _ in range(self.N_membs_min, self.N_membs_max):
            # Sample data
            s_pmRA, s_pmDE, s_Plx = self.dataSample(data_3, data_err)

            # # Estimate VPD center
            # vpd_c = self.centVPD(np.array([s_pmRA, s_pmDE]).T)
            # # Estimate Plx center
            # plx_c = self.centPlx(s_pmRA, s_pmDE, s_Plx, vpd_c)
            # print(_, vpd_c, plx_c)

            # Obtain indexes and distances of stars to the VPD+Plx center
            dist, dist_idxs, dist_sorted = self.getDists(
                s_pmRA, s_pmDE, s_Plx, vpd_c, plx_c)

            # Most probable members given their coordinates distribution
            st_idx = self.getStars(
                _, rads, Kest, C_thresh_N, lon, lat, dist_idxs, dist_sorted)

            # plt.subplot(231)
            # plt.title(len(st_idx))
            # plt.scatter(lon, lat, marker='.', s=1)
            # plt.scatter(lon[st_idx], lat[st_idx], alpha=.5)
            # plt.subplot(232)
            # plt.scatter(pmRA, pmDE, marker='.')
            # plt.scatter(pmRA[st_idx], pmDE[st_idx], alpha=.5)
            # plt.scatter(*vpd_c, marker='x', c='r')
            # plt.xlim(pmRA[st_idx].min(), pmRA[st_idx].max())
            # plt.ylim(pmDE[st_idx].min(), pmDE[st_idx].max())
            # plt.subplot(233)
            # plt.scatter(plx[st_idx], dist[st_idx])
            # plt.axvline(plx_c, c='r')
            # plt.subplot(234)
            # plt.hist(plx[st_idx])
            # plt.axvline(plx_c, c='r')
            # plt.subplot(235)
            # plt.hist(pmRA[st_idx])
            # plt.axvline(vpd_c[0], c='r')
            # plt.subplot(236)
            # plt.hist(pmDE[st_idx])
            # plt.axvline(vpd_c[1], c='r')
            # plt.show()

            # if st_idx:
            #     ww = abs(np.median(dist_sorted[st_idx]) - dist_sorted[st_idx]) /\
            #         np.median(dist_sorted[st_idx])
            #     ww = 1 / (ww + 1e-6)
            #     pmRA_c = np.average(s_pmRA[st_idx], weights=ww)
            #     pmDE_c = np.average(s_pmDE[st_idx], weights=ww)
            #     # vpd_c = np.array([pmRA_c, pmDE_c])
            #     # plx_c = np.average(s_Plx[st_idx], weights=ww)
            #     vpd_c_all.append([pmRA_c, pmDE_c])
            #     plx_c_all.append(np.average(s_Plx[st_idx], weights=ww))

            #     vpd_c = np.median(vpd_c_all, 0)
            #     plx_c = np.median(plx_c_all)
            #     print(len(st_idx), vpd_c, plx_c)

            idx_selected += st_idx
            nruns += 1

        probs_final = self.assignProbs(msk_accpt, idx_selected, nruns)

        return probs_final

    def outlRjct(self, data):
        """
        Remove outliers and nans
        """
        msk_all = []
        # Process each dimension separately
        for arr in data:
            med, std = np.nanmedian(arr), np.nanstd(arr)
            dmin, dmax = med - self.outl_nstd * std, med + self.outl_nstd * std
            msk = (arr > dmin) & (arr < dmax) & ~np.isnan(arr)
            msk_all.append(msk.data)
        # Combine into a single mask
        msk_accpt = np.logical_and.reduce(msk_all)

        return msk_accpt, data.T[msk_accpt].T

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
        rads = np.linspace(lmin * 0.01, lmin * .25, 10)

        area = (xmax - xmin) * (ymax - ymin)
        C_thresh_N = 1.68 * np.sqrt(area)

        return rads, Kest, C_thresh_N

    def dataSample(self, data_3, data_err):
        """
        Gaussian random sample
        """
        return data_3
        if self.N_resample == 0:
            return data_3
        grs = np.random.normal(0., 1., data_3.shape[1])
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

    def centPlx(self, pmRA, pmDE, Plx, vpd_c, M=4):
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
        idx = dist.argsort()[: M * self.N_membs_min]  # M IS HARDCODED

        # Center in Plx
        # plx_c = np.median(Plx[idx])

        # Center estimated as the  mode
        H, ex = np.histogram(Plx[idx])
        plx_c = ex[np.argmax(H) + 1]

        return plx_c

    def getDists(self, pmRA, pmDE, Plx, vpd_c, plx_c):
        """
        Obtain the distances of all stars to the center and sort by the
        smallest value.
        """
        cent = np.array([vpd_c[0], vpd_c[1], plx_c])

        # Normalize
        if self.norm_data:
            data = np.concatenate((np.array(
                [pmRA, pmDE, Plx]), cent.reshape(3, -1)), 1).T

            data0 = StandardScaler().fit(data).transform(data)

            data -= data.mean(0)
            data /= data.std(0)

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

        return dist, d_idxs, dist_sorted

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
        N_stars = len(lon)

        # Select those clusters where the stars are different enough from a
        # random distribution
        idxs_survived = []

        # for N_clust in range(self.N_membs_min, self.N_membs_max):
        # N_clust = self.N_membs_min
        N_clust = i

        C_thresh = C_thresh_N / N_clust

        last_dists = [100000]
        N_break_count, step_old, ld_avrg, ld_std, d_avrg = 0, 0, 0, 0, -np.inf
        for step in np.arange(N_clust, N_stars, N_clust):
            msk = d_idxs[step_old:step]
            xy = np.array([lon[msk], lat[msk]]).T

            C_s = self.rkfunc(xy, rads, Kest)
            if not np.isnan(C_s):
                # Cluster survived
                if C_s >= C_thresh:
                    idxs_survived += list(msk)

                    # Break condition 1
                    d_avrg = np.median(dist_sorted[step_old:step])
                    # d_std = np.std(dist_sorted[step_old:step])
                    ld_avrg = np.median(last_dists)
                    ld_std = np.std(last_dists)
                    last_dists = 1. * dist_sorted[step_old:step]

                    # Reset break condition 2
                    N_break_count = 0
                else:
                    # Break condition 2
                    N_break_count += 1

            if N_break_count == self.N_break:
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

    def assignProbs(self, msk_accpt, idx_selected, nruns):
        """
        """
        N_stars = msk_accpt.sum()
        # Assign probabilities as averages of counts
        values, counts = np.unique(idx_selected, return_counts=True)

        # N_tot = (self.N_resample + 1) * (self.N_membs_max - self.N_membs_min)
        N_tot = nruns

        probs = counts / N_tot
        probs_all = np.zeros(N_stars)
        probs_all[values] = probs

        # Mark nans with '-1'
        probs_final = np.zeros(len(msk_accpt)) - 1
        probs_final[msk_accpt] = probs_all

        return probs_final
