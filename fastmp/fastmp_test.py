
import numpy as np
from scipy.special import gamma
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


class fastMP:

    def __init__(self,
                 N_membs_min=25,
                 N_resample=100,
                 outl_nstd=5,
                 norm_data=True,
                 vpd_c=None):
        self.N_membs_min = N_membs_min
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

        # Pack PMs+Plx data
        data_3 = np.array([pmRA, pmDE, plx])
        data_err = np.array([e_pmRA, e_pmDE, e_plx])

        # Estimate VPD center
        vpd_c = self.centVPD(np.array([pmRA, pmDE]).T)
        # Estimate Plx center
        xy_c, plx_c = self.centXYPlx(lon, lat, pmRA, pmDE, plx, vpd_c)
        print(xy_c, vpd_c, plx_c)

        cents_old = [[xy_c[0], xy_c[1], vpd_c[0], vpd_c[1], plx_c]]
        cent_old = cents_old[0]
        nruns = 0

        idx_selected = []
        for _ in range(self.N_resample + 1):
            # Sample data
            s_pmRA, s_pmDE, s_Plx = self.dataSample(data_3, data_err)

            # Estimate VPD center
            vpd_c = self.centVPD(np.array([s_pmRA, s_pmDE]).T)
            # Estimate coordinates+Plx center
            xy_c, plx_c = self.centXYPlx(
                lon, lat, s_pmRA, s_pmDE, s_Plx, vpd_c)
            cents_old += [[xy_c[0], xy_c[1], vpd_c[0], vpd_c[1], plx_c]]
            cents = np.median(cents_old, 0)
            xy_c = cents[:2]
            vpd_c = cents[2:4]
            plx_c = cents[-1]
            if np.allclose(cent_old, cents, rtol=1e-04):
                break
            cent_old = cents
            print(_, xy_c, vpd_c, plx_c)

            # Obtain indexes and distances of stars to the VPD+Plx center
            dist, dist_idxs, dist_sorted = self.getDists(
                lon, lat, s_pmRA, s_pmDE, s_Plx, xy_c, vpd_c, plx_c)

            Ns = self.N_membs_min
            rads = dist_sorted[::Ns]
            vols = self.HyperSphereVol(rads)
            dens_Ns = Ns / vols
            csum = np.cumsum(dens_Ns)
            aa = csum / csum.max()
            # slopes = aa[1:] - aa[:-1]
            # idx = np.searchsorted(slopes[1:] / slopes[:-1], .9) - 1
            plt.plot(aa)
            plt.show()
            idx = np.searchsorted(aa, .9) - 1
            Ns_idx = Ns * idx

            # plt.plot(slopes)
            # plt.show()
            # breakpoint()

            st_idx = list(dist_idxs[:Ns_idx])

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

    def dataSample(self, data_3, data_err):
        """
        Gaussian random sample
        """
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

    def centXYPlx(self, lon, lat, pmRA, pmDE, Plx, vpd_c, M=4):
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

        # Center X
        H, ex = np.histogram(lon[idx])
        xy_c = [ex[np.argmax(H) + 1]]
        # Center Y
        H, ex = np.histogram(lat[idx])
        xy_c += [ex[np.argmax(H) + 1]]

        return xy_c, plx_c

    def getDists(self, lon, lat, s_pmRA, s_pmDE, s_Plx, xy_c, vpd_c, plx_c):
        """
        Obtain the distances of all stars to the center and sort by the
        smallest value.
        """
        cent = np.array([xy_c[0], xy_c[1], vpd_c[0], vpd_c[1], plx_c])

        # Normalize
        if self.norm_data:
            data = np.concatenate((np.array(
                [lon, lat, s_pmRA, s_pmDE, s_Plx]), cent.reshape(5, -1)), 1).T

            data -= data.mean(0)
            data /= data.std(0)

            cent = data[-1]
            lon, lat, s_pmRA, s_pmDE, s_Plx = data[:-1].T

        # 3D distance to the estimated (VPD + Plx) center
        cent = cent.reshape(1, 5)
        # Distance of all the stars to the center
        data = np.array([lon, lat, s_pmRA, s_pmDE, s_Plx]).T
        dist = cdist(data, cent).T[0]
        # Sort by smallest distance
        d_idxs = dist.argsort()
        dist_sorted = dist[d_idxs]

        return dist, d_idxs, dist_sorted

    def HyperSphereVol(self, rad, dim=5):
        """
        """
        return rad**(dim) * np.pi**(dim / 2) / gamma(dim / 2 + 1)

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
