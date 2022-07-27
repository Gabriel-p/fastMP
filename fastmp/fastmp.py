
import warnings
import numpy as np
from astropy.stats import RipleysKEstimator
from scipy.spatial.distance import cdist


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
                 N_membs=None,
                 N_zoom=10,
                 zoom_f=5,
                 N_groups_large_dist=100,
                 large_dist_perc=0.75,
                 N_resample=100,
                 N_loop=2,
                 N_std=3,
                 vpd_c=None):
        self.N_zoom = N_zoom
        self.zoom_f = zoom_f
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
        # Remove nans
        msk_nonans, X = self.remNans(X)

        # Prepare input data
        lon, lat, pmRA, e_pmRA, pmDE, e_pmDE, plx, e_plx = X
        rads, Kest = self.rkparams(lon, lat)
        data_3 = np.array([pmRA, pmDE, plx])
        data_err = np.array([e_pmRA, e_pmDE, e_plx])

        if self.N_membs is None:
            self.N_membs = min(25, max(int(.1 * len(lon)), 10))

        # # Estimate initial VPD center
        # vpd_c = self.initVPDCenter(Kest, rads, lon, lat, data_3)

        idx_selected = []
        # vpd_c_all = [vpd_c]
        for _ in range(self.N_resample):
            # Sample data
            s_pmRA, s_pmDE, s_Plx = self.dataSample(data_3, data_err)

            # Estimate VPD center
            if self.vpd_c is None:
                vpd_c = self.centVPD(np.array([s_pmRA, s_pmDE]).T)

            # # Estimate VPD center. Use initial estimate for the first run
            # if _ > 0:
            #     vpd_c = self.centVPD(np.array([s_pmRA, s_pmDE]).T)
            #     vpd_c_all.append(vpd_c)
            #     vpd_c = np.median(vpd_c_all, 0)

            # Estimate Plx center
            plx_c = self.centPlx(s_pmRA, s_pmDE, s_Plx, vpd_c)

            # Obtain indexes of stars closest to the VPD+Plx center
            cent_all = np.array([vpd_c[0], vpd_c[1], plx_c])
            dist_idxs = self.getDists(s_pmRA, s_pmDE, s_Plx, cent_all)

            # Find probable members comparing 
            st_idx = self.getStars(Kest, rads, lon, lat, dist_idxs)

            # Remove outliers
            st_idx = self.sigmaClip(s_pmRA, s_pmDE, s_Plx, st_idx)

            idx_selected += st_idx

        # Assign probabilities as averages of counts
        values, counts = np.unique(idx_selected, return_counts=True)
        probs = counts / self.N_resample
        probs_all = np.zeros(len(lon))
        probs_all[values] = probs

        # Mark nans with '-1'
        probs_final = np.zeros(len(probs)) - 1
        probs_final[~msk_nonans] = probs_all

        return probs_final

    def remNans(X):
        """
        Remove nans
        """
        msk_old = [False for _ in range(len(X[0]))]
        for dim in X:
            msk = np.isnan(dim)
            msk = msk_old | msk
        msk_nonans = ~msk
        X_nonans = [_[msk_nonans] for _ in X]

        return msk_nonans, X_nonans

    # def initVPDCenter(self, Kest, rads, lon, lat, data_3):
    #     """
    #     """
    #     pmRA, pmDE, Plx = data_3
    #     # Estimate VPD center
    #     if self.vpd_c is None:
    #         vpd_c = self.centVPD(np.array([pmRA, pmDE]).T)
    #     else:
    #         vpd_c = self.vpd_c

    #     # Estimate Plx center
    #     plx_c = self.centPlx(pmRA, pmDE, Plx, vpd_c)

    #     # Obtain indexes of stars closest to the VPD+Plx center
    #     cent_all = np.array([vpd_c[0], vpd_c[1], plx_c])
    #     dist_idxs = self.getDists(pmRA, pmDE, Plx, cent_all)

    #     # Find probable members comparing 
    #     st_idx = self.getStars(Kest, rads, lon, lat, dist_idxs)

    #     # Remove outliers
    #     st_idx = self.sigmaClip(pmRA, pmDE, Plx, st_idx)

    #     vpd_c = (np.median(pmRA[st_idx]), np.median(pmDE[st_idx]))

    #     return vpd_c

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

        return rads, Kest

    def dataSample(self, data_3, data_err):
        """
        Gaussian random sample
        """
        grs = np.random.normal(0., 1., data_3.shape[1])
        return data_3 + grs * data_err

    def centVPD(self, vpd, n_clusters=1000, N_dim=2):
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
        break_flag = False
        if vpd.shape[0] > self.N_membs * 4 * 2:
            break_flag = True

        for _ in range(self.N_zoom):
            N_stars = vpd.shape[0]

            if break_flag and N_stars < self.N_membs * 4:
                break

            N_bins = max(3, int(N_stars / self.N_membs))
            if N_bins**N_dim > n_clusters:
                N_bins = int(n_clusters**(1 / N_dim))

            H, edges = np.histogramdd(vpd, bins=(N_bins, N_bins))
            edgx, edgy = edges

            # Find center coordinates
            flat_idx = H.argmax()
            cbx, cby = np.unravel_index(flat_idx, H.shape)
            cx = (edgx[cbx + 1] + edgx[cbx]) / 2.
            cy = (edgy[cby + 1] + edgy[cby]) / 2.

            # Zoom in
            x, y = vpd.T
            rx, ry = edgx[1] - edgx[0], edgy[1] - edgy[0]
            msk = (x < (cx + self.zoom_f * rx))\
                & (x > (cx - self.zoom_f * rx))\
                & (y < (cy + self.zoom_f * ry))\
                & (y > (cy - self.zoom_f * ry))
            vpd = vpd[msk]

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
        idx = dist.argsort()[: M * self.N_membs]  # HARDCODED
        # Center in Plx
        plx_c = np.median(Plx[idx])

        return plx_c

    def getDists(self, pmRA, pmDE, Plx, cent):
        """
        Obatin the distances of all stars to the center and sort by the
        smallest value.
        """
        # 3D distance to the estimated (VPD + Plx) center
        cent = cent.reshape(1, 3)
        # Distance of all the stars to the center
        data = np.array([pmRA, pmDE, Plx]).T
        dist = cdist(data, cent).T[0]
        # Sort by smallest distance
        d_idxs = dist.argsort()
        return d_idxs

    def getStars(self, Kest, rads, lon, lat, d_idxs):
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
        # Estimate average C_s of large distance stars
        C_S_field = []

        #
        N_low1 = N_stars - self.N_groups_large_dist * self.N_membs
        N_low2 = int(N_stars * self.large_dist_perc)
        N_low = max(N_low1, N_low2)
        if N_low > N_stars - self.N_membs:
            N_low = N_stars - self.N_membs - 1

        step_old = -1
        for step in np.arange(N_stars - self.N_membs, N_low, -self.N_membs):
            msk = d_idxs[step:step_old]
            xy = np.array([lon[msk], lat[msk]]).T
            C_s = self.rkfunc(xy, rads, Kest)
            if not np.isnan(C_s):
                C_S_field.append(C_s)
            step_old = step

        # This value is associated to the field stars' distribution. When it
        # is achieved, the block below breaks out, having achieved the value
        # associated to the field distribution. The larger this value,
        # the more stars will be included in the members selection returned.
        if C_S_field:
            C_thresh = np.median(C_S_field)
        else:
            C_thresh = np.inf

        # Find Ripley's K value where the stars differ from the estimated
        # field value
        step_old = 0
        for step in np.arange(self.N_membs, N_stars, self.N_membs):
            msk = d_idxs[step_old:step]
            xy = np.array([lon[msk], lat[msk]]).T
            C_s = self.rkfunc(xy, rads, Kest)
            if not np.isnan(C_s):
                if C_s < C_thresh:
                    break
            step_old = step

        return d_idxs[:step]

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

    def sigmaClip(self, s_pmRA, s_pmDE, s_Plx, st_idx):
        """
        Remove outliers in the VPD+Plx space.
        """
        xyz = np.array([s_pmRA[st_idx], s_pmDE[st_idx], s_Plx[st_idx]]).T
        idxs = np.array(st_idx)
        for _ in range(self.N_loop):
            xy_d = cdist(xyz, xyz.mean(0).reshape(-1, 3)).T[0]
            xy_std = xyz.std(0).mean()
            msk_s3 = xy_d < self.N_std * xy_std
            xyz, idxs = xyz[msk_s3], idxs[msk_s3]

        return list(idxs)
