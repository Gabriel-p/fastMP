
import warnings
import numpy as np
from astropy.stats import RipleysKEstimator
from scipy.spatial.distance import cdist


class fastMP:
    """Summary

    Attributes
    ----------
    CG_data_file : str
        Description
    large_dist_perc : float
        Description
    n_clusters : int
        Description
    N_groups_large_dist : int
        Description
    N_loop : int
        Description
    N_membs : int
        Description
    N_resample : int
        Description
    out_path : str
        Description
    read_PM_center : bool
        Description
    zoom_f : int
        Description
    """

    def __init__(self,
                 method=None,
                 classifDims=5,
                 N_membs=25,
                 n_clusters=1000,
                 N_zoom=10,
                 zoom_f=3,
                 N_groups_large_dist=100,
                 large_dist_perc=0.75,
                 N_resample=10,
                 read_PM_center=None,
                 N_loop=2,
                 N_std=3,
                 vpd_c=None):
        self.method = method
        self.classifDims = classifDims

        self.n_clusters = n_clusters
        self.N_zoom = N_zoom
        self.zoom_f = zoom_f
        self.N_groups_large_dist = N_groups_large_dist
        self.large_dist_perc = large_dist_perc
        self.N_resample = N_resample
        self.read_PM_center = read_PM_center
        self.N_membs = N_membs
        self.N_loop = N_loop
        self.N_std = N_std
        self.vpd_c = vpd_c

    def fit(self, X):
        """
        """
        lon, lat, pmRA, e_pmRA, pmDE, e_pmDE, plx, e_plx = X

        # 
        rads, Kest = self.rkparams(lon, lat)

        from sklearn.naive_bayes import GaussianNB
        from sklearn.calibration import CalibratedClassifierCV
        gnb = GaussianNB()
        gnb_method = CalibratedClassifierCV(gnb, method=self.method)

        data_3 = np.array([pmRA, pmDE, plx])
        data_err = np.array([e_pmRA, e_pmDE, e_plx])

        # idx_selected = []
        probs_all = []
        for _ in range(self.N_resample):
            # Sample data
            s_pmRA, s_pmDE, s_Plx = self.dataSample(data_3, data_err)

            # Estimate VPD center
            if self.vpd_c is None:
                vpd_c = self.centVPD(np.array([s_pmRA, s_pmDE]).T)

            # Estimate Plx center
            plx_c = self.centPlx(s_pmRA, s_pmDE, s_Plx, vpd_c)

            # Obtain indexes of stars closest to the VPD+Plx center
            cent_all = np.array([vpd_c[0], vpd_c[1], plx_c])
            dist_idxs = self.getDists(s_pmRA, s_pmDE, s_Plx, cent_all)

            # Find probable members comparing 
            st_idx = self.getStars(Kest, rads, lon, lat, dist_idxs)

            # Remove outliers
            st_idx = self.sigmaClip(
                s_pmRA, s_pmDE, s_Plx, st_idx, self.N_loop, self.N_std)

            # idx_selected += st_idx

            cl_probs = self.kdeClassif(
                np.array([lon, lat, s_pmRA, s_pmDE, s_Plx]).T, st_idx,
                gnb_method, self.classifDims)

            if cl_probs is not None:
                probs_all.append(cl_probs)

        # # Assign probabilities
        # values, counts = np.unique(idx_selected, return_counts=True)
        # probs = counts / N_resample
        # probs_all = np.zeros(len(lon))
        # probs_all[values] = probs

        probs_all = np.array(probs_all).mean(0)

        return probs_all

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

    def centVPD(self, vpd, N_dim=2):
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
        for _ in range(self.N_zoom):
            N_stars = vpd.shape[0]

            if N_stars < self.N_membs * 4:
                break

            N_bins = max(3, int(N_stars / self.N_membs))
            if N_bins**N_dim > self.n_clusters:
                N_bins = int(self.n_clusters**(1 / N_dim))

            H, edges = np.histogramdd(vpd, bins=(N_bins, N_bins))
            edgx, edgy = edges

            # Find center coordinates
            flat_idx = H.argmax()
            cbx, cby = np.unravel_index(flat_idx, H.shape)
            cx = (edgx[cbx + 1] + edgx[cbx]) / 2.
            cy = (edgy[cby + 1] + edgy[cby]) / 2.

            # This makes the process very non-robust
            # Hg = gaussian_filter(H, sigma=5)
            # flat_idx = Hg.argmax()
            # cbx, cby = np.unravel_index(flat_idx, Hg.shape)
            # cx = (edgx[cbx + 1] + edgx[cbx]) / 2.
            # cy = (edgy[cby + 1] + edgy[cby]) / 2.

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
        N_low1 = N_stars - self.N_groups_large_dist * self.N_membs
        N_low2 = int(N_stars * self.large_dist_perc)
        N_low = max(N_low1, N_low2)
        step_old = -1
        for step in np.arange(N_stars - self.N_membs, N_low, -self.N_membs):
            msk = d_idxs[step:step_old]
            xy = np.array([lon[msk], lat[msk]]).T
            C_s = self.rkfunc(xy, rads, Kest)
            if not np.isnan(C_s):
                C_S_field.append(C_s)
            step_old = step

        # This value is associated to the field stars' distribution. When it
        # is achieved, the block below breaks out. The larger this value,
        # the more restrictive the process becomes.
        C_thresh = np.median(C_S_field)

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

    def sigmaClip(self, s_pmRA, s_pmDE, s_Plx, st_idx, N_loop, N_std):
        """
        Remove outliers in the VPD+Plx space.
        """
        xyz = np.array([s_pmRA[st_idx], s_pmDE[st_idx], s_Plx[st_idx]]).T
        idxs = np.array(st_idx)
        for _ in range(N_loop):
            xy_d = cdist(xyz, xyz.mean(0).reshape(-1, 3)).T[0]
            xy_std = xyz.std(0).mean()
            msk_s3 = xy_d < N_std * xy_std
            xyz, idxs = xyz[msk_s3], idxs[msk_s3]

        return list(idxs)

    def kdeClassif(self, data, st_idx, gnb_method, Ndims, Nst_max=5000):
        """
        Assign probabilities to all stars after generating the KDEs for
        field and member stars. The Cluster probability is obtained
        applying the formula for two mutually exclusive and exhaustive
        hypotheses.
        """
        # Split into the two populations.
        msk = np.array(st_idx)

        labels = np.zeros(len(data))
        labels[msk] = 1

        # from sklearn.linear_model import LogisticRegression
        # logreg = LogisticRegression()
        # clf = logreg.fit(data, labels)
        # probs = clf.predict_proba(data)
        # clf = logreg.fit(data[:, 2:], labels)
        # probs = clf.predict_proba(data[:, 2:])

        clf = gnb_method.fit(data[:, Ndims:], labels)
        probs = clf.predict_proba(data[:, Ndims:])

        return probs.T[1]


        # msk_f = np.arange(0, len(data))
        # f_idx = np.array(list(set(msk_f) - set(msk)))
        # membs_stars = data[st_idx]
        # field_stars = data[f_idx]

        # # To improve the performance, cap the number of stars using a random
        # # selection of 'Nf_max' elements.
        # if field_stars.shape[0] > Nst_max:
        #     idxs = np.arange(field_stars.shape[0])
        #     np.random.shuffle(idxs)
        #     field_stars = field_stars[idxs[:Nst_max]]

        # # Evaluate all stars in both KDEs
        # try:
        #     kd_field = gaussian_kde(field_stars.T)
        #     kd_memb = gaussian_kde(membs_stars.T)

        #     L_memb = kd_memb.evaluate(data.T) + 1e-6
        #     L_field = kd_field.evaluate(data.T)

        #     # with warnings.catch_warnings():
        #     #     warnings.simplefilter("ignore")
        #     # Probabilities for mutually exclusive and exhaustive
        #     # hypotheses
        #     cl_probs = 1. / (1. + (L_field / L_memb))

        # except (np.linalg.LinAlgError, ValueError):
        #     # print("WARNING: Could not perform KDE probabilities estimation")
        #     return None

        # return cl_probs
