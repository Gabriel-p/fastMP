
import warnings
import numpy as np
from astropy.stats import RipleysKEstimator
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from scipy import spatial
import matplotlib.pyplot as plt


"""
TODO:

1. Estimate 'plx_rad' from plx error data
2. Remove function to detect 2nd cluster in coords
3. Use only 'N_clust' or (50, 25) in 'estimate_nmembs'?

"""


class fastMP:
    """Summary

    Attributes
    ----------
    coord_dens_perc : TYPE
        Description
    fix_N_clust : TYPE
        Description
    fixed_centers : TYPE
        Description
    N_break : TYPE
        Description
    N_clust : TYPE
        Description
    N_resample : TYPE
        Description
    pc_rad : TYPE
        Description
    plx_c : TYPE
        Description
    plx_rad : float
        Maximum cluster radius in parsec
        This value should be enough for most clusters but there are clusters
        with a more extended region. In these cases this parameter needs to
        be increased.
    PM_cent_rad : TYPE
        Description
    PM_rad : TYPE
        Description
    vpd_c : TYPE
        Description
    xy_c : TYPE
        Description
    """

    def __init__(self,
                 N_resample=100,
                 N_clust=50,
                 fix_N_clust=False,
                 coord_dens_perc=0.05,
                 PM_rad=3,
                 PM_cent_rad=0.5,
                 plx_rad=0.2,
                 pc_rad=15,
                 N_break=5,
                 xy_c=None,
                 vpd_c=None,
                 plx_c=None,
                 fixed_centers=False):
        self.N_resample = N_resample
        self.N_clust = N_clust
        self.fix_N_clust = fix_N_clust
        self.coord_dens_perc = coord_dens_perc
        self.PM_rad = PM_rad
        self.PM_cent_rad = PM_cent_rad
        self.plx_rad = plx_rad
        self.pc_rad = pc_rad
        self.N_break = N_break
        self.xy_c = xy_c
        self.vpd_c = vpd_c
        self.plx_c = plx_c
        self.fixed_centers = fixed_centers

    def fit(self, X):
        """
        """
        # Remove nans
        msk_accpt, X = self.reject_nans(X)

        # Unpack input data
        if self.N_resample > 0:
            lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx, Gmag, BPRP = X
        else:
            lon, lat, pmRA, pmDE, plx, Gmag, BPRP = X
            # Dummy variables
            e_pmRA, e_pmDE, e_plx = [np.array([]) for _ in range(3)]

        # Remove the most obvious field stars to speed up the process
        msk_accpt, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c, e_pmRA,\
            e_pmDE, e_plx, Gmag, BPRP = self.first_filter(
                msk_accpt, lon, lat, pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx, Gmag, BPRP)

        # cx, cy = .5 * (max(lon) + min(lon)), .5 * (max(lat) + min(lat))
        # d_pc = 1000 / plx
        # l_rad, b_rad = np.deg2rad(lon - cx), np.deg2rad(lat - cy)
        # x = d_pc * np.sin(l_rad) * np.cos(b_rad)
        # y = d_pc * np.sin(b_rad)
        # p_pmRA, p_pmDE = d_pc * pmRA, d_pc * pmDE

        # import matplotlib.pyplot as plt
        # plt.subplot(221)
        # plt.scatter(lon, lat)
        # plt.subplot(222)
        # plt.scatter(pmRA, pmDE)
        # plt.subplot(223)
        # plt.scatter(x, y)
        # plt.subplot(224)
        # plt.scatter(p_pmRA, p_pmDE)
        # plt.show()

        # Prepare Ripley's K data
        self.init_ripley(lon, lat)

        # Estimate the number of members
        if self.fix_N_clust is False:
            N_survived = self.estimate_nmembs(
                lon, lat, pmRA, pmDE, plx, vpd_c, plx_c)
        else:
            N_survived = int(self.fix_N_clust)

        N_runs, idx_selected, centers = 0, [], [[], [], []]
        for _ in range(self.N_resample + 1):

            # print(xy_c, vpd_c, plx_c)

            centers[0].append(xy_c)
            centers[1].append(vpd_c)
            centers[2].append(plx_c)

            # Sample data
            s_pmRA, s_pmDE, s_plx = self.data_sample(
                pmRA, pmDE, plx, e_pmRA, e_pmDE, e_plx)

            # Indexes of the sorted 5D distances 5D to the estimated center
            d_idxs = self.get_5d_dists(
                xy_c, vpd_c, plx_c, lon, lat, s_pmRA, s_pmDE, s_plx)

            # Star selection
            st_idx = d_idxs[:N_survived]

            # Filter outlier field stars
            msk = self.filter_pms_plx(
                lon[st_idx], lat[st_idx], pmRA[st_idx], pmDE[st_idx],
                plx[st_idx], xy_c, vpd_c, plx_c)
            st_idx = st_idx[msk]

            # plt.suptitle(len(st_idx))
            # plt.subplot(221)
            # plt.scatter(lon, lat, alpha=.5)
            # plt.scatter(lon[st_idx], lat[st_idx], alpha=.5)
            # plt.subplot(222)
            # plt.scatter(pmRA, pmDE, alpha=.5)
            # plt.scatter(pmRA[st_idx], pmDE[st_idx], alpha=.5)
            # plt.subplot(223)
            # plt.scatter(BPRP, Gmag, alpha=.5)
            # plt.scatter(BPRP[st_idx], Gmag[st_idx], alpha=.5)
            # plt.gca().invert_yaxis()
            # plt.subplot(224)
            # plt.hist(plx, alpha=.5, density=True)
            # plt.hist(plx[st_idx], alpha=.5, density=True)
            # plt.show()

            # Re-estimate centers using the selected stars
            xy_c, vpd_c, plx_c = self.get_5D_center(
                lon[st_idx], lat[st_idx], pmRA[st_idx], pmDE[st_idx],
                plx[st_idx])

            idx_selected += list(st_idx)
            N_runs += 1

        # Check to see if any center moved around
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c_xy, c_vpd, c_plx = [np.array(_) for _ in centers]
            zs_xy = ((c_xy - c_xy.mean()) / c_xy.std()).mean(0)
            zs_vpd = ((c_vpd - c_vpd.mean()) / c_vpd.std()).mean(0)
            zs_plx = ((c_plx - c_plx.mean()) / c_plx.std()).mean()
            if zs_xy.any() > 2:
                warnings.warn("Center in xy varied substantially", UserWarning)
            if zs_vpd.any() > 2:
                warnings.warn("Center in PMs varied substantially",
                              UserWarning)
            if zs_plx.any() > 2:
                warnings.warn("Center in Plx varied substantially",
                              UserWarning)

        probs_final = self.assign_probs(msk_accpt, idx_selected, N_runs)

        # Test for duplicity of clusters or similar problems
        N_peak = min(N_survived, (probs_final > .5).sum())

        NN = 25 # HARDCODED
        if N_peak < 4 * NN:  # HARDCODED:
            return probs_final, N_survived, False

        msk = np.argsort(probs_final[msk_accpt])[::-1][:N_peak]
        #
        xys = np.array([lon[msk], lat[msk]]).T
        dist_xy = cdist(xys, np.array([xy_c])).T[0]
        # Find nearest neighbors.
        tree = spatial.cKDTree(xys)
        inx = tree.query(xys, k=NN)
        # Max distance to the NN neighbors.
        NN_dist = inx[0].max(1)
        # Convert to densities
        dens = 1. / NN_dist
        # Normalize
        dens /= dens.max()

        x_range = np.linspace(0, dist_xy.max(), 10)
        dens_peak_x, x_min, dens_peak_old = [], 0, np.inf
        for x_max in x_range[3:]:
            msk_x = (dist_xy >= x_min) & (dist_xy < x_max)
            if msk_x.sum() > 1:
                dens_peak = np.percentile(dens[msk_x], 95)
                # dens_peak = max(dens[msk_x])
                if dens_peak > dens_peak_old:
                    # print(x_min, x_max, dens_peak_old, dens_peak)
                    dens_peak_x.append((x_max + x_min) * .5)
            x_min = x_max
            dens_peak_old = dens_peak

        mc_flag = False
        if dens_peak_x:
            warnings.warn("Possible multiple clusters in frame", UserWarning)
            mc_flag = True

            print(N_survived, (probs_final>.5).sum())
            plt.subplot(121)
            plt.scatter(lon[msk], lat[msk])
            plt.subplot(122)
            for peak_x in dens_peak_x:
                plt.axvline(peak_x, c='r', ls='-')
            plt.scatter(dist_xy, dens)
            plt.show()
            # breakpoint()

        return probs_final, N_survived, mc_flag

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

        return msk_accpt, data.T[msk_accpt].T

    def first_filter(self, msk_accpt, lon, lat, pmRA, pmDE, plx, e_pmRA,
                     e_pmDE, e_plx, Gmag, BPRP):
        """
        """
        # Estimate initial center
        xy_c, vpd_c, plx_c = self.get_5D_center(lon, lat, pmRA, pmDE, plx)

        # Remove obvious field stars
        msk = self.filter_pms_plx(
            lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c, True)
        # Update arrays
        lon, lat, pmRA, pmDE, plx = lon[msk], lat[msk], pmRA[msk], pmDE[msk],\
            plx[msk]
        if self.N_resample > 0:
            e_pmRA, e_pmDE, e_plx = e_pmRA[msk], e_pmDE[msk], e_plx[msk]
            Gmag, BPRP = Gmag[msk], BPRP[msk]

        # Update mask of accepted elements
        N_accpt = len(msk_accpt)
        # Indexes accepted by both masks
        idxs = np.arange(0, N_accpt)[msk_accpt][msk]
        msk_accpt = np.full(N_accpt, False)
        msk_accpt[idxs] = True

        return msk_accpt, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c,\
            e_pmRA, e_pmDE, e_plx, Gmag, BPRP

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
        idx = dist.argsort()[:M * self.N_clust]  # HARDCODED

        # Estimate 5D center with KDE
        d1, d2, d3, d4, d5 = lon[idx], lat[idx], pmRA[idx], pmDE[idx], plx[idx]
        d1_5 = np.vstack([d1, d2, d3, d4, d5])
        # Define Gaussian KDE
        kde = gaussian_kde(d1_5)
        # Evaluate in the selected closest stars to the center
        try:
            density = kde(d1_5)
            # Extract new centers as those associated to the maximum density
            x_c, y_c, pmra_c, pmde_c, plx_c = d1_5[:, density.argmax()]
        except:
            warnings.warn("Could not estimate the KDE 5D center", UserWarning)
            x_c, y_c, pmra_c, pmde_c, plx_c = self.xy_c[0], self.xy_c[1],\
                vpd_c[0], vpd_c[1], self.plx_c
        vpd_c = (pmra_c, pmde_c)

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
        cx, cxm = None, None
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

        return (cx, cy)

    def filter_pms_plx(
            self, lon, lat, pmRA, pmDE, plx, xy_c, vpd_c, plx_c,
            ff_flag=False, NN=5):
        """
        Identify obvious field stars
        """

        # Don't use on the first filter, or for too few or too many stars
        if ff_flag is False and len(lon) > 10 and len(lon) < 1000:
            # Find the distance to the XY center where the density of stars
            # drops below a fixed threshold. Use this as a value to determine
            # the maximum XY radius.
            xys = np.array([lon, lat]).T
            dist_xy = cdist(xys, np.array([xy_c])).T[0]

            # Find nearest neighbors.
            tree = spatial.cKDTree(xys)
            inx = tree.query(xys, k=NN) # HARDCODED
            # Max distance to the NN neighbors.
            NN_dist = inx[0].max(1)
            # Convert to densities
            dens = 1. / NN_dist
            # Normalize
            dens /= dens.max()
            msk_dens = dens <= self.coord_dens_perc
            if any(msk_dens):
                idx = np.argmax(msk_dens)
            else:
                idx = np.argmax(dist_xy)
            msk_xy = (dist_xy < dist_xy[idx])

            # # if msk_xy.sum() < 5:
            # if not any(msk_dens):
            #     print("error")

            #     print(idx, dist_xy[idx])
            #     ax = plt.subplot(121)
            #     plt.scatter(lon, lat)
            #     circle = plt.Circle(
            #         (np.median(lon), np.median(lat)), dist_xy[idx], color='r',
            #         fill=False)
            #     ax.add_patch(circle)
            #     plt.subplot(122)
            #     plt.scatter(dist_xy, dens)
            #     plt.axhline(0.05, c='r')
            #     plt.axvline(dist_xy[idx], c='r')
            #     plt.show()
            #     breakpoint()

        else:
            msk_xy = np.full(len(pmRA), True)

        # Hard PMs limit
        pms = np.array([pmRA, pmDE]).T
        dist_pm = cdist(pms, np.array([vpd_c])).T[0]
        pmRad = max(self.PM_rad, self.PM_cent_rad * abs(vpd_c[0]))

        # Hard Plx limit
        d_pc = 1000 / plx_c
        plx_min = 1000 / (d_pc + self.pc_rad)
        plx_max = max(.05, 1000 / (d_pc - self.pc_rad))  # HARDCODED
        plxRad_min = max(self.plx_rad, plx_c - plx_min)
        plxRad_max = max(self.plx_rad, plx_max - plx_c)

        msk = (dist_pm < pmRad) & (plx - plx_c < plxRad_max)\
            & (plx_c - plx < plxRad_min) & msk_xy

        return msk

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
        self, lon, lat, pmRA, pmDE, plx, vpd_c, plx_c, prob_cut=.5):
        """
        Estimate the number of cluster members
        """
        d_pm_plx_idxs = self.get_3d_dists(pmRA, pmDE, plx, vpd_c, plx_c)

        xy = np.array([lon, lat]).T
        N_stars = xy.shape[0]
        # C_thresh = self.C_thresh_N / self.N_clust

        # Select those clusters where the stars are different enough from a
        # random distribution
        # N_break_count, step_old, idx_survived = 0, 0, []
        # for step in np.arange(self.N_clust, N_stars, self.N_clust):

        for N_clust_surv in (50, 25):
            C_thresh = self.C_thresh_N / N_clust_surv
            N_break_count, step_old, idx_survived = 0, 0, []
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
                if N_break_count > self.N_break:
                    break
                step_old = step

            if len(idx_survived) > 0:
                break

        N_survived = 0
        if idx_survived:
            cl_probs = self.kde_probs(lon, lat, idx_survived)
            N_survived = int((cl_probs > prob_cut).sum())

        if N_survived < N_clust_surv: #self.N_clust:
            warnings.warn(
                f"The estimated number of cluster members is <{N_clust_surv}",
                UserWarning)
            N_survived = N_clust_surv # self.N_clust

        return N_survived

    def get_3d_dists(self, pmRA, pmDE, plx, vpd_c, plx_c):
        """
        Obtain indexes and distances of stars to the PMs+Plx center
        """
        all_c = np.array([list(vpd_c) + [plx_c]])
        # Distances to this center
        PMsPlx_data = np.array([pmRA, pmDE, plx]).T
        dist_3d = cdist(PMsPlx_data, all_c).T[0]
        # Indexes that sort the distances
        d_pm_plx_idxs = dist_3d.argsort()

        return d_pm_plx_idxs

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

    def kde_probs(self, lon, lat, idx_survived, Nst_max=5000):
        """
        Assign probabilities to all stars after generating the KDEs for field
        and member stars. The cluster member probability is obtained applying
        the formula for two mutually exclusive and exhaustive hypotheses.
        """

        all_idxs = set(np.arange(0, len(lon)))
        field_idxs = np.array(list(
            all_idxs.symmetric_difference(idx_survived)))

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
        if self.N_resample == 0:
            return pmRA, pmDE, plx

        data_3 = np.array([pmRA, pmDE, plx])
        grs = np.random.normal(0., 1., data_3.shape[1])
        data_err = np.array([e_pmRA, e_pmDE, e_plx])
        return data_3 + grs * data_err

    def get_5d_dists(self, xy_c, vpd_c, plx_c, lon, lat, s_pmRA, s_pmDE,
                     s_plx):
        """
        Obtain the distances of all stars to the center and sort by the
        smallest value.
        """
        all_data = np.array([lon, lat, s_pmRA, s_pmDE, s_plx]).T
        all_c = np.array([xy_c + vpd_c + [plx_c]])
        all_dist = cdist(all_data, all_c).T[0]
        d_idxs = all_dist.argsort()

        return d_idxs

    def assign_probs(self, msk_accpt, idx_selected, N_runs):
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
