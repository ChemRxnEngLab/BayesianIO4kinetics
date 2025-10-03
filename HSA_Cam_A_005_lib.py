from scipy.integrate import odeint
import pytensor as pt
from pytensor.compile.ops import as_op
import numpy as np
from scipy.constants import R
from numba import njit


class Chem:

    @staticmethod
    @njit
    def reaction(y, t, params):
        V_r = 1e-3  # m3
        n_g_con, c_Al, c_Bl, c_Cl = y[0], y[1], y[2], y[3]
        p = params[0]
        T = params[1]
        kla = params[2]
        A0 = params[3]
        EA = params[4]

        k = A0 * np.exp(-EA / R * (1 / T - 1 / 298))
        k_H = 7.8e-6 * np.exp(500 * (1 / T - 1 / 298))

        dn_g_con_dt = kla * (k_H * p - c_Al) * V_r
        dc_Al_dt = kla * (k_H * p - c_Al) - k * c_Al * c_Bl
        dc_Bl_dt = -k * c_Al * c_Bl
        dc_Cl_dt = k * c_Al * c_Bl

        dydt = [
            dn_g_con_dt,
            dc_Al_dt,
            dc_Bl_dt,
            dc_Cl_dt,
        ]
        return dydt

    @staticmethod
    @as_op(
        itypes=[pt.tensor.fvector, pt.tensor.dvector, pt.tensor.dvector],
        otypes=[pt.tensor.dmatrix],
    )
    def pt_forward(y0, t, theta):
        return odeint(func=Chem.reaction, y0=y0, t=t, args=(theta,))
