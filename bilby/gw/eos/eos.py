import os
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from numpy.polynomial import chebyshev as cheb
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp

from .tov_solver import IntegrateTOV
from ...core import utils

C_SI = utils.speed_of_light  # m/s
C_CGS = C_SI * 100.0
G_SI = utils.gravitational_constant  # m^3 kg^-1 s^-2
MSUN_SI = utils.solar_mass  # Kg

# Stores conversions from geometerized to cgs or si unit systems
conversion_dict = {
    "pressure": {
        "cgs": C_SI**4.0 / G_SI * 10.0,
        "si": C_SI**4.0 / G_SI,
        "geom": 1.0,
    },
    "energy_density": {
        "cgs": C_SI**4.0 / G_SI * 10.0,
        "si": C_SI**4.0 / G_SI,
        "geom": 1.0,
    },
    "density": {
        "cgs": C_SI**2.0 / G_SI / 1000.0,
        "si": C_SI**2.0 / G_SI,
        "geom": 1.0,
    },
    "pseudo_enthalpy": {"dimensionless": 1.0},
    "mass": {
        "g": C_SI**2.0 / G_SI * 1000,
        "kg": C_SI**2.0 / G_SI,
        "geom": 1.0,
        "m_sol": C_SI**2.0 / G_SI / MSUN_SI,
    },
    "radius": {"cm": 100.0, "m": 1.0, "km": 0.001},
    "tidal_deformability": {"geom": 1.0},
}


# construct dictionary of pre-shipped EOS pressure denstity table
path_to_eos_tables = os.path.join(os.path.dirname(__file__), "eos_tables")
list_of_eos_tables = os.listdir(path_to_eos_tables)
valid_eos_files = [i for i in list_of_eos_tables if "LAL" in i]
valid_eos_file_paths = [
    os.path.join(path_to_eos_tables, filename) for filename in valid_eos_files
]
valid_eos_names = [
    i.split("_", maxsplit=1)[-1].strip(".dat") for i in valid_eos_files
]
valid_eos_dict = dict(zip(valid_eos_names, valid_eos_file_paths))


class TabularEOS(object):
    """
    Given a valid eos input format, such as 2-D array, an ascii file, or a string, parse, and interpolate

    Parameters
    ==========
    eos: (numpy.ndarray, str, ASCII TABLE)
        if `numpy.ndarray` then user supplied pressure-density 2D numpy array.
        if `str` then given a valid eos name, relevant preshipped ASCII table will be loaded
        if ASCII TABLE then given viable file extensions, which include .txt,.dat, etc (np.loadtxt used),
        read in pressure density from file.
    sampling_flag: bool
        Do you plan on sampling the parameterized EOS? Highly recommended. Defaults to False.
    warning_flag: bool
        Keeps track of status of various physical checks on EoS.

    Attributes
    ==========
    msg: str
        Human readable string describing the exception.
    code: int
        Exception error code.
    """

    def __init__(self, eos, sampling_flag=False, warning_flag=False):
        from scipy.integrate import cumulative_trapezoid

        self.sampling_flag = sampling_flag
        self.warning_flag = warning_flag

        if isinstance(eos, str):
            if eos in valid_eos_dict.keys():
                table = np.loadtxt(valid_eos_dict[eos])
            else:
                table = np.loadtxt(eos)
        elif isinstance(eos, np.ndarray):
            table = eos
        else:
            raise ValueError(
                "eos provided is invalid type please supply a str name, str path to ASCII file, "
                "or a numpy array"
            )

        table = self.__remove_leading_zero(table)

        # all have units of m^-2
        self.pressure = table[:, 0]
        self.energy_density = table[:, 1]

        self.minimum_pressure = min(self.pressure)
        self.minimum_energy_density = min(self.energy_density)
        if (
            not self.check_monotonicity() and self.sampling_flag
        ) or self.warning_flag:
            self.warning_flag = True
        else:
            integrand = self.pressure / (self.energy_density + self.pressure)
            self.pseudo_enthalpy = (
                cumulative_trapezoid(
                    integrand, np.log(self.pressure), initial=0
                )
                + integrand[0]
            )

            self.interp_energy_density_from_pressure = CubicSpline(
                np.log10(self.pressure),
                np.log10(self.energy_density),
            )

            self.interp_energy_density_from_pseudo_enthalpy = CubicSpline(
                np.log10(self.pseudo_enthalpy), np.log10(self.energy_density)
            )

            self.interp_pressure_from_pseudo_enthalpy = CubicSpline(
                np.log10(self.pseudo_enthalpy), np.log10(self.pressure)
            )

            self.interp_pseudo_enthalpy_from_energy_density = CubicSpline(
                np.log10(self.energy_density), np.log10(self.pseudo_enthalpy)
            )

            self.__construct_all_tables()

            self.minimum_pseudo_enthalpy = min(self.pseudo_enthalpy)
            if not self.check_causality() and self.sampling_flag:
                self.warning_flag = True

    def __remove_leading_zero(self, table):
        """
        For interpolation of lalsimulation tables;
        loglog interpolation breaks if the first entries are 0s
        """

        if table[0, 0] == 0.0 or table[0, 1] == 0.0:
            return table[1:, :]

        else:
            return table

    def energy_from_pressure(self, pressure, interp_type="CubicSpline"):
        """
        Find value of energy_from_pressure
        as in lalsimulation, return e = K * p**(3./5.) below min pressure

        Parameters
        ==========
        pressure: float
            pressure in geometerized units.
        interp_type: str
            String specifying which interpolation type to use.
            Currently implemented: 'CubicSpline', 'linear'.
        energy_density: float
            energy-density in geometerized units.
        """
        pressure = np.atleast_1d(pressure)
        energy_returned = np.zeros(pressure.size)
        indices_less_than_min = np.nonzero(pressure < self.minimum_pressure)
        indices_greater_than_min = np.nonzero(
            pressure >= self.minimum_pressure
        )

        # We do this special for less than min pressure
        energy_returned[indices_less_than_min] = 10 ** (
            np.log10(self.energy_density[0])
            + (3.0 / 5.0)
            * (
                np.log10(pressure[indices_less_than_min])
                - np.log10(self.pressure[0])
            )
        )

        if interp_type == "CubicSpline":
            energy_returned[indices_greater_than_min] = (
                10.0
                ** self.interp_energy_density_from_pressure(
                    np.log10(pressure[indices_greater_than_min])
                )
            )
        elif interp_type == "linear":
            energy_returned[indices_greater_than_min] = np.interp(
                pressure[indices_greater_than_min],
                self.pressure,
                self.energy_density,
            )
        else:
            raise ValueError(
                "Interpolation scheme must be linear or CubicSpline"
            )

        if energy_returned.size == 1:
            return energy_returned[0]
        else:
            return energy_returned

    def pressure_from_pseudo_enthalpy(
        self, pseudo_enthalpy, interp_type="CubicSpline"
    ):
        """
        Find p(h)
        as in lalsimulation, return p = K * h**(5./2.) below min enthalpy

        :param pseudo_enthalpy (`float`): Dimensionless pseudo-enthalpy.
        :interp_type (`str`): String specifying interpolation type.
                              Current implementations are 'CubicSpline', 'linear'.

        :return pressure (`float`): pressure in geometerized units.
        """
        pseudo_enthalpy = np.atleast_1d(pseudo_enthalpy)
        pressure_returned = np.zeros(pseudo_enthalpy.size)
        indices_less_than_min = np.nonzero(
            pseudo_enthalpy < self.minimum_pseudo_enthalpy
        )
        indices_greater_than_min = np.nonzero(
            pseudo_enthalpy >= self.minimum_pseudo_enthalpy
        )

        pressure_returned[indices_less_than_min] = 10.0 ** (
            np.log10(self.pressure[0])
            + 2.5
            * (
                np.log10(pseudo_enthalpy[indices_less_than_min])
                - np.log10(self.pseudo_enthalpy[0])
            )
        )

        if interp_type == "CubicSpline":
            pressure_returned[indices_greater_than_min] = (
                10.0
                ** self.interp_pressure_from_pseudo_enthalpy(
                    np.log10(pseudo_enthalpy[indices_greater_than_min])
                )
            )
        elif interp_type == "linear":
            pressure_returned[indices_greater_than_min] = np.interp(
                pseudo_enthalpy[indices_greater_than_min],
                self.pseudo_enthalpy,
                self.pressure,
            )
        else:
            raise ValueError(
                "Interpolation scheme must be linear or CubicSpline"
            )

        if pressure_returned.size == 1:
            return pressure_returned[0]
        else:
            return pressure_returned

    def energy_density_from_pseudo_enthalpy(
        self, pseudo_enthalpy, interp_type="CubicSpline"
    ):
        """
        Find energy_density_from_pseudo_enthalpy(pseudo_enthalpy)
        as in lalsimulation, return e = K * h**(3./2.) below min enthalpy

        :param pseudo_enthalpy (`float`): Dimensionless pseudo-enthalpy.
        :param interp_type (`str`): String specifying interpolation type.
                                    Current implementations are 'CubicSpline', 'linear'.

        :return energy_density (`float`): energy-density in geometerized units.
        """
        pseudo_enthalpy = np.atleast_1d(pseudo_enthalpy)
        energy_returned = np.zeros(pseudo_enthalpy.size)
        indices_less_than_min = np.nonzero(
            pseudo_enthalpy < self.minimum_pseudo_enthalpy
        )
        indices_greater_than_min = np.nonzero(
            pseudo_enthalpy >= self.minimum_pseudo_enthalpy
        )

        energy_returned[indices_less_than_min] = 10 ** (
            np.log10(self.energy_density[0])
            + 1.5
            * (
                np.log10(pseudo_enthalpy[indices_less_than_min])
                - np.log10(self.pseudo_enthalpy[0])
            )
        )
        if interp_type == "CubicSpline":
            x = np.log10(pseudo_enthalpy[indices_greater_than_min])
            energy_returned[indices_greater_than_min] = (
                10 ** self.interp_energy_density_from_pseudo_enthalpy(x)
            )
        elif interp_type == "linear":
            energy_returned[indices_greater_than_min] = np.interp(
                pseudo_enthalpy[indices_greater_than_min],
                self.pseudo_enthalpy,
                self.energy_density,
            )
        else:
            raise ValueError(
                "Interpolation scheme must be linear or CubicSpline"
            )

        if energy_returned.size == 1:
            return energy_returned[0]
        else:
            return energy_returned

    def pseudo_enthalpy_from_energy_density(
        self, energy_density, interp_type="CubicSpline"
    ):
        """
        Find h(epsilon)
        as in lalsimulation, return h = K * e**(2./3.) below min enthalpy

        :param energy_density (`float`): energy-density in geometerized units.
        :param interp_type (`str`): String specifying interpolation type.
                                    Current implementations are 'CubicSpline', 'linear'.

        :return pseudo_enthalpy (`float`): Dimensionless pseudo-enthalpy.
        """
        energy_density = np.atleast_1d(energy_density)
        pseudo_enthalpy_returned = np.zeros(energy_density.size)
        indices_less_than_min = np.nonzero(
            energy_density < self.minimum_energy_density
        )
        indices_greater_than_min = np.nonzero(
            energy_density >= self.minimum_energy_density
        )

        pseudo_enthalpy_returned[indices_less_than_min] = 10 ** (
            np.log10(self.pseudo_enthalpy[0])
            + (2.0 / 3.0)
            * (
                np.log10(energy_density[indices_less_than_min])
                - np.log10(self.energy_density[0])
            )
        )

        if interp_type == "CubicSpline":
            x = np.log10(energy_density[indices_greater_than_min])
            pseudo_enthalpy_returned[indices_greater_than_min] = (
                10 ** self.interp_pseudo_enthalpy_from_energy_density(x)
            )
        elif interp_type == "linear":
            pseudo_enthalpy_returned[indices_greater_than_min] = np.interp(
                energy_density[indices_greater_than_min],
                self.energy_density,
                self.pseudo_enthalpy,
            )
        else:
            raise ValueError(
                "Interpolation scheme must be linear or CubicSpline"
            )

        if pseudo_enthalpy_returned.size == 1:
            return pseudo_enthalpy_returned[0]
        else:
            return pseudo_enthalpy_returned

    def dedh(self, pseudo_enthalpy, rel_dh=1e-5, interp_type="CubicSpline"):
        """
        Value of [depsilon/dh](p)

        :param pseudo_enthalpy (`float`): Dimensionless pseudo-enthalpy.
        :param interp_type (`str`): String specifying interpolation type.
                                    Current implementations are 'CubicSpline', 'linear'.
        :param rel_dh (`float`): Relative step size in pseudo-enthalpy space.

        :return dedh (`float`): Derivative of energy-density with respect to pseudo-enthalpy
                                evaluated at `pseudo_enthalpy` in geometerized units.
        """

        # step size=fraction of value
        dh = pseudo_enthalpy * rel_dh

        eps_upper = self.energy_density_from_pseudo_enthalpy(
            pseudo_enthalpy + dh, interp_type=interp_type
        )
        eps_lower = self.energy_density_from_pseudo_enthalpy(
            pseudo_enthalpy - dh, interp_type=interp_type
        )

        return (eps_upper - eps_lower) / (2.0 * dh)

    def dedp(self, pressure, rel_dp=1e-5, interp_type="CubicSpline"):
        """
        Find value of [depsilon/dp](p)

        :param pressure (`float`): pressure in geometerized units.
        :param rel_dp (`float`): Relative step size in pressure space.
        :param interp_type (`float`): String specifying interpolation type.
                                      Current implementations are 'CubicSpline', 'linear'.

        :return dedp (`float`): Derivative of energy-density with respect to pressure
                                evaluated at `pressure`.
        """

        # step size=fraction of value
        dp = pressure * rel_dp

        eps_upper = self.energy_from_pressure(
            pressure + dp, interp_type=interp_type
        )
        eps_lower = self.energy_from_pressure(
            pressure - dp, interp_type=interp_type
        )

        return (eps_upper - eps_lower) / (2.0 * dp)

    def velocity_from_pseudo_enthalpy(
        self, pseudo_enthalpy, interp_type="CubicSpline"
    ):
        """
        Returns the speed of sound in geometerized units in the
        neutron star at the specified pressure.

        Assumes the equation
        vs = c (de/dp)^{-1/2}

        :param pseudo_enthalpy (`float`): Dimensionless pseudo-enthalpy.
        :param interp_type (`str`): String specifying interpolation type.
                                    Current implementations are 'CubicSpline', 'linear'.

        :return v_s (`float`): Speed of sound at `pseudo-enthalpy` in geometerized units.
        """
        pressure = self.pressure_from_pseudo_enthalpy(
            pseudo_enthalpy, interp_type=interp_type
        )
        return self.dedp(pressure, interp_type=interp_type) ** -0.5

    def check_causality(self):
        """
        Checks to see if the equation of state is causal i.e. the speed
        of sound in the star is less than the speed of light.
        Returns True if causal, False if not.
        """
        pmax = self.pressure[-1]
        emax = self.energy_from_pressure(pmax)
        hmax = self.pseudo_enthalpy_from_energy_density(emax)
        vsmax = self.velocity_from_pseudo_enthalpy(hmax)
        if vsmax < 1.1:
            return True
        else:
            return False

    def check_monotonicity(self):
        """
        Checks to see if the equation of state is monotonically increasing
        in energy density-pressure space. Returns True if monotonic, False if not.
        """
        e1 = self.energy_density[1:]
        e2 = self.energy_density[:-1]
        ediff = e1 - e2
        e_negatives = len(np.where(ediff < 0))

        p1 = self.pressure[1:]
        p2 = self.pressure[:-1]
        pdiff = p1 - p2
        p_negatives = len(np.where(pdiff < 0))
        if e_negatives > 1 or p_negatives > 1:
            return False
        else:
            return True

    def __get_plot_range(self, data):
        """
        Determines default plot range based on data provided.
        """
        low = np.amin(data)
        high = np.amax(data)
        dx = 0.05 * (high - low)

        xmin = low - dx
        xmax = high + dx
        xlim = [xmin, xmax]

        return xlim

    def __construct_all_tables(self):
        """Pressure and epsilon already tabular, now create array of enthalpies"""
        edat = self.energy_density
        hdat = [self.pseudo_enthalpy_from_energy_density(e) for e in edat]
        self.pseudo_enthalpy = np.array(hdat)

    def plot(self, rep, xlim=None, ylim=None, units=None):
        """
        Given a representation in the form 'energy_density-pressure', plot the EoS in that space.

        Parameters
        ==========
        rep: str
            Representation to plot. For example, plotting in energy_density-pressure space,
            specify 'energy_density-pressure'
        xlim: list
            Plotting bounds for x-axis in the form [low, high].
            Defaults to 'None' which will plot from 10% below min x value to 10% above max x value
        ylim: list
            Plotting bounds for y-axis in the form [low, high].
            Defaults to 'None' which will plot from 10% below min y value to 10% above max y value
        units: str
            Specifies unit system to plot. Currently can plot in CGS:'cgs', SI:'si', or geometerized:'geom'

        Returns
        =======
        fig: matplotlib.figure.Figure
            EOS plot.
        """
        import matplotlib.pyplot as plt

        # Set data based on specified representation
        varnames = rep.split("-")

        assert varnames[0] != varnames[1], (
            "Cannot plot the same variable against itself. Please choose another representation"
        )

        # Correspondence of rep parameter, data, and latex symbol
        # rep_dict = {'energy_density': [self.epsilon, r'$\epsilon$'],
        # 'pressure': [self.p, r'$p$'], 'pseudo_enthalpy': [pseudo_enthalpy, r'$h$']}

        # FIXME: The second element in these arrays should be tex labels, but tex's not working rn
        rep_dict = {
            "energy_density": [self.energy_density, "energy_density"],
            "pressure": [self.pressure, "pressure"],
            "pseudo_enthalpy": [self.pseudo_enthalpy, "pseudo_enthalpy"],
        }

        xname = varnames[1]
        yname = varnames[0]

        # Set units
        eos_default_units = {
            "pressure": "cgs",
            "energy_density": "cgs",
            "density": "cgs",
            "pseudo_enthalpy": "dimensionless",
        }
        if units is None:
            units = [
                eos_default_units[yname],
                eos_default_units[xname],
            ]  # Default unit system is cgs
        elif isinstance(units, str):
            units = [
                units,
                units,
            ]  # If only one unit system given, use for both

        xunits = units[1]
        yunits = units[0]

        # Ensure valid units
        if xunits not in list(
            conversion_dict[xname].keys()
        ) or yunits not in list(conversion_dict[yname].keys()):
            s = """
                Invalid unit system. Valid variable-unit pairs are:
                p: {p_units}
                e: {e_units}
                rho: {rho_units}
                h: {h_units}.
                """.format(
                p_units=list(conversion_dict["pressure"].keys()),
                e_units=list(conversion_dict["energy_density"].keys()),
                rho_units=list(conversion_dict["density"].keys()),
                h_units=list(conversion_dict["pseudo_enthalpy"].keys()),
            )
            raise ValueError(s)

        xdat = rep_dict[xname][0] * conversion_dict[xname][xunits]
        ydat = rep_dict[yname][0] * conversion_dict[yname][yunits]

        xlabel = rep_dict[varnames[1]][1].replace("_", " ")
        ylabel = (
            rep_dict[varnames[0]][1].replace("_", " ") + "(" + xlabel + ")"
        )

        # Determine plot ranges. Currently shows 10% wider than actual data range.
        if xlim is None:
            xlim = self.__get_plot_range(xdat)

        if ylim is None:
            ylim = self.__get_plot_range(ydat)

        fig, ax = plt.subplots()

        ax.loglog(xdat, ydat)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig


def spectral_adiabatic_index(gammas, x):
    arg = 0
    for i in range(len(gammas)):
        arg += gammas[i] * x**i

    return np.exp(arg)


class SpectralDecompositionEOS(TabularEOS):
    """
    Parameterized EOS using a spectral
    decomposition per Lindblom
    arXiv: 1009.0738v2. Inherits from TabularEOS.

    Parameters
    ==========
    gammas: list
        List of adiabatic expansion parameters used
        to construct the equation of state in various
        spaces.
    p0: float
        The starting point in pressure of the high-density EoS. This is stitched to
        the low-density portion of the SLY EoS model. The default value chosen is set to
        a sufficiently low pressure so that the high-density EoS will never be
        overconstrained.
    e0/c**2: float
        The starting point in energy-density of the high-density EoS. This is stitched to
        the low-density portion of the SLY EoS model. The default value chosen is set to
        a sufficiently low energy density so that the high-density EoS will never be
        overconstrained.
    xmax: float
        highest dimensionless pressure value in EoS
    npts: float (optional)
        number of points in pressure-energy density data.
    """

    def __init__(
        self,
        gammas,
        p0=3.01e33,
        e0=2.03e14,
        xmax=None,
        npts=100,
        sampling_flag=False,
        warning_flag=False,
    ):
        self.warning_flag = warning_flag
        self.gammas = gammas
        self.p0 = p0
        self.e0 = e0
        self.xmax = xmax
        self.npts = npts
        if self.xmax is None:
            self.xmax = self.__determine_xmax()
        self.sampling_flag = sampling_flag
        self.__construct_a_of_x_table()

        # Construct pressure-energy density table and
        # set up interpolating functions.

        if self.warning_flag and self.sampling_flag:
            # If sampling prior is enabled and adiabatic check
            # has failed, empty the array values
            self.e_pdat = np.zeros((2, 2))
        else:
            self.e_pdat = self.__construct_e_of_p_table()

        super().__init__(
            self.e_pdat,
            sampling_flag=self.sampling_flag,
            warning_flag=self.warning_flag,
        )

    def __determine_xmax(self, a_max=6.0):
        highest_order_gamma = np.abs(self.gammas[-1])[0]
        expansion_order = float(len(self.gammas) - 1)

        xmax = (np.log(a_max) / highest_order_gamma) ** (1.0 / expansion_order)
        return xmax

    def __mu_integrand(self, x):

        return 1.0 / spectral_adiabatic_index(self.gammas, x)

    def mu(self, x):
        from scipy.integrate import quad

        return np.exp(-quad(self.__mu_integrand, 0, x)[0])

    def __eps_integrand(self, x):

        return (
            np.exp(x) * self.mu(x) / spectral_adiabatic_index(self.gammas, x)
        )

    def energy_density(self, x, eps0):
        from scipy.integrate import quad

        quad_result, quad_err = quad(self.__eps_integrand, 0, x)
        eps_of_x = (eps0 * C_CGS**2.0) / self.mu(x) + self.p0 / self.mu(
            x
        ) * quad_result
        return eps_of_x

    def __construct_a_of_x_table(self):

        xdat = np.linspace(0, self.xmax, num=self.npts)

        # Generate adiabatic index points until a point is out of prior range
        if self.sampling_flag:
            adat = np.empty(self.npts)
            for i in range(self.npts):
                if 0.6 < spectral_adiabatic_index(self.gammas, xdat[i]) < 4.6:
                    adat[i] = spectral_adiabatic_index(self.gammas, xdat[i])
                else:
                    break

            # Truncate arrays to last point within range
            adat = adat[:i]
            xmax_new = xdat[i - 1]

            # If EOS is too short, set prior to 0, else resample the function and set new xmax
            if xmax_new < 4.0 or i == 0:
                self.warning_flag = True
            else:
                xdat = np.linspace(0, xmax_new, num=self.npts)
                adat = spectral_adiabatic_index(self.gammas, xdat)
                self.xmax = xmax_new
        else:
            adat = spectral_adiabatic_index(self.gammas, xdat)

        self.adat = adat

    def __construct_e_of_p_table(self):
        """
        Creates p, epsilon table for a given set of spectral parameters
        """

        # make p range
        # to match lalsimulation tables: array = [pressure, density]
        x_range = np.linspace(0, self.xmax, self.npts)
        p_range = self.p0 * np.exp(x_range)

        eos_vals = np.zeros((self.npts, 2))
        eos_vals[:, 0] = p_range

        for i in range(0, len(x_range)):
            eos_vals[i, 1] = self.energy_density(x_range[i], self.e0)

        # convert eos to geometrized units in *m^-2*
        # IMPORTANT
        eos_vals = eos_vals * 0.1 * G_SI / C_SI**4

        # doing as those before me have done and using SLY4 as low density region
        # SLY4 in geometrized units
        low_density_path = os.path.join(
            os.path.dirname(__file__),
            "eos_tables",
            "LALSimNeutronStarEOS_SLY4.dat",
        )
        low_density = np.loadtxt(low_density_path)

        cutoff = eos_vals[0, :]

        # Then find overlap point
        break_pt = len(low_density)
        for i in range(1, len(low_density)):
            if (
                low_density[-i, 0] < cutoff[0]
                and low_density[-i, 1] < cutoff[1]
            ):
                break_pt = len(low_density) - i + 1
                break

        # stack EOS arrays
        eos_vals = np.vstack((low_density[0:break_pt, :], eos_vals))

        return eos_vals


class ChebSpectralDecompositionEOS(TabularEOS):
    def __init__(
        self,
        upsilons,
        pressure,
        energy_density,
        p0=3.01e33,
        e0=2.03e14,
        xmax=None,
        pmax=None,
        npts=100,
        sampling_flag=False,
        warning_flag=False,
    ):

        self.warning_flag = warning_flag
        self.sampling_flag = sampling_flag

        # Inputs are geometrized (m^-2)
        self.upsilons = upsilons
        self.pressure = np.asarray(pressure, float)
        self.energy_density = np.asarray(energy_density, float)

        # Build SLY4 interpolator (geom)
        self._sly_interp = self._load_sly4_interp()

        # Unit conversions
        self.geom_factor = 0.1 * G_SI / (C_SI**4)  # cgs to geom
        self.p0_cgs = float(p0)
        self.e0_cgs_mass = float(e0)  # e0 / c^2 in cgs
        self.p0_geom = self.p0_cgs * self.geom_factor
        self.e0_geom = float(self._sly_interp(self.p0_geom))

        # Domain (keep p0 fixed)
        if pmax is None:
            self.pmax_geom = self.p0_geom * np.exp(6.0)  # ~403 * p0
        else:
            # current fix
            pmax_val = float(pmax)
            self.pmax_geom = (
                pmax_val * self.geom_factor if pmax_val > 1e20 else pmax_val
            )
        if self.pmax_geom <= self.p0_geom:
            raise ValueError(
                "With fixed p0, pmax must be strictly greater than p0."
            )

        self.xmax = (
            float(np.log(self.pmax_geom / self.p0_geom))
            if xmax is None
            else float(xmax)
        )
        self.npts = int(npts)

        # upsilon0 situation
        self.upsilon0 = self._compute_upsilon0()

        self.__construct_a_of_x_table()

        # Build stitched EOS table (geom units)
        if self.warning_flag and self.sampling_flag:
            self.e_pdat = np.zeros((2, 2))
        else:
            self.e_pdat = self.__construct_e_of_p_table()

        super().__init__(
            self.e_pdat,
            sampling_flag=self.sampling_flag,
            warning_flag=self.warning_flag,
        )

    def _load_sly4_interp(self):
        low_density_path = os.path.join(
            os.path.dirname(__file__),
            "eos_tables",
            "LALSimNeutronStarEOS_SLY4.dat",
        )
        ld = np.loadtxt(low_density_path)
        ld = ld[(ld[:, 0] > 0) & (ld[:, 1] > 0)]  # keep only positive rows
        self._sly_table = ld
        return PchipInterpolator(ld[:, 0], ld[:, 1], extrapolate=True)

    def _compute_upsilon0(self):
        # interp_e = PchipInterpolator(self.pressure, self.energy_density, extrapolate=True)
        dede_p0_sly = float(self._sly_interp.derivative()(self.p0_geom))
        gamma0 = ((self.e0_geom + self.p0_geom) / self.p0_geom) / dede_p0_sly
        return float(np.clip(gamma0, 0.6, 4.6))

    # try without clamp=False
    def generating_function(self, x, upsilons):
        x = np.asarray(x, float)
        y = -1.0 + 2.0 * (x / self.xmax)
        s = cheb.chebval(y, upsilons)
        arg = (1.0 + y) * s
        gamma = self.upsilon0 * np.exp(arg)
        return gamma

    def compute_energy_density_array(self, x_array, upsilons):
        # array and float
        x = np.asarray(x_array, float)
        # output array that will be returned with the correct(?) values

        # Find the range of x values to integrate over
        x_min, x_max = x.min(), x.max()

        # Ensure we include x=0 in our integration range
        if x_min > 0:
            x_min = 0.0
        if x_max < 0:
            x_max = 0.0

        # integration
        integration_points = np.unique(
            np.concatenate(
                [
                    [x_min],  # Start point
                    x,  # All original x values
                    [x_max],  # End point
                ]
            )
        )

        # Sort the integration points
        integration_points = np.sort(integration_points)

        # integration of dimensionless x values
        # use generating function
        G = self.generating_function(integration_points, upsilons)
        # inverse of generating function
        s = 1.0 / G
        # pressure back to cgs units
        p = self.p0_geom * np.exp(integration_points)
        # interpolators
        s_interp = interp1d(
            integration_points, s, kind="linear", fill_value="extrapolate"
        )
        p_interp = interp1d(
            integration_points, p, kind="linear", fill_value="extrapolate"
        )

        # diff eq
        def energy_density_ode(x_val, e_val):
            s_val = s_interp(x_val)
            p_val = p_interp(x_val)
            return s_val * (e_val + p_val)

        # compare the below to chebyshev_eos file if errors
        # dense_points = np.linspace(x_min, x_max, max(200, len(integration_points)))
        # Solve the ivp over the entire range
        solution = solve_ivp(
            energy_density_ode,
            t_span=(x_min, x_max),  # Integration range
            y0=[self.e0_geom],  # Initial condition: e(0) = e0_geom
            t_eval=integration_points,  # Points where to evaluate the solution
            method="RK45",  # Runge-Kutta method
            rtol=1e-8,  # Tolerances
            atol=1e-10,  # using integration not dense points
        )

        # Create interpolator
        e_interp = interp1d(
            integration_points,
            solution.y[0],
            kind="linear",
            fill_value="extrapolate",
        )
        # change integration_points to solution.t if error
        # Return energy density values at the original x positions
        return e_interp(x)

    def compute_energy_density(self, x, upsilons):
        return float(
            self.compute_energy_density_array(np.array([x], float), upsilons)[
                0
            ]
        )

    def __construct_a_of_x_table(self):
        xdat = np.linspace(0.0, self.xmax, num=self.npts)
        if self.sampling_flag:
            adat = np.empty(self.npts)
            i = 0
            for i in range(self.npts):
                g = self.generating_function(
                    xdat[i], self.upsilons
                )  # , clamp=True
                if 0.6 < g < 4.6:
                    adat[i] = g
                else:
                    break
            adat = adat[:i]
            xmax_new = xdat[i - 1] if i > 0 else 0.0
            if xmax_new < 4.0 or i == 0:
                self.warning_flag = True
            else:
                xdat = np.linspace(0.0, xmax_new, num=self.npts)
                adat = self.generating_function(
                    xdat, self.upsilons
                )  # , clamp=True
                self.xmax = xmax_new
        else:
            adat = self.generating_function(xdat, self.upsilons)
        self.adat = adat

    def __construct_e_of_p_table(self):
        """
        Creates p, epsilon table for a given set of spectral parameters
        """
        # Spectral branch (geom)
        x_range = np.linspace(0.0, self.xmax, self.npts)
        p_range_geom = self.p0_geom * np.exp(x_range)
        eps_range_geom = self.compute_energy_density_array(
            x_range, self.upsilons
        )
        eos_vals = np.column_stack([p_range_geom, eps_range_geom])

        # Load SLY4 (geom)
        low_density = self._sly_table

        # Find overlap point
        cutoff = eos_vals[0, :]
        break_pt = len(low_density)
        for i in range(1, len(low_density)):
            if (
                low_density[-i, 0] < cutoff[0]
                and low_density[-i, 1] < cutoff[1]
            ):
                break_pt = len(low_density) - i + 1
                break

        # Stack without reordering (keeps SLY4 order as-is)
        eos_vals = np.vstack((low_density[0:break_pt, :], eos_vals))
        # print('eos_vals', eos_vals)

        return eos_vals


def ChebyshevNeutronStarEOSSpectralDecomposition(upsilons):

    upsilons = np.asarray(upsilons)  # turn the upsilons into an array

    ndat_low = 69
    ndat = ndat_low + 500

    # Minimum pressure and energy density (cgs)
    e0 = 2.03e14
    # 9.54629006e-11
    p0 = 3.01e33
    # 4.43784199e-13

    xmax = 12.3081  # very relaxed upper limit
    pmax = p0 * np.exp(xmax)  # give units back to xmax

    pressure = np.array(
        [
            8.34363176e-38,
            8.34363158e-37,
            8.34363158e-36,
            8.34363204e-35,
            9.99590323e-34,
            1.15654576e-32,
            1.40438658e-31,
            4.80792427e-30,
            1.56955470e-28,
            8.04954335e-27,
            4.10409137e-26,
            2.00824991e-25,
            9.50846400e-25,
            4.35032199e-24,
            1.91489794e-23,
            8.05870148e-23,
            3.23083653e-22,
            4.34456135e-22,
            1.18546722e-21,
            3.16643535e-21,
            8.31069246e-21,
            2.15123216e-20,
            5.51515411e-20,
            7.21853315e-20,
            1.34573133e-19,
            2.50233629e-19,
            3.41104119e-19,
            4.16022078e-19,
            5.66714695e-19,
            1.05080103e-18,
            1.94635992e-18,
            3.60350588e-18,
            4.67734957e-18,
            6.36270771e-18,
            8.65766111e-18,
            1.17719556e-17,
            1.60101658e-17,
            2.06777667e-17,
            2.81208180e-17,
            3.82314916e-17,
            4.91456998e-17,
            6.68234984e-17,
            9.08719781e-17,
            1.23502352e-16,
            1.67945004e-16,
            2.14547164e-16,
            2.38904345e-16,
            2.71784343e-16,
            3.69523312e-16,
            4.80467457e-16,
            6.44778806e-16,
            6.51794987e-16,
            6.89962998e-16,
            7.51587087e-16,
            8.12147785e-16,
            8.94672702e-16,
            1.00619290e-15,
            1.15571864e-15,
            7.67630482e-15,
            1.54310830e-14,
            2.76006879e-14,
            4.25728200e-14,
            5.97616412e-14,
            7.89063920e-14,
            9.98687894e-14,
            1.22577195e-13,
            1.47001397e-13,
            1.73138786e-13,
            2.01005994e-13,
            2.30633510e-13,
            2.62062083e-13,
            2.95340235e-13,
            3.30522500e-13,
            3.67668165e-13,
            4.06840346e-13,
            4.48105310e-13,
            4.91531976e-13,
            5.36657977e-13,
            7.42975692e-13,
            1.90396753e-12,
            4.20379487e-12,
            8.33964365e-12,
            1.54208076e-11,
            2.54206991e-11,
            3.95859812e-11,
            5.93051173e-11,
            8.57893204e-11,
            1.18964736e-10,
            1.59082469e-10,
            2.06792439e-10,
            2.61740144e-10,
            3.23777876e-10,
            3.93053343e-10,
            4.69861963e-10,
            5.54351445e-10,
            6.45930954e-10,
            7.44305072e-10,
            8.48439836e-10,
            9.56858157e-10,
            1.06882149e-09,
            1.18447755e-09,
            1.30515571e-09,
        ]
    )

    energy_density = np.array(
        [
            5.76064628e-24,
            5.80495921e-24,
            5.83450097e-24,
            6.01913680e-24,
            8.56711563e-24,
            1.21121289e-23,
            3.33083545e-23,
            1.56571417e-22,
            8.49326096e-22,
            7.71040409e-21,
            1.93646347e-20,
            4.86479199e-20,
            1.22155249e-19,
            3.06939069e-19,
            7.71040347e-19,
            1.93646353e-18,
            4.86553076e-18,
            6.12474889e-18,
            1.22229097e-17,
            2.43867369e-17,
            4.86626935e-17,
            9.71185959e-17,
            1.93794050e-16,
            2.44015087e-16,
            3.86775695e-16,
            6.13065718e-16,
            7.71778906e-16,
            9.71924446e-16,
            1.22376810e-15,
            1.93941772e-15,
            3.07529885e-15,
            4.87513190e-15,
            6.13878125e-15,
            7.72517467e-15,
            9.73401567e-15,
            1.22524524e-14,
            1.54355785e-14,
            1.94311035e-14,
            2.44679768e-14,
            3.08120733e-14,
            3.88031234e-14,
            4.88694867e-14,
            6.15355240e-14,
            7.74733127e-14,
            9.76355731e-14,
            1.22893792e-13,
            1.36187588e-13,
            1.54798914e-13,
            1.94975724e-13,
            2.45566027e-13,
            3.17500257e-13,
            3.29390815e-13,
            3.86111031e-13,
            4.88177866e-13,
            5.88176782e-13,
            7.18455992e-13,
            8.83299112e-13,
            1.08639886e-12,
            4.97965079e-12,
            9.96810241e-12,
            1.49620990e-11,
            1.99607387e-11,
            2.49634397e-11,
            2.99697938e-11,
            3.49795013e-11,
            3.99923351e-11,
            4.50081194e-11,
            5.00267156e-11,
            5.50480142e-11,
            6.00719276e-11,
            6.50983862e-11,
            7.01273342e-11,
            7.51587276e-11,
            8.01925324e-11,
            8.52287224e-11,
            9.02672780e-11,
            9.53081864e-11,
            1.00351439e-10,
            1.25434385e-10,
            1.76068991e-10,
            2.27176265e-10,
            2.79022084e-10,
            3.31754156e-10,
            3.85815608e-10,
            4.41501857e-10,
            4.99108322e-10,
            5.59225839e-10,
            6.22149824e-10,
            6.88323404e-10,
            7.57894289e-10,
            8.31305605e-10,
            9.08705061e-10,
            9.90388074e-10,
            1.07650235e-09,
            1.16719561e-09,
            1.26261555e-09,
            1.36290988e-09,
            1.46822632e-09,
            1.57900798e-09,
            1.69422091e-09,
            1.81386511e-09,
            1.93794057e-09,
        ]
    )

    # initialize class
    model = ChebSpectralDecompositionEOS(
        upsilons=upsilons,
        pressure=pressure,
        energy_density=energy_density,
        p0=p0,
        e0=e0,
        pmax=pmax,
        xmax=xmax,
        npts=ndat,
    )

    return model


class EOSFamily(object):
    """
    Create a EOS family and get mass-radius information

    Parameters
    ==========
    eos: object
        Supply a `TabularEOS` class (or subclass)
    npts: float
        Number of points to calculate for mass-radius relation. Default is 500.

    Notes
    =====
    The mass-radius and mass-k2 data should be
    populated here via the TOV solver upon object construction.
    """

    def __init__(self, eos, npts=500):
        from scipy.optimize import minimize_scalar

        self.eos = eos

        # FIXME: starting_energy_density is set somewhat arbitrarily
        starting_energy_density = 1.6e-10
        ending_energy_density = max(self.eos.energy_density)
        log_starting_energy_density = np.log(starting_energy_density)
        log_ending_energy_density = np.log(ending_energy_density)
        log_energy_density_grid = np.linspace(
            log_starting_energy_density, log_ending_energy_density, num=npts
        )
        energy_density_grid = np.exp(log_energy_density_grid)

        # Generate m, r, and k2 lists
        mass = []
        radius = []
        k2love_number = []
        for i in range(len(energy_density_grid)):
            tov_solver = IntegrateTOV(self.eos, energy_density_grid[i])
            m, r, k2 = tov_solver.integrate_TOV()
            mass.append(m)
            radius.append(r)
            k2love_number.append(k2)

            # Check if maximum mass has been found
            if i > 0 and mass[i] <= mass[i - 1]:
                break

        # If we're not at the end of the array, determine actual maximum mass. Else, assume
        # last point is the maximum mass and proceed.
        if i < (npts - 1):
            # Now replace with point with interpolated maximum mass
            # This is done by interpolating the last three points and then
            # minimizing the negative of the interpolated function
            x = [
                energy_density_grid[i - 2],
                energy_density_grid[i - 1],
                energy_density_grid[i],
            ]
            y = [mass[i - 2], mass[i - 1], mass[i]]

            f = interp1d(
                x,
                y,
                kind="quadratic",
                bounds_error=False,
                fill_value="extrapolate",
            )

            res = minimize_scalar(lambda x: -f(x))

            # Integrate max energy density to get maximum mass
            tov_solver = IntegrateTOV(self.eos, res.x)
            mfin, rfin, k2fin = tov_solver.integrate_TOV()

            mass[-1] = mfin
            radius[-1] = rfin
            k2love_number[-1] = k2fin

        # Currently, everything is in geometerized units.
        # The mass variables have dimensions of length, k2 is dimensionless
        # and radii have dimensions of length. Calculate dimensionless lambda
        # with these quantities, then convert to SI.

        # Calculating dimensionless lambda values from k2, radii, and mass
        tidal_deformability = [
            2.0 / 3.0 * k2 * r**5.0 / m**5.0
            for k2, r, m in zip(k2love_number, radius, mass)
        ]

        # As a last resort, if highest mass is still smaller than second
        # to last point, remove the last point from each array
        if mass[-1] < mass[-2]:
            mass = mass[:-1]
            radius = radius[:-1]
            k2love_number = k2love_number[:-1]
            tidal_deformability = tidal_deformability[:-1]

        self.mass = np.array(mass)
        self.radius = np.array(radius)
        self.k2love_number = np.array(k2love_number)
        self.tidal_deformability = np.array(tidal_deformability)
        self.maximum_mass = mass[-1] * conversion_dict["mass"]["m_sol"]

    def radius_from_mass(self, m):
        """
        :param m: mass of neutron star in solar masses
        :return: radius of neutron star in meters
        """
        f = CubicSpline(
            self.mass, self.radius, bc_type="natural", extrapolate=True
        )

        mass_converted_to_geom = m * MSUN_SI * G_SI / C_SI**2.0
        return f(mass_converted_to_geom)

    def k2_from_mass(self, m):
        """
        :param m: mass of neutron star in solar masses.
        :return: dimensionless second tidal love number.
        """
        f = CubicSpline(
            self.mass, self.k2love_number, bc_type="natural", extrapolate=True
        )

        m_geom = m * MSUN_SI * G_SI / C_SI**2.0
        return f(m_geom)

    def lambda_from_mass(self, m):
        """
        Convert from equation of state model parameters to
        component tidal parameters.

        :param m: Mass of neutron star in solar masses.
        :return: Tidal parameter of neutron star of mass m.
        """

        # Get lambda from mass and equation of state

        r = self.radius_from_mass(m)
        k = self.k2_from_mass(m)
        m_geom = m * MSUN_SI * G_SI / C_SI**2.0
        c = m_geom / r
        lmbda = (2.0 / 3.0) * k / c**5.0

        return lmbda

    def __get_plot_range(self, data):
        low = np.amin(data)
        high = np.amax(data)
        dx = 0.05 * (high - low)

        xmin = low - dx
        xmax = high + dx
        xlim = [xmin, xmax]

        return xlim

    def plot(self, rep, xlim=None, ylim=None, units=None):
        """
        Given a representation in the form 'm-r', plot the family in that space.

        Parameters
        ==========
        rep: str
            Representation to plot. For example, plotting in mass-radius space, specify 'm-r'
        xlim: list
            Plotting bounds for x-axis in the form [low, high].
            Defaults to 'None' which will plot from 10% below min x value to 10% above max x value
        ylim: list
            Plotting bounds for y-axis in the form [low, high].
            Defaults to 'None' which will plot from 10% below min y value to 10% above max y value
        units: str
            Specifies unit system to plot. Currently can plot in CGS:'cgs', SI:'si', or geometerized:'geom'

        Returns
        =======
        fig: matplotlib.figure.Figure
            EOS Family plot.
        """
        import matplotlib.pyplot as plt

        # Set data based on specified representation
        varnames = rep.split("-")

        assert varnames[0] != varnames[1], (
            "Cannot plot the same variable against itself. Please choose another representation"
        )

        # Correspondence of rep parameter, data, and latex symbol
        rep_dict = {
            "mass": [self.mass, r"$M$"],
            "radius": [self.radius, r"$R$"],
            "k2": [self.k2love_number, r"$k_2$"],
            "tidal_deformability": [self.tidal_deformability, r"$l$"],
        }

        xname = varnames[1]
        yname = varnames[0]

        # Set units
        fam_default_units = {
            "mass": "m_sol",
            "radius": "km",
            "tidal_deformability": "geom",
        }
        if units is None:
            units = [
                fam_default_units[yname],
                fam_default_units[xname],
            ]  # Default unit system is cgs
        elif isinstance(units, str):
            units = [
                units,
                units,
            ]  # If only one unit system given, use for both

        xunits = units[1]
        yunits = units[0]

        # Ensure valid units
        if xunits not in list(
            conversion_dict[xname].keys()
        ) or yunits not in list(conversion_dict[yname].keys()):
            s = """
                        Invalid unit system. Valid variable-unit pairs are:
                        m: {m_units}
                        r: {r_units}
                        l: {l_units}.
                        """.format(
                m_units=list(conversion_dict["mass"].keys()),
                r_units=list(conversion_dict["radius"].keys()),
                l_units=list(conversion_dict["tidal_deformability"].keys()),
            )
            raise ValueError(s)

        xdat = rep_dict[varnames[1]][0] * conversion_dict[xname][xunits]
        ydat = rep_dict[varnames[0]][0] * conversion_dict[yname][yunits]

        xlabel = rep_dict[varnames[1]][1].replace("_", " ")
        ylabel = (
            rep_dict[varnames[0]][1].replace("_", " ") + "(" + xlabel + ")"
        )

        # Determine plot ranges. Currently shows 10% wider than actual data range.
        if xlim is None:
            xlim = self.__get_plot_range(xdat)

        if ylim is None:
            ylim = self.__get_plot_range(ydat)

        # Plot the data with appropriate labels
        fig, ax = plt.subplots()

        ax.loglog(xdat, ydat)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig
