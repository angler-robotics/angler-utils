import numpy as np
from scipy.integrate import solve_ivp


class ThermalSystem:
    def __init__(self, thermal_capacitances):
        """
        A list of thermal capacitances in J/K. Every node in the system has its own temperature and  capacitance.
        If the temperature is constant (i.e. the environment) assign it a capacitance of np.inf.

        :param thermal_capacitances: The list of thermal capacitances in J/K.
        """
        self.thermal_capacitances = np.asarray(thermal_capacitances)  # m*c
        self.heat_relations = []  # functions that map the list of temperatures to a list of heat transfer rates

    def add_heat_source(self, idx, Q, time_dependant=False):
        """
        Adds a constant heat source to the node with the given index.

        :param idx:
        :param Q:
        :param time_dependant:
        :return:
        """
        def heat_source(t, T):
            heat_flux = np.zeros_like(T)
            heat_flux[idx] = Q(t) if time_dependant else Q
            return heat_flux

        self.heat_relations.append(heat_source)

    def add_conductive_contact(self, i, j, k, A, L):
        # dQ/dt = -(kA/L)*(delta T)
        alpha = k * A / L  # thermal diffusivity

        def conductive_contact(_, T):
            heat_flux = np.zeros_like(T)
            heat_flux[i] = -alpha * (T[i] - T[j])
            heat_flux[j] = -heat_flux[i]
            return heat_flux

        self.heat_relations.append(conductive_contact)

    def add_convective_contact(self, i, j, h, A, b=1):
        """

        :param i: The index of the object radiating heat.
        :param j: The index of the fluid receiving heat.
        :param h:
        :param A:
        :param b:
        """
        # dQ/dt = hA * (T_fluid - T)^b
        alpha = h * A

        def contact(_, T):
            heat_flux = np.zeros_like(T)
            if T[i] > T[j]:
                heat_flux[i] = -alpha * (T[i] - T[j]) ** b
                heat_flux[j] = -heat_flux[i]
            else:
                heat_flux[j] = -alpha * (T[j] - T[i]) ** b
                heat_flux[i] = -heat_flux[j]
            return heat_flux

        self.heat_relations.append(contact)

    def heat_step(self, t, T):
        """
        Calculates dT/dt for each node.

        :param t: The time.
        :param T: The temperatures of the nodes.
        :return: The rate of change of the temperature of the nodes (with respect to time).
        """
        # dT/dt = (1/mc)*dQ/dt
        heat_flux = sum(relation(t, T) for relation in self.heat_relations)
        return heat_flux / self.thermal_capacitances

    def integrate(self, T_0, t_0, t_f, method='RK45', rtol=1e-9, atol=1e-12):
        result = solve_ivp(self.heat_step, (t_0, t_f), T_0, method=method, rtol=rtol, atol=atol)
        # noinspection PyUnresolvedReferences
        return result.t, result.y
