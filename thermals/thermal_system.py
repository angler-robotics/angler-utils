import numpy as np
from scipy.integrate import solve_ivp


class ThermalSystem:
    def __init__(self, thermal_capacitances, node_labels=None):
        """
        A list of thermal capacitances in J/K. Every node in the system has its own temperature and  capacitance.
        If the temperature is constant (i.e. the environment) assign it a capacitance of np.inf.

        :param thermal_capacitances: The list of thermal capacitances in J/K.
        """
        self.thermal_capacitances = np.asarray(thermal_capacitances)  # m*c
        self.heat_relations = []  # functions that map the list of temperatures to a list of heat transfer rates
        self.node_labels = node_labels

    def add_heat_source(self, node, Q, time_dependant=False):
        """
        Adds a heat source to the node with the given index.

        :param node: An identifier for the node to apply the heat source to. Can be an integer corresponding to the
                     node's index, or a string corresponding to the node's label (if provided).
        :param Q: The heat to apply to the specified node in W. Can be a constant or a function of time if
                  time_dependant is set to True.
        :param time_dependant: If True, then Q is a function of time. Otherwise, Q is a constant.
        """
        if type(node) is str:
            node = self.node_labels.index(node)

        def heat_source(t, T):
            heat_flux = np.zeros_like(T)
            heat_flux[node] = Q(t) if time_dependant else Q
            return heat_flux

        self.heat_relations.append(heat_source)

    def add_conductive_contact(self, node1, node2, k, A, L):
        """

        :param node1: An identifier for the first node. Can be an integer corresponding to the
                      node's index, or a string corresponding to the node's label (if provided).
        :param node2: An identifier for the second node. Can be an integer corresponding to the
                      node's index, or a string corresponding to the node's label (if provided).
        :param k:
        :param A:
        :param L:
        """
        if type(node1) is str:
            node1 = self.node_labels.index(node1)
        if type(node2) is str:
            node2 = self.node_labels.index(node2)

        # dQ/dt = -(kA/L)*(delta T)
        alpha = k * A / L  # thermal diffusivity

        def conductive_contact(_, T):
            heat_flux = np.zeros_like(T)
            heat_flux[node1] = -alpha * (T[node1] - T[node2])
            heat_flux[node2] = -heat_flux[node1]
            return heat_flux

        self.heat_relations.append(conductive_contact)

    def add_convective_contact(self, solid_node, fluid_node, h, A, b=1):
        """
        Adds a convective contact between a solid and a fluid node. Intuitively, heat is transferred at the boundary
        layer of the fluid, and the rest of the fluid is brought to equilibrium at a rate which is controlled by h.

        :param solid_node: An identifier for the node that is a solid. Can be an integer corresponding to the
                      node's index, or a string corresponding to the node's label (if provided).
        :param fluid_node: An identifier for the node that is a fluid. Can be an integer corresponding to the
                      node's index, or a string corresponding to the node's label (if provided).
        :param h:
        :param A:
        :param b:
        """
        if type(solid_node) is str:
            solid_node = self.node_labels.index(solid_node)
        if type(fluid_node) is str:
            fluid_node = self.node_labels.index(fluid_node)

        # dQ/dt = hA * (T_fluid - T)^b
        alpha = h * A

        def contact(_, T):
            heat_flux = np.zeros_like(T)
            if T[solid_node] > T[fluid_node]:
                heat_flux[solid_node] = -alpha * (T[solid_node] - T[fluid_node]) ** b
                heat_flux[fluid_node] = -heat_flux[solid_node]
            else:
                heat_flux[fluid_node] = -alpha * (T[fluid_node] - T[solid_node]) ** b
                heat_flux[solid_node] = -heat_flux[fluid_node]
            return heat_flux

        self.heat_relations.append(contact)

    def heat_step(self, t, T):
        """
        Calculates the rate of change of temperature (with respect to time) for each node.

        :param t: The current time.
        :param T: A list of the starting temperatures of the nodes.
                  If a single integer is given, then all nodes will start at that temperature.
        :return: The rate of change of the temperatures of the nodes (with respect to time).
        """
        # dT/dt = (1/mc)*dQ/dt
        heat_flux = sum(relation(t, T) for relation in self.heat_relations)
        return heat_flux / self.thermal_capacitances

    def integrate(self, T_0, t_f, method='RK45', rtol=1e-9, atol=1e-12):
        if type(T_0) is not list:
            T_0 = [T_0] * len(self.thermal_capacitances)
        result = solve_ivp(self.heat_step, (0, t_f), T_0, method=method, rtol=rtol, atol=atol)
        # noinspection PyUnresolvedReferences
        return result.t, result.y
