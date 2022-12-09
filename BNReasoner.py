from typing import Union
from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    def query(self, query_var: str, evidence: dict) -> float:
        """
        :param query_var: name of the query variable
        :param evidence: dictionary of evidence variables and their values
        :return: the probability of the query variable given the evidence
        """
        pass

    def prune(self, query_var: str, evidence: dict) -> BayesNet:
        """ Edge-prune the Bayesian network s.t. queries of the form P(Q|E) can still be correctly calculated.
        :param evidence: dictionary of evidence variables and their values
        :return: a new BayesNet object with the evidence variables removed
        """
        pass

    def d_separation(self, x: str, y: str, z: list) -> bool:
        """ determine whether X is d-separated of Y given Z.

        :param x: name of variable x
        :param y: name of variable y
        :param z: list of variables z
        :return: True if x and y are d-separated given z, False otherwise
        """
        pass

    def independence(self, x: str, y: str, z: list) -> bool:
        """ determine whether X is independent of Y given Z.

        :param x: name of variable x
        :param y: name of variable y
        :param z: list of variables z
        :return: True if x and y are independent given z, False otherwise
        """
        pass

    def marginalization(self, x: str, evidence: dict) -> float:
        """ Given a factor and a variable X, compute the CPT in which X is summed-out. 

        :param x: name of variable x
        :param evidence: dictionary of evidence variables and their values
        :return: the marginal probability of x given the evidence
        """
        pass

    def maxing_out(self, x: str, evidence: dict) -> float:
        """ Given a factor and a variable X, compute the CPT in which X is maxed-out.

        :param x: name of variable x
        :param evidence: dictionary of evidence variables and their values
        :return: the maximum probability of x given the evidence
        """
        pass

    def factor_multiplication(self, x: str, y: str) -> float:
        """ Given two factors, compute the product of the two factors.

        :param x: name of variable x
        :param y: name of variable y
        :return: the product of the two factors
        """
        pass

    def ordering(self, x: str, heuristic: str) -> list:
        """ Given X, compute a good ordering for the elimination of X based on the heuristic

        :return: a topological ordering of the variables in the Bayesian network
        """
        pass

    def variable_elimination(self, x: str, evidence: dict) -> float:
        """ Given a variable X and evidence E, compute the probability of X given E using variable elimination.

        :param x: name of variable x
        :param evidence: dictionary of evidence variables and their values
        :return: the probability of x given the evidence
        """
        pass

    def marginal_distributions(self, query_variables: str, evidence: dict) -> dict:
        """ Given query variables Q, evidence E, compute the marginal distributions.

        :param evidence: dictionary of evidence variables and their values
        :return: a dictionary of the marginal distributions of all variables in the Bayesian network
        """
        pass

    def MAP(self, evidence: dict) -> dict:
        """ Given evidence E, compute the MAP assignment of query variables in the Bayesian network.

        :param evidence: dictionary of evidence variables and their values
        :return: a dictionary of the MAP assignment of query variables in the Bayesian network
        """
        pass

    def MEP(self, evidence: dict) -> dict:
        """ Given evidence E, compute the MEP assignment of query variables in the Bayesian network.

        :param evidence: dictionary of evidence variables and their values
        :return: a dictionary of the MEP assignment of query variables in the Bayesian network
        """
        pass

a = BNReasoner('testing/lecture_example.bifxml').bn 
# a = BNReasoner('testing/lecture_example2.bifxml').bn
a.draw_structure()