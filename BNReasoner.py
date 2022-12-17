from typing import Union
from BayesNet import BayesNet
import pandas as pd
import itertools
import networkx as nx
from typing import List


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

    def marginalization(self, x: str, factor: pd.DataFrame) -> pd.DataFrame:
        """ 
        param x: name of variable x
        param factor: CPT of variables
        return: Factor in which X is summed-out
        """
        updated_factor_variables = [v for v in factor.columns if v != x]
        updated_factor_variables.pop()
        
        updated_factor = factor.groupby(updated_factor_variables).sum()
        updated_factor.reset_index(inplace=True)
        if x in updated_factor.columns:
            updated_factor.drop(columns = [x], inplace = True)
        return updated_factor


    def maxing_out(self, x: str, factor: pd.DataFrame) -> pd.DataFrame:
        """ 
        :param x: name of variable x
        :param factor: dictionary of evidence variables and their values
        :return: the CPT in which X is maxed-out
        """
        updated_factor_variables = [v for v in factor.columns if v != x]
        updated_factor_variables.pop()
        # updated_factor_variables = [v for v in factor.columns if v != 'p'] 

        updated_factor = factor.groupby(updated_factor_variables).max()
        updated_factor.reset_index(inplace=True)
        updated_factor.drop(columns = [x], inplace = True) # Drop the variable that was maxed out
        
        tracked_factor = factor.groupby(updated_factor_variables)['p'].idxmax()
        tracked_factor = factor.loc[tracked_factor] # Keep track of the instances where the max occurs

        return tracked_factor

    def factor_multiplication(self, f: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
        """ Given two factors, compute the product of the two factors. h = fg

        :param f: factor 1
        :param g: factor 2
        :return: the product of the two factors
        """
        #print(f, "\n", g)
        g.rename(columns = {'p':'p1'}, inplace = True) # prevent name collision
        common_variables = [v for v in f.columns if v in g.columns] 
        h = f.join(g.set_index(common_variables), on=common_variables).reset_index(drop=True) 
        h['p1'] = h['p1'] * h['p'] # multiply the probabilities
        h.drop(columns = ['p'], inplace = True)
        h.rename(columns = {'p1':'p'}, inplace = True) # rename back to p
        g.rename(columns = {'p1':'p'}, inplace = True) # rename the original cpt back to p
        #print(h)
        return h

    def ordering(self, X: str, heuristic: str) -> list:
        """ Given a set of variables X, compute a good ordering for the elimination of X based on the heuristic

        :return: a topological ordering of the variables in the Bayesian network
        """

        interaction_graph = self.bn.get_interaction_graph()
        ordering = []
        var_count = len(X)

        if heuristic == "min-fill":
            for i in range(var_count):
                cost = {x: 0 for x in X}
                # Find interactions added by each variable
                for x in X:
                    x_neighbours = set(interaction_graph.neighbors(x))
                    for n in x_neighbours:
                        n_non_neighbours = nx.non_neighbors(interaction_graph, n)
                        cost[x] += len(x_neighbours.intersection(n_non_neighbours))
                # Find the variable with the minimum cost
                min_fill_node = min(cost, key=cost.get)
                ordering.append(min_fill_node)
                interaction_graph.remove_node(min_fill_node)
                X.remove(min_fill_node)
                        
        elif heuristic == "min-degree":
            for i in range(var_count):
                x_degrees = {interaction_graph.degree[x]: x for x in X}
                min_node = x_degrees[min(x_degrees)]
                ordering.append(min_node) # Queue the variable with the minimum degree to the ordering
                min_node_neighbours = list(interaction_graph.neighbors(min_node))
                for n in min_node_neighbours: # Sum out the variable with the minimum degree
                    e = list(itertools.zip_longest([], min_node_neighbours, fillvalue=n))
                    interaction_graph.add_edges_from(e)
                    interaction_graph.remove_edge(n, n)
                interaction_graph.remove_node(min_node)
                X.remove(min_node)
        
        return ordering
            

    def variable_elimination(self, X: List[str]) -> pd.DataFrame:
        """ Sum out a set of variables by using variable elimination (according to given order).

        :return: the resulting factor
        """

        tau = pd.DataFrame()
        visited = set()
        for x in X:
            if x not in visited and not tau.empty:
                tau = self.factor_multiplication(tau, self.bn.get_cpt(x))

            if tau.empty:
                tau = self.bn.get_cpt(x)

            to_visit = set(self.bn.get_children(x)) - set(visited)
            for child in to_visit:
                tau = self.factor_multiplication(tau, self.bn.get_cpt(child))
                visited.add(child)
            tau = self.marginalization(x, tau)
            visited.add(x)
        return tau

    def marginal_distributions(self, Q: List[str], e: dict, heuristic="min-degree") -> pd.DataFrame:
        """ Given query variables Q, evidence E, compute the marginal distributions.

        :param evidence: dictionary of evidence variables and their values
        :return: Posterior Marginal
        """

        e = pd.Series(e)
        to_be_eliminated = set(self.bn.get_all_variables()) - set(Q)
        ordering = self.ordering(to_be_eliminated, heuristic)
        ordering = ordering + Q

        tau = pd.DataFrame()
        visited = set()
        for x in ordering:
            if x not in visited and not tau.empty:
                tau = self.factor_multiplication(tau, self.bn.get_cpt(x))
            
            if tau.empty:
                tau = self.bn.get_cpt(x)
                if any([var in e.index for var in tau.columns]):
                    tau = self.bn.reduce_factor(e, tau)
            
            to_visit = set(self.bn.get_children(x)) - set(visited)
            for child in to_visit:
                child_reduced = self.bn.get_cpt(child)
                if any([var in e.index for var in child_reduced.columns]):
                    child_reduced = self.bn.reduce_factor(e, child_reduced)
                tau = self.factor_multiplication(tau, child_reduced)
                visited.add(child)
            if not x in Q:
                tau = self.marginalization(x, tau)
            visited.add(x)

        tau['p'] = tau['p'] / tau['p'].sum()
        return tau

    def MAP(self, Q: List[str], e: dict) -> dict:
        """ Computes the MAP instantiation + value of query variables Q, given a possibly empty evidence e. 

        :return: a dictionary of the MAP assignment of query variables in the Bayesian network
        """


    def MEP(self, e: dict) -> dict:
        """ Given evidence e, compute the MEP assignment of query variables in the Bayesian network.

        :return: a dictionary of the MEP assignment of query variables in the Bayesian network
        """
        pass

if __name__ == '__main__':
    # Playground for testing your code
    # rnr = BNReasoner('testing/lecture_example.bifxml')
    # rnr = BNReasoner('testing/lecture_example2.bifxml')
    rnr = BNReasoner('testing/lecture_example3.bifxml')
    a = rnr.bn 
    # a = BNReasoner('testing/lecture_example2.bifxml').bn
    #a.draw_structure()
    # print(a.get_all_cpts().keys())
    # print(a.get_all_cpts().values())
    # print(a.get_cpt('Sprinkler?'))
    # print(a.get_cpt('Winter?'))
    # print(a.get_all_variables())
    # print(a.get_cpt('Slippery Road?'))
    # rnr.marginalization('Wet Grass?', a.get_cpt('Wet Grass?'))
    # print(rnr.marginalization('Wet Grass?', a.get_cpt('Wet Grass?')))
    # print(rnr.maxing_out('Wet Grass?', a.get_cpt('Wet Grass?')))
    # print(rnr.factor_multiplication(a.get_cpt('Wet Grass?'), a.get_cpt('Sprinkler?')))
    # a.get_compatible_instantiations_table('B', {'A': 0, 'C': 1})
    # ['Winter?' : A, 'Sprinkler?' : B, 'Rain?' : C, 'Wet Grass?' : D, 'Slippery Road?' : E]
    # print(rnr.ordering(['Winter?', 'Sprinkler?', 'Rain?', 'Wet Grass?', 'Slippery Road?'], 'min-degree'))
    # print(rnr.ordering(['Winter?', 'Sprinkler?', 'Rain?', 'Wet Grass?', 'Slippery Road?'], 'min-fill'))

    # Manual test for VE
    # t = rnr.factor_multiplication(a.get_cpt('Sprinkler?'), a.get_cpt('Winter?'))
    # t = rnr.factor_multiplication(t, a.get_cpt('Rain?'))
    # # print(t)
    # t = rnr.marginalization('Winter?', t)
    # # print(t)
    # t = rnr.factor_multiplication(t, a.get_cpt('Wet Grass?'))
    # t = rnr.marginalization('Sprinkler?', t)
    # t = rnr.factor_multiplication(t, a.get_cpt('Slippery Road?'))
    # t = rnr.marginalization('Rain?', t)
    # print(t)
    # print(a.get_children('Winter?'))
    
    # order = rnr.ordering(['Winter?', 'Sprinkler?', 'Rain?'], 'min-degree')
    # print(order)
    # print(rnr.variable_elimination(order))
    # print(rnr.variable_elimination(['Sprinkler?', 'Slippery Road?']))

    # order = rnr.ordering(['A', 'B'], 'min-degree')
    # print(rnr.variable_elimination(order))

    # print(a.get_cpt('A'))
    # print(a.get_cpt('B'))
    # print(a.get_cpt('C'))

    # print(a.reduce_factor(pd.Series({'A': True}), a.get_cpt('A')))
    # print(a.reduce_factor(pd.Series({'A': True}), a.get_cpt('B')))
    # print(a.reduce_factor(pd.Series({'A': True}), a.get_cpt('C')))

    # Q = ['C']
    # to_be_eliminated = set(a.get_all_variables()) - set(Q)
    # print(to_be_eliminated)
    # posterior = rnr.marginal_distributions(['C'], {'A': True})
    # posterior = rnr.marginal_distributions(['B', 'C'], {'A': True})
    # posterior = rnr.marginal_distributions(['B', 'C'], None)
    
    # posterior = rnr.marginal_distributions(['O', 'X'], {"J": True})
    posterior = rnr.marginal_distributions(['O', 'X'], None)
    # posterior = rnr.marginal_distributions(['I', 'J'], {"O": True})
    print(posterior)

    # Checking VE for Q = [O, X]
    # Q = ['O', 'X']
    # wo_x = rnr.variable_elimination(['I', 'J', 'Y'])
    # c = ['I', 'J', 'Y']
    # for var in ['I', 'J', 'Y']:
    #     c = c + rnr.bn.get_children(var)
    # for var in Q:
    #     if var not in c:
    #         wo_x = rnr.factor_multiplication(wo_x, rnr.bn.get_cpt(var))
    # print(wo_x)

    # Checking VE for Q = [B, C]
    # Q = ['B', 'C']
    # wo_x = rnr.variable_elimination(['A'])
    # c = ['A']
    # for var in ['A']:
    #     c = c + rnr.bn.get_children(var)
    # for var in Q:
    #     if var not in c:
    #         wo_x = rnr.factor_multiplication(rnr.bn.get_cpt(var), wo_x)
    # print(wo_x)