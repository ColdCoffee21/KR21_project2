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

    def networkPrune(self, query: Union[str, list], evidence: Union[str, list, pd.Series], updateCPT=False, copy =False) -> BayesNet:
        """
        Edge-prunes and iteratively Node-prunes the Bayesian network s.t. queries of the form P(Q|E)
        can still be correctly calculated.
        :param query:    a variable (str) or a list of variables containing the query
        :param evidence: a series of assignments as tuples. E.g.: pd.Series({"A": True, "B": False})
        :returns:        The pruned version of the network w.r.t the query and evidence given.
        """
        # working out the different options for the input params
        if copy:
            net = copy.deepcopy(self.bn)
        else:
            net = self.bn
        if type(query) == str:
            query = [query]
        if type(evidence) == list and updateCPT:
            print("Evidence should be pd.Series in order to compute the CPTs")
        elif type(evidence) == str:
            evidence = [evidence]
            var_names = evidence
        elif type(evidence) == list:
            var_names = evidence
        else:
            var_names = evidence.index.values

        # Performs edge pruning
        for var in var_names:
            # Updates CPT of the evidence
            if updateCPT:
                cpt = net.get_cpt(var)
                if evidence[var]:
                    indexNames = cpt[cpt[var] == False].index
                    cpt.drop(indexNames, inplace=True)
                else:
                    indexNames = cpt[cpt[var] == True].index
                    cpt.drop(indexNames, inplace=True)

            children = net.get_children(var)
            for child in children:
                # Removes edges
                net.del_edge((var, child))
                # Updates CPTs for the children
                if updateCPT:
                    net.update_cpt(child, net.get_compatible_instantiations_table(evidence, net.get_cpt(child)))
                    net.get_cpt(child).drop(var, inplace=True, axis=1)

        # Performs node pruning iteratively
        union = list(var_names) + query
        options = [v for v in net.get_all_variables() if v not in union]

        done = False
        while not done:
            done = True
            for var in options:
                children = net.get_children(var)
                if children == []:  # If there are still leaf nodes, we delete them and iter one more time
                    net.del_var(var)
                    options.remove(var)
                    done = False
        # net.draw_structure()
        return net

    def d_separation(self, x: Union[str, list], y: Union[str, list], z: Union[str, list]) -> bool:
        """ determine whether X is d-separated of Y given Z.
        :param x: name of variable x (or list of variables)
        :param y: name of variable y (or list of variables)
        :param z: list of variables z (or list of variables)
        :return: True if x and y are d-separated given z, False otherwise
        """
        if type(x) == str:
            x = [x]
        if type(y) == str:
            y = [y]
        if type(z) == str:
            z = [z]
        union = x + y
        pruned = self.networkPrune(union, z)
        interaction_graph = pruned.get_interaction_graph()

        queue = []
        visited = set()

        # add the variables in set X to the queue
        for i in x:
            queue.append(i)
            visited.add(i)

        # perform BFS
        while queue != []:
            # get the next variable in the queue
            current_variable = queue.pop(0)
            # check if the current variable is in set Y
            if current_variable in y:
                return False

            # add the neighbors of the current variable to the queue
            for neighb in interaction_graph.neighbors(current_variable):
                if neighb not in visited:
                    queue.append(neighb)
                    visited.add(neighb)

        # if the BFS completes without finding a path from X to Y, return True
        return True

    def independence(self, x: Union[str, list], y: Union[str, list], z: Union[str, list]) -> bool:
        """ determine whether X is independent of Y given Z.

        :param x: name of variable x
        :param y: name of variable y
        :param z: list of variables z
        :return: True if x and y are independent given z, False otherwise
        """
        if type(x) == str:
            x = [x]
        if type(y) == str:
            y = [y]
        if type(z) == str:
            z = [z]
        e={}
        ey={}
        for i in z:
            e[i]=1
            ey[i]=1
        for i in y:
            ey[i]=1
        if not self.d_separation(x, y, z):
            return False
        else:
            if abs(self.marginal_distribution2(x,e)-self.marginal_distribution2(x,ey))<0.0000001:
                return True
            else:
                return False

    def marginalization(self, x: str, factor: pd.DataFrame) -> pd.DataFrame:
        """ 
        param x: name of variable x
        param factor: CPT of variables
        return: Factor in which X is summed-out
        """
        updated_factor_variables = [v for v in factor.columns if v != x]
        updated_factor_variables.pop()
        for var in updated_factor_variables:
            if "track_" in var:
                updated_factor_variables.remove(var)
        
        updated_factor = pd.DataFrame()
        if len(updated_factor_variables) == 0:
            updated_factor = factor['p'].sum()
            return updated_factor
    
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

    def maxing_out2(self, x: str, factor: pd.DataFrame) -> pd.DataFrame:
        """ v2: Keep track by renaming the column
        :param x: name of variable x
        :param factor: dictionary of evidence variables and their values
        :return: the CPT in which X is maxed-out
        """
        updated_factor_variables = [v for v in factor.columns if v not in ['p', x]]
        for var in updated_factor_variables:
            if "track_" in var:
                updated_factor_variables.remove(var)

        if len(updated_factor_variables) == 0:
            updated_factor = factor['p'].max()
            return updated_factor

        tracked_factor = factor.groupby(updated_factor_variables)['p'].idxmax()
        tracked_factor = factor.loc[tracked_factor] # Keep track of the instances where the max occurs
        tracked_factor.rename(columns = {x: 'track_' + x}, inplace = True) # For tracking purposes

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
        for var in common_variables:
            if "track_" in var:
                common_variables.remove(var)
        if common_variables == []:
            print("Factor Multiplication Warning: No common variables")
            g.rename(columns = {'p1':'p'}, inplace = True)
            return pd.DataFrame()

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

    def variable_elimination_fix(self, X: List[str]) -> pd.DataFrame:
        """ Sum out a set of variables by using variable elimination (according to given order).
        
        Fix: If the factor multiplication is empty, try later
        :return: the resulting factor
        """

        tau = pd.DataFrame()
        visited = set()
        later = []
        for x in X:
            if x not in visited and not tau.empty:
                tau_ = self.factor_multiplication(tau, self.bn.get_cpt(x))
                if tau_.empty:
                    later.append(x)
                    continue
                tau = tau_

            if tau.empty:
                tau = self.bn.get_cpt(x)

            to_visit = set(self.bn.get_children(x)) - set(visited)
            for child in to_visit:
                tau = self.factor_multiplication(tau, self.bn.get_cpt(child))
                visited.add(child)
            
            for v in later:
                if v in tau.columns:
                    tau = self.factor_multiplication(tau, self.bn.get_cpt(v))
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
    
    def marginal_distribution2(self, Q: List[str], e: dict, heuristic="min-degree") -> pd.DataFrame:
        """ Given query variables Q, evidence E, compute the marginal distribution.
        v2: Computes Marginal Distribution inplace

        :param evidence: dictionary of evidence variables and their values
        :return: Posterior Marginal
        """

        e = pd.Series(e)
        to_be_eliminated = set(self.bn.get_all_variables()) - set(Q)
        ordering = self.ordering(to_be_eliminated, heuristic)
        
        # Reduce all cpts wrt evidence
        cpts = self.bn.get_all_cpts()
        for var, cpt in cpts.items():
            if any([var in e.index for var in cpt.columns]):
                cpt = self.bn.reduce_factor(e, cpt)
            self.bn.update_cpt(var, cpt)

        tau = self.variable_elimination(ordering)
        visited = list(ordering)    
        for var in ordering:
            visited = visited + self.bn.get_children(var)

        to_visit = [var for var in Q if var not in visited]
        if tau.empty:
            tau = self.bn.get_cpt(to_visit[0])
            to_visit = to_visit[1:]
        for var in to_visit:
            tau = self.factor_multiplication(tau, self.bn.get_cpt(var))

        # Change to sum-out
        tau['p'] = tau['p'] / tau['p'].sum() # Normalize
        return tau

    def MAP(self, Q: List[str], e: dict) -> dict:
        """ Computes the MAP instantiation + value of query variables Q, given a possibly empty evidence e. 

        :return: a dictionary of the MAP assignment of query variables in the Bayesian network
        """
        tau = self.marginal_distributions(Q, e)
        tau = tau.sort_values(by=['p'])

        return tau.iloc[-1].to_dict()


    def MPE(self, e: dict) -> dict:
        """ Given evidence e, compute the MEP assignment of query variables in the Bayesian network.

        :return: a dictionary of the MEP assignment of query variables in the Bayesian network
        """
        # self.networkPrune(self.bn.get_all_variables(), evidence = pd.Series(e), updateCPT=True)
        # self.bn.draw_structure()
        # print(self.bn.get_all_cpts())

        q_wo_e = set(self.bn.get_all_variables()) - set(e.keys())
        ordering = self.ordering(q_wo_e, "min-degree")
        ordering = ordering + list(e.keys())
        e = pd.Series(e)

        tau = pd.DataFrame()
        visited = set()
        for x in ordering:
            if x not in visited and not tau.empty:
                tau = self.factor_multiplication(tau, self.bn.get_cpt(x))
            
            if tau.empty:
                tau = self.bn.get_cpt(x)
                if any([var in e.index for var in tau.columns]):
                    # tau = self.bn.reduce_factor(e, tau)
                    tau = self.bn.get_compatible_instantiations_table(e, tau)
            
            to_visit = set(self.bn.get_children(x)) - set(visited)
            for child in to_visit:
                child_reduced = self.bn.get_cpt(child)
                if any([var in e.index for var in child_reduced.columns]):
                    # child_reduced = self.bn.reduce_factor(e, child_reduced)
                    child_reduced = self.bn.get_compatible_instantiations_table(e, child_reduced)
                tau = self.factor_multiplication(tau, child_reduced)
                visited.add(child)
            tau = self.maxing_out2(x, tau)
            visited.add(x)

        # Renaming
        tau_cols = list(tau.columns)
        tau_cols.remove('p')
        tau.rename(columns={col: col.split('_')[1] for col in tau_cols}, inplace=True)
        
        return tau

if __name__ == '__main__':
    # Playground for testing your code
    # rnr = BNReasoner('testing/lecture_example.bifxml')
    rnr = BNReasoner('testing/lecture_example2.bifxml')
    # rnr = BNReasoner('testing/lecture_example3.bifxml')
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
    
    # t = rnr.maxing_out('Wet Grass?', a.get_cpt('Wet Grass?'))
    # t = rnr.maxing_out('Sprinkler?', t)

    # t = rnr.maxing_out2('Wet Grass?', a.get_cpt('Wet Grass?'))
    # t = rnr.maxing_out2('Sprinkler?', t)

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
    # posterior = rnr.marginal_distributions(['A', 'B', 'C'], None)
    
    # posterior = rnr.marginal_distributions(['O', 'X'], {"J": True})
    # posterior = rnr.marginal_distributions(['O', 'X'], None)
    # posterior = rnr.marginal_distributions(['I', 'J'], {"O": True})
    # print(posterior)

    # posterior = rnr.marginal_distribution2(['C'], {'A': True})
    # posterior = rnr.marginal_distribution2(['B', 'C'], {'A': True})
    # posterior = rnr.marginal_distribution2(['A', 'B', 'C'], None)
    # posterior = rnr.marginal_distribution2(['O', 'X'], None)
    # posterior = rnr.marginal_distribution2(['I', 'J'], {"O": True})
    # posterior = rnr.marginal_distribution2(['O', 'X'], {"J": True})
    # print(posterior)

    # test = rnr.marginalization('Wet Grass?', a.get_cpt('Wet Grass?')) 
    # print(test)
    # test = rnr.marginalization('Sprinkler?', test)
    # print(test)
    # test = rnr.marginalization('Rain?', test)
    # print(test)

    # MAP
    # map_res = rnr.MAP(['I', 'J'], {"O": True})
    # print(map_res)

    # MPE
    mpe_res = rnr.MPE({"J": True, "O": False})
    print(mpe_res)

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