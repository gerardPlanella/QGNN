import numpy as np

class Factor(object):
    
    def __init__(self, variables=None, potentials=None):
        """
            A tensor associated with a list of variables / indices. Without any parameters the resulting empty factor does not modify another factor through multiplication.
            
            Parameters
                potentials (np.ndarray): A multidimensional array. The array's axes should be ordered according to the variable list.
                variables (str): A list containing the variable names of all variables this factor should represent.
        """
        if potentials is None:
            potentials = np.array(1)
        if variables is None:
            variables = []
        if isinstance(variables, str):
            variables = list(variables)
        self.data = potentials
        self.variables = variables

    def potential(self, instantiation):
        """ The current potential for the specified instantiation of all contained variables. """
        index = [instantiation[v] if v in instantiation else slice(None) for v in self.variables]
        return np.squeeze(np.copy(self.data[np.ix_(*index)]))

    def marginalize(self, variables):
        """
            Creates a new factor where the specified variables are summed out.
            
            Parameters
                variables (list[str]): A list containing the names of all the variables that should be summed out.

            Returns
                Factor: A Factor where the specified variables have been summed out from this Factor.
        """
        if not isinstance(variables, (list,set)):
            variables = [variables]

        res = self.copy()
        for v in variables:
            # Sum out the corresponding dimension for each variable and remove the variable from the list
            res.data = np.sum(res.data, axis=res.variables.index(v))
            res.variables.remove(v)

        return res

    def multiply(self, other_factor):
        """
            Creates a new factor, which is the resulting product of multiplying
            this factor with the given other factor. The initial factors are
            not modified.

            Parameters
                other_factor (Factor): The factor to multiply with.

            Returns
                Factor: A new factor which is the product of this factor and the other factor.
        """
        def add_indices(target: Factor, like: Factor, init=0):
            """
                Adds new dimensions to the target array, so that it has the same number of dimensions as the like array.
                The new dimensions are added to the end of the array.

                Parameters
                    target (np.ndarray): The array to add dimensions to.
                    like (np.ndarray): The array to mimic the dimensions of.
                    init (float): The initial value for the new dimensions.

                Returns
                    np.ndarray: The target array with the new dimensions added.
            """
            extra_vars = set(like.variables) - set(target.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(target.variables) + [np.newaxis] * len(extra_vars)
                target.data = target.data[*slice_]
                # Update the variables list
                target.variables.extend(extra_vars)

        # Shortcuts for trivial factors
        if len(self.variables) == 0:
            res = other_factor.copy()
            res.data = self.data * res.data
            return res

        if len(other_factor.variables) == 0:
            res = self.copy()
            res.data = res.data * other_factor.data
            return res

        res = self.copy()
        f2 = other_factor.copy()

        # Extend res factor by the new variables from the other factor
        add_indices(res, f2)

        # Extend other factor by the new variables from the res factor, to have the same dimensions
        add_indices(f2, res)

        # Rearrange f2 data so that dimensions align to the order in res
        f2.transpose(res.variables, inplace=True)

        # Multiply the potentials element-wise
        # See Definition 6.3 in "Modeling and Reasoning with Bayesian Networks" - Adnan Darwiche Chapter 6    
        res.data = res.data * f2.data

        return res

    def __mul__(self, other):
        return self.multiply(other)

    def reduce(self, evidence):
        """
            Creates a new factor which has been reduced to conform to the 
            provided evidence.

            Parameters
                evidence (dict): A dictionary containing variable:outcome pairs

            Returns
                Factor: A new factor which has been reduced to conform to the provided evidence.
        """
        # Construct the index for the desired potential
        res = self.copy()
        index = [evidence[v] if v in evidence else slice(None) for v in self.variables]

        # We use the access mask provided by np.ix_ to prevent the cells conforming to the evidence from being multiplied by 0
        tmp = np.zeros(res.data.shape)
        tmp[np.ix_(*index)] = 1
        res.data *= tmp
        return res
    
    def transpose(self, new_variable_order, inplace=False):
        """
            Transposes the potentials of this factor according to the provided axes.

            Parameters
                axes (list[str]): The new order of the variables.

            Returns
                Factor: A new factor with the transposed potentials.
        """
        if inplace:
            res = self
        else:
            res = self.copy()
        res.data = np.transpose(res.data, [res.variables.index(v) for v in new_variable_order])
        res.variables = new_variable_order
        return res
    
    def normalize(self, inplace=False):
        """
            Normalizes the potentials of this factor.
        """
        if inplace:
            res = self
        else:
            res = self.copy()
        res.data = res.data / np.sum(res.data)
        return res

    def __str__(self):
        if np.prod(self.data.shape) < 20:
            return "Factor: " + str(self.variables) + " " + str(self.data.shape) + " " + str(self.data)
        return "Factor: " + str(self.variables) + " " + str(self.data.shape)
    
    def __repr__(self):
        return self.__str__()

    def copy(self):
        """ Creates a deep copy of this factor. """
        res = Factor()
        res.data = np.copy(self.data)
        res.variables = list(self.variables)
        return res