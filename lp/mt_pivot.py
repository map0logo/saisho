# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 00:37:11 2013

@author: mapologo

Program Assignment for Coursera's Linear and Integer Programming


"""

import numpy as np


def get_values(line, num_type="int"):
    """(str) -> list

    Return a list of numbers of type num_type from a line of values
    separated with spaces

    >>> get_values("3 4\n")
    [3, 4]
    >>> get_values("1 -1 0 -1\n")
    [1, -1, 0, -1]
    >>> get_values("-1.0 3.0 -1.0 -2.0 \n", "float")
    [-1.0, 3.0, -1.0, -2.0]
    """

    convert = {"int": int, "float": float, "complex": complex}
    return [convert[num_type](item) for item in line.split()]


def check_array(array, shape, dtype):
    """(array, tuple, dtype) -> boolean

    Check if array is of given shape and dtype.

    """

    return isinstance(array, np.ndarray) and array.shape == shape and array.dtype == dtype


class SimplexPivoting():
    """
    Simplex pivoting dictionary
    
    m: number of constraints
    n: number of variables
    B: array of basic indices (m integers)
    N: array of non-basic indices (n integers)
    b: array of rhs (values of basic variables)
    a: matrix of constraints coefficients (m x n)
    c: array of objective function coeffcients
    z: objective function value
    num_iter: number of completed iterations
        
    enter_var: index of entering variable
    enter: position of enter_var in N
    leave_var: index of leaving variable
    leave: position of leave_var in B
    unbounded: True if there isn't leaving variable
    optimal: True if there isn't entering variable

    """

    def check(self):
        """(SimplexPivoting) -> bool

        Check Pivot Dictionary parameters.
        todo: check dimensions

        """

        print(isinstance(self.m, int))
        print(isinstance(self.n, int))
        print(check_array(self.B, (self.m, ), int))
        print(check_array(self.N, (self.n, ), int))
        print(check_array(self.b, (self.m, ), float))
        print(check_array(self.A, (self.m, self.n), float))
        print(isinstance(self.z, float))
        print(check_array(self.c, (self.n, ), float))

    def read_dfile(self, fdict):
        """(SimplexPivoting, text) -> NoneType

        Get dictionary parameters from a fdict text file with the
        following format:

        [Line 1] m n
        [Line 2] B1 B2 ... Bm [list of basic indices m integers]
        [Line 3] N1 N2 ... Nn [list of non-basic indices n integers]
        [Line 4] b1 .. bm (m floating point numbers)
        [Line 5] a11 ... a1n (first row of A matrix)
        ....
        [Line m+4] am1 ... amn (mth row of A matrix)
        [Line m+5] z0 c1 .. cn (objective coefficients
                   (n+1 floating point numbers)) 
        """

        fd = open(fdict, "r")
        self.m, self.n = get_values(fd.readline())
        self.B = np.array(get_values(fd.readline()))
        self.N = np.array(get_values(fd.readline()))
        self.b = np.array(get_values(fd.readline(), "float"))
        self.A = np.array(get_values(fd.readline(), "float"))
        for i in range(self.m - 1):
            self.A = np.vstack((self.A, get_values(fd.readline(), "float")))
        Z = np.array(get_values(fd.readline(), "float"))
        self.z = Z[0]
        self.c = Z[1:]
        self.unbounded = False
        self.optimal = False
        fd.close()

    def __repr__(self):
        """(SimplexPivoting) -> str


        """

        s = 'm: {} n: {}\n'.format(self.m, self.n)
        s = s + 'B:\n{}\n'.format(self.B)
        s = s + 'N:\n{}\n'.format(self.N)
        s = s + 'b:\n{}\n'.format(self.b)
        s = s + 'A:\n{}\n'.format(self.A)
        s = s + 'z:\n{}\n'.format(self.z)
        s = s + 'c:\n{}\n'.format(self.c)
        return s

    def __entering(self):
        """(SimplexPivoting) -> int

        Selects and returns the entering variable and set "enter",
        its position in N array.

        Use Bland's rule to prevent cycling: Choose the
        lowest-numbered nonbasic with a positive c coeff.

        """

        try:
            self.enter_var = self.N[self.c > 0].min()
            self.enter = np.where(self.N == self.enter_var)[0][0]
        except ValueError:
            self.optimal = True
            return False

        return True

    def __leaving(self):
        """(SimplexPivoting) -> int

        After entering is executed, selects and returns the leaving
        variable and set "leave", its position in B array.

        Use Bland's rule to prevent cycling: If the minimum ratio is
        shared by several rows, choose the lowest-numbered one of
        them.

        """

        enter_col = -self.A[:, self.enter]
        enter_col = np.ma.masked_array(enter_col, enter_col <= 0)
        ratios = self.b / enter_col
        if ratios.count():
            min_ratios = np.ma.filled(ratios == ratios.min(), False)
            # if masked when ratios.min() == 0, ratios == ratios.min()
            # return all True
            self.leave_var = self.B[min_ratios].min()
            self.leave = np.where(self.B == self.leave_var)[0][0]
            return True
        else:
            self.unbounded = True
            return False

    def pivoting(self):
        """(SimplexPivoting) -> float

        Makes one pivoting.
        
        todo: return properly Unbouded and optimal as exceptions

        """
        if not self.__entering():
            return "OPTIMAL"
        if not self.__leaving():
            return "UNBOUNDED"
        pivot = -self.A[self.leave, self.enter]
        self.A[self.leave, self.enter] = -1.0
        self.b[self.leave] = self.b[self.leave] / pivot
        self.A[self.leave, :] = self.A[self.leave, :] / pivot
        idxB = np.arange(self.m)
        for i in idxB[idxB != self.leave]:
            pivot = self.A[i, self.enter]
            self.A[i, self.enter] = 0.0
            self.A[i, :] = self.A[self.leave, :] * pivot + self.A[i, :]
            self.b[i] = self.b[self.leave] * pivot + self.b[i]
        pivot = self.c[self.enter]
        self.c[self.enter] = 0.0
        self.c = self.A[self.leave, :] * pivot + self.c
        self.z = self.b[self.leave] * pivot + self.z
        # After pivoting exchange entering and leaving variables in
        # N and B
        self.B[self.leave] = self.enter_var
        self.N[self.enter] = self.leave_var

        return "STEP"

    def output4step_one(self, filename=""):
        """(SimplexPivoting) -> str

        Return the output for "Program the Pivot: Step 1"
        If filename is given writes output for this file.

        """

        if not self.unbounded:
            out_str = "{}\n{}\n{}".format(self.enter_var,
                                          self.leave_var,
                                          self.z)
        else:
            out_str = "UNBOUNDED"
        if filename:
            out = file(filename, "w")
            out.write(out_str)
            out.close()
        return out_str

    def iterate(self):
        """(SimplexPivoting) -> str
        
        Iterates over a feasible dictionary
        
        """
        self.num_iter = 0
        while True:
            state = self.pivoting()
            if state == "OPTIMAL" or state == "UNBOUNDED":
                break
            self.num_iter += 1
        return state

    def output4step_two(self, filename=""):
        """(SimplexPivoting) -> str

        Return the output for "Program the Pivot: Step 2"
        If filename is given writes output for this file.

        """

        if not self.unbounded:
            out_str = "{}\n{}".format(self.z,
                                      self.num_iter)
        else:
            out_str = "UNBOUNDED"
        if filename:
            out = file(filename, "w")
            out.write(out_str)
            out.close()
        return out_str

    def initialize(self):
        """(SimplexPivoting) -> str
        
        Iterates over a non-feasible dictionary
        
        """
        self.num_iter = 0
        while True:
            state = self.pivoting()
            if state == "OPTIMAL" or state == "UNBOUNDED":
                break
            self.num_iter += 1
        return state

    def __add_variable(self, index):
        """

        """
        pass
        
        

