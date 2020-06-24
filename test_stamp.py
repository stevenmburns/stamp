import numpy as np

class GMat:
    def __init__(self):
        self.tbl = {}
        self.n = 0
        self.t = .01
        self.h = 1
        self.v = 1

    def new_node(self):
        result = self.n
        self.n += 1
        return result

    def incr( self, i, j, r):
        if i < 0 or j < 0: return
        assert i < self.n and j < self.n
        if (i,j) not in self.tbl:
            self.tbl[(i,j)] = 0
        self.tbl[(i,j)] += r

    def stamp( self, i, j, r):
        self.incr( i, i,  r)
        self.incr( j, j,  r)
        self.incr( i, j, -r)
        self.incr( j, i, -r)

    def semantic( self):
        self.g = np.zeros( (self.n, self.n))
        for (k,v) in self.tbl.items():
            (i,j) = k
            self.g[i,j] = v
        self.ginv = np.linalg.inv(self.g)

    def compute_resistance_to_ground( self, idx):
        # I = G V
        # G^-1 I = V
        I = np.zeros( (self.n,))
        I[idx] = 1
        return self.ginv.dot(I)[idx]

    @staticmethod
    def par( x, y):
        return x*y/(x+y)

    def row_conductance(self):
        t1 = 2 * self.par( self.t, 0.5*self.h) + self.t
        t2 = 2 * self.par( self.t, self.h)
        return t1, t2

    def cc_array_conductance(self, t1, t2):
        q1 = self.par( t1, self.v) + t2
        q2 = self.par( q1, self.v) + t2
        return self.par( q2, self.v) + t1

    def row(self):
        # O I O I O
        outer_l = self.new_node()
        outer_m = self.new_node()
        outer_r = self.new_node()
        inner_l = self.new_node()
        inner_m = self.new_node()
        inner_r = self.new_node()

        self.stamp( -1, outer_l, self.t)
        self.stamp( -1, outer_m, self.t)
        self.stamp( -1, outer_r, self.t)
        self.stamp( outer_l, outer_m, 0.5*self.h)
        self.stamp( outer_m, outer_r, 0.5*self.h)

        self.stamp( -1, inner_l, self.t)
        self.stamp( -1, inner_r, self.t)
        self.stamp( inner_l, inner_m, self.h)
        self.stamp( inner_m, inner_r, self.h)

        return outer_m,inner_m

    def cc_array( self):
        # A B A B A
        # B A B A B
        # B A B A B    
        # A B A B A

        A0, B0 = self.row()
        B1, A1 = self.row()
        B2, A2 = self.row()
        A3, B3 = self.row()

        self.stamp( A0, A1, self.v)
        self.stamp( A1, A2, self.v)
        self.stamp( A2, A3, self.v)

        self.stamp( B0, B1, self.v)
        self.stamp( B1, B2, self.v)
        self.stamp( B2, B3, self.v)

        return A0, B0

    def res_divider( self, n):
        nodes = [ self.new_node() for i in range( n)]
        self.stamp( -1, nodes[0], 1)
        for i in range(n-1):
            self.stamp( nodes[i], nodes[i+1], 1)
        return nodes[-1]

    def parallel( self, n):
        out = self.new_node()
        for i in range(n):
            self.stamp( -1, out, 1)
        return out

    def prnt( self, indices):
        for idx in indices:
            print( idx, self.compute_resistance_to_ground( idx))


def test_divider():
    g = GMat()

    last = g.res_divider( 10)
    g.semantic()
    g.prnt( [last])

    assert np.isclose( g.compute_resistance_to_ground( last), 10.0)

def test_parallel():
    g = GMat()

    last = g.parallel( 10)
    g.semantic()
    g.prnt( [last])

    assert np.isclose( g.compute_resistance_to_ground( last), 0.1)


def test_cc_array():
    g = GMat()

    A0, B0 = g.cc_array()
    g.semantic()
    g.prnt( [A0, B0])

    t1, t2 = g.row_conductance()
    g1, g2 = g.cc_array_conductance( t1, t2), g.cc_array_conductance( t2, t1)

    assert np.isclose( g.compute_resistance_to_ground( A0), 1/g1)
    assert np.isclose( g.compute_resistance_to_ground( B0), 1/g2)
