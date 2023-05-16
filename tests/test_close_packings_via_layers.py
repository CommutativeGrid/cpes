from cpes.close_packings_via_layers import *


def test_Layer():
    layer = Layer(nx=3, ny=3, type="A")
    assert len(layer.coords) == 9


def test_FaceCenteredCubic():
    fcc = FaceCenteredCubic(num=8, radius=1, num_vector="auto")
    assert abs(fcc.distance_array()[3] - 3.4641) < 0.001
    #   will encounter
    #   /opt/anaconda3/envs/TDA/lib/python3.9/site-packages/sklearn/utils/extmath.py:152: RuntimeWarning: invalid value encountered in matmul
    # ret = a @ b
    #  if num is <=7


def test_HexagonalClosePacking():
    hcp = HexagonalClosePacking(num=8, radius=1, num_vector="auto")
    # notice that the diagram in
    # https://msestudent.com/wp-content/uploads/2021/02/OPT-HCP-coordination-number.svg
    # is not correct. The distance between the first and the second atom is 2.828
    # instead of c=3.2659 when r=1.
    # That distance can be realized by (0,0,0) and
    assert abs(hcp.distance_array()[2] - 2.828) < 0.001


def test_neighboursCounting_FCC():
    fcc = FaceCenteredCubic(10)
    assert len(fcc.df.loc[fcc.df["neighbours"].apply(lambda x: len(x))==12])==492
    assert max(set(fcc.df["neighbours"].apply(lambda x: len(x))))==12
    
def test_neighboursCounting_HCP():
    hcp = HexagonalClosePacking(10)
    assert len(hcp.df.loc[hcp.df["neighbours"].apply(lambda x: len(x))==12])==512
    assert max(set(hcp.df["neighbours"].apply(lambda x: len(x))))==12

