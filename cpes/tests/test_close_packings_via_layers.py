from cpes.close_packings_via_layers import *


def test_layer():
    l=layer(nx=3,ny=3,type='A')
    assert (len(l.coords)==9)

def test_face_centered_cubic():
    fcc=face_centered_cubic(num=5,radius=1,num_vector='auto')
    assert (abs(fcc.distance_array()[3]-3.4641)<0.001)

def test_hexagonal_close_packing():
    hcp=hexagonal_close_packing(num=5,radius=1,num_vector='auto')
    # notice that the diagram in https://msestudent.com/wp-content/uploads/2021/02/OPT-HCP-coordination-number.svg
    # is not correct. The distance between the first and the second atom is 2.828 instead of c=3.2659 when r=1.
    # That distance can be realized by (0,0,0) and 
    assert (abs(hcp.distance_array()[2]-2.828)<0.001)
    

