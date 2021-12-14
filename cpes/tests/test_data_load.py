from cpes.data_load import *
import pytest

# def test_fcc_au_cart_len_online():
#         fcc_au_online = fcc_au_cart(mode="online")
        
#         assert (len(fcc_au_online.normalized) >= 4000)

# def test_fcc_au_cart_len_offline():
#         fcc_au_offline = fcc_au_cart(mode='offline')
#         assert (len(fcc_au_offline.normalized) >= 4000)
@pytest.mark.data_download
@pytest.mark.parametrize("mode", ['online', 'offline'])
def test_fcc_au_cart(mode):
        fcc_au_cart_data = fcc_au_cart(mode=mode)
        assert (len(fcc_au_cart_data.normalized) >= 4000 and len(fcc_au_cart_data.original)>=4000)

@pytest.mark.data_download
@pytest.mark.parametrize("mode", ['online', 'offline'])
def test_hcp_ru_frac(mode):
        hcp_ru_frac_data = hcp_ru_frac(mode=mode)
        assert (len(hcp_ru_frac_data.normalized) >= 8000)

@pytest.mark.data_download
@pytest.mark.parametrize("mode", ['online', 'offline'])
def test_fcc_si_frac(mode):
        fcc_si_frac_data = fcc_si_frac(mode=mode)
        assert (len(fcc_si_frac_data.normalized) >= 1500)