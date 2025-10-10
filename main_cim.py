from BEM_AC_NVM_PDN import PDN
from CIM_DC_RES import main_res
from input_AH import input_path, stackup_path, R11_PowerSI

from pdn_io.spd_parser import parse_spd
from pdn_io.stackup_parser import read_stackup
 
def gen_brd_data(
    brd,
    spd_path: str,
    stackup_path: str,
):
    # --- 1) Parse SPD (board shapes, vias, layers, etc.) ---
    result = parse_spd(brd, 
                       spd_path, ground_net="gnd", power_net="pwr", 
                       ic_port_tag="ic_port", decap_port_tag="decap_port",
                       verbose=True)
    
    # --- 2) Stackup ---
    die_t, er_list, d_r = read_stackup(stackup_path)
    brd.er_list = er_list
    brd.die_t = die_t
    brd.d_r = d_r

    # --- 3) IC R11 computation ---
    res_matrix = main_res(
        brd=brd,
        die_t=die_t,
        d=d_r,
        stackup=brd.stackup,
        start_layer=brd.start_layers,
        stop_layer=brd.stop_layers,
        decap_via_type=brd.decap_via_type,
        decap_via_xy=brd.decap_via_xy,
        decap_via_loc=brd.decap_via_loc,
        ic_via_xy=brd.ic_via_xy,
        ic_via_loc=brd.ic_via_loc,
        ic_via_type=brd.ic_via_type
    )

    return res_matrix

if __name__ == '__main__':

    brd = PDN()

    R11_Python = gen_brd_data(
        brd=brd,
        spd_path=input_path,
        stackup_path=stackup_path,
    )
    print("Python IC R11: ", R11_Python*1e3, "mohm")
    print("PowerSI IC R11: ", float(R11_PowerSI)*1e3, "mohm")



    
    
