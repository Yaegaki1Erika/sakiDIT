import numpy as np

enable_teacache=True
rel_l1_thresh=0.15
coefficients_dict = {
    "CogVideoX-2b":[-3.10658903e+01,  2.54732368e+01, -5.92380459e+00,  1.75769064e+00, -3.61568434e-03],
    "CogVideoX-5b":[-1.53880483e+03,  8.43202495e+02, -1.34363087e+02,  7.97131516e+00, -5.23162339e-02],
    "CogVideoX-5b-I2V":[-1.53880483e+03,  8.43202495e+02, -1.34363087e+02,  7.97131516e+00, -5.23162339e-02],
    "CogVideoX1.5-5B":[ 2.50210439e+02, -1.65061612e+02,  3.57804877e+01, -7.81551492e-01, 3.58559703e-02],
    "CogVideoX1.5-5B-I2V":[ 1.22842302e+02, -1.04088754e+02,  2.62981677e+01, -3.06009921e-01, 3.71213220e-02],
}


def get_should_calc(
        cnt: int = 0,
        accumulated_rel_l1_distance: float = 0,
        emb= None,
        previous_modulated_input= None,
    ):
    # print("pre is ",accumulated_rel_l1_distance)
    if enable_teacache == False:
        return accumulated_rel_l1_distance,True
    if cnt==0 or cnt ==29 or cnt==15 or cnt==2 or cnt==10 or cnt==20:
        return accumulated_rel_l1_distance,True
    rescale_func = np.poly1d(coefficients_dict["CogVideoX-2b"])
    accumulated_rel_l1_distance += rescale_func(((emb-previous_modulated_input).abs().mean() / previous_modulated_input.abs().mean()).cpu().item())
    # print(accumulated_rel_l1_distance)
    if accumulated_rel_l1_distance < rel_l1_thresh:
        should_calc = False
    else:
        should_calc = True
    return accumulated_rel_l1_distance,should_calc