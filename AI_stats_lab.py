import numpy as np
# Question 1: Continuous pair on the unit square

def joint_cdf_unit_square(x, y):

    if x <= 0 or y <= 0:
        return 0
    elif 0 < x < 1 and 0 < y < 1:
        return x * y
    elif 0 < x < 1 and y >= 1:
        return x
    elif x >= 1 and 0 < y < 1:
        return y
    else:  
        return 1

def rectangle_probability(x1, x2, y1, y2):
   
    return (
        joint_cdf_unit_square(x2, y2)
        - joint_cdf_unit_square(x1, y2)
        - joint_cdf_unit_square(x2, y1)
        + joint_cdf_unit_square(x1, y1)
    )

def marginal_fx_unit_square(x):
   
    if 0 < x < 1:
        return 1
    else:
        return 0

def marginal_fy_unit_square(y):
   
    if 0 < y < 1:
        return 1
    else:
        return 0

# Question 2: Joint PMF, marginals, independence

def joint_pmf_heads(x, y):
   
    if x == 0 and y == 0:
        return 1/4
    elif x == 0 and y == 1:
        return 1/4
    elif x == 0 and y == 2:
        return 0
    elif x == 1 and y == 0:
        return 0
    elif x == 1 and y == 1:
        return 1/4
    elif x == 1 and y == 2:
        return 1/4
    else:
        return 0

def marginal_px_heads(x):
   
    total = 0
    for y in [0, 1, 2]:
        total += joint_pmf_heads(x, y)
    return total

def marginal_py_heads(y):
   
    total = 0
    for x in [0, 1]:
        total += joint_pmf_heads(x, y)
    return total

def check_independence_heads():
   
    for x in [0, 1]:
        for y in [0, 1, 2]:
            joint = joint_pmf_heads(x, y)
            product = marginal_px_heads(x) * marginal_py_heads(y)
            if not np.isclose(joint, product):
                return False
    return True
