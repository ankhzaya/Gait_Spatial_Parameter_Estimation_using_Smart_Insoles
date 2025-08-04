def add_value_dict_SL(f_fn, R_SL, L_SL, out):
    """Append output in the list of dictionary"""
    if f_fn in R_SL.keys():
        R_SL[f_fn].append(out[0])
        L_SL[f_fn].append(out[1])
    else:
        R_SL[f_fn] = [out[0]]
        L_SL[f_fn] = [out[1]]

    return R_SL, L_SL

def add_value_spat_dict(f_fn, R_StrL, L_StrL, R_SL, L_SL, R_StrW, L_StrW, R_SW, L_SW, out):
    """Append output in the list of dictionary"""
    if f_fn in R_StrL.keys():
        R_StrL[f_fn].append(out[0])
        L_StrL[f_fn].append(out[1])
        R_SL[f_fn].append(out[2])
        L_SL[f_fn].append(out[3])
        R_StrW[f_fn].append(out[4])
        L_StrW[f_fn].append(out[5])
        R_SW[f_fn].append(out[6])
        L_SW[f_fn].append(out[7])
    else:
        R_StrL[f_fn] = [out[0]]
        L_StrL[f_fn] = [out[1]]
        R_SL[f_fn] = [out[2]]
        L_SL[f_fn] = [out[3]]
        R_StrW[f_fn] = [out[4]]
        L_StrW[f_fn] = [out[5]]
        R_SW[f_fn] = [out[6]]
        L_SW[f_fn] = [out[7]]

    return R_StrL, L_StrL, R_SL, L_SL, R_StrW, L_StrW, R_SW, L_SW

def add_value_dict(f_fn, R_SL, L_SL, R_SW, L_SW, out):
    """Append output in the list of dictionary"""
    if f_fn in R_SL.keys():
        R_SL[f_fn].append(out[0])
        L_SL[f_fn].append(out[1])
        R_SW[f_fn].append(out[2])
        L_SW[f_fn].append(out[3])
    else:
        R_SL[f_fn] = [out[0]]
        L_SL[f_fn] = [out[1]]
        R_SW[f_fn] = [out[2]]
        L_SW[f_fn] = [out[3]]

    return R_SL, L_SL, R_SW, L_SW

def merge_dicts(dict, L_HS, L_TO, R_HS, R_TO):
    dict['L_HS'] = L_HS
    dict['L_TO'] = L_TO
    dict['R_HS'] = R_HS
    dict['R_TO'] = R_TO
    return dict