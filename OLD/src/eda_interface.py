import pyAether as ae

def extract_params(lib, cell, view, param_file_name):
    cv = ae.dbOpenCV(lib, cell, view)
    if(cv is None):
        print("dbOpenCV failed !!!")
        return

    with open(param_file_name, 'w') as param_file:
        insts = cv.instances
        for inst in insts:
            # print("=====start ", inst.name, "=====")
            if (inst.name[0] == "R"  or  inst.name[0] == 'r'):
                retVal = ae.aeEmyInstGetParameter('segW', inst)
                strlog = f"parameter, {inst.name}_segW, {retVal}\n"
                param_file.write(strlog)

                retVal = ae.aeEmyInstGetParameter('segL', inst)
                strlog = f"parameter, {inst.name}_segL, {retVal}\n"
                param_file.write(strlog)

            elif (inst.name[0] == "C"  or  inst.name[0] == 'c'):
                retVal = ae.aeEmyInstGetParameter('l', inst)
                strlog = f"parameter, {inst.name}_l, {retVal}\n"
                param_file.write(strlog)

            elif (str(inst.name).startswith("PM")  or  str(inst.name).startswith("NM")):
                retVal = ae.aeEmyInstGetParameter('m', inst)
                strlog = f"parameter, {inst.name}_m, {retVal}\n"
                param_file.write(strlog)

                retVal = ae.aeEmyInstGetParameter('l', inst)
                strlog = f"parameter, {inst.name}_l, {retVal}\n"
                param_file.write(strlog)

                retVal = ae.aeEmyInstGetParameter('fw', inst)
                strlog = f"parameter, {inst.name}_fw, {retVal}\n"
                param_file.write(strlog)

            else:
                # print(f"==== Instance {inst.name} no params need extract.")
                continue

            # print("=====finish ", inst.name, "=====")
    ae.dbCloseCV(cv)
