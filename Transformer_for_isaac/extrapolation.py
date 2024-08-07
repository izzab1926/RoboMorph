import re

def extrapolation(list_file, *args):
    # This function extrapolate for each of the variables involved in the randomization
    # their value: 


    # Initialization
    seed = []
    frequency = []
    bounds = []
    task=[]
    rand_initial_position = []  
    joint_randomization = []
    mass_randomization = []
    random_coms = []

    # If the name provided is a file composed by many simulation
    try:
        with open(list_file) as text_file:
            lines = [line.rstrip() for line in text_file]

    # Otherwise it is a single simulation such as: '10_envs_32_steps_1000_osc_FC_1010_bounds_mass_10_10.pt'
    except FileNotFoundError:
        lines = list_file 

    flag_test = False
    for line in lines:
        # End of file 
        if line =='':
            break
        # name_tensor was given as input 
        if len(line) == 1:
            line = lines
            flag_test = True
            # print("Test") # helpful to understand if it is a file.txt or directly the name of tensor
            
        if "KUKA" in line:
            # Temporary removing kuka from line 
            idx_kuka = str(line).find('KUKA')
            line = line[:idx_kuka-1] + line[idx_kuka+4:]
            
        # SEED EXTRACTION
        seed_idx = str(line).find('_envs')
        seed.append(int(str(line)[0:seed_idx]))
        
        # Type of input extraction
            
        if "_f_" in line:
            frequency_idx = str(line).find('_f_')
            try:
                # f_0_15
                frequency.append(float(str(line)[frequency_idx+3:frequency_idx+7].replace('_','.')))
            except ValueError:
                # f_0_1
                frequency.append(float(str(line)[frequency_idx+3:frequency_idx+6].replace('_','.')))
            frequency = list(set(frequency))
            if "_CH_" in line:
                task.append("CH")
            elif "_MS_" in line :
                task.append("MS")
            
        elif "osc_" in line: 


            if 'VS' in line:
                task.append("osc[VS]")
            elif 'FS' in line:
                task.append("osc[FS]")
            elif 'FC' in line:
                task.append("osc[FC]")

        if "ss_" in line:
            bounds_idx = str(line).find('ss_')
            end_idx = str(line).find('.pt')
            bounds.append((str(line)[bounds_idx+3:end_idx].replace('_',' ')))
            bounds = list(set(bounds))  

        if "_1111_" in line:
            rand_initial_position.append(True)
            joint_randomization.append(True)
            mass_randomization.append(True)
            random_coms.append(True)
        if "_1110_" in line:
            rand_initial_position.append(True)
            joint_randomization.append(True)
            mass_randomization.append(True)
            random_coms.append(False)
        if "_1100_" in line:
            rand_initial_position.append(True)
            joint_randomization.append(True)
            mass_randomization.append(False)
            random_coms.append(False)
        if "_1010_" in line:
            rand_initial_position.append(True)
            joint_randomization.append(False)
            mass_randomization.append(True)
            random_coms.append(False)

        if flag_test:
            break
    
    max_bounds = []
    for i in range(len(bounds)):
            space_index = bounds[i].find(" ") 
            # print(space_index)
            first_test_bound = int(bounds[i][0:space_index])
            second_test_bound = int(bounds[i][space_index+1:])
            # print(first_test_bound,second_test_bound)
            if max_bounds == []:
                max_bounds = [first_test_bound,second_test_bound]
            elif max_bounds != []:
                if first_test_bound >= max_bounds[0] and first_test_bound >= max_bounds[1]:
                    max_bounds = [first_test_bound,second_test_bound]

    # print(max_bounds)

    task = list(set(task))
    bounds = list(set(bounds))

    rand_initial_position = list(set(rand_initial_position))
    joint_randomization = list(set(joint_randomization))
    mass_randomization = list(set(mass_randomization))
    random_coms = list(set(random_coms))

    my_dictionary = {"seed":seed,
                    "frequencies": frequency,
                    "max_bounds" :max_bounds,
                    "bounds": bounds,
                    "task" :task,
                    "rand_init_pos":rand_initial_position,
                    "rand_mass":mass_randomization,
                    "rand_joints":joint_randomization,
                    "rand_coms": random_coms}

    return my_dictionary

def check_ID_OOD(name_tensor,file_training_dataset,*args):

    flag_validation = False
    # to be commented out
    # print("Searching in file...")

    with open(file_training_dataset) as text_file:
        lines = [line.rstrip() for line in text_file]
    for line in lines:
        # End of file 
        if line =='':
            break
        else:
            if line == name_tensor:
                print("Attention, this exact file is already present! It is a validation file.\n")
                flag_validation  = True
                output = "Error_validation"

    if not flag_validation:
        # print("Well, let's see if it is ID or OOD...")
        training = extrapolation(file_training_dataset) 
        test = extrapolation(name_tensor)

        flag1 = test["rand_joints"][0] in training["rand_joints"]  
        flag2 = test["rand_mass"][0] in training["rand_mass"] 
        flag3 = test["rand_coms"][0] in training["rand_coms"]
        flag4 = test["rand_init_pos"][0] in training["rand_init_pos"] 
        flag5 = test["task"][0] in training["task"]

        if test["task"][0] == 'MS' or test["task"][0] == 'CH':
            # print("Torque is imposed in test...")
            # flag6 = test["frequencies"][0] in training["frequencies"]
            flag6 = test["frequencies"][0] <= max(training["frequencies"])
        else:
            # print("Ok, OSC task...")    
            # if it is an OSC Task, frequency is not considered.
            if flag5:
                flag6 = True
            else:
                flag6 = True

        if test["max_bounds"] != []:
            
            flag7 = test["max_bounds"][0] <= training["max_bounds"][0] and test["max_bounds"][1] <= training["max_bounds"][1]
        else:
            flag7 = True

        # print(flag1)
        # print(flag2)
        # print(flag3)
        # print(flag4)
        # print(flag5)
        # print(flag6)
        # print(flag7)

        if flag1 and flag2 and flag3 and flag4 and flag5 and flag6 and flag7:
            # print("In-Distribution")    
            output = "In-Distribution"
        else:
            # print("Out Of Distribution")
            output = "Out-Of-Distribution"

        # output = {"rand_init_pos": flag4,
        #         "rand_mass":flag2,
        #         "rand_joints":flag1,
        #         "rand_coms": flag3,
        #         "frequencies": flag6,
        #         "max_bounds" :flag7,
        #         "task" :flag5}
        
        boolean_output = [ flag1,flag2,flag3,flag4,flag5,flag6,flag7 ]
        
    else:
        return None,None,None,None
        
    return training,test,output,boolean_output
            

if __name__ == '__main__':

    name_tensor = '337_envs_28_steps_1110_osc_VS_1010_bounds_mass_10_10.pt'

    file_training_dataset= "training_ds_VS_scratch_list.txt"
    training,test,output,boolean_output = check_ID_OOD(name_tensor,file_training_dataset)
    
    print(test["rand_joints"])
    print(training["rand_joints"])
    print(output)

    lista_example = ["training_ds1_list.txt",  
            "training_ds2_list.txt", 
            "training_ds3_list.txt"] 

    for i in range(len(lista_example)):
        
        training,test,output,boolean_output = check_ID_OOD(name_tensor,lista_example[i])
        print("\ textbf{"+lista_example[i]+"}")
        print("\ begin{itemize}[noitemsep]")
        print("\item Frequencies → ", training["frequencies"])
        print("\item Bounds → ", training["bounds"])
        print("\item Tasks → ",training["task"])
        print("\item Random initial Positions → ",training["rand_init_pos"])
        print("\item Random Mass → ",training["rand_mass"])
        print("\item Random Joint stiffness → ", training["rand_joints"])
        print("\item Random center of masses → ",training["rand_coms"]) 
        print("\end{itemize}")
        print("\n")
