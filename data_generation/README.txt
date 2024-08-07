execute:
sh runs.sh

In "out_tensors" can be 2 folders


!!!!!!! NOTICE !!!!!!!!
IF the user want to impose from OSC file and there are randomizations in it, 
the seed must be exactly the same.
Gravity must be disabled or enabled depending on the torque file.

The problem with a simulation which leads to a a result of discarded different from 0, is that
if you create another simulation, it is difficult to "risalire" at the same randomization.

This is the format in which the tensor is saved:

    SS         --> starting SEED
    _envs_     --> number of envs 
    
    (example, if envs is 128 and num_runs is 10 --> potentially 1280 envs 
    [without considering % of collided envs])

    frequency -->  is the base frequency of the control action 
                    (which is maipulated internally by the function)

    _input_   -->  type of input: MS,CH | Multi-sinusoidal, chirp, combination

    _random_  -->  boolean variables in 0/1 | The case below, produce _random_1010

    random_initial_positions = True         
    random_stifness_dofs = False
    random_masses = True
    random_coms = False
