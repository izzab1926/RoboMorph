"""
This is my training dataset generator, in order to familiarize with the environment.

MANUEL BIANCHI BAZZI  -  POLITECNICO DI MILANO - created: 19/10/2023 ---> [...]

"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from trajectory_generator import control_action_function
from chirp import control_action_chirp

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path

torch.cuda.empty_cache()

args = gymutil.parse_arguments(description="Data Generation for multiple Franka systems",
        custom_parameters=[
        {"name": "--num-envs", "type": int, "default": 32, "help": "Number of environments to create"},
        #General
        {"name": "--disable-gravity-flag", "type": gymutil.parse_bool, "const": True, "default": False, "help": "If False, the gravity is not manually compensated"},  
        {"name": "--control-imposed", "type": gymutil.parse_bool, "const": True, "default": False, "help": "If False, it does the xy circle - task or waypoint"},
        {"name": "--control-imposed-file", "type": gymutil.parse_bool, "const": True, "default": False, "help": "If False, it does the xy circle - task or waypoint"},
        {"name": "--osc-task", "type": gymutil.parse_bool, "const": True, "default": True, "help": "if False, waypoints are generated"},
        {"name": "--type-of-task", "type": str, "default": 'VS', "help": "'VS','FS','FC'"},
        {"name": "--random-kp-kv", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Random kp and kv"},      
        {"name": "--dynamical-inclusion-flag", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Include/exclude masses from input tensor"},
        
        # Display Window, Save tensors, show figures
        {"name": "--headless-mode", "type": gymutil.parse_bool, "const": True, "default": True, "help": "Hide the simulation window (Faster)"},
        {"name": "--plot-diagrams", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Plot Diagrams of u an y tensors"},
        {"name": "--plot-only-torques", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Plot only Joint torques u"},
        {"name": "--save-tensors", "type": gymutil.parse_bool, "const": True, "default": True, "help": "Save tensors in file"},
        {"name": "--type-of-dataset", "type": str, "default": 'train', "help": "This is the subfolder in which we want to save ['train' or 'test' ]"},
        # Randomness of the simulation
        {"name": "--random-stiffness-dofs", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Stifness and damping of each Dofs"},
        {"name": "--random-initial-positions", "type": gymutil.parse_bool, "const": True, "default": True, "help": "Initial random position"},
        {"name": "--random-masses", "type": gymutil.parse_bool, "const": True, "default": True, "help": "Random mass of each link"},
        {"name": "--random-coms", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Random position of com of each link"},
        {"name": "--lower-bound-mass", "type": int, "default": 15, "help": "lower bound mass randomization"},
        {"name": "--higher-bound-mass", "type": int, "default": 15, "help": "higher bound mass randomization"},

        # Model settings
        {"name": "--tot-coordinates", "type": int, "default": 14, "help": "Stands for (x,y,z) + quaternions + 7 joints space"},
        {"name": "--joints", "type": int, "default": 9, "help": "Actual number of dofs in which we are interested (arms)"},
        {"name": "--body-links", "type": int, "default": 11, "help": "This represents the number of links (fingers included)"},  
        # Input settings 
        {"name": "--type-of-input", "type": str, "default": 'multi_sinusoidal', "help": "'combination','chirp','multi_sinusoidal'"},
        {"name": "--frequency", "type": float, "default": 0.1, "help": "Master frequency of the imposed control action"},
        # Simulation parameters
        {"name": "--max-iteration", "type": int, "default": 1000, "help": "Number of time-steps in simulation"},
        {"name": "--num-of-runs", "type": int, "default": 1, "help": "Number of runs (consecutive)"},
        {"name": "--seed", "type": int, "default": 0, "help": "imposing_seed"}])

#  General Simulation Parameters

disable_gravity_flag = args.disable_gravity_flag                  # if True, the gravity isn't manually compensated
control_imposed = args.control_imposed                            # if False, it does the xy circle task or waypoints task
control_imposed_file = args.control_imposed_file                        # if False ...
osc_task = args.osc_task                                            # if False it does the waypoints task (waypoints are generated every x iterations)
type_of_task = args.type_of_task

plot_diagrams = args.plot_diagrams                                # If True, could take much more time
plot_only_torques = args.plot_only_torques                        # if False it plots also the value of masses

random_initial_positions = args.random_initial_positions         
random_stiffness_dofs = args.random_stiffness_dofs
random_masses = args.random_masses
random_coms = args.random_coms
random_kp_kv = args.random_kp_kv                                  # OSC control parameters THIS IS NOT USED if control_imposed = True

higher_bound_mass = args.higher_bound_mass
lower_bound_mass = args.lower_bound_mass

type_of_input = args.type_of_input                                # 'combination','chirp','multi_sinusoidal'
tot_coordinates = args.tot_coordinates                            # Stands for (x,y,z) + quaternions + 7 joints space
joints = args.joints                                              # This represents the actual number of dofs
body_links = args.body_links

# Important for simulation 
headless_mode = args.headless_mode                                # if False, displays the simulation window
save_tensors = args.save_tensors 
dynamical_inclusion_flag = args.dynamical_inclusion_flag          # This include/exclude masses from input tensor
type_of_dataset = args.type_of_dataset                            # This is the subfolder in which we want to save ['train' or 'test' ]
frequency = args.frequency                                        # master frequency
num_envs = args.num_envs                                          # number of robots in each run
num_of_runs = args.num_of_runs                                    # number of runs 
max_iteration = args.max_iteration                                # number of time_steps in simulation

#Internal variable used to manage 
# conflicts with disable_gravity_flag
compensate = True

name_of_test_file = '12_envs_32_steps_1000_f_0_1_MS_rand_0010_bounds_mass_15_15.pt'

# ----------------------------------------- MANAGE BOOLEAN CHOICES ----------------------------------------

# disable_gravity_flag = False              # if False, the gravity isn't manually compensated
# headless_mode = False                    # if False, displays the simulation window

# control_imposed = True         
# control_imposed_file = False
# osc_task = False

# headless_mode= False 
# control_imposed = False         
# control_imposed_file = False
# osc_task = True
# type_of_task = 'FS'
# random_initial_positions = True
# random_masses = True
# save_tensors = True 
# num_envs = 32  
# save_tensors = False

# plot_diagrams=True

# random_stiffness_dofs = False
# frequency = .15
# type_of_input = 'chirp'

# control_imposed = True         
# control_imposed_file = True
# osc_task = False

# random_initial_positions = True
# plot_diagrams = True                    # If True, could take much more time
# plot_only_torques = False               # if False it plots also the dynamical inclusion

# random_masses = True
# lower_bound = 30
# higher_bound = 30

# random_coms = False
# random_kp_kv = False                    # OSC control parameters THIS IS NOT USED if control_imposed = True

# type_of_input = 'multi_sinusoidal'      # 'combination','chirp','multi_sinusoidal'

# tot_coordinates = 14                    # Stands for (x,y,z) + quaternions + 7 joints space
# joints = 9                              # This represents the actual number of dofs
# body_links = 11

# # Important for simulation 
# save_tensors = True 
# dynamical_inclusion_flag = False        # This include/exclude masses from input tensor
# type_of_dataset = 'train'               # This is the subfolder in which we want to save ['train' or 'test' ]
# frequency = .1                           # master frequency
# num_envs = 32                       # number of robots in each run
# num_of_runs = 1                        # number of runs 
# max_iteration = 1000                    # number of time_steps in simulation

# ---------------------------------------------------------------------------------------------------------

if control_imposed_file:
    max_iteration = max_iteration - 1 

# IF there's an imposition from file which had gravity, it must consider it the robot.
if control_imposed_file and ('no_gravity' in name_of_test_file) == False:
    disable_gravity_flag = False

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


for iter_run in range(num_of_runs):

    #check if the output directory is present, otherwise create them.
    if type_of_dataset=='test': complementary_folder = 'train'
    elif type_of_dataset=='train': complementary_folder = 'test'
    
    out_folder = Path("./out_tensors/")
    out_folder.mkdir(exist_ok=True)

    current_folder = Path("./out_tensors/"+type_of_dataset)
    current_folder.mkdir(exist_ok=True)
    _complementary_folder = Path("./out_tensors/"+complementary_folder)
    _complementary_folder.mkdir(exist_ok=True)

    # Check if in the other folder's is present the same seed 
    copies = True
    
    # print(args.seed)
    # if args.seed == 0:
    generated_seed = np.random.randint(0,9999)
    # else:
    #     generated_seed = args.seed
    
    print("\nGenerated seed:"+str(generated_seed))

    while copies:    
        #look in the other folder's is present the same seed
        list_of_tensors = os.listdir("./out_tensors/"+complementary_folder)
        generated_seed_str = 'seed_'+str(generated_seed)
        if any(generated_seed_str in s for s in list_of_tensors):
            generated_seed = np.random.randint(0,9999)
            print("\nCurrent Folder: "+type_of_dataset+" --> Found the same seed in "+complementary_folder)
            print("\nGenerated seed:"+str(generated_seed)) 
        else:
            copies=False

    torch.manual_seed(generated_seed) # Variable seed for each big iteration - reproducibility!

    #------------------------------------------------------------------------------------------------         
    print("\n ---------- This is run number :",iter_run, "----------")

    # ====================================== Simulation Settings  ==========================================

    gym = gymapi.acquire_gym()

    # Configure sim
    type_of_contact = gymapi.ContactCollection.CC_LAST_SUBSTEP 

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.dt = 1.0/60  # [s]
    sim_params.substeps = 2
    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.contact_collection = type_of_contact
    else:
        raise Exception("This example can only be used with PhysX")

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

    if sim is None:
        raise Exception("Failed to create sim")

    if not headless_mode:
        # Create viewer
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise Exception("Failed to create viewer")

    # Add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # Load franka asset
    asset_root = "./" 
    # Specify the folder of the Franka urdf file
    franka_asset_file = "franka_description/robots/franka_panda.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = True
    asset_options.armature = 0.01
    asset_options.disable_gravity = disable_gravity_flag

    print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
    franka_asset = gym.load_asset(
        sim, asset_root, franka_asset_file, asset_options)

    # get joint limits and ranges for Franka
    franka_dof_props = gym.get_asset_dof_properties(franka_asset)
    franka_effort_limits = franka_dof_props['effort'] # added
    franka_lower_limits = franka_dof_props['lower']
    franka_upper_limits = franka_dof_props['upper']
    franka_ranges = franka_upper_limits - franka_lower_limits
    franka_mids =  0.5 * (franka_upper_limits + franka_lower_limits) 
    franka_num_dofs = len(franka_dof_props)

    # set default DOF states
    default_dof_state = np.ones(franka_num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"][:7] = franka_mids[:7] 
    # franka_dof_props["stiffness"][:7].fill(0.0)
    # franka_dof_props["damping"][:7].fill(0.0)

    # Set up the env grid
    num_per_row = int(math.sqrt(num_envs))
    spacing = 1.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # default franka pose
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0)
    pose.r = gymapi.Quat(0, 0, 0, 1)

    print("Creating %d environments" % num_envs)

    envs = []
    hand_idxs = []
    init_pos_list = []
    init_orn_list = []

    if random_coms or random_masses:
        try:
            
            lower_bound = args.lower_bound_mass
            higher_bound = args.higher_bound_mass
            # comma_idx=random_bounds.index('')
            lower_bound = 1 - (args.lower_bound_mass)/100
            higher_bound = 1 + (args.higher_bound_mass)/100
            
            print('\n Mass randomization - Lower bound -->'+str(round(lower_bound,2)) 
                  +' | Higher bound -->' +str(round(higher_bound,2)))
  
        except ValueError:
            print('Incorrect or missing Bounds | Default values +-15%')
            lower_bound_mass = 0.85
            higher_bound_mass = 1.15

    # This is the initializing tensor for acquisition of the dynamical embedded parameters (masses of each joint)
    
    dynamical_inclusion = torch.zeros(0,body_links,dtype=torch.float32,device=args.graphics_device_id)

    for i in range(num_envs):
        # Create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)
        # Add franka
        franka_handle = gym.create_actor(env, franka_asset, pose, "franka", i, 0) # from 1 to 0

        #set DOF initial position 

        if  not random_initial_positions:
            default_dof_state["pos"][:7] = franka_mids[:7] 
        else:
            #magnitude controls "how" far is from middle position (0.5 is mids)
            magnitude = torch.rand(1).uniform_(0.2,0.8).numpy()
            default_dof_state["pos"][0] =  magnitude * np.sign(torch.rand(1).uniform_(-1,1).numpy()) * torch.rand(1).uniform_(franka_lower_limits[0],franka_upper_limits[0]).numpy() 
            default_dof_state["pos"][1:7] = franka_mids[1:7] + np.sign(torch.rand(1).uniform_(-1,1).numpy()) * 0.25 * torch.rand(6).numpy()
            
            # default_dof_state["pos"][0] = 0.2 * torch.rand(1).uniform_(franka_lower_limits[0],franka_upper_limits[0]).numpy() 
            # default_dof_state["pos"][1:7] = franka_mids[1:7] + 0.1 * torch.rand(6).numpy()
 
        gym.set_actor_dof_states(env, franka_handle, default_dof_state , gymapi.STATE_ALL)
        
        franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)

        if  not random_stiffness_dofs:
            #set DOF control properties (except grippers)
            franka_dof_props["stiffness"][:7].fill(0) 
            franka_dof_props["damping"][:7].fill(0.0) # default values inside URDF! 
        else:
            lower_bound_damping,higher_bound_damping,lower_random_stiff,higher_bound_stiff =  5, 10 ,0.01, 0.05 # 0.1, 0.5 ,0.01, 0.05 
            # Generating random arrays
            random_stiff = np.random.uniform(lower_random_stiff,higher_bound_stiff,7)
            random_damping = np.random.uniform(lower_bound_damping,higher_bound_damping,7)

            # set DOF control properties (except grippers)
            franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            franka_dof_props["stiffness"][:7]=random_stiff
            franka_dof_props["damping"][:7]=random_damping

            gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

        # set DOF control properties for grippers - indipendently from the choice to randomize joints!
        franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        # franka_dof_props["stiffness"][7:].fill(800.0)
        # franka_dof_props["damping"][7:].fill(40.0)

        # ADDED 12/01
        # gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

        # Set random mass and centers of masses for each link 
        if  not random_masses:

            # Acquire the rigid body properties of each env
            rigid_body_properties = gym.get_actor_rigid_body_properties(env, franka_handle)
            # Initialize tensor for local environment collection of the masses
            link_mass_tensor = torch.zeros(body_links,dtype=torch.float32,device=args.graphics_device_id)

            for l in range(len(rigid_body_properties)):
                #This is the only way to access to this class!
                link_mass_tensor[l]=rigid_body_properties[l].mass
        else:
            
            rigid_body_properties = gym.get_actor_rigid_body_properties(env, franka_handle)
            # initializing buffer for single env inclusion
            link_mass_tensor = torch.zeros(body_links,dtype=torch.float32,device=args.graphics_device_id)

            for l in range(0,len(rigid_body_properties)):
                
                rigid_body_properties[l].mass = rigid_body_properties[l].mass * torch.rand(1).uniform_(lower_bound,higher_bound).numpy()
                link_mass_tensor[l]=rigid_body_properties[l].mass

                # if random_coms:
                #     # rigid_body_properties[l].com.x = rigid_body_properties[l].com.x * torch.rand(1).uniform_(0.85,1.15).numpy() # np.random.uniform(0.85, 1.15)
                #     # rigid_body_properties[l].com.y = rigid_body_properties[l].com.y * torch.rand(1).uniform_(0.85,1.15).numpy() # np.random.uniform(0.85, 1.15)
                #     # rigid_body_properties[l].com.z = rigid_body_properties[l].com.z * torch.rand(1).uniform_(0.85,1.15).numpy() # np.random.uniform(0.85, 1.15)
                #     # # WIP - is there a way to compute inertia from xyz and coms? 
                #     # rigid_body_properties[l].inertia.x = rigid_body_properties[l].inertia.x * torch.rand(1).uniform_(0.85,1.15).numpy() # np.random.uniform(0.85, 1.15)
                #     # rigid_body_properties[l].inertia.y = rigid_body_properties[l].inertia.y * torch.rand(1).uniform_(0.85,1.15).numpy() # np.random.uniform(0.85, 1.15)
                #     # rigid_body_properties[l].inertia.z = rigid_body_properties[l].inertia.z * torch.rand(1).uniform_(0.85,1.15).numpy() #np.random.uniform(0.85, 1.15)
                    
                update = gym.set_actor_rigid_body_properties(env, franka_handle, rigid_body_properties,0)

                if not update:
                    print("Failed to overwrite link randomization.")

        #Stacking the dynamical values onto dimension 0             
        dynamical_inclusion = torch.cat( (dynamical_inclusion,link_mass_tensor.unsqueeze(0)) , dim = 0)
                        
        # Get inital hand pose
        hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
        hand_pose = gym.get_rigid_transform(env, hand_handle)
        init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
        init_orn_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

        # Get global index of hand in rigid body state tensor
        hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
        hand_idxs.append(hand_idx)

    dynamical_inclusion = dynamical_inclusion.unsqueeze(0)

    print("Example of masses:", link_mass_tensor)

    print("Stiffness", franka_dof_props["stiffness"][:7])
    print("Dampings", franka_dof_props["damping"][:7])
    
    print("\n--- Succesfully Created %d environments ----" % num_envs)    
    
    if not headless_mode:
        # Point camera at middle env
        cam_pos = gymapi.Vec3(4, 4, 4) # gymapi.Vec3(4, 3, 3)
        cam_target = gymapi.Vec3(-4, -3, -2)  #gymapi.Vec3(-4, -3, 0)
        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    # ===================================== prepare Exectution of simulation ==================================

    # from now on, we will use the tensor API to access and control the physics simulation

    gym.prepare_sim(sim)

    # initial hand position and orientation tensors
    init_pos = torch.Tensor(init_pos_list).view(num_envs, 3)
    init_orn = torch.Tensor(init_orn_list).view(num_envs, 4)

    if args.use_gpu_pipeline:
        init_pos = init_pos.to('cuda:0')
        init_orn = init_orn.to('cuda:0')

    # desired hand positions and orientations
    pos_des = init_pos.clone()
    orn_des = init_orn.clone()

    # Prepare jacobian tensor
    # For franka, tensor shape is (num_envs, 10, 6, 9)
    _jacobian = gym.acquire_jacobian_tensor(sim, "franka")
    jacobian = gymtorch.wrap_tensor(_jacobian)

    # Jacobian entries for end effector
    hand_index = gym.get_asset_rigid_body_dict(franka_asset)["panda_hand"]
    j_eef = jacobian[:, hand_index - 1, :]

    # Prepare mass matrix tensor
    # For franka, tensor shape is (num_envs, 9, 9)
    _massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
    mm = gymtorch.wrap_tensor(_massmatrix)

    # Randomizing OSC control parameters
    if not random_kp_kv:
        kp = 10
        kv = 2 * math.sqrt(kp)
    else:
        kp=torch.FloatTensor(num_envs, 1).uniform_(1, 5).to(device=args.graphics_device_id) 
        kv=torch.FloatTensor(num_envs, 1).uniform_(1,2 * math.sqrt(5)).to(device=args.graphics_device_id) 

    # Rigid body state tensor
    _rb_states = gym.acquire_rigid_body_state_tensor(sim)
    rb_states = gymtorch.wrap_tensor(_rb_states)

    # DOF state tensor
    _dof_states = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(_dof_states)
    dof_vel = dof_states[:, 1].view(num_envs, 9, 1)
    dof_pos = dof_states[:, 0].view(num_envs, 9, 1)

    # ------------------------------- initializing buffer tensor ------------------------------------

    buffer_position = torch.empty((0,num_envs,tot_coordinates), dtype=torch.float32).to(device=args.graphics_device_id) 
    buffer_target  = torch.empty((0,num_envs,3), dtype=torch.float32)

    # No dynamical inclusion 

    if dynamical_inclusion_flag:
        buffer_control_action = torch.empty((0,num_envs,joints+dynamical_inclusion.shape[2]), dtype=torch.float32).to(device=args.graphics_device_id)  
    
    if control_imposed_file or osc_task:
        print("Buffer intilized for xy_task or control_imposed_by_file \n ")
        buffer_control_action = torch.zeros((num_envs,joints,1), dtype=torch.float32).to(device=args.graphics_device_id) 

    else:
        print("Buffer intilized for control imposed directly in Nm\n ")
        buffer_control_action = torch.zeros((1,num_envs,joints), dtype=torch.float32).to(device=args.graphics_device_id) 
    
    # ----------------------------- Set control action as torque tensor -----------------------------

    if control_imposed and not control_imposed_file:
        if type_of_input == 'multi_sinusoidal':
            # Multi-sinusoidal 
            my_control_action = control_action_function(num_envs,max_iteration+1,joints,frequency) # +1 for bugs
        elif type_of_input == 'chirp':
            # chirp signal 
            my_control_action = control_action_chirp(num_envs,max_iteration+1,joints,frequency).to(device=args.graphics_device_id) 
        else:
            print("Error! No valid input type. Choosing 'multi_sinusoidal' by default.\n")
            my_control_action = control_action_function(num_envs,max_iteration+1,joints,frequency) # +1 for bugs

    # ----------------------- ADDED --------------------------- #
    if control_imposed_file:
        print("Control imposed by file: ",name_of_test_file,"\n")
        # name_of_test_file = 'circonferenza_nominale.pt'
        loaded = torch.load(name_of_test_file,map_location="cuda:0") #
        @ torch.no_grad()
        def loading():
            control_action_extracted = loaded['control_action']
            control_action_extracted = control_action_extracted.movedim(1,0)
            control_action_extracted = control_action_extracted.movedim(2,1)
            return control_action_extracted

        my_control_action = loading() 
        
    # -----------------------------------------------------------

    # These are the list of each self-colliding environment   
    black_list = []  
    # This is the list of each environment which quaternion has an abnormal change
    out_of_range_quaternion = [] 
    saturated_ll_idxs = []
    saturated_ul_idxs = []

    # ================================ SIMULATIONS STARTS =====================================

    # This flag helps reducing computational requirements by the simulation
    if not headless_mode:
        condition_window = gym.query_viewer_has_closed(viewer)
    else:
        condition_window = 0 

    _dof_states = gym.acquire_dof_state_tensor(sim)
    itr = 0 # control variable for inner loop

    while not condition_window  and itr <= max_iteration-1: # while not gym.query_viewer_has_closed(viewer):  #ORIGINAL

        itr += 1
        # Update jacobian and mass matrix and contact collection
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)
        gym.refresh_mass_matrix_tensors(sim)
        gym.refresh_net_contact_force_tensor(sim) 

        # Get current hand poses
        pos_cur = rb_states[hand_idxs, :3]
        orn_cur = rb_states[hand_idxs, 3:7]
        
        # Set desired hand positions # ORIGINAL
        if  osc_task == True:
            # radius = 0.05
            if itr ==1:
                radius = torch.rand((1,num_envs)).uniform_(0.01,0.12).to(device=args.graphics_device_id)
                period = torch.rand(1).uniform_(20,100).to(device=args.graphics_device_id)
                z_speed = torch.rand(1).uniform_(0.1,0.4).to(device=args.graphics_device_id)
                sign = torch.sign(torch.rand(1).uniform_(-1,1)).to(device=args.graphics_device_id)
                # offset =  torch.sign(torch.rand(1).uniform_(-0.3,0.3)).to(device=args.graphics_device_id)
            #This was used for testC!
            if  type_of_task == 'VS': # Vertical spyral
                pos_des[:, 0] = init_pos[:, 0] + math.sin(itr / period) * radius 
                pos_des[:, 1] = init_pos[:, 1] + math.cos(itr / period) * radius
                pos_des[:, 2] = init_pos[:, 2] - 0.1 + sign * z_speed * itr/max_iteration
            elif type_of_task == 'FS':
                radius = 0.1           # Fixed spyral
                pos_des[:, 0] = init_pos[:, 0] + math.sin(itr / 80) * radius 
                pos_des[:, 1] = init_pos[:, 1] + math.cos(itr / 80) * radius
                pos_des[:, 2] = init_pos[:, 2] + - 0.1 + 0.2 * itr/max_iteration
            elif type_of_task == 'FC': # Fixed circle
                # radius = 0.1
                pos_des[:, 0] = init_pos[:, 0] 
                pos_des[:, 1] = init_pos[:, 1] + math.sin(itr / 50) * radius #EDITED
                pos_des[:, 2] = init_pos[:, 2] + math.cos(itr / 50) * radius #EDITED

            # pos_des[:, 0] = init_pos[:, 0] - 0.05
            # pos_des[:, 1] = math.sin(itr / 50) * 0.15
            # pos_des[:, 2] = init_pos[:, 2] + math.cos(itr / 50) * 0.15

            # Solve for control (Operational Space Control)
            m_inv = torch.inverse(mm)
            m_eef = torch.inverse(j_eef @ m_inv @ torch.transpose(j_eef, 1, 2)) 
            orn_cur /= torch.norm(orn_cur, dim=-1).unsqueeze(-1)
            orn_err = orientation_error(orn_des, orn_cur)
            pos_err = kp * (pos_des - pos_cur)
            dpose = torch.cat([pos_err, orn_err], -1)
            u = torch.transpose(j_eef, 1, 2) @ m_eef @ (kp * dpose).unsqueeze(-1) - kv * mm @ dof_vel 

        # In these case, control is manually imposed (directly in Nm)
                
        if control_imposed and not control_imposed_file:
            # by function
            # u_custom = my_control_action[:,:,itr].unsqueeze(-1).to(device=args.graphics_device_id) 
            u_custom = my_control_action[:,:,itr].unsqueeze(-1)
            u = u_custom.contiguous().to(device=args.graphics_device_id) 

        if control_imposed_file:
            # by file
            u = my_control_action[:,:,itr].unsqueeze(-1).contiguous()
            
        # ok for OSC file with gravity
        if disable_gravity_flag == False and control_imposed_file and ('no_gravity' in name_of_test_file)==True:
            compensate = True
        # ok for OSC file without gravity
        if control_imposed_file==True and ('no_gravity' in name_of_test_file)==False:
            compensate = False

        # If the gravity is not compensated by the robot, manually compensate.
        if compensate:              #if not disable_gravity_flag:   
            if itr ==1: 
                print("I'm compensating")
            dof_count = gym.get_asset_dof_count(franka_asset)
            g = torch.zeros(num_envs, dof_count+1, 6, 1, dtype=torch.float, device=args.graphics_device_id)
            g[:, :, 2, :] = 9.81

            # dynamical_inclusion.squeeze(0)[:,1:] --> pick the bodies excluding the first one,
            # which is fixed to the ground! jacobian is [num_envs,10,6,9], where 10 is the number of body.
            #  Look at the documentation for further explanation about jacobian and how they're calculated. 
            # This compensation is taken from https://github.com/NVlabs/oscar/blob/main/oscar/agents/franka.py

            g_force = dynamical_inclusion.squeeze(0)[:,1:].unsqueeze(-1).unsqueeze(-1) * g
            j_link = jacobian[:, :dof_count+1, :, :dof_count]
            g_torque = (torch.transpose(j_link, 2, 3) @ g_force).squeeze(-1)
            g_torque = torch.sum(g_torque, dim=1, keepdim=False)
            g_torque = g_torque.unsqueeze(-1)
            u += g_torque       # u = u + g_torque --> more efficent

    # ------------------------------------- APPLICATION OF U -------------------------------------------------
            
        # Set control action as torque tensor, or position tensor
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(u))

    # ------------------------------------ CONTACT COLLECTION -------------------------------------------------

        # ATTENTION! gym.refresh_net_contact_force_tensor(sim) need to be added! remember
        _contact_forces = gym.acquire_net_contact_force_tensor(sim) 
        # returns: (envs x 11, 3) --> 16 envs --> (176,3), xyz components for each body
        # wrap it in a PyTorch Tensor
        contact_forces = gymtorch.wrap_tensor(_contact_forces)

    # -- COLLISION DETECTION  
        #body contact contains all the nonzero indeces about contact forces
        body_contact = torch.nonzero(abs(contact_forces)>0.01)

        # This for processes in all the istant of the simulation the indexes to be taken into account: 
        # Anyway, in the black list they enter only the first time.

        for j in range(body_contact.shape[0]):
            _body_contact = body_contact[j].to("cpu").numpy()
            env_idx_collision = int(np.ceil(_body_contact[0]/body_links)-1)

            if  not env_idx_collision in black_list : 
                
                black_list.append(env_idx_collision)
                # print("In env: ",env_idx_collision,", body n°",_body_contact[0] - body_links * env_idx_collision, "collided: step ",itr)
                # print(contact_forces[(body_links) * (env_idx_collision) ,:])
                # print("contact indexes:",body_contact[j])
                # print("body_contact: ",contact_forces[body_contact[j,0],:].to("cpu").numpy())

                # Visual purposes - coloring in red the colliding robots
                
                mesh = gymapi.MESH_VISUAL_AND_COLLISION # MESH_VISUAL works fine
                color = gymapi.Vec3(.9,.25,.15)
                env_handle = gym.get_env(sim,env_idx_collision)
                for k in range(body_links):
                    gym.set_rigid_body_color(env_handle,franka_handle, k , mesh ,color)

        # -------------------------------------- Step the physics --------------------------------------------------
        
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if not headless_mode:
            # Step rendering
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False) # True - collision | False - visual
            # gym.sync_frame_time(sim)

        # --------------------------------------- Stacking in the buffers -------------------------------------------------
            
        if control_imposed and control_imposed_file:
                #from file
                if itr==1:
                    print("\n Torque Imposed by FILE")
                control_action = u
        
        elif control_imposed and not control_imposed_file:
            if itr==1:
                print("\n Torque Imposed by function")
            control_action = u 
            control_action = control_action.view(1,num_envs,9)
            control_action = control_action[:,:,:joints]

        if osc_task:
            if itr==1:
                print(f"\n Torque Imposed by OSC - {type_of_task}")
            control_action = u  
            # if xy_task , I want [2, 9, 1001]

        # dynamical inclusion happens, both if the masses changes, both if the masses don't change
        if dynamical_inclusion_flag == True:
            control_action = torch.cat((control_action,dynamical_inclusion),dim=2)

        if control_imposed_file or osc_task:
            buffer_control_action = torch.cat((buffer_control_action, control_action), 2)  
        else: 
            buffer_control_action = torch.cat((buffer_control_action, control_action), 0) 


        dof_states = gymtorch.wrap_tensor(_dof_states)
        dof_pos = dof_states[:, 0]

        # Be careful using view | movedim is an alternative most of the times
        dof_pos = dof_pos.to("cpu").view(1,num_envs,9)
        dof_pos = dof_pos[:,:,:7]  # pick up only the 7 dofs! gripper fingers are excluded for output.
        pos_cur = pos_cur.to("cpu").view(1,num_envs,3)  # x y z
        orn_cur = orn_cur.to("cpu").view(1,num_envs,4)  # orientation angles

        # -------------------------- INCLUDING dof_pos for 7 - dimension state space -------------------------------

        full_pose = torch.cat((pos_cur,orn_cur,dof_pos),dim = 2).to(device=args.graphics_device_id) 
        # stacking onto the 0 dimension, each acquisition        
        buffer_position = torch.cat((buffer_position, full_pose), 0)  
        pos_desired=pos_des.to("cpu").view( 1,num_envs, 3) 
        if osc_task:
            buffer_target = torch.cat((buffer_target, pos_desired), 0)  #target position

        # --------------------------- Abnormal change in quaternion ---------------------------------------------------
        if itr > 2:

            #This is the increment of the undesired changes in quaternion
            increment = abs(buffer_position[itr-1,:,4:7]- buffer_position[itr-2,:,4:7])

            #This is the increment of the undesired changes in all dofs!
            # increment = abs(buffer_position[itr-1,:,:7]- buffer_position[itr-2,:,:7])

            #These are the out of range simulations
            out_of_range = torch.nonzero(increment > .1) #  np.rad2deg(.15) = 9°

            if out_of_range.shape[0] != 0:   
                abnormal_idxs = out_of_range.shape[0]
                for j in range(out_of_range.shape[0]):
                    out_of_range_quaternion.append(int(out_of_range[j,0].to('cpu').numpy()))

        # ---------------------------- Saturation check | Position --------------------------------------------------

        ll = torch.tensor(franka_lower_limits[:7]).repeat(num_envs,1) 
        ul = torch.tensor(franka_upper_limits[:7]).repeat(num_envs,1) 
        saturation_ll = torch.nonzero(abs(dof_pos-ll) < 0.05)
        saturation_ul = torch.nonzero(abs(dof_pos-ul) < 0.05)

        if saturation_ll.shape[0] != 0:
            abnormal_idxs = saturation_ll.shape[0]
            for j in range(saturation_ll.shape[0]):
                saturated_ll_idxs.append(int(saturation_ll[j,1].to('cpu').numpy()))

        if saturation_ul.shape[0] != 0:
            abnormal_idxs = saturation_ul.shape[0]
            for j in range(saturation_ul.shape[0]):
                saturated_ul_idxs.append(int(saturation_ul[j,1].to('cpu').numpy()))

        # ---------------------------- Saturation check | Torque --------------------------------------------------
        if osc_task:
            torques_limit = torch.tensor(franka_effort_limits[:7]).repeat(num_envs,1).to(device=args.graphics_device_id) 
            saturation_torques = torch.nonzero( (torques_limit - abs(control_action.squeeze(-1)[:,:7]) ) < 1)

            if saturation_torques.shape[0] != 0:
                abnormal_idxs = saturation_torques.shape[0]
                for j in range(saturation_torques.shape[0]):
                    saturated_ul_idxs.append(int(saturation_torques[j,0].to('cpu').numpy())) # append in another

        # ------------------------------------ CLEANING BUFFERS --------------------------------------------
        
        ## in this block, all the envs that have collided at least one step, are removed from the acquisitioncle

        if itr == max_iteration:
            
            saturation_idxs = list(set(saturated_ul_idxs + saturated_ll_idxs))
            print("\n----Number of saturated simulations: ",len(saturation_idxs),"/", num_envs,"----\n" ) 

            out_of_range_quaternion = list(set(out_of_range_quaternion))
            # this is the numbers of the colliding simulations
            print("---- Number of simulations with abnormal change in quaternions: ",len(out_of_range_quaternion),"/", num_envs,"----\n" )  
            print("---- Number of the colliding simulations: ",len(black_list),"/", num_envs,"----\n" ) 

            # Maybe a colliding env and abnormal change env coincide! Use always set
            black_list = black_list + out_of_range_quaternion + saturation_idxs 
            black_list = list(set(black_list))
            failed_percentage = len(black_list)/num_envs*100
            print("\n---- Number of bad simulations: ",len(black_list),"/", num_envs,"----\n" )  
            print("Percentage of total discarded simulations:", round(failed_percentage,2), "%") 

            #---- excluding all the invalid environment from the simulation ------ 
            non_valid_envs = len(black_list)
            num_valid_envs = num_envs - non_valid_envs

            black_list.sort(reverse=True)
            # for i in range(non_valid_envs):

            #     row_exclude = black_list[i]
            #     buffer_control_action = torch.cat((buffer_control_action [:,:row_exclude,:],
            #                                             buffer_control_action [:,row_exclude+1:,:]),1)
            #     buffer_position = torch.cat((buffer_position [:,:row_exclude,:],
            #                                             buffer_position [:,row_exclude+1:,:]),1)
                
            for i in range(non_valid_envs):

                row_exclude = black_list[i]
                if control_imposed and not control_imposed_file:
                    buffer_control_action = torch.cat((buffer_control_action [:,:row_exclude,:],
                                                            buffer_control_action [:,row_exclude+1:,:]),1)
                elif osc_task:
                    #in OSC task control action has a different order
                    buffer_control_action = torch.cat((buffer_control_action [:row_exclude,:,:],
                                        buffer_control_action [row_exclude+1:,:,:]),0)
                    buffer_target = torch.cat((buffer_position [:,:row_exclude,:],
                                        buffer_position [:,row_exclude+1:,:]),1)
                
                buffer_position = torch.cat((buffer_position [:,:row_exclude,:],
                    buffer_position [:,row_exclude+1:,:]),1)

    
    # --------------------------------- SAVING THE BUFFERS ----------------------------------------------------
    
    # This flag overcome the case in which all the simulation have collided
    # Avoiding strange errors [trying to save tensors with zero dimension]
    if  osc_task or (control_imposed_file):       
        buffer_control_action = buffer_control_action.movedim(1,2)
        buffer_control_action = buffer_control_action.movedim(0,1)

    if buffer_control_action.shape[1] == 0: 
        error_all_collided = True
    else:
        error_all_collided = False

    if save_tensors and not(error_all_collided) and not(control_imposed_file): 

        if disable_gravity_flag:
            addition_gravity = '_no_gravity'
        else:
            addition_gravity = ''

        print("\n\nSaving tensor to file ... ")

        tensors_from_isaacGym = {
            'control_action': buffer_control_action.to('cpu'), 
            'position': buffer_position.to('cpu'),
            'masses': dynamical_inclusion.to('cpu')
             }

        if type_of_input == 'multi_sinusoidal':
            type_of_input_save = 'MS'
        elif type_of_input == 'chirp':
            type_of_input_save = 'CH'
        else: 
            type_of_input_save = 'COMB'

        # Check if the file already exists in the chosen subfolder (train or dataset):
        list_of_tensors = os.listdir("./out_tensors/"+type_of_dataset)
        
        name_tensor = (str(generated_seed) +'_envs_' + str(num_valid_envs) + '_steps_' + str(max_iteration)+ 
                       '_f_'+ str(frequency).replace('.','_') +'_'+ type_of_input_save + '_rand_'+ str(int(random_initial_positions))+ str(int(random_stiffness_dofs)) + str(int(random_masses)) + str(int(random_coms))
                       + addition_gravity + '_bounds_mass_'+str(lower_bound_mass)+'_'+str(higher_bound_mass))
        
        if osc_task:
            
            tensors_from_isaacGym = {'control_action': buffer_control_action,
                                      'position': buffer_position.to('cpu'), 
                                      'target': buffer_target.to('cpu'),
                                      'masses': dynamical_inclusion.to('cpu')}
            
            name_tensor = (str(generated_seed) +'_envs_' + str(num_valid_envs) + '_steps_' + str(max_iteration)+ 
                            '_osc_' + type_of_task +'_'+ str(int(random_initial_positions))+ str(int(random_stiffness_dofs)) + 
                            str(int(random_masses)) + str(int(random_coms))+ addition_gravity)
                        
            if random_masses == True:
                name_tensor = name_tensor + '_bounds_mass_'+str(lower_bound_mass)+'_'+str(higher_bound_mass)

        print("\nThis is the ideal name of the current simulation:\n\n",name_tensor) 

        # check the index of simulation
        if any(name_tensor in s for s in list_of_tensors):
            print(" --- Warning, there was a copy! This has been overrided. --- ")

        # create the file if it is not created yet (opening in append mode)
        f = open("training_dataset_list.txt", "a")
        f.close()    

        with open("training_dataset_list.txt") as text_file:
            lines = [line.rstrip() for line in text_file]
        
        no_copies = True

        for line in lines:
            if line =='':
                break
            else:
                # check if there is an ononimy (already created)
                if (name_tensor + '.pt') == line:
                    no_copies = False
                    

        if no_copies:
            f = open("training_dataset_list.txt", "a")
            f.write(f'{name_tensor}.pt\n')
            f.close()        
            output = torch.save(tensors_from_isaacGym,'./out_tensors/'+type_of_dataset+'/'+ name_tensor+'.pt')
            print("\nSaved Done.\n ")
        else:
            print("\nAlready created! Not proceede to saving.\n")

    print("\nThis is control action dimension:", buffer_control_action.shape)
    print("This is position dimension:", buffer_position.shape)



    if not headless_mode:
        gym.destroy_viewer(viewer)

    # =================================== END OF SIMULATION ==================================================
    gym.destroy_sim(sim)

    # print(buffer_position.size()) # that's the size of the acquisition x-y-z
    # print(buffer_control_action.size()) # that's the size of the control action u

    # # results: matrix of steps x num_envs
    # =================================== PLOT TRAJECTORIES ==================================================
    
    # i = 0 ,1, 2   # x y z [m]
    # i = 3 - 6   # quaternion orientation

    if plot_diagrams and not(error_all_collided):
        
        print("\n Plotting ...")
        #If there's no dynamical inclusion in input, plot only_torques automatically.
        if dynamical_inclusion_flag == False: 
            plot_only_torques = True 

        fig, axs = plt.subplots(int(tot_coordinates/2),2,figsize=(20,20)) 
        fig.suptitle('Output: full pose and joint positions')
        label_coordinates = ['x','y','z','$X$','$Y$','$Z$','$W$',
                             '$q_0$','$q_1$','$q_2$','$q_3$','$q_4$','$q_5$','$q_6$'] 
        
        k = 0
        for j in range(2):
            for i in range(int(tot_coordinates/2)): 
                if k <=2:
                    axs[i,j].plot(buffer_position[:,:,k].to("cpu").numpy())
                    axs[i,j].set(ylabel='m', title=label_coordinates[k])
                    axs[i,j].grid()
                    if buffer_target != []:
                        axs[i,j].plot(buffer_target[:,:,k].to("cpu").numpy(),'r-',label='target') 
                        # axs[i,j].legend()
                elif k>2 and k<=6:
                    axs[i,j].plot(buffer_position[:,:,k].to("cpu").numpy())
                    axs[i,j].set(ylabel='[-]', title=label_coordinates[k])
                    axs[i,j].grid()
                else:
                    axs[i,j].plot(np.rad2deg(buffer_position[:,:,k].to("cpu").numpy()))
                    axs[i,j].set(ylabel='deg', title=label_coordinates[k])
                    axs[i,j].grid()
                k = k+1

        axs[i,0].set(xlabel='iteration steps')
        axs[i,1].set(xlabel='iteration steps')

        #-----------------------------------------------------------------------------------

        # if osc_task or control_imposed_file:

        #     buffer_control_action = buffer_control_action.movedim(0,1)  #added
        #     buffer_control_action = buffer_control_action.movedim(1,2)  #added
            
        #     fig3, axs3 = plt.subplots(joints, figsize=(20,20))
        #     fig3.suptitle('Control action among Dofs')
        #     for i in range(joints):
        #         temporary=buffer_control_action[i,1:,:].to("cpu").numpy()   
        #         if i <=joints-1:
        #             axs3[i].set(ylabel='torque', title='joint'+str(i)+'')
        #             axs3[i].plot(temporary)
        #         axs3[i].grid()
        #     axs3[i].set(xlabel='iteration steps')    
        # else:
            # if plot_only_torques:
        fig3, axs3 = plt.subplots(joints, figsize=(20,20))
        fig3.suptitle('Control action among Dofs',y=1.2)

        fig3.suptitle('Control action among Dofs')
        for i in range(joints):
            temporary=buffer_control_action[1:,:,i].to("cpu").numpy()   
            if i <=joints-1:
                axs3[i].set(ylabel='torque', title='joint'+str(i)+'')
                axs3[i].plot(temporary)
            axs3[i].grid()
        axs3[i].set(xlabel='iteration steps')

        if plot_only_torques == False:

            fig2, axs2 = plt.subplots(dynamical_inclusion.shape[2], figsize=(20,20))  
            fig2.suptitle('Masses of each body') 

            for i in range(dynamical_inclusion.shape[2]):  
                temporary=buffer_control_action[:,:,joints+i].to("cpu").numpy()    
                axs2[i].grid()
                axs2[i].set(xlabel='iteration steps')
                axs2[i].set(ylabel='mass', title='body'+str(i)+'')
                axs2[i].plot(temporary)

        # label_coordinates = ['x','y','z']
        # fig4, axs4 = plt.subplots(3)
        # fig4.suptitle('Position Target')
        # for i in range(3):
        #     axs4[i].plot(buffer_target[:,:,i].to("cpu").numpy())
        #     axs4[i].set(xlabel='iteration steps', ylabel='m', title=label_coordinates[i])
        #     axs4[i].grid()

        fig3.tight_layout(pad=3)
        fig.tight_layout(pad=3)
        fig.subplots_adjust(top = .96)
        fig3.subplots_adjust(left=0.07,
                    bottom=0.06, 
                    right=0.97, 
                    top=0.92, 
                    wspace=0.2, 
                    hspace=0.54)
        
        fig.tight_layout()
        plt.show()
        
        
    # clearing associated memory --> Apparently it seems to not effecting 
    # the required Used Dedicated Memory

    # # if control action is imposed, clear variable u_custom

    # if control_imposed:  
    # del u_custom
    # del u 
    # del buffer_position
    # del buffer_control_action
    # del full_pose 
    # del contact_forces
    # del _contact_forces
    # del out_of_range

    torch.cuda.empty_cache()

print("Simulation finished.")

# ============================== SAVING TENSORS TO FILE ===============================

    # This is the format in which the tensor is saved:

    #  NNN         -->  SEED
    # _envs_     --> number of envs 
    # _num_runs_ --> number of runs 
    # 
    # (example, if envs is 128 and num_runs is 10 --> potentially 1280 envs 
    # [without considering % of collided envs])

    #steps      -->  number of time steps
    # frequency -->  is the base frequency of the control action 
    #                 (which is maipulated internally by the function)

    # _input_   -->  type of input: MS,CH,COMB | Multi-sinusoidal, chirp, combination

    # _rand_  -->  boolean variables in 0/1 | The case below, produce _random_1010

    # # random_initial_positions = True         
    # # random_stiffness_dofs = False
    # # random_masses = True
    # # random_coms = False

    # _bounds_mass_ --> lower_bound and higher_bound 



