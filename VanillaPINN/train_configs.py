class config:
    """
    Train configs
    Args:
        act: activation func used for MLP 
        
        n_adam: Number of steps used for supervised training
        
        n_neural: Hidden dim fo each MLP layer (N_h)
        
        n_layer: total MLP layers used in model (N_l)


    """
    act = 'tanh'
    n_adam = 1000
    n_neural = 10
    n_layer = 16  
    cp_step = 1
    bc_step = 1
    method = "L-BFGS-B"
    