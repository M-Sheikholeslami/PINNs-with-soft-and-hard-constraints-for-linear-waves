import numpy as np
import tensorflow as tf
import math
from tensorflow.keras import models, layers, optimizers, activations, Sequential
from tensorflow.keras.models import load_model
from PINN import PINNs
from matplotlib import pyplot as plt
from time import time
from train_configs import config
from error import l2norm_err
import time
from matplotlib.animation import FuncAnimation
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

RunningTime = []

for i in range(3):
    num = str(i)
    start = time.time()
    # Domain discretization
    xn = 41
    yn = 41
    tn = 41

    g = 9.8
    k = 1
    L = 2*math.pi
    h = L
    omega = np.sqrt(g * k * np.tanh(k*h))
    TT = 2*math.pi/omega

    xx = np.linspace(0, L, xn)
    yy = np.linspace(-h, 0, yn)
    tt = np.linspace(0, TT, tn)
    x, y, t = np.meshgrid(xx, yy, tt)

    # Training Parameters
    act = config.act
    nn = config.n_neural
    nl = config.n_layer
    n_adam = config.n_adam
    cp_step = config.cp_step
    bc_step = config.bc_step

    class CustomLayer(tf.keras.layers.Layer):
        def __init__(self, num_X, num_T):
            super(CustomLayer, self).__init__()
            self.num_X = num_X
            self.num_T = num_T

        def build(self, input_shape):
            self.phasesX = self.add_weight(name='phasesX',
                                        shape=(self.num_X,),
                                        initializer='uniform',
                                        trainable=True)            

        def call(self, inputs):
            xPure = inputs[:, :1]   
            yPure = inputs[:, 1:2]  
            tPure = inputs[:, 2:3]  

            xPeiodicList = []
            for i in range(self.num_X):
                xPeriodic = tf.math.sin(xPure+ self.phasesX[i])
                xPeriodic = tf.math.tanh(xPeriodic)
                xPeiodicList.append(xPeriodic)            

            xperiodicTensor = tf.concat(xPeiodicList, axis=-1)
            return tf.concat([xperiodicTensor, yPure, tPure], axis=-1)

    # Training Data
    # Collection points
    cp = np.concatenate((x[:, ::cp_step].reshape((-1, 1)), 
                        y[:, ::cp_step].reshape((-1, 1)),
                        t[:, ::cp_step].reshape((-1, 1))), axis = 1)
    n_cp = len(cp)

    # Boundary points
    ind_bc = np.zeros(x.shape, dtype = bool)
    ind_bc[ [0, -1], ::bc_step, ::bc_step] = True
    ind_bc[ ::bc_step, [0, -1], ::bc_step] = True
    ind_bc[ ::bc_step, ::bc_step, [0, -1]] = True
    x_bc = x[ind_bc].flatten()
    y_bc = y[ind_bc].flatten()
    t_bc = t[ind_bc].flatten()
    bc = np.array([x_bc, y_bc, t_bc]).T

    ni = 3 
    nv = 1 
    n_bc = len(bc)
    domain = "periodicPhiPhix" 
    test_name = f'UnknownOmega_{nn}_{nl}_{act}_{n_adam}_{n_cp}_{n_bc}_{num}'

    #%%
    #################
    # Compiling Model
    #################

    inp = layers.Input(shape = (ni,))
    custom_layer = CustomLayer(10,10)
    custom_layer_output = custom_layer(inp)  
    hl = custom_layer_output
    # hl = inp
    for i in range(nl):
        hl = layers.Dense(nn, activation = act)(hl)
    out = layers.Dense(nv)(hl)

    model = models.Model(inp, out)
    print(model.summary())
    lr = 1e-3
    opt = optimizers.Adam(lr)
    pinn = PINNs(model, opt, n_adam)
    weights = model.get_weights()

    # Training Process
    print(f"INFO: Start training case : {test_name}")
    st_time = time.time()
    hist, omegaList = pinn.fit(cp)

    en_time = time.time()
    comp_time = en_time - st_time
    # Prediction
    xn_predict = 31
    yn_predict = 31
    tn_predict = 31

    xx_predict = np.linspace(0, L, xn_predict)
    yy_predict = np.linspace(-h, 0, yn_predict)
    tt_predict = np.linspace(0, TT, tn_predict)
    x_predict, y_predict, t_predict = np.meshgrid(xx_predict, yy_predict, tt_predict)

    cp_predict = np.concatenate((x_predict[:, ::cp_step].reshape((-1, 1)), 
                        y_predict[:, ::cp_step].reshape((-1, 1)),
                        t_predict[:, ::cp_step].reshape((-1, 1))), axis = 1)    

    pred, PhiX, PhiY = pinn.predict(cp_predict)
    Phi_p = pred[:, 0].reshape(x_predict.shape)
    PhiX = PhiX.reshape(x_predict.shape)
    PhiY = PhiY.reshape(x_predict.shape)
    pred = np.stack((Phi_p, PhiX, PhiY))
    np.savez_compressed('pred/' + test_name, pred = pred, x = x_predict, y = y_predict, t = t_predict, hist = hist, ct = comp_time, omegaList = omegaList)

    model.save('models/model_' + test_name + '.h5')
    print("INFO: Prediction and model have been saved!")
    end = time.time()
    print("Running time was: ", end - start)
    RunningTime.append(end - start)
