import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from lbfgs import optimizer as lbfgs_op
import math
import matplotlib.pyplot as plt 
from train_configs import config

class PINNs(models.Model):
    def __init__(self, model, optimizer, epochs, **kwargs):
        super(PINNs, self).__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.hist = []
        self.epoch = 0
        self.sopt = lbfgs_op(self.trainable_variables)
              
    @tf.function
    def net_f(self, cp):

        self.xn = 41
        self.yn = 41
        self.tn = 41

        x = cp[:, 0]
        y = cp[:, 1]
        t = cp[:, 2]
        x = tf.expand_dims(x, axis=-1)
        y = tf.expand_dims(y, axis=-1)
        t = tf.expand_dims(t, axis=-1)
        self.h = 2*math.pi

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            
            X = tf.stack([x, y, t], axis = -1)
            X = tf.squeeze(X, axis=1)
            pred = self.model(X)
            Phi = ((y+self.h)**2) * pred
            Phi_x = tape.gradient(Phi, x)
            Phi_y = tape.gradient(Phi, y)
        Phi_xx = tape.gradient(Phi_x, x)
        Phi_yy = tape.gradient(Phi_y, y)
        A = 0.01
        g = 9.8
        k = 1
        L = 2*math.pi
        h = L
        omega = np.sqrt(g * self.k * np.tanh(self.k*self.h))

        # === Bottom Boundary: z = -h, enforce ∂Φ/∂z = 0 ===
        mask_bottom = tf.squeeze(tf.abs(y + h) < 1e-5)
        loss_kbbc = tf.reduce_mean(tf.square(tf.boolean_mask(Phi_y, mask_bottom)))

        # === Free Surface: z = 0, enforce ∂Φ/∂z = A ω sin(kx - ωt) ===
        mask_surface = tf.squeeze(tf.abs(y) < 1e-5)
        x_surf = tf.boolean_mask(x, mask_surface)
        t_surf = tf.boolean_mask(t, mask_surface)
        Phi_z_surf = tf.boolean_mask(Phi_y, mask_surface)
        target_KFBC = A * omega * tf.sin(k * x_surf - omega * t_surf)
        loss_kfbc = tf.reduce_mean(tf.square(Phi_z_surf - target_KFBC))
        f1 = tf.abs(Phi_xx + Phi_yy)
        f5 = Phi - Phi
        f1 = tf.reduce_mean(tf.square(f1))
        f5 = tf.reduce_mean(tf.square(f5))

        return f1, loss_kbbc, loss_kfbc , f5 
    
    @tf.function
    def train_step(self, cp):
        print("train step is called")
        with tf.GradientTape() as tape:
            f1, f2, f4, f5 = self.net_f(cp)
            loss_GE = f1
            loss_KBBC = f2
            loss_KFBC = f4
            loss_dummy = f5
            loss = loss_GE + loss_KBBC + loss_KFBC + loss_dummy
         
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        l1 = tf.reduce_mean(loss)
        l2 = tf.reduce_mean(loss_GE)
        l3 = tf.reduce_mean(loss_KBBC)
        l4 = tf.reduce_mean(loss_KFBC)
        l6 = tf.reduce_mean(loss_dummy)

        tf.print('loss:', l1)
        return loss, grads, tf.stack([l1, l2, l3, l4, l6]) #, l7    
    
    def fit(self, cp):
        cpNumpy = cp
        cp = tf.convert_to_tensor(cp, tf.float32)
        self.k = 1
        self.L = 2*math.pi

        def func(params_1d):
            self.sopt.assign_params(params_1d)
            tf.print('epoch:', self.epoch)
            loss, grads, hist = self.train_step(cp)
            grads = tf.dynamic_stitch(self.sopt.idx, grads)
            self.epoch += 1
            self.hist.append(hist.numpy())
            return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)

        for epoch in range(self.epochs):
            tf.print('epoch:', self.epoch)
            loss, grads, hist = self.train_step(cp)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.hist.append(hist.numpy())
            self.epoch += 1
        self.sopt.minimize(func)
        return np.array(self.hist)

    def predict(self, cp):
        cp = tf.convert_to_tensor(cp, tf.float32)
        x = cp[:, 0]
        y = cp[:, 1]
        t = cp[:, 2]

        x = tf.expand_dims(x, axis=-1)
        y = tf.expand_dims(y, axis=-1)
        t = tf.expand_dims(t, axis=-1)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            X = tf.stack([x, y, t], axis = -1)
            X = tf.squeeze(X, axis=1)
            pred = self.model(X)
            Phi = ((y+self.h)**2) * pred
            Phi_x = tape.gradient(Phi, x)
            Phi_y = tape.gradient(Phi, y)
        return Phi.numpy(), Phi_x.numpy(), Phi_y.numpy()     
