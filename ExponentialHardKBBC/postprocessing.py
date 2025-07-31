#%%
import numpy as np 
import math
import matplotlib.pyplot as plt 
import seaborn as sns
from error import l2norm_err
from train_configs import config
import statistics
from matplotlib.ticker import FuncFormatter
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, activations, Sequential
import os

# Enable LaTeX rendering for text
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

act = config.act
nn = config.n_neural
nl = config.n_layer
n_adam = config.n_adam
data_name = f"ExpHardKBBC_{nn}_{nl}_{act}_{n_adam}_9261_2402"

xn = 31
yn = 31
tn = 31
L = 2*math.pi
h = L
k = 1
g = 9.8
omega = np.sqrt(g * k * np.tanh(k*h))
# print("omega =", omega)
TT = 2*math.pi/omega
A = 0.01
Phi = np.zeros((yn,xn,tn))
PhiX = np.zeros((yn,xn,tn))
PhiY = np.zeros((yn,xn,tn))
xBoundaryPhiDiff = np.zeros((tn,yn))
xBoundaryUDiff = np.zeros((tn,yn))
xBoundaryVDiff = np.zeros((tn,yn))

tBoundaryPhiDiff = np.zeros((xn,yn))
tBoundaryUDiff = np.zeros((xn,yn))

xDomain = np.linspace(0, L, xn)
yDomain = np.linspace(-h, 0, yn)
tDomain = np.linspace(0, TT, tn)

###### Analytical solution #######
xxEXACT, yyEXACT, ttEXACT = np.meshgrid(xDomain, yDomain, tDomain)
# omega = 3.10
for m in range(tn):
    for j in range(yn):
        for i in range(xn):
            Phi[j,i,m] = A*g/omega * (np.cosh(k*(h+yyEXACT[j,i,m]))/np.cosh(k*h)) * np.sin(k*xxEXACT[j,i,m]-omega*ttEXACT[j,i,m])

for m in range(tn):
    for j in range(yn):
        for i in range(xn):
            PhiX[j,i,m] = A*g/omega * (np.cosh(k*(h+yyEXACT[j,i,m]))/np.cosh(k*h)) * k * np.cos(k*xxEXACT[j,i,m]-omega*ttEXACT[j,i,m])

for m in range(tn):
    for j in range(yn):
        for i in range(xn):
            PhiY[j,i,m] = A * g/omega * k * (np.sinh(k*(h+yyEXACT[j,i,m]))/np.cosh(k*h)) * np.sin(k*xxEXACT[j,i,m]-omega*ttEXACT[j,i,m])

PhiMeanPointDifference = []
PhiXMeanPointDifference = []
PhiYMeanPointDifference = []
ctList = []
uPeriodicityErrorList = []
vPeriodicityErrorList = []

trialNum = 3

for numerator in range(trialNum):
    num = str(numerator)
    cmp = sns.color_palette('RdBu_r',n_colors = 100, as_cmap=True)
    plt.set_cmap(cmp)

    ###### Loading the PINN solution #######
    domain = "periodicPhiPhix" 
    data = np.load(f"pred/{data_name}_{num}.npz")

    xxPINN, yyPINN = np.meshgrid(xDomain, yDomain)
    tttPINN, yyyPINN = np.meshgrid(tDomain, yDomain)


    pred = data["pred"]
    ###### Shifting the analytical solution #######
    RefPred = pred[0,0,0,0]
    RefExact = Phi[0,0,0] 
    for k in range(tn):
        for j in range(yn):
            for i in range(xn):
                pred[0,j,i,k] += RefExact - RefPred

    ###### Point-to-Point difference #######
    PhiMeanPointDifferenceInTime = []
    PhipointDifferenceWhole = []
    PhiXMeanPointDifferenceInTime = []
    PhiXpointDifferenceWhole = []    
    PhiYMeanPointDifferenceInTime = []
    PhiYpointDifferenceWhole = []

    for k in range(tn):
        PhipointDifference = []
        PhiXpointDifference = []
        PhiYpointDifference = []
        for j in range(yn):
            for i in range(xn):
                PhipointDifference.append(np.abs((Phi[j,i,k]-pred[0,j,i,k])/(np.max(Phi[:,:,:]))*100))
                PhipointDifferenceWhole.append(np.abs((Phi[j,i,k]-pred[0,j,i,k])/(np.max(Phi[:,:,:]))*100))
                PhiXpointDifference.append(np.abs((PhiX[j,i,k]-pred[1,j,i,k])/(np.max(PhiX[:,:,:]))*100))
                PhiXpointDifferenceWhole.append(np.abs((PhiX[j,i,k]-pred[1,j,i,k])/(np.max(PhiX[:,:,:]))*100))
                PhiYpointDifference.append(np.abs((PhiY[j,i,k]-pred[2,j,i,k])/(np.max(PhiY[:,:,:]))*100))
                PhiYpointDifferenceWhole.append(np.abs((PhiY[j,i,k]-pred[2,j,i,k])/(np.max(PhiY[:,:,:]))*100))

        PhiMeanPointDifferenceInTime.append(statistics.mean(PhipointDifference))
        PhiXMeanPointDifferenceInTime.append(statistics.mean(PhiXpointDifference))

    PhiMeanPointDifference.append(statistics.mean(PhipointDifferenceWhole))
    PhiXMeanPointDifference.append(statistics.mean(PhiXpointDifferenceWhole))
    PhiYMeanPointDifference.append(statistics.mean(PhiYpointDifferenceWhole))
    ###### Phi difference at x ends #######
    for k in range(tn):
        for j in range(yn):
            xBoundaryPhiDiff[k,j] = np.abs(pred[0,j,0,k]-pred[0,j,-1,k])
            xBoundaryUDiff[k,j] = np.abs(pred[1,j,0,k]-pred[1,j,-1,k])    
            xBoundaryVDiff[k,j] = np.abs(pred[2,j,0,k]-pred[2,j,-1,k])    

    ###### Plots #######
    xComparison = 1
    tComparison = 0
    tContour = 0

    ###### Error Contour #######
    PhiError = np.abs((Phi[:,:,:]-pred[0,:,:,:])/(np.max(Phi[:,:,:]))*100)
    PhiXError = np.abs((PhiX[:,:,:]-pred[1,:,:,:])/(np.max(PhiX[:,:,:]))*100)
    PhiYError = np.abs((PhiY[:,:,:]-pred[2,:,:,:])/(np.max(PhiY[:,:,:]))*100)

    rows = 6 
    cols = 6
    fig, axes = plt.subplots(rows, cols, figsize=(25, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < 21:        
            contour_PINN = ax.contourf(xxPINN/L, yyPINN/h, pred[1,:,:,i], levels=100, alpha=0.5)
            cbar_PINN = plt.colorbar(contour_PINN, ax=ax)
            cbar_PINN.set_ticks([np.min(pred[1,:,:,i]), np.max(pred[1,:,:,i])])
            cbar_PINN.set_label('u PINN [m/s]', fontsize=14)
            contour_PINN.set_cmap('viridis')
            ax.set_xlabel('x / L', fontsize=14)
            ax.set_ylabel('z / h', fontsize=14)
            ax.text(0.95, 0.95, f"t={i}", ha='right', va='top', transform=ax.transAxes, fontsize=12)

    rows = 6 
    cols = 6
    fig, axes = plt.subplots(rows, cols, figsize=(25, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < 21:        
            contour_PINN = ax.contourf(xxPINN/L, yyPINN/h, pred[2,:,:,i], levels=100, alpha=0.5)
            cbar_PINN = plt.colorbar(contour_PINN, ax=ax)
            cbar_PINN.set_ticks([np.min(pred[2,:,:,i]), np.max(pred[2,:,:,i])])
            cbar_PINN.set_label('w PINN [m/s]', fontsize=14)
            contour_PINN.set_cmap('viridis')
            ax.set_xlabel('x / L', fontsize=14)
            ax.set_ylabel('z / h', fontsize=14)
            ax.text(0.95, 0.95, f"t={i}", ha='right', va='top', transform=ax.transAxes, fontsize=12)

    rows = 6 
    cols = 6
    fig, axes = plt.subplots(rows, cols, figsize=(25, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < 21:        
            contour_PINN = ax.contourf(xxPINN/L, yyPINN/h, pred[0,:,:,i], levels=100, alpha=0.5)
            cbar_PINN = plt.colorbar(contour_PINN, ax=ax)
            cbar_PINN.set_ticks([np.min(pred[0,:,:,i]), np.max(pred[0,:,:,i])])
            cbar_PINN.set_label('Phi PINN [m2/s]', fontsize=14)
            contour_PINN.set_cmap('viridis')
            ax.set_xlabel('x / L', fontsize=14)  
            ax.set_ylabel('z / h', fontsize=14) 
            ax.text(0.95, 0.95, f"t={i}", ha='right', va='top', transform=ax.transAxes, fontsize=12)

    save_dir = 'pred'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(9)
    for k in range(tn):
        label = f"Time _{k}"
        plt.plot(yDomain,xBoundaryUDiff[k,:], label=label)
        plt.xlabel('z [m]', fontsize= 14)
    plt.ylabel('u Difference at x ends [m/s]', fontsize= 14)
    plt.legend(loc='upper center', ncol=4, prop={'size': 8})
    plt.savefig(os.path.join(save_dir, f'UPeriodError_case{num}.png'), bbox_inches='tight')
    plt.close()

    plt.figure(10)
    for k in range(tn):
        label = f"Time _{k}"
        plt.plot(yDomain,xBoundaryVDiff[k,:], label=label)
        plt.xlabel('z [m]', fontsize= 14)
    plt.ylabel('w Difference at x ends [m/s]', fontsize= 14)
    plt.legend(loc='upper center', ncol=4, prop={'size': 8})
    plt.savefig(os.path.join(save_dir, f'VPeriodError_case{num}.png'), bbox_inches='tight')
    plt.close()

    plt.figure(12)
    plt.semilogy(data["hist"][:,0],label=r'$L_\mathrm{Total}$')
    plt.semilogy(data["hist"][:,1],label=r'$L_\mathrm{GE}$')
    plt.semilogy(data["hist"][:,2],label=r'$L_\mathrm{KBBC}$')
    plt.semilogy(data["hist"][:,3],label=r'$L_\mathrm{KFSBC}$')
    plt.legend(fontsize=14)
    plt.xlabel(r'$Epoch$', fontsize=20)
    plt.ylabel(r'$Loss$', fontsize=20)
    plt.xticks(fontsize=18) 
    plt.yticks(fontsize=18) 
    plt.savefig(f'Loss_case{num}.png' , dpi=300)
    plt.close()

    plt.figure(13)
    plt.plot(xDomain, pred[1,0,:,1], label='U')
    plt.legend()

    # plt.show()


    times = [0, 10, 20]
    time_labels = ['t = 0', 't = T/3', 't = 2T/3']

    for i, time_label in zip(times, time_labels):
    # Plot u PINN
        if i == 0:
            time_label = 't = 0'
        elif i == 10:
            time_label = 't = T/3'
        elif i == 20:
            time_label = 't = 2T/3'
        else:
            time_label = f"t={i}" 
        plt.figure()
        plt.contourf(xxPINN/L, yyPINN/h, pred[1, :, :, i], levels=20, alpha=1, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label(r'$u \, \mathrm{PINN} \, [\mathrm{m/s}]$', fontsize=22)
        cbar.ax.tick_params(labelsize=20)  
        plt.xlabel(r'$x / L$', fontsize=22)
        plt.ylabel(r'$z / h$', fontsize=22)
        plt.xticks(fontsize=20)  
        plt.yticks(fontsize=20)  
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'u_PINN_{i}_case{num}.eps'), format='eps', bbox_inches='tight')
        plt.close()

        # Plot v PINN
        plt.figure()
        plt.contourf(xxPINN/L, yyPINN/h, pred[2, :, :, i], levels=20, alpha=1, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label(r'$w \, \mathrm{PINN} \, [\mathrm{m/s}]$', fontsize=22)
        cbar.ax.tick_params(labelsize=20)
        plt.xlabel(r'$x / L$', fontsize=22)
        plt.ylabel(r'$z / h$', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'v_PINN_{i}_case{num}.eps'), format='eps', bbox_inches='tight')
        plt.close()

        # Plot u Exact
        plt.figure()
        plt.contourf(xxPINN/L, yyPINN/h, PhiX[:, :, i], levels=20, alpha=1, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label(r'$u \, \mathrm{Analytical} \, [\mathrm{m/s}]$', fontsize=22)
        cbar.ax.tick_params(labelsize=20)
        plt.xlabel(r'$x / L$', fontsize=22)
        plt.ylabel(r'$z / h$', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'u_Exact_{i}_case{num}.eps'), format='eps', bbox_inches='tight')
        plt.close()

        # Plot v Exact
        plt.figure()
        plt.contourf(xxPINN/L, yyPINN/h, PhiY[:, :, i], levels=20, alpha=1, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label(r'$w \, \mathrm{Analytical} \, [\mathrm{m/s}]$', fontsize=22)
        cbar.ax.tick_params(labelsize=20)
        plt.xlabel(r'$x / L$', fontsize=22)
        plt.ylabel(r'$z / h$', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'v_Exact_{i}_case{num}.eps'), format='eps', bbox_inches='tight')
        plt.close()

        # Plot u Error
        u_error = PhiXError[:, :, i]
        plt.figure()
        plt.contourf(xxPINN/L, yyPINN/h, u_error, levels=20, alpha=1, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label(r'$u \, \mathrm{Error} \, [\mathrm{\%}]$', fontsize=22)
        cbar.ax.tick_params(labelsize=20)
        plt.xlabel(r'$x / L$', fontsize=22)
        plt.ylabel(r'$z / h$', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'u_Error_{i}_case{num}.eps'), format='eps', bbox_inches='tight')
        plt.close()

        # Plot v Error
        v_error = PhiYError[:, :, i]
        plt.figure()
        plt.contourf(xxPINN/L, yyPINN/h, v_error, levels=20, alpha=1, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label(r'$w \, \mathrm{Error} \, [\mathrm{\%}]$', fontsize=22)
        cbar.ax.tick_params(labelsize=20)
        plt.xlabel(r'$x / L$', fontsize=22)
        plt.ylabel(r'$z / h$', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'v_Error_{i}_case{num}.eps'), format='eps', bbox_inches='tight')
        plt.close()

    ctList.append(data["ct"])
    uPeriodicityErrorList.append(np.mean(xBoundaryUDiff)/np.max(PhiX[:,:,:])*100)
    vPeriodicityErrorList.append(np.mean(xBoundaryVDiff)/np.max(PhiY[:,:,:])*100)

PhiXMeanError = np.mean(PhiXMeanPointDifference)
PhiXStandardDeviation = np.std(PhiXMeanPointDifference)
PhiYMeanError = np.mean(PhiYMeanPointDifference)
PhiYStandardDeviation = np.std(PhiYMeanPointDifference)
ctMean = np.mean(ctList)
ctStandardDeviation = np.std(ctList)
uPeriodicityError = np.mean(uPeriodicityErrorList)
vPeriodicityError = np.mean(vPeriodicityErrorList)
uPeriodicityErrorStandardDeviation = np.std(uPeriodicityErrorList)
vPeriodicityErrorStandardDeviation = np.std(vPeriodicityErrorList)

print("PhiXMeanPointDifference", PhiXMeanPointDifference)
print("PhiXMeanError", PhiXMeanError)
print("PhiXStandardDeviation", PhiXStandardDeviation)

print("PhiYMeanPointDifference", PhiYMeanPointDifference)
print("PhiYMeanError", PhiYMeanError)
print("PhiYStandardDeviation", PhiYStandardDeviation)

print("ctList", ctList)
print("ctMean", ctMean)
print("ctStandardDeviation", ctStandardDeviation)

print("uPeriodicityError", uPeriodicityError)
print("uPeriodicityErrorStandardDeviation", uPeriodicityErrorStandardDeviation)
print("vPeriodicityError", vPeriodicityError)
print("vPeriodicityErrorStandardDeviation", vPeriodicityErrorStandardDeviation)
