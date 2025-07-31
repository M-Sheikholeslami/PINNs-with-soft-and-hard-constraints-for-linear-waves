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
data_name = f"SoftPeriodic_{nn}_{nl}_{act}_{1000}_9261_2402"

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
    # data = np.load(f"pred/res_{data_name}_{domain}_{num}.npz")
    data = np.load(f"pred/{data_name}_{num}.npz")
    # data = np.load(f"pred/res_{data_name}.npz")

    xxPINN, yyPINN = np.meshgrid(xDomain, yDomain)
    tttPINN, yyyPINN = np.meshgrid(tDomain, yDomain)


    pred = data["pred"]
    ###### Shifting the analytical solution #######
    RefPred = pred[0,0,0,0]
    RefExact = Phi[0,0,0] 
    # print("RefExact", RefExact)
    # print("pred[0,0,0,0] = ", pred[0,0,0,0])
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
    xComparison = 1 # the x index at which the contours the exact and PINN solutions are plotted
    tComparison = 0 # the time index at which the contours the exact and PINN solutions are plotted
    tContour = 0 # the time index at which the contours are plotted

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
            ax.set_xlabel('x / L', fontsize=14)  # Set x label for the subplot
            ax.set_ylabel('z / h', fontsize=14)  # Set y label for the subplot
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
            ax.set_xlabel('x / L', fontsize=14)  # Set x label for the subplot
            ax.set_ylabel('z / h', fontsize=14)  # Set y label for the subplot
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
            ax.set_xlabel('x / L', fontsize=14)  # Set x label for the subplot
            ax.set_ylabel('z / h', fontsize=14)  # Set y label for the subplot
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
    plt.legend(loc='upper center', ncol=4, prop={'size': 8})  # Adjusted legend
    plt.savefig(os.path.join(save_dir, f'UPeriodError_case{num}.png'), bbox_inches='tight')
    plt.close()

    plt.figure(10)
    for k in range(tn):
        label = f"Time _{k}"
        plt.plot(yDomain,xBoundaryVDiff[k,:], label=label)
        plt.xlabel('z [m]', fontsize= 14)
    plt.ylabel('w Difference at x ends [m/s]', fontsize= 14)
    plt.legend(loc='upper center', ncol=4, prop={'size': 8})  # Adjusted legend
    plt.savefig(os.path.join(save_dir, f'VPeriodError_case{num}.png'), bbox_inches='tight')
    plt.close()

    plt.figure(12)
    plt.semilogy(data["hist"][:,0],label=r'$L_\mathrm{Total}$')
    plt.semilogy(data["hist"][:,1],label=r'$L_\mathrm{GE}$')
    # epsilon = 1e-10
    # plt.semilogy(data["hist"][:,2]+ epsilon,label="loss_KBBC")
    plt.semilogy(data["hist"][:,2],label=r'$L_\mathrm{KBBC}$')
    plt.semilogy(data["hist"][:,3],label=r'$L_\mathrm{KFSBC}$')
    plt.semilogy(data["hist"][:,5],label="loss_periodic")
    # plt.semilogy(data["hist"][:,4],label=r'$L_\mathrm{DFSBC}$')
    # plt.semilogy(data["hist"][:,5],label="loss_Phi_x")
    # plt.semilogy(data["hist"][:,6],label="loss_Phi_y")
    plt.legend(fontsize=14)
    plt.xlabel(r'$Epoch$', fontsize=20)
    plt.ylabel(r'$Loss$', fontsize=20)
    plt.xticks(fontsize=18)  # Increase x-axis number font size
    plt.yticks(fontsize=18)  # Increase y-axis number font size
    plt.savefig(f'Loss_case{num}.png' , dpi=300)
    # plt.savefig(os.path.join(save_dir, f'Loss_case{num}.eps'), format='eps', bbox_inches='tight')
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
        # plt.contourf(xxPINN/L, yyPINN/h, pred[1, :, :, i]/(np.max(PhiX[:,:,:])), levels=20, alpha=1, cmap='viridis')
        plt.contourf(xxPINN/L, yyPINN/h, pred[1, :, :, i], levels=20, alpha=1, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label(r'$u \, \mathrm{PINN} \, [\mathrm{m/s}]$', fontsize=22)
        cbar.ax.tick_params(labelsize=20)  # Increase colorbar tick font size
        plt.xlabel(r'$x / L$', fontsize=22)
        plt.ylabel(r'$z / h$', fontsize=22)
        plt.xticks(fontsize=20)  # Increase x-axis number font size
        plt.yticks(fontsize=20)  # Increase y-axis number font size
        # plt.title(f'u PINN at {time_label}')  # Uncommented and using time_label
        plt.tight_layout()
        # plt.text(0.95, 0.95, time_label, ha='right', va='top', transform=plt.gca().transAxes, fontsize=12)
        plt.savefig(os.path.join(save_dir, f'u_PINN_{i}_case{num}.eps'), format='eps', bbox_inches='tight')
        plt.close()

        # Plot v PINN
        plt.figure()
        # plt.contourf(xxPINN/L, yyPINN/h, pred[2, :, :, i]/(np.max(PhiY[:,:,:])), levels=20, alpha=1, cmap='viridis')
        plt.contourf(xxPINN/L, yyPINN/h, pred[2, :, :, i], levels=20, alpha=1, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label(r'$w \, \mathrm{PINN} \, [\mathrm{m/s}]$', fontsize=22)
        cbar.ax.tick_params(labelsize=20)  # Increase colorbar tick font size
        plt.xlabel(r'$x / L$', fontsize=22)
        plt.ylabel(r'$z / h$', fontsize=22)
        plt.xticks(fontsize=20)  # Increase x-axis number font size
        plt.yticks(fontsize=20)  # Increase y-axis number font size
        # plt.title(f'v PINN at {time_label}')  # Uncommented and using time_label
        plt.tight_layout()
        # plt.text(0.95, 0.95, time_label, ha='right', va='top', transform=plt.gca().transAxes, fontsize=12)
        plt.savefig(os.path.join(save_dir, f'v_PINN_{i}_case{num}.eps'), format='eps', bbox_inches='tight')
        plt.close()

        # Plot u Exact
        plt.figure()
        plt.contourf(xxPINN/L, yyPINN/h, PhiX[:, :, i], levels=20, alpha=1, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label(r'$u \, \mathrm{Analytical} \, [\mathrm{m/s}]$', fontsize=22)
        cbar.ax.tick_params(labelsize=20)  # Increase colorbar tick font size
        plt.xlabel(r'$x / L$', fontsize=22)
        plt.ylabel(r'$z / h$', fontsize=22)
        plt.xticks(fontsize=20)  # Increase x-axis number font size
        plt.yticks(fontsize=20)  # Increase y-axis number font size
        # plt.title(f'v PINN at {time_label}')  # Uncommented and using time_label
        plt.tight_layout()
        # plt.text(0.95, 0.95, time_label, ha='right', va='top', transform=plt.gca().transAxes, fontsize=12)
        plt.savefig(os.path.join(save_dir, f'u_Exact_{i}_case{num}.eps'), format='eps', bbox_inches='tight')
        plt.close()

        # Plot v Exact
        plt.figure()
        plt.contourf(xxPINN/L, yyPINN/h, PhiY[:, :, i], levels=20, alpha=1, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label(r'$w \, \mathrm{Analytical} \, [\mathrm{m/s}]$', fontsize=22)
        cbar.ax.tick_params(labelsize=20)  # Increase colorbar tick font size
        plt.xlabel(r'$x / L$', fontsize=22)
        plt.ylabel(r'$z / h$', fontsize=22)
        plt.xticks(fontsize=20)  # Increase x-axis number font size
        plt.yticks(fontsize=20)  # Increase y-axis number font size
        # plt.title(f'v PINN at {time_label}')  # Uncommented and using time_label
        plt.tight_layout()
        # plt.text(0.95, 0.95, time_label, ha='right', va='top', transform=plt.gca().transAxes, fontsize=12)
        plt.savefig(os.path.join(save_dir, f'v_Exact_{i}_case{num}.eps'), format='eps', bbox_inches='tight')
        plt.close()

        # Plot u Error
        u_error = PhiXError[:, :, i]  # Adjust this line to correctly reference v_error
        plt.figure()
        plt.contourf(xxPINN/L, yyPINN/h, u_error, levels=20, alpha=1, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label(r'$u \, \mathrm{Error} \, [\mathrm{\%}]$', fontsize=22)
        cbar.ax.tick_params(labelsize=20)  # Increase colorbar tick font size
        plt.xlabel(r'$x / L$', fontsize=22)
        plt.ylabel(r'$z / h$', fontsize=22)
        plt.xticks(fontsize=20)  # Increase x-axis number font size
        plt.yticks(fontsize=20)  # Increase y-axis number font size
        # plt.title(f'u Error at {time_label}')  # Uncommented and using time_label
        # plt.text(0.95, 0.95, time_label, ha='right', va='top', transform=plt.gca().transAxes, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'u_Error_{i}_case{num}.eps'), format='eps', bbox_inches='tight')
        plt.close()

        # Plot v Error
        v_error = PhiYError[:, :, i]  # Adjust this line to correctly reference v_error
        plt.figure()
        plt.contourf(xxPINN/L, yyPINN/h, v_error, levels=20, alpha=1, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label(r'$w \, \mathrm{Error} \, [\mathrm{\%}]$', fontsize=22)
        cbar.ax.tick_params(labelsize=20)  # Increase colorbar tick font size
        plt.xlabel(r'$x / L$', fontsize=22)
        plt.ylabel(r'$z / h$', fontsize=22)
        plt.xticks(fontsize=20)  # Increase x-axis number font size
        plt.yticks(fontsize=20)  # Increase y-axis number font size
        # plt.title(f'v Error at {time_label}')  # Uncommented and using time_label
        # plt.text(0.95, 0.95, time_label, ha='right', va='top', transform=plt.gca().transAxes, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'v_Error_{i}_case{num}.eps'), format='eps', bbox_inches='tight')
        plt.close()

    ctList.append(data["ct"])
    uPeriodicityErrorList.append(np.mean(xBoundaryUDiff)/np.max(PhiX[:,:,:])*100)
    vPeriodicityErrorList.append(np.mean(xBoundaryVDiff)/np.max(PhiY[:,:,:])*100)
# print("PhiMeanPointDifference", PhiMeanPointDifference)
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

# # Plot PhiXMeanPointDifference vs. Number of Cases
# plt.figure()
# plt.plot(range(trialNum), PhiXMeanPointDifference, label='PhiXMeanPointDifference', color='blue', linewidth=2)
# plt.xlabel('Number of Cases', fontsize=14)
# plt.ylabel('PhiXMeanPointDifference', fontsize=14)
# plt.title('PhiXMeanPointDifference vs. Number of Cases', fontsize=16)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("PhiXMeanPointDifference_vs_Number_of_Cases.png", dpi=300)

# # Plot PhiYMeanPointDifference vs. Number of Cases
# plt.figure()
# plt.plot(range(trialNum), PhiYMeanPointDifference, label='PhiYMeanPointDifference', color='red', linewidth=2)
# plt.xlabel('Number of Cases', fontsize=14)
# plt.ylabel('PhiYMeanPointDifference', fontsize=14)
# plt.title('PhiYMeanPointDifference vs. Number of Cases', fontsize=16)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("PhiYMeanPointDifference_vs_Number_of_Cases.png", dpi=300)

# # Plot ct vs. Number of Cases
# plt.figure()
# plt.plot(range(trialNum), ctList, label='ct', color='green', linewidth=2)
# plt.xlabel('Number of Cases', fontsize=14)
# plt.ylabel('ct', fontsize=14)
# plt.title('ct vs. Number of Cases', fontsize=16)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("ct_vs_Number_of_Cases.png", dpi=300)


# # Placeholder for actual values - these should be filled with the final loss values over 100 cases
# Loss_Total = []
# Loss_GE = []
# Loss_KBBC = []
# Loss_KFSBC = []

# # Loop to extract the last loss value of each type from each saved file
# for i in range(trialNum):
#     try:
#         data = np.load(f"pred/res_{data_name}_{i}.npz")
#         hist = data["hist"]
#         Loss_Total.append(hist[-1, 0])   # Last epoch's total loss
#         Loss_GE.append(hist[-1, 1])      # Last epoch's GE loss
#         Loss_KBBC.append(hist[-1, 2])    # Last epoch's KBBC loss
#         Loss_KFSBC.append(hist[-1, 3])   # Last epoch's KFSBC loss
#     except FileNotFoundError:
#         continue

# # Create one plot with all 4 curves
# plt.figure(figsize=(10, 6))
# plt.semilogy(range(len(Loss_Total)), Loss_Total, label=r'$L_\mathrm{Total}$', linewidth=2)
# plt.semilogy(range(len(Loss_GE)), Loss_GE, label=r'$L_\mathrm{GE}$', linewidth=2)
# plt.semilogy(range(len(Loss_KBBC)), Loss_KBBC, label=r'$L_\mathrm{KBBC}$', linewidth=2)
# plt.semilogy(range(len(Loss_KFSBC)), Loss_KFSBC, label=r'$L_\mathrm{KFSBC}$', linewidth=2)
# plt.xlabel('Case Number', fontsize=14)
# plt.ylabel('Final Loss Value (log scale)', fontsize=14)
# plt.title('Final Loss Values Across 100 Cases', fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(True, which="both", ls='--', alpha=0.6)
# plt.tight_layout()
# plt.savefig("Final_Losses_Over_100_Cases.png", dpi=300)

# # Plot all three metrics
# plt.figure(figsize=(10, 6))
# plt.semilogy(range(len(Loss_Total)), Loss_Total, label=r'$L_\mathrm{Total}$', linewidth=2, color='black')
# plt.plot(range(len(PhiXMeanPointDifference)), PhiXMeanPointDifference, label='PhiX Mean Difference', linewidth=2, color='blue')
# plt.plot(range(len(PhiYMeanPointDifference)), PhiYMeanPointDifference, label='PhiY Mean Difference', linewidth=2, color='red')
# plt.xlabel('Case Number', fontsize=14)
# plt.ylabel('Value', fontsize=14)
# plt.title('Final Total Loss and Point Differences Across 100 Cases', fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("Loss_vs_PhiDiff_Over_100_Cases.png", dpi=300)

# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Left y-axis for loss (log scale)
# ax1.set_xlabel('Case Number', fontsize=14)
# ax1.set_ylabel('Final Loss Value (log scale)', fontsize=14, color='black')
# ax1.semilogy(range(len(Loss_Total)), Loss_Total, label=r'$L_\mathrm{Total}$', color='black', linewidth=2)
# ax1.tick_params(axis='y', labelcolor='black')

# # Right y-axis for point differences (linear scale)
# ax2 = ax1.twinx()
# ax2.set_ylabel('Mean Point Difference [\%]', fontsize=14, color='blue')
# ax2.plot(range(len(PhiXMeanPointDifference)), PhiXMeanPointDifference, label='PhiX Mean Difference', color='blue', linewidth=2)
# ax2.tick_params(axis='y', labelcolor='blue')

# # Add legends
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=12)

# plt.title('Final Total Loss and Point Differences Across Cases', fontsize=16)
# fig.tight_layout()
# plt.grid(True, linestyle='--', alpha=0.7)
# # Add dashed lines connecting each case for visual link
# for i in range(trialNum):
#     ax1.plot([i, i], [Loss_Total[i], ax2.get_ylim()[0]], linestyle='--', color='gray', alpha=0.5)
#     ax2.plot([i, i], [ax1.get_ylim()[0], PhiXMeanPointDifference[i]], linestyle='--', color='gray', alpha=0.5)
# plt.savefig("Loss_vs_PhiDiffX_TwoYAxis.png", dpi=300)
# plt.close()

# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Left y-axis for loss (log scale)
# ax1.set_xlabel('Case Number', fontsize=14)
# ax1.set_ylabel('Final Loss Value (log scale)', fontsize=14, color='black')
# ax1.semilogy(range(len(Loss_Total)), Loss_Total, label=r'$L_\mathrm{Total}$', color='black', linewidth=2)
# ax1.tick_params(axis='y', labelcolor='black')

# # Right y-axis for point differences (linear scale)
# ax2 = ax1.twinx()
# ax2.set_ylabel('Mean Point Difference [\%]', fontsize=14, color='blue')
# ax2.plot(range(len(PhiYMeanPointDifference)), PhiYMeanPointDifference, label='PhiY Mean Difference', color='red', linewidth=2)
# ax2.tick_params(axis='y', labelcolor='blue')

# # Add legends
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=12)

# plt.title('Final Total Loss and Point Differences Across Cases', fontsize=16)
# fig.tight_layout()
# plt.grid(True, linestyle='--', alpha=0.7)
# # Add dashed lines connecting each case for visual link
# for i in range(trialNum):
#     ax1.plot([i, i], [Loss_Total[i], ax2.get_ylim()[0]], linestyle='--', color='gray', alpha=0.5)
#     ax2.plot([i, i], [ax1.get_ylim()[0], PhiYMeanPointDifference[i]], linestyle='--', color='gray', alpha=0.5)

# plt.savefig("Loss_vs_PhiDiffY_TwoYAxis.png", dpi=300)
# plt.close()

# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Left y-axis for loss (log scale)
# ax1.set_xlabel('Case Number', fontsize=14)
# ax1.set_ylabel('Final Loss Value (log scale)', fontsize=14, color='black')
# ax1.semilogy(range(len(Loss_GE)), Loss_GE, label=r'$L_\mathrm{GE}$', color='black', linewidth=2)
# ax1.tick_params(axis='y', labelcolor='black')

# # Right y-axis for point differences (linear scale)
# ax2 = ax1.twinx()
# ax2.set_ylabel('Mean Point Difference [\%]', fontsize=14, color='blue')
# ax2.plot(range(len(PhiXMeanPointDifference)), PhiXMeanPointDifference, label='PhiX Mean Difference', color='red', linewidth=2)
# ax2.tick_params(axis='y', labelcolor='blue')

# # Add legends
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=12)

# plt.title('Final Total Loss and Point Differences Across Cases', fontsize=16)
# fig.tight_layout()
# plt.grid(True, linestyle='--', alpha=0.7)
# # Add dashed lines connecting each case for visual link
# for i in range(trialNum):
#     ax1.plot([i, i], [Loss_GE[i], ax2.get_ylim()[0]], linestyle='--', color='gray', alpha=0.5)
#     ax2.plot([i, i], [ax1.get_ylim()[0], PhiXMeanPointDifference[i]], linestyle='--', color='gray', alpha=0.5)

# plt.savefig("Loss_GE_vs_PhiDiffY_TwoYAxis.png", dpi=300)
# plt.close()

# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Left y-axis for loss (log scale)
# ax1.set_xlabel('Case Number', fontsize=14)
# ax1.set_ylabel('Final Loss Value (log scale)', fontsize=14, color='black')
# ax1.semilogy(range(len(Loss_KBBC)), Loss_KBBC, label=r'$L_\mathrm{KBBC}$', color='black', linewidth=2)
# ax1.tick_params(axis='y', labelcolor='black')

# # Right y-axis for point differences (linear scale)
# ax2 = ax1.twinx()
# ax2.set_ylabel('Mean Point Difference [\%]', fontsize=14, color='blue')
# ax2.plot(range(len(PhiXMeanPointDifference)), PhiXMeanPointDifference, label='PhiX Mean Difference', color='red', linewidth=2)
# ax2.tick_params(axis='y', labelcolor='blue')

# # Add legends
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=12)

# plt.title('Final Total Loss and Point Differences Across Cases', fontsize=16)
# fig.tight_layout()
# plt.grid(True, linestyle='--', alpha=0.7)
# # Add dashed lines connecting each case for visual link
# for i in range(trialNum):
#     ax1.plot([i, i], [Loss_KBBC[i], ax2.get_ylim()[0]], linestyle='--', color='gray', alpha=0.5)
#     ax2.plot([i, i], [ax1.get_ylim()[0], PhiXMeanPointDifference[i]], linestyle='--', color='gray', alpha=0.5)

# plt.savefig("Loss_KBBC_vs_PhiDiffX_TwoYAxis.png", dpi=300)
# plt.close()

# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Left y-axis for loss (log scale)
# ax1.set_xlabel('Case Number', fontsize=14)
# ax1.set_ylabel('Final Loss Value (log scale)', fontsize=14, color='black')
# ax1.semilogy(range(len(Loss_KFSBC)), Loss_KFSBC, label=r'$L_\mathrm{KFSBC}$', color='black', linewidth=2)
# ax1.tick_params(axis='y', labelcolor='black')

# # Right y-axis for point differences (linear scale)
# ax2 = ax1.twinx()
# ax2.set_ylabel('Mean Point Difference [\%]', fontsize=14, color='blue')
# ax2.plot(range(len(PhiXMeanPointDifference)), PhiXMeanPointDifference, label='PhiX Mean Difference', color='red', linewidth=2)
# ax2.tick_params(axis='y', labelcolor='blue')

# # Add legends
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=12)

# plt.title('Final Total Loss and Point Differences Across Cases', fontsize=16)
# fig.tight_layout()
# plt.grid(True, linestyle='--', alpha=0.7)
# # Add dashed lines connecting each case for visual link
# for i in range(trialNum):
#     ax1.plot([i, i], [Loss_KFSBC[i], ax2.get_ylim()[0]], linestyle='--', color='gray', alpha=0.5)
#     ax2.plot([i, i], [ax1.get_ylim()[0], PhiXMeanPointDifference[i]], linestyle='--', color='gray', alpha=0.5)

# plt.savefig("Loss_KFSBC_vs_PhiDiffX_TwoYAxis.png", dpi=300)
# plt.close()