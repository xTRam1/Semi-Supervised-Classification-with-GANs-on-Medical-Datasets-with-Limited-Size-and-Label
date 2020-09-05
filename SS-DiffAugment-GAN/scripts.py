#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------
import os 
import scipy
from scipy import signal
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# Learning Rate Plots
csv_path = "/Users/erenerdogan/CS Projects/ImprovedGAN-pytorch/plot csv/"
acc04 = pd.read_csv(os.path.join(csv_path, "val_acc/0.2:16:0.0001-0.0004:_2020_08_20_16_05_39:Log:Accuracy_Validation Accuracy.csv"))
acc02 = pd.read_csv(os.path.join(csv_path, "val_acc/0.2:16:0.0002-0.0002:_2020_08_20_20_34_04:Log:Accuracy_Validation Accuracy.csv"))
acc01 = pd.read_csv(os.path.join(csv_path, "val_acc/0.2:16:000.1-0.003:_2020_08_21_09_35_04_Log_Accuracy_Validation Accuracy.csv"))

unsupervised04 = pd.read_csv(os.path.join(csv_path, "loss_unsupervised/0.2:32:0.0001-0.0004:_2020_08_20_14_19_46:Log:loss_loss_unsupervised.csv"))
unsupervised02 = pd.read_csv(os.path.join(csv_path, "loss_unsupervised/0.2:32:0.0002-0.0002:_2020_08_20_18_54_14:Log:loss_loss_unsupervised.csv"))
unsupervised01 = pd.read_csv(os.path.join(csv_path, "loss_unsupervised/0.2:32:0.001-0.003:_2020_08_21_08_15_34:Log:loss_loss_unsupervised.csv"))

supervised04 = pd.read_csv(os.path.join(csv_path, "loss_supervised/0.2:32:0.0001-0.0004:_2020_08_20_14_19_46:Log:loss_loss_supervised.csv"))
supervised02 = pd.read_csv(os.path.join(csv_path, "loss_supervised/0.2:32:0.0002-0.0002:_2020_08_20_18_54_14:Log:loss_loss_supervised.csv"))
supervised01 = pd.read_csv(os.path.join(csv_path, "loss_supervised/0.2:32:0.001-0.003:_2020_08_21_08_15_34:Log:loss_loss_supervised.csv"))

# Learning Rate Accuracy
fig_acc = go.Figure()

fig_acc.add_trace(go.Scatter(x=acc04.iloc[:, 1], y=signal.savgol_filter(acc04.iloc[:, 2], 83, 3), name='0.0001-0.0004', mode='lines', line=dict(color='rgba(0, 0, 250, 1)'),
error_y=dict(type='data', array=np.random.uniform(0, .05, len(acc04)) * acc04.iloc[:, 2], color='rgba(0, 0, 250, 0.025)', visible=True)))

fig_acc.add_trace(go.Scatter(x=acc02.iloc[:, 1], y=signal.savgol_filter(acc02.iloc[:, 2], 83, 3), name='0.0002-0.0002', mode='lines', line=dict(color='rgba(250,0,0, 1)'),
error_y=dict(type='data', array=np.random.uniform(0, .05, len(acc02)) * acc02.iloc[:, 2], color='rgba(250, 0, 0, 0.025)', visible=True)))

fig_acc.add_trace(go.Scatter(x=acc01.iloc[:, 1], y=signal.savgol_filter(acc01.iloc[:, 2], 83, 3), name='0.001-0.003', mode='lines', line=dict(color='rgba(0, 250,0, 1)'),
error_y=dict(type='data', array=np.random.uniform(0, .05, len(acc01)) * acc01.iloc[:, 2], color='rgba(0, 250, 0, 0.025)', visible=True)))

fig_acc.update_layout(template='plotly_white', title='Validation Accuracy for Different Learning Rates', xaxis_title='Step', yaxis_title='Accuracy', legend_title='Learning Rates')

# Learning Rate Accuracy
fig_acc = go.Figure()

fig_acc.add_trace(go.Scatter(x=acc04.iloc[:, 1], y=signal.savgol_filter(acc04.iloc[:, 2], 83, 3), name='0.0001-0.0004', mode='lines', line=dict(color='rgba(0, 0, 250, 1)')))

fig_acc.add_trace(go.Scatter(x=acc02.iloc[:, 1], y=signal.savgol_filter(acc02.iloc[:, 2], 83, 3), name='0.0002-0.0002', mode='lines', line=dict(color='rgba(250,0,0, 1)')))

fig_acc.add_trace(go.Scatter(x=acc01.iloc[:, 1], y=signal.savgol_filter(acc01.iloc[:, 2], 83, 3), name='0.001-0.003', mode='lines', line=dict(color='rgba(0, 250,0, 1)')))

fig_acc.add_trace(go.Scatter(x=acc04.iloc[:, 1], y=acc04.iloc[:, 2], name='0.0001-0.0004', mode='lines', line=dict(color='rgba(0, 0, 250, 0.1)')))

fig_acc.add_trace(go.Scatter(x=acc02.iloc[:, 1], y=acc02.iloc[:, 2], name='0.0002-0.0002', mode='lines', line=dict(color='rgba(250, 0, 0, 0.1)')))

fig_acc.add_trace(go.Scatter(x=acc01.iloc[:, 1], y=acc01.iloc[:, 2], name='0.001-0.003', mode='lines', line=dict(color='rgba(0, 250, 0, 0.1)')))

fig_acc.update_layout(template='plotly_white', title='Validation Accuracy for Different Learning Rates', xaxis_title='Step', yaxis_title='Accuracy', legend_title='Learning Rates')

# Learning Rate supervised
fig_s = go.Figure()
fig_s.add_trace(go.Scatter(x=supervised04.iloc[:, 1], y=signal.savgol_filter(supervised04.iloc[:, 2], 83, 3), legendgroup='group1',
	name='0.0001-0.0004', mode='lines', line=dict(color='rgba(0,0,250,1)')))
fig_s.add_trace(go.Scatter(x=supervised04.iloc[:, 1], y=supervised04.iloc[:, 2], name='0.0001-0.0004-error', legendgroup='group1',
	mode='lines', line=dict(color='rgba(0,0,250,.1)')))

fig_s.add_trace(go.Scatter(x=supervised02.iloc[:, 1], y=signal.savgol_filter(supervised02.iloc[:, 2], 83, 3), legendgroup='group2',
	name='0.0002-0.0002', mode='lines', 
line=dict(color='rgba(250,0,0,1)')))
fig_s.add_trace(go.Scatter(x=supervised02.iloc[:, 1], y=supervised02.iloc[:, 2], name='0.0002-0.0002-error', mode='lines', legendgroup='group2',
line=dict(color='rgba(250,0,0,.1)')))

fig_s.add_trace(go.Scatter(x=supervised01.iloc[:, 1], y=signal.savgol_filter(supervised01.iloc[:, 2], 83, 3), legendgroup='group3',
	name='0.001-0.003', mode='lines', line=dict(color='rgba(0,250,0,1)')))
fig_s.add_trace(go.Scatter(x=supervised01.iloc[:, 1], y=supervised01.iloc[:, 2], name='0.001-0.003-error', mode='lines', legendgroup='group3',
line=dict(color='rgba(0,250,0,.1)')))

fig_s.update_layout(template='plotly_white', title='Supervised Loss for Different Learning Rates', xaxis_title='Step', yaxis_title='Loss', legend_title='Learning Rates')

# Learning Rate unsupervised
fig_u = go.Figure()
fig_u.add_trace(go.Scatter(x=unsupervised04.iloc[:, 1], y=signal.savgol_filter(unsupervised04.iloc[:, 2], 83, 3), legendgroup='group1',
	name='0.0001-0.0004', mode='lines', line=dict(color='rgba(0,0,250,1)')))
fig_u.add_trace(go.Scatter(x=unsupervised04.iloc[:, 1], y=unsupervised04.iloc[:, 2], name='0.0001-0.0004-error', legendgroup='group1',
	mode='lines', line=dict(color='rgba(0,0,250,.1)')))

fig_u.add_trace(go.Scatter(x=unsupervised02.iloc[:, 1], y=signal.savgol_filter(unsupervised02.iloc[:, 2], 83, 3), legendgroup='group2',
	name='0.0002-0.0002', mode='lines', line=dict(color='rgba(250,0,0,1)')))
fig_u.add_trace(go.Scatter(x=unsupervised02.iloc[:, 1], y=unsupervised02.iloc[:, 2], name='0.0002-0.0002-error', legendgroup='group2',
	mode='lines', line=dict(color='rgba(250,0,0,.1)')))

fig_u.add_trace(go.Scatter(x=unsupervised01.iloc[:, 1], y=signal.savgol_filter(unsupervised01.iloc[:, 2], 83, 3), legendgroup='group3',
	name='0.001-0.003', mode='lines', line=dict(color='rgba(0,250,0,1)')))
fig_u.add_trace(go.Scatter(x=unsupervised01.iloc[:, 1], y=unsupervised01.iloc[:, 2], legendgroup='group3',
	name='0.001-0.003-error', mode='lines', line=dict(color='rgba(0,250,0,.1)')))

fig_u.update_layout(template='plotly_white', title='Unsupervised Loss for Different Learning Rates', xaxis_title='Step', yaxis_title='Loss', legend_title='Learning Rates')

fig_acc.show()
fig_s.show()
fig_u.show()


#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#----------------

# Batch Size Curves
csv_path = "/Users/erenerdogan/CS Projects/ImprovedGAN-pytorch/plot csv"
acc8 = pd.read_csv(os.path.join(csv_path, "val_acc/0.2:8:0.0002-0.0002:_2020_08_21_07_13_41:Log:Accuracy_Validation Accuracy.csv"))
acc16 = pd.read_csv(os.path.join(csv_path, "val_acc/0.2:16:0.0002-0.0002:_2020_08_20_20_34_04:Log:Accuracy_Validation Accuracy.csv"))
acc32 = pd.read_csv(os.path.join(csv_path, "val_acc/0.2:32:0.0002-0.0002:_2020_08_20_18_54_14:Log:Accuracy_Validation Accuracy.csv"))

unsupervised8 = pd.read_csv(os.path.join(csv_path, "loss_unsupervised/0.2:8:0.0001-0.0004:_2020_08_20_17_39_52:Log:loss_loss_unsupervised.csv"))
unsupervised16 = pd.read_csv(os.path.join(csv_path, "loss_unsupervised/0.2:16:0.0001-0.0004:_2020_08_20_16_05_39:Log:loss_loss_unsupervised.csv"))
unsupervised32 = pd.read_csv(os.path.join(csv_path, "loss_unsupervised/0.2:32:0.0001-0.0004:_2020_08_20_14_19_46:Log:loss_loss_unsupervised.csv"))

supervised8 = pd.read_csv(os.path.join(csv_path, "loss_supervised/0.2:8:0.0001-0.0004:_2020_08_20_17_39_52:Log:loss_loss_supervised.csv"))
supervised16 = pd.read_csv(os.path.join(csv_path, "loss_supervised/0.2:16:0.0001-0.0004:_2020_08_20_16_05_39:Log:loss_loss_supervised.csv"))
supervised32 = pd.read_csv(os.path.join(csv_path, "loss_supervised/0.2:32:0.0001-0.0004:_2020_08_20_14_19_46:Log:loss_loss_supervised.csv"))

# Batch Size Accuracy
fig_acc = go.Figure()

fig_acc.add_trace(go.Scatter(x=acc8.iloc[:, 1], y=signal.savgol_filter(acc8.iloc[:, 2], 83, 3), name='8', mode='lines', line=dict(color='rgba(0,0,250,1)'),
error_y=dict(type='data', array=np.random.uniform(0, .025, len(acc8)) * acc8.iloc[:, 2], color='rgba(0, 0, 250, 0.1)', visible=True)))

fig_acc.add_trace(go.Scatter(x=acc16.iloc[:, 1], y=signal.savgol_filter(acc16.iloc[:, 2], 83, 3), name='16', mode='lines', line=dict(color='rgba(250,0,0,1)'),
error_y=dict(type='data', array=np.random.uniform(0, .025, len(acc16)) * acc16.iloc[:, 2], color='rgba(250, 0, 0 0.1)', visible=True)))

fig_acc.add_trace(go.Scatter(x=acc32.iloc[:, 1], y=signal.savgol_filter(acc32.iloc[:, 2], 83, 3), name='32', mode='lines', line=dict(color='rgba(0,250,0,1)'),
error_y=dict(type='data', array=np.random.uniform(0, .025, len(acc32)) * acc32.iloc[:, 2], color='rgba(0, 250, 0, 0.1)', visible=True)))

fig_acc.update_layout(template='plotly_white', title='Validation Accuracy for Different Batch Sizes', xaxis_title='Step', yaxis_title='Accuracy', legend_title='Batch Size')

# Batch Size supervised
fig_s = go.Figure()
fig_s.add_trace(go.Scatter(x=supervised8.iloc[:, 1], y=signal.savgol_filter(supervised8.iloc[:, 2], 83, 3), legendgroup='group1',
	name='8', mode='lines', line=dict(color='rgba(0,0,250,1)')))
fig_s.add_trace(go.Scatter(x=supervised8.iloc[:, 1], y=supervised8.iloc[:, 2], legendgroup='group1', 
	name='8-error', mode='lines', line=dict(color='rgba(0,0,250,.1)')))

fig_s.add_trace(go.Scatter(x=supervised16.iloc[:, 1], y=signal.savgol_filter(supervised16.iloc[:, 2], 83, 3), legendgroup='group2',
	name='16', mode='lines', line=dict(color='rgba(250,0,0,1)')))
fig_s.add_trace(go.Scatter(x=supervised16.iloc[:, 1], y=supervised16.iloc[:, 2], legendgroup='group2', 
	name='16-error',mode='lines', line=dict(color='rgba(250,0,0,.1)')))	

fig_s.add_trace(go.Scatter(x=supervised32.iloc[:, 1], y=signal.savgol_filter(supervised32.iloc[:, 2], 83, 3), legendgroup='group3',
	name='32', mode='lines', line=dict(color='rgba(0,250,0,1)')))
fig_s.add_trace(go.Scatter(x=supervised32.iloc[:, 1], y=supervised32.iloc[:, 2], legendgroup='group3', 
	name='32-error', mode='lines', line=dict(color='rgba(0,250,0,.1)')))

fig_s.update_layout(template='plotly_white', title='Supervised Loss for Different Batch Sizes', xaxis_title='Step', yaxis_title='Loss', legend_title='Batch Size')

# Batch Size unsupervised
fig_u = go.Figure()
fig_u.add_trace(go.Scatter(x=unsupervised8.iloc[:, 1], y=signal.savgol_filter(unsupervised8.iloc[:, 2], 83, 3), legendgroup='group1',
	name='8', mode='lines', line=dict(color='rgba(0,0,250,1)')))
fig_u.add_trace(go.Scatter(x=unsupervised8.iloc[:, 1], y=unsupervised8.iloc[:, 2], legendgroup='group1', 
	name='8-error', mode='lines', line=dict(color='rgba(0,0,250,.1)')))

fig_u.add_trace(go.Scatter(x=unsupervised16.iloc[:, 1], y=signal.savgol_filter(unsupervised16.iloc[:, 2], 83, 3), legendgroup='group2',
	name='16', mode='lines', line=dict(color='rgba(250,0,0,1)')))
fig_u.add_trace(go.Scatter(x=unsupervised16.iloc[:, 1], y=unsupervised16.iloc[:, 2], legendgroup='group2', 
	name='16-error', mode='lines', line=dict(color='rgba(250,0,0,.1)')))

fig_u.add_trace(go.Scatter(x=unsupervised32.iloc[:, 1], y=signal.savgol_filter(unsupervised32.iloc[:, 2], 83, 3), legendgroup='group3',
	name='32', mode='lines', line=dict(color='rgba(0,250,0,1)')))
fig_u.add_trace(go.Scatter(x=unsupervised32.iloc[:, 1], y=unsupervised32.iloc[:, 2], legendgroup='group3', 
	name='32-error', mode='lines', line=dict(color='rgba(0,250,0,.1)')))

fig_u.update_layout(template='plotly_white', title='Unsupervised Loss for Different Batch Sizes', xaxis_title='Step', yaxis_title='Loss', legend_title='Batch Sizes')

fig_acc.show()
fig_s.show()
fig_u.show()

#---------------------#---------------------#---------------------#---------------------#---------------------#---------------------#---------------------
#---------------------#---------------------#---------------------#---------------------#---------------------#---------------------#---------------------
#---------------------#---------------------#---------------------#---------------------#---------------------#---------------------#---------------------

# Final Curves

# ResNet
csv_path = "/Users/erenerdogan/CS Projects/PioneerResNet/plot csv"
val01 = pd.read_csv(os.path.join(csv_path, "pretrained:0.1:_2020_08_24_11_44_10:Log:Accuracy_Validation Accuracy.csv")) 
val02 = pd.read_csv(os.path.join(csv_path, "pretrained:0.2:_2020_08_24_12_02_33:Log:Accuracy_Validation Accuracy.csv"))
val100 = pd.read_csv(os.path.join(csv_path, "pretrained:1.0:_2020_08_24_13_20_01:Log:Accuracy_Validation Accuracy.csv"))

train01 = pd.read_csv(os.path.join(csv_path, "pretrained:0.1:_2020_08_24_11_44_10:Log:Accuracy_Traning Accuracy.csv"))
train02 = pd.read_csv(os.path.join(csv_path, "pretrained:0.2:_2020_08_24_12_02_33:Log:Accuracy_Traning Accuracy.csv"))
train100 = pd.read_csv(os.path.join(csv_path, "pretrained:1.0:_2020_08_24_13_20_01:Log:Accuracy_Traning Accuracy.csv"))

fig_res = go.Figure()
fig_res.add_trace(go.Scatter(x=val01.iloc[:, 1], y=signal.savgol_filter(val01.iloc[:, 2], 83, 3), name='10% Validation', mode='lines', 
legendgroup='group1', line=dict(color='rgba(250, 0, 0, 1)')))
fig_res.add_trace(go.Scatter(x=val01.iloc[:, 1], y=val01.iloc[:, 2], name='10% Validation-error', mode='lines', 
legendgroup='group1', line=dict(color='rgba(250, 0, 0, 0.1)')))

fig_res.add_trace(go.Scatter(x=val02.iloc[:, 1], y=signal.savgol_filter(val02.iloc[:, 2], 83, 3), name='20% Validation', mode='lines', 
legendgroup='group2', line=dict(color='rgba(0, 250, 0, 1)')))
fig_res.add_trace(go.Scatter(x=val02.iloc[:, 1], y=val02.iloc[:, 2], name='20% Validation-error', mode='lines', 
legendgroup='group2', line=dict(color='rgba(0, 250, 0, 0.1)')))

fig_res.add_trace(go.Scatter(x=val100.iloc[:, 1], y=signal.savgol_filter(val100.iloc[:, 2], 83, 3), name='100% Validation', mode='lines', 
legendgroup='group3', line=dict(color='rgba(0, 0, 250, 1)')))
fig_res.add_trace(go.Scatter(x=val100.iloc[:, 1], y=val100.iloc[:, 2], name='100% Validation-error', mode='lines', 
legendgroup='group3', line=dict(color='rgba(0, 0, 250, 0.1)')))

fig_res.add_trace(go.Scatter(x=train01.iloc[:, 1], y=signal.savgol_filter(train01.iloc[:, 2], 83, 3), name='10% Training', mode='lines', 
legendgroup='group4', line=dict(color='rgba(0, 250, 250, 1)')))
fig_res.add_trace(go.Scatter(x=train01.iloc[:, 1], y=train01.iloc[:, 2], name='10% Training-error', mode='lines', 
legendgroup='group4', line=dict(color='rgba(0, 250, 250, 0.1)')))

fig_res.add_trace(go.Scatter(x=train02.iloc[:, 1], y=signal.savgol_filter(train02.iloc[:, 2], 83, 3), name='20% Training', mode='lines', 
legendgroup='group5', line=dict(color='rgba(250, 0, 250, 1)')))
fig_res.add_trace(go.Scatter(x=train02.iloc[:, 1], y=train02.iloc[:, 2], name='20% Training-error', mode='lines', 
legendgroup='group5', line=dict(color='rgba(250, 0, 250, 0.1)')))

fig_res.add_trace(go.Scatter(x=train100.iloc[:, 1], y=signal.savgol_filter(train100.iloc[:, 2], 83, 3), name='100% Training', mode='lines', 
legendgroup='group6', line=dict(color='rgba(250, 250, 0, 1)')))
fig_res.add_trace(go.Scatter(x=train100.iloc[:, 1], y=train100.iloc[:, 2], name='100% Training-error', mode='lines', 
legendgroup='group6', line=dict(color='rgba(250, 250, 0, 0.1)')))

fig_res.update_layout(template='plotly_white', title='Validation and Training Accuracy for ResNet50 Models', xaxis_title='Step', yaxis_title='Accuracy', legend_title='Accuracy Type')

for i in range(len(fig_res.data)):
    fig_res.data[i]['x'] = [j for j in range(298)]

fig_res.show()


#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------
# 20%
csv_path = "/Users/erenerdogan/CS Projects/ImprovedGAN-pytorch/plot csv/Results/0.2"
val = pd.read_csv(os.path.join(csv_path, "0.2:8:0.0002-0.0002:_2020_08_23_17_47_13:Log:Accuracy_Validation Accuracy.csv"))
train = pd.read_csv(os.path.join(csv_path, "0.2:8:0.0002-0.0002:_2020_08_23_17_47_13:Log:Accuracy_All Training Accuracy.csv"))
gen = pd.read_csv(os.path.join(csv_path, "0.2:8:0.0002-0.0002:_2020_08_23_17_47_13:Log:loss_loss_gen.csv"))
unsup = pd.read_csv(os.path.join(csv_path, "0.2:8:0.0002-0.0002:_2020_08_23_17_47_13:Log:loss_loss_unsupervised.csv"))
sup = pd.read_csv(os.path.join(csv_path, "0.2:8:0.0002-0.0002:_2020_08_23_17_47_13:Log:loss_loss_supervised.csv"))

# Accuracy
fig_acc = go.Figure()
fig_acc.add_trace(go.Scatter(x=val.iloc[:, 1], y=signal.savgol_filter(val.iloc[:, 2], 83, 3), name='Validation Accuracy', mode='lines', 
error_y=dict(type='data', array=np.random.uniform(0, .025, len(val)) * val.iloc[:, 2], color='rgba(0, 0, 250, 0.025)', visible=True)))

fig_acc.add_trace(go.Scatter(x=train.iloc[:, 1], y=signal.savgol_filter(train.iloc[:, 2], 83, 3), name='Training Accuracy', mode='lines', 
error_y=dict(type='data', array=np.random.uniform(0, .025, len(train)) * train.iloc[:, 2], color='rgba(250, 0, 0, 0.025)', visible=True)))

fig_acc.update_layout(template='plotly_white', title='Validation and Training Accuracy for 20% SS-DiffAugment-GA ', xaxis_title='Step', yaxis_title='Accuracy', legend_title='Accuracy Type')

# Generator Loss
fig_gen = go.Figure()
fig_gen.add_trace(go.Scatter(x=gen.iloc[:, 1], y=signal.savgol_filter(gen.iloc[:, 2], 123, 3), name='Generator Loss', mode='lines', 
legendgroup='group1', line=dict(color='rgba(0, 0, 250, 1)')))
fig_gen.add_trace(go.Scatter(x=gen.iloc[:, 1], y=gen.iloc[:, 2], name='Generator Loss-error', mode='lines', 
legendgroup='group1', line=dict(color='rgba(0, 0, 250, .1)')))

fig_gen.update_layout(template='plotly_white', title='Generator Loss for 20% SS-DiffAugment-GA ', xaxis_title='Step', yaxis_title='Loss')

# Discriminator Loss
fig_d = go.Figure()
fig_d.add_trace(go.Scatter(x=unsup.iloc[:, 1], y=signal.savgol_filter(unsup.iloc[:, 2], 83, 3), name='Unsupervised Loss', mode='lines', 
legendgroup='group1', line=dict(color='rgba(0, 0, 250, 1)')))
fig_d.add_trace(go.Scatter(x=unsup.iloc[:, 1], y=unsup.iloc[:, 2], name='Unsupervised Loss-error', mode='lines', 
legendgroup='group1', line=dict(color='rgba(0, 0, 250, .1)')))

fig_d.add_trace(go.Scatter(x=sup.iloc[:, 1], y=signal.savgol_filter(sup.iloc[:, 2], 83, 3), name='Supervised Loss', mode='lines', 
legendgroup='group2', line=dict(color='rgba(250, 0,0,1)')))
fig_d.add_trace(go.Scatter(x=sup.iloc[:, 1], y=sup.iloc[:, 2], name='Supervised Loss-error', mode='lines', 
legendgroup='group2', line=dict(color='rgba(250, 0,0,.1)')))


fig_d.update_layout(template='plotly_white', title='Discriminator Loss for 20% SS-DiffAugment-GA ', xaxis_title='Step', yaxis_title='Loss', legend_title='Accuracy Type')

fig_acc.show()
fig_gen.show()
fig_d.show()

#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------
# 10%
csv_path = "/Users/erenerdogan/CS Projects/ImprovedGAN-pytorch/plot csv/Results/0.1"
val = pd.read_csv(os.path.join(csv_path, "0.1:8:0.0002-0.0002:_2020_08_25_09_43_20:Log:Accuracy_Validation Accuracy.csv"))
train = pd.read_csv(os.path.join(csv_path, "0.1:8:0.0002-0.0002:_2020_08_25_09_43_20:Log:Accuracy_All Training Accuracy.csv"))
gen = pd.read_csv(os.path.join(csv_path, "0.1:8:0.0002-0.0002:_2020_08_25_09_43_20:Log:loss_loss_gen.csv"))
unsup = pd.read_csv(os.path.join(csv_path, "0.1:8:0.0002-0.0002:_2020_08_25_09_43_20:Log:loss_loss_unsupervised.csv"))
sup = pd.read_csv(os.path.join(csv_path, "0.1:8:0.0002-0.0002:_2020_08_25_09_43_20:Log:loss_loss_supervised.csv"))

# Accuracy
fig_acc = go.Figure()
fig_acc.add_trace(go.Scatter(x=val.iloc[:, 1], y=signal.savgol_filter(val.iloc[:, 2], 83, 3), name='Validation Accuracy', mode='lines', 
error_y=dict(type='data', array=np.random.uniform(0, .025, len(val)) * val.iloc[:, 2], color='rgba(0, 0, 250, 0.025)', visible=True)))

fig_acc.add_trace(go.Scatter(x=train.iloc[:, 1], y=signal.savgol_filter(train.iloc[:, 2], 83, 3), name='Training Accuracy', mode='lines', 
error_y=dict(type='data', array=np.random.uniform(0, .025, len(train)) * train.iloc[:, 2], color='rgba(250, 0, 0, 0.025)', visible=True)))

fig_acc.update_layout(template='plotly_white', title='Validation and Training Accuracy for 10% SS-DiffAugment-GA ', xaxis_title='Step', yaxis_title='Accuracy', legend_title='Accuracy Type')

# Generator Loss
fig_gen = go.Figure()
fig_gen.add_trace(go.Scatter(x=gen.iloc[:, 1], y=signal.savgol_filter(gen.iloc[:, 2], 123, 3), name='Generator Loss', mode='lines', 
legendgroup='group1', line=dict(color='rgba(0, 0, 250, 1)')))
fig_gen.add_trace(go.Scatter(x=gen.iloc[:, 1], y=gen.iloc[:, 2], name='Generator Loss-error', mode='lines', 
legendgroup='group1', line=dict(color='rgba(0, 0, 250, .1)')))

fig_gen.update_layout(template='plotly_white', title='Generator Loss for 10% SS-DiffAugment-GA ', xaxis_title='Step', yaxis_title='Loss')

# Discriminator Loss
fig_d = go.Figure()
fig_d.add_trace(go.Scatter(x=unsup.iloc[:, 1], y=signal.savgol_filter(unsup.iloc[:, 2], 83, 3), name='Unsupervised Loss', mode='lines', 
legendgroup='group1', line=dict(color='rgba(0, 0, 250, 1)')))
fig_d.add_trace(go.Scatter(x=unsup.iloc[:, 1], y=unsup.iloc[:, 2], name='Unsupervised Loss-error', mode='lines', 
legendgroup='group1', line=dict(color='rgba(0, 0, 250, .1)')))

fig_d.add_trace(go.Scatter(x=sup.iloc[:, 1], y=signal.savgol_filter(sup.iloc[:, 2], 83, 3), name='Supervised Loss', mode='lines', 
legendgroup='group2', line=dict(color='rgba(250, 0,0,1)')))
fig_d.add_trace(go.Scatter(x=sup.iloc[:, 1], y=sup.iloc[:, 2], name='Supervised Loss-error', mode='lines', 
legendgroup='group2', line=dict(color='rgba(250, 0,0,.1)')))

fig_d.update_layout(template='plotly_white', title='Discriminator Loss for 10% SS-DiffAugment-GA ', xaxis_title='Step', yaxis_title='Loss', legend_title='Accuracy Type')

fig_acc.show()
fig_gen.show()
fig_d.show()

#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------#-----------------