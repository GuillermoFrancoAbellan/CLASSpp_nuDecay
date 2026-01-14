"""
This plotting script can be used to test a models performance on the
stored test data. The syntax is 'python test_model.py <path to model>',
and up to three models can be given. Only the first model will be used
to produce cmb spectra, but all models given will be included in the
error plot. The error plot will be saved within the first model specified
as '<path to model>/plots/error.pdf'.

Author: Andreas Nygaard (2022)

"""

import os
import sys
import warnings
import pickle as pkl

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib

import PlanckLogLinearScale

# index of test data to use for cmb spectra
n=0

# color of axes and all text in the error plot
color_of_axis_and_text = 'k'

# use latex?
latex = True


################################################################

name = sys.argv[1]
model = tf.keras.models.load_model(name, compile=False)
with open(name+'/test_data.pkl', 'rb') as f:
    test_data = pkl.load(f)


try:
    model_params = np.array(test_data[0])
    output_data     = np.array(test_data[1])
except:
    test_data = tuple(zip(*test_data))
    model_params = np.array(test_data[0])
    output_data     = np.array(test_data[1])

try:    
    with open(name+'/output_info.pkl', 'rb') as f:
        output_info = pkl.load(f)
    pickle_file = True
    warnings.warn("You are using CONNECT models from an old version (before v23.6.0). Support for this is deprecated and will be removed in a later update.")
    
except:
    pickle_file = False
    output_info = eval(model.get_raw_info().numpy().decode('utf-8'))

print(output_info)

if pickle_file:
    try:
        if output_info['normalize']['method'] == 'standardization':
            normalize = 'standardization'
        elif output_info['normalize']['method'] == 'log':
            normalize = 'log'
        elif output_info['normalize']['method'] == 'min-max':
            normalize = 'min-max'
        elif output_info['normalize']['method'] == 'factor':
            normalize = 'factor'
        else:
            normalize = 'factor'
    except:
        normalize = 'standardization'

output_predict  = model.predict(model_params, verbose=0)

    
if pickle_file and normalize == 'standardization':
    mean = output_info['normalize']['mean']
    var  = output_info['normalize']['variance']
    output_predict = output_predict * np.sqrt(var) + mean
    output_data = output_data * np.sqrt(var) + mean
elif pickle_file and normalize == 'min-max':
    x_min = np.array(output_info['normalize']['x_min'])
    x_max = np.array(output_info['normalize']['x_max'])
    output_predict = output_predict * (x_max - x_min) + x_min
    output_data = output_data * (x_max - x_min) + x_min


out_predict = {}
out_data    = {}
for output in output_info['output_Cl']:
    lim0 = output_info['interval']['Cl'][output][0]
    lim1 = output_info['interval']['Cl'][output][1]
    out_data[output]    = output_data[n][lim0:lim1]
    out_predict[output] = output_predict[n][lim0:lim1]
    if pickle_file and normalize == 'log':
        for offset in list(reversed(output_info['normalize']['Cl'][output])):
            out_predict[output]=np.exp(out_predict[output]) - offset
            out_data[output]=np.exp(out_data[output]) - offset


if 'output_derived' in output_info.keys():
    for output in output_info['output_derived']:
        if output != '100*theta_s':
            idx = output_info['interval']['derived'][output]
            out_data[output]    = output_data[n][idx]
            out_predict[output] = output_predict[n][idx]
            if pickle_file and normalize == 'log':
                for offset in list(reversed(output_info['normalize']['derived']['100*theta_s'])):
                    out_predict[output]=np.exp(out_predict[output]) - offset
                    out_data[output]=np.exp(out_data[output]) - offset

for output in output_info['output_Cl']:
    if pickle_file and normalize == 'factor':
        normalize_factor = output_info['normalize']['Cl'][output]
    plt.figure(figsize=(12,10))
    ell        = output_info['ell']
    ll         = np.linspace(2,max(ell)+7,int(max(ell)-1+7))
    Cl_predict = out_predict[output]
    Cl_data    = out_data[output]
    Cl_pre_sp  = CubicSpline(ell,Cl_predict, bc_type = 'natural', extrapolate=True)
    Cl_dat_sp  = CubicSpline(ell,Cl_data,  bc_type = 'natural', extrapolate=True)
    if pickle_file and normalize == 'factor':
        plt.plot(ll, Cl_dat_sp(ll)/normalize_factor,'k-',lw=3,label='CLASS')
    else:
        plt.plot(ll, Cl_dat_sp(ll),'k-',lw=3,label='CLASS')
    if pickle_file and normalize == 'factor':
        plt.plot(ll, Cl_pre_sp(ll)/normalize_factor,'r-',lw=3,label='CONNECT')
    else:
        plt.plot(ll, Cl_pre_sp(ll),'r-',lw=3,label='CONNECT')
    plt.legend()
    plt.xscale('log')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_{\ell}\times\ell(\ell+1)/2\pi$')
    plt.title(output)
#    plt.savefig(name+f'/plots/Cl_{output.upper()}.pdf')

if 'output_derived' in output_info.keys(): 
    for output in output_info['output_derived']:
        if output != '100*theta_s':
            if pickle_file and normalize == 'factor':
                normalize_factor = output_info['normalize']['derived'][output]
            print(output)
            if pickle_file and normalize == 'factor':
                print('CLASS:',out_data[output]/normalize_factor)
            else:
                print('CLASS:',out_data[output])
            if pickle_file and normalize == 'factor':
                print('CONNECT:',out_predict[output]/normalize_factor)
            else:
                print('CONNECT:',out_predict[output])


def rms(x):
    return np.sqrt(np.mean(x**2))


    
l = np.linspace(2,2500,2499)


def get_error_Cl(path,spectrum):
    model = tf.keras.models.load_model(path, compile=False)

    with open(path + '/test_data.pkl', 'rb') as f:
        test_data = pkl.load(f)
    try:
        with open(path + '/output_info.pkl', 'rb') as f:
            output_info = pkl.load(f)
        pickle_file = True
    except:
        pickle_file = False
        output_info = eval(model.get_raw_info().numpy().decode('utf-8'))

    ell = output_info['ell']
    if pickle_file:
        try:
            if output_info['normalize']['method'] == 'standardization':
                normalize = 'standardization'
            elif output_info['normalize']['method'] == 'log':
                normalize = 'log'
            elif output_info['normalize']['method'] == 'min-max':
                normalize = 'min-max'
            elif output_info['normalize']['method'] == 'factor':
                normalize = 'factor'
            else:
                normalize = 'factor'
        except:
            normalize = 'standardization'
            
    try:
        model_params = test_data[0]
        Cls_data     = test_data[1]
    except:
        test_data = tuple(zip(*test_data))
        model_params = np.array(test_data[0])
        Cls_data     = np.array(test_data[1])

    v = tf.constant(model_params)
    Cls_predict = model(v).numpy()

    if pickle_file and normalize == 'standardization':
        mean = output_info['normalize']['mean']
        var  = output_info['normalize']['variance']
        Cls_predict = Cls_predict * np.sqrt(var) + mean
        Cls_data = Cls_data * np.sqrt(var) + mean
    elif pickle_file and normalize == 'min-max':
        x_min = np.array(output_info['normalize']['x_min'])
        x_max = np.array(output_info['normalize']['x_max'])
        Cls_predict = Cls_predict * (x_max - x_min) + x_min
        Cls_data = Cls_data * (x_max - x_min) + x_min
    
    lim0 = output_info['interval']['Cl'][spectrum][0]
    lim1 = output_info['interval']['Cl'][spectrum][1]

    errors = []
    for j, (cls_d, cls_p) in enumerate(zip(Cls_data, Cls_predict)):
        if pickle_file and normalize == 'factor':
            err = ((np.array(cls_d[lim0:lim1])-np.array(cls_p[lim0:lim1]))/output_info['normalize']['Cl'][spectrum])/rms(np.array(cls_d[lim0:lim1]))
        elif pickle_file and normalize == 'log':
            clsp = cls_p[lim0:lim1]
            clsd = cls_d[lim0:lim1]
            for offset in list(reversed(output_info['normalize']['Cl'][spectrum])):
                clsp=np.exp(clsp) - offset
                clsd=np.exp(clsd) - offset
            err = (np.array(clsd)-np.array(clsp))/rms(np.array(clsd))
        else:
            err = (np.array(cls_d[lim0:lim1])-np.array(cls_p[lim0:lim1]))/rms(np.array(cls_d[lim0:lim1]))
        errors.append(abs(err))

    errors = np.array(errors).T
    return errors, ell

zz = np.linspace(0.26,2.37,100)

def get_error_background(path, bg_quantity):
    model = tf.keras.models.load_model(path, compile=False)

    with open(path + '/test_data.pkl', 'rb') as f:
        test_data = pkl.load(f)
    try:
        with open(path + '/output_info.pkl', 'rb') as f:
            output_info = pkl.load(f)
        pickle_file = True
    except:
        pickle_file = False
        output_info = eval(model.get_raw_info().numpy().decode('utf-8'))

    z_bg = output_info['z_bg']
    if pickle_file:
        try:
            if output_info['normalize']['method'] == 'standardization':
                normalize = 'standardization'
            elif output_info['normalize']['method'] == 'log':
                normalize = 'log'
            elif output_info['normalize']['method'] == 'min-max':
                normalize = 'min-max'
            elif output_info['normalize']['method'] == 'factor':
                normalize = 'factor'
            else:
                normalize = 'factor'
        except:
            normalize = 'standardization'
            
    try:
        model_params = test_data[0]
        Bg_data     = test_data[1]
    except:
        test_data = tuple(zip(*test_data))
        model_params = np.array(test_data[0])
        Bg_data     = np.array(test_data[1])

    v = tf.constant(model_params)
    Bg_predict = model(v).numpy()

    if pickle_file and normalize == 'standardization':
        mean = output_info['normalize']['mean']
        var  = output_info['normalize']['variance']
        Bg_predict = Bg_predict * np.sqrt(var) + mean
        Bg_data = Bg_data * np.sqrt(var) + mean
    elif pickle_file and normalize == 'min-max':
        x_min = np.array(output_info['normalize']['x_min'])
        x_max = np.array(output_info['normalize']['x_max'])
        Bg_predict = Bg_predict * (x_max - x_min) + x_min
        Bg_data = Bg_data * (x_max - x_min) + x_min
    
    lim0 = output_info['interval']['bg'][bg_quantity][0]
    lim1 = output_info['interval']['bg'][bg_quantity][1]

    errors = []
    for j, (bg_d, bg_p) in enumerate(zip(Bg_data, Bg_predict)):
        if pickle_file and normalize == 'factor':
            err = ((np.array(bg_d[lim0:lim1])-np.array(bg_p[lim0:lim1]))/output_info['normalize']['bg'][bg_quantity])/rms(np.array(bg_d[lim0:lim1]))
        elif pickle_file and normalize == 'log':
            bgp = bg_p[lim0:lim1]
            bgd = bg_d[lim0:lim1]
            for offset in list(reversed(output_info['normalize']['bg'][bg_quantity])):
                bgp=np.exp(bgp) - offset
                bgd=np.exp(bgd) - offset
            err = (np.array(bgd)-np.array(bgp))/rms(np.array(bgd))
        else:
            err = (np.array(bg_d[lim0:lim1])-np.array(bg_p[lim0:lim1]))/rms(np.array(bg_d[lim0:lim1]))
        errors.append(abs(err))

    errors = np.array(errors).T
    return errors, z_bg


height = 6.0
width = 6.0

fontsize = 11/1.2*1.5

if latex:
    latex_preamble = [
        r'\usepackage{lmodern}',
        r'\usepackage{amsmath}',
        r'\usepackage{amsfonts}',
        r'\usepackage{amssymb}',
        r'\usepackage{mathtools}',
    ]
    matplotlib.rcParams.update({
        'text.usetex'        : True,
        'font.family'        : 'serif',
        'font.serif'         : 'cmr10',
        'font.size'          : fontsize,
        'mathtext.fontset'   : 'cm',
        'text.latex.preamble': latex_preamble,
    })


model_paths = []
model_names = []
for arg in sys.argv[1:]:
    model_paths.append(arg)
    model_names.append(arg.split('/')[-1])

#percentiles = [0.682,0.954]
percentiles = [0.954]

alpha_list  = [0.2,    0.1,   0.05]
if len(model_names) == 3:
    c_list      = ['royalblue','seagreen','crimson']
elif len(model_names) == 2:
    c_list      = ['red','blue','green']
else:
    c_list      = ['blue','red','green']
fc_array    = [['cyan',   'blue', 'navy'],
               ['orange', 'red',  'crimson'],
               ['lightgreen','forestgreen','darkgreen']]



change=200

PlanckLogLinearScale.new_change(change)

fig, axs = plt.subplots(2,2,figsize=(width, height))
fig.subplots_adjust(wspace=0,hspace=0)

for k, spectrum in enumerate(output_info['output_Cl']):
    y_max = 0
    y_min = 0
    id_x, id_y = divmod(k, 2)
    for i, path in enumerate(model_paths): 
        model_name = model_names[i]
        errors, l_red =  get_error_Cl(path, spectrum)
        max_error=[]
        err_lower_array = []
        err_upper_array = []
        for errs in errors:
            err_lower_list = []
            err_upper_list = []
            for p in percentiles:
                err_upper_list.append(np.percentile(errs, 100*p))

            err_upper_array.append(err_upper_list)
            max_error.append(max(errs))
        if max(np.array(err_upper_array).flatten()) > y_max:
            y_max = max(np.array(err_upper_array).flatten())
            
        for j, p in reversed(list(enumerate(sorted(percentiles)))):
            err_u = CubicSpline(l_red,np.array(err_upper_array).T[j])
            
            axs[id_x, id_y].axvline(x=change, linestyle="-", color="lightgrey", zorder=-3)
            if j==len(percentiles)-1 and k==0:
                axs[id_x,id_y].plot(l,abs(err_u(l)),
                              c=c_list[i],lw=1, zorder=i, alpha=list(reversed(percentiles))[j])
            else:
                axs[id_x,id_y].plot(l,abs(err_u(l)),
                              c=c_list[i],lw=1, zorder=i, alpha=list(reversed(percentiles))[j])

    if k==3:
        custom_lines=[matplotlib.lines.Line2D([],[],c=c_list[0]),
                      matplotlib.lines.Line2D([],[],c=c_list[1]),
                      matplotlib.lines.Line2D([],[],c=c_list[2])]
        if len(model_names) == 2:
            custom_lines = custom_lines[1:]
        elif len(model_names) == 1:
            custom_lines = custom_lines[2:]
        model_names = [r'$\nu_{1,2,3} \rightarrow \nu_4 + \phi$',r'$\nu_{3} \rightarrow \nu_{1,2} + \phi$',r'$\nu_{1,2} \rightarrow \nu_3 + \phi$']
        axs[id_x,id_y].legend(custom_lines,model_names,bbox_to_anchor=(0.35,0.6),fontsize=fontsize/1.5)

    axs[id_x,id_y].set_yscale('log')
    axs[id_x,id_y].set_xscale('planck')

    axs[id_x,id_y].set_xlim([2,2500])
    axs[1,1].set_xlabel(r'$\ell$')

    if k==2:
        axs[id_x,id_y].set_xlabel(r'$\ell$')
    if k == 0:
        if latex:
            axs[id_x,id_y].set_ylabel(r'$\dfrac{\left\vert \mathcal{D}_{\ell}^{\textsc{connect}}-\mathcal{D}_{\ell}^{\textsc{class}}\right\vert}{{\rm rms}\left(\mathcal{D}_{\ell}^{\textsc{class}}\right)}$',labelpad=10)
            axs[id_x,id_y].yaxis.set_label_coords(-0.23, 0.0)
        else:
            axs[id_x,id_y].set_ylabel(r'$|D_{\ell}^{\rm connect}-D_{\ell}^{\rm class}| /{\rm rms}(D_{\ell}^{\rm class})$',labelpad=10)
            axs[id_x,id_y].yaxis.set_label_coords(-0.23, 0.0)
    if (k==1) or (k==3):
        axs[id_x,id_y].yaxis.set_ticklabels([])
    axs[id_x,id_y].set_ylim([5e-6,5e0])
    if spectrum == 'pp':
    	axs[id_x,id_y].set_title(r'$\phi\phi$', color=color_of_axis_and_text,y=0.85,fontsize=fontsize)
    else:
    	axs[id_x,id_y].set_title(f'{spectrum.upper()}', color=color_of_axis_and_text,y=0.85,fontsize=fontsize)
    axs[id_x,id_y].tick_params(axis='both', which='both', direction='in')
    xticks = [1e+1, 1e+2, 1e+3, 2e+3]
    xticks_minor = [2,3,4,5,6,7,8,9,20,30,40,50,60,70,80,90,200,300,400,500,600,700,800,900,1100,1200,1300,1400,1500,1600,1700,1800,1900,2100,2200,2300,2400,2500] 
    yticklabels=[r'$10^{0}$',r'$10^{-1}$',r'$10^{-2}$',r'$10^{-3}$',r'$10^{-4}$',r'$10^{-5}$']
    xticklabels=[r'$10^{1}$',r'$10^{2}$','1000','2000']
    axs[id_x,id_y].set_xticks(xticks)
    axs[id_x,id_y].set_xticks(xticks_minor, minor=True)
    axs[id_x,id_y].xaxis.set_ticklabels(xticklabels, color=color_of_axis_and_text)
    axs[id_x,id_y].set_yticks([1e0,1e-1,1e-2,1e-3,1e-4,1e-5])
    if k==0:
        axs[id_x,id_y].yaxis.set_ticklabels(yticklabels, color=color_of_axis_and_text)

    
if not os.path.isdir(model_paths[0]+'/plots'):
    os.mkdir(model_paths[0]+'/plots')

plt.savefig(model_paths[0]+f'/plots/error_Cl.pdf', facecolor=fig.get_facecolor(),bbox_inches='tight')

plt.show()




height = 6.0
width = 3.0
fontsize = 11/1.2*1.5

fig, axs = plt.subplots(2, 1, figsize=(width, height))

fig.subplots_adjust(wspace=0,hspace=0)

for k, bg_quantity in enumerate([s for s in output_info['output_bg'] if s != 'conf. time [Mpc]']):
    y_max = 0
    y_min = 0
    for i, path in enumerate(model_paths): 
        model_name = model_names[i]
        errors, z_red =  get_error_background(path, bg_quantity)
        max_error=[]
        err_lower_array = []
        err_upper_array = []
        for errs in errors:
            err_lower_list = []
            err_upper_list = []
            for p in percentiles:
                err_upper_list.append(np.percentile(errs, 100*p))

            err_upper_array.append(err_upper_list)
            max_error.append(max(errs))
        if max(np.array(err_upper_array).flatten()) > y_max:
            y_max = max(np.array(err_upper_array).flatten())

        for j, p in reversed(list(enumerate(sorted(percentiles)))):
            err_u = CubicSpline(z_red,np.array(err_upper_array).T[j])

            if j==len(percentiles)-1 and k==0:
                axs[k].plot(zz,abs(err_u(zz)),c=c_list[i],lw=1, zorder=i, alpha=list(reversed(percentiles))[j])
                axs[k].scatter(z_red,abs(np.array(err_upper_array).T[j]),marker = '*',c=c_list[i],lw=1, zorder=i, alpha=list(reversed(percentiles))[j])
            else:
                axs[k].plot(zz,abs(err_u(zz)),c=c_list[i],lw=1, zorder=i, alpha=list(reversed(percentiles))[j])
                axs[k].scatter(z_red,abs(np.array(err_upper_array).T[j]),marker = '*',c=c_list[i],lw=1, zorder=i, alpha=list(reversed(percentiles))[j])


    if k==0:
        axs[k].legend(custom_lines,model_names,bbox_to_anchor=(0.81,0.8),fontsize=fontsize/1.5)

    axs[k].set_yscale('log')
    axs[k].set_xscale('linear')

    axs[k].set_xlim([0.26,2.37])
#    axs[0].set_xlabel(r'$z$')

    if k==1:
        axs[k].set_xlabel(r'$z$')
    if k == 0:
        if latex:
            axs[k].set_ylabel(r'$\dfrac{\left\vert D_A^{\textsc{connect}}-D_A^{\textsc{class}}\right\vert}{{\rm rms}\left(D_A^{\textsc{class}}\right)}$',labelpad=10)
        else:
            axs[k].set_ylabel(r'$|D_A^{\rm connect}-D_A^{\rm class}| /{\rm rms}(D_A^{\rm class})$',labelpad=10)
    if k == 1:
        if latex:
            axs[k].set_ylabel(r'$\dfrac{\left\vert H^{\textsc{connect}}-H^{\textsc{class}}\right\vert}{{\rm rms}\left(H^{\textsc{class}}\right)}$',labelpad=10)
        else:
            axs[k].set_ylabel(r'$|H^{\rm connect}-H^{\rm class}| /{\rm rms}(H^{\rm class})$',labelpad=10)
    axs[k].set_ylim([5e-6,5e0])
#    axs[k].set_title(f'{bg_quantity}', color=color_of_axis_and_text)
    axs[k].tick_params(axis='both', which='both', direction='in')
    yticklabels=[r'$10^{0}$',r'$10^{-1}$',r'$10^{-2}$',r'$10^{-3}$',r'$10^{-4}$',r'$10^{-5}$']
    axs[k].set_xticks([0.5, 1.0, 1.5, 2.0])
    axs[k].set_yticks([1e0,1e-1,1e-2,1e-3,1e-4,1e-5])
    if k==0:
        axs[k].yaxis.set_ticklabels(yticklabels, color=color_of_axis_and_text)

plt.savefig(model_paths[0]+f'/plots/error_bg.pdf', facecolor=fig.get_facecolor(),bbox_inches='tight')

plt.show()
