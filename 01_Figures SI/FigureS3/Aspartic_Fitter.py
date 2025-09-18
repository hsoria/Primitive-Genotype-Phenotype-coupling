import pandas as pd


from scipy.integrate import odeint, solve_ivp, lsoda
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from lmfit import Parameters, minimize, Model, report_fit, conf_interval
from sklearn.metrics import mean_squared_error
import numdifftools
from PIL import Image
from sklearn.metrics import r2_score
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import scipy.optimize
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as tkr






def kinetic_plotting(z, t, params):
    F, Ac, An, W = z  
    k0, k1, a, k4 = params
    
    dAcdt = -k1*F*Ac + k4*An + ((k1*a*Ac*F)/(a+1))
    dFdt = -k1*F*Ac - k0*F
    dWdt = +k1*F*Ac + k0*F
    dAndt = +((k1*Ac*F)/(a+1)) - k4*An
    
    return [dFdt, dAcdt, dAndt, dWdt]


def ode_model(z, t, k0, k1, a, k4):
    
    """
    takes a vector of the initial concentrations that are previously defined. 
    YOu have to provide also initial guesses for the kinetic constants. 
    You must define as much constants as your system requires. Note that the reactions are expressed as differential
    equations
    
    F: Fuel
    Ac: Precuros
    An: Anhydride
    W: Waste
    
    Time is considered to be in **minutes**. Concentrations are in **mM**
    
    """
    F, Ac, An, W = z  
    
    dAcdt = -k1*F*Ac + k4*An + ((k1*a*Ac*F)/(a+1))
    dFdt = -k1*F*Ac - k0*F
    dWdt = k1*F*Ac + k0*F
    dAndt = ((k1*Ac*F)/(a+1)) - k4*An
    
    
    return [dFdt,dAcdt,dAndt,dWdt]

def ode_solver(t, initial_conditions, params):
    
    """
    Solves the ODE system given initial conditions for both initial concentrations and initial guesses for k

    
    """
        
    F, Ac, An, W = initial_conditions
    k0, k1, a, k4 = params['k0'].value, params['k1'].value, params['a'].value, params['k4'].value
    res = odeint(ode_model, [F, Ac, An, W], t, args=(k0, k1, a, k4))
    return res

def error(params, initial_conditions, tspan, data):
    sol = ode_solver(tspan, initial_conditions, params)
    sol_subset = sol[:, [0, 1, 2,]]
    data_subset = data[:, [0, 1, 2,]]
    return (sol_subset-data_subset)





def load_data_frame(excel_name, sheet_name):
    """
    Loads the excel file with data sortened in different sheets. Each sheet corresponds to different parameter change. 


    """
    a = pd.read_excel(excel_name, sheet_name= f"{sheet_name}", skiprows=(range(0,2)))
    a.columns = ["time", "F", "Ac", "An", "W", "Condition"] 

    a = a.sort_values(by = "time")

    return a



def load_initial_conditions(df, k0):
    
    
    
    """
    
    Provided a dataframe we extract the initial concentrations, the time window and provide initial guesses for the kinetic modelling. 
    Note that one can modifify the boundaries of the kinetic constants.
    Provide a df with the proper formatting
    
    """
    tspan = np.linspace(df["time"][0],float(df["time"].iloc[-1])+20,1000)
    F = df["F"][0] 
    Ac = df["Ac"][0]
    An = df["An"][0] 
    W = df["W"][0] 

    initial_conditions = [F, Ac, An, W]
    
    #initial guesses.
    
    k1 = 0.1      
    a = .1 #k2/k3
    k4 = .1  

    params = Parameters()
    params.add('k0', value=k0, vary = False)
    params.add('k1', value=k1, min=1e-6, max= 10)
    params.add('a', value=a, min=1e-6, max= 10)
    params.add('k4', value=k4, min=1e-6, max= 10)
    
    
    return initial_conditions, params, tspan

def sort_condition(df):

    list_df = []
    conditions = df["Condition"].unique()
    for condition in conditions:
        g = df[df["Condition"] == condition].sort_values(by = ["time", "Condition"]).reset_index(drop = True)
        list_df.append(g)

    return list_df, conditions
    

def get_fitted_curve(initial_conditions, tspan, params):
    """
    You provide the initial conditions, and the fitted values for the kinetic constants to simulate data.
    Is the "line" in the fitting curves.
    
    initial_conditions 
    tspan: time window simulation
    params: fitted parameters
    
    returns fitted data
    """
    
    y = pd.DataFrame(odeint(kinetic_plotting, initial_conditions, tspan, args=(params,)), columns = ['F', 'Ac', 'An','W'])
    y['min'] = tspan
    return y 




def fit_data(data_to_fit, fit_method = "least_squares", k0 = 1.3600e-04):

    initial_conditions, params, tspan = load_initial_conditions(data_to_fit, k0)

    data = data_to_fit[['F', 'Ac', 'An', 'W']].values
    t = data_to_fit['time']
    result = minimize(error, 
                        params, 
                        args=(initial_conditions, t, data), 
                        method=fit_method, nan_policy='omit')
        
    
    # Extract parameter values
    k_values = pd.DataFrame({"k_values": result.params.valuesdict().values()})
    
    simulated_data = get_fitted_curve(initial_conditions,
                                       tspan = tspan, 
                                       params = k_values.values.flatten())



    return k_values, data_to_fit, simulated_data, tspan






def fit_bootstrapping(data_to_fit, fit_method = "least_squares", k0 = 1.3600e-04, n_bt = 1000):
    
    """
    We sample our data n times doing fitting of each sampled data. We store the solutions for the k values. 
    In the end we obtain a distribution of kinetic constants. 
    One can also calcualate other parameters, for instance the half-life of the anhydride.
    
    
    n_iter: number of bootstrapping steps. 
    df: parent dataframe
    method: Minimization method
    error: Loss function
    initial_conditions, 
    params, 
    tspan
    
    """
    initial_conditions, params, tspan = load_initial_conditions(data_to_fit, k0)


    data = data_to_fit[['F', 'Ac', 'An', 'W']].values
    t = data_to_fit['time']


    n_iter = n_bt
    k1_bt = []
    a_bt = []
    k4_bt = []
    half_life = []

    
        
    for i in range(0, n_bt):
        
        """
        This is the most crucial sentence of the code. 
        It minimize out error function according to our initial values, guesses and time window. 
        There are several algorithms to minimize the function.
        
        Here's a list of those available. Note that some of them may require long computation period. 
        https://lmfit.github.io/lmfit-py/fitting.html
        
        """
        
        
        bt = data_to_fit.sample(n = len(data_to_fit), replace=True).reset_index(drop = True)
        bt = bt.sort_values(by = "time")
        data = bt[[ 'F', 'Ac', 'An','W']].values
        t = bt['time']
        result = minimize(error, 
                          params, 
                          args=(initial_conditions, t, data), 
                          method=fit_method, nan_policy='omit')
        
        k1_bt.append(result.params['k1'].value)
        a_bt.append(result.params['a'].value)
        k4_bt.append(result.params['k4'].value)
        half_life.append(np.log(2)/(result.params['k4'].value))

        res = pd.DataFrame({"k1":k1_bt,
                           "a":a_bt,
                           'k4':k4_bt,
                            "half-life":half_life
                           })
        params.update(result.params)


    return res



def get_rmse(data_to_fit, params_fitted):
    
    """
    
    Calculates the goodness of the fit through the R2. Which is a well-known parameter to decide whether a fit
    was done properly. Ranges between 0 and 1. Close to 1 equals to good fitting. 
    
    initial_conditions 
    df_real: original data
    params: fitted parameters
    
    returns fitted data
    
    """

    y_real = data_to_fit[['F', 'Ac', 'An']]
    t = data_to_fit['time']
    y_predict = pd.DataFrame(odeint(kinetic_plotting, 
                                    y_real.iloc[0].to_list() + [0], 
                                    t, 
                                    args = (params_fitted,)), 
                                     columns = ['F', 'Ac', 'An', "W"])
    
    rmse_F = np.sqrt(np.mean((y_predict["F"] - y_real["F"])**2, axis=0))
    rmse_Ac = np.sqrt(np.mean((y_predict["Ac"] - y_real["Ac"])**2, axis=0))
    rmse_An = np.sqrt(np.mean((y_predict["An"] - y_real["An"])**2, axis=0))
    
    rmse = [rmse_F, rmse_Ac, rmse_An]

    return rmse


def get_r2(data_to_fit, params_fitted):
    
    """
    
    Calculates the goodness of the fit through the R2. Which is a well-known parameter to decide whether a fit
    was done properly. Ranges between 0 and 1. Close to 1 equals to good fitting. 
    
    initial_conditions 
    df_real: original data
    params: fitted parameters
    
    returns fitted data
    
    """
    

    y_real = data_to_fit[['F', 'Ac', 'An']]
    t = data_to_fit['time']
    y_predict = pd.DataFrame(odeint(kinetic_plotting, 
                                    y_real.iloc[0].to_list() + [0], 
                                    t, 
                                    args = (params_fitted,)), 
                                    columns = ['F', 'Ac', 'An', "W"])
    
    r2_F = round(r2_score(y_real["F"], y_predict["F"]),4)
    r2_Ac = round(r2_score(y_real["Ac"], y_predict["Ac"]),4)
    r2_An = round(r2_score(y_real["An"], y_predict["An"]),4)
    
    r2 = [r2_F, r2_Ac, r2_An]

    return r2



def plot_fitted_error(df, y,rmse, tspan):
    
    
    """
    Creates a 4 column plots. In each column there is a different reagent. It plots both the original data and
    the fitted. Error is the 95 Confidence interval of the median. It is showed as filled area.
    
    df: data
    y: fitted
    y_max: upper limit
    y_min: low limit
    
    
    """

    fig, ax = plt.subplots(ncols = 3, figsize = (6.125,2.25), sharex=True, sharey=False, dpi = 600)

    sns.lineplot(data = df_fit, x = "min", y = "F", ax = ax[0], color ="C9", lw = 1, alpha = 0.68)
    sns.lineplot(data = df_fit, x = "min", y = "Ac", ax = ax[1], color ="C9", lw = 1, alpha = 0.68)
    sns.lineplot(data = df_fit, x = "min", y = "An", ax = ax[2], color ="C9", lw = 1, alpha = 0.68,)

    i = 20
    sns.scatterplot(data = df, x = "time", y = "F", ax = ax[0], color ="#B8336A", s = i, alpha = 0.8, edgecolor = "none",)
    sns.scatterplot(data = df, x = "time", y = "Ac", ax = ax[1], color ="C0", s = i, alpha = 0.8, edgecolor = "none",)
    sns.scatterplot(data = df, x = "time", y = "An", ax = ax[2], color = "C5", s = i, alpha = 0.8, edgecolor = "none",)


    ax[0].fill_between(tspan, y["F"]+(1.96*rmse[0]), y["F"]-(1.96*rmse[0]), alpha = 0.2, color = "#B8336A", lw = 0)
    ax[1].fill_between(tspan, y["Ac"]+(1.96*rmse[1]), y["Ac"]-(1.96*rmse[1]), alpha = 0.2, color = "C0", lw = 0)
    ax[2].fill_between(tspan, y["An"]+(1.96*rmse[2]), y["An"]-(1.96*rmse[2]), alpha = 0.2, color = "C5", lw = 0)

    ax[0].set(xlabel = "", ylabel = "", 
            xticks = np.linspace(0, 16, 3),xlim =  (-2, 18), 

            yticks = np.linspace(0, 60, 3), ylim = (-5, 62))


    ax[1].set(xlabel = "", ylabel = "", 
            xticks = np.linspace(0, 16, 3),xlim =  (-1, 18), 

            yticks = np.linspace(0, 20, 3), ylim = (-1.5, 21))

    ax[2].set(xlabel = "", ylabel = "", 
            xticks = np.linspace(0, 16, 3),xlim =  (-1, 18), 

            yticks = np.linspace(0, 20, 3), ylim = (-1.5, 21))


    sns.despine(fig, top = True, right = True)

    plt.tight_layout()





    return fig, ax

def plot_fitted(df, df_fit):

    fig, ax = plt.subplots(ncols = 3, figsize = (6.125,2.25), sharex=True, sharey=False, dpi = 600)

    sns.lineplot(data = df_fit, x = "min", y = "F", ax = ax[0], color ="C9", lw = 1, alpha = 0.68)
    sns.lineplot(data = df_fit, x = "min", y = "Ac", ax = ax[1], color ="C9", lw = 1, alpha = 0.68)
    sns.lineplot(data = df_fit, x = "min", y = "An", ax = ax[2], color ="C9", lw = 1, alpha = 0.68,)

    i = 20
    sns.scatterplot(data = df, x = "time", y = "F", ax = ax[0], color ="#B8336A", s = i, alpha = 0.8, edgecolor = "none",)
    sns.scatterplot(data = df, x = "time", y = "Ac", ax = ax[1], color ="C0", s = i, alpha = 0.8, edgecolor = "none",)
    sns.scatterplot(data = df, x = "time", y = "An", ax = ax[2], color = "C5", s = i, alpha = 0.8, edgecolor = "none",)


    ax[0].set(xlabel = "", ylabel = "", 
            xticks = np.linspace(0, 16, 3),xlim =  (-2, 18), 

            yticks = np.linspace(0, 60, 3), ylim = (-5, 62))


    ax[1].set(xlabel = "", ylabel = "", 
            xticks = np.linspace(0, 16, 3),xlim =  (-1, 18), 

            yticks = np.linspace(0, 20, 3), ylim = (-1.5, 21))

    ax[2].set(xlabel = "", ylabel = "", 
            xticks = np.linspace(0, 16, 3),xlim =  (-1, 18), 

            yticks = np.linspace(0, 20, 3), ylim = (-1.5, 21))


    sns.despine(fig, top = True, right = True)

    plt.tight_layout()
    return fig, ax