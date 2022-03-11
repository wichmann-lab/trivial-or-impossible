# Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
import torch
import os
import re
import pandas as pd
import warnings
#import krippendorff

# Set params
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Seaborn settings
sns.set_palette(sns.color_palette("magma"))
sns.set_style("ticks")
sns.set_style({'font.family': 'Lato'})
sns.set_context("paper", font_scale = 1.8)


# Natural sorting algorithm, taken from
# https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


# Function that loads datafiles for defined condition
def load_data(folder_path):
    """
    :param condition: Condition name
    :return: Results array
    """

    # Set path in which result files of conditions are found
    results = []
    val_acc = []
    model_list = os.listdir(folder_path)

    # Go through model list in result file path
    for model in sorted(model_list):

        # New list for each model for better list structure
        model_results = []

        # File list for every file of model
        file_list = os.listdir(folder_path + model + "/")

        # Loop through files for this model
        for filename in natural_sort(file_list):

            # Only append to results if it is a .txt file
            if filename.startswith("RESULTS"):
                filepath = folder_path + model + "/" + filename
                model_results.append(np.array(torch.load(filepath, map_location=torch.device('cpu')), dtype=object))

            if filename.startswith("VAL_ACC"):
                filepath = folder_path + model + "/" + filename
                val_acc.append(np.array(torch.load(filepath, map_location=torch.device('cpu'))))

        # Append list to overall result list
        results.append(model_results)

    return results, np.array(val_acc)


# Plot development of accuracy for all models
def base_plot(array, metric, title, lim):
    """
    :param array: Array to display
    :param metric: Which metric is displayed?
    :param title: Title of output figure
    :param lim: What are the y-axis limits
    :return: Plot figure
    """

    plt.figure(figsize=(8, 4))
    ax = sns.lineplot(data=array, legend=False);
    plt.setp(ax.lines, alpha=.3)
    ax.set(xlabel="Epoch", ylabel="{}".format(metric), title="{}".format(title))
    sns.lineplot(data=np.mean(array, axis=1))

    plt.ylim(lim)
    plt.tight_layout()

# Calculate error consistency to base network
def calc_econ(results_1, results_2, model_1, model_2, epoch_1, epoch_2, errs, correct):
    """
    :param results_1: Result file 1
    :param results_2: Result file 2
    :param model_1: Model index for result file 1
    :param model_2: Model index for result file 2
    :param epoch_1: Epoch index for result file 1
    :param epoch_2: Epoch index for result file 2
    :return:
    """

    # Calculate error consistency to base model
    ep_1 = np.equal(np.array(results_1[model_1][epoch_1][1]), np.array(results_1[model_1][epoch_1][2]))
    ep_2 = np.equal(np.array(results_2[model_2][epoch_2][1]), np.array(results_2[model_2][epoch_2][2]))
        
    if correct == True:
        print("Deleting falsely labeled items")
        ep_1 = np.delete(ep_1, errs)
        ep_2 = np.delete(ep_2, errs)
        
    # Calculate kappa
    econ = cohen_kappa_score(ep_1, ep_2)

    return econ


# Calculate error consistency to base network
def calc_krip(results_1, results_2, model_1, model_2, epoch_1, epoch_2):
    """
    :param results_1: Result file 1
    :param results_2: Result file 2
    :param model_1: Model index for result file 1
    :param model_2: Model index for result file 2
    :param epoch_1: Epoch index for result file 1
    :param epoch_2: Epoch index for result file 2
    :return:
    """

    # Calculate error consistency to base model
    ep_1 = np.array(results_1[model_1][epoch_1][1])
    ep_2 = np.array(results_2[model_2][epoch_2][1])

    # Calculate kappa
    krip = cohen_kappa_score(ep_1, ep_2)

    return krip

def order_econs(econs):
    """
    :param econs: Array including condition name and error consistency for all models of conditions in last epoch
    :return: Ordered econs array and ordered mean over models of each condition
    """

    # Loop through condition and calculate mean for each condition
    num_conditions = econs.shape[0]
    econ_means = np.zeros(num_conditions)
    for ind in range(num_conditions):
        econ_means[ind] = np.mean(econs[ind, 1])

    # Order conditions by highest mean
    order_high_low = np.flip(np.argsort(econ_means))
    ordered_data = econs[order_high_low, :]
    ordered_means = econ_means[order_high_low]

    return ordered_data, ordered_means


def three_plots(val_acc, error_con_subseq, error_con_tobase, condition, figure_path, base_network):
    """
    :param val_acc:
    :param error_con_subseq:
    :param error_con_tobase:
    :param condition:
    :return:
    """

    # Reformat condition name
    title = condition.replace(base_network, "").replace("_", " ")

    # Plot results
    base_plot(val_acc.T, "Test accuracy",
              "{}: \nTest accuracy over epochs".format(title), lim=(0, 1))
    plt.savefig(figure_path + '{}_accuracy.png'.format(condition), bbox_inches='tight')
    plt.close()

    base_plot(error_con_subseq.T, "Error consistency",
              "{}: \nError consistency compared to following epoch".format(title), lim=(0, 1))
    plt.savefig(figure_path + '{}_econ_subseq.png'.format(condition), bbox_inches='tight')
    plt.close()

    base_plot(error_con_tobase.T, "Error consistency",
              "{}: \nError consistency compared to base network".format(title), lim=(0, 1))
    plt.savefig(figure_path + '{}_econ_tobase.png'.format(condition), bbox_inches='tight')
    plt.close()

    return


# Function to prepare everything for this condition
def prep_condition(condition, folder_name, base_network):
    """
    :param condition: Which condition are we in
    :param folder_name: Name of data folder
    :param base_network: Name of base network
    :return:
    """

    # Load data, conditions with more epochs than base are stored in base file
    if condition.endswith("Plus_1ep") or condition.endswith("Plus_10ep"):
        results, val_acc = load_data(folder_name + "{}_Base_condition/".format(base_network))
    elif condition.endswith("Different_data"):
        results, val_acc = load_data(folder_name + "{}_Half_data/".format(base_network))
    else:
        results, val_acc = load_data(folder_name + condition + "/")

    # Set number of epochs depending on model family
    if base_network == "Res18":
        num_epochs = 90
        num_base_epochs = 90
    elif base_network == "Dense121":
        num_epochs = 30
        num_base_epochs = 30
    elif base_network == "VGG11":
        num_epochs = 30
        num_base_epochs = 30
    elif base_network == "Res18fc100":
        num_epochs = 30
        num_base_epochs = 30
    elif base_network == "Res18CIFAR":
        num_epochs = 30
        num_base_epochs = 30
    elif base_network == "Res18FRACTAL":
        num_epochs = 30
        num_base_epochs = 30
    else:
        raise Exception("No correct model family defined! Run script using: sbatch ./run_analysis MODELFAMILY")

    # Set number of models depending on condition
    if condition.endswith("Base_condition") or condition.endswith("Different_batchsize") or condition.endswith("Different_optimizer") or condition.endswith("Different_architecture") or condition.endswith("Plus_1ep") or condition.endswith("Plus_10ep") or condition.endswith("Base_comparison") or condition.endswith("ASGD") or condition.endswith("Rprop") or condition.endswith("Adam") or condition.endswith("Adagrad") or condition.endswith("RMSprop") or condition.endswith("SGD_Low_LR"):
        num_models = 1
    elif condition.endswith("Half_data") or condition.endswith("Different_data"):
        num_models = 2
    else:
        num_models = 5

    # For plus epoch conditions, set number of epochs higher
    if condition.endswith("Plus_1ep"):
        num_epochs += 1
    elif condition.endswith("Plus_10ep"):
        num_epochs += 10
    elif condition.endswith("Base_comparison"):
        num_epochs = 90

    # Add plus 1 to num_epochs and num_base_epochs because model is given test set once before training
    num_epochs += 1
    num_base_epochs += 1

    return results, val_acc, num_epochs, num_models, num_base_epochs


def heat_plot(heat_array, conditions, figure_path, base_network, epoch):
    """
    :param heat_array: Array for heat map
    :param conditions: Names of conditions
    :param figure_path: Path for figure to be saved
    :param base_network: Name of base network
    :return:
    """

    # Format conditions
    plt.figure(figsize=(12, 12))
    cond_clear = [cond.replace(base_network, "").replace("_", " ") for cond in conditions]

    # Make and save heatmap
    sns.heatmap(heat_array, annot=True, annot_kws={"fontsize":8}, square=True,
                xticklabels=cond_clear, yticklabels=cond_clear, cmap="Blues", cbar_kws={"shrink": .60})
    plt.title(label=f'Error consistency compared to base model for Epoch: {epoch}', fontsize=26, y=1.1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(figure_path + f'{base_network}_heatmap_ep{epoch}.png')

    return


def plot_decisions(decisions, overlay, num_epochs, figure_path, condition, correct):
    """
    :param decisions: Decisions array of shape: (size validation set x epochs)
    :return: Plot matrix
    """
    print(f"Condition: {condition}, array shape: {decisions.shape}")
       
    if condition.endswith("superimp"):
        dpi = 150
        col_map = "Reds_r"
    else:
        dpi = 300
        col_map = "Blues_r"
           
    plt.figure(figsize=(24, 24))
    sns.heatmap(decisions, cmap=col_map, cbar=False, yticklabels=False)
    sns.heatmap(overlay, cmap="copper", cbar=False, yticklabels=False)
    sns.despine(top=True, right=True, left=True, offset=10, trim=True)
    
    plt.xticks(np.arange(0, decisions.shape[1], 5), labels=np.arange(0, decisions.shape[1], 5), fontsize=30)

    plt.xlabel("Epoch", fontsize=40, labelpad=10)
    plt.ylabel("Image from validation set", fontsize=40, labelpad=10)
    plt.tight_layout()
    
    if correct:
        plt.savefig(figure_path + f'decisions/{condition}_decisions_full_corrected.png', dpi=dpi, bbox_inches='tight')
    else:
        plt.savefig(figure_path + f'decisions/{condition}_decisions_full.png', dpi=dpi, bbox_inches='tight')
    plt.close()

#    plt.figure(figsize=(24, 24))
#    sns.heatmap(decisions[-1000:, :], cmap="binary_r", cbar=False, xticklabels=np.arange(0, num_epochs, 1), yticklabels=False)
#    sns.despine(top=True, right=True, left=True, offset=10, trim=True)
#    plt.title(label="Decisions on ImageNet validation set over epochs: last 1000 images", fontsize=24, fontweight="bold", pad=10)
#    plt.xticks(fontsize=10, rotation=30)
#    plt.xlabel("Epoch")
#    plt.ylabel("Image from ImageNet validation set")
#    plt.tight_layout()
#    plt.savefig(figure_path + f'decisions/{condition}_decisions_1000_model{model}.png', dpi=1000, bbox_inches='tight')
#    plt.close()
#
#    plt.figure(figsize=(24, 24))
#    sns.heatmap(decisions[-100:, :], cmap="binary_r", cbar=False, xticklabels=np.arange(0, num_epochs, 1), yticklabels=False)
#    sns.despine(top=True, right=True, left=True, offset=10, trim=True)
#    plt.title(label="Decisions on ImageNet validation set over epochs: last 100 images", fontsize=24, fontweight="bold", pad=10)
#    plt.xticks(fontsize=10, rotation=30)
#    plt.xlabel("Epoch")
#    plt.ylabel("Image from ImageNet validation set")
#    plt.tight_layout()
#    plt.savefig(figure_path + f'decisions/{condition}_decisions_100_model{model}.png', bbox_inches='tight')
#    plt.close()

    return


def plot_decisions_change(decisions, figure_path, base_network, conditions, num_items):
    """
    :param decisions: Decisions array of shape: (epochs)
    :return: Plot
    """
    sns.set_context("paper", font_scale=5.0)
    sns.set_style("white")
    
    plt.figure(figsize=(24, 24))
    for ind, cond in enumerate(conditions):
        sns.lineplot(x=np.arange(0, decisions.shape[1], 1), y=decisions[ind, :], palette="viridis", linewidth=3,
                     label=cond.replace("Res18_", "").replace("_", " "))
    plt.xticks(np.arange(0, decisions.shape[1]+1, 10))
    plt.yticks(np.arange(0, 1.1, 0.1), labels=np.arange(0, num_items + (num_items / 10), num_items / 10, dtype=np.int32))
    plt.legend(fontsize=40)
    plt.xlabel("Epoch", fontsize=60, labelpad=30)
    plt.ylabel("Number of different decisions", fontsize=60, labelpad=30)
    plt.ylim(0,1.05)
    sns.despine(top=True, right=True, offset=10, trim=True)
    sns.color_palette("viridis", as_cmap=True)
    
    plt.tight_layout()
    #plt.savefig(figure_path + f'{base_network}_decisions_change.png', bbox_inches='tight')

    return


def main_plot(main, epoch, figure_path, base_network, order=None):
    """
    :param main: Results array
    :param epochs: Which epochs to include in plot
    :param figure_path: Output path for figures
    :param base_network: Name of base network
    :return:
    """
    # Order main array according to last epoch
    num_epochs = main.shape[0]
    num_conditions = main.shape[1]
    econ_means = np.zeros(num_conditions)

    # Order conditions by highest mean
    fig = plt.figure(figsize=(20, 10))
    
    # Fill array for mean values
    econ_means = np.zeros(num_conditions)
    for ind in range(num_conditions):
        econ_means[ind] = np.mean(main[epoch, ind, 1])

    # Order arrays from high to low
    if type(order) == np.ndarray:
        order_high_low = order
    else:
        order_high_low = np.flip(np.argsort(econ_means))
        
    ordered_means = econ_means[order_high_low]
    ordered_econs = main[epoch, order_high_low, 1]
    
    # Remove first \n from labels
    labels = main[0, order_high_low, 0]
    labels_correct = np.empty(len(labels), dtype=object)
    for ind, label in enumerate(labels):
        labels_correct[ind] = label[1:]
        
    #plt.xticks(np.arange(0, len(main[0, :, 0]), 1), labels = np.arange(1,12,1), rotation=60)
    for ind in range(num_conditions):
        ax = sns.scatterplot(y=ordered_econs[ind], x=ind, s=250, marker='o', color='black', alpha=0.5)

    # Plot means and individual models
    ax = sns.scatterplot(data=ordered_means, s=250, marker='o', color='black', alpha=1.0)

    # Plot settings
    sns.set(font_scale = 2)
    ax.set_ylabel("Error Consistency", fontsize=30, labelpad=25)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, len(main[0, :, 0]), 1))
    ax.set_xticklabels(labels_correct, ha='right')
    

    # Context setting and saving plot
    sns.despine(top=True, right=True, offset=20, trim=True)
    ax.tick_params(axis = 'x', pad=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    # plt.savefig(figure_path + f'{base_network}_main_plot.png', bbox_inches='tight')

    return order_high_low

def main_plot_acc(main, epoch, figure_path, base_network, labels, ylabel, order=None):
    """
    :param main: Results array
    :param epochs: Which epochs to include in plot
    :param figure_path: Output path for figures
    :param base_network: Name of base network
    :return:
    """
    
    # Order conditions by highest mean
    fig = plt.figure(figsize=(20, 10))
    sns.set_context("paper")
    sns.set_style("white")
    
    # Order arrays from high to low. If order is passed, it is assumed that labels are already sorted aswell!
    if type(order) == np.ndarray:
        print("Using Res18 order")
        warnings.warn("If passing an order, it is assumed that labels are already ordered aswell!")
        order_high_low = order
        ordered_labels = labels
    else:
        # Careful! If no order is passed, labels are sorted by high low also. 
        order_high_low = np.flip(np.argsort(main))
        ordered_labels = labels[order_high_low]
        
    ordered_means = main[order_high_low]

    # Plot means and individual models
    ax = sns.scatterplot(data=ordered_means, s=250, marker='o', color='black', alpha=1.0)

    # Plot settings
    sns.set(font_scale = 2)
    ax.set_ylabel(ylabel, fontsize=30, labelpad=25)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, len(main), 1))
    ax.set_xticklabels(ordered_labels, ha='right')
    

    # Context setting and saving plot
    sns.despine(top=True, right=True, offset=20, trim=True)
    ax.tick_params(axis = 'x', pad=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    # plt.savefig(figure_path + f'{base_network}_main_plot_acc.png', bbox_inches='tight')

    return order_high_low


def main_plot_both(main, main2, epoch, figure_path, order_high_low):
    """
    :param main: Results array
    :param epochs: Which epochs to include in plot
    :param figure_path: Output path for figures
    :param base_network: Name of base network
    :return:
    """
    # Order main array according to last epoch
    num_epochs = main.shape[0]
    num_conditions = main.shape[1]
    econ_means = np.zeros(num_conditions)
    econ2_means = np.zeros(num_conditions)

    # Order conditions by highest mean
    fig = plt.figure(figsize=(20, 10))
    
    # Fill array for mean values
    econ_means = np.zeros(num_conditions)
    for ind in range(num_conditions):
        econ_means[ind] = np.mean(main[epoch, ind, 1])

    # Fill array for mean values
    econ2_means = np.zeros(num_conditions)
    for ind in range(num_conditions):
        econ2_means[ind] = np.mean(main2[epoch, ind, 1])
        
    # Order arrays from high to low
    ordered_means = econ_means[order_high_low]
    ordered_means2 = econ2_means[order_high_low]
    ordered_econs = main[epoch, order_high_low, 1]
    ordered_econs2 = main2[epoch, order_high_low, 1]
    
    
    # Remove first \n from labels
    labels = main[0, order_high_low, 0]
    labels_correct = np.empty(len(labels), dtype=object)
    for ind, label in enumerate(labels):
        labels_correct[ind] = label[1:]
        
    #plt.xticks(np.arange(0, len(main[0, :, 0]), 1), labels = np.arange(1,12,1), rotation=60)
    for ind in range(num_conditions):
        ax = sns.scatterplot(y=ordered_econs[ind], x=ind, s=250, marker='o', color='orange', alpha=0.5)
        ax = sns.scatterplot(y=ordered_econs2[ind], x=ind, s=250, marker='x', color='purple', alpha=0.5)

    # Plot means and individual models
    ax = sns.scatterplot(data=ordered_means, s=250, marker='o', color='orange', alpha=1.0, label="Gaussian")
    ax = sns.scatterplot(data=ordered_means2, s=250, marker='x', color='purple', alpha=1.0, label="CIFAR-100")
    
    # Plot settings
    sns.set(font_scale = 2)
    ax.set_ylabel("Error Consistency", fontsize=30, labelpad=25)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, len(main[0, :, 0]), 1))
    ax.set_xticklabels(labels_correct, ha='right')
    ax.legend(loc="lower left", fontsize=30, frameon=False)

    # Context setting and saving plot
    sns.despine(top=True, right=True, offset=20, trim=True)
    ax.tick_params(axis = 'x', pad=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    # plt.savefig(figure_path + f'{base_network}_main_plot.png', bbox_inches='tight')

    return


def plotSingle(indexNumber,imageEasy, imageImpossible, Training=False):
    plt.close()

    LeftRight = np.random.rand()

    plt.figure(figsize=(20,16))
    plt.axis('on')

    ax = plt.subplot(121)
    if(LeftRight>0.5): 
        if(Training==True):
            ax.patch.set_edgecolor('green')  
            ax.patch.set_linewidth('50')
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        plt.imshow(imageEasy)
    else:
        plt.imshow(imageImpossible)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.axis('on')
    plt.title('L', fontsize=50)
    plt.ylabel(str(indexNumber), fontsize=50,rotation=0)
    plt.xticks([], [])
    plt.yticks([], [])
    ax.yaxis.set_label_coords(-0.2,0.5)

    ax =plt.subplot(122)

    if(LeftRight>0.5): 
        plt.imshow(imageImpossible)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    else:
        if(Training==True):
            ax.patch.set_edgecolor('green')  
            ax.patch.set_linewidth('50') 
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        plt.imshow(imageEasy)

    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('R', fontsize=50)
    if Training==True:
        plt.savefig('./experiment/ExpImage_train_' + str(indexNumber) + '.jpg')
    else:
        plt.savefig('./experiment/ExpImage_' + str(indexNumber) + '.jpg')
    plt.show()
    if(LeftRight > 0.5):
        return 'l'
    else:
        return 'r'

