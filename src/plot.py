"""Plotting functions
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(log_history, save_path):
    train_loss = []
    valid_loss = []
    for element in log_history:
        if "loss" in element.keys():
            loss = element["loss"]
            train_loss.append(loss)
        elif "eval_loss" in element.keys():
            loss = element["eval_loss"]
            valid_loss.append(loss)
            
    plt.plot(np.arange(0, len(train_loss)), train_loss, marker = 'o', label='Train Loss')
    plt.plot(np.arange(0, len(valid_loss)), valid_loss, marker = 'o', label='Valid Loss')
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.savefig(save_path) 
    plt.cla()
    plt.close()


def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset, save_path):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    # plt.xlim([0, 1500])
    plt.savefig(save_path)
    plt.cla()
    plt.close()


def plot_temp_error(temp, error, save_path):
    # plt.style.use(['science','ieee'])
    plt.figure()
    for key in error.keys():
     plt.plot(temp, error[key], label=key)

    plt.xlabel('Temperature')
    plt.ylabel('Error (MRE)')
    plt.legend(loc="upper left")
    plt.savefig(save_path)
    plt.cla()
    plt.close()
 
 
 
def plot_pred(ground_truth, predictions, save_path):
    
    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(ground_truth, predictions, color='blue', label='Predictions vs Ground Truth')
    plt.plot([min(ground_truth), max(ground_truth)], [min(ground_truth), max(ground_truth)], 'r--', label='Perfect Prediction')

    # Adding labels and title
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title('Prediction vs Ground Truth Comparison')
    plt.legend()

    # Adding grid
    plt.grid(True)

    # Show the plot
    plt.savefig(save_path)
    