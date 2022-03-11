# Imports
import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader
from helpers import process_sys, parameter_check, set_seeds, set_cuda_randomness, set_optim, make_info, lr_update, run_check
from init import init_model, init_check
from data import init_dataset, split_dataset

# Process input variables
MODEL, DSET, INIT, OPTIM, DATA, ORDER, LR, BATCHES, EPOCHS, CUDA, NUM, VERBOSE, CONDITION, BATCH_SIZE = process_sys(sys.argv)

# Set device by checking CUDA availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Print parameters for manual check
parameter_check(MODEL, DSET, INIT, OPTIM, DATA, ORDER, LR, BATCHES, EPOCHS, CUDA, NUM, VERBOSE, DEVICE, CONDITION)

# Prepare paths, filenames and information for output
filename, info = make_info(MODEL, DSET, INIT, OPTIM, DATA, ORDER, LR, BATCHES, EPOCHS, CUDA, NUM, CONDITION)

# Check if condition has already been run and break if that is the case
run_check(filename, EPOCHS)

# Define global and initialization seed
GLOBAL_SEED = 1312
INIT_SEED = 1312

# Initialize model and check whether initialization strategy has worked correctly
model = init_model(INIT, INIT_SEED, MODEL, NUM, DEVICE)
init_check(model, INIT, INIT_SEED, MODEL, DEVICE)

# Set seeds and set CUDA to be deterministic or non-deterministic
set_seeds(GLOBAL_SEED)
set_cuda_randomness(CUDA)

# Initialize datasets
train_set = init_dataset(DSET, NUM, DATA, MODEL, CONDITION, train=True)

# Split FRACTAL dataset if it is used
if DSET == "FRACTAL":
    train_set, val_set = torch.utils.data.dataset.random_split(train_set, [int(len(train_set))-50000, 50000])
    train_set, _ = torch.utils.data.dataset.random_split(train_set, [75000, int(len(train_set))-75000])
else:
    val_set = init_dataset(DSET, NUM, DATA, MODEL, CONDITION, train=False)


# For different data condition, split dataset into two parts
if DATA == "different":
    train_set = split_dataset(train_set, NUM, GLOBAL_SEED)

# Initialise data loaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=30)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=30)  # Do not shuffle to keep same order

# Set loss function
if DSET == "ARN":
    criterion = nn.BCELoss().to(DEVICE)
else:
    criterion = nn.CrossEntropyLoss().to(DEVICE)

# Set optimizer
optimizer = set_optim(OPTIM, model, LR)

# Pre-allocate arrays for training and results (+1 for test set before model enters training)
loss_train = torch.zeros(EPOCHS).to(DEVICE)
acc_train = torch.zeros(EPOCHS).to(DEVICE)
loss_val = torch.zeros(EPOCHS+1).to(DEVICE)
acc_val = torch.zeros(EPOCHS+1).to(DEVICE)


# Start training model
for epoch in range(EPOCHS + 1):

    # Run test set before model has been trained at all
    if epoch != 0:

        # Run training dataset
        model.train()

        # Change seed before data is randomly drawn from train loader
        if ORDER == "different":
            set_seeds(GLOBAL_SEED + (NUM*100) + epoch)
        else:
            set_seeds(GLOBAL_SEED + epoch)

        # Get images from dataloader
        for i, (images, targets) in enumerate(train_loader):

            # Load images and targets onto GPU
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # For different batch size condition: halve sample and do two gradient updates
            if BATCHES == "different":

                # Loop through half batches
                for ind in range(2):

                    # Set start and end of half batch and calculate everything for this half
                    if(targets.shape[0] == BATCH_SIZE):
                        start = int(ind*(BATCH_SIZE/2))
                        end = int((ind+1)*(BATCH_SIZE/2))
                    else:
                        start = int(ind*(targets.shape[0]/2))
                        end = int((ind+1)*(targets.shape[0]/2))
                        
                    output = model(images[start:end, :, :, :])
                    loss = criterion(output, targets[start:end])
                    loss_train[epoch - 1] += loss.item()

                    # Compute accuracy: for each batch accuracy divided by dataset size, yields mean after last batch
                    if DSET == "ARN":
                        acc_train[epoch - 1] += torch.sum(torch.eq(targets[start:end], torch.round(output))) / len(train_loader.dataset)
                    else:
                        acc_train[epoch - 1] += torch.sum(torch.eq(targets[start:end], torch.argmax(output, dim=1))) / len(train_loader.dataset)

                    # Compute gradient and do optimizer step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            else:

                # Get output and loss
                output = model(images)
                loss = criterion(output, targets)
                loss_train[epoch-1] += loss.item()

                # Compute accuracy: for each batch accuracy divided by dataset size, this yields mean after last batch
                if DSET == "ARN":
                    acc_train[epoch-1] += torch.sum(torch.eq(targets, torch.round(output))) / len(train_loader.dataset)
                else:
                    acc_train[epoch-1] += torch.sum(torch.eq(targets, torch.argmax(output, dim=1))) / len(train_loader.dataset)

                # Compute gradient and do optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Disable gradient for test dataset
    model.eval()
    with torch.no_grad():

        # Re-set seed to global seed
        set_seeds(GLOBAL_SEED)

        # Run test set
        for i, (images, targets) in enumerate(val_loader):

            # Load images and targets onto GPU
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # Get output and loss
            output = model(images)
            loss = criterion(output, targets)
            loss_val[epoch] += loss.item()

            # Compute accuracy
            if DSET == "ARN":
                acc_val[epoch] += torch.sum(torch.eq(targets, torch.round(output))) / len(val_loader.dataset)
            else:
                acc_val[epoch] += torch.sum(torch.eq(targets, torch.argmax(output, dim=1))) / len(val_loader.dataset)

            # Prepare outputs to be saved for each epoch
            if i == 0:
                epoch_output = output
                epoch_targets = targets
            else:
                epoch_output = torch.cat((epoch_output, output))
                epoch_targets = torch.cat((epoch_targets, targets))

    # Manual learning rate scheduler
    lrTmp = lr_update(epoch, LR, EPOCHS)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrTmp
    print(f"Current learning rate: {lrTmp}")

    # Print output if wanted
    if VERBOSE == 1:
        print('Epoch {}/{}, Train loss: {:.4f}, Train accuracy: {:.4f}, Test loss: {:.4f}, Test accuracy:  {:.4f}'
              .format(epoch, EPOCHS,
                      loss_train[epoch-1], acc_train[epoch-1],
                      loss_val[epoch], acc_val[epoch]))

    # Save output after every epoch
    result = [info, torch.argmax(epoch_output, dim=1), epoch_targets]

    # Use torch save to write results and model into file
    torch.save(result, filename + "RESULTS_EP{}".format(epoch) + ".txt")
    torch.save(model, filename + "MODEL_EP{}".format(epoch))

# Save accuracy and loss over all epochs
torch.save(loss_train, filename + "TRAIN_LOSS.txt")
torch.save(loss_val, filename + "VAL_LOSS.txt")
torch.save(acc_train, filename + "TRAIN_ACC.txt")
torch.save(acc_val, filename + "VAL_ACC.txt")
