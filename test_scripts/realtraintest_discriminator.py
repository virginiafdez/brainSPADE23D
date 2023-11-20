import monai.networks.nets as mnn
import numpy as np
import os
from data.rrt_dataset import getTrainingDataloaders
from options.realtraintest_disc_options import RTT_DiscOptions
import torch
import monai
from torch.utils.tensorboard import SummaryWriter

# Auxiliary functions
def findLRGamma(startingLR, endingLR, num_epochs):
    '''
    Gamma getter. Based on a minimum and maximum learning rate, calculates the Gamma
    necessary to go from minimum to maimum in num_epochs.
    :param startingLR: First Learning Rate.
    :param endingLR: Final Learning Rate.
    :param num_epochs: Number of epochs.
    :return:
    '''

    gamma = np.e ** (np.log(endingLR / startingLR) / num_epochs)
    return gamma

# Define options
options = RTT_DiscOptions().parse()

# Define network
rtt_classifier = mnn.DenseNet169(spatial_dims=2,
                                 in_channels = 1, out_channels=1, init_features=options.init_features,
                                 norm = 'instance'
                                 )

# Define datasets (train, est)
train_loader, val_loader = getTrainingDataloaders(options)

# Load state
if os.path.isdir(os.path.join(options.checkpoints_dir, options.name)):
    with open(os.path.join(options.checkpoints_dir, options.name, 'current_epoch.txt'), 'r') as f:
        lines = f.readlines()
    for l in lines:
        current_epoch = float(l.replace("\n", ""))
else:
    os.makedirs(os.path.join(options.checkpoints_dir, options.name))
    current_epoch = 0
    with open(os.path.join(options.checkpoints_dir, options.name, 'loss_log.txt'), 'w') as f:
        f.write("Epoch\tLoss\tAccuracy\n")
        f.close()

# Get learning rate, optimizer and scheduler
gamma = findLRGamma(options.lr_init, options.lr_end, options.num_epochs)
lr = options.lr_init*gamma**(current_epoch)
optimiser = torch.optim.Adam(rtt_classifier.parameters(), lr = lr, betas=(0.99, 0.999))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma = gamma)

# Define loss function
loss_function = torch.nn.BCEWithLogitsLoss()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define tensorboard
tboard_train = SummaryWriter(log_dir=os.path.join(options.checkpoints_dir, options.name))
tboard_validation = SummaryWriter(log_dir=os.path.join(options.checkpoints_dir, options.name))

for e in range(current_epoch, options.num_epochs):
    print("Epoch %d/%d" %(e, options.num_epochs))
    epoch_accuracy = []
    epoch_loss = []
    for ind, i in enumerate(train_loader):
        #### Forward pass and backward pass
        image = i["image"]
        brain_mask = torch.argmax(i["label"], 1)
        label = i["tag"]

        # Skullstrip if needed
        if options.skullstrip:
            image[brain_mask==0] = 0.0 # Zero non-brain voxels

        # Send to device
        image = image.to(device)
        label = label.to(device)
        rtt_classifier = rtt_classifier.to(device).train()

        # Forward pass
        out_classified = rtt_classifier(image)

        # Calculate loss function, backward and step
        loss = loss_function(out_classified, label)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Calculate accuracy and store
        accuracy = (torch.sigmoid(out_classified.detach().cpu().round()) == label).mean()
        epoch_loss.append(loss.item())
        epoch_accuracy.append(accuracy.numpy())

    # VALIDATION
    validation_accuracy = []
    validation_loss = []
    for ind, i in enumerate(val_loader):
        #### Forward pass and backward pass
        image = i["image"]
        brain_mask = torch.argmax(i["label"], 1)
        label = i["tag"]

        # Skullstrip if needed
        if options.skullstrip:
            image[brain_mask == 0] = 0.0  # Zero non-brain voxels

        # Send to device
        image = image.to(device)
        label = label.to(device)
        rtt_classifier = rtt_classifier.to(device).eval()

        # Forward pass
        out_classified = rtt_classifier(image)

        # Calculate loss function, backward and step
        loss = loss_function(out_classified, label)

        # Calculate accuracy and store
        accuracy = (torch.sigmoid(out_classified.detach().cpu().round()) == label).mean()
        validation_loss.append(loss.item())
        validation_accuracy.append(accuracy.numpy())

    # Print / Store / Tensorboard
    tboard_train.add_scalar("loss", np.mean(epoch_loss), e)
    tboard_train.add_scalar("accuracy", np.mean(epoch_accuracy), e)
    tboard_validation.add_scalar("loss", np.mean(validation_loss), e)
    tboard_validation.add_scalar("accuracy", np.mean(validation_accuracy), e)
    print("TRAINING: Loss: %.3f, Accuracy: %.3f\nVALIDATION: Loss: %.3f, Accuracy: %.3f\n" %(np.mean(epoch_loss),
                                                                                             np.mean(epoch_accuracy),
                                                                                             np.mean(validation_loss),
                                                                                             np.mean(validation_accuracy)))

    # Save model
    save_path = os.path.join(options.checkpoints_dir, options.name, "latest_net_RTT.pth")
    torch.save(rtt_classifier.cpu().state_dict(), save_path)
    rtt_classifier = rtt_classifier.to(device)

    # Advance scheduler and move on current epoch
    scheduler.step()
    with open(os.path.join(options.checkpoints_dir, options.name, 'current_epoch.txt'), 'w') as f:
        f.write(str(e))
        f.close()

