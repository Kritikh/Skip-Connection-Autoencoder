#from model import Autoencoder
from AE_skip import Skip_connect
from dataloader import get_dataloaders
from train import train_model
from visualize import visualize_results

if __name__ == "__main__":
    # Get the dataloaders
    train_loader, test_loader, val_loader = get_dataloaders()

    # Initialize the model
    model = Skip_connect()
    #model = Autoencoder()

    # Train the model
    train_model(model, train_loader, val_loader)

    # Visualize the results
    visualize_results(model, test_loader)
