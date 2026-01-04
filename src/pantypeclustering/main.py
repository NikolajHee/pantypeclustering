
import torch
import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt

from pantypeclustering.utils import get_args
from pantypeclustering.dataloader import get_mnist_dataloaders
from pantypeclustering.model import Recogniser, YGenerator, PriorGenerator, GMVAE
from pantypeclustering.config import get_training_parameters

def main():

    cfg = get_training_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    recog = Recogniser(input_size=cfg.y_size,
                       hidden_size=cfg.hidden_size,
                       x_size=cfg.x_size,
                       w_size=cfg.w_size,
                       number_of_mixtures=cfg.number_of_mixtures)
    
    ygen = YGenerator(input_size=cfg.x_size,
                      hidden_size=cfg.hidden_size,
                      output_size=cfg.y_size,
                      continuous=True)
    
    priorgen = PriorGenerator(input_size=cfg.w_size,
                              hidden_size=cfg.hidden_size,
                              output_size=cfg.x_size,
                              number_of_mixtures=cfg.number_of_mixtures)

    model = GMVAE(recogniser=recog,
                  ygenerator=ygen,
                  priorgenerator=priorgen,
                  mc=cfg.mc,
                  continuous=True)


    model.to(device)

    train_loader, test_loader = get_mnist_dataloaders(batch_size=256, binarize=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    num_epochs = cfg.max_epochs
    loss_train_save = torch.zeros(num_epochs)
    loss_test_save = torch.zeros(num_epochs)
    accuracy = torch.zeros(num_epochs)
    for epoch in tqdm.tqdm(range(num_epochs), disable=False):
        model.train()
        train_loss = 0.0
        for batch_idx, (images, _) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), disable=False):
            optimizer.zero_grad()
            loss, (y, y_recon) = model(images.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        loss_train_save[epoch] = avg_train_loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        model.eval()
        test_loss = 0.0
        K = 10
        batch_size = 256
        test_class_probs = torch.zeros((len(test_loader.dataset), K))
        test_label = torch.zeros((len(test_loader.dataset)))

        with torch.no_grad():
            for i, (images, label) in enumerate(test_loader):
                loss, (y, y_recon) = model(images.to(device))
                test_loss += loss.item()
                test_class_probs[i*batch_size : (i+1)*batch_size] = model.get_class_prob(images.to(device)).T
                test_label[i*batch_size : (i+1)*batch_size] = label
                if i == 0:
                    for j in range(10):
                        fig, axs = plt.subplots(1, 2)
                        axs[0].imshow(y[j].squeeze().detach().cpu().numpy())
                        axs[1].imshow(y_recon[0][j].squeeze().detach().cpu().numpy())
                        plt.savefig(f"img{j}.png")
                        plt.close()
        
            acc = model.acc_evaluation(test_class_probs, test_label)
        
        accuracy[epoch] = acc
        avg_test_loss = test_loss / len(test_loader)
        loss_test_save[epoch] = avg_test_loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}, ACC: {acc}")

    # Save model
    torch.save(model.state_dict(), "gmvae_mnist.pth")
    torch.save(loss_train_save, "avg_train.npy")
    torch.save(loss_test_save, "avg_test.npy")
    torch.save(accuracy, f"accuracy.npy")

if __name__ == "__main__":
    main()
    
