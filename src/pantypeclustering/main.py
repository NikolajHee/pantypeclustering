import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from pantypeclustering.config import get_training_parameters
from pantypeclustering.dataloader import get_mnist_dataloaders
from pantypeclustering.model import GMVAE


def main() -> None:
    """ Train the VAE model on MNIST dataset. """


    cfg = get_training_parameters()

    torch.manual_seed(cfg.seed)


    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )

    model = GMVAE(
        y_size=cfg.y_size,
        x_size=cfg.x_size,
        w_size=cfg.w_size,
        hidden_size=cfg.hidden_size,
        number_of_mixtures=cfg.number_of_mixtures,
        mc=cfg.mc,
        continuous=cfg.continuous,
        lambda_threshold=cfg.lambda_threshold,
    )

    model.to(device)

    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=cfg.batch_size,
        binarize=False,
        seed=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)


    loss_train_save = torch.zeros(cfg.max_epochs)
    loss_test_save = torch.zeros(cfg.max_epochs)
    accuracy = torch.zeros(cfg.max_epochs)
    db_scores = torch.zeros(cfg.max_epochs)
    adj_rand_scores = torch.zeros(cfg.max_epochs)

    disable_tqdm = False

    for epoch in tqdm(range(cfg.max_epochs), disable=disable_tqdm):
        model.train()
        train_loss = torch.zeros(len(train_loader))
        test_loss = torch.zeros(len(test_loader))

        for i, (images, _) in tqdm(enumerate(train_loader), total=len(train_loader), disable=disable_tqdm):
            optimizer.zero_grad()
            loss, (y, y_recon) = model(images.to(device))
            loss.backward()
            optimizer.step()
            train_loss[i] = loss.item()

        loss_train_save[epoch] = train_loss.mean()
        print(f"Epoch [{epoch+1}/{cfg.max_epochs}], Train Loss: {train_loss.mean():.4f}")

        model.eval()

        test_class_probs = torch.zeros((len(test_loader.dataset), cfg.number_of_mixtures))
        test_label = torch.zeros((len(test_loader.dataset)))
        test_x = torch.zeros((len(test_loader.dataset), cfg.x_size))

        with torch.no_grad():
            for i, (images, label) in enumerate(test_loader):
                loss, (y, y_recon) = model(images.to(device))
                test_loss[i] = loss.item()

                start_idx = i * cfg.batch_size
                end_idx = start_idx + len(label)
                test_class_probs[start_idx:end_idx] = model.get_class_prob(images.to(device)).T
                test_label[start_idx:end_idx] = label
                (mean_x, logvar_x), (mean_w, logvar_w) = model.recogniser(images.to(device))

                test_x[start_idx:end_idx] = mean_x

                # Save sample images from first batch of each epoch
                if i == 0:
                    for j in range(min(10, len(y))):
                        _, axs = plt.subplots(1, 2)
                        axs[0].imshow(y[j].squeeze().detach().cpu().numpy(), cmap="gray")
                        axs[1].imshow(y_recon[0][j].squeeze().detach().cpu().numpy(), cmap="gray")
                        plt.savefig(f"img_sample{j}.png")
                        plt.close()

            acc = model.acc_evaluation(test_class_probs, test_label)
            db_score = model.get_db_score(test_x, test_class_probs)
            adj_rand_score = model.get_adjust_rand(test_label, test_class_probs)

        accuracy[epoch] = acc
        db_scores[epoch] = db_score
        adj_rand_scores[epoch] = adj_rand_score

        loss_test_save[epoch] = test_loss.mean()

        print(f"Epoch [{epoch+1}/{cfg.max_epochs}], Test Loss: {test_loss.mean():.4f}, ACC: {acc:.4f}, DB: {db_score:.4f}, ADJ; {adj_rand_score}")

    # Save model and results
    #torch.save(model.state_dict(), "gmvae_mnist.pth")
    #torch.save(loss_train_save, "avg_train.npy")
    #torch.save(loss_test_save, "avg_test.npy")
    torch.save(accuracy, f"accuracy.npy")
    torch.save(db_scores, f"db_scores.npy")
    torch.save(adj_rand_scores, f"adj_rand_scores.npy")



if __name__ == "__main__":
    main()