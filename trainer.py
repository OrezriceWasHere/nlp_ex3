from datetime import datetime

import torch
from tqdm import tqdm
from hyper_parameters import *
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt


def train(train_loader, test_loader, model, criterion, optimizer, epochs, device, name=None):
    # Prepare training

    loss_per_epoch = []
    accuracy_per_epoch = []

    # Start training
    for epoch in range(epochs):
        predictions = []
        truths = []
        total_loss_train = 0

        print("Epoch: ", epoch)
        print("\n----------------")
        print("Train:")
        model.train()
        for text, label in (pbar := tqdm(train_loader)):
            text = text.to(device)
            label = label.to(device)
            pbar.set_description(f"Training epoch {epoch}")

            # input(":(")
            output = model(text)
            # input()
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            total_loss_train += batch_loss
            predictions.extend(output.argmax(dim=1).tolist())
            truths.extend(torch.argmax(label, dim=1).tolist())
        print(f'Train Loss: {total_loss_train / len(train_loader): .3f}')
        # print(classification_report(truths, predictions))
        print(f'Train accuracy: {len([0 for t, p in zip(truths, predictions) if t == p]) / len(truths)}')

        ratio = len([0 for p in predictions if p == 0]) / len(predictions)
        if ratio < 0.15:
            print(f"Too much 1! {ratio}")
            criterion = torch.nn.BCELoss(torch.tensor((0.75, 0.25)).to(device))
            criterion = torch.nn.BCELoss(torch.tensor((0.9, 0.1)).to(device))
        elif ratio > 0.85:
            print(f"Too much 0! {ratio}")
            criterion = torch.nn.BCELoss(torch.tensor((0.1, 0.9)).to(device))
        else:
            print(f"Back to normality {ratio}")
            criterion = torch.nn.BCELoss()

        predictions = []
        truths = []
        total_loss_test = 0

        print("\n----------------")
        print("Test: ")

        with torch.no_grad():
            model.eval()
            for text, label in (pbar := tqdm(test_loader)):
                text = text.to(device)
                label = label.to(device)
                pbar.set_description(f"Evaluation epoch {epoch}")
                output = model(text)
                loss = criterion(output, label)
                batch_loss = loss.item()
                total_loss_test += batch_loss
                predictions.extend(output.argmax(dim=1).tolist())
                truths.extend(label.argmax(dim=1).tolist())

        print(f'Test Loss: {total_loss_test / len(test_loader): .3f}')
        loss_per_epoch.append(total_loss_test / len(test_loader))
        accuracy_per_epoch.append(len([0 for t, p in zip(truths, predictions) if t == p]) / len(truths))
        print(f'Test accuracy: {accuracy_per_epoch[-1]}')

        # print(classification_report(truths, predictions))

        if epoch % 20 == 0 and False:
            matrix = confusion_matrix(truths, predictions)
            cm_display = ConfusionMatrixDisplay(confusion_matrix=matrix)
            cm_display.plot()
            plt.show()

    if name is None:
        # Assign name to none according top finish time of training
        name = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    plt.plot(loss_per_epoch)
    plt.savefig(name + '_loss.png')

    plt.plot(accuracy_per_epoch)
    plt.savefig(name + '_accuracy.png')
