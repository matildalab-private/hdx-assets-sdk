import copy
import os
import os.path
import time

import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from torch import optim, nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from asset_hub.torchbinding.folder import ImageFolder as AssetHubImageFolder
from asset_hub.asset_hub_api import AssetHubAPI

use_asset_hub = True


def main():
    api = AssetHubAPI('.asset_mlops')
    vision_dataset = api.assets(assets_id=134)

    writer = SummaryWriter('./experiment_AIhub')
    batch_size = 32
    epochs = 30

    data_transforms = {'train': transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.RandomRotation(0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize((256, 256))
        ])}

    if use_asset_hub:
        base_path = '/OF'
        image_datasets = {x: AssetHubImageFolder(vision_dataset,
                                                 os.path.join(base_path, x),
                                                 data_transforms[x])
                          for x in ['train', 'val']}
    else:
        data_dir = '/Users/codex/works/dataset/OF'
        from torchvision import datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(dataset_sizes, class_names)
    print(class_names)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=5)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()

            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            # writer.add_graph('epoch loss', epoch_loss, epoch)
            # writer.add_graph('epoch acc', epoch_acc, epoch)
            writer.add_scalar('epoch loss', epoch_loss, epoch)
            writer.add_scalar('epoch acc', epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

    writer.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    torch.save(best_model_weights, './best_weights_b5_class_AIhub.pth')


#
# def orig_main():
#     batch_size = 32
#     epochs = 30
#     data_dir = '/Users/codex/works/dataset/OF'
#     writer = SummaryWriter('./orig/experiment_AIhub')
#
#     data_transforms = {'train': transforms.Compose([
#         transforms.Resize((256, 256)),
#         # transforms.RandomRotation(0),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#         'val': transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             transforms.Resize((256, 256))
#         ])}
#
#     from torchvision import datasets
#     image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                               data_transforms[x])
#                       for x in ['train', 'val']}
#
#     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
#                                                   shuffle=True, num_workers=4)
#                    for x in ['train', 'val']}
#
#     dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#     class_names = image_datasets['train'].classes
#     print(dataset_sizes, class_names)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=5)
#     model.to(device)
#
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#
#     since = time.time()
#
#     best_model_weights = copy.deepcopy(model.state_dict())
#
#     best_acc = 0.0
#
#     for epoch in range(epochs):
#         print('Epoch {}/{}'.format(epoch, epochs - 1))
#         print('-' * 10)
#
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#
#             else:
#                 model.eval()
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#                 optimizer.zero_grad()
#
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)
#
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#
#             if phase == 'train':
#                 scheduler.step()
#
#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]
#             # writer.add_graph('epoch loss', epoch_loss, epoch)
#             # writer.add_graph('epoch acc', epoch_acc, epoch)
#             writer.add_scalar('epoch loss', epoch_loss, epoch)
#             writer.add_scalar('epoch acc', epoch_acc, epoch)
#
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))
#
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_weights = copy.deepcopy(model.state_dict())
#
#     writer.close()
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))
#
#     torch.save(best_model_weights, './best_weights_b5_class_AIhub.pth')


if __name__ == '__main__':
    main()
