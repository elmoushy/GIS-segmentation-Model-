TRAIN, VALID = 0, 1
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
import glob
from PIL import Image
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import ConcatDataset as MyDataset

torch.set_float32_matmul_precision("high")
data_dir = r"/content/drive/MyDrive/Dataset"
output_dir = r"/content/drive/MyDrive/Models/Main/Final"


import os
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
import os


def get_images_and_masks(data_dir, dataset_type):
    images = []
    masks = []

    dataset_path = data_dir + f"/{dataset_type}/{dataset_type}"
    categories = ["Rural", "Urban"]

    for category in categories:
        images_path = os.path.join(dataset_path, category, "images_png")
        masks_path = os.path.join(dataset_path, category, "masks_png")

        images.extend(
            sorted([os.path.join(images_path, img) for img in os.listdir(images_path)])
        )
        masks.extend(
            sorted([os.path.join(masks_path, mask) for mask in os.listdir(masks_path)])
        )

    return list(sorted(images)), list(sorted(masks))


class CustomDataset(Dataset):
    def __init__(self, root_dir, split_index=TRAIN, transform=None, augmentation=True):
        self.root_dir = root_dir
        self.transform = transform
        self.max_retries = 5
        self.augmentation = augmentation
        self.image_files, self.mask_files = get_images_and_masks(
            data_dir, self.split[split_index]
        )

    def __len__(self):
        return len(self.image_files) * 2  # Double the length for augmentation

    def __getitem__(self, idx):
        # Check if the current index corresponds to the original or augmented data
        is_augmented = idx >= len(self.image_files)
        if is_augmented:
            idx -= len(self.image_files)  # Adjust index for augmented data

        img_name = os.path.join(self.images_folder, self.image_files[idx])
        mask_name = os.path.join(self.masks_folder, self.mask_files[idx])

        retries = 0
        while retries < self.max_retries:
            try:
                # Read images using cv2
                image = cv2.imread(img_name)  # Read image in BGR format

                if image is None:
                    raise ValueError(
                        f"Error in loading image at index {idx}. File path: {img_name}"
                    )

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

                mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

                if self.augmentation and is_augmented:
                    # Rotate image and mask by 90 degrees if augmentation is enabled and data is augmented
                    image = np.rot90(image)
                    mask = np.rot90(mask)

                if self.transform:
                    # Apply any additional transformations if needed
                    # Note: You might need to adjust this based on your specific transformation requirements
                    image = self.transform(image)

                return image, mask

            except Exception as e:
                print(
                    f"Error in loading image at index {idx}. Retrying... ({retries}/{self.max_retries})"
                )
                retries += 1

        # If all retries fail, raise an error
        raise ValueError(
            f"Failed to load image after {self.max_retries} retries. File path: {img_name}"
        )

    split = ["Train", "VALID"]


def custom_color_adjustment(img, xp=[0, 64, 128, 192, 255], fp=[0, 64, 128, 225, 255]):
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype("uint8")
    adjusted_img = cv2.LUT(img, table)
    return adjusted_img


def Load_Data(
    root_dir,
    batch_size=1,
    shuffle=False,
):
    transform = transforms.Compose(
        [
            custom_color_adjustment,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = CustomDataset(root_dir, split_index=TRAIN, transform=transform)
    val_dataset = CustomDataset(root_dir, split_index=VALID, transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2
    )
    return train_loader, val_loader


# Edge Deteciton Model
class Edge_Detection_Model(nn.Module):
    def __init__(self):
        super(Edge_Detection_Model, self).__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netScoreOne = torch.nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreTwo = torch.nn.Conv2d(
            in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreThr = torch.nn.Conv2d(
            in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreFou = torch.nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreFiv = torch.nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0
        )

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, tenInput):
        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(
            input=tenScoreOne,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tenScoreTwo = torch.nn.functional.interpolate(
            input=tenScoreTwo,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tenScoreThr = torch.nn.functional.interpolate(
            input=tenScoreThr,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tenScoreFou = torch.nn.functional.interpolate(
            input=tenScoreFou,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tenScoreFiv = torch.nn.functional.interpolate(
            input=tenScoreFiv,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        return self.netCombine(
            torch.cat(
                [tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv], 1
            )
        )


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()
        self.query = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(q.size(0), q.size(1), -1)
        k = k.view(k.size(0), k.size(1), -1)
        v = v.view(v.size(0), v.size(1), -1)

        attention_weights = torch.matmul(q.permute(0, 2, 1), k)
        attention_weights = attention_weights / k.size(-1)  # Normalize by key dimension
        attention_weights = self.softmax(attention_weights)

        output = torch.matmul(v, attention_weights.permute(0, 2, 1))
        output = output.view_as(x)

        return output


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nin, nout, stride=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.residual = nin == nout and stride == 1
        self.depthwise = nn.Conv2d(
            nin,
            nin,
            kernel_size=3,
            padding=dilation,
            groups=nin,
            stride=stride,
            dilation=dilation,
        )
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        self.bn = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x if self.residual else None
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        if residual is not None:
            x += residual
        x = self.relu(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.avg_pool(x)
        return x


class CustomBackbone(nn.Module):
    def __init__(self):
        super(CustomBackbone, self).__init__()
        self.init_layers = nn.Sequential(
            DepthwiseSeparableConv(3, 64),
            DepthwiseSeparableConv(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2_x = self._make_layer(64, 128, 2, dilation=1)
        self.trans1 = TransitionLayer(128, 128)
        self.conv3_x = self._make_layer(128, 256, 3, dilation=2)
        self.trans2 = TransitionLayer(256, 256)
        self.conv4_x = self._make_layer(256, 512, 3, dilation=4)
        self.trans3 = TransitionLayer(512, 512)
        self.conv5_x = self._make_layer(512, 512, 3, dilation=4)

    def _make_layer(self, in_channels, out_channels, num_blocks, dilation):
        layers = []
        for _ in range(num_blocks):
            layers.append(
                DepthwiseSeparableConv(in_channels, out_channels, dilation=dilation)
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_layers(x)
        x = self.conv2_x(x)
        x = self.trans1(x)
        x = self.conv3_x(x)
        x = self.trans2(x)
        x = self.conv4_x(x)
        x = self.trans3(x)
        x = self.conv5_x(x)
        return x


class Combined_Model(nn.Module):
    def __init__(self):
        super(Combined_Model, self).__init__()
        self.Backbone1 = Edge_Detection_Model()
        self.BackBone2 = CustomBackbone()
        self.attention = Attention(1, 1)
        self.Convertion = nn.Conv2d(
            in_channels=5, out_channels=512, kernel_size=32, stride=16, padding=8
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=2, stride=2
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=2, stride=2
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=2, stride=2
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=8, kernel_size=2, stride=2
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        edges = self.Backbone1(input)
        attention_weights = self.attention(edges)
        Backbone = self.BackBone2(input)
        x = torch.cat([input, attention_weights, edges], dim=1)
        x = self.Convertion(x)
        x = torch.cat([Backbone, x], dim=1)
        return self.decoder(x)


class Custom_Edge_Unet_net(pl.LightningModule):
    def __init__(self):
        super(Custom_Edge_Unet_net, self).__init__()
        self.model = Combined_Model()

    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        inputs, masks = batch
        outputs = self(inputs)
        masks = masks.to(torch.long)

        loss = nn.CrossEntropyLoss()(outputs, masks)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, masks = batch
        outputs = self(inputs)
        masks = masks.to(torch.long)

        loss = nn.CrossEntropyLoss()(outputs, masks)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
        return [optimizer], [scheduler]


def main():
    train_loader, val_loader = Load_Data(data_dir, batch_size=2)

    # Checkpoint for the best model based on validation loss
    best_model_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename="best_model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    # Checkpoint to save the model at the end of each epoch
    every_epoch_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename="epoch-{epoch:02d}",
        save_top_k=1,  # Save all checkpoints
        verbose=True,
        monitor=None,  # No monitoring for every epoch checkpoint
        mode="min",  # Doesn't matter since monitor is None
    )
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", patience=3)

    logger = TensorBoardLogger(output_dir)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=200,
        devices=1,
        accelerator="gpu",
        callbacks=[best_model_checkpoint, every_epoch_checkpoint, early_stop_callback],
    )
    model = Custom_Edge_Unet_net()

    trainer.fit(model, train_loader, val_loader)


# if __name__=='__main__':
# import multiprocessing
# multiprocessing.freeze_support()
main()
