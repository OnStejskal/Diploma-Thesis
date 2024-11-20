from os import makedirs
from os.path import exists, join
from json import dump
import argparse
import sys


from torch import device
from torch.nn import CrossEntropyLoss
from torch.cuda import is_available, set_device
from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor
from carotids.segmentation.dataset import SegmentationDataset
from carotids.segmentation.dataset import SegmentationDatamodule
from carotids.segmentation.transformations import (
    SegCompose,
    SegCrop,
    SegRandomRotation,
    SegRandomContrast,
    SegRandomGammaCorrection,
    SegRandomBrightness,
    SegRandomHorizontalFlip,
    SegRandomVerticalFlip,
)
from carotids.segmentation.metrics import SegAccuracy
from carotids.segmentation.model import Unet, UnetDVCFS
from carotids.segmentation.module import SegModule
from carotids.segmentation.loss_functions import logcosh_dice_loss
from carotids.utils import split_dataset, split_dataset_into_dataloaders


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, help="number of CUDA device")
args = parser.parse_args()
if args.device or args.device == 0:
    DEVICE_NUMBER = args.device
    # print(f'DEVICE NUMBER is set to {DEVICE_NUMBER}')
else:
    print("No number was provided. Please Provide number")
    sys.exit()


EXPERIMENTS_FOLDER = "/datagrid/personal/stejson3/dp/carotids/experiments"
EXPERIMENT_NAME = "UNET_replicate_best_possible_with_75_25_50_split"
EXPERIMENT_PATH = join(EXPERIMENTS_FOLDER, EXPERIMENT_NAME)

N_CLASSES = 4

if not exists(EXPERIMENT_PATH):
    makedirs(EXPERIMENT_PATH)

GPUS = [DEVICE_NUMBER]
MAX_EPOCHS = 800

data_module = SegmentationDatamodule(
    "/mnt/datagrid/personal/stejson3/dp/carotids/data_samples/data/trans",
    "/mnt/datagrid/personal/stejson3/dp/carotids/data_samples/references/trans",
    SegCompose(
        [
            SegRandomRotation(),
            SegRandomContrast(),
            SegRandomGammaCorrection(),
            SegRandomBrightness(),
            SegRandomHorizontalFlip(),
            SegRandomVerticalFlip(),
            SegCrop(random_t=25),
        ]
    ),
    SegCompose([SegCrop(),]),
    Compose(
        [
            ToTensor(),
            Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        ]
    ),
    batch_size=14,
    num_workers=4,
    val_split = 1/6,
    test_split = 1/3,

)


module = SegModule(
    UnetDVCFS(N_CLASSES), #Unet(N_CLASSES),
    logcosh_dice_loss,
    learning_rate=0.00001,
    patience=25,
    accuracy=SegAccuracy((256, 256)),
    loss_weights=[1.0, 1.5, 1.75, 1.0]
)

device = device("cuda") if is_available() else device("cpu")

if GPUS is not None and is_available():
    set_device(GPUS[0])

module.train_model(
    data_module.train_loader, data_module.val_loader, device, MAX_EPOCHS, join(EXPERIMENT_PATH, EXPERIMENT_NAME)
)

module.load_model(join(EXPERIMENT_PATH, EXPERIMENT_NAME) + ".pt")

module.plot_datasets(
    [
        data_module.train_eval_set,
        data_module.val_eval_set,
        data_module.test_eval_set,
    ],
    EXPERIMENT_PATH,
    device
)

module.plot_datasets_differences(
    [
        data_module.train_eval_set,
        data_module.val_eval_set,
        data_module.test_eval_set,
    ],
    EXPERIMENT_PATH,
    device
)


RESULTS = {}

train_results = module.evaluate_dataloader(data_module.train_eval_loader, device)
RESULTS["train_loss"] = train_results[0]
RESULTS["train_acc"] = train_results[1]

val_results = module.evaluate_dataloader(data_module.val_eval_loader, device)
RESULTS["val_loss"] = val_results[0]
RESULTS["val_acc"] = val_results[1]

test_results = module.evaluate_dataloader(data_module.test_eval_loader, device)
RESULTS["test_loss"] = test_results[0]
RESULTS["test_acc"] = test_results[1]


RESULTS["train_cnf_matrix"] = module.confusion_matrix(data_module.train_eval_set, device).tolist()
RESULTS["val_cnf_matrix"] = module.confusion_matrix(data_module.val_eval_set, device).tolist()
RESULTS["test_cnf_matrix"] = module.confusion_matrix(data_module.test_eval_set, device).tolist()

ious = module.datasets_iou(
    [
        data_module.train_eval_set,
        data_module.val_eval_set,
        data_module.test_eval_set,
    ],
    device,
    N_CLASSES
)

RESULTS["train_set_iou"] = ious["train_set"]
RESULTS["validation_set_iou"] = ious["validation_set"]
RESULTS["test_set_iou"] = ious["test_set"]

with open(join(EXPERIMENT_PATH, "results.json"), "w") as fp:
    dump(RESULTS, fp)



#early_stop_callback = EarlyStopping(
#   monitor="val_loss_epoch",
#   min_delta=0.00,
#   patience=250,
#   #verbose=False,
#   mode="min"
#)
#
#checkpoint_callback = ModelCheckpoint(
#    monitor="val_loss_epoch",
#    dirpath=EXPERIMENT_PATH,
#    filename=EXPERIMENT_NAME,
#    save_top_k=1
#)
#trainer = Trainer(
#    callbacks=[early_stop_callback, checkpoint_callback],
#    gpus=GPUS,
#    accelerator="dp",
#    max_epochs=MAX_EPOCHS,
#    default_root_dir=EXPERIMENT_PATH
#)
#
#trainer.fit(module, data_module.train_loader, data_module.val_loader)
#
#module = SegModule.load_from_checkpoint(
#    join(EXPERIMENT_PATH, EXPERIMENT_NAME + ".ckpt"),
#    model=UnetDVCFS(4),
#    loss=LogCoshDiceLoss(),
#)
#
#
#RESULTS = {}
#
#train_results = trainer.test(test_dataloaders=data_module.train_eval_loader)
#RESULTS["train_loss"] = train_results[0]["test_loss"]
#RESULTS["train_acc"] = train_results[0]["test_acc"]
#
#val_results = trainer.test(test_dataloaders=data_module.val_eval_loader)
#RESULTS["val_loss"] = val_results[0]["test_loss"]
#RESULTS["val_acc"] = val_results[0]["test_acc"]
#
#test_results = trainer.test(test_dataloaders=data_module.test_eval_loader)
#RESULTS["test_loss"] = test_results[0]["test_loss"]
#RESULTS["test_acc"] = test_results[0]["test_acc"]
#
#
#device = device("cuda") if is_available() else device("cpu")
#
#if GPUS is not None and is_available():
#    set_device(GPUS[0])
#
#RESULTS["train_cnf_matrix"] = module.confusion_matrix(data_module.train_eval_set, device).tolist()
#RESULTS["val_cnf_matrix"] = module.confusion_matrix(data_module.val_eval_set, device).tolist()
#RESULTS["test_cnf_matrix"] = module.confusion_matrix(data_module.test_eval_set, device).tolist()
#
#with open(join(EXPERIMENT_PATH, "results.json"), "w") as fp:
#    dump(RESULTS, fp)
#
#module.plot_datasets(
#    [
#        data_module.train_eval_set,
#        data_module.val_eval_set,
#        data_module.test_eval_set,
#    ],
#    EXPERIMENT_PATH,
#    device
#)
