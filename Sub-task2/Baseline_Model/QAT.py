"""
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2022, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2022.04.20.
"""

from dataset import ETRIDataset_color
from networks import *

import pandas as pd
import os
import argparse
import time

import torch
import torch.utils.data
import torch.utils.data.distributed

import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# import wandb

import random
import numpy as np

import copy

# Set up warnings
import warnings

warnings.filterwarnings(action="ignore", category=DeprecationWarning, module=r".*")
warnings.filterwarnings(action="default", module=r"torch.quantization")

# wandb.init(name="sub-task2-baseline", project="Fashion-How", entity="hayeon0808")

parser = argparse.ArgumentParser()

parser.add_argument("--version", type=str, default="QAT_ResNet_color")
parser.add_argument(
    "--epochs", default=5, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--lr", default=0.0001, type=float, metavar="N", help="learning rate"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=64,
    type=int,
    metavar="N",
    help="mini-batch size (default: 64), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--seed", default=42, type=int, help="seed for initializing training. "
)

a = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_metrics(y_true, y_pred, verbose=True):
    """
    :return: asca, pre, rec, spe, f1_ma, f1_mi, g_ma, g_mi
    """
    y_true, y_pred = y_true.astype(np.int8), y_pred.astype(np.int8)

    cnf_matrix = confusion_matrix(y_true, y_pred)
    if verbose:
        print(cnf_matrix)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    top_1 = np.sum(TP) / np.sum(np.sum(cnf_matrix))
    cs_accuracy = TP / cnf_matrix.sum(axis=1)

    return top_1, cs_accuracy.mean()


# 모델 평가
def validation(trained_model, val_dataloader, step):
    trained_model.eval()  # 평가 모드로 설정
    total_loss = 0
    total_samples = 0
    gt_list = np.array([])
    pred_list = np.array([])

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    with torch.no_grad():  # gradient 계산 비활성화
        for sample in tqdm(val_dataloader):
            for key in sample:
                sample[key] = sample[key].to(DEVICE)

            out = trained_model(sample["image"])
            loss = criterion(out, sample["color_label"])
            total_loss += loss.item() * sample["color_label"].size(0)
            total_samples += sample["color_label"].size(0)

            gt = sample["color_label"].cpu().numpy()
            gt_list = np.concatenate([gt_list, gt], axis=0)

            _, indx = out.max(1)
            pred_list = np.concatenate([pred_list, indx.cpu().numpy()], axis=0)

    avg_loss = total_loss / total_samples
    top_1, acsa = get_test_metrics(gt_list, pred_list)

    print("------------------------------------------------------")
    print(
        f"step {step}: Validation Loss: {avg_loss:.5f}, Top-1: {top_1:.5f}, ACSA: {acsa:.5f}"
    )
    print("------------------------------------------------------")

    return avg_loss, top_1, acsa


# 양자화된 모델 평가
def quantization_evaluation(
    trained_model, val_dataloader, path, step: int = None, epoch: int = None
):
    net_q = copy.deepcopy(trained_model)

    # 양자화된 모델을 평가하기 위해 eval 모드로 설정 후 Quantization 모델로 변환
    net_q.cpu().eval()
    net_q = torch.quantization.convert(net_q, inplace=True)

    total_loss = 0
    total_samples = 0

    gt_list = np.array([])
    pred_list = np.array([])

    criterion = nn.CrossEntropyLoss().to("cpu")

    with torch.no_grad():
        for sample in tqdm(val_dataloader):
            for key in sample:
                sample[key] = sample[key].to("cpu")  # 양자화 모델은 CPU에서 실행

            out = net_q(sample["image"])

            loss = criterion(out, sample["color_label"])

            total_loss += loss.item() * sample["color_label"].size(0)
            total_samples += sample["color_label"].size(0)

            gt = sample["color_label"].numpy()
            gt_list = np.concatenate([gt_list, gt], axis=0)

            _, indx = out.max(1)
            pred_list = np.concatenate([pred_list, indx.numpy()], axis=0)

    avg_loss = total_loss / total_samples
    top_1, acsa = get_test_metrics(gt_list, pred_list)
    # wandb.log({"val_loss": avg_loss, "val_top_1": top_1, "val_acsa": acsa}, step=epoch)
    print("---------------------------------------------------------------------")
    print(
        f"Step {step}: Quantized Validation Loss: {avg_loss:.5f}, Top-1: {top_1:.5f}, ACSA: {acsa:.5f}"
    )
    print("---------------------------------------------------------------------")

    if epoch == None:
        torch.save(net_q.state_dict(), path + f"/q_RN_step_{step}.pt")
    else:
        # 양자화된 모델 저장
        torch.save(net_q.state_dict(), path + f"/q_RN_e_{epoch}.pt")


def main():
    """The main function for model training."""

    # fix the seed for reproducibility
    seed = a.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmarks = False

    if os.path.exists(f"./model/{a.version}") is False:
        os.makedirs(f"./model/{a.version}")

    save_path = "model/" + a.version

    """ Quantizable ResNet50 """
    net_int = Quantizable_ResNet_color("50")

    net_int.eval()

    # quantization configuration 설정
    net_int.qconfig = torch.quantization.get_default_qconfig("x86")  # default qconfig
    # net_int.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # fuse_model을 통해 Conv2d, BatchNorm2d, ReLU 등을 하나의 연산으로 통합
    net_int.fuse_model()

    # QAT(Quantization Aware Training)을 위한 준비
    net_int.train()
    net_int = torch.quantization.prepare_qat(net_int, inplace=True)

    df = pd.read_csv("./Dataset/Fashion-How24_sub2_train_aug.csv")

    train_dataset = ETRIDataset_color(df, base_path="./Dataset/train+val/")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=a.batch_size, shuffle=True, num_workers=2
    )

    valid_dataset = ETRIDataset_color(
        pd.read_csv("./Dataset/Fashion-How24_sub2_val_aug.csv"),
        base_path="./Dataset/train+val/",
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=64, shuffle=False, num_workers=2
    )

    optimizer = torch.optim.Adam(net_int.parameters(), lr=a.lr)

    # Classification Loss
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    net_int = net_int.to(DEVICE)

    total_step = len(train_dataloader)
    step = 0
    t0 = time.time()

    for epoch in range(a.epochs):
        print(f"Epoch [{epoch + 1}/{a.epochs}]")
        for i, sample in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            step += 1
            sample = {key: sample[key].to(DEVICE) for key in sample}

            out = net_int(sample["image"])

            loss = criterion(out, sample["color_label"])

            loss.backward()
            optimizer.step()

            # 10 step마다 loss 출력 및 모델 저장
            if (i + 1) % 10 == 0:
                print(
                    "\n Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time : {:2.3f}".format(
                        epoch + 1,
                        a.epochs,
                        i + 1,
                        total_step,
                        loss.item(),
                        time.time() - t0,
                    )
                )
                # wandb.log({"loss": loss.item()}, step=(i + 1) + 170 * epoch)
                quantization_evaluation(
                    net_int, valid_dataloader, step=i + 1 + 170 * epoch, path=save_path
                )
                t0 = time.time()

        quantization_evaluation(
            net_int, valid_dataloader, step=(epoch + 1) * 170, path=save_path
        )
        net_int = net_int.to(DEVICE)
        net_int.train()

        if (epoch + 1) % 10 == 0:
            a.lr *= 0.9
            optimizer = torch.optim.Adam(net_int.parameters(), lr=a.lr)
            print("learning rate is decayed")


if __name__ == "__main__":
    main()
