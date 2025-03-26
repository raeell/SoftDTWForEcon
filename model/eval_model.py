import torch
import numpy as np
from tslearn.metrics import dtw


from data.data_preprocessing import (
    train_test_val_split,
    to_tensor_and_normalize,
    to_array_and_normalize,
)


def eval_models_insee(
    models,
    value,
    df,
    device,
    split_train=0.6,
    split_val=0.2,
    input_size=20,
    output_size=5,
):
    X_train, y_train, X_val, y_val, X_test, y_test = train_test_val_split(
        df, value, split_train, split_val, input_size, output_size
    )
    X_test = to_tensor_and_normalize(X_test).to(device)
    res = []
    for m in range(len(models)):
        result = models[m](X_test)
        res.append(result)
    return res


def error_insee(
    res, value, df, device, split_train=0.6, split_val=0.2, input_size=20, output_size=5
):
    X_train, y_train, X_val, y_val, X_test, y_test = train_test_val_split(
        df, value, split_train, split_val, input_size, output_size
    )
    gt = to_array_and_normalize(y_test)
    res = np.array(
        [r.cpu().detach().numpy() if isinstance(r, torch.Tensor) else r for r in res]
    )

    # MSE
    mse = np.mean((gt - res[0].squeeze(-1)) ** 2, axis=1)
    std_mse = np.std((gt - res[0].squeeze(-1)) ** 2)
    mse = np.mean(mse)

    # DTW
    dtw_models = np.zeros((len(res), gt.shape[1]))
    for m in range(len(res)):
        for ts in range(gt.shape[1]):
            dist = dtw(gt[0, ts], res[m][ts])
            dtw_models[m][ts] = dist
    std_dtw = np.std(dtw_models, axis=1)
    dtws = np.mean(dtw_models, axis=1)
    print("MSE: {} +- {}".format(np.round(mse, 2), np.round(std_mse, 2)))
    print("DTW: {} +- {}".format(np.round(dtws, 2), np.round(std_dtw, 2)))
