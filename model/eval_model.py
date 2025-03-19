import numpy as np
from data/data_preprocessing import train_test_split

def eval_models_insee(models, value, df, input_size=20, output_size=5):
    X_train, y_train, X_test, y_test = train_test_split(df, value, 0.6, input_size, output_size)
    x_test = torch.Tensor(np.array(X_test)).unsqueeze(-1).to(device)
    x_test = (x_test - x_test.mean(dim=0))/x_test.std(dim=0)
    input_size = int(x_test.shape[1])
    print(input_size)
    res = []
    for m in range(len(models)):
        result = models[m](x_test)
        print(result.shape)
        res.append(result)
    return res



def error_insee(res, value,df,input_size=20,output_size=5):
    X_train,y_train,X_test,y_test = train_test_split(df,value,0.6,input_size,output_size)
    y_test = np.array(y_test)
    input_size = int(y_test.shape[1])
    output_size = int(y_test.shape[1])
    gt = y_test
    gt = (gt - gt.mean(axis=1, keepdims=True))/gt.std(axis=1, keepdims=True)
    res = np.array([r.cpu().detach().numpy() if isinstance(r, torch.Tensor) else r for r in res])


    # MSE
    mse = np.mean((gt - res[0].squeeze(-1))**2, axis=1)
    std_mse = np.std((gt - res[0].squeeze(-1))**2)
    mse = np.mean(mse)

    # DTW
    dtw_models = np.zeros((len(res), gt.shape[1]))
    for m in range(len(res)):
        for ts in range(gt.shape[1]):
            dist = dtw(gt[0, ts], res[m][ts])
            dtw_models[m][ts] = dist
    std_dtw = np.std(dtw_models, axis=1)
    dtws = np.mean(dtw_models, axis=1)
    print("MSE: {} +- {}".format(np.round(mse,2), np.round(std_mse,2)))
    print("DTW: {} +- {}".format(np.round(dtws,2), np.round(std_dtw,2)))