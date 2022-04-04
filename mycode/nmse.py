# calculate NMSE
def calc_NMSE(x_hat,x_test,T=10,pow_diff=None):
    results = {
                "mse_truncated": [0]*T,
                "nmse_truncated": [0]*T,
                "mse_full": [0]*T,
                "nmse_full": [0]*T
              }

    if T == 1:
        x_test_real = np.reshape(x_test_pre[:, 0, :, :], (len(x_test), -1))
        x_test_imag = np.reshape(x_test_pre[:, 1, :, :], (len(x_test), -1))
        x_test_temp = x_test_real + 1j*(x_test_imag)
        x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
        x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
        x_hat_temp = x_hat_real + 1j*(x_hat_imag)
    else:
        x_test_real = np.reshape(x_test_pre[:, :,0, :, :], (x_test.shape[0]*x_test.shape[1], -1))
        x_test_imag = np.reshape(x_test_pre[:, :,1, :, :], (x_test.shape[0]*x_test.shape[1], -1))
        x_test_temp = x_test_real + 1j*(x_test_imag)
        x_hat_real = np.reshape(x_hat[:, :,0, :, :], (x_hat.shape[0]*x_hat.shape[1], -1))
        x_hat_imag = np.reshape(x_hat[:, :,1, :, :], (x_hat.shape[0]*x_hat.shape[1], -1))
        x_hat_temp = x_hat_real + 1j*(x_hat_imag)

    power = np.sum(abs(x_test_temp)**2, axis=1)
    mse = np.sum(abs(x_test_temp-x_hat_temp)**2, axis=1)
    temp = mse/power

    #mse = mse[np.nonzero(power)] 
    #pow_diff_temp = pow_diff_temp[np.nonzero(power)] if type(pow_diff) != type(None) else None
    #power = power[np.nonzero(power)] 
    #temp = mse[np.nonzero(power)] / power[np.nonzero(power)]

    results["avg_mse_truncated"] = np.mean(mse)
    results["avg_nmse_truncated"] = 10*math.log10(np.mean(temp))
    print(f"Average Truncated | NMSE = {results['avg_nmse_truncated']} | MSE = {results['avg_mse_truncated']:.4E}")
    
    if T != 1:
        for t in range(T):
            x_test_temp =  np.reshape(x_test_temp[:, t, :, :], (len(x_test_temp[:, t, :, :]), -1))
            x_hat_temp =  np.reshape(x_hat_temp[:, t, :, :], (len(x_hat_temp[:, t, :, :]), -1))
            power = np.sum(abs(x_test_temp)**2, axis=1)
            mse = np.sum(abs(x_test_temp-x_hat_temp)**2, axis=1)
            temp = mse/power

            #mse = mse[np.nonzero(power)] 
            #power = power[np.nonzero(power)] 
            #temp = mse/power

            results["mse_truncated"][t] = np.mean(mse) 
            results["nmse_truncated"][t] = 10*math.log10(np.mean(temp))
            if type(pow_diff) != type(None):
                temp = (mse + pow_diff_temp) / (power + pow_diff_temp)
                results["mse_full"][t] = np.mean(mse) 
                results["nmse_full"][t] = 10*math.log10(np.mean(temp))
                print(f"t{t+1} | Truncated NMSE = {results['nmse_truncated'][t]} | Full NMSE = {results['nmse_full'][t]}")
            else:
                print(f"t{t+1} | Truncated NMSE = {results['nmse_truncated'][t]}")
    return results