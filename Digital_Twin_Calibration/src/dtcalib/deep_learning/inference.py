from pathlib import Path
import pandas as pd
from model import RCInverseCNN
from calibration import RCNeuralCalibrator

def main():
    cal = RCNeuralCalibrator.load(Path("models/rc_inverse_best_xxx.pth"))

    df = pd.read_csv("some_experiment.csv")
    vin = df.iloc[:, 1].values
    vout = df.iloc[:, 2].values

    c_hat = cal.predict(vin, vout)
    print("C_hat =", c_hat)

if __name__ == "__main__":
    main()