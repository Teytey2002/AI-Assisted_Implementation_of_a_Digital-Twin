from pathlib import Path
import pandas as pd
from model import RCInverseCNN
from calibration import RCNeuralCalibrator

"""
Lors de l’inférence, il faut :

sauvegarder ces stats
les recharger
dénormaliser la sortie :

C = ypred * ystd + 𝑦𝑚𝑒𝑎𝑛 
"""
def load_model(path):
    model = RCInverseCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict(model, x):
    with torch.no_grad():
        return model(x.unsqueeze(0)).item()

def main():
    cal = RCNeuralCalibrator.load(Path("models/rc_inverse_best_xxx.pth"))

    df = pd.read_csv("some_experiment.csv")
    vin = df.iloc[:, 1].values
    vout = df.iloc[:, 2].values

    c_hat = cal.predict(vin, vout)
    print("C_hat =", c_hat)

if __name__ == "__main__":
    main()