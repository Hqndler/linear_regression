import pandas as pd
import os.path as path
import sys

def get_data():
    data = pd.read_csv("theta.csv")
    return data["t0"][0], data["t1"][0] 

def predict_price(mileage):
    t0, t1 = get_data()
    return t0 + (t1 * mileage)

def main():
    if not path.exists("theta.csv"):
        print("Value not present. Please train the model.")
        sys.exit()
    mileage = float(input("Input the car mileage : "))
    price = predict_price(mileage)
    print(f"Predicted price for {mileage} = {price}")

if __name__ == "__main__":
    main()