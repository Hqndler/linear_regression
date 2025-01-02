import sys, os
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

class LinearRegression:
    t0 = 0
    t1 = 0
    lr = 0.1
    min_k = 0
    min_p = 0
    max_k = 0
    max_p = 0

    def __str__(self):
        return f"t0 = {self.t0:.3f}, t1 = {self.t1:.8f}, learning rate = {self.lr}"

def estimate_price(mileage : float):
    return lr.t0 + (lr.t1 * mileage)

def get_data() -> dict:
    data : dict = pd.read_csv("data.csv")
    km = data['km']
    price = data['price']
    
    if len(price) != len(km):
        print("Something went wrong with the csv file. Please check all line are populated")
    return km, price

def gradient0(km : list, price : list) -> float:
    return lr.lr * sum(estimate_price(km) - price) / len(km)

def gradient1(km : list, price : list) -> float:
    return lr.lr * (sum((estimate_price(km) - price) * km) / len(km))

def standardize(km, price):
    lr.min_k, lr.max_k = min(km), max(km)
    lr.min_p, lr.max_p = min(price), max(price)

    km = (km - lr.min_k) / (lr.max_k - lr.min_k)
    price = (price - lr.min_p) / (lr.max_p - lr.min_p)

    return km, price

def main():
    k, p = get_data()
    km, price = standardize(*get_data())
    epoch = 0

    plt.ion()
    plt.plot(k, p, "og")
    plt.xticks(range(0,300_000, 50_000))
    plt.yticks(range(3000, 9000, 500))
    plt.xlim(0,300_000)
    plt.ylim(3000,9000)
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    line, = plt.plot(k, [6000] * len(k), color="red", label="Prediction") # visual purpose
    plt.legend()
    plt.title("Linear Regression - Waiting to train")
    plt.draw()
    plt.show()
    input()

    while epoch != 3000:
        lr.t0 -= gradient0(km, price)
        lr.t1 -= gradient1(km, price)
        epoch += 1
        if epoch % 100 == 0:
            tt0 = lr.t0
            tt1 = lr.t1
            lr.t1 = (lr.max_p - lr.min_p) * lr.t1 / (lr.max_k - lr.min_k)
            lr.t0 = lr.min_p + ((lr.max_p - lr.min_p) * lr.t0) + lr.t1 * (1 - lr.min_k)
            print(lr, f"epoch = {epoch}")
            plt.title(f"Epoch {epoch}/3000")
            line.set_ydata([estimate_price(i) for i in k])
            plt.draw()
            plt.show()
            input()
            lr.t0 = tt0
            lr.t1 = tt1

    lr.t1 = (lr.max_p - lr.min_p) * lr.t1 / (lr.max_k - lr.min_k)
    lr.t0 = lr.min_p + ((lr.max_p - lr.min_p) * lr.t0) + lr.t1 * (1 - lr.min_k)

    plt.ioff()
    plt.show()

    print(lr)
    
    std_km, std_price = get_data()
    mse = sqrt(sum(((estimate_price(std_km) - std_price) ** 2)) / len(std_km))
    print(f"Precision : +/- {mse:.3f}")

    with open("theta.csv", "w") as file:
        file.write(f"t0,t1,precision\n{lr.t0},{lr.t1},{mse}")

if __name__ == "__main__":
    if not os.path.exists("data.csv"):
        print("data.csv not found. Terminating...")
        sys.exit()
    lr = LinearRegression()
    main()