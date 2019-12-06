"""
Author: Miles Bennett
Main File to Run Simulation and Process results
"""
import numpy as np
import matplotlib.pyplot as plt
import algorithms as algs
import generate_data as gen_data

if __name__ == "__main__":
    print("*************")
    print("1D Analysis")
    print("*************")
    algs.compare_1d_results()
    print("*************")
    print("2D Analysis")
    print("*************")
    algs.compare_2d_results()
    plt.show()