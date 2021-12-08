import os
import pandas as pd
import matplotlib.pyplot as plt

input_path = './output/'
result_path = './output/AP.csv'


if __name__ == "__main__":
    os.chdir('../')

    result_df = pd.read_csv(result_path, index_col=0)
    result_df = result_df[['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']].transpose()

    result_df.plot.bar()
    plt.show()
