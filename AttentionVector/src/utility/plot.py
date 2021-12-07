# source: https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b#file-lossevalhook-py
import json
import matplotlib.pyplot as plt

experiment_folder = '../output'


def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def run():
    experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')

    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
        [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
    plt.legend(['total_loss', 'validation_loss'], loc='upper left')
    plt.show()

    experiment_metrics.reverse()
    for value in experiment_metrics:
        if 'total_loss' in value:
            print(value['iteration'], value['total_loss'])
            break

    for value in experiment_metrics:
        if 'validation_loss' in value:
            print(value['iteration'], value['validation_loss'])
            break


if __name__ == '__main__':
    run()
