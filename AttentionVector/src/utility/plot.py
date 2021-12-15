import json
import matplotlib.pyplot as plt


# load metrics file
def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


# plot train loss and validation loss
# edit source from: https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b#file-lossevalhook-py
def run(input_path):
    experiment_metrics = load_json_arr(input_path + '/metrics.json')

    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'total_loss' in x],
        [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
    plt.legend(['total_loss', 'validation_loss'], loc='upper left')
    plt.show()

    experiment_metrics.reverse()

    # print each loss at last checkpoint
    for value in experiment_metrics:
        if 'total_loss' in value:
            print(value['iteration'], value['total_loss'])
            break

    for value in experiment_metrics:
        if 'validation_loss' in value:
            print(value['iteration'], value['validation_loss'])
            break


if __name__ == '__main__':
    run('../output')
