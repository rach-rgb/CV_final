import torch
import sample_config
import sample_train
import sample_inference


def run():
    # config
    cfg = sample_config.run()

    # train
    sample_train.run(cfg)

    # print('inference')
    # pre = sample_inference.run(cfg, balloon_meta, True)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    run()

