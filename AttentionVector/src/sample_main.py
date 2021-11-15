import torch
import sample_register_custom_dataset as add_balloon
import sample_train
import sample_inference


def run():
    # register balloon dataset
    print('register dataset')
    balloon_meta = add_balloon.register()

    # train with balloon dataset
    print('train')
    cfg = sample_train.run()

    print('inference')
    pre = sample_inference.run(cfg, balloon_meta, True)

    # print("evaluate")
    # sample_evaluate.run(cfg, pre)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    run()

