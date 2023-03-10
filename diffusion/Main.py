from Diffusion.Train import train, eval


def main(model_config = None):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 200,
        "batch_size": 20,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 16 * 1024,
        "grad_clip": 1.,
        "device": "cuda",
        "training_load_weight": "ckpt_150_1.pt",
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_570_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "nrow": 8
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
