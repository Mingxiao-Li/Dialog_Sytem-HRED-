import logging
import os
import pickle
import torch
from math import sqrt
from datetime import datetime
from hred import hred
from train import run_epochs
from test_model import chat,test
from build_dataset import DailyDataset
from utils import build_vocab


os.makedirs("./Ubuntu_models/log",exist_ok = True)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
logfilename = "./Ubuntu_models/log/train_{}.log".format(datetime.now().strftime("%Y%m%d%H%M%S"))
format = "%(message)s"
logger = logging.getLogger("logger")
handler = logging.FileHandler(filename=logfilename)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(format))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    hparams = {
        "PAD_id":0,
        "SOS_id":1,
        "EOS_id":2,
        "UNK_id":3,
        "MAX_DIAL_LEN":4,
        "MAX_UTTR_LEN":150,
        "hidden_size":256,
        "num_layers":2,
        "batch_size":2,
        "max_epoch":40,
        "max_gradient":50.0,
        "learning_rate":1e-4,
        "decoder_learning_ratio":5.0,
        "decay_step":6000,
        "lr_decay":1/sqrt(3),
        "dropout":0.1,
        "teacher_forcing_ratio":1.0,
        "mmi_lambda":0.4,
        "mmi_gamma":10,
        "itf_lambda":0.4,
        "l2_pooling":"True",
        "beam_width":10,
        "suppress_lambda":1.0,
        "len_alpha":0.6,
        "loss":None,
        "accumulate_step":50,
    }
    data_dir = "./Data/Ubuntu/"
    vocab_path = os.path.join(data_dir,"vocab.pkl")
    valid_every = 100
    save_every = 1
    checkpoint_path = False
    fix_embedding = "False"
    model_pre = "./Ubuntu_models/pkl"
    mode = "train"


    logger.info("Data directory: {}".format(data_dir))
    logger.info("Vocabulary file: {}".format(vocab_path))

    os.makedirs(os.path.dirname(model_pre),exist_ok=True)

    with open(vocab_path,"rb") as f:
        vocab = pickle.load(f)
    n_words = len(vocab["wtoi"])

    if mode != "train":
        assert checkpoint_path is not None


    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = None

    if checkpoint:
        hparams = checkpoint["hparams"]
        hparams["beam_width"] = 10

        for k,v in hparams.items():
            logger.info("{}: {}".format(k,v))

    if mode == "train":
        data_pre = data_dir+"ub_valid"
        print("Loading valid dataset...")
        valid_dataset = DailyDataset(hparams,data_pre,vocab)

        data_pre = data_dir + "ub_train"
    else:
        data_pre = data_dir + "ub_test"
    print("Loading dataset...")
    dataset = DailyDataset(hparams,data_pre,vocab)
    if hparams["loss"] == "itf":
        itfloss_weight = dataset.itfloos_weight
    else:
        itfloss_weight = None

    print("Building model...")
    model = hred(hparams = hparams,n_words = n_words,
                 itfloss_weights = itfloss_weight,
                 fix_embedding = fix_embedding).cuda()

    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    print("Model built and ready to go !")

    if mode == "train":

        print("Training model...")
        print(torch.cuda.current_device())
        run_epochs(hparams = hparams,
                   model = model,
                   dataset = dataset,
                   valid_dataset = valid_dataset,
                   model_pre = model_pre,
                   valid_every = valid_every,
                   save_every = save_every,
                   )
    elif mode == "inference":
        print("Inference utterences...")
        test(
            hparams,model,dataset,
            os.path.join(
                os.path.dirname(checkpoint_path),
                "inf."+os.path.basename(checkpoint_path))
        )

    elif mode == "chat":
        print("Chatting with bot...")
        chat(hparams,model,vocab)
    else:
        raise ValueError("Unknown mode !")

    print('Done')



