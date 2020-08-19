import logging
import pickle
import os
import torch
from hred import hred
from test_model import chat,test
from build_dataset import DailyDataset


def create_log(logfilename):
    logger = logging.getLogger('logger')
    format = '%(message)s'
    handler = logging.FileHandler(filename=logfilename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)



def run_test(model_dir,data_dir,mode,beam_width = 10):

    test_dir = model_dir + 'test'
    os.makedirs(test_dir, exist_ok=True)

    log_filename = test_dir + '/test_data.log'
    checkpoint_path = model_dir + 'pkl.tar'

    create_log(log_filename)
    vocab_path = os.path.join(data_dir,'vocab.pkl')
    data_pre = data_dir + 'test'

    with open(vocab_path,'rb') as f:
        vocab = pickle.load(f)


    n_words = len(vocab['wtoi'])
    checkpoint = torch.load(checkpoint_path)
    hparams = checkpoint['hparams']
    hparams['beam_width'] = beam_width

    print('Building model...')
    model = hred(hparams=hparams, n_words=n_words,
                 itfloss_weights=None).cuda()
    model.load_state_dict(checkpoint['model'])

    if mode == 'test':
        print("Loading test dataset...")
        dataset = DailyDataset(hparams,data_pre,vocab)

        print('Inference utterences ...')
        test(hparams,model,dataset,
         os.path.join(os.path.dirname(checkpoint_path),
                      'inf.'+os.path.basename(checkpoint_path)))
    elif mode == 'chat':
        print('Chatting with bot...')
        chat(hparams,model,vocab)
    else:
        raise ValueError('Unknown mode !')

if __name__ == '__main__':
    os.chdir('/export/home2/NoCsBack/hci/mingxiao/HRED')

    run_test('./Cornell_models/','./Data/Cornell_movie_dialogs/',mode = 'chat')