import torch
import logging
import tqdm
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from utils import loginfo_and_print, collate_fn
from batch_sampler import RandomBatchSampler

logger = logging.getLogger("logger").getChild("train")

def run_epochs(hparams, model, dataset, valid_dataset, model_pre,
               valid_every = 1000, save_every =1, checkpoint = None,
               pretrained=False):

    learning_rate = hparams["learning_rate"]
    decoder_learning_ratio = hparams["decoder_learning_ratio"]
    decay_step = hparams["decay_step"]
    lr_decay = hparams["lr_decay"]
    max_epoch = hparams["max_epoch"]
    batch_size = hparams["batch_size"]
    max_gradient = hparams["max_gradient"]
    accumulate_step = hparams['accumulate_step']

    model.train()

    encoder_optimizer = optim.Adam(model.encoder.parameters(),lr=learning_rate)
    encoder_scheduler = optim.lr_scheduler.StepLR(encoder_optimizer,step_size=decay_step,gamma=lr_decay)
    decoder_optimizer = optim.Adam(model.decoder.parameters(),lr=learning_rate*decoder_learning_ratio)
    decoder_scheduler = optim.lr_scheduler.StepLR(decoder_optimizer,step_size=decay_step,gamma=lr_decay)
    context_optimizer = optim.Adam(model.context.parameters(),lr = learning_rate)
    context_scheduler = optim.lr_scheduler.StepLR(context_optimizer,step_size=decay_step,gamma=lr_decay)

    start_epoch = 1
    loginfo_and_print(logger,
                      "Valid (Epoch{}):{:.4f}".format(start_epoch-1,valid(model,valid_dataset,batch_size,hparams)))
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=collate_fn, drop_last=True,
        num_workers=4, sampler=RandomBatchSampler(dataset, batch_size))

    for epoch in range(start_epoch, max_epoch + 1):
        pbar = tqdm.tqdm(enumerate(dataloader),total=len(dataloader))
        loss = 0
        for step,data in pbar:
            loss += iteration(
                model,data,step,accumulate_step,
                encoder_optimizer, context_optimizer,
                decoder_optimizer,max_gradient
            )
            pbar.set_description("Train (Epoch{}):{:.4f}".format(epoch,loss/(step+1)))

            if step % accumulate_step == 0:
                encoder_scheduler.step()
                decoder_scheduler.step()
                context_scheduler.step()

            if (step + 1) % valid_every == 0:
                logger.info("Valid: {:.4f}".format(valid(model,valid_dataset,batch_size,hparams)))

        logger.info("Train (Epoch {}):{:.4f}".format(epoch, loss/(step+1)))
        loginfo_and_print(logger,"Valid (Epoch {}):{:.4f}".format(epoch,valid(
            model, valid_dataset,batch_size,hparams
        )))

        if epoch % save_every == 0:
            print("Saving model...")
            torch.save({
                "epoch": epoch,
                "hparams": hparams,
                "model": model.state_dict(),
                "en_opt": encoder_optimizer.state_dict(),
                "cn_opt": context_optimizer.state_dict(),
                "de_opt": decoder_optimizer.state_dict(),
                "en_sch": encoder_scheduler.state_dict(),
                "cn_sch": context_scheduler.state_dict(),
                "de_sch": decoder_scheduler.state_dict()
            }, model_pre+"_{}.tar".format(epoch))
            print("Model is saved successfully!!")
        dataset.init_epoch(epoch + 1)

def valid(model, valid_dataset, batch_size, hparams):

    model.eval()
    dataloader = DataLoader(
        valid_dataset,batch_size=batch_size,
        collate_fn=collate_fn,drop_last=True,
        num_workers=4,sampler=RandomBatchSampler(valid_dataset,batch_size)
    )
    loss = 0
    for idx, data in enumerate(dataloader):
        loss += iteration(model,data,None,None)
    model.train()
    return loss/(idx+1)

def iteration(model, data,step,accumulate_step, encoder_optimizer = None,context_optimizer = None,
              decoder_optimizer = None, max_gradient = None):

    with torch.set_grad_enabled(model.training):
        loss, print_loss = model(data)

        if model.training:
            loss = loss / accumulate_step
            loss.backward()

            if step % accumulate_step == 0:

                _ = clip_grad_norm_(model.encoder.parameters(), max_gradient)
                _ = clip_grad_norm_(model.decoder.parameters(), max_gradient)
                _ = clip_grad_norm_(model.context.parameters(), max_gradient)

                encoder_optimizer.step()
                decoder_optimizer.step()
                context_optimizer.step()


                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                context_optimizer.zero_grad()
    return print_loss
