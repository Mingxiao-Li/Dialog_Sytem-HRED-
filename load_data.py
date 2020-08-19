import os
import sys
import logging
from datetime import datetime
from utils import loginfo_and_print,build_vocab

os.makedirs("./log",exist_ok = True)
logfilename = "./log/load_dailydialog_{}.log".format(datetime.now().strftime("%Y%m%d%H%M%S"))
format = "%(message)s"
logger = logging.getLogger("logger")
handler = logging.FileHandler(filename=logfilename)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(format))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

def load_data(in_dir,data):
    if data == "train":
        dial_path = os.path.join(in_dir,"train.txt")
    elif data == "validation":
        dial_path = os.path.join(in_dir,"validation.txt")
    elif data == "test":
        dial_path = os.path.join(in_dir,"test.txt")
    else:
        raise ValueError("Cannot find file")

    dial_list = []
    with open(dial_path,"r") as f:
        for line_dial in f:
            dial = line_dial.split("__eou__")[:-1]
            tmp = []
            for uttr in dial:
                if uttr[0] == " ":
                    uttr = uttr[1:]
                if uttr[-1] == " ":
                    uttr = uttr[:-1]
                tmp.append(uttr)
            dial_list.append(tmp)
    return dial_list,data

def extend_dial(dial_list):
    extend_dial_list = []
    for d in dial_list:
        for i in range(2,len(d) + 1):
            extend_dial_list += [d[:i]]
    return extend_dial_list

def save_data(out_dir,prefix,dial_list):
    out_dial_path = os.path.join(out_dir,prefix+"_text.txt")
    with open(out_dial_path,"w") as f:
        for dial in dial_list:
            f.write("<dial>\n")
            for uttr in dial:
                f.write(uttr + "\n")
            f.write("</dial>\n")

if __name__ == "__main__":
    out_dir = "./Data/DailyDialogues/"
    in_dir = "./Data/DailyDialogues/"
    os.makedirs(out_dir,exist_ok=True)
    dial_list, prefix = load_data(in_dir,"test")
    loginfo_and_print(logger,"Load {} dialogues from {}".format(len(dial_list),in_dir))
    if prefix == "train":
        build_vocab(out_dir,dial_list)
    print("Finish building vocab")
    dial_list = extend_dial(dial_list)
    loginfo_and_print(logger,"Extend to {} dialogues".format(len(dial_list)))
    save_data(out_dir,prefix,dial_list)
    print("Data saved successfully !!")
