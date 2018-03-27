# source \virtualenv\TFmetrics\bin\activate

"""datamaker.py: Creates new file structures with symbolic links for use with Keras' ImageDataGenerator for example
. DISCLAIMER: Running this program might delete your files, use with care.
No warranty is expressed or implied and no technical support will be offered."""

__authors__ = ["Tor Jan Derek Berstad"]
__copyright__ = "Tor Jan Derek Berstad"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Tor Jan Derek Berstad"
__email__ = "tjbersta@ifi.uio.no"
__status__ = "Development"
# This file is subject to the terms and conditions defined in
# file 'LICENSE.md', which is part of this source code package.


import os
import platform
import argparse
import random
import sys
import shutil
import json


def walkerror(error):
    print(error)


def osdetect():
    osname = os.name
    env = platform.system()
    version = platform.release()

    return osname, env, version


def create_symlink(oldpath, newpath, env, verbose, dry):
    if verbose:
        print("Making symbolic link from", oldpath, "to", newpath)
    if not dry:
        if env == "Windows":
            try:
                kdll = ctypes.windll.LoadLibrary("kernel32.dll")
                kdll.CreateSymbolicLinkW(newpath, oldpath, 1)
            except OSError:
                print("OSError: ", sys.exc_info()[0], "\n During linking of:", oldpath, "->", newpath)
        elif env == "Linux" or "Darwin":
            try:
                os.symlink(oldpath, newpath)
            except OSError:
                print("OSError: ", sys.exc_info()[0], "\n During linking of:", oldpath, "->", newpath)
        else:
            print("Unsupported environment!")
            sys.exit(1)


def get_class_dirs(dataset_dir):
    classes_from_directories = []  # to determine the classes from the root folder structure automatically
    root_dir = os.path.dirname(os.path.abspath(__file__)) + dataset_dir
    su = []
    for directory, subdirectories, files in os.walk(root_dir, onerror=walkerror):
        subdirectories.sort()
        su.append(directory)
        for subdirectory in subdirectories:
            classes_from_directories.append(subdirectory)
    return su, classes_from_directories


def make_class_folder(name, completedir, binary, verbose, subdir = ""):
    if verbose:
        print("Making folder: ", completedir+subdir+"/"+name)
    if not args_d["dry"]:
        os.makedirs(completedir+subdir+"/"+name)
    if binary:
        if verbose:
            print("Making folder: ", completedir+subdir+"/"+name+"/"+"positive")
        if not args_d["dry"]:
            os.makedirs(completedir+subdir+"/"+name+"/"+"positive")
        if verbose:
            print("Making folder: ", completedir+subdir+"/"+name+"/"+"negative")
        if not args_d["dry"]:
            os.makedirs(completedir+subdir+"/"+name+"/"+"negative")


def make_folders(args_d,classnames, outp):
    completedir = os.path.dirname(os.path.abspath(__file__)) + outp
    verbose = args_d["verbose"]
    if os.path.exists(completedir):
        print("Warning: Folder already exists!")
        if not args_d["yes"]:
            keyin = input("Are you sure you want to overwrite what it contains " +
                          "(This will DELETE the contents first)? (y) ")
            if "y" not in keyin.lower():
                sys.exit(1)
        for the_file in os.listdir(completedir):
            file_path = os.path.join(completedir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
                sys.exit(1)
    if verbose:
        print("Making folder: ", completedir)
    os.makedirs(completedir, exist_ok=True)
    if args_d["folds"]:
        for fold in range(args_d["folds"]):
            if verbose:
                print("Making folder: ", completedir+str(fold))
            if not args_d["dry"]:
                os.makedirs(completedir+str(fold))
            for name in classnames:
                make_class_folder(name, completedir, args_d["binary"], verbose, subdir=str(fold))

    else:
        if args_d["train"]:
            if verbose:
                print("Making folder: ", completedir+"train")
            if not args_d["dry"]:
                os.makedirs(completedir+"train")
            for name in classnames:
                make_class_folder(name,completedir,args_d["binary"], verbose, subdir="train")
        if args_d["validation"]:
            if verbose:
                print("Making folder: ", completedir+"validation")
            if not args_d["dry"]:
                os.makedirs(completedir+"validation")
            for name in classnames:
                make_class_folder(name,completedir,args_d["binary"], verbose, subdir="validation")

        if args_d["test"]:
            if verbose:
                print("Making folder: ", completedir+"test")
            if not args_d["dry"]:
                os.makedirs(completedir+"test")
            for name in classnames:
                make_class_folder(name,completedir,args_d["binary"], verbose, subdir="test")


def make_links(subdirs,classes, args_d, outp):
    verbose = args_d["verbose"]
    dry = args_d["dry"]
    name, env, version = osdetect()
    outdir = os.path.dirname(os.path.abspath(__file__)) + outp
    for subdir in subdirs[1:]:
        patharr = subdir.split("/")
        cls = patharr[-1]
        if verbose:
            print("Making symlinks for", subdir)
        file_list = os.listdir(subdir)
        random.shuffle(file_list)
        if args_d["folds"]:
            f = 0
            i = 0
            for path in file_list:
                if not args_d["binary"]:
                    create_symlink(subdir + "/" + path, outdir + str(f) + "/" + cls
                                   + "/" + path, env, verbose, dry)
                else:
                    create_symlink(subdir + "/" + path, outdir + str(f) + "/" + cls
                                   + "/positive/" + path, env, verbose, dry)
                    for cls2 in classes:
                        if cls2 != cls:
                            create_symlink(subdir + "/" + path, outdir + str(f) + "/" + cls2
                                           + "/negative/" + path, env, verbose, dry)
                i += 1
                next_checkpoint = (f+1)*(len(file_list)//args_d["folds"])
                if i == next_checkpoint:
                    f += 1
        else:
            train_lim = (len(file_list)//100)*args_d["train"]
            valid_lim = (len(file_list)//100)*args_d["validation"]
            test_lim = (len(file_list)//100)*args_d["test"]
            if verbose:
                print("Percentage of dataset class used: ", ((train_lim+valid_lim+test_lim)//len(file_list))*100)
            i = 0
            for path in file_list:
                if not args_d["binary"]:
                    if i < train_lim:
                        create_symlink(subdir + "/" + path, outdir + "train" + "/" + cls
                                       + "/" + path, env, verbose, dry)
                    elif i < train_lim + valid_lim:
                        create_symlink(subdir + "/" + path, outdir + "validation" + "/" + cls
                                       + "/" + path, env, verbose, dry)
                    elif i < train_lim + valid_lim + test_lim:
                        create_symlink(subdir + "/" + path, outdir + "test" + "/" + cls
                                       + "/" + path, env, verbose, dry)
                else:
                    if i < train_lim:
                        create_symlink(subdir + "/" + path, outdir + "train" + "/" + cls
                                       + "/positive/" + path, env, verbose, dry)
                        for cls2 in classes:
                            if cls2 != cls:
                                create_symlink(subdir + "/" + path, outdir + "train" + "/" + cls2
                                               + "/negative/" + path, env, verbose, dry)
                    elif i < train_lim + valid_lim:
                        create_symlink(subdir + "/" + path, outdir + "validation" + "/" + cls
                                       + "/positive/" + path, env, verbose, dry)
                        for cls2 in classes:
                            if cls2 != cls:
                                create_symlink(subdir + "/" + path, outdir + "validation" + "/" + cls2
                                               + "/negative/" + path, env, verbose, dry)
                    elif i < train_lim + valid_lim + test_lim:
                        create_symlink(subdir + "/" + path, outdir + "test" + "/" + cls
                                       + "/positive/" + path, env, verbose, dry)
                        for cls2 in classes:
                            if cls2 != cls:
                                create_symlink(subdir + "/" + path, outdir + "test" + "/" + cls2
                                               + "/negative/" + path, env, verbose, dry)
                i += 1


def make_all(args_d):
    if args_d["dry"]:
        print("Dry Run Mode, will not actually do anything, best used with verbose")
    osname, env, version = osdetect()
    if args_d["seed"]:
        seed = args_d["seed"]
    else:
        seed = random.randrange(sys.maxsize)
    print("Running with parameters: ", args_d)
    print("Environment: ", osname,", ", env, ", ", version)
    print("Seed: ", seed)
    for dataset in args_d["dataset"]:
        subdirs, classes_from_directories = get_class_dirs(dataset)
        folders = 3
        if not args_d["train"] or args_d["train"] == 0:
            folders -= 1
            args_d["train"] = 0
        if not args_d["validation"] or args_d["validation"] == 0:
            folders -= 1
            args_d["validation"] = 0
        if not args_d["test"] or args_d["test"] == 0:
            folders -= 1
            args_d["test"] = 0
        if args_d["folds"]:
            folders = args_d["folds"]
        folders *= len(classes_from_directories)
        if args_d["binary"]:
            folders *= 3
        folders += 1
        tot_files = 0
        for current in subdirs[1:]:
            tot_files += len(os.listdir(current))
        print(len(subdirs)-1, "classes")
        print(tot_files, "files found")
        if args_d["binary"]:
            tot_files *= len(classes_from_directories)
        if args_d["out"]:
            outp = args_d["out"]
        else:
            outp = dataset
        print("Warning: This operation will create", folders, "new folders containing", tot_files, "new symlinks in",
              outp)
        if not args_d["yes"]:
            keyin = input("Are you sure you want to do this? (y) ")
            if "y" not in keyin.lower():
                sys.exit(1)
            make_folders(args_d, classes_from_directories, outp)
            make_links(subdirs, classes_from_directories, args_d, outp)
    print("Done!")
    if args_d["save"] and not args_d["dry"]:
        print("Saving params to", outp, "as params.json")
        if not args_d["seed"]:
            args_d["seed"] = seed
        with open(os.path.dirname(os.path.abspath(__file__)) + outp + "datamaker_params.json", 'w') as f:
            json.dump(args_d, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates new file structures with symbolic links for use with keras " +
                                                 "ImageDataGenerator for example. " +
                                                 "\nDefault mode: Multiclass mode with a " +
                                                 "50% train, 25% validation, 25% test split")
    parser.add_argument("dataset", metavar ="D", nargs='+',
                        help="A dataset path to make new file structures from, must be a set of " +
                             "images in seperate folders, one per class")
    parser.add_argument("--train", type=int, dest="train", help="The percentage for training data (not used in folds)")
    parser.add_argument("--valid", type=int, dest="validation", help="The percentage for validation data "
                                                                     + "(not used in folds)")
    parser.add_argument("--test", type=int, dest="test", help="The percentage for validation data (not used in folds)")
    parser.add_argument("--folds", type=int, dest="folds", help="Number of folds, divides the data equally, " +
                                                      "train,valid,test parameters are ignored.")
    parser.add_argument('--seed', type=int, dest="seed", help="Seed for the pseudo-random file selection, use this to "
                                                              + "reproduce a selection")
    parser.add_argument('--binary', dest='binary', action='store_const',
                        const=True, default=False,
                        help="Select binary mode, if you use a seed the file selections should be " +
                             "the same as for multiclass (default) mode")
    parser.add_argument('--verbose', dest='verbose', action='store_const',
                        const=True, default=False,
                        help="Print all the things")
    parser.add_argument('--dry', dest='dry', action='store_const',
                        const=True, default=False,
                        help="Dryrun mode, won't actually change anything")
    parser.add_argument('--yes', dest='yes', action='store_const',
                        const=True, default=False,
                        help="Yesmode, will do things without asking first (Not recommended)")
    parser.add_argument('--out', dest="out", help="Output directory, where to make the new files, default is current")
    parser.add_argument('--save', dest="save",action='store_const',
                        const=True, default=False, help="Save the params to the output directory as json")

    args = parser.parse_args()
    args_d = vars(args)
    make_all(args_d)
