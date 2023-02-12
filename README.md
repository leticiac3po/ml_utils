# Machine Learning Tools
Tools for Machine Learning dataset prep and model evaluation. Metrics.py is pretty much bug proof, but I haven't tested every scenario in Dataset_prep.py.

## DATASET PREP

It needs the directory where the dataset is, and the structure must be: dataset -> folders for classes. It can deal with .npy files and images (theoretically, I haven't tested the images), I also want to add video pre processing eventually.

def from_folders_to_split(base_dir,print_=False,save_all=False,img=False,val=False,split_train=0.7,split_val=0.2,one_class=False,save_dir=None,which_class=None):

The breakdown on the other arguments:

- print_ : will print updates on the matrices' shapes if True (default = False)
- save_all : saves the matrices in each step of the process (class concatenation, train/test splitting in each class, and final concatenation) in subfolders (default = False)
- img : if the dataset is of images (default = False)
- val : if you want to include a validation set (default = False)
- split_train : percentage for training (default = 0.7)
- split_val : percentag for validation, if applicable (default = 0.2)
- one_class : set as True if it's a one class model (default = False)
- save_dir : really recommend having one, unless you don't mind a huge mess because by default it'll save everything in the base_dir (default = None)
- which_class : if it's a one class model, please specify which class is the training class, super important, I should probably even add an exception (default = None)

## METRICS

### get_metrics

The main method is:

get_metrics(y,Y_pred,threshold=None,which_output=0,print_=False,roc_curve=False,complete=False,smaller_than=True)

It receives the labels and the raw model prediction, and returns them. It does not save them, although I should probably get around to adding that. It can also receive a threshold and the following flags:

- which_output : will get either the first or second row of predictions, if there's more than one (default = 0)
- print_ : will print updates on the metrics if True (default = False)
- roc_curve : will calculate the roc curve if True (defalt = False)
- complete : will calculate the metrics associated with the confusion matrix if True (defalt = False)
- smaller_than : will apply the threshold as a smaller than if True (default = True)

### save_metrics

The secondary method is:

save_metrics(save_path,metrics,save_roc=False,print_=False)

It receives save path and the metrics dictionary. It can receive the following flags:
- save_roc : will save the roc curve in a png file if True (defalt = False)
- print : will print updates on the metrics if True (defalt = False)

### save_mult_metrics

And finally, for your convenience when comparing multiple sets of metrics to choose the best:

save_mult_metrics(save_path,metrics_list,flags,save_roc=False,name_key='tag',print_=False)

It saves multiple metrics in one .txt file for easier comparison, one model's metrics per line. It receives a list of metrics dictionaries, the same kind that get_metrics returns. It works better if you add an ID to each set of metrics so they can be identified. The default key for that is 'tag' but a different one can be set. I realise it would be better to have a dictionary of metrics dictionaries, and I'll update that when possible. The first row of the .txt file have all the keys to the metrics dictionary, and the last column will be athe identifier.

It also receives save path and and it can receive the following flags:

- save_roc : will save the roc curves in png files if True (defalt = False)
- name_key : is the key to the identifier in the dictionary (default = 'tag')
- print_ : will print updates on the metrics if True (defalt = False)
