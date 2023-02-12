# ml_utils
Stuff I'm making to hopefully make my job easier.

Metrics.py is pretty much bug proof, but I haven't tested every scenario in Dataset_prep.py. I also want to add video pre processing eventually.

## GET METRICS
The main method is:

get_metrics(y,Y_pred,threshold=None,which_output=0,print_=False,roc_curve=False,complete=False,smaller_than=True)

It receives the labels and the raw model prediction, and returns them. It does not save them, although I should probably get around to adding that. It can also receive a threshold and the following flags:

- which_output : will get either the first or second row of predictions, if there's more than one (default = 0)
- print_ : will print updates on the metrics if True (default = False)
- roc_curve : will calculate the roc curve if True (defalt = False)
- complete : will calculate the metrics associated with the confusion matrix if True (defalt = False)
- smaller_than : will apply the threshold as a smaller than if True (default = True)

The secondary method is:

save_metrics(save_path,metrics,save_roc=False,print_=False)

It receives save path and the metrics dictionary. It can receive the following flags:
- save_roc : will save the roc curve in a png file if True (defalt = False)
- print : will print updates on the metrics if True (defalt = False)

And finally, for your convenience when comparing multiple sets of metrics to choose the best:

save_mult_metrics(save_path,metrics_list,flags,save_roc=False,name_key='tag',print_=False)

It saves multiple metrics in one .txt file for easier comparison, one model's metrics per line. It receives a list of metrics dictionaries, the same kind that get_metrics returns. It works better if you add an ID to each set of metrics so they can be identified. The default key for that is 'tag' but a different one can be set. I realise it would be better to have a dictionary of metrics dictionaries, and I'll update that when possible. The first row of the .txt file have all the keys to the metrics dictionary, and the last column will be athe identifier.

It also receives save path and and it can receive the following flags:

- save_roc : will save the roc curves in png files if True (defalt = False)
- name_key : is the key to the identifier in the dictionary (default = 'tag')
- print_ : will print updates on the metrics if True (defalt = False)
