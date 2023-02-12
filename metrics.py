import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, roc_curve, auc, confusion_matrix, accuracy_score
import numpy as np

def _make_roc(y_true,Y_pred,metrics,roc_curve_,which_output=0,print_=False):
	if roc_curve_:
		if Y_pred.shape==(Y_pred.shape[0],2):
			if which_output == 0:
				fpr, tpr, thresholds = roc_curve(y_true, Y_pred[:,1], pos_label=1)
			else:
				fpr, tpr, thresholds = roc_curve(y_true, Y_pred[:1,], pos_label=1)
		else:
			fpr, tpr, thresholds = roc_curve(y_true, Y_pred, pos_label=1)
		valor_auc = auc(fpr, tpr)
		metrics["auc"] = valor_auc
		metrics["thresholds"] = thresholds
		plt.figure()
		plt.plot(fpr, tpr, label='ROC curve (area = %0.4f) ' % (valor_auc))
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver Operating Characteristic Curve')
		plt.legend(loc="lower right")
		plt.close()
		metrics['rocc'] = plt
		if print_:
			print('-- ROC Curve OK')
	return metrics

def _matrix_info(matrix,complete,acc,metrics,print_=False):
	if complete and acc != 1:
		TP = float(matrix[0][0])
		FP = float(matrix[0][1])
		FN = float(matrix[1][0])
		TN = float(matrix[1][1])

		if (TP+FN)>0:
			sensitivity = TP/(TP+FN)
		else:
			sensitivity = 0
		if (TN+FP)>0:
			specificity = TN/(TN+FP)
		else:
			specificity = 0

		if (TP+FP)>0:
			precision = TP/(TP+FP)
		else:
			precision = (TP+FP)

		if (sensitivity > 0) and (precision > 0):
			f1 = 2/((sensitivity**-1)+(precision**-1))
		else:
			f1 = 0

		metrics["sensitivity"] = sensitivity
		metrics["specificity"] = specificity
		metrics["f1"] = f1
		metrics["precision"] = precision

		if print_:
			print('-- Confusion Matrix OK')

	return metrics

def _classify(Y_pred,threshold,smaller_than=True,which_output=0,print_=False):

	if smaller_than:
		if Y_pred.shape==(Y_pred.shape[0],2):
			if which_output == 0:
				y_pred = (Y_pred[:,1]<=threshold)*1
			else:
				y_pred = (Y_pred[:1,]<=threshold)*1
		else:
			y_pred = (Y_pred<=threshold)*1
	else:
		if Y_pred.shape==(Y_pred.shape[0],2):
			if which_output == 0:
				y_pred = (Y_pred[:,1]>=threshold)*1
			else:
				y_pred = (Y_pred[:1,]>=threshold)*1
		else:
			y_pred = (Y_pred>=threshold)*1

	if print_:
		print('-- Classified')
	return y_pred

def get_metrics(y,Y_pred,threshold=None,which_output=0,print_=False,roc_curve_=False,complete=False,smaller_than=True):
	metrics = {}
	y_true = np.argmax(y, axis=-1)

	if not (threshold is None):

		y_pred = _classify(Y_pred,threshold,smaller_than,which_output,print_)

		acc = accuracy_score(y_true, y_pred)
		kappa = cohen_kappa_score(y_true, y_pred)
		matrix = confusion_matrix(y_true, y_pred)
		metrics["acc"] = acc
		metrics["kappa"] = kappa
		metrics["matrix"] = matrix

		metrics = _matrix_info(matrix,complete,acc,metrics,print_)

	metrics = _make_roc(y_true,Y_pred,metrics,roc_curve_,which_output,print_)

	return metrics

def save_metrics(save_path,metrics,save_roc=False,print_=False):
	fl = open(save_path + 'metrics.txt','w+')

	for key in metrics.keys():
		if 'roc' in key and save_roc:
			metrics[key].savefig(save_path + 'roc_curve')
		else:
			fl.write(key+' = ' + str(metrics[key]).replace('\n','') + '\n')
		if print_:
			if 'matrix' in key:
				print('-- Confusion Matrix saved')
			elif 'roc' in key:
				print('-- ROC Curve saved')
			elif 'acc' in key:
				if metrics[key] == 1:
					print('PROBLEM: ACCURACY EQUALS 1')
				elif metrics[key] == 0:
					print('PROBLEM: ACCURACY EQUALS 0')
				else:
					print('-- Accuracy OK')
			elif 'auc' in key:
				if metrics[key] == 1:
					print('PROBLEM: AREA UNDER CURVE EQUALS 1')
				elif metrics[key] == 0:
					print('PROBLEM: AREA UNDER CURVE EQUALS 0')
				else:
					print('-- Area Under Curve OK')
		
	fl.close()

def save_mult_metrics(save_path,metrics_list,flags,save_roc=False,name_key='tag',print_=False):
	fl = open(save_path + 'metrics_list.txt','w+')

	is_name_key = False
	for metrics in metrics_list:
		for key in metrics.keys():
			if key == name_key:
				is_name_key = True
				continue
			fl.write(key+',')
		if is_name_key:
			fl.write(name_key+'\n')
		else:
			if print_:
				print('''WARNING: there is no model identifier, 
				include a unique "name_key" in the flags dictionary 
				for each model in order to know the metrics they are associated with''')
			fl.write('\n')
		break

	for metrics in metrics_list:
		for key in metrics.keys():
			if key == name_key:
				continue
			if 'roc' in key and save_roc:
				metrics[key].savefig(save_path + 'roc_curve' + metrics[name_key])
			else:
				fl.write(str(metrics[key]).replace('\n','') + ',')
		if metrics[name_key] == None:
			fl.write('\n')
			if print_:
				if is_name_key:
					print('WARNING: model without identifier')
		else:
			fl.write(metrics[name_key]+'\n')

	fl.close()
