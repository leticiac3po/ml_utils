import numpy as np
import os, cv2

# structure must be: dataset -> folders for classes
# if it's a one class model: one_class = True and which_class = CLASS_FOLDER_NAME
# def from_folders_to_split(base_dir,print_=False,save_all=False,img=False,val=False,split_train=0.7,
        # split_val=0.2,one_class=False,save_dir=None,which_class=None):

def _print_dic(data,print_):
    if print_:
        print('VERIFY SHAPES:')
        for key in data.keys():
            if 'name' in key:
                continue
            else:
                print(key,data[key].shape)
        print('-------------')

def _shuffle_together(files):
    seed = np.random.randint(0, 10000)
    for fl in files:
        np.random.seed(seed)
        np.random.shuffle(fl)

def _read_names(base_dir,filename):
    fl = open(os.path.join(base_dir, filename+'.txt'),'r')
    names = []
    for line in fl:
        line = line.replace('\n','')
        names.append(line)
    return names

def _save_names(base_dir,filename,names):
    fl = open(os.path.join(base_dir, filename),'w+')
    for name in names:
        fl.write(name+'\n')
    fl.close()

def _is_train(one_class,class_,which_class):
    if one_class:
        if class_ == which_class:
            return True
        else:
            return False
    else:
        return True

def _save_dic(save_dir,data):
    for key in data.keys():
        if 'name' in key:
            _save_names(save_dir,key+'.txt',data[key])
        else:
            np.save(os.path.join(save_dir, key+'.npy'),data[key])

def _load_filenames(base_dir,filenames,tag=''):
    data = {}
    new_filenames = []
    if tag != '':
        tag = '_' + tag
    for filename in filenames:
        key = filename.split('.')[0]
        if 'names' in key:
            data[key] = _read_names(base_dir,filename)
            new_filenames.append(filename+'.txt')
        else:
            data[key] = np.load(os.path.join(base_dir,filename+'.npy'))
            new_filenames.append(filename+'.npy')
    return data,new_filenames

def _is_in_data(keys,key):
    split = ['train_','test_','val_']
    for part in split:
        if (part + key) in keys:
            return True
    return False

def _load_from_dir(base_dir,data={}):
    keys = os.listdir(base_dir)
    for key in keys:
        if not('.' in key) or _is_in_data(data.keys(),key.split('.')[0]):
            continue
        if 'name' in key:
            data[key.split('.')[0]] = _read_names(base_dir,key.replace('.txt',''))
        else:
            data[key.split('.')[0]] = np.load(os.path.join(base_dir,key))
    return data,keys

def _delete_filenames(base_dir,filenames=None):
    if filenames == None:
        filenames = os.listdir(base_dir)
    for filename in filenames:
        os.remove(os.path.join(base_dir,filename))

def _prep_split(tag,data,train,val,split_train,split_val):
    if tag == None:
        fls = ['data','lbls','names']
    else:
        fls = ['data_'+str(tag),'lbls_'+str(tag),'names_'+str(tag)]

    if train:
        split_train = int(data[fls[0]].shape[0]*split_train)
    if val:
        split_val = int(data[fls[0]].shape[0]*(split_train + split_val))

    if val:
        if train:
            split = ['train','test','val']
        else:
            split = ['test','val']
    elif train:
        split = ['train','test']
    else:
        split = ['test']

    return fls,split_train,split_val, split

def _make_save_dirs(base_dir,save_dir,save_all):
    if save_dir == None:
        save_dir = base_dir
    if save_all:
        save_one = os.path.join(save_dir,'class_files')
        save_two = os.path.join(save_dir,'class_files_split')
        save_three = os.path.join(save_dir,'final_files')
        if not os.path.exists(save_one):
            os.mkdir(save_one)
        if not os.path.exists(save_two):
            os.mkdir(save_two)
        if not os.path.exists(save_three):
            os.mkdir(save_three)
        return save_one, save_two, save_three
    else:
        return save_dir, save_dir, save_dir

def _prep_concat(val):
    if val:
        split = ['train','test','val']
    else:
        split = ['train','test']

    return split

def concat_one_folder(base_dir,print_=False,img=False,save_dir=None,tag=''):
    if tag != '':
        tag = '_' + tag
    if save_dir == None:
        save_dir = base_dir
    files = os.listdir(base_dir)
    fl = np.load(os.path.join(base_dir, files[0]))
    fl = np.expand_dims(fl,axis=0)

    data={}
    data['data'+tag] = np.zeros((1,fl.shape[1]))
    data['lbls'+tag] = np.zeros((1,1))
    data['names'+tag] = []

    for fl in files:
        data['names'+tag].append(os.path.join(base_dir, fl))
        if img:
            fl = cv2.imread(os.path.join(base_dir, fl))
        else:
            fl = np.load(os.path.join(base_dir, fl))
        fl = np.expand_dims(fl,axis=0)
        lbls = np.full((fl.shape[0],1),0)
        data['data'+tag] = np.concatenate([data['data'+tag],fl],axis=0)
        data['lbls'+tag] = np.concatenate([data['lbls'+tag],lbls],axis=0)

    data['data'+tag] = data['data'+tag][1:]
    data['lbls'+tag] = data['lbls'+tag][1:]

    _shuffle_together([data['data'+tag],data['lbls'+tag],data['names'+tag]])
    _save_dic(save_dir,data)
    _print_dic(data,print_)
    return list(data.keys())

def split_one_class(base_dir,filenames,train=True,val=False,split_train=0.7,split_val=0.2,
        save=True,save_dir=None,tag=None,print_=False):

    if save_dir == None:
        save_dir = base_dir
    data,filenames = _load_filenames(base_dir,filenames,tag=tag)
    fls,split_train,split_val,split = _prep_split(tag,data,train,val,split_train,split_val)

    new_data = {}
    if val:
        if train:
            for fl in fls:
                new_data['train_'+fl] = data[fl][:split_train]
                new_data['test_'+fl] = data[fl][split_train:split_val]
                new_data['val_'+fl] = data[fl][split_val:]
        else:
            new_data['test_'+fl] = data[fl][:split_val]
            new_data['val_'+fl] = data[fl][split_val:]
    elif train:
        for fl in fls:
            new_data['train_'+fl] = data[fl][:split_train]
            new_data['test_'+fl] = data[fl][split_train:]
    else:
        for fl in fls:
            new_data['test_'+fl] = data[fl]

    for part in split:
        _shuffle_together([new_data[part+'_'+fls[0]],new_data[part+'_'+fls[1]],new_data[part+'_'+fls[2]]])
    if not save:
        _delete_filenames(base_dir,filenames)
    _save_dic(save_dir,new_data)
    _print_dic(new_data,print_)
    return list(new_data.keys())

def concat_classes_after_split(base_dir,class_names,val=False,save=True,save_dir=None,one_class=False,which_class=None,one_class_dir=None,print_=False):
    
    if save_dir == None:
        save_dir = base_dir
        
    data,filenames = _load_from_dir(base_dir)
    split = _prep_concat(val)

    new_data = {}
    train = True
    for i,class_ in enumerate(class_names):
        if i == 0:
            for part in split:
                if one_class and class_ != which_class and part == 'train':
                    train = False
                    continue
                new_data[part+'_data'] = data[part+'_data_'+str(class_)]
                new_data[part+'_lbls'] = data[part+'_lbls_'+str(class_)]
                new_data[part+'_names'] = data[part+'_names_'+str(class_)]
        else:
            for part in split:
                if one_class and class_ != which_class and part == 'train':
                    continue
                if train == False:
                    train = True
                    new_data[part+'_data'] = data[part+'_data_'+str(class_)]
                    new_data[part+'_lbls'] = data[part+'_lbls_'+str(class_)]
                    new_data[part+'_names'] = data[part+'_names_'+str(class_)]
                else:
                    new_data[part+'_data'] = np.concatenate([new_data[part+'_data'], data[part+'_data_'+str(class_)]],axis=0)
                    new_data[part+'_lbls'] = np.concatenate([new_data[part+'_lbls'], data[part+'_lbls_'+str(class_)]],axis=0)
                    new_data[part+'_names'] = new_data[part+'_names']+data[part+'_names_'+str(class_)]
    
    for part in split:
        _shuffle_together([new_data[part+'_data'],new_data[part+'_lbls'],new_data[part+'_names']])
    if not save:
        _delete_filenames(base_dir,filenames)
    _save_dic(save_dir,new_data)
    _print_dic(new_data,print_)
    return list(new_data.keys())
    
def from_folders_to_split(base_dir,print_=False,save_all=False,img=False,val=False,split_train=0.7,
        split_val=0.2,one_class=False,save_dir=None,which_class=None):
    save_one, save_two, save_three = _make_save_dirs(base_dir,save_dir,save_all)
    class_names = os.listdir(base_dir)
    for class_name in class_names:
        class_dir = os.path.join(base_dir,class_name)
        saved_files = concat_one_folder(class_dir,print_,img,tag=class_name,save_dir=save_one)
        train = _is_train(one_class,class_name,which_class)
        split_one_class(save_one,saved_files,train,val,split_train,split_val,save_all,save_two,class_name,print_)
    return concat_classes_after_split(save_two,class_names,val,save_all,save_three,one_class,which_class,save_one,print_)
