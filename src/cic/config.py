import os, sys

path = os.path.split(__file__)[0]
# print("abs path is %s" %(os.path.abspath()))

config = {
    'batch_size' : 32,
    'val_batch_size': 1,
    'num_iters': 1000000,
    'seed' : 1,
    'lr': 3.16e-4,

    'lr_update_step':375,
    'test_cycle': 200,

    'cuda' : True,
    'gpus' : 1,
    'gpuargs' : {'num_workers': 4,
               'pin_memory' : True
              },

    'model':'ColorizationNet',
    'bachnorm':True,
    'pretrained':False,

    # 'opt_config':{
    #     'lr' : 0.001,
    #     'betas' : (0.9, 0.99),
    #     'eps': 1e-8,
    #     'weight_decay': 0.004
    # },

    'save': os.path.join(os.environ["SR_PROJECT_OUTS_PATH"], "cic"), 

    'image_folder_train' : {
        'root' : '%s/' % path,
        'file' : os.path.join(os.environ["SR_PROJECT_DATA_PATH"], "DIV2K/DIV2K_train_HR/*.png") ,
        'replicates': 1,
        'train':True
    },
    'image_folder_val' : {
        'root' : '%s/' % path,
        'file' : os.path.join(os.environ["SR_PROJECT_DATA_PATH"], "SET5/HR/*.png"),
        'replicates': 1,
        'train':False
    },

    'log_frequency': 1, #frequency for the number of epotch
    'save_iamge':'%s/work/img/' % path
}

# print(config['save'])
