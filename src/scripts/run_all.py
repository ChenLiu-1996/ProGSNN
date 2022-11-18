import os

random_seed = 2022

protein_name = 'BPTI'

#os.system('python step_01_get_data.py --protein_name %s' % (protein_name))
for alpha in [0.001, 0.01, 0.1, 0.5, 1, 10]:
    for model in ['gsae', 'progsnn']:
        os.system('python step_02_train_model.py --protein_name %s --model %s --alpha %s --random_seed %s' %
                  (protein_name, model, alpha, random_seed))
        os.system('python step_03_compute_outputs.py --protein_name %s --model %s --alpha %s --random_seed %s' %
                  (protein_name, model, alpha, random_seed))
        os.system('python step_04_plots.py --protein_name %s --model %s --alpha %s --random_seed %s' %
                  (protein_name, model, alpha, random_seed))
