import sys
from data_loader import *

sys.path.append("..")
from permute_toolbox.permute_toolbox import *
from permute_toolbox.rsfc_tools import *

X, y = load_timeseries_data()
# np.save('../data/X_scale.npy', X_scale)
# np.save('../data/y.npy', y)

features_to_permute = [get_flat_inds_for_net('Default')]
results = loocv_svm_permute_features(X,y,features_to_permute)
# np.save('../results/permute_dmn_LOOCV_SVM.npy', results)
