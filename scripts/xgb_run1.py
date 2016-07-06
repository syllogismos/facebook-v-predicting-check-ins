import xgboost_model as xgbm
import grid_generation as grid

g = grid.Grid(200, 50, 20, 5, pref = 'grid', files_flag=True, train_file='../main_train_0.02_5.csv')
xgbm1 = xgbm.XGB_Model(grid = g, threshold = 3, cross_validation_file='../main_cv_0.02_5.csv')

if __name__ == '__main__':
    """
    eta = 0.1
    max_depth = 13
    min_child_weight = 5
    gamma = 0.3
    subsample = 0.9
    colsample_bytree = 0.711
{'colsample_bytree': 0.6, 'silent': 1, 'scale_pos_weight': 1, 'nthread': 8, 'min_child_weight': 6, 'subsample': 0.9, 'eta': 0.1, 'objective': 'multi:softprob', 'alpha': 0.005, 'max_depth': 3, 'gamma': 0.1}
    """
    xgbm1.train_and_predict_parallel('../default_params_with_n100.csv', upload_to_s3=True)
