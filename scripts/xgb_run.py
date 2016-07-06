import xgboost_time_features as xgbm
import grid_generation as grid
import gc


if __name__ == '__main__':

    g1 = grid.Grid(200, 20, 30, 10, pref = 'grid', files_flag=True)
    xgbm1 = xgbm.XGB_Model(grid = g1, threshold = 7)
    params1 = {
        'objective': 'multi:softprob',
        'eta': 0.1,
        'max_depth': 5,
        'min_child_weight': 4,
        'gamma': 0.3,
        'subsample': 0.9,
        'colsample_bytree': 0.7,
	'colsample_bylevel': 0.9,
        'scale_pos_weight': 1,
        'nthread': 4,
        'silent': 1,
        'max_delta_step': 6
    }
    xgbm1.ini_params(params1)
    name1 = 'grid_200_20_30_10_params_01_5_4_03_09_07_09_6_th7.csv'
    print name1
    xgbm1.train_and_predict_parallel(name1, upload_to_s3=True)

    g1 = grid.Grid(280, 50, 30, 10, pref = 'grid', files_flag=True)
    xgbm1 = xgbm.XGB_Model(grid = g1, threshold = 7)
    params1 = {
        'objective': 'multi:softprob',
        'eta': 0.1,
        'max_depth': 5,
        'min_child_weight': 4,
        'gamma': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.65,
	'colsample_bylevel': 0.9,
        'scale_pos_weight': 1,
        'nthread': 4,
        'silent': 1,
        'max_delta_step': 7
    }
    xgbm1.ini_params(params1)
    name1 = 'grid_280_50_30_10_params_01_5_4_01_09_065_09_7_th7.csv'
    print name1
    xgbm1.train_and_predict_parallel(name1, upload_to_s3=True)


#    gc.collect()
#
#    g1 = grid.Grid(250, 20, 30, 10, pref = 'grid', files_flag=True)
#    xgbm1 = xgbm.XGB_Model(grid = g1, threshold = 10)
#    params1 = {
#        'objective': 'multi:softprob',
#        'eta': 0.1,
#        'max_depth': 7,
#        'min_child_weight': 4,
#        'gamma': 0.3,
#        'subsample': 0.9,
#        'colsample_bytree': 0.6,
#	'colsample_bylevel': 0.7,
#        'scale_pos_weight': 1,
#        'nthread': 4,
#        'silent': 1,
#        'max_delta_step': 8
#    }
#    xgbm1.ini_params(params1)
#    name1 = 'grid_250_20_30_10_params_01_7_4_03_09_06_07_8_th10.csv'
#    xgbm1.train_and_predict_parallel(name1, upload_to_s3=True)
#
#    gc.collect()
#
#    g1 = grid.Grid(200, 20, 30, 10, pref = 'grid', files_flag=True)
#    xgbm1 = xgbm.XGB_Model(grid = g1, threshold = 7)
#    params1 = {
#        'objective': 'multi:softprob',
#        'eta': 0.1,
#        'max_depth': 7,
#        'min_child_weight': 4,
#        'gamma': 0.1,
#        'subsample': 0.85,
#        'colsample_bytree': 0.65,
#	'colsample_bylevel': 0.6,
#        'scale_pos_weight': 1,
#        'nthread': 4,
#        'silent': 1,
#        'max_delta_step': 6
#    }
#    xgbm1.ini_params(params1)
#    name1 = 'grid_200_20_30_10_params_01_7_4_01_085_065_06_6_th7.csv'
#    xgbm1.train_and_predict_parallel(name1, upload_to_s3=True)
#
#    gc.collect()
#
#    g1 = grid.Grid(200, 20, 30, 10, pref = 'grid', files_flag=True)
#    xgbm1 = xgbm.XGB_Model(grid = g1, threshold = 10)
#    params1 = {
#        'objective': 'multi:softprob',
#        'eta': 0.1,
#        'max_depth': 7,
#        'min_child_weight': 4,
#        'gamma': 0.1,
#        'subsample': 0.85,
#        'colsample_bytree': 0.65,
#	'colsample_bylevel': 0.6,
#        'scale_pos_weight': 1,
#        'nthread': 4,
#        'silent': 1,
#        'max_delta_step': 6
#    }
#    xgbm1.ini_params(params1)
#    name1 = 'grid_200_20_30_10_params_01_7_4_01_085_065_06_6_th10.csv'
#    xgbm1.train_and_predict_parallel(name1, upload_to_s3=True)
#
#    gc.collect()
#
#    g1 = grid.Grid(200, 50, 20, 5, pref = 'grid', files_flag=True)
#    xgbm1 = xgbm.XGB_Model(grid = g1, threshold = 8)
#    params1 = {
#        'objective': 'multi:softprob',
#        'eta': 0.1,
#        'max_depth': 5,
#        'min_child_weight': 4,
#        'gamma': 0.1,
#        'subsample': 0.85,
#        'colsample_bytree': 0.65,
#	'colsample_bylevel': 0.8,
#        'scale_pos_weight': 1,
#        'nthread': 4,
#        'silent': 1,
#        'max_delta_step': 6
#    }
#    xgbm1.ini_params(params1)
#    name1 = 'grid_200_50_20_5_params_01_5_4_01_085_065_08_6_th8.csv'
#    xgbm1.train_and_predict_parallel(name1, upload_to_s3=True)
