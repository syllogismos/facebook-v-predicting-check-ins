import xgboost_model as xgbm
import grid_generation as grid

g1 = grid.Grid(100, 50 ,20, 5, pref = 'grid', files_flag=True, train_file='../main_train_0.02_5.csv')
xgbm1 = xgbm.XGB_Model(grid = g1, threshold = 3, cross_validation_file='../main_cv_0.02_5.csv')

# g2 = grid.Grid(200, 50 ,20, 5, pref = 'grid', files_flag=True, train_file='../main_train_0.02_5.csv')
# xgbm2 = xgbm.XGB_Model(grid = g2, threshold = 3, cross_validation_file='../main_cv_0.02_5.csv')
if __name__ == '__main__':
    """
    eta = 0.1
    max_depth = 13
    min_child_weight = 5
    gamma = 0.3
    subsample = 0.9
    colsample_bytree = 0.7
    """
    print "grid 1 training"
    xgbm1.train_and_predict_parallel('../ec2_xgb_eta0_1_max_depth_8_min_child_wt5_gamma0.3_subsample0_9_colsample_bytree0_7_scale_pos_wt1_th3_100_50_20_5_edges.csv', upload_to_s3=False)
    print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
#     print "grid 2 training"
#     xgbm2.train_and_predict_parallel('../ec2_xgb_eta0_1_max_depth_8_min_child_wt5_gamma0.3_subsample0_9_colsample_bytree0_7_scale_pos_wt1_th3_200_50_20_5.csv', upload_to_s3=True)
