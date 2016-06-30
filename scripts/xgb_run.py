import xgboost_model as xgbm
import grid_generation as grid

g = grid.Grid(400, 100, 50 ,10, pref = 'grid1', files_flag=True, train_file='../main_train_0.02_5.csv')
xgbm1 = xgbm.XGB_Model(grid = g, threshold = 7, cross_validation_file='../main_cv_0.02_5.csv')

if __name__ == '__main__':
	xgbm1.train_and_predict_parallel('../ec2_xgb_eta0_1_max_depth_10_min_child_wt6_gamma0_subsample0_8_colsample_bytree0_8_scale_pos_wt1_th7.csv', upload_to_s3=True)
