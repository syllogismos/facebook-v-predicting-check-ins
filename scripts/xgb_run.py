import xgboost_time_features as xgbm
import grid_generation as grid


if __name__ == '__main__':

    g1 = grid.Grid(200, 50, 20, 5, pref = 'grid', files_flag=True)
    xgbm1 = xgbm.XGB_Model(grid = g1, threshold = 7)
    name1 = 'run1 results'
    xgbm1.train_and_predict_parallel(name1, upload_to_s3=True)
