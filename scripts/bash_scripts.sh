# sed '1d' -i train.csv # to remove the header from training data set in place

cat train.csv | sort -nk6 -t ',' > sorted_train.csv # Sort the file based on the place id

