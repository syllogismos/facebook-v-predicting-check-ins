# 1403031520
# 1820452146
# 2220607398
# 2639496044
# 3054534752
# 3472342559
# 3890200660
# 4305698270
# 4715539546
# 5136400615
# 5559039358
# 5970648397
# 6388849530
# 6811212719
# 7238542045
# 7655890385
# 8063763003
# 8472470001
# 8885844238
# 9312478920
# 9723297030

# splitting commands


[2016-06-08 16:31] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/1_train sorted_train.csv '/1403031520/'
59645798
1209284608
[2016-06-08 16:33] anil at pan in ~/Code/facebook on master [?]
$ ls split_train/
1_train00 1_train01
[2016-06-08 16:33] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/2_train split_train/1_train01 '/1820452146/'
58942203
1150342405
[2016-06-08 16:34] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/3_train split_train/2_train01 '/2220607398/'
58786555
1091555850
[2016-06-08 16:35] anil at pan in ~/Code/facebook on master [?]
$ ls split_train/
1_train00 1_train01 2_train00 2_train01 3_train00 3_train01
[2016-06-08 16:35] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/4_train split_train/3_train01 '/2639496044/'
59199003
1032356847
[2016-06-08 16:36] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/5_train split_train/4_train01 '/3054534752/'
57145372
975211475
[2016-06-08 16:37] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/6_train split_train/5_train01 '/3472342559/'
59659974
915551501
[2016-06-08 16:37] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/7_train split_train/6_train01 '/3890200660/'
57711216
857840285
[2016-06-08 16:37] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/8_train split_train/7_train01 '/4305698270/'
58261240
799579045
[2016-06-08 16:38] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/9_train split_train/8_train01 '/4715539546/'
56826634
742752411
[2016-06-08 16:38] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/10_train split_train/9_train01 '/5136400615/'
56775268
685977143
[2016-06-08 16:39] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/11_train split_train/10_train01 '/5559039358/'
56994152
628982991
[2016-06-08 16:39] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/12_train split_train/11_train01 '/5970648397/'
58871130
570111861
[2016-06-08 16:39] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/13_train split_train/12_train01 '/6388849530/'
60221102
509890759
[2016-06-08 16:40] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/14_train split_train/13_train01 '/6811212719/'
58218091
451672668
[2016-06-08 16:40] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/15_train split_train/14_train01 '/7238542045/'
59696511
391976157
[2016-06-08 16:41] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/16_train split_train/15_train01 '/7655890385/'
59266284
332709873
[2016-06-08 16:41] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/17_train split_train/16_train01 '/8063763003/'
58252105
274457768
[2016-06-08 16:41] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/18_train split_train/17_train01 '/8472470001/'
59270503
215187265
[2016-06-08 16:42] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/19_train split_train/18_train01 '/8885844238/'
59102171
156085094
[2016-06-08 16:42] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/20_train split_train/19_train01 '/9312478920/'
58391811
97693283
[2016-06-08 16:42] anil at pan in ~/Code/facebook on master [?]
$ csplit -f split_train/21_train split_train/20_train01 '/9723297030/'
57783921
39909362