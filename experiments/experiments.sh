

##############
# BENCHMARKS #
##############

# TESTING
fake_logo_2019-03-23T18:06:23.106382: a fake benchmark used for early testing

# Case study 1
# logo



# initial fake tests
./bench_dreamcoder.sh "benches/fake_logo_2019-03-23T18:06:23.106382"
./bench_stitch.sh "benches/fake_logo_2019-03-23T18:06:23.106382" "benches/fake_logo_2019-03-23T18:06:23.106382"
python3 analyze.py graphs bar "benches/fake_logo_2019-03-23T18:06:23.106382/out/stitch/2022-04-02_19-52-54" "benches/fake_logo_2019-03-23T18:06:23.106382/out/dc/2021-03-02_00-00-00"

# 2022-03-02 overnight
# launching on stitch commit 9d528f08 and stitch_dreamcoder commit 3387fd2f
./bench_dreamcoder.sh benches/logo_all
./bench_dreamcoder.sh benches/regex_all
./bench_dreamcoder.sh benches/rec-fp_arity3_2019-07-11T19:49:10.899159

# the logo and regex barely made progress overnight, the rec-fp is quick and finished in a few min
# looking at the progress so  far

python3 analyze.py process dreamcoder /scratch/mlbowers/proj/stitch/experiments/benches/regex_all/out/dc/2022-04-02_22-36-48
./bench_stitch.sh benches/regex_all /scratch/mlbowers/proj/stitch/experiments/benches/regex_all/out/dc/2022-04-02_22-36-48 loose







