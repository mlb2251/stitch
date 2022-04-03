

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

# 2022-03-02
# launching on stitch commit e9b0b142 and stitch_dreamcoder commit 3387fd2f
./bench_dreamcoder.sh benches/logo_all
./bench_dreamcoder.sh benches/regex_all


