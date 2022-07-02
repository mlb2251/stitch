
mkdir -p benches_small_cogsci

SUBDOMAIN="nuts-bolts"
mkdir -p benches_small_cogsci/$SUBDOMAIN
python programs_list_to_dc.py pre_benches_small_cogsci/$SUBDOMAIN.json benches_small_cogsci/$SUBDOMAIN/bench000it0.json 1 graphics

SUBDOMAIN="dials"
mkdir -p benches_small_cogsci/$SUBDOMAIN
python programs_list_to_dc.py pre_benches_small_cogsci/$SUBDOMAIN.json benches_small_cogsci/$SUBDOMAIN/bench000it0.json 1 graphics

SUBDOMAIN="wheels"
mkdir -p benches_small_cogsci/$SUBDOMAIN
python programs_list_to_dc.py pre_benches_small_cogsci/$SUBDOMAIN.json benches_small_cogsci/$SUBDOMAIN/bench000it0.json 1 graphics

SUBDOMAIN="furniture"
mkdir -p benches_small_cogsci/$SUBDOMAIN
python programs_list_to_dc.py pre_benches_small_cogsci/$SUBDOMAIN.json benches_small_cogsci/$SUBDOMAIN/bench000it0.json 1 graphics


SUBDOMAIN="house"
mkdir -p benches_small_cogsci/$SUBDOMAIN
python programs_list_to_dc.py pre_benches_small_cogsci/$SUBDOMAIN.json benches_small_cogsci/$SUBDOMAIN/bench000it0.json 1 towers

SUBDOMAIN="city"
mkdir -p benches_small_cogsci/$SUBDOMAIN
python programs_list_to_dc.py pre_benches_small_cogsci/$SUBDOMAIN.json benches_small_cogsci/$SUBDOMAIN/bench000it0.json 1 towers

SUBDOMAIN="bridge"
mkdir -p benches_small_cogsci/$SUBDOMAIN
python programs_list_to_dc.py pre_benches_small_cogsci/$SUBDOMAIN.json benches_small_cogsci/$SUBDOMAIN/bench000it0.json 1 towers

SUBDOMAIN="castle"
mkdir -p benches_small_cogsci/$SUBDOMAIN
python programs_list_to_dc.py pre_benches_small_cogsci/$SUBDOMAIN.json benches_small_cogsci/$SUBDOMAIN/bench000it0.json 1 towers
