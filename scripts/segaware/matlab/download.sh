#!/bin/bash
#rsync -vaP -e "ssh -p 24" aharley@141.117.231.7:~/.bashrc .
#rsync -vaP -e "ssh -p 24" aharley@141.117.231.7:~/.screenrc .
#rsync -vaP -e "ssh -p 24" aharley@141.117.231.7:~/.bash_profile .
#rsync -vaP -e "ssh -p 24" aharley@141.117.231.7:~/.bash_aliases .
#rsync -vaP -e "ssh -p 24" aharley@141.117.231.7:~/segaware-light* .
#rsync -vaP -e "ssh -p 24" aharley@141.117.231.7:~/newscripts .
#rsync -vaP -e "ssh -p 24" aharley@141.117.231.7:~/datasets/VOC2012 datasets/
#rsync -vaP -e "ssh -p 24" aharley@141.117.231.7:~/datasets .
#rsync -vaP -e "ssh -p 24" aharley@141.117.231.7:~/datasets .
#rsync -vaP -e "ssh -p 24" aharley@141.117.231.7:~/newscripts .
#rsync -vaP -e "ssh " flow-release-1.0 lop1.autonlab.org:~/
#rsync -vaP -e "ssh " list lop1.autonlab.org:~/newscripts/segaware/
#rsync -vaP -e "ssh -p 24" aharley@141.117.231.10:~/newscripts/segaware/list/one* list/
#rsync -vaP -e "ssh -p 24" aharley@141.117.231.10:~/newscripts/segaware/matlab/merge* .
rsync -vaP -e "ssh -p 24" aharley@141.117.231.10:~/newscripts/segaware/matlab/inspect_flow/get_metrics.m inspect_flow/
