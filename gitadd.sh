#########################################################################
# File Name: gitadd.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Jul  1 12:51:29 2021
#########################################################################
#!/bin/bash

for afile in `git status | grep "modified" | grep ".py$" | awk 'BEGIN{FS=":"}{print $2}'`
do
	echo "git add $afile"
	git add $afile
done

for afile in `git status | grep "modified" | grep ".sh$" | awk 'BEGIN{FS=":"}{print $2}'`
do
	echo "git add $afile"
	git add $afile
done

for afile in `git status | grep "modified" | grep ".cpp$" | awk 'BEGIN{FS=":"}{print $2}'`
do
	echo "git add $afile"
	git add $afile
done


