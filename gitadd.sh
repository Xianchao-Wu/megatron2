#########################################################################
# File Name: gitadd.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Jul  1 12:51:29 2021
#########################################################################
#!/bin/bash

prockey(){
	key=$1 # e.g., "modified"

	for ext in .py .sh .cpp
	do
		print $ext
		for afile in `git status | grep -v ".pyc$" | grep $key | grep $ext"$" | awk 'BEGIN{FS=":"}{print $2}'`
		do
			echo "---git add $afile"
			git add $afile
		done

		for afile in `git status | grep -v ":" | grep -v ".npy" | grep $ext"$" | awk '{print $1}'`
		do
			echo "===git add $afile"
			git add $afile
		done
	done
}

prockey "modified"
prockey "deleted"
